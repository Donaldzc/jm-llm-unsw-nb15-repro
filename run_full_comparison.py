"""Reproduce Table 2 of the JM-LLM paper on the FULL UNSW-NB15 split
(175,341 training flows / 82,332 testing flows).

Protocol
--------
* train/val  : official training set, stratified 80/20
* test       : official testing set (the paper's held-out split)
* graph      : KNN(k=3) gaussian-kernel graph over all 257,673 flows
* models     : JM, SVM, LSTM, Random-Forest-GNN, JM-GNN, JM-LSTM, JM-LLM

Run:  python run_full_comparison.py
"""
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)  # avoid thread oversubscription on 32-core hosts
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

import common
from correction import (PlattCalibrator, PriorShift, best_threshold,
                        calc_metrics)
from models import (DEVICE, CollaborativeModel, GNNClassifier, GNNModel,
                    JordanModel, LMModel, LSTMClassifier, LSTMModel,
                    batched_predict, set_seed)
from common import scipy_to_torch_sparse
from normal_profile import fit_normal_profile

N_COMPONENTS = 26
EPOCHS = 120
PATIENCE = 10
BATCH = 8192  # large batches keep CPU kernel launch/sync overhead low


# ---------------------------------------------------------------------------
# generic trainers
# ---------------------------------------------------------------------------
def _logits(model, X_all, adj, idx):
    if adj is not None:
        return model(X_all, adj)["class_pred"][idx]
    return model(X_all[idx])["class_pred"]


def _proba(model, X_all, adj, idx, batch=BATCH):
    """P(attack) for the given node indices."""
    model.eval()
    with torch.no_grad():
        if adj is not None:
            return torch.softmax(model(X_all, adj)["class_pred"][idx], 1)[
                :, 1].cpu().numpy()
        outs = []
        for i in range(0, len(idx), batch):
            part = idx[i:i + batch]
            outs.append(torch.softmax(model(X_all[part])["class_pred"], 1)[
                :, 1].cpu().numpy())
    return np.concatenate(outs)


def _predict(model, X_all, adj, idx, batch=BATCH):
    model.eval()
    with torch.no_grad():
        if adj is not None:
            return model(X_all, adj)["class_pred"][idx].argmax(1).cpu().numpy()
        outs = []
        for i in range(0, len(idx), batch):
            part = idx[i:i + batch]
            outs.append(model(X_all[part])["class_pred"].argmax(1).cpu().numpy())
    return np.concatenate(outs)


def train_model(model, X_all, adj, y, masks, epochs=EPOCHS, patience=PATIENCE,
                lr=1e-3, weight_decay=5e-4, tag="model"):
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["val"])[0]
    counts = np.bincount(y[train_idx], minlength=2).astype(np.float32)
    cls_w = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                         dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=cls_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_loss, counter, best_state = -1.0, float("inf"), 0, None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        if adj is not None:                      # full-batch graph training
            opt.zero_grad()
            loss = crit(_logits(model, X_all, adj, train_idx), y_t[train_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        else:                                     # mini-batch MLP/LSTM
            perm = np.random.permutation(train_idx)
            for i in range(0, len(perm), BATCH):
                part = perm[i:i + BATCH]
                opt.zero_grad()
                loss = crit(model(X_all[part])["class_pred"], y_t[part])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        pred = _predict(model, X_all, adj, val_idx)
        # model selection on validation macro-F1 (robust to class imbalance)
        val_score = float(f1_score(y[val_idx], pred, average="macro"))
        if val_score >= best_val:
            best_val, counter = val_score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                break
        if epoch % 10 == 0:
            print(f"  [{tag}] epoch {epoch}: val_F1={val_score:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    return best_val


def evaluate(model, X_all, adj, y, masks, splits=("train", "val", "test")):
    res = {}
    for split in splits:
        idx = np.where(masks[split])[0]
        pred = _predict(model, X_all, adj, idx)
        m = calc_metrics(y[idx], pred)
        res[f"{split}_acc"] = m["acc"]
        if split == "test":
            res["f1"] = m["macro_f1"]
            res["test_bacc"] = m["bacc"]
            res["normal_recall"] = m["normal_recall"]
            res["attack_recall"] = m["attack_recall"]
    return res


def run_p1(p_all, y, masks, method="none", tag="P1"):
    """Decision-layer correction: Platt calibration on validation, then either a
    validation-tuned threshold or an estimated target-prior shift."""
    tr = np.where(masks["train"])[0]
    va = np.where(masks["val"])[0]
    te = np.where(masks["test"])[0]

    cal = PlattCalibrator().fit(p_all[va], y[va])
    p = cal.transform(p_all)
    info = f"a={cal.a:.3f} b={cal.b:.3f}"
    if method == "none":
        thr = best_threshold(p[va], y[va])
        pred = (p > thr).astype(int)
        info += f" thr={thr:.2f}"
    else:
        shift = PriorShift(y[tr].mean(), method=method).fit(p[va], y[va], p[te])
        pred = (shift.apply(p) > 0.5).astype(int)
        info += (f" prior {shift.pi_s:.3f}->{shift.pi_t:.3f} (true test "
                 f"{y[te].mean():.3f})")

    res = {"loss": np.nan}
    for split, idx in (("train", tr), ("val", va), ("test", te)):
        m = calc_metrics(y[idx], pred[idx])
        res[f"{split}_acc"] = m["acc"]
        if split == "test":
            res["f1"] = m["macro_f1"]
            res["test_bacc"] = m["bacc"]
            res["normal_recall"] = m["normal_recall"]
            res["attack_recall"] = m["attack_recall"]
    print(f"  [{tag}] {info} -> test_acc={res['test_acc']:.4f} "
          f"bacc={res['test_bacc']:.4f} normal_recall={res['normal_recall']:.4f}")
    return res


# ---------------------------------------------------------------------------
# individual models
# ---------------------------------------------------------------------------
def run_jm(X_all, y, masks, adjacency, jordan):
    """Jordan propagation model (paper: JM)."""
    initial = np.zeros(len(y))
    initial[masks["train"] & (y == 1)] = 1.0
    t0 = time.time()
    predicted = jordan.predict_infection(initial, t=3)
    print(f"  [JM] propagation done in {time.time() - t0:.0f}s")

    tr_idx = np.where(masks["train"])[0]
    tr_pred = (predicted[tr_idx] > 0.5).astype(int)
    train_acc = float((tr_pred == y[tr_idx]).mean())

    # nearest training flow in feature space carries the propagated score
    nn1 = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_all[tr_idx])
    res = {"loss": np.nan, "train_acc": train_acc, "val_acc": np.nan}
    for split in ("val", "test"):
        idx = np.where(masks[split])[0]
        _, nn_idx = nn1.kneighbors(X_all[idx])
        pred = (predicted[tr_idx[nn_idx.ravel()]] > 0.5).astype(int)
        m = calc_metrics(y[idx], pred)
        res[f"{split}_acc"] = m["acc"]
        if split == "test":
            res["f1"] = m["macro_f1"]
            res["test_bacc"] = m["bacc"]
            res["normal_recall"] = m["normal_recall"]
            res["attack_recall"] = m["attack_recall"]
    print(f"  [JM] {res}")
    return res


def run_svm(X_all, y, masks):
    """Linear SVM on all 39 numeric + one-hot features (RBF SVC is O(n^2) at
    this scale, so the linear kernel is used - documented adaptation)."""
    tr_idx = np.where(masks["train"])[0]
    clf = LinearSVC(C=1.0, max_iter=3000, dual="auto", random_state=42)
    t0 = time.time()
    clf.fit(X_all[tr_idx].numpy(), y[tr_idx])
    print(f"  [SVM] fitted in {time.time() - t0:.0f}s")
    res = {"loss": np.nan}
    for split in ("train", "val", "test"):
        idx = np.where(masks[split])[0]
        pred = clf.predict(X_all[idx].numpy())
        m = calc_metrics(y[idx], pred)
        res[f"{split}_acc"] = m["acc"]
        if split == "test":
            res["f1"] = m["macro_f1"]
            res["test_bacc"] = m["bacc"]
            res["normal_recall"] = m["normal_recall"]
            res["attack_recall"] = m["attack_recall"]
    print(f"  [SVM] {res}")
    return res


def run_rf_gnn(X_all, adj, y, masks, seed=42):
    """GNN embeddings + RandomForest head (Random-Forest-GNN row)."""
    set_seed(seed)
    gnn = GNNModel(X_all.shape[1], 128, 64).to(DEVICE)
    model = GNNClassifier(gnn, 64, 2).to(DEVICE)
    train_model(model, X_all, adj, y, masks, tag="GNN")
    with torch.no_grad():
        emb = gnn(X_all, adj).cpu().numpy()
    tr_idx = np.where(masks["train"])[0]
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(emb[tr_idx], y[tr_idx])
    p_all = rf.predict_proba(emb)[:, 1].astype(np.float64)
    res = {"loss": np.nan}
    for split in ("train", "val", "test"):
        idx = np.where(masks[split])[0]
        pred = (p_all[idx] > 0.5).astype(int)
        m = calc_metrics(y[idx], pred)
        res[f"{split}_acc"] = m["acc"]
        if split == "test":
            res["f1"] = m["macro_f1"]
            res["test_bacc"] = m["bacc"]
            res["normal_recall"] = m["normal_recall"]
            res["attack_recall"] = m["attack_recall"]
    print(f"  [RF-GNN] {res}")
    return res, p_all


def run_jm_llm(X_all, y, masks, seed=42, return_models=False):
    """JM-LLM closed-loop collaborative model (Section 3.3)."""
    set_seed(seed)
    tr_idx = np.where(masks["train"])[0]
    X_tr = X_all[tr_idx]
    y_tr = y[tr_idx]
    y_tr_f = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
    y_tr_l = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)

    jordan = JordanModel(X_all.shape[1], 1).to(DEVICE)
    lm = LMModel(X_all.shape[1], 2).to(DEVICE)
    mse = nn.MSELoss()
    counts = np.bincount(y_tr, minlength=2).astype(np.float32)
    ce = nn.CrossEntropyLoss(
        weight=torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                            dtype=torch.float32).to(DEVICE))
    opt_j = torch.optim.Adam(jordan.parameters(), lr=1e-3)
    opt_l = torch.optim.Adam(lm.parameters(), lr=1e-3)

    threshold, max_iter, total_err, it = 0.15, 30, float("inf"), 0
    while total_err > threshold and it < max_iter:
        jp = batched_predict(jordan, X_tr)
        lp = batched_predict(lm, X_tr, softmax=True)
        range_err = float(np.mean((y_tr.reshape(-1, 1) - jp) ** 2))
        node_err = float(np.mean((jp > 0.5).astype(int).ravel() != y_tr))
        dyn_err = float(np.mean((jp - lp.argmax(1).reshape(-1, 1)) ** 2))
        total_err = 0.4 * range_err + 0.4 * node_err + 0.2 * dyn_err
        print(f"  [JM-LLM] iteration {it + 1}: total error = {total_err:.4f}")
        if total_err <= threshold:
            break
        for _ in range(10):
            opt_j.zero_grad()
            loss = mse(jordan(X_tr), y_tr_f.view(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(jordan.parameters(), 5.0)
            opt_j.step()

            perm = np.random.permutation(len(X_tr))
            for i in range(0, len(perm), BATCH):
                part = torch.tensor(perm[i:i + BATCH]).to(DEVICE)
                opt_l.zero_grad()
                loss_l = ce(lm(X_tr[part]), y_tr_l[part])
                loss_l.backward()
                torch.nn.utils.clip_grad_norm_(lm.parameters(), 5.0)
                opt_l.step()
        it += 1

    def proba(idx):
        jp = batched_predict(jordan, X_all[idx]).ravel()
        lp = batched_predict(lm, X_all[idx], softmax=True)[:, 1]
        return 0.5 * jp + 0.5 * lp

    val_idx = np.where(masks["val"])[0]
    val_p = proba(val_idx)
    thr = max(np.linspace(0.05, 0.95, 19),
              key=lambda t: f1_score(y[val_idx], (val_p > t).astype(int),
                                     average="macro"))

    res = {"loss": float(mse(jordan(X_tr), y_tr_f.view(-1, 1)).item())}
    p_all = proba(np.arange(len(y)))
    for split in ("train", "val", "test"):
        idx = np.where(masks[split])[0]
        pred = (p_all[idx] > thr).astype(int)
        m = calc_metrics(y[idx], pred)
        res[f"{split}_acc"] = m["acc"]
        if split == "test":
            res["f1"] = m["macro_f1"]
            res["test_bacc"] = m["bacc"]
            res["normal_recall"] = m["normal_recall"]
            res["attack_recall"] = m["attack_recall"]
    print(f"  [JM-LLM] thr={thr:.2f} {res}")
    if return_models:
        return res, p_all, {"jordan": jordan, "lm": lm, "thr": float(thr)}
    return res, p_all


# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test on a small subsample")
    ap.add_argument("--protocol", choices=["official", "random"],
                    default="official",
                    help="official = train on UNSW_NB15_training-set and test on "
                         "the held-out testing-set; random = stratified 80/10/10 "
                         "over the combined corpus (the split used by the paper)")
    ap.add_argument("--p1", action="store_true",
                    help="P1: Platt calibration on validation + threshold tuning")
    ap.add_argument("--bcts", choices=["none", "em", "bbse"], default="none",
                    help="P1 target-prior estimation (implies --p1); the prior is "
                         "estimated from unlabelled predictions only")
    ap.add_argument("--normal-profile", action="store_true",
                    help="P2.5: append AutoEncoder normal-profile score to the "
                         "JM-LLM input features")
    ap.add_argument("--suffix", default="",
                    help="suffix for the output csv (e.g. _p1)")
    args = ap.parse_args()
    if args.bcts != "none":
        args.p1 = True

    set_seed()
    X_tr, y_tr, X_te, y_te = common.load_full_dataset()
    if args.quick:
        rng = np.random.default_rng(0)
        keep_tr = rng.choice(len(y_tr), 6000, replace=False)
        keep_te = rng.choice(len(y_te), 2000, replace=False)
        X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr]
        X_te, y_te = X_te[keep_te], y_te[keep_te]
        print(f"quick mode: train={X_tr.shape} test={X_te.shape}")
    X_all_np = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    n_tr = len(y_tr)

    masks = common.make_masks(n_tr, y_tr)
    full_masks = {}
    for split in ("train", "val"):
        m = np.zeros(len(y_all), dtype=bool)
        m[:n_tr] = masks[split]
        full_masks[split] = m
    test_mask = np.zeros(len(y_all), dtype=bool)
    test_mask[n_tr:] = True
    full_masks["test"] = test_mask

    if args.protocol == "random":
        # stratified 80/10/10 over the combined corpus, as in the paper
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y_all))
        tr_i, rest = train_test_split(idx, test_size=0.2, random_state=42,
                                      stratify=y_all)
        va_i, te_i = train_test_split(rest, test_size=0.5, random_state=42,
                                      stratify=y_all[rest])
        full_masks = {}
        for name, part in [("train", tr_i), ("val", va_i), ("test", te_i)]:
            m = np.zeros(len(y_all), dtype=bool)
            m[part] = True
            full_masks[name] = m
        print(f"random split: train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    if args.normal_profile:
        # P2.5 -- fitted on training-split Normal flows only
        normal_mask = full_masks["train"] & (y_all == 0)
        t0 = time.time()
        z_score, _, _ = fit_normal_profile(X_all_np, normal_mask)
        print(f"  [P2.5] normal profile fitted on {int(normal_mask.sum())} "
              f"Normal flows in {time.time() - t0:.0f}s")
        X_jmllm_np = np.hstack([X_all_np, z_score[:, None]])
    else:
        X_jmllm_np = X_all_np

    t0 = time.time()
    adjacency = common.build_knn_adjacency(X_all_np, k_neighbors=3)
    print(f"graph built in {time.time() - t0:.0f}s")
    adj_norm = common.normalized_adjacency_sparse(adjacency)
    adj_t = scipy_to_torch_sparse(adj_norm)
    X_all = torch.tensor(X_all_np, dtype=torch.float32).to(DEVICE)

    t0 = time.time()
    jordan = common.JordanMatrixDecomposition(adjacency, n_components=N_COMPONENTS)
    print(f"Jordan decomposition (n={N_COMPONENTS}) in {time.time() - t0:.0f}s")

    results, probas = {}, {}
    results["JM"] = run_jm(X_all_np, y_all, full_masks, adjacency, jordan)
    results["SVM"] = run_svm(X_all, y_all, full_masks)

    set_seed()
    lstm = LSTMClassifier(X_all.shape[1]).to(DEVICE)
    # wd=0 for LSTM: Adam+L2 suppresses the recurrent gradients on this
    # architecture (verified empirically: wd=5e-4 freezes the model at ln 2)
    train_model(lstm, X_all, None, y_all, full_masks, tag="LSTM",
                weight_decay=0.0)
    results["LSTM"] = evaluate(lstm, X_all, None, y_all, full_masks)
    probas["LSTM"] = _proba(lstm, X_all, None, np.arange(len(y_all)))

    results["Random-Forest-GNN"], probas["Random-Forest-GNN"] = run_rf_gnn(
        X_all, adj_t, y_all, full_masks)

    set_seed()
    jm_gnn = CollaborativeModel(GNNModel(X_all.shape[1]).to(DEVICE), 64,
                                jordan).to(DEVICE)
    train_model(jm_gnn, X_all, adj_t, y_all, full_masks, tag="JM-GNN")
    results["JM-GNN"] = evaluate(jm_gnn, X_all, adj_t, y_all, full_masks)
    probas["JM-GNN"] = _proba(jm_gnn, X_all, adj_t, np.arange(len(y_all)))

    set_seed()
    from models import LSTMModel
    jm_lstm = CollaborativeModel(LSTMModel(X_all.shape[1]).to(DEVICE), 64,
                                 jordan).to(DEVICE)
    train_model(jm_lstm, X_all, None, y_all, full_masks, tag="JM-LSTM",
                weight_decay=0.0)
    results["JM-LSTM"] = evaluate(jm_lstm, X_all, None, y_all, full_masks)
    probas["JM-LSTM"] = _proba(jm_lstm, X_all, None, np.arange(len(y_all)))

    X_jmllm = torch.tensor(X_jmllm_np, dtype=torch.float32).to(DEVICE)
    results["JM-LLM"], probas["JM-LLM"] = run_jm_llm(X_jmllm, y_all, full_masks)

    if args.p1:
        for name in ("LSTM", "Random-Forest-GNN", "JM-GNN", "JM-LSTM",
                     "JM-LLM"):
            tag = f"{name}+P1"
            if args.bcts != "none":
                tag += f"+{args.bcts.upper()}"
            results[tag] = run_p1(probas[name], y_all, full_masks,
                                  method=args.bcts, tag=tag)

    cols = ["loss", "train_acc", "val_acc", "test_acc", "f1", "test_bacc",
            "normal_recall", "attack_recall"]
    table = pd.DataFrame(results).T.reindex(columns=cols)
    print(f"\n===== Full UNSW-NB15 Performance Comparison "
          f"({args.protocol} split) =====")
    print(table.round(4).to_string())
    out_csv = ("results_full.csv" if args.protocol == "official"
               else "results_full_random.csv")
    out_csv = out_csv.replace(".csv", f"{args.suffix}.csv")
    table.round(4).to_csv(out_csv, encoding="utf-8-sig")
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
