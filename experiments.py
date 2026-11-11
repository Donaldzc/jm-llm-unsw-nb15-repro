"""Additional experiments requested by the pre-submission review round.

Every experiment writes its raw output to ``experiments/`` (CSV/JSON/npz) so
that the numbers quoted in the manuscript can be traced back to a file.

Sub-commands
------------
cache   train the seven models for one seed and cache predictions
e1      per-class and per-attack-category metrics (needs a cache)
e2      repeated runs over several seeds (mean +- std)
e3      mode-guided attention: implementation and ablation
e4      directed (non-normal) propagation matrix vs symmetrised kNN graph
e5      random-graph and capacity-matched controls
e6      adversarial perturbation and false-positive cost

Usage
-----
python experiments.py cache --protocol official --seed 42
python experiments.py e1    --protocol official --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

import common
from common import scipy_to_torch_sparse
from correction import calc_metrics
from models import (DEVICE, LSTMClassifier, LSTMModel, batched_predict,
                    set_seed)
from run_full_comparison import (N_COMPONENTS, _proba, evaluate, run_jm_llm,
                                 run_rf_gnn, run_svm, train_model)
from models import CollaborativeModel, GNNModel

HERE = Path(__file__).resolve().parent
OUT = HERE / "experiments"
OUT.mkdir(exist_ok=True)
BATCH = 8192

MODELS = ["JM", "SVM", "LSTM", "Random-Forest-GNN", "JM-GNN", "JM-LSTM",
          "JM-LLM"]
KEY = {"JM": "JM", "SVM": "SVM", "LSTM": "LSTM",
       "Random-Forest-GNN": "RF_GNN", "JM-GNN": "JM_GNN",
       "JM-LSTM": "JM_LSTM", "JM-LLM": "JM_LLM"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_with_categories(protocol="official"):
    """Return X_all, y, attack_cat, full_masks, n_train."""
    train_df = pd.read_csv(common.FULL_TRAIN)
    test_df = pd.read_csv(common.FULL_TEST)
    train_df, _ = common._clean(train_df)
    test_df, _ = common._clean(test_df)
    X_tr, X_te = common._fit_transform(train_df, test_df)
    y_tr = train_df["label"].astype(int).values
    y_te = test_df["label"].astype(int).values
    cat_tr = train_df["attack_cat"].fillna("Normal").astype(str).values
    cat_te = test_df["attack_cat"].fillna("Normal").astype(str).values

    X_all = np.vstack([X_tr, X_te]).astype(np.float32)
    y = np.concatenate([y_tr, y_te])
    attack_cat = np.concatenate([cat_tr, cat_te])
    n_tr = len(y_tr)

    if protocol == "official":
        masks = common.make_masks(n_tr, y_tr)
        full = {}
        for split in ("train", "val"):
            m = np.zeros(len(y), dtype=bool)
            m[:n_tr] = masks[split]
            full[split] = m
        test = np.zeros(len(y), dtype=bool)
        test[n_tr:] = True
        full["test"] = test
    elif protocol == "random":
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        tr_i, rest = train_test_split(idx, test_size=0.2, random_state=42,
                                      stratify=y)
        va_i, te_i = train_test_split(rest, test_size=0.5, random_state=42,
                                      stratify=y[rest])
        full = {}
        for name, part in [("train", tr_i), ("val", va_i), ("test", te_i)]:
            m = np.zeros(len(y), dtype=bool)
            m[part] = True
            full[name] = m
    else:
        raise ValueError(protocol)
    return X_all, y, attack_cat, full, n_tr


def graph_and_jordan(X_all):
    t0 = time.time()
    adjacency = common.build_knn_adjacency(X_all, k_neighbors=3)
    print(f"graph built in {time.time() - t0:.0f}s")
    adj_norm = common.normalized_adjacency_sparse(adjacency)
    adj_t = scipy_to_torch_sparse(adj_norm)
    t0 = time.time()
    jordan = common.JordanMatrixDecomposition(adjacency,
                                              n_components=N_COMPONENTS)
    print(f"decomposition in {time.time() - t0:.0f}s")
    return adjacency, adj_t, jordan


# ---------------------------------------------------------------------------
# JM prediction (paper row "JM")
# ---------------------------------------------------------------------------
def jm_predict(X_all, y, masks, jordan):
    initial = np.zeros(len(y))
    initial[masks["train"] & (y == 1)] = 1.0
    predicted = jordan.predict_infection(initial, t=3)
    tr_idx = np.where(masks["train"])[0]
    nn1 = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_all[tr_idx])
    pred = np.zeros(len(y), dtype=np.int8)
    pred[tr_idx] = (predicted[tr_idx] > 0.5).astype(np.int8)
    for split in ("val", "test"):
        idx = np.where(masks[split])[0]
        _, nn = nn1.kneighbors(X_all[idx])
        pred[idx] = (predicted[tr_idx[nn.ravel()]] > 0.5).astype(np.int8)
    return pred


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def _part_path(protocol, seed, name):
    return OUT / f"part_{protocol}_{seed}_{KEY[name]}.npz"


def train_one(name, X_all, X_t, y, masks, adj_t, jordan, seed):
    """Train one model and return (binary predictions, probabilities|None)."""
    tr_idx = np.where(masks["train"])[0]
    if name == "JM":
        return jm_predict(X_all, y, masks, jordan), None
    if name == "SVM":
        svm = LinearSVC(C=1.0, max_iter=3000, dual="auto", random_state=seed)
        svm.fit(X_all[tr_idx], y[tr_idx])
        dec = svm.decision_function(X_all)
        return svm.predict(X_all).astype(np.int8), 1.0 / (1.0 + np.exp(-dec))
    if name == "LSTM":
        set_seed(seed)
        model = LSTMClassifier(X_t.shape[1]).to(DEVICE)
        train_model(model, X_t, None, y, masks, tag="LSTM", weight_decay=0.0)
        p = _proba(model, X_t, None, np.arange(len(y)))
        return (p > 0.5).astype(np.int8), p
    if name == "Random-Forest-GNN":
        _, p = run_rf_gnn(X_t, adj_t, y, masks, seed=seed)
        return (p > 0.5).astype(np.int8), p
    if name == "JM-GNN":
        set_seed(seed)
        model = CollaborativeModel(GNNModel(X_t.shape[1]).to(DEVICE), 64,
                                   jordan).to(DEVICE)
        train_model(model, X_t, adj_t, y, masks, tag="JM-GNN")
        p = _proba(model, X_t, adj_t, np.arange(len(y)))
        return (p > 0.5).astype(np.int8), p
    if name == "JM-LSTM":
        set_seed(seed)
        model = CollaborativeModel(LSTMModel(X_t.shape[1]).to(DEVICE), 64,
                                   jordan).to(DEVICE)
        train_model(model, X_t, None, y, masks, tag="JM-LSTM",
                    weight_decay=0.0)
        p = _proba(model, X_t, None, np.arange(len(y)))
        return (p > 0.5).astype(np.int8), p
    if name == "JM-LLM":
        _, p = run_jm_llm(X_t, y, masks, seed=seed)
        val = np.where(masks["val"])[0]
        p = p.astype(np.float32)
        thr = max(np.linspace(0.05, 0.95, 19),
                  key=lambda t: f1_score(y[val], (p[val] > t).astype(int),
                                         average="macro"))
        print(f"  [JM-LLM] validation-tuned threshold {thr:.2f}", flush=True)
        return (p > thr).astype(np.int8), p
    raise ValueError(name)


def build_cache(protocol="official", seed=42, force=False, only=None):
    path = OUT / f"cache_{protocol}_seed{seed}.npz"
    if path.exists() and not force and only is None:
        print(f"cache exists: {path}")
        return path
    X_all, y, attack_cat, masks, n_tr = load_with_categories(protocol)
    print(f"{protocol}: X={X_all.shape} train={masks['train'].sum()} "
          f"val={masks['val'].sum()} test={masks['test'].sum()}", flush=True)
    adjacency, adj_t, jordan = graph_and_jordan(X_all)
    X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)

    names = MODELS if only is None else [only]
    for name in names:
        part = _part_path(protocol, seed, name)
        if part.exists() and not force:
            print(f"  [{name}] cached", flush=True)
            continue
        t0 = time.time()
        pred, prob = train_one(name, X_all, X_t, y, masks, adj_t, jordan, seed)
        extra = {"proba": prob.astype(np.float32)} if prob is not None else {}
        np.savez_compressed(part, pred=pred.astype(np.int8), **extra)
        print(f"  [{name}] {time.time() - t0:.0f}s -> {part.name}", flush=True)

    parts = [_part_path(protocol, seed, n) for n in MODELS]
    if not all(p.exists() for p in parts):
        print("partial cache saved; rerun to complete", flush=True)
        return None

    payload = {"y": y.astype(np.int8),
               "attack_cat": attack_cat.astype("U16"),
               "split": np.where(masks["train"], 0,
                                 np.where(masks["val"], 1, 2)).astype(np.int8),
               "seed": np.array(seed), "protocol": np.array(protocol)}
    for name, p in zip(MODELS, parts):
        z = np.load(p, allow_pickle=True)
        payload[f"pred_{KEY[name]}"] = z["pred"]
        if "proba" in z.files:
            payload[f"proba_{KEY[name]}"] = z["proba"]
    np.savez_compressed(path, **payload)
    print(f"saved -> {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# E1: per-class and per-attack-category metrics
# ---------------------------------------------------------------------------
def run_e1(protocol="official", seed=42):
    path = OUT / f"cache_{protocol}_seed{seed}.npz"
    if not path.exists():
        build_cache(protocol, seed)
    d = np.load(path, allow_pickle=True)
    y = d["y"].astype(int)
    cat = d["attack_cat"].astype(str)
    split = d["split"]
    te = split == 2

    rows, cm_rows = [], []
    for name in MODELS:
        pred = d[f"pred_{KEY[name]}"].astype(int)
        rep = classification_report(y[te], pred[te], labels=[0, 1],
                                    target_names=["Normal", "Attack"],
                                    output_dict=True, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y[te], pred[te],
                                          labels=[0, 1]).ravel()
        rows.append({
            "model": name,
            "normal_precision": rep["Normal"]["precision"],
            "normal_recall": rep["Normal"]["recall"],
            "normal_f1": rep["Normal"]["f1-score"],
            "attack_precision": rep["Attack"]["precision"],
            "attack_recall": rep["Attack"]["recall"],
            "attack_f1": rep["Attack"]["f1-score"],
            "accuracy": rep["accuracy"],
            "macro_f1": rep["macro avg"]["f1-score"],
            "fpr": fp / max(fp + tn, 1),
            "fnr": fn / max(fn + tp, 1),
        })
        cm_rows.append({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp})

    pd.DataFrame(rows).round(4).to_csv(OUT / f"e1_per_class_{protocol}.csv",
                                       index=False, encoding="utf-8-sig")
    pd.DataFrame(cm_rows).to_csv(OUT / f"e1_confusion_{protocol}.csv",
                                 index=False, encoding="utf-8-sig")

    cats = sorted(set(cat[te]) - {"Normal"})
    cat_rows = []
    for name in MODELS:
        pred = d[f"pred_{KEY[name]}"].astype(int)
        row = {"model": name}
        for c in cats:
            m = (cat == c) & te
            row[c] = float(pred[m].mean()) if m.sum() else np.nan
        m_normal = (cat == "Normal") & te
        row["Normal TP rate (1-FPR)"] = float((pred[m_normal] == 0).mean())
        cat_rows.append(row)
    cat_rows.append({"model": "support",
                     **{c: int(((cat == c) & te).sum()) for c in cats},
                     "Normal TP rate (1-FPR)": int(((cat == "Normal") & te).sum())})
    pd.DataFrame(cat_rows).round(4).to_csv(
        OUT / f"e1_attack_categories_{protocol}.csv", index=False,
        encoding="utf-8-sig")

    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print()
    print(pd.DataFrame(cat_rows).round(3).to_string(index=False))
    return rows, cat_rows


def run_e1_jm_stability(protocol="official", repeats=5):
    """Re-decompose the same graph several times and score the JM row.

    ``scipy.sparse.linalg.eigsh`` is started from a random vector, so the
    retained eigenpairs of a graph with clustered top eigenvalues are not
    bit-reproducible.  This measures how far the eigen-only row moves.
    """
    X_all, y, _, masks, _ = load_with_categories(protocol)
    adjacency = common.build_knn_adjacency(X_all, k_neighbors=3)
    rows = []
    for r in range(repeats):
        jordan = common.JordanMatrixDecomposition(adjacency,
                                                  n_components=N_COMPONENTS)
        pred = jm_predict(X_all, y, masks, jordan)
        m = calc_metrics(y[masks["test"]], pred[masks["test"]])
        rows.append({"repeat": r + 1, "test_acc": m["acc"],
                     "macro_f1": m["macro_f1"],
                     "normal_recall": m["normal_recall"],
                     "attack_recall": m["attack_recall"]})
        print(f"  repeat {r + 1}: acc={m['acc']:.4f} "
              f"macro_f1={m['macro_f1']:.4f}", flush=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(OUT / f"e1_jm_stability_{protocol}.csv", index=False,
              encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(df[["test_acc", "macro_f1"]].agg(["mean", "std", "min", "max"]))
    return df


# ---------------------------------------------------------------------------
# E2: repeated runs
# ---------------------------------------------------------------------------
def run_e2(protocol="official", seeds=(1, 2, 3, 4, 5)):
    frames = []
    for s in seeds:
        p = build_cache(protocol, s)
        d = np.load(p, allow_pickle=True)
        y = d["y"].astype(int)
        te = d["split"] == 2
        for name in MODELS:
            pred = d[f"pred_{KEY[name]}"].astype(int)
            m = calc_metrics(y[te], pred[te])
            frames.append({"seed": s, "model": name,
                           "test_acc": m["acc"], "macro_f1": m["macro_f1"],
                           "fpr": 1 - m["normal_recall"]})
    df = pd.DataFrame(frames)
    df.to_csv(OUT / f"e2_per_seed_{protocol}.csv", index=False,
              encoding="utf-8-sig")
    rows = []
    for name in MODELS:                      # keep in the order of the tables
        sub = df[df["model"] == name]
        rows.append({"model": name, "n_seeds": len(sub),
                     "test_acc_mean": sub["test_acc"].mean(),
                     "test_acc_std": sub["test_acc"].std(ddof=1),
                     "macro_f1_mean": sub["macro_f1"].mean(),
                     "macro_f1_std": sub["macro_f1"].std(ddof=1),
                     "fpr_mean": sub["fpr"].mean(),
                     "fpr_std": sub["fpr"].std(ddof=1)})
    agg = pd.DataFrame(rows).round(4)
    agg.to_csv(OUT / f"e2_mean_std_{protocol}.csv", index=False,
               encoding="utf-8-sig")
    print(agg.to_string(index=False))

    # paired per-seed differences (same seed, same split) against the two
    # reference models that the claims in the paper are phrased against
    wide = df.pivot(index="seed", columns="model", values="test_acc")
    p_rows = []
    for a, b in [("JM-LLM", "LSTM"), ("JM-LLM", "Random-Forest-GNN"),
                 ("JM-LLM", "JM-LSTM"), ("LSTM", "JM-LSTM"),
                 ("Random-Forest-GNN", "JM-GNN")]:
        diff = wide[a] - wide[b]
        p_rows.append({
            "comparison": f"{a} minus {b}",
            "mean_diff": diff.mean(), "sd_diff": diff.std(ddof=1),
            "se_5seeds": diff.std(ddof=1) / np.sqrt(len(diff)),
            "t_stat": diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))),
            "wins": int((diff > 0).sum()), "n_seeds": len(diff)})
    paired = pd.DataFrame(p_rows).round(4)
    paired.to_csv(OUT / f"e2_paired_{protocol}.csv", index=False,
                  encoding="utf-8-sig")
    print(paired.to_string(index=False))
    return agg


# ---------------------------------------------------------------------------
# E3: mode-guided attention (implementation + ablation)
# ---------------------------------------------------------------------------
class AttentionEncoder(nn.Module):
    """Compact encoder with mode-guided cross-attention (paper Sec. 3.4).

    sim(h_i, V_i) = cos(W_q h_i, W_k p_i) / tau, where p_i is the mean input
    feature vector of the nodes assigned to mode i.  The enhanced hidden state
    is h_i + sum_i alpha_i W_v p_i.
    """

    def __init__(self, input_dim, n_modes, tau=0.5, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(input_dim, 64)
        self.v = nn.Linear(input_dim, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.tau = tau
        self.n_modes = n_modes

    def forward(self, x, protos):
        import torch.nn.functional as F
        h = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        q = F.normalize(self.q(h), dim=1)
        k = F.normalize(self.k(protos), dim=1)
        alpha = torch.softmax(q @ k.t() / self.tau, dim=1)
        ctx = alpha @ self.v(protos)
        return self.classifier(h + ctx), alpha


class AttentionLMModel(nn.Module):
    """Semantic branch with attention; same interface as models.LMModel."""

    def __init__(self, input_dim, n_modes, tau=0.5, num_classes=2):
        super().__init__()
        self.encoder = AttentionEncoder(input_dim, n_modes, tau, num_classes)
        self.protos = None

    def forward(self, x):
        out, _ = self.encoder(x, self.protos)
        return out

    def forward_alpha(self, x):
        return self.encoder(x, self.protos)


def mode_prototypes(X, trans_matrix, n_modes):
    """Mean input vector of the nodes assigned to each mode (Sec. 3.4)."""
    assign = np.abs(trans_matrix[:, :n_modes]).argmax(1)
    protos = np.zeros((n_modes, X.shape[1]), dtype=np.float32)
    for c in range(n_modes):
        m = assign == c
        protos[c] = X[m].mean(0) if m.any() else X.mean(0)
    return protos, assign


def train_jmllm_attn(X_all, y, masks, jordan, seed=42, tau=0.5):
    """JM-LLM with mode-guided attention instead of the plain MLP branch."""
    import torch.nn.functional as F
    set_seed(seed)
    tr = np.where(masks["train"])[0]
    va = np.where(masks["val"])[0]
    X_tr, y_tr = X_all[tr], y[tr]
    y_tr_f = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
    y_tr_l = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    protos_np, assign = mode_prototypes(X_all.cpu().numpy(), jordan.trans_matrix,
                                        jordan.trans_matrix.shape[1])
    protos = torch.tensor(protos_np, dtype=torch.float32).to(DEVICE)

    from models import JordanModel
    jordan_m = JordanModel(X_all.shape[1], 1).to(DEVICE)
    lm = AttentionLMModel(X_all.shape[1], protos.shape[0], tau=tau).to(DEVICE)
    lm.protos = protos
    mse = nn.MSELoss()
    counts = np.bincount(y_tr, minlength=2).astype(np.float32)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(
        counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32).to(DEVICE))
    opt_j = torch.optim.Adam(jordan_m.parameters(), lr=1e-3)
    opt_l = torch.optim.Adam(lm.parameters(), lr=1e-3)

    # sigma_0: spread of the attention distribution over the validation split
    with torch.no_grad():
        _, a_val = lm.forward_alpha(X_all[va])
        sigma0 = float(a_val.std(0).mean().item())
    prev_mean = None
    threshold, max_iter, total_err, it = 0.15, 30, float("inf"), 0
    while total_err > threshold and it < max_iter:
        jp = batched_predict(jordan_m, X_tr)
        lp = batched_predict(lm, X_tr, softmax=True)
        range_err = float(np.mean((y_tr.reshape(-1, 1) - jp) ** 2))
        node_err = float(np.mean((jp > 0.5).astype(int).ravel() != y_tr))
        dyn_err = float(np.mean((jp - lp.argmax(1).reshape(-1, 1)) ** 2))
        total_err = 0.4 * range_err + 0.4 * node_err + 0.2 * dyn_err
        with torch.no_grad():
            alpha_mean = lm.forward_alpha(X_tr[:4096])[1].mean(0)
        fluct = np.inf if prev_mean is None else float(
            (alpha_mean.cpu().numpy() - prev_mean).__abs__().sum())
        prev_mean = alpha_mean.cpu().numpy()
        print(f"  [JM-LLM-Attn] iter {it + 1}: err={total_err:.4f} "
              f"sigma0={sigma0:.4f} fluct={fluct:.4f}")
        if total_err <= threshold:
            break
        if fluct < sigma0:      # modal fluctuation below sigma_0: freeze attention
            for p in lm.encoder.q.parameters():
                p.requires_grad_(False)
        for _ in range(10):
            opt_j.zero_grad()
            mse(jordan_m(X_tr), y_tr_f.view(-1, 1)).backward()
            torch.nn.utils.clip_grad_norm_(jordan_m.parameters(), 5.0)
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
        jp = batched_predict(jordan_m, X_all[idx]).ravel()
        lp = batched_predict(lm, X_all[idx], softmax=True)[:, 1]
        return 0.5 * jp + 0.5 * lp

    val_p = proba(va)
    thr = max(np.linspace(0.05, 0.95, 19),
              key=lambda t: f1_score(y[va], (val_p > t).astype(int),
                                     average="macro"))
    p_all = proba(np.arange(len(y)))
    pred = (p_all > thr).astype(int)
    res = {}
    for split in ("train", "val", "test"):
        idx = np.where(masks[split])[0]
        m = calc_metrics(y[idx], pred[idx])
        res[f"{split}_acc"] = m["acc"]
        if split == "test":
            res["f1"] = m["macro_f1"]
    res.update({"threshold": thr, "sigma0": sigma0, "rounds": it})
    return res, p_all


def run_e3(protocol="official", seeds=(1, 2, 3, 4, 5)):
    """Attention on/off ablation, paired over seeds.

    The attention branch and the plain semantic branch are trained with the
    same seed and the same data split, so the difference per seed is paired.
    """
    from scipy import stats as st
    import models as _models

    X_all, y, cat, masks, _ = load_with_categories(protocol)
    adjacency, adj_t, jordan = graph_and_jordan(X_all)
    X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)

    n_modes = int(jordan.trans_matrix.shape[1])
    n_attn = sum(p.numel() for p in
                 AttentionLMModel(X_t.shape[1], n_modes).parameters())
    n_plain = sum(p.numel() for p in _models.LMModel(X_t.shape[1]).parameters())

    frames, metas = [], {}
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        res, _ = train_jmllm_attn(X_t, y, masks, jordan, seed=seed)
        d = np.load(build_cache(protocol, seed), allow_pickle=True)
        te = d["split"] == 2
        base = calc_metrics(y[te], d["pred_JM_LLM"].astype(int)[te])
        frames.append({
            "seed": seed,
            "attn_test_acc": res["test_acc"], "attn_macro_f1": res["f1"],
            "base_test_acc": base["acc"], "base_macro_f1": base["macro_f1"],
            "diff_acc_pp": (res["test_acc"] - base["acc"]) * 100,
            "threshold": res["threshold"], "rounds": res["rounds"],
            "sigma0": res["sigma0"]})
        metas[str(seed)] = res
    df = pd.DataFrame(frames).round(4)
    df.to_csv(OUT / f"e3_attention_per_seed_{protocol}.csv", index=False,
              encoding="utf-8-sig")

    d_acc = df["attn_test_acc"] - df["base_test_acc"]
    d_f1 = df["attn_macro_f1"] - df["base_macro_f1"]
    t_acc = st.ttest_rel(df["attn_test_acc"], df["base_test_acc"])
    t_f1 = st.ttest_rel(df["attn_macro_f1"], df["base_macro_f1"])
    summary = pd.DataFrame([
        {"variant": "JM-LLM (equal-weight fusion, no attention)",
         "n_seeds": len(df), "test_acc_mean": df["base_test_acc"].mean(),
         "test_acc_std": df["base_test_acc"].std(ddof=1),
         "macro_f1_mean": df["base_macro_f1"].mean(),
         "macro_f1_std": df["base_macro_f1"].std(ddof=1)},
        {"variant": "JM-LLM + mode-guided attention",
         "n_seeds": len(df), "test_acc_mean": df["attn_test_acc"].mean(),
         "test_acc_std": df["attn_test_acc"].std(ddof=1),
         "macro_f1_mean": df["attn_macro_f1"].mean(),
         "macro_f1_std": df["attn_macro_f1"].std(ddof=1)},
    ]).round(4)
    summary.to_csv(OUT / f"e3_attention_ablation_{protocol}.csv", index=False,
                   encoding="utf-8-sig")

    paired = pd.DataFrame([
        {"metric": "test_acc", "mean_diff_pp": d_acc.mean() * 100,
         "sd_diff_pp": d_acc.std(ddof=1) * 100, "t_stat": t_acc.statistic,
         "p_value": t_acc.pvalue, "wins": int((d_acc > 0).sum()),
         "n_seeds": len(d_acc)},
        {"metric": "macro_f1", "mean_diff_pp": d_f1.mean() * 100,
         "sd_diff_pp": d_f1.std(ddof=1) * 100, "t_stat": t_f1.statistic,
         "p_value": t_f1.pvalue, "wins": int((d_f1 > 0).sum()),
         "n_seeds": len(d_f1)},
    ]).round(4)
    paired.to_csv(OUT / f"e3_attention_paired_{protocol}.csv", index=False,
                  encoding="utf-8-sig")

    with open(OUT / f"e3_attention_meta_{protocol}.json", "w",
              encoding="utf-8") as f:
        json.dump({"n_modes": n_modes, "tau": 0.5,
                   "parameters_attention": n_attn,
                   "parameters_plain": n_plain, "per_seed": metas}, f,
                  indent=2)

    print(df.to_string(index=False))
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# E4: directed (non-normal) propagation matrix
# ---------------------------------------------------------------------------
def run_e4(replica=1500, seed=0):
    """Spectrum and Jordan structure of the directed propagation matrix.

    The full graph has 257k nodes, and ARPACK does not converge on its
    non-symmetric version because the eigenvalues cluster at unit modulus, so
    the eigenvalue analysis is carried out on a 1,500-record replica built with
    the same kNN construction, while the non-normality statistic and the
    propagation comparison use the full graph.
    """
    from scipy.sparse import diags

    X, y, cat, masks, _ = load_with_categories("official")
    adj = common.build_knn_adjacency(X, k_neighbors=3)      # directed
    n = adj.shape[0]
    deg = np.asarray(adj.sum(1)).ravel()
    A = (diags(1.0 / np.maximum(deg, 1e-12)) @ adj).tocsr().astype(np.float64)

    # non-normality on the full graph: ||A^T A v - A A^T v|| / ||A v||
    rng = np.random.default_rng(seed)
    etas = []
    for _ in range(5):
        v = rng.standard_normal(n)
        av = A @ v
        etas.append(float(np.linalg.norm(A.T @ av - A @ (A.T @ v)) /
                          max(np.linalg.norm(av), 1e-12)))

    # spectrum on a replica of the same construction
    sub = np.sort(rng.choice(n, size=replica, replace=False))
    R = common.build_knn_adjacency(X[sub], k_neighbors=3, verbose=False)
    rd = np.asarray(R.sum(1)).ravel()
    Ar = (diags(1.0 / np.maximum(rd, 1e-12)) @ R).toarray().astype(np.float64)
    ev = np.linalg.eigvals(Ar)
    ev = ev[np.argsort(-np.abs(ev))]
    print(f"replica spectrum computed ({len(ev)} eigenvalues)", flush=True)

    # cluster the spectrum at 1e-6 and compare algebraic with geometric
    # multiplicity: a defective (Jordan) eigenvalue has geometric < algebraic.
    # Numerical zeros are excluded from that count, because a rank-deficient
    # graph Laplacian trivially produces a large zero eigenspace.
    clusters: list[list[int]] = []
    for i, lam in enumerate(ev):
        placed = False
        for c in clusters:
            if abs(lam - ev[c[0]]) < 1e-6:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    eye = np.eye(replica)
    normA = np.linalg.norm(Ar)
    defective, worst = 0, 0
    for c in clusters:
        lam = ev[c[0]]
        alg = len(c)
        if alg < 2 or abs(lam) < 1e-3:
            continue
        geo = replica - np.linalg.matrix_rank(Ar - lam * eye,
                                              tol=1e-8 * normA)
        worst = max(worst, alg - geo)
        if geo < alg:
            defective += 1

    # leading part of the spectrum: the block orders the paper would use
    lead = [c for c in clusters if abs(ev[c[0]]) >= 1e-3][:26]
    lead_worst = 0
    for c in lead:
        lam = ev[c[0]]
        alg = len(c)
        if alg < 2:
            continue
        geo = replica - np.linalg.matrix_rank(Ar - lam * eye,
                                              tol=1e-8 * normA)
        lead_worst = max(lead_worst, alg - geo)

    top = ev[:200]
    gaps = [float(np.min(np.abs(top[i] - top[i + 1:])))
            for i in range(len(top) - 1)]

    nonnull = [c for c in clusters if abs(ev[c[0]]) >= 1e-3]
    unit = min(nonnull, key=lambda c: abs(ev[c[0]] - 1.0))
    top_spectrum = [{"modulus": round(float(abs(ev[c[0]])), 6),
                     "real": round(float(ev[c[0]].real), 6),
                     "imag": round(float(ev[c[0]].imag), 6),
                     "algebraic": len(c)} for c in nonnull[:10]]

    # propagation: three diffusion steps on the directed matrix against the
    # symmetrised Jordan reconstruction, both scored on the same transfer rule
    # and with a threshold tuned on the validation split
    dir_score = _diffusion_score(X, y, masks, A, t=3)
    sym_score = _jm_score(X, y, masks, common.JordanMatrixDecomposition(
        adj, n_components=N_COMPONENTS), t=3)
    m_dir, thr_dir = _tuned_metrics(dir_score, y, masks)
    m_sym, thr_sym = _tuned_metrics(sym_score, y, masks)
    m_dir_05 = calc_metrics(y[masks["test"]],
                            (dir_score[masks["test"]] > 0.5).astype(int))
    m_sym_05 = calc_metrics(y[masks["test"]],
                            (sym_score[masks["test"]] > 0.5).astype(int))

    out = {
        "n_nodes": int(n),
        "n_replica": int(replica),
        "n_modes": N_COMPONENTS,
        "non_normality_eta": float(np.mean(etas)),
        "max_abs_imag": float(np.max(np.abs(ev.imag))),
        "n_distinct_eigenvalues": len(clusters),
        "n_repeated_clusters": int(sum(len(c) > 1 for c in clusters)),
        "max_algebraic_multiplicity": int(max(len(c) for c in clusters)),
        "max_algebraic_multiplicity_nonzero": int(max(len(c) for c in nonnull)),
        "unit_eigenvalue_multiplicity": len(unit),
        "unit_eigenvalue_geometric": int(replica - np.linalg.matrix_rank(
            Ar - ev[unit[0]] * eye, tol=1e-8 * normA)),
        "n_eigenvalues_above_0_99": int(np.sum(np.abs(ev) > 0.99)),
        "top_spectrum": top_spectrum,
        "n_defective_clusters": int(defective),
        "max_algebraic_minus_geometric": int(worst),
        "leading_block_order_above_one": int(lead_worst),
        "n_zero_eigenvalues": int(np.sum(np.abs(ev) < 1e-8)),
        "min_eigenvalue_gap": float(min(gaps)) if gaps else float("nan"),
        "n_modes_near_unit_circle": int(np.sum(np.abs(ev) > 1 - 1e-3)),
        "directed_diffusion_test_acc": m_dir["acc"],
        "directed_diffusion_macro_f1": m_dir["macro_f1"],
        "directed_threshold": float(thr_dir),
        "directed_acc_at_0.5": m_dir_05["acc"],
        "jm_symmetric_test_acc": m_sym["acc"],
        "jm_symmetric_macro_f1": m_sym["macro_f1"],
        "jm_symmetric_threshold": float(thr_sym),
        "jm_symmetric_acc_at_0.5": m_sym_05["acc"],
    }
    with open(OUT / "e4_directed_spectrum.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


def _transfer_score(X, score, masks):
    """Give each validation and test node the score of its nearest training
    node, which is the labelling rule used for the eigen-only model."""
    tr_idx = np.where(masks["train"])[0]
    nn1 = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X[tr_idx])
    out = score.astype(np.float64).copy()
    for split in ("val", "test"):
        idx = np.where(masks[split])[0]
        _, nn = nn1.kneighbors(X[idx])
        out[idx] = score[tr_idx[nn.ravel()]]
    return out


def _tuned_metrics(score, y, masks):
    va = np.where(masks["val"])[0]
    thr = max(np.linspace(0.05, 0.95, 19),
              key=lambda t: f1_score(y[va], (score[va] > t).astype(int),
                                     average="macro"))
    return calc_metrics(y[masks["test"]], (score[masks["test"]] > thr).astype(int)), thr


def _diffusion_score(X, y, masks, A, t=3):
    """Three steps of infection diffusion on the directed matrix."""
    n = A.shape[0]
    v = np.zeros(n, dtype=np.float64)
    v[masks["train"] & (y == 1)] = 1.0
    score = v.copy()
    for _ in range(t):
        score = A @ score
    return _transfer_score(X, score, masks)


def _jm_score(X, y, masks, jordan, t=3):
    """The eigen-only branch: propagate the initial infection through the
    Jordan form and transfer the score as above."""
    initial = np.zeros(len(y), dtype=np.float64)
    initial[masks["train"] & (y == 1)] = 1.0
    return _transfer_score(X, jordan.predict_infection(initial, t=t), masks)


# ---------------------------------------------------------------------------
# E5: random-graph and capacity-matched controls
# ---------------------------------------------------------------------------
class PlainMLP(nn.Module):
    """Feature-only MLP with the funnel shape and parameter count of the
    semantic branch of JM-LLM.

    The default is deliberately identical to ``models.LMModel`` in shape
    (input -> W -> W/2 -> 2, ReLU, no dropout); only the width changes, so the
    control differs from the semantic branch in capacity alone and not in
    regularisation.
    """

    def __init__(self, input_dim, width=97, num_classes=2, dropout=0.0):
        super().__init__()
        hidden = max(width // 2, 8)
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(width, hidden), nn.ReLU(),
                   nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, adj=None):
        return {"class_pred": self.net(x)}


def shuffled_graph(adj, seed=0):
    """Same in/out degrees, randomly rewired (configuration-style)."""
    from scipy.sparse import coo_matrix
    coo = adj.tocoo()
    rng = np.random.default_rng(seed)
    cols = coo.col.copy()
    rng.shuffle(cols)
    keep = coo.row != cols
    m = coo_matrix((coo.data[keep], (coo.row[keep], cols[keep])),
                   shape=adj.shape).tocsr()
    return m.maximum(m.T)


def _n_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def match_width(input_dim, target_params, lo=16, hi=512):
    """Smallest MLP width whose parameter count is closest to the target."""
    best = min(range(lo, hi + 1),
               key=lambda w: abs(_n_params(PlainMLP(input_dim, w)) -
                                 target_params))
    return best, _n_params(PlainMLP(input_dim, best))


def run_e5(protocol="official", seeds=(1, 2, 3, 4, 5)):
    """Random-graph and capacity-matched controls, paired over seeds.

    Two questions are separated here: whether the neighbourhood structure
    carries information (kNN graph vs a shuffled graph with the same edge
    weights) and whether the attention gain of E3 survives when the plain
    branch is given the same number of parameters.
    """
    from scipy import stats as st
    import models as _models

    X, y, cat, masks, _ = load_with_categories(protocol)
    adj = common.build_knn_adjacency(X, k_neighbors=3)
    jordan = common.JordanMatrixDecomposition(adj, n_components=N_COMPONENTS)
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    n_plain = _n_params(_models.LMModel(X.shape[1]))
    n_attn = _n_params(AttentionLMModel(X.shape[1], N_COMPONENTS))
    w_plain, p_plain = match_width(X.shape[1], n_plain)
    w_attn, p_attn = match_width(X.shape[1], n_attn)
    print(f"targets: plain={n_plain} (width {w_plain}={p_plain}), "
          f"attention={n_attn} (width {w_attn}={p_attn})", flush=True)

    frames = []
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        part = OUT / f"e5_part_{protocol}_{seed}.npz"
        if part.exists():
            z = np.load(part)
            p_rg, probs = z["p_rg"], {"mlp_plain": z["p_plain"],
                                      "mlp_attn": z["p_attn"]}
            print("  cached", flush=True)
        else:
            radj = shuffled_graph(adj, seed=seed)
            radj_t = scipy_to_torch_sparse(
                common.normalized_adjacency_sparse(radj))
            set_seed(seed)
            rg = CollaborativeModel(GNNModel(X_t.shape[1]).to(DEVICE), 64,
                                    jordan).to(DEVICE)
            train_model(rg, X_t, radj_t, y, masks,
                        tag=f"JM-GNN-random-graph-{seed}")
            p_rg = _proba(rg, X_t, radj_t, np.arange(len(y)))
            del rg

            probs = {}
            for tag, width in (("mlp_plain", w_plain), ("mlp_attn", w_attn)):
                print(f"  [{tag}] width {width}", flush=True)
                set_seed(seed)
                mlp = PlainMLP(X_t.shape[1], width=width).to(DEVICE)
                train_model(mlp, X_t, None, y, masks, tag=f"{tag}-{seed}")
                probs[tag] = _proba(mlp, X_t, None, np.arange(len(y)))
                del mlp
            np.savez_compressed(part, p_rg=p_rg,
                                p_plain=probs["mlp_plain"],
                                p_attn=probs["mlp_attn"])

        d = np.load(OUT / f"cache_{protocol}_seed{seed}.npz",
                    allow_pickle=True)
        te = d["split"] == 2
        row = {"seed": seed}
        for label, prob in [("gnn_knn", None), ("gnn_random", p_rg),
                            ("jmllm", None), ("mlp_plain", probs["mlp_plain"]),
                            ("mlp_attn", probs["mlp_attn"])]:
            if prob is None:
                pred = d[f"pred_{KEY['JM-GNN' if label == 'gnn_knn' else 'JM-LLM']}"]
            else:
                pred = (prob > 0.5).astype(np.int8)
            m = calc_metrics(y[te], pred.astype(int)[te])
            row[f"{label}_acc"] = m["acc"]
            row[f"{label}_f1"] = m["macro_f1"]
        frames.append(row)

    df = pd.DataFrame(frames)
    df.round(4).to_csv(OUT / f"e5_controls_per_seed_{protocol}.csv",
                       index=False, encoding="utf-8-sig")

    variants = [("gnn_knn", "JM-GNN, kNN graph"),
                ("gnn_random", "JM-GNN, shuffled graph"),
                ("jmllm", "JM-LLM, equal-weight fusion"),
                ("mlp_plain", f"MLP, width {w_plain} ({p_plain} params)"),
                ("mlp_attn", f"MLP, width {w_attn} ({p_attn} params)")]
    summary = pd.DataFrame([
        {"variant": label, "n_seeds": len(df),
         "test_acc_mean": df[f"{k}_acc"].mean(),
         "test_acc_std": df[f"{k}_acc"].std(ddof=1),
         "macro_f1_mean": df[f"{k}_f1"].mean(),
         "macro_f1_std": df[f"{k}_f1"].std(ddof=1)}
        for k, label in variants]).round(4)
    summary.to_csv(OUT / f"e5_controls_{protocol}.csv", index=False,
                   encoding="utf-8-sig")

    prow = []
    for a, b, note in [("gnn_random", "gnn_knn", "shuffled minus kNN graph"),
                       ("mlp_plain", "jmllm", "plain-capacity MLP minus JM-LLM"),
                       ("mlp_attn", "jmllm", "attention-capacity MLP minus JM-LLM")]:
        diff = df[f"{a}_acc"] - df[f"{b}_acc"]
        t = st.ttest_rel(df[f"{a}_acc"], df[f"{b}_acc"])
        prow.append({"comparison": note, "mean_diff_pp": diff.mean() * 100,
                     "sd_diff_pp": diff.std(ddof=1) * 100,
                     "t_stat": t.statistic, "p_value": t.pvalue,
                     "wins": int((diff > 0).sum()), "n_seeds": len(diff)})
    paired = pd.DataFrame(prow).round(4)
    paired.to_csv(OUT / f"e5_paired_{protocol}.csv", index=False,
                  encoding="utf-8-sig")

    with open(OUT / f"e5_meta_{protocol}.json", "w", encoding="utf-8") as f:
        json.dump({"plain_parameters": n_plain,
                   "attention_parameters": n_attn,
                   "mlp_plain_width": w_plain, "mlp_plain_parameters": p_plain,
                   "mlp_attn_width": w_attn,
                   "mlp_attn_parameters": p_attn}, f, indent=2)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# E6a: false-positive cost at the operating point
# ---------------------------------------------------------------------------
def run_e6_cost(protocol="official", seeds=(1, 2, 3, 4, 5)):
    rows = []
    for seed in seeds:
        d = np.load(OUT / f"cache_{protocol}_seed{seed}.npz",
                    allow_pickle=True)
        y = d["y"].astype(int)
        split = d["split"]
        va, te = split == 1, split == 2
        yt = y[te]
        for name in MODELS:
            if f"proba_{KEY[name]}" not in d.files:
                continue
            p = d[f"proba_{KEY[name]}"].astype(float)
            thr = 0.5
            if name == "JM-LLM":
                thr = max(np.linspace(0.05, 0.95, 19),
                          key=lambda t: f1_score(
                              y[va], (p[va] > t).astype(int),
                              average="macro"))
            pred = (p[te] > thr).astype(int)
            tn = int(((pred == 0) & (yt == 0)).sum())
            fp = int(((pred == 1) & (yt == 0)).sum())
            fn = int(((pred == 0) & (yt == 1)).sum())
            tp = int(((pred == 1) & (yt == 1)).sum())
            row = {"seed": seed, "model": name, "threshold": thr,
                   "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1)}
            for ratio in (1, 5, 10):
                row[f"cost_per_10k_1to{ratio}"] = \
                    (fn + fp / ratio) / len(yt) * 1e4
            rows.append(row)

    raw = pd.DataFrame(rows).round(4)
    raw.to_csv(OUT / f"e6_fpr_cost_per_seed_{protocol}.csv", index=False,
               encoding="utf-8-sig")
    agg = (raw.groupby("model", sort=False)
              .agg(n_seeds=("seed", "count"), fpr_mean=("fpr", "mean"),
                   fpr_std=("fpr", "std"), fnr_mean=("fnr", "mean"),
                   fnr_std=("fnr", "std"),
                   cost_1to1_mean=("cost_per_10k_1to1", "mean"),
                   cost_1to1_std=("cost_per_10k_1to1", "std"),
                   cost_1to5_mean=("cost_per_10k_1to5", "mean"),
                   cost_1to5_std=("cost_per_10k_1to5", "std"),
                   cost_1to10_mean=("cost_per_10k_1to10", "mean"),
                   cost_1to10_std=("cost_per_10k_1to10", "std"))
              .reset_index().round(4))
    agg.to_csv(OUT / f"e6_fpr_cost_{protocol}.csv", index=False,
               encoding="utf-8-sig")
    print(agg.to_string(index=False))
    return agg


# ---------------------------------------------------------------------------
# E6b: adversarial evasion (one-step bounded perturbation)
# ---------------------------------------------------------------------------
def _attack_prob_lstm(model, x):
    return torch.softmax(model(x)["class_pred"], dim=1)[:, 1]


def _attack_prob_jmllm(pack, x):
    jp = torch.sigmoid(pack["jordan"](x)).ravel()
    lp = torch.softmax(pack["lm"](x), dim=1)[:, 1]
    return 0.5 * jp + 0.5 * lp


def _fgsm(prob_fn, X, idx, eps, batch=8192):
    """One bounded step that pushes the attack score towards zero.

    The step is taken in standardised feature units, so ``eps`` is measured in
    standard deviations of the corresponding feature.
    """
    out = X.clone()
    for i in range(0, len(idx), batch):
        part = torch.as_tensor(idx[i:i + batch], device=DEVICE)
        x = X[part].clone().detach().requires_grad_(True)
        p = prob_fn(x).clamp(1e-6, 1 - 1e-6)
        loss = -(1.0 - p).log().mean()
        g = torch.autograd.grad(loss, x)[0]
        out[part] = (x.detach() - eps * g.sign()).clamp_(-10.0, 10.0)
    return out


def run_e6_adv(protocol="official", seed=42,
               eps_grid=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05)):
    X_all, y, cat, masks, _ = load_with_categories(protocol)
    X_t = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
    te = np.where(masks["test"])[0]
    yt = y[te]

    print("training LSTM for the evasion study ...", flush=True)
    set_seed(seed)
    lstm = LSTMClassifier(X_t.shape[1]).to(DEVICE)
    train_model(lstm, X_t, None, y, masks, tag="LSTM-adv", weight_decay=0.0)
    lstm.eval()

    print("training JM-LLM for the evasion study ...", flush=True)
    _, _, pack = run_jm_llm(X_t, y, masks, seed=seed, return_models=True)
    pack["jordan"].eval()
    pack["lm"].eval()

    rows = []
    te_t = torch.as_tensor(te, device=DEVICE)
    for name, fn, thr in [("LSTM", lambda x: _attack_prob_lstm(lstm, x), 0.5),
                          ("JM-LLM",
                           lambda x: _attack_prob_jmllm(pack, x),
                           pack["thr"])]:
        with torch.no_grad():
            clean = fn(X_t[te_t]).cpu().numpy()
        base = calc_metrics(yt, (clean > thr).astype(int))
        for attack in ("FGSM", "random"):
            for eps in eps_grid:
                if eps == 0.0:
                    if attack == "random":
                        continue
                    p = clean
                else:
                    if attack == "FGSM":
                        Xa = _fgsm(fn, X_t, te, eps)
                    else:
                        rng = np.random.default_rng(seed)
                        noise = torch.as_tensor(
                            rng.integers(0, 2, size=(len(te),
                                                      X_t.shape[1])) * 2 - 1,
                            dtype=torch.float32, device=DEVICE)
                        Xa = X_t.clone()
                        Xa[te_t] = X_t[te_t] + eps * noise
                    with torch.no_grad():
                        p = fn(Xa[te_t]).cpu().numpy()
                m = calc_metrics(yt, (p > thr).astype(int))
                row = {"model": name, "attack": attack, "epsilon_sigma": eps,
                       "threshold": round(float(thr), 2),
                       "test_acc": m["acc"],
                       "attack_recall": m["attack_recall"],
                       "normal_recall": m["normal_recall"],
                       "attack_recall_drop_pp":
                           (base["attack_recall"] - m["attack_recall"]) * 100}
                rows.append(row)
                print(f"  {row}", flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(OUT / f"e6_adversarial_{protocol}.csv", index=False,
               encoding="utf-8-sig")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["cache", "e1", "e1-jm", "e2", "e3",
                                        "e4", "e5", "e6", "e6-adv"])
    ap.add_argument("--protocol", choices=["official", "random"],
                    default="official")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", default=None, choices=MODELS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.command == "cache":
        build_cache(args.protocol, args.seed, args.force, only=args.model)
    elif args.command == "e1":
        run_e1(args.protocol, args.seed)
    elif args.command == "e1-jm":
        run_e1_jm_stability(args.protocol)
    elif args.command == "e2":
        run_e2(args.protocol, args.seeds)
    elif args.command == "e3":
        run_e3(args.protocol, args.seeds)
    elif args.command == "e4":
        run_e4()
    elif args.command == "e5":
        run_e5(args.protocol, args.seeds)
    elif args.command == "e6":
        run_e6_cost(args.protocol, args.seeds)
    elif args.command == "e6-adv":
        run_e6_adv(args.protocol, args.seed)


if __name__ == "__main__":
    main()
