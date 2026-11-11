"""Reproduce the n-sweep experiment (paper Figures 3-6) on the FULL dataset.

By default the sweep runs on the complete official split (257,673 nodes).
To keep 40 full-graph trainings affordable, the Jordan eigensystem is solved
ONCE with k = max(N_LIST) leading eigenpairs and each n simply truncates that
basis: the leading n eigenpairs of a k = 40 ARPACK solve are the same ones a
dedicated k = n solve would return (documented adaptation, mathematically
equivalent up to the ARPACK tolerance).

Usage:  python run_n_sweep.py [--size 0]   # 0 = full corpus; >0 = subsample
"""
import argparse
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import common
from common import scipy_to_torch_sparse
from models import DEVICE, CollaborativeModel, GNNModel, set_seed

EPOCHS = 80
PATIENCE = 15
N_LIST = range(1, 41)
EIGEN_SHOW = (10, 20, 30, 40)
MAX_N = max(N_LIST)


def stratified_subsample(X_tr, y_tr, X_te, y_te, size):
    rng = np.random.default_rng(0)
    n_te = int(size * len(y_te) / (len(y_tr) + len(y_te)))
    n_tr = size - n_te

    def take(X, y, k):
        parts = []
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            take_n = max(1, int(k * len(idx) / len(y)))
            parts.append(rng.choice(idx, min(take_n, len(idx)), replace=False))
        sel = np.sort(np.concatenate(parts))
        return X[sel], y[sel]

    Xs_tr, ys_tr = take(X_tr, y_tr, n_tr)
    Xs_te, ys_te = take(X_te, y_te, n_te)
    return Xs_tr, ys_tr, Xs_te, ys_te


def evaluate(model, X_all, adj, y, masks, splits=("train", "val", "test")):
    model.eval()
    with torch.no_grad():
        pred = model(X_all, adj)["class_pred"].argmax(1).cpu().numpy()
    out = {}
    for split in splits:
        idx = np.where(masks[split])[0]
        out[split] = float((pred[idx] == y[idx]).mean())
        if split == "test":
            out["rpa"] = float(f1_score(y[idx], pred[idx], average="macro"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=0,
                    help="0 = full official split; >0 = stratified subsample")
    args = ap.parse_args()
    torch.set_num_threads(8)

    set_seed()
    X_tr, y_tr, X_te, y_te = common.load_full_dataset()
    if args.size > 0:
        X_tr, y_tr, X_te, y_te = stratified_subsample(X_tr, y_tr, X_te, y_te,
                                                      args.size)
    X_all_np = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    n_tr = len(y_tr)
    print(f"n-sweep sample: {X_all_np.shape}, train={n_tr}, test={len(y_te)}")

    masks = common.make_masks(n_tr, y_tr, val_size=0.2)
    full = {}
    for split in ("train", "val"):
        m = np.zeros(len(y_all), dtype=bool)
        m[:n_tr] = masks[split]
        full[split] = m
    tm = np.zeros(len(y_all), dtype=bool)
    tm[n_tr:] = True
    full["test"] = tm

    t0 = time.time()
    adjacency = common.build_knn_adjacency(X_all_np, k_neighbors=3)
    adj_t = scipy_to_torch_sparse(common.normalized_adjacency_sparse(adjacency))
    X_all = torch.tensor(X_all_np, dtype=torch.float32).to(DEVICE)
    print(f"graph ready in {time.time() - t0:.0f}s")

    y_t = torch.tensor(y_all, dtype=torch.long).to(DEVICE)
    train_idx = np.where(full["train"])[0]
    counts = np.bincount(y_all[train_idx], minlength=2).astype(np.float32)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                            dtype=torch.float32).to(DEVICE))

    t_eig = time.time()
    jordan_full = common.JordanMatrixDecomposition(adjacency,
                                                   n_components=MAX_N,
                                                   solver="arpack")
    ev_all = jordan_full.eigenvalues
    print(f"eigensystem (k={MAX_N}) solved in {time.time() - t_eig:.0f}s")

    results = []
    try:                      # resume from a partial sweep if one exists
        prev = pd.read_csv("n_sweep_full.csv")
        results = prev.to_dict("records")
        done = {int(v) for v in prev["n"]}
        print(f"resuming: {len(done)} points already done", flush=True)
    except FileNotFoundError:
        done = set()
    for n in N_LIST:
        if n in done:
            continue
        t_n = time.time()
        # Leading-n truncation of the k=40 basis (same eigenpairs as a k=n solve)
        jordan = SimpleNamespace(eigenvalues=ev_all[:n])
        if n in EIGEN_SHOW:
            plt.figure(figsize=(8, 5))
            plt.plot(ev_all[:n], "o-", ms=3)
            plt.title(f"Eigenvalue distribution (n={n})")
            plt.xlabel("index"); plt.ylabel("eigenvalue"); plt.grid(alpha=.3)
            plt.tight_layout(); plt.savefig(f"eigenvalues_n{n}.png", dpi=150)
            plt.close()

        set_seed()
        model = CollaborativeModel(GNNModel(X_all.shape[1]).to(DEVICE), 64,
                                   jordan).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        best_f1, best_loss, counter, best_state = -1.0, float("inf"), 0, None
        for epoch in range(1, EPOCHS + 1):
            model.train(); opt.zero_grad()
            logits = model(X_all, adj_t)["class_pred"][train_idx]
            loss = crit(logits, y_t[train_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            met = evaluate(model, X_all, adj_t, y_all, full)
            if met["val"] >= best_f1:
                best_f1, counter, best_loss = met["val"], 0, loss.item()
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
            else:
                counter += 1
                if counter >= PATIENCE:
                    break
        if best_state:
            model.load_state_dict(best_state)
        met = evaluate(model, X_all, adj_t, y_all, full)
        results.append({"n": n, "loss": best_loss, **met})
        pd.DataFrame(results).sort_values("n").to_csv(
            "n_sweep_full.csv", index=False, encoding="utf-8-sig")
        print(f"n={n}: loss={best_loss:.4f} train={met['train']:.4f} "
              f"val={met['val']:.4f} test={met['test']:.4f} "
              f"rpa={met['rpa']:.4f} ({time.time() - t_n:.0f}s)", flush=True)

    df = pd.DataFrame(results)
    df.to_csv("n_sweep_full.csv", index=False, encoding="utf-8-sig")
    print("\nbest row by val F1:")
    print(df.loc[df["val"].idxmax()].to_string())

    for col, title in [("loss", "Loss"), ("rpa", "RPA"),
                       ("test", "Test accuracy"), ("train", "Train accuracy")]:
        plt.figure(figsize=(8, 5))
        plt.plot(df["n"], df[col], "o-", ms=3)
        plt.title(f"Influence of n on {title.lower()}")
        plt.xlabel("n (number of Jordan eigen-components)")
        plt.ylabel(title); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"n_influence_{col}.png", dpi=150)
        plt.close()
    print("saved n_sweep_full.csv and n_influence_*.png")


if __name__ == "__main__":
    main()
