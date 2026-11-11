"""P2.5: normal-behaviour profile branch.

A small auto-encoder is fitted on training-split *Normal* flows only.  Its
per-flow reconstruction error is standardised with the statistics of that same
Normal subset and appended as one extra input feature of the JM-LLM branch.
No validation/test label is used while fitting.
"""
import numpy as np
import torch
import torch.nn as nn

from models import DEVICE


class NormalProfileAE(nn.Module):
    def __init__(self, input_dim, hidden=64, latent=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.net(x)


def recon_scores(model, X, batch=8192):
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            part = torch.tensor(X[i:i + batch], dtype=torch.float32).to(DEVICE)
            rec = model(part)
            out[i:i + len(part)] = ((rec - part) ** 2).mean(1).cpu().numpy()
    return out


def fit_normal_profile(X, normal_mask, epochs=60, batch=4096, lr=1e-3,
                       seed=42, verbose=True):
    """Return the standardised anomaly score for every flow in ``X``.

    ``normal_mask`` must only cover training-split Normal flows.
    """
    torch.manual_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    Xn = X[normal_mask]
    model = NormalProfileAE(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xn_t = torch.tensor(Xn, dtype=torch.float32).to(DEVICE)
    n = len(Xn_t)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        total = 0.0
        model.train()
        for i in range(0, n, batch):
            part = Xn_t[perm[i:i + batch]]
            opt.zero_grad()
            loss = lossf(model(part), part)
            loss.backward()
            opt.step()
            total += loss.item() * len(part)
        if verbose and ep % 20 == 0:
            print(f"  [P2.5] AE epoch {ep}: normal-train MSE={total / n:.4f}",
                  flush=True)

    raw = recon_scores(model, X)
    mu = float(raw[normal_mask].mean())
    sd = float(raw[normal_mask].std()) + 1e-6
    z = ((raw - mu) / sd).astype(np.float32)
    if verbose:
        print(f"  [P2.5] score: normal(train) mean={mu:.4f} sd={sd:.4f} | "
              f"z normal={z[normal_mask].mean():.2f} z attack="
              f"{z[~normal_mask].mean():.2f}")
    return z, model, (mu, sd)
