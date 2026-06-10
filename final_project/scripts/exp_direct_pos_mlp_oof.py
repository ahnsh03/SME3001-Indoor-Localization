from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import load_dataset


class PosMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def make_feat(d_hat: np.ndarray) -> np.ndarray:
    X = d_hat.T  # (N,18)
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    top8 = np.sort(X, axis=1)[:, :8]
    return np.hstack([X, mins, maxs, means, stds, top8])  # 30 dims


def main() -> None:
    data = load_dataset(None)
    d_hat, p = data["d_hat"], data["p"]  # p: (2,N)
    X = make_feat(d_hat)
    y = p.T  # (N,2)
    n = X.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((n, 2), dtype=np.float64)
    torch.manual_seed(42)

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        Xtr, Xva = X[tr], X[va]
        ytr = y[tr]

        mu = Xtr.mean(axis=0, keepdims=True)
        sig = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtrn = (Xtr - mu) / sig
        Xvan = (Xva - mu) / sig

        model = PosMLP(in_dim=Xtrn.shape[1], hidden=256, dropout=0.1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss(beta=2.0)
        xt = torch.from_numpy(Xtrn.astype(np.float32))
        yt = torch.from_numpy(ytr.astype(np.float32))

        model.train()
        for _ in range(400):
            opt.zero_grad()
            pred = model(xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xvan.astype(np.float32))).numpy()
        oof[va] = pv
        e = np.sqrt(np.sum((pv - y[va]) ** 2, axis=1))
        print(f"fold={fold} rmse={float(np.sqrt(np.mean(e**2))):.3f}")

    oof_p = oof.T
    e = np.hypot(oof_p[0] - p[0], oof_p[1] - p[1])
    print(f"direct_pos_mlp_oof rmse={float(np.sqrt(np.mean(e**2))):.3f} median={float(np.median(e)):.3f} p90={float(np.percentile(e,90)):.3f}")


if __name__ == "__main__":
    main()

