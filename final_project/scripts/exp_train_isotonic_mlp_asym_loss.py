from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import load_dataset, geometric_distances
from lib.spatial import X_BOUNDS, Y_BOUNDS
from lib.trilat import huber_trilat
from lib.post_refine import fit_pos_affine, apply_pos_affine


class DistMLP(nn.Module):
    def __init__(self, hidden: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 18),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual: keep distance scale interpretable
        return x + self.net(x)


def _rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]

    # stage 1: isotonic mapping (per BS channel)
    d_true = geometric_distances(p, bs)  # (18,N)
    iso_x, iso_y = [], []
    for k in range(18):
        ir = IsotonicRegression(out_of_bounds="clip")
        x = d_hat[k].astype(np.float64).ravel()
        y = d_true[k].astype(np.float64).ravel()
        ir.fit(x, y)
        iso_x.append(np.asarray(ir.X_thresholds_, dtype=np.float64))
        iso_y.append(np.asarray(ir.y_thresholds_, dtype=np.float64))

    def apply_iso(d_hat_u: np.ndarray) -> np.ndarray:
        out = np.empty(18, dtype=np.float64)
        for k in range(18):
            xk = iso_x[k]
            yk = iso_y[k]
            out[k] = np.interp(d_hat_u[k], xk, yk, left=yk[0], right=yk[-1])
        return out

    # dataset for MLP: input = d_iso (N,18), target = d_true (N,18)
    N = d_hat.shape[1]
    dh = np.zeros((N, 18), dtype=np.float64)
    y = d_true.T  # (N,18)
    for u in range(N):
        dh[u] = apply_iso(d_hat[:, u])

    in_mean = dh.mean(axis=0)
    in_std = np.maximum(dh.std(axis=0), 1e-3)
    xn = (dh - in_mean) / in_std

    xt = torch.from_numpy(xn.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    t_std = torch.from_numpy(in_std.astype(np.float32))
    t_mean = torch.from_numpy(in_mean.astype(np.float32))

    # stage 2: MLP with asymmetric loss on overestimation (pred > target)
    model = DistMLP(hidden=64, dropout=0.15)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    asym_pos_weight = 5.0
    epochs = 120
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(xt) * t_std + t_mean  # (N,18)
        diff = pred - yt
        w = torch.where(diff > 0, torch.tensor(asym_pos_weight, device=diff.device), torch.tensor(1.0, device=diff.device))
        loss = torch.mean((diff**2) * w)
        loss.backward()
        opt.step()

    # inference with v30 trilat: use calibrated distances (iso+mlp) then huber_trilat + global affine
    model.eval()
    with torch.no_grad():
        pred_d = (model(xt) * t_std + t_mean).numpy()  # (N,18)

    # trilat for each UE
    preds = np.zeros((2, N), dtype=np.float64)
    for u in range(N):
        xy, _ = huber_trilat(
            pred_d[u],
            bs,
            loss="huber",
            f_scale=1.0,
            weight_gamma=1.0,
            asym_pos_weight=5.0,
            xy_bounds=(X_BOUNDS, Y_BOUNDS),
        )
        preds[:, u] = xy

    # 2-pass affine on positions
    M_aff = fit_pos_affine(preds, p)
    preds_aff = np.zeros_like(preds)
    for u in range(N):
        preds_aff[:, u] = apply_pos_affine(preds[:, u], M_aff)
        preds_aff[0, u] = np.clip(preds_aff[0, u], X_BOUNDS[0], X_BOUNDS[1])
        preds_aff[1, u] = np.clip(preds_aff[1, u], Y_BOUNDS[0], Y_BOUNDS[1])

    rmse = _rmse(p, preds_aff)
    print(f"asym-distance-MLP trainfit rmse={rmse:.3f}")


if __name__ == "__main__":
    main()

