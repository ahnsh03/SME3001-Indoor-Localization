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

from lib.calib import fit_calib
from lib.cv import clone_pipeline_cfg
from lib.io_mat import load_dataset, geometric_distances
from lib.mlp_calib import DistMLP
from lib.pipeline import VERSION_REGISTRY
from lib.post_refine import fit_pos_affine
from lib.spatial import bounds_from_positions
from lib.trilat import huber_trilat


class ResidualPosMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.15):
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


def _rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def _feat(d_hat_u: np.ndarray, p_base_u: np.ndarray) -> np.ndarray:
    d = np.asarray(d_hat_u, dtype=np.float64).reshape(-1)  # (18,)
    return np.hstack(
        [
            d,
            p_base_u.reshape(-1),  # (2,)
            [np.min(d), np.max(d), np.mean(d), np.std(d)],
        ]
    )  # in_dim=18+2+4=24


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    cfg = VERSION_REGISTRY["v30"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_base = np.zeros((2, n), dtype=np.float64)
    oof_final = np.zeros((2, n), dtype=np.float64)

    torch.manual_seed(42)

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)

        calib = fit_calib(
            d_hat,
            p,
            bs,
            tr,
            cfg.calib,
            quantile_tau=cfg.quantile_tau,
            mlp_noise_std=cfg.mlp_noise_std,
        )

        # fit pos_affine on train fold only (same strategy as lib/cv)
        cfg_no_aff = clone_pipeline_cfg(cfg, pos_refine_affine=False)
        xy_bounds_pre = bounds_from_positions(p[:, tr])
        p_tr_raw = np.zeros((2, len(tr)), dtype=np.float64)
        for j, u in enumerate(tr):
            xy, _ = huber_trilat(
                # localize_user uses calib.apply + huber_trilat + clip; we can reuse it by calling localize_user
                # but to keep script minimal, use calib+trilat directly:
                calib.apply(d_hat[:, int(u)]),
                bs,
                loss=cfg.loss,
                f_scale=cfg.huber_f_scale,
                weight_gamma=cfg.weight_gamma,
                asym_pos_weight=cfg.asym_pos_weight,
                xy_bounds=None,
            )
            p_tr_raw[:, j] = xy

        calib.pos_affine = fit_pos_affine(p_tr_raw, p[:, tr])

        # base predictions
        xy_bounds = bounds_from_positions(p[:, tr])
        p_base_tr = np.zeros((2, len(tr)), dtype=np.float64)
        p_base_va = np.zeros((2, len(va)), dtype=np.float64)

        for j, u in enumerate(tr):
            xy, _ = huber_trilat(
                calib.apply(d_hat[:, int(u)]),
                bs,
                loss=cfg.loss,
                f_scale=cfg.huber_f_scale,
                weight_gamma=cfg.weight_gamma,
                asym_pos_weight=cfg.asym_pos_weight,
                xy_bounds=None,
            )
            # apply affine
            if calib.pos_affine is not None:
                v = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
                xy = (np.asarray(calib.pos_affine, dtype=np.float64) @ v).reshape(2)
            p_base_tr[:, j] = xy
        for j, u in enumerate(va):
            xy, _ = huber_trilat(
                calib.apply(d_hat[:, int(u)]),
                bs,
                loss=cfg.loss,
                f_scale=cfg.huber_f_scale,
                weight_gamma=cfg.weight_gamma,
                asym_pos_weight=cfg.asym_pos_weight,
                xy_bounds=None,
            )
            if calib.pos_affine is not None:
                v = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
                xy = (np.asarray(calib.pos_affine, dtype=np.float64) @ v).reshape(2)
            p_base_va[:, j] = xy

        oof_base[:, tr] = p_base_tr
        oof_base[:, va] = p_base_va

        # residual targets
        y_res_tr = (p[:, tr] - p_base_tr).T  # (len(tr),2)

        # feature scaling (simple z-score)
        X_tr = np.stack([_feat(d_hat[:, u], p_base_tr[:, j]) for j, u in enumerate(tr)], axis=0)
        X_va = np.stack([_feat(d_hat[:, u], p_base_va[:, j]) for j, u in enumerate(va)], axis=0)

        mu = X_tr.mean(axis=0, keepdims=True)
        sig = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_trn = (X_tr - mu) / sig
        X_van = (X_va - mu) / sig

        in_dim = X_trn.shape[1]
        model = ResidualPosMLP(in_dim=in_dim, hidden=64, dropout=0.15)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss(beta=1.0)  # robust regression

        xt = torch.from_numpy(X_trn.astype(np.float32))
        yt = torch.from_numpy(y_res_tr.astype(np.float32))
        model.train()
        for _ in range(80):
            opt.zero_grad()
            pred = model(xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            delta_va = model(torch.from_numpy(X_van.astype(np.float32))).numpy().astype(np.float64)

        p_final_va = p_base_va.T + delta_va  # (len(va),2)
        oof_final[:, va] = p_final_va.T

    print(f"base_v30_rmse={_rmse(p, oof_base):.3f}")
    print(f"base+residual_pos_mlp_rmse={_rmse(p, oof_final):.3f}")


if __name__ == "__main__":
    main()

