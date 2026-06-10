from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lib.io_mat import load_dataset
from lib.calib import CalibParams
from lib.mlp_calib import bundle_from_dict as mlp_bundle_from_dict
from lib.pipeline import PipelineConfig
from lib.spatial import X_BOUNDS, Y_BOUNDS


def _rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def _load_bundle(root: Path) -> tuple[PipelineConfig, CalibParams]:
    z = torch.load(root / "model_mlp.pt", map_location="cpu", weights_only=False)
    calib = CalibParams.from_dict(z["calib"])
    calib.mlp_bundle = mlp_bundle_from_dict(z["meta"], z["state_dict"])
    pipe = z["pipeline"]
    pcfg = PipelineConfig(
        version=z.get("production_version", "v30"),
        loss=pipe.get("loss", "huber"),
        huber_f_scale=float(pipe.get("huber_f_scale", 1.0)),
        calib=pipe.get("calib", "isotonic_mlp"),
        gate=False,
        weight_gamma=float(pipe["weight_gamma"]) if pipe.get("weight_gamma") is not None else None,
        asym_pos_weight=float(pipe.get("asym_pos_weight", 5.0)),
        pos_refine_affine=bool(pipe.get("pos_refine_affine", True)),
    )
    return pcfg, calib


def lambda_mu_huber_trilat(
    d_obs: np.ndarray,
    bs: np.ndarray,
    *,
    huber_f_scale: float,
    weight_gamma: Optional[float],
    asym_pos_weight: float,
    xy_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    lambda_bounds: Tuple[float, float] = (0.5, 2.0),
    mu_bounds: Tuple[float, float] = (0.0, 40.0),
    lambda_init: float = 1.0,
    mu_init: float = 0.0,
) -> np.ndarray:
    d_obs = np.asarray(d_obs, dtype=np.float64).reshape(-1)
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    x0 = bs.mean(axis=1)
    bx, by = bs[0], bs[1]

    sqrt_w = None
    if weight_gamma is not None and float(weight_gamma) > 0:
        dd = np.maximum(d_obs, 5.0)
        w = (1.0 / dd) ** float(weight_gamma)
        w = w / (w.mean() + 1e-12)
        sqrt_w = np.sqrt(w)

    def residual(z: np.ndarray) -> np.ndarray:
        x, y, lam, mu = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        pred = np.sqrt((bx - x) ** 2 + (by - y) ** 2)
        r = pred - (lam * d_obs + mu)
        if asym_pos_weight != 1.0:
            w_pos = np.where(r > 0, asym_pos_weight, 1.0)
            r = r * np.sqrt(w_pos)
        if sqrt_w is not None:
            r = r * sqrt_w
        return r

    xb, yb = xy_bounds
    r = least_squares(
        residual,
        x0=np.array([x0[0], x0[1], lambda_init, mu_init], dtype=np.float64),
        loss="huber",
        f_scale=huber_f_scale,
        bounds=([xb[0], yb[0], lambda_bounds[0], mu_bounds[0]], [xb[1], yb[1], lambda_bounds[1], mu_bounds[1]]),
        max_nfev=250,
    )
    return np.array([r.x[0], r.x[1]], dtype=np.float64)


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    pcfg, calib = _load_bundle(ROOT)
    xy_bounds = (X_BOUNDS, Y_BOUNDS)

    for N in [100, 300, 700]:
        N = min(N, d_hat.shape[1])
        preds = np.zeros((2, N), dtype=np.float64)
        for u in range(N):
            d_corr = calib.apply(d_hat[:, u])
            xy = lambda_mu_huber_trilat(
                d_corr,
                bs,
                huber_f_scale=pcfg.huber_f_scale,
                weight_gamma=pcfg.weight_gamma,
                asym_pos_weight=pcfg.asym_pos_weight,
                xy_bounds=xy_bounds,
            )
            if calib.pos_affine is not None:
                v = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
                xy = (np.asarray(calib.pos_affine, dtype=np.float64) @ v).reshape(2)
            preds[:, u] = np.array(
                [np.clip(xy[0], X_BOUNDS[0], X_BOUNDS[1]), np.clip(xy[1], Y_BOUNDS[0], Y_BOUNDS[1])],
                dtype=np.float64,
            )
        rmse = _rmse(p[:, :N], preds)
        print(f"lambda_mu_trilat N={N}: rmse={rmse:.3f}")


if __name__ == "__main__":
    main()

