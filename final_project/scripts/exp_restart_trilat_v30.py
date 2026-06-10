from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import load_dataset
from lib.calib import CalibParams
from lib.mlp_calib import bundle_from_dict as mlp_bundle_from_dict
from lib.pipeline import PipelineConfig
from lib.trilat import huber_trilat
from lib.post_refine import apply_pos_affine
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
        weight_gamma=float(pipe["weight_gamma"])
        if pipe.get("weight_gamma", None) is not None
        else None,
        asym_pos_weight=float(pipe.get("asym_pos_weight", 5.0)),
        pos_refine_affine=True,
    )
    return pcfg, calib


def _initial_points(d_obs: np.ndarray, bs: np.ndarray) -> List[np.ndarray]:
    if bs.shape[0] != 2:
        bs = bs.T
    bx = bs[0]
    by = bs[1]
    pts = []
    # 1) anchor centroid
    pts.append(np.array([bx.mean(), by.mean()], dtype=np.float64))
    # 2) inverse-distance weighted mean
    w = 1.0 / np.maximum(d_obs, 1.0)
    w = w / (w.mean() + 1e-12)
    pts.append(np.array([(bx * w).mean(), (by * w).mean()], dtype=np.float64))
    # 3) mean of k nearest anchors
    idx = np.argsort(d_obs)[:3]
    pts.append(np.array([bx[idx].mean(), by[idx].mean()], dtype=np.float64))
    # 4) mean of two far anchors
    idx2 = np.argsort(d_obs)[-2:]
    pts.append(np.array([bx[idx2].mean(), by[idx2].mean()], dtype=np.float64))
    # dedupe
    out = []
    for p in pts:
        if not any(np.allclose(p, q) for q in out):
            out.append(p)
    return out


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    pcfg, calib = _load_bundle(ROOT)

    n = 700
    preds = np.zeros((2, n), dtype=np.float64)
    xb = (X_BOUNDS[0], X_BOUNDS[1])
    yb = (Y_BOUNDS[0], Y_BOUNDS[1])

    for u in range(n):
        d_corr = calib.apply(d_hat[:, u])
        inits = _initial_points(d_corr, bs)
        best_xy = None
        best_cost = float("inf")
        for x0 in inits:
            xy, meta = huber_trilat(
                d_corr,
                bs,
                loss=pcfg.loss,
                f_scale=pcfg.huber_f_scale,
                x0=x0,
                weight_gamma=pcfg.weight_gamma,
                asym_pos_weight=pcfg.asym_pos_weight,
                xy_bounds=(xb, yb),
            )
            cost = float(meta.get("cost", float("inf")))
            if cost < best_cost:
                best_cost = cost
                best_xy = xy
        xy = best_xy
        if calib.pos_affine is not None:
            xy = apply_pos_affine(xy, calib.pos_affine)
        preds[:, u] = [
            float(np.clip(xy[0], xb[0], xb[1])),
            float(np.clip(xy[1], yb[0], yb[1])),
        ]

    print(f"restart_trilat_v30_trainfit_rmse={_rmse(p[:, :n], preds):.3f}")


if __name__ == "__main__":
    main()

