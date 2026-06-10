from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

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
from lib.spatial import X_BOUNDS, Y_BOUNDS, zone_from_min_dist


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


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    pcfg, calib = _load_bundle(ROOT)

    xb = (X_BOUNDS[0], X_BOUNDS[1])
    yb = (Y_BOUNDS[0], Y_BOUNDS[1])

    rules = [
        ("base_18", {0: 18, 1: 18, 2: 18}),
        ("near18_mid14_far10", {0: 18, 1: 14, 2: 10}),
        ("near18_mid15_far12", {0: 18, 1: 15, 2: 12}),
        ("near18_mid12_far8", {0: 18, 1: 12, 2: 8}),
    ]

    for name, kmap in rules:
        preds = np.zeros((2, 700), dtype=np.float64)
        for u in range(700):
            zone = int(zone_from_min_dist(d_hat[:, u]))
            k = int(kmap[zone])
            d_corr = calib.apply(d_hat[:, u])
            idx = np.argsort(d_corr)[:k]  # closest calibrated distances
            bs_sub = bs[:, idx]
            d_sub = d_corr[idx]
            xy, _ = huber_trilat(
                d_sub,
                bs_sub,
                loss=pcfg.loss,
                f_scale=pcfg.huber_f_scale,
                weight_gamma=pcfg.weight_gamma,
                asym_pos_weight=pcfg.asym_pos_weight,
                xy_bounds=(xb, yb),
            )
            if calib.pos_affine is not None:
                xy = apply_pos_affine(xy, calib.pos_affine)
            preds[:, u] = [
                float(np.clip(xy[0], xb[0], xb[1])),
                float(np.clip(xy[1], yb[0], yb[1])),
            ]
        rmse = _rmse(p[:, :700], preds)
        print(f"{name}: rmse={rmse:.3f}")


if __name__ == "__main__":
    main()

