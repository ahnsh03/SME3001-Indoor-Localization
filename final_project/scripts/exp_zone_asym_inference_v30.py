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
        weight_gamma=float(pipe["weight_gamma"]) if pipe.get("weight_gamma", None) is not None else None,
        asym_pos_weight=float(pipe.get("asym_pos_weight", 5.0)),
        pos_refine_affine=True,
    )
    return pcfg, calib


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    pcfg, calib = _load_bundle(ROOT)

    combos = [
        (1.0, 3.0, 8.0),
        (1.0, 5.0, 8.0),
        (1.0, 2.0, 5.0),
        (1.0, 3.0, 5.0),
        (2.0, 3.0, 8.0),
        (5.0, 5.0, 5.0),  # baseline
    ]

    for w0, w1, w2 in combos:
        preds = np.zeros((2, 700), dtype=np.float64)
        for u in range(700):
            zone = int(zone_from_min_dist(d_hat[:, u]))
            asym = {0: w0, 1: w1, 2: w2}[zone]
            d_corr = calib.apply(d_hat[:, u])
            xy, _ = huber_trilat(
                d_corr,
                bs,
                loss=pcfg.loss,
                f_scale=pcfg.huber_f_scale,
                weight_gamma=pcfg.weight_gamma,
                asym_pos_weight=asym,
                xy_bounds=(X_BOUNDS, Y_BOUNDS),
            )
            if calib.pos_affine is not None:
                xy = apply_pos_affine(xy, calib.pos_affine)
            preds[0, u] = float(np.clip(xy[0], X_BOUNDS[0], X_BOUNDS[1]))
            preds[1, u] = float(np.clip(xy[1], Y_BOUNDS[0], Y_BOUNDS[1]))
        rmse = _rmse(p[:, :700], preds)
        print(f"zone_asym (w_near,w_mid,w_far)=({w0},{w1},{w2}) rmse={rmse:.3f}")


if __name__ == "__main__":
    main()

