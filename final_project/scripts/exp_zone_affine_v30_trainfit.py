from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import load_dataset
from lib.calib import CalibParams
from lib.mlp_calib import bundle_from_dict as mlp_bundle_from_dict
from lib.pipeline import VERSION_REGISTRY, PipelineConfig
from lib.pipeline import localize_user as _localize_user
from lib.post_refine import fit_pos_affine, apply_pos_affine
from lib.spatial import zone_from_min_dist
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
        weight_gamma=float(pipe["weight_gamma"]) if pipe.get("weight_gamma", None) is not None else None,
        asym_pos_weight=float(pipe.get("asym_pos_weight", 5.0)),
        pos_refine_affine=True,
    )
    return pcfg, calib


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    pcfg, calib = _load_bundle(ROOT)

    calib_coarse = copy.deepcopy(calib)
    calib_coarse.pos_affine = None
    calib_coarse.pos_affine_far = None

    n = d_hat.shape[1]
    p_coarse = np.zeros((2, n), dtype=np.float64)
    zones = np.zeros(n, dtype=np.int64)

    mins = np.min(d_hat, axis=0)
    e1 = float(np.percentile(mins, 33))
    e2 = float(np.percentile(mins, 66))

    for u in range(n):
        # quantile-based zones to avoid extremely small partitions
        zidx = 0 if float(mins[u]) < e1 else (1 if float(mins[u]) < e2 else 2)
        zones[u] = zidx
        xy = _localize_user(d_hat[:, u], bs, calib_coarse, pcfg, xy_bounds=None)[0]
        # coarse output already respects trilat bounds
        p_coarse[:, u] = xy.reshape(-1)

    # fit separate affine for each zone
    Ms = {}
    for zidx in [0, 1, 2]:
        mask = zones == zidx
        if int(mask.sum()) < 50:
            continue
        Ms[zidx] = fit_pos_affine(p_coarse[:, mask], p[:, mask])

    p_final = np.zeros_like(p_coarse)
    for u in range(n):
        zidx = int(zones[u])
        if zidx in Ms:
            xy = apply_pos_affine(p_coarse[:, u], Ms[zidx])
        else:
            xy = p_coarse[:, u]
        p_final[0, u] = float(np.clip(xy[0], X_BOUNDS[0], X_BOUNDS[1]))
        p_final[1, u] = float(np.clip(xy[1], Y_BOUNDS[0], Y_BOUNDS[1]))

    rmse = _rmse(p, p_final)
    print(f"zone_affine_trainfit_rmse={rmse:.3f}")
    for zidx in [0, 1, 2]:
        mask = zones == zidx
        if int(mask.sum()) == 0:
            continue
        print(f"  zone {zidx} n={int(mask.sum())} rmse={_rmse(p[:, mask], p_final[:, mask]):.3f}")


if __name__ == "__main__":
    main()

