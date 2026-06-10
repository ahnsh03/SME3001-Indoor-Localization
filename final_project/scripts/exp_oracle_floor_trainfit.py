from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import CalibParams
from lib.io_mat import load_dataset
from lib.mlp_calib import bundle_from_dict as mlp_bundle_from_dict
from lib.pipeline import PipelineConfig
from lib.post_refine import apply_pos_affine
from lib.spatial import X_BOUNDS, Y_BOUNDS
from lib.trilat import huber_trilat


def _err(p_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    return np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])


def _rmse(e: np.ndarray) -> float:
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


def _predict_v30(
    d_hat: np.ndarray, bs: np.ndarray, calib: CalibParams, cfg: PipelineConfig, margin: float = 0.0
) -> np.ndarray:
    xb = (X_BOUNDS[0] - margin, X_BOUNDS[1] + margin)
    yb = (Y_BOUNDS[0] - margin, Y_BOUNDS[1] + margin)
    n = d_hat.shape[1]
    pred = np.zeros((2, n), dtype=np.float64)
    for u in range(n):
        dc = calib.apply(d_hat[:, u])
        xy, _ = huber_trilat(
            dc,
            bs,
            loss=cfg.loss,
            f_scale=cfg.huber_f_scale,
            weight_gamma=cfg.weight_gamma,
            asym_pos_weight=cfg.asym_pos_weight,
            xy_bounds=(xb, yb),
        )
        if calib.pos_affine is not None:
            xy = apply_pos_affine(xy, calib.pos_affine)
        pred[0, u] = np.clip(xy[0], xb[0], xb[1])
        pred[1, u] = np.clip(xy[1], yb[0], yb[1])
    return pred


def _predict_hgb(d_hat: np.ndarray, p: np.ndarray) -> np.ndarray:
    X = d_hat.T
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    Xf = np.hstack([X, mins, maxs, means, stds, np.sort(X, axis=1)[:, :6]])
    y = p.T
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=700, learning_rate=0.04, max_depth=8, random_state=42)
    )
    model.fit(Xf, y)
    return model.predict(Xf).T


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    cfg, calib = _load_bundle(ROOT)

    cands: dict[str, np.ndarray] = {}
    cands["v30_margin0"] = _predict_v30(d_hat, bs, calib, cfg, margin=0.0)
    cands["v30_margin2"] = _predict_v30(d_hat, bs, calib, cfg, margin=2.0)
    cands["v30_margin5"] = _predict_v30(d_hat, bs, calib, cfg, margin=5.0)
    cands["hgb_fullfit"] = _predict_hgb(d_hat, p)

    errs = {}
    for k, pred in cands.items():
        e = _err(p, pred)
        errs[k] = e
        print(f"{k}: rmse={_rmse(e):.3f} median={np.median(e):.3f} p90={np.percentile(e,90):.3f}")

    E = np.stack([errs[k] for k in cands.keys()], axis=0)  # (K,N)
    e_oracle = np.min(E, axis=0)
    print(f"oracle_min_over_{len(cands)}: rmse={_rmse(e_oracle):.3f} median={np.median(e_oracle):.3f} p90={np.percentile(e_oracle,90):.3f}")


if __name__ == "__main__":
    main()

