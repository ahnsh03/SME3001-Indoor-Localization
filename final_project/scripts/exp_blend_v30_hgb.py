from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.cv import clone_pipeline_cfg
from lib.io_mat import load_dataset
from lib.pipeline import VERSION_REGISTRY, localize_user
from lib.post_refine import fit_pos_affine
from lib.spatial import bounds_from_positions


def _rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def _build_feats(d_hat: np.ndarray) -> np.ndarray:
    mins = d_hat.min(axis=1, keepdims=True)
    maxs = d_hat.max(axis=1, keepdims=True)
    means = d_hat.mean(axis=1, keepdims=True)
    stds = d_hat.std(axis=1, keepdims=True)
    return np.hstack([d_hat, mins, maxs, means, stds, np.sort(d_hat, axis=1)[:, :6]])


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    x = d_hat.T  # (N,18)
    y = p.T  # (N,2)
    xf = _build_feats(x)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_v30 = np.zeros((2, d_hat.shape[1]), dtype=np.float64)
    oof_hgb = np.zeros((2, d_hat.shape[1]), dtype=np.float64)
    base_cfg = VERSION_REGISTRY["v30"]

    for fold, (tr, va) in enumerate(cv.split(np.arange(d_hat.shape[1]))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        calib = fit_calib(
            d_hat,
            p,
            bs,
            tr,
            base_cfg.calib,
            quantile_tau=base_cfg.quantile_tau,
            mlp_noise_std=base_cfg.mlp_noise_std,
        )
        cfg_no_aff = clone_pipeline_cfg(base_cfg, pos_refine_affine=False)
        xy_bounds_pre = bounds_from_positions(p[:, tr])
        p_tr_raw = np.zeros((2, len(tr)), dtype=np.float64)
        for j, u in enumerate(tr):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg_no_aff, xy_bounds=xy_bounds_pre)
            p_tr_raw[:, j] = xy
        calib.pos_affine = fit_pos_affine(p_tr_raw, p[:, tr])
        xy_bounds = bounds_from_positions(p[:, tr])
        for u in va:
            xy, _ = localize_user(d_hat[:, u], bs, calib, base_cfg, xy_bounds=xy_bounds)
            oof_v30[:, u] = xy

        hgb = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=700, learning_rate=0.04, max_depth=8, random_state=42 + fold)
        )
        hgb.fit(xf[tr], y[tr])
        oof_hgb[:, va] = hgb.predict(xf[va]).T

    print(f"v30_rmse={_rmse(p, oof_v30):.3f}")
    print(f"hgb_rmse={_rmse(p, oof_hgb):.3f}")
    best = (None, 1e9)
    for a in np.linspace(0.0, 1.0, 21):
        blend = a * oof_v30 + (1.0 - a) * oof_hgb
        r = _rmse(p, blend)
        if r < best[1]:
            best = (a, r)
        print(f"alpha_v30={a:.2f} rmse={r:.3f}")
    print(f"best_alpha={best[0]:.2f} best_rmse={best[1]:.3f}")


if __name__ == "__main__":
    main()
