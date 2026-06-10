from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

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


def _feat_from_dhat(d_hat: np.ndarray) -> np.ndarray:
    d = np.asarray(d_hat, dtype=np.float64).reshape(-1)
    mins = float(np.min(d))
    maxs = float(np.max(d))
    means = float(np.mean(d))
    stds = float(np.std(d))
    top8 = np.sort(d)[:8]
    return np.hstack([d, top8, [mins, maxs, means, stds]])


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    N = d_hat.shape[1]

    X_d = d_hat.T  # (N,18)
    y = p.T  # (N,2)

    # base model configs
    cfg = VERSION_REGISTRY["v30"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_v30 = np.zeros((2, N), dtype=np.float64)
    oof_hgb = np.zeros((2, N), dtype=np.float64)
    oof_meta = np.zeros((2, N), dtype=np.float64)

    # direct HGB feature builder (same across folds)
    X_feat = np.stack([_feat_from_dhat(X_d[i]) for i in range(N)], axis=0)

    for fold, (tr, va) in enumerate(kf.split(np.arange(N))):
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
        cfg_no_aff = clone_pipeline_cfg(cfg, pos_refine_affine=False)
        xy_bounds_pre = bounds_from_positions(p[:, tr])
        p_tr_raw = np.zeros((2, len(tr)), dtype=np.float64)
        for j, u in enumerate(tr):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg_no_aff, xy_bounds=xy_bounds_pre)
            p_tr_raw[:, j] = xy
        calib.pos_affine = fit_pos_affine(p_tr_raw, p[:, tr])

        xy_bounds = bounds_from_positions(p[:, tr])
        # v30 base predictions for meta training (tr) and evaluation (va)
        p_v30_tr = np.zeros((2, len(tr)), dtype=np.float64)
        p_v30_va = np.zeros((2, len(va)), dtype=np.float64)
        for j, u in enumerate(tr):
            xy, _ = localize_user(d_hat[:, int(u)], bs, calib, cfg, xy_bounds=xy_bounds)
            p_v30_tr[:, j] = xy
        for j, u in enumerate(va):
            xy, _ = localize_user(d_hat[:, int(u)], bs, calib, cfg, xy_bounds=xy_bounds)
            p_v30_va[:, j] = xy
            oof_v30[:, int(u)] = xy

        # direct HGB regressor
        meta_hgb = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=800, learning_rate=0.04, max_depth=7, random_state=100 + fold
            )
        )
        meta_hgb.fit(X_feat[tr], y[tr])
        p_hgb_tr = meta_hgb.predict(X_feat[tr]).T
        p_hgb_va = meta_hgb.predict(X_feat[va]).T
        oof_hgb[:, va] = p_hgb_va

        # meta model (nested): train on tr, predict on va
        base_tr = np.hstack(
            [
                p_v30_tr.T,
                p_hgb_tr.T,
                X_feat[tr, :6],
                X_feat[tr, -4:],
            ]
        )
        base_va = np.hstack(
            [
                p_v30_va.T,
                p_hgb_va.T,
                X_feat[va, :6],
                X_feat[va, -4:],
            ]
        )
        meta = MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=600,
                random_state=42 + fold,
                n_jobs=-1,
                min_samples_leaf=5,
                max_depth=24,
            )
        )
        meta.fit(base_tr, y[tr])
        pred_meta_va = meta.predict(base_va).T
        oof_meta[:, va] = pred_meta_va

    print(f"v30_oof_rmse={_rmse(p, oof_v30):.3f}")
    print(f"hgb_oof_rmse={_rmse(p, oof_hgb):.3f}")
    print(f"stack_meta_oof_rmse={_rmse(p, oof_meta):.3f}")


if __name__ == "__main__":
    main()

