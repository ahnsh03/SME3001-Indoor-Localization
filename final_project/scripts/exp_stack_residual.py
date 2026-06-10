from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
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


def _feat(d_u: np.ndarray, pv: np.ndarray, ph: np.ndarray, pe: np.ndarray) -> np.ndarray:
    d = np.asarray(d_u, dtype=np.float64).ravel()
    return np.hstack(
        [
            d,
            np.sort(d)[:8],
            [np.min(d), np.max(d), np.mean(d), np.std(d)],
            pv,
            ph,
            pe,
        ]
    )


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    x = d_hat.T
    y = p.T

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cfg = VERSION_REGISTRY["v30"]
    oof_v30 = np.zeros((2, d_hat.shape[1]), dtype=np.float64)
    oof_hgb = np.zeros((2, d_hat.shape[1]), dtype=np.float64)
    oof_et = np.zeros((2, d_hat.shape[1]), dtype=np.float64)
    oof_stack = np.zeros((2, d_hat.shape[1]), dtype=np.float64)

    for fold, (tr, va) in enumerate(cv.split(np.arange(d_hat.shape[1]))):
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
        cfg_no = clone_pipeline_cfg(cfg, pos_refine_affine=False)
        xy_bounds_pre = bounds_from_positions(p[:, tr])
        p_tr_raw = np.zeros((2, len(tr)), dtype=np.float64)
        for j, u in enumerate(tr):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg_no, xy_bounds=xy_bounds_pre)
            p_tr_raw[:, j] = xy
        calib.pos_affine = fit_pos_affine(p_tr_raw, p[:, tr])
        xy_bounds = bounds_from_positions(p[:, tr])
        for u in tr:
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg, xy_bounds=xy_bounds)
            oof_v30[:, u] = xy
        for u in va:
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg, xy_bounds=xy_bounds)
            oof_v30[:, u] = xy

        hgb = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=800, learning_rate=0.03, max_depth=8, random_state=fold + 42)
        )
        et = ExtraTreesRegressor(n_estimators=700, random_state=fold + 42, n_jobs=-1, min_samples_leaf=1)
        hgb.fit(x[tr], y[tr])
        et.fit(x[tr], y[tr])
        oof_hgb[:, va] = hgb.predict(x[va]).T
        oof_et[:, va] = et.predict(x[va]).T
        pred_hgb_tr = hgb.predict(x[tr]).T
        pred_et_tr = et.predict(x[tr]).T

        xtr_meta = np.stack(
            [_feat(d_hat[:, u], oof_v30[:, u], pred_hgb_tr[:, j], pred_et_tr[:, j]) for j, u in enumerate(tr)],
            axis=0,
        )
        xva_meta = np.stack(
            [_feat(d_hat[:, u], oof_v30[:, u], oof_hgb[:, u], oof_et[:, u]) for u in va],
            axis=0,
        )
        yres_tr = (p[:, tr] - oof_v30[:, tr]).T

        meta = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_depth=6, random_state=100 + fold)
        )
        meta.fit(xtr_meta, yres_tr)
        res_va = meta.predict(xva_meta).T
        oof_stack[:, va] = oof_v30[:, va] + res_va

    print(f"v30_rmse={_rmse(p, oof_v30):.3f}")
    print(f"hgb_rmse={_rmse(p, oof_hgb):.3f}")
    print(f"et_rmse={_rmse(p, oof_et):.3f}")
    print(f"stack_residual_rmse={_rmse(p, oof_stack):.3f}")


if __name__ == "__main__":
    main()
