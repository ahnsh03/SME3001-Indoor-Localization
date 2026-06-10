"""5-fold OOF: HLOS-Rwgh-WLS (+ optional residual), raw/calib switch."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.hlos_rwgh import (
    HlosConfig,
    build_bs_features,
    fit_los_classifier,
    localize_user_hlos,
    los_labels,
    ls_init_weighted,
)
from lib.io_mat import load_dataset
from lib.pipeline import VERSION_REGISTRY
from lib.post_refine import apply_pos_affine, fit_pos_affine
from lib.trilat import huber_trilat


def _stats(e: np.ndarray) -> str:
    return (
        f"rmse={float(np.sqrt(np.mean(e**2))):.3f} "
        f"median={float(np.median(e)):.3f} p90={float(np.percentile(e, 90)):.3f}"
    )


def _residual_features(d: np.ndarray, bs: np.ndarray, p_geo: np.ndarray) -> np.ndarray:
    bx, by = bs[0], bs[1]
    d_pred = np.sqrt((p_geo[0] - bx) ** 2 + (p_geo[1] - by) ** 2)
    resid = d - d_pred
    return np.hstack(
        [
            d,
            p_geo,
            d_pred,
            resid,
            np.abs(resid),
            resid / (np.std(resid) + 1e-6),
            [np.min(d), np.max(d), np.mean(d), np.std(d), np.linalg.norm(resid)],
        ]
    )


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    v30 = VERSION_REGISTRY["v30"]
    cfg = HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=True)
    use_v30_calib = False
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_hlos = np.zeros((2, n))
    oof_hlos_res = np.zeros((2, n))
    t0 = time.perf_counter()

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        if use_v30_calib:
            calib = fit_calib(
                d_hat, p, bs, tr, v30.calib,
                quantile_tau=v30.quantile_tau,
                mlp_noise_std=v30.mlp_noise_std,
            )
            get_d = lambda u: calib.apply(d_hat[:, u])
        else:
            get_d = lambda u: d_hat[:, u]

        X_list, y_list = [], []
        for u in tr:
            dc = get_d(u)
            p0 = ls_init_weighted(dc, bs)
            X_list.append(build_bs_features(dc, bs, p0))
            y_list.append(los_labels(dc, p[:, u], bs, cfg.los_thresh_m, use_abs=cfg.los_use_abs))
        Xtr = np.vstack(X_list)
        ytr = np.concatenate(y_list)
        los_clf = fit_los_classifier(Xtr, ytr, cfg, seed=42 + fold)

        # residual boost train: delta = p_true - p_hlos on train
        X_res, Y_res = [], []
        for u in tr:
            dc = get_d(u)
            ph = localize_user_hlos(dc, bs, los_clf, cfg)
            X_res.append(_residual_features(dc, bs, ph))
            Y_res.append(p[:, u] - ph)
        hgb = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=8, random_state=42 + fold)
        )
        hgb.fit(np.stack(X_res), np.stack(Y_res))

        for u in va:
            dc = get_d(u)
            ph = localize_user_hlos(dc, bs, los_clf, cfg)
            oof_hlos[:, u] = ph
            delta = hgb.predict(_residual_features(dc, bs, ph)[None, :])[0]
            oof_hlos_res[:, u] = ph + delta

        e = np.hypot(p[0, va] - oof_hlos[0, va], p[1, va] - oof_hlos[1, va])
        print(f"fold={fold} hlos {_stats(e)}")

    e_h = np.hypot(p[0] - oof_hlos[0], p[1] - oof_hlos[1])
    e_r = np.hypot(p[0] - oof_hlos_res[0], p[1] - oof_hlos_res[1])
    elapsed = time.perf_counter() - t0
    print(f"hlos_rwgh_oof {_stats(e_h)}  ({elapsed:.1f}s)")
    print(f"hlos+hgb_residual {_stats(e_r)}")

    # compare: v30 affine only on same folds (quick)
    oof_v30 = np.zeros((2, n))
    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        calib = fit_calib(d_hat, p, bs, tr, v30.calib, quantile_tau=v30.quantile_tau, mlp_noise_std=v30.mlp_noise_std)
        from lib.spatial import bounds_from_positions

        xy_bounds = bounds_from_positions(p[:, tr])
        p_tr = np.zeros((2, len(tr)))
        for j, u in enumerate(tr):
            dc = calib.apply(d_hat[:, u])
            xy, _ = huber_trilat(
                dc, bs, loss=v30.loss, f_scale=v30.huber_f_scale,
                weight_gamma=v30.weight_gamma, asym_pos_weight=v30.asym_pos_weight,
                xy_bounds=xy_bounds,
            )
            p_tr[:, j] = xy
        M = fit_pos_affine(p_tr, p[:, tr])
        for u in va:
            dc = calib.apply(d_hat[:, u])
            xy, _ = huber_trilat(
                dc, bs, loss=v30.loss, f_scale=v30.huber_f_scale,
                weight_gamma=v30.weight_gamma, asym_pos_weight=v30.asym_pos_weight,
                xy_bounds=xy_bounds,
            )
            oof_v30[:, u] = apply_pos_affine(xy, M)
    e_v = np.hypot(p[0] - oof_v30[0], p[1] - oof_v30[1])
    print(f"v30_oof_ref {_stats(e_v)}")


if __name__ == "__main__":
    main()
