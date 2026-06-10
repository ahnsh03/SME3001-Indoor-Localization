from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.io_mat import geometric_distances, load_dataset
from lib.pipeline import VERSION_REGISTRY
from lib.post_refine import apply_pos_affine, fit_pos_affine
from lib.spatial import X_BOUNDS, Y_BOUNDS, bounds_from_positions
from lib.trilat import _distance_weights


def _rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def weighted_huber_trilat(
    d_obs: np.ndarray,
    bs: np.ndarray,
    *,
    weights: np.ndarray,
    f_scale: float,
    asym_pos_weight: float,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]],
    x0: np.ndarray | None = None,
) -> np.ndarray:
    d_obs = np.asarray(d_obs, dtype=np.float64).ravel()
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    bx, by = bs[0], bs[1]
    sqrt_w = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64).ravel(), 1e-8))
    if x0 is None:
        x0 = bs.mean(axis=1)
    xb, yb = xy_bounds

    def residual(xy: np.ndarray) -> np.ndarray:
        pred = np.sqrt((bx - xy[0]) ** 2 + (by - xy[1]) ** 2)
        r = pred - d_obs
        if asym_pos_weight != 1.0:
            rw = np.where(r > 0, asym_pos_weight, 1.0)
            r = r * np.sqrt(rw)
        return r * sqrt_w

    out = least_squares(
        residual,
        x0=np.asarray(x0, dtype=np.float64),
        loss="huber",
        f_scale=f_scale,
        bounds=([xb[0], yb[0]], [xb[1], yb[1]]),
        max_nfev=150,
    )
    return np.asarray(out.x, dtype=np.float64)


def anchor_features(d_corr_u: np.ndarray, bs: np.ndarray) -> np.ndarray:
    d = np.asarray(d_corr_u, dtype=np.float64).ravel()
    order = np.argsort(d)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(d.size, dtype=np.float64) / max(d.size - 1, 1)
    if bs.shape[0] != 2:
        bs = bs.T
    bx, by = bs[0], bs[1]
    mn, mx, mu, sd = d.min(), d.max(), d.mean(), d.std()
    feats = np.stack(
        [
            d,
            rank,
            bx,
            by,
            np.full_like(d, mn),
            np.full_like(d, mx),
            np.full_like(d, mu),
            np.full_like(d, sd),
        ],
        axis=1,
    )
    return feats


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    cfg = VERSION_REGISTRY["v30"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_base = np.zeros((2, n), dtype=np.float64)
    oof_learned = np.zeros((2, n), dtype=np.float64)

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        xy_bounds = bounds_from_positions(p[:, tr])

        calib = fit_calib(
            d_hat,
            p,
            bs,
            tr,
            cfg.calib,
            quantile_tau=cfg.quantile_tau,
            mlp_noise_std=cfg.mlp_noise_std,
        )

        # baseline weights from v30 gamma
        def base_weights(dcorr: np.ndarray) -> np.ndarray:
            wg = cfg.weight_gamma if cfg.weight_gamma is not None else 1.0
            return _distance_weights(dcorr, wg)

        # train anchor reliability regressor on train users
        rows = []
        targets = []
        d_true_tr = geometric_distances(p[:, tr], bs)  # (18, n_tr)
        for j, u in enumerate(tr):
            dc = calib.apply(d_hat[:, u])
            feats = anchor_features(dc, bs)
            err = np.abs(dc - d_true_tr[:, j])
            # reliability target: inverse error
            w_tgt = 1.0 / (1.0 + err)
            rows.append(feats)
            targets.append(w_tgt)
        Xw = np.vstack(rows)  # (n_tr*18, f)
        yw = np.concatenate(targets)
        w_model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=30,
            random_state=42 + fold,
        )
        w_model.fit(Xw, yw)

        # fit affine for baseline and learned separately on train fold
        p_tr_base = np.zeros((2, len(tr)), dtype=np.float64)
        p_tr_learn = np.zeros((2, len(tr)), dtype=np.float64)
        for j, u in enumerate(tr):
            dc = calib.apply(d_hat[:, u])
            wb = base_weights(dc)
            wl = np.clip(w_model.predict(anchor_features(dc, bs)), 1e-4, None)
            wl = wl / (wl.mean() + 1e-12)
            p_tr_base[:, j] = weighted_huber_trilat(
                dc,
                bs,
                weights=wb,
                f_scale=cfg.huber_f_scale,
                asym_pos_weight=cfg.asym_pos_weight,
                xy_bounds=xy_bounds,
            )
            p_tr_learn[:, j] = weighted_huber_trilat(
                dc,
                bs,
                weights=wl,
                f_scale=cfg.huber_f_scale,
                asym_pos_weight=cfg.asym_pos_weight,
                xy_bounds=xy_bounds,
            )
        M_base = fit_pos_affine(p_tr_base, p[:, tr])
        M_learn = fit_pos_affine(p_tr_learn, p[:, tr])

        for u in va:
            dc = calib.apply(d_hat[:, u])
            wb = base_weights(dc)
            wl = np.clip(w_model.predict(anchor_features(dc, bs)), 1e-4, None)
            wl = wl / (wl.mean() + 1e-12)
            xb = weighted_huber_trilat(
                dc,
                bs,
                weights=wb,
                f_scale=cfg.huber_f_scale,
                asym_pos_weight=cfg.asym_pos_weight,
                xy_bounds=xy_bounds,
            )
            xl = weighted_huber_trilat(
                dc,
                bs,
                weights=wl,
                f_scale=cfg.huber_f_scale,
                asym_pos_weight=cfg.asym_pos_weight,
                xy_bounds=xy_bounds,
            )
            xb = apply_pos_affine(xb, M_base)
            xl = apply_pos_affine(xl, M_learn)
            xb[0] = np.clip(xb[0], X_BOUNDS[0], X_BOUNDS[1])
            xb[1] = np.clip(xb[1], Y_BOUNDS[0], Y_BOUNDS[1])
            xl[0] = np.clip(xl[0], X_BOUNDS[0], X_BOUNDS[1])
            xl[1] = np.clip(xl[1], Y_BOUNDS[0], Y_BOUNDS[1])
            oof_base[:, u] = xb
            oof_learned[:, u] = xl

    print(f"baseline_v30_style_oof_rmse={_rmse(p, oof_base):.3f}")
    print(f"learned_anchor_weight_oof_rmse={_rmse(p, oof_learned):.3f}")


if __name__ == "__main__":
    main()

