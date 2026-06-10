"""Leakage-safe HLOS-Rwgh pipeline: per-BS affine + LOS/Rwgh + residual + optional blend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier

from .hlos_rwgh import (
    HlosConfig,
    build_bs_features,
    fit_los_classifier,
    localize_user_hlos,
    los_labels,
    ls_init_weighted,
)
from .post_refine import apply_pos_affine, fit_pos_affine

Array = np.ndarray


@dataclass
class PerBsAffine:
    scale: Array  # (18,)
    bias: Array  # (18,)
    sigma: Array  # (18,) robust residual scale after fit

    def apply(self, d: Array) -> Array:
        d = np.asarray(d, dtype=np.float64).ravel()
        out = self.scale * d + self.bias
        return np.clip(out, 1.0, None)


def fit_per_bs_affine(
    d_hat: Array,
    p: Array,
    bs: Array,
    train_idx: Array,
    *,
    p_low: float = 1.0,
    p_high: float = 99.0,
    mad_z: float = 3.5,
) -> PerBsAffine:
    """Per-anchor affine d' = a_i d_i + c_i with percentile + MAD trimming (train only)."""
    d_hat = np.asarray(d_hat, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    n_bs = d_hat.shape[0]
    scale = np.ones(n_bs, dtype=np.float64)
    bias = np.zeros(n_bs, dtype=np.float64)
    sigma = np.ones(n_bs, dtype=np.float64)
    bx, by = bs[0], bs[1]

    for i in range(n_bs):
        d = d_hat[i, train_idx]
        r = np.sqrt((p[0, train_idx] - bx[i]) ** 2 + (p[1, train_idx] - by[i]) ** 2)
        lo, hi = np.percentile(d, [p_low, p_high])
        mask = (d >= lo) & (d <= hi)
        resid = d - r
        med = float(np.median(resid[mask])) if mask.any() else 0.0
        mad = float(np.median(np.abs(resid[mask] - med))) if mask.any() else 1.0
        mad = 1.4826 * mad + 1e-6
        sigma[i] = mad
        mask &= np.abs(resid - med) < mad_z * mad
        if int(mask.sum()) < 20:
            mask = np.ones_like(d, dtype=bool)
        X = np.column_stack([d[mask], np.ones(int(mask.sum()))])
        coef, *_ = np.linalg.lstsq(X, r[mask], rcond=None)
        scale[i] = float(coef[0])
        bias[i] = float(coef[1])
    return PerBsAffine(scale=scale, bias=bias, sigma=sigma)


def geometry_residual_features(d: Array, bs: Array, p_geo: Array, sigma_bs: Optional[Array] = None) -> Array:
    """Rich features for delta = p_true - p_geo (no absolute UE coords)."""
    d = np.asarray(d, dtype=np.float64).ravel()
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    p_geo = np.asarray(p_geo, dtype=np.float64).reshape(2)
    bx, by = bs[0], bs[1]
    d_pred = np.sqrt((p_geo[0] - bx) ** 2 + (p_geo[1] - by) ** 2)
    resid = d - d_pred
    abs_r = np.abs(resid)
    if sigma_bs is None:
        sigma_bs = np.ones_like(d)
    norm_r = resid / (np.asarray(sigma_bs, dtype=np.float64).ravel() + 1e-6)
    J = np.column_stack(
        [
            (p_geo[0] - bx) / (d_pred + 1e-3),
            (p_geo[1] - by) / (d_pred + 1e-3),
        ]
    )
    s = np.linalg.svd(J, compute_uv=False)
    cond = float(s[0] / (s[-1] + 1e-9))
    pct = np.percentile(np.r_[d, resid, abs_r], [10, 25, 50, 75, 90])
    return np.hstack(
        [
            d,
            d_pred,
            resid,
            abs_r,
            norm_r,
            p_geo,
            [d.min(), d.max(), d.mean(), d.std(), abs_r.mean(), abs_r.std(), np.linalg.norm(resid)],
            pct,
            [cond],
        ]
    ).astype(np.float64)


@dataclass
class HlosFoldModel:
    per_bs: Optional[PerBsAffine]
    hlos_cfg: HlosConfig
    los_clf: MLPClassifier
    residual: Optional[MultiOutputRegressor]
    pos_affine: Optional[Array]
    blend_alpha: Optional[float] = None
    anchor_predictor: Optional[object] = None

    def transform_d(self, d_raw: Array) -> Array:
        d_raw = np.asarray(d_raw, dtype=np.float64).ravel()
        if self.per_bs is not None:
            return self.per_bs.apply(d_raw)
        return d_raw

    def predict_hlos(self, d_raw: Array, bs: Array) -> Array:
        d = self.transform_d(d_raw)
        p = localize_user_hlos(d, bs, self.los_clf, self.hlos_cfg)
        if self.residual is not None:
            sig = self.per_bs.sigma if self.per_bs is not None else None
            feat = geometry_residual_features(d, bs, p, sig)
            p = p + self.residual.predict(feat[None, :])[0]
        if self.pos_affine is not None:
            p = apply_pos_affine(p, self.pos_affine)
        return np.asarray(p, dtype=np.float64).reshape(2)


def fit_hlos_fold_model(
    d_hat: Array,
    p: Array,
    bs: Array,
    train_idx: Array,
    *,
    hlos_cfg: HlosConfig,
    use_per_bs_affine: bool = True,
    use_residual: bool = True,
    use_pos_affine: bool = True,
    seed: int = 42,
) -> HlosFoldModel:
    per_bs = fit_per_bs_affine(d_hat, p, bs, train_idx) if use_per_bs_affine else None

    def get_d(u: int) -> Array:
        raw = d_hat[:, u]
        return per_bs.apply(raw) if per_bs is not None else raw

    X_list, y_list = [], []
    for u in train_idx:
        du = get_d(int(u))
        p0 = ls_init_weighted(du, bs)
        X_list.append(build_bs_features(du, bs, p0))
        y_list.append(los_labels(du, p[:, u], bs, hlos_cfg.los_thresh_m, use_abs=hlos_cfg.los_use_abs))
    los_clf = fit_los_classifier(np.vstack(X_list), np.concatenate(y_list), hlos_cfg, seed=seed)

    residual = None
    if use_residual:
        X_res, Y_res = [], []
        sig = per_bs.sigma if per_bs is not None else None
        for u in train_idx:
            du = get_d(int(u))
            ph = localize_user_hlos(du, bs, los_clf, hlos_cfg)
            X_res.append(geometry_residual_features(du, bs, ph, sig))
            Y_res.append(p[:, u] - ph)
        X_mat = np.stack(X_res)
        Y_mat = np.stack(Y_res)
        hgb_m = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.05, max_depth=10, random_state=seed
            )
        )
        et_m = MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=400, max_depth=14, min_samples_leaf=2, random_state=seed + 1, n_jobs=-1
            )
        )
        hgb_m.fit(X_mat, Y_mat)
        et_m.fit(X_mat, Y_mat)
        residual = _EnsembleResidual(hgb_m, et_m)

    pos_affine = None
    if use_pos_affine:
        p_tr = np.zeros((2, len(train_idx)), dtype=np.float64)
        m = HlosFoldModel(per_bs, hlos_cfg, los_clf, residual, None)
        for j, u in enumerate(train_idx):
            p_tr[:, j] = m.predict_hlos(d_hat[:, int(u)], bs)
        pos_affine = fit_pos_affine(p_tr, p[:, train_idx])

    return HlosFoldModel(per_bs, hlos_cfg, los_clf, residual, pos_affine)


class _EnsembleResidual:
    def __init__(self, hgb_m: MultiOutputRegressor, et_m: MultiOutputRegressor) -> None:
        self.hgb_m = hgb_m
        self.et_m = et_m

    def predict(self, X: Array) -> Array:
        return 0.5 * (self.hgb_m.predict(X) + self.et_m.predict(X))


def run_hlos_oof_cv(
    d_hat: Array,
    p: Array,
    bs: Array,
    *,
    hlos_cfg: HlosConfig,
    use_per_bs_affine: bool = True,
    use_residual: bool = True,
    use_pos_affine: bool = True,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    n = d_hat.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((2, n), dtype=np.float64)
    fold_rmse = []

    for fold_id, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        model = fit_hlos_fold_model(
            d_hat, p, bs, tr,
            hlos_cfg=hlos_cfg,
            use_per_bs_affine=use_per_bs_affine,
            use_residual=use_residual,
            use_pos_affine=use_pos_affine,
            seed=seed + fold_id,
        )
        for u in va:
            oof[:, u] = model.predict_hlos(d_hat[:, u], bs)
        e = np.hypot(p[0, va] - oof[0, va], p[1, va] - oof[1, va])
        rm = float(np.sqrt(np.mean(e**2)))
        fold_rmse.append(rm)

    e_all = np.hypot(p[0] - oof[0], p[1] - oof[1])
    return {
        "oof_pred": oof,
        "rmse_m": float(np.sqrt(np.mean(e_all**2))),
        "median_m": float(np.median(e_all)),
        "p90_m": float(np.percentile(e_all, 90)),
        "fold_rmse": fold_rmse,
    }
