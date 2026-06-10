"""HLOS-Rwgh-WLS: LOS classifier + residual-weighted subset trilateration (inference-safe features)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from sklearn.neural_network import MLPClassifier

from .spatial import X_BOUNDS, Y_BOUNDS

Array = np.ndarray


@dataclass
class HlosConfig:
    los_thresh_m: float = 2.0
    los_use_abs: bool = False
    top_k: int = 8
    subset_k: int = 4
    gamma_r: float = 2.0
    gamma_w: float = 4.0
    rwgh_keep_frac: float = 0.5
    mlp_hidden: tuple[int, ...] = (64, 64)
    mlp_max_iter: int = 100
    nls_loss: str = "soft_l1"
    nls_f_scale: float = 1.0
    box_margin_m: float = 30.0


def _true_dist(p: Array, bs: Array) -> Array:
    bx, by = bs[0], bs[1]
    return np.sqrt((p[0] - bx) ** 2 + (p[1] - by) ** 2)


def ls_init_weighted(d: Array, bs: Array, weights: Optional[Array] = None) -> Array:
    """Closed-form LS; reference anchor = minimum measured distance."""
    d = np.maximum(np.asarray(d, dtype=np.float64).ravel(), 1e-3)
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    ref = int(np.argmin(d))
    if weights is None:
        w = 1.0 / (d + 1.0) ** 2
    else:
        w = np.maximum(np.asarray(weights, dtype=np.float64).ravel(), 1e-6)
    w = w / (w.sum() + 1e-12)

    bx, by = bs[0], bs[1]
    rows, rhs, ww = [], [], []
    for i in range(d.size):
        if i == ref:
            continue
        a = -2.0 * np.array([bx[i] - bx[ref], by[i] - by[ref]])
        b = (
            d[i] ** 2
            - d[ref] ** 2
            - (bx[i] ** 2 - bx[ref] ** 2)
            - (by[i] ** 2 - by[ref] ** 2)
        )
        rows.append(a)
        rhs.append(b)
        ww.append(0.5 * (w[i] + w[ref]))
    if not rows:
        return np.array([bx[ref], by[ref]], dtype=np.float64)
    A = np.stack(rows)
    bvec = np.asarray(rhs)
    W = np.diag(ww)
    AtW = A.T @ W
    try:
        p = np.linalg.solve(AtW @ A + 1e-6 * np.eye(2), AtW @ bvec)
    except np.linalg.LinAlgError:
        p = np.array([bx[ref], by[ref]], dtype=np.float64)
    return p


def _clip_to_box(p: Array, bs: Array, margin: float) -> Array:
    x0 = float(bs[0].min() - margin)
    x1 = float(bs[0].max() + margin)
    y0 = float(bs[1].min() - margin)
    y1 = float(bs[1].max() + margin)
    xb = (max(x0, X_BOUNDS[0]), min(x1, X_BOUNDS[1]))
    yb = (max(y0, Y_BOUNDS[0]), min(y1, Y_BOUNDS[1]))
    p = np.asarray(p, dtype=np.float64).reshape(2)
    return np.array(
        [np.clip(p[0], xb[0], xb[1]), np.clip(p[1], yb[0], yb[1])],
        dtype=np.float64,
    )


def build_bs_features(d: Array, bs: Array, p0: Array) -> Array:
    """Per-anchor features (18, F) — no absolute UE coordinates."""
    d = np.asarray(d, dtype=np.float64).ravel()
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    bx, by = bs[0], bs[1]
    d_pred = np.sqrt((p0[0] - bx) ** 2 + (p0[1] - by) ** 2)
    resid = d - d_pred
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    scale = 1.4826 * mad + 1e-6
    z = (resid - med) / scale
    order = np.argsort(d)
    rank = np.empty_like(d)
    rank[order] = np.linspace(0.0, 1.0, d.size)
    dmin = float(d.min())
    return np.column_stack(
        [d, d_pred, resid, np.abs(resid), z, rank, np.full(d.size, med), np.full(d.size, mad), d - dmin]
    )


def los_labels(d: Array, p_true: Array, bs: Array, thresh: float, use_abs: bool = False) -> Array:
    dt = _true_dist(p_true, bs)
    diff = d - dt
    return ((np.abs(diff) < thresh) if use_abs else (diff < thresh)).astype(np.int64)


def fit_los_classifier(
    X: Array, y: Array, cfg: HlosConfig, seed: int = 42
) -> MLPClassifier:
    pos = max(1, int(y.sum()))
    neg = max(1, int((1 - y).sum()))
    pos_weight = neg / pos
    sample_w = np.where(y > 0, pos_weight, 1.0).astype(np.float64)
    clf = MLPClassifier(
        hidden_layer_sizes=cfg.mlp_hidden,
        activation="relu",
        max_iter=cfg.mlp_max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )
    clf.fit(X, y, sample_weight=sample_w)
    return clf


def _nls_point(d: Array, bs: Array, x0: Array, cfg: HlosConfig) -> tuple[Array, float]:
    d = np.asarray(d, dtype=np.float64).ravel()
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    bx, by = bs[0], bs[1]
    x0 = np.asarray(x0, dtype=np.float64).reshape(2)
    xb = (
        max(float(bs[0].min() - cfg.box_margin_m), X_BOUNDS[0]),
        min(float(bs[0].max() + cfg.box_margin_m), X_BOUNDS[1]),
    )
    yb = (
        max(float(bs[1].min() - cfg.box_margin_m), Y_BOUNDS[0]),
        min(float(bs[1].max() + cfg.box_margin_m), Y_BOUNDS[1]),
    )
    x0 = np.array(
        [np.clip(x0[0], xb[0], xb[1]), np.clip(x0[1], yb[0], yb[1])],
        dtype=np.float64,
    )

    def fun(xy: Array) -> Array:
        pred = np.sqrt((bx - xy[0]) ** 2 + (by - xy[1]) ** 2)
        return pred - d

    r = least_squares(
        fun,
        x0,
        loss=cfg.nls_loss,
        f_scale=cfg.nls_f_scale,
        bounds=([xb[0], yb[0]], [xb[1], yb[1]]),
    )
    cost = float(np.sqrt(np.mean(fun(r.x) ** 2)))
    return r.x, cost


def rwgh_midpoint(
    d: Array,
    bs: Array,
    p_los: Array,
    p0: Array,
    cfg: HlosConfig,
) -> Array:
    d = np.asarray(d, dtype=np.float64).ravel()
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    p_los = np.asarray(p_los, dtype=np.float64).ravel()
    top = np.argsort(-p_los)[: cfg.top_k]
    if top.size < cfg.subset_k:
        top = np.arange(d.size)

    pts, costs = [], []
    for sub in combinations(top.tolist(), cfg.subset_k):
        idx = np.asarray(sub, dtype=np.int64)
        d_sub = d[idx]
        bs_sub = bs[:, idx]
        x_init = ls_init_weighted(d_sub, bs_sub)
        xy, c = _nls_point(d_sub, bs_sub, x_init, cfg)
        k = idx.size
        r_norm = c / max(np.sqrt(k - 2), 1.0)
        pts.append(xy)
        costs.append(r_norm)

    if not pts:
        return p0.copy()
    costs = np.asarray(costs)
    pts = np.stack(pts)
    thr = np.quantile(costs, cfg.rwgh_keep_frac)
    mask = costs <= thr
    if not np.any(mask):
        mask = np.ones_like(costs, dtype=bool)
    c_sel = costs[mask]
    p_sel = pts[mask]
    w = 1.0 / (c_sel ** cfg.gamma_r + 1e-6)
    w = w / w.sum()
    return (p_sel * w[:, None]).sum(axis=0)


def final_weighted_nls(
    d: Array,
    bs: Array,
    p_los: Array,
    p_mid: Array,
    cfg: HlosConfig,
) -> Array:
    d = np.asarray(d, dtype=np.float64).ravel()
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    p_los = np.clip(np.asarray(p_los, dtype=np.float64).ravel(), 1e-4, 1.0 - 1e-4)
    w = np.maximum(p_los**cfg.gamma_w, 1e-3)
    sqrt_w = np.sqrt(w / (w.mean() + 1e-12))

    bx, by = bs[0], bs[1]

    def fun(xy: Array) -> Array:
        pred = np.sqrt((bx - xy[0]) ** 2 + (by - xy[1]) ** 2)
        return (pred - d) * sqrt_w

    xb = (
        max(float(bs[0].min() - cfg.box_margin_m), X_BOUNDS[0]),
        min(float(bs[0].max() + cfg.box_margin_m), X_BOUNDS[1]),
    )
    yb = (
        max(float(bs[1].min() - cfg.box_margin_m), Y_BOUNDS[0]),
        min(float(bs[1].max() + cfg.box_margin_m), Y_BOUNDS[1]),
    )
    r = least_squares(
        fun,
        np.array(
            [
                np.clip(float(np.asarray(p_mid)[0]), xb[0], xb[1]),
                np.clip(float(np.asarray(p_mid)[1]), yb[0], yb[1]),
            ],
            dtype=np.float64,
        ),
        loss=cfg.nls_loss,
        f_scale=cfg.nls_f_scale,
        bounds=([xb[0], yb[0]], [xb[1], yb[1]]),
    )
    return r.x


def localize_user_hlos(
    d: Array,
    bs: Array,
    los_clf: MLPClassifier,
    cfg: HlosConfig,
) -> Array:
    d = np.asarray(d, dtype=np.float64).ravel()
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    p0 = ls_init_weighted(d, bs)
    p0 = _clip_to_box(p0, bs, cfg.box_margin_m)
    feats = build_bs_features(d, bs, p0)
    p_los = los_clf.predict_proba(feats)[:, 1]
    p_mid = rwgh_midpoint(d, bs, p_los, p0, cfg)
    p_mid = _clip_to_box(p_mid, bs, cfg.box_margin_m)
    p_hat = final_weighted_nls(d, bs, p_los, p_mid, cfg)
    return _clip_to_box(p_hat, bs, cfg.box_margin_m)
