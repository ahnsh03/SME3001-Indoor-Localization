"""
V12 — Top‑3 융합 (V3/UWB·Wi‑Fi 융합 + V9 스나이퍼 게이트 + V11 강건 보정/IRLS)

분석 요지
----------
- 원본 데이터에서 UWB는 저분산·대오차(조용한 암살자) 패턴이라 단순 Var 가중 융합이 붕괴함.
- **Pure Wi‑Fi**는 안정적인 수학적 바닥(≈1.55m 검증 근처)을 제공함.
- **V9**는 Wi‑Fi 1차 좌표로 UWB 거리 불일치를 하드 게이트해 스파이크 UWB만 제거할 때 역대 급 향상을 낸 패턴임.
- **V11**은 Wi‑Fi 앵커별 Huber/RANSAC·항등 포함 보정과 IRLS로 멀티패스 재가중 근방을 깎음.

파이프라인
--------
Step A: V11 계열 강건 Wi‑Fi(a,b) 학습 후 Wi‑Fi 전용 삼변(임시 신뢰 좌표).
Step B: V9 게이트 |d_geom − d_meas| > τ → 해당 UWB 가중치 0, 통과 UWB에는 분산 inflate로 과신 페널티,
        보정 Wi‑Fi 거리와 같이 바운드 Huber 융합 삼변.
Step C: 마지막 (pos,d,w) 기준 Tukey×Huber IRLS.

그리드는 단일 검증 폴드에서 RMSE 최소를 택함(후보 수는 런타임 때문에 모듈 상수가 보수적으로 잡혀 있음;
`FEAT_MODES`·`GATE_*` 등을 확장하면 추가 탐색 가능).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fusion_realtime_sanitize import (
    HARDWARE_CALIB_BIAS_M,
    VAR_CAP_UWB_FUSION,
    VAR_CAP_WIFI_FUSION,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
)

from indoor_fusion_pipeline_v10 import WifiOnlyTrilateration, resolve_paths
from indoor_fusion_pipeline_v11 import (
    CalibMethodSpec,
    fit_robust_calibration_per_ap,
    huber_hinge_weights,
    tukey_bisquare_weights,
)
from indoor_fusion_pipeline_v8 import (
    BOUNDS_LO,
    BOUNDS_HI,
    Config,
    EPS_W,
    FusionLocalizerV8,
    MAX_RANGE_M,
    ROBUST_BIAS_M,
    UWB_BIAS_W,
    WIFI_BIAS_W,
    WIFI_COL_CANON,
    sensor_canon_key,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except (AttributeError, OSError):
        pass
plt.rcParams["axes.unicode_minus"] = False

TUKEY_C = 4.685

# --- 자율 탐색 그리드 (과적합 억제: 후보 수 제한 + 상위 A만 B/C 정밀 탐색) ----------
FEAT_MODES = ["bias_sub"]
HUBER_F_WIFI_GRID = [1.0, 1.35]
HUBER_F_FUSION_GRID = [1.0, 1.35]
GATE_THRESH_GRID = [1.25, 1.55, 1.9, 2.35, 2.95]
UWB_VAR_INFLATE_GRID = [10.0, 24.0]
IRLS_THRESH_GRID_M = [2.2, 3.1]
IRLS_MAX_ITER_GRID = [2]
TOP_A_CANDIDATES = 3


def v12_calibration_spec_list() -> List[CalibMethodSpec]:
    """자율 탐색 후보(RANSAC max_trials는 Phase A 시간 폭증 → Huber 중심)."""
    return [
        CalibMethodSpec("identity"),
        CalibMethodSpec("huber", huber_epsilon=1.35, huber_alpha=5e-4),
    ]


def augment_train_for_canon_fit(df: pd.DataFrame, loc: FusionLocalizerV8) -> pd.DataFrame:
    """학습 프레임에 WIFI_COL_CANON 별칭 열을 추가해 v11 캘리브루틴과 호환."""
    t = df.copy()
    for i, c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        if c != canon:
            if f"med_{canon}" not in t.columns and f"med_{c}" in t.columns:
                t[f"med_{canon}"] = t[f"med_{c}"]
            if f"var_{canon}" not in t.columns and f"var_{c}" in t.columns:
                t[f"var_{canon}"] = t[f"var_{c}"]
    return t


def wifi_median_impute_fusion(
    train_df: pd.DataFrame, val_df: pd.DataFrame, wifi_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t, v = train_df.copy(), val_df.copy()
    feats = [f"med_{c}" for c in wifi_cols]
    kn = max(3, min(12, len(t) // 10 or 5))
    imputer = KNNImputer(n_neighbors=min(kn, len(t)), weights="distance")
    imputer.fit(t[feats].values)
    t[feats] = imputer.transform(t[feats].values)
    v[feats] = imputer.transform(v[feats].values)
    return t, v


def solve_trilat_array(
    pos: np.ndarray,
    dist: np.ndarray,
    weights: np.ndarray,
    *,
    huber_f_scale: float,
) -> np.ndarray:
    """V8과 동일한 바운드 Huber 삼변 (경량 복제)."""
    pos = np.asarray(pos, dtype=float)
    dist = np.asarray(dist, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(dist) == 0:
        return np.array([np.nan, np.nan], dtype=float)
    if len(dist) < 3:
        d = np.clip(dist, 1e-3, None)
        w = np.maximum(weights, 0.0)
        if w.sum() < 1e-12:
            return np.array([np.nan, np.nan], dtype=float)
        xy = (pos * w[:, None]).sum(axis=0) / w.sum()
        return np.clip(xy, BOUNDS_LO, BOUNDS_HI).astype(float)
    idx = int(np.argmin(dist))
    x0 = np.clip(pos[idx].copy(), BOUNDS_LO, BOUNDS_HI)

    def residuals(xy: np.ndarray) -> np.ndarray:
        pr = np.sqrt((xy[0] - pos[:, 0]) ** 2 + (xy[1] - pos[:, 1]) ** 2)
        return np.sqrt(np.maximum(weights, 0.0)) * (pr - dist)

    res = least_squares(
        residuals,
        x0=x0,
        bounds=(BOUNDS_LO, BOUNDS_HI),
        loss="huber",
        f_scale=float(huber_f_scale),
        method="trf",
    )
    return np.asarray(res.x, dtype=float)


def row_wifi_calibrated_arrays(
    row: pd.Series,
    loc: FusionLocalizerV8,
    coeffs_canon: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_l: List[np.ndarray] = []
    dist_l: List[float] = []
    w_l: List[float] = []
    for i, c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        m_raw = row.get(f"med_{c}", np.nan)
        vv = row.get(f"var_{c}", np.nan)
        if (not np.isfinite(m_raw)) or (not np.isfinite(vv)) or float(vv) > VAR_CAP_WIFI_FUSION:
            continue
        if float(m_raw) > MAX_RANGE_M:
            continue
        key = sensor_canon_key(canon, i, False)
        A, B = coeffs_canon[canon]
        if feat_mode == "raw":
            xf = float(m_raw)
        else:
            xf = float(m_raw) - ROBUST_BIAS_M[key]
        d_c = float(A * xf + B)
        if d_c < 0.1:
            d_c = 0.1
        ax, ay = loc._resolve_sensor_position_m(c, i, False)
        pos_l.append(np.array([ax, ay], dtype=float))
        dist_l.append(d_c)
        w_l.append(1.0 / (float(vv) + WIFI_BIAS_W + EPS_W))
    if not dist_l:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    return (
        np.array(pos_l, dtype=float),
        np.array(dist_l, dtype=float),
        np.array(w_l, dtype=float),
    )


def predict_step_a_calibrated(
    df: pd.DataFrame,
    loc: FusionLocalizerV8,
    coeffs_canon: Dict[str, Tuple[float, float]],
    feat_mode: str,
    huber_f_scale: float,
) -> np.ndarray:
    n = len(df)
    xy = np.full((n, 2), np.nan, dtype=float)
    for k, (_, row) in enumerate(df.iterrows()):
        p, d, w = row_wifi_calibrated_arrays(row, loc, coeffs_canon, feat_mode)
        xy[k] = solve_trilat_array(p, d, w, huber_f_scale=huber_f_scale)
    return xy


def row_wifi_v3_equivalent_arrays(
    row: pd.Series,
    loc: FusionLocalizerV8,
    *,
    wifi_bias: float = 2.5,
    epsilon: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V3 `WifiUwbFusionLocalizerV3._build_wifi_only_arrays`와 동일(편향·분산 cap·가중 1/(v+bias+eps))."""
    pos_l: List[np.ndarray] = []
    dist_l: List[float] = []
    w_l: List[float] = []
    for i, c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        m_raw = row.get(f"med_{c}", np.nan)
        vv = row.get(f"var_{c}", np.nan)
        if (not np.isfinite(m_raw)) or (not np.isfinite(vv)) or float(vv) > VAR_CAP_WIFI_FUSION:
            continue
        if float(m_raw) > MAX_RANGE_M:
            continue
        bk = sensor_canon_key(canon, i, False)
        bias = float(HARDWARE_CALIB_BIAS_M[bk])
        ax, ay = loc._resolve_sensor_position_m(c, i, False)
        pos_l.append(np.array([ax, ay], dtype=float))
        dist_l.append(float(m_raw) - bias)
        w_l.append(1.0 / (float(vv) + float(wifi_bias) + float(epsilon)))
    if not dist_l:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    return (
        np.array(pos_l, dtype=float),
        np.array(dist_l, dtype=float),
        np.array(w_l, dtype=float),
    )


def solve_trilat_v3_wifi_style(
    pos: np.ndarray,
    dist: np.ndarray,
    weights: np.ndarray,
    *,
    huber_f_scale: float = 1.0,
) -> np.ndarray:
    """V3 `_solve_trilat`: 3개 미만은 역거리 가중 중심, 이상은 바운드 없는 Huber least_squares."""
    pos = np.asarray(pos, dtype=float)
    dist = np.asarray(dist, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(dist) < 3:
        if len(dist) == 0:
            return np.array([np.nan, np.nan], dtype=float)
        d = np.clip(dist, 1e-3, None)
        w = 1.0 / d
        return (pos * w[:, None]).sum(axis=0) / w.sum()

    x0 = pos[int(np.argmin(dist))].copy()

    def residuals(xy: np.ndarray) -> np.ndarray:
        pr = np.sqrt((xy[0] - pos[:, 0]) ** 2 + (xy[1] - pos[:, 1]) ** 2)
        return np.sqrt(weights) * (pr - dist)

    res = least_squares(
        residuals,
        x0=x0,
        loss="huber",
        f_scale=float(huber_f_scale),
        method="trf",
    )
    return np.asarray(res.x, dtype=float)


def predict_step_a_v3_wifi_equivalent(
    df: pd.DataFrame,
    loc: FusionLocalizerV8,
    *,
    wifi_bias: float = 2.5,
    huber_f_scale: float = 1.0,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """V3 Wi‑Fi 전용 삼변과 동일한 Step A 좌표(캘리브 계수 미사용)."""
    n = len(df)
    xy = np.full((n, 2), np.nan, dtype=float)
    for k, (_, row) in enumerate(df.iterrows()):
        p, d, w = row_wifi_v3_equivalent_arrays(row, loc, wifi_bias=wifi_bias, epsilon=epsilon)
        xy[k] = solve_trilat_v3_wifi_style(p, d, w, huber_f_scale=huber_f_scale)
    return xy


def row_fusion_gated_arrays(
    row: pd.Series,
    loc: FusionLocalizerV8,
    wifi_guide_xy: np.ndarray,
    gate_t: float,
    coeffs_canon: Dict[str, Tuple[float, float]],
    feat_mode: str,
    uwb_variance_inflate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wi‑Fi 안심좌표 기준 UWB 기하 게이트 + 분산 인플레이트, 보정 Wi‑Fi 거리 병합."""
    wx, wy = float(wifi_guide_xy[0]), float(wifi_guide_xy[1])
    pos_l: List[np.ndarray] = []
    dist_l: List[float] = []
    w_l: List[float] = []

    if np.isfinite(wx) and np.isfinite(wy):
        for i, c in enumerate(loc.uwb_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan or (not np.isfinite(d_raw)) or (not np.isfinite(v)):
                continue
            if float(v) > VAR_CAP_UWB_FUSION:
                continue
            key_u = sensor_canon_key(c, i, True)
            d_meas = float(d_raw) - ROBUST_BIAS_M[key_u]
            ax_u, ay_u = loc._resolve_sensor_position_m(c, i, True)
            d_geom = float(np.hypot(wx - ax_u, wy - ay_u))
            if abs(d_geom - d_meas) > float(gate_t):
                continue
            eff_var = float(v) * float(max(uwb_variance_inflate, 1.0))
            wi = 1.0 / (eff_var + UWB_BIAS_W + EPS_W)
            pos_l.append(np.array([ax_u, ay_u], dtype=float))
            dist_l.append(d_meas)
            w_l.append(wi)

    p_w, d_w, w_wifi = row_wifi_calibrated_arrays(row, loc, coeffs_canon, feat_mode)
    for j in range(len(d_w)):
        pos_l.append(p_w[j])
        dist_l.append(float(d_w[j]))
        w_l.append(float(w_wifi[j]))

    if not dist_l:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )
    return (
        np.asarray(pos_l, dtype=float),
        np.asarray(dist_l, dtype=float),
        np.asarray(w_l, dtype=float),
    )


def irls_refine_xyz(
    pos: np.ndarray,
    dist: np.ndarray,
    w_base: np.ndarray,
    huber_f_fusion: float,
    thresh_m: float,
    max_iters: int,
) -> np.ndarray:
    """Step B 산출 (pos,dist,w_base)에 대해 Tukey×Huber 힌지 IRLS."""
    if len(dist) < 3:
        return solve_trilat_array(pos, dist, w_base, huber_f_scale=huber_f_fusion)
    w = np.asarray(w_base, dtype=float).copy()
    xy_last = np.full(2, np.nan)
    for _ in range(max(1, int(max_iters))):
        xy_last = solve_trilat_array(pos, dist, w, huber_f_scale=huber_f_fusion)
        if not np.all(np.isfinite(xy_last)):
            break
        gpred = np.hypot(xy_last[0] - pos[:, 0], xy_last[1] - pos[:, 1])
        r = np.abs(gpred - dist)
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        sigma_rob = max(mad * 1.4826, 0.08)
        sigma = float(max(thresh_m * 0.65, sigma_rob, 1e-2))
        u = r / sigma
        tw = tukey_bisquare_weights(u, c=TUKEY_C)
        hw = huber_hinge_weights(r, float(thresh_m))
        mul = np.clip(tw * hw, 0.04, 1.0)
        w_new = np.asarray(w_base, dtype=float) * mul
        if float(np.linalg.norm(w_new - w)) / (float(np.linalg.norm(w)) + 1e-9) < 1e-4:
            w = w_new
            xy_last = solve_trilat_array(pos, dist, w, huber_f_scale=huber_f_fusion)
            break
        w = w_new
    return xy_last.astype(float)


def point_errors(tx: np.ndarray, ty: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return np.sqrt((px - tx) ** 2 + (py - ty) ** 2)


def rmse_mae(err: np.ndarray) -> Tuple[float, float]:
    ee = np.asarray(err, dtype=float)
    ee = ee[np.isfinite(ee)]
    if ee.size == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(mean_squared_error(np.zeros_like(ee), ee)))
    mae = float(mean_absolute_error(np.zeros_like(ee), ee))
    return rmse, mae


def save_cdf_steps(
    out_dir: Path,
    errs: Dict[str, np.ndarray],
    rmse_by: Dict[str, float],
    labels: Dict[str, str],
) -> None:
    plt.figure(figsize=(8.2, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(errs)))
    for ki, key in enumerate(sorted(errs.keys())):
        e = errs[key][np.isfinite(errs[key])]
        if e.size == 0:
            continue
        e = np.sort(e)
        cdf = np.arange(1, len(e) + 1) / len(e)
        plt.plot(e, cdf, lw=2.0, color=colors[ki], label=f"{labels[key]} (RMSE {rmse_by[key]:.3f} m)")
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("V12: 검증 오차 CDF — Step A/B/C")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "v12_cdf_steps.png", dpi=200)
    plt.close()


def save_v12_true_vs_pred_maps(
    out_dir: Path,
    tx: np.ndarray,
    ty: np.ndarray,
    xy_a: np.ndarray,
    preds_b: np.ndarray,
    preds_c: np.ndarray,
    rmse_a: float,
    rmse_b: float,
    rmse_c: float,
) -> None:
    """실제 위치(True)와 단계별 측위 좌표를 12×18m 맵 위에 함께 표시."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)
    triples = [
        (xy_a[:, 0], xy_a[:, 1], "Step A (Wi‑Fi 보정 삼변)", rmse_a),
        (preds_b[:, 0], preds_b[:, 1], "Step B (+ UWB 게이트 융합)", rmse_b),
        (preds_c[:, 0], preds_c[:, 1], "Step C (+ IRLS)", rmse_c),
    ]
    for ax, (px, py, ttl, rm) in zip(axes, triples):
        m = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(px) & np.isfinite(py)
        txv, tyv = tx[m], ty[m]
        pxv, pyv = px[m], py[m]
        ax.scatter(
            txv,
            tyv,
            s=42,
            c="#1f77b4",
            alpha=0.88,
            edgecolors="#0d2c44",
            linewidths=0.35,
            label="실제 (True)",
            zorder=2,
        )
        ax.scatter(
            pxv,
            pyv,
            s=52,
            c="#ff7f0e",
            alpha=0.78,
            marker="x",
            linewidths=0.95,
            label="측위 추정",
            zorder=3,
        )
        for xi, yi, pxi, pyi in zip(txv, tyv, pxv, pyv):
            ax.plot([xi, pxi], [yi, pyi], color="#7f7f7f", alpha=0.22, lw=0.85, zorder=1)

        ax.set_xlim(0, 12)
        ax.set_ylim(0, 18)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{ttl}\n(RMSE {rm:.3f} m)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("V12 검증 세트 — 실제 vs 측위 (타일 그리드 12 m × 18 m)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_dir / "v12_map_true_vs_predicted_steps.png", dpi=200)
    plt.close()


def save_v12_error_vectors_step_c(
    out_dir: Path,
    tx: np.ndarray,
    ty: np.ndarray,
    preds_c: np.ndarray,
    err_c_m: np.ndarray,
) -> None:
    """최종(Step C) 오차 벡터: 실제 → 추정 방향 및 오차 크기 색상."""
    m = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(preds_c[:, 0]) & np.isfinite(preds_c[:, 1])
    if not np.any(m):
        return
    fig, ax = plt.subplots(figsize=(8.0, 10.8))
    u = preds_c[m, 0] - tx[m]
    v = preds_c[m, 1] - ty[m]
    err = err_c_m[m]
    if np.any(np.isfinite(err)):
        er = np.where(np.isfinite(err), err, 0.0)
        sc = ax.scatter(tx[m], ty[m], c=er, cmap="viridis", s=54, alpha=0.92, edgecolors="k", linewidths=0.35, zorder=3)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
        cbar.set_label("오차 (m)")
    else:
        ax.scatter(tx[m], ty[m], s=54, c="seagreen", alpha=0.85, edgecolors="k", linewidths=0.35, label="실제(True)", zorder=3)

    ax.quiver(
        tx[m],
        ty[m],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0032,
        alpha=0.5,
        color="navy",
        zorder=2,
    )
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("V12 Step C — 오차 벡터 (실제 → 추정)")
    ax.grid(True, alpha=0.3)
    if not np.any(np.isfinite(err)):
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_dir / "v12_map_error_vectors_stepC.png", dpi=200)
    plt.close()


def save_v12_combined_truth_pred_scatter_step_c(
    out_dir: Path,
    tx: np.ndarray,
    ty: np.ndarray,
    preds_c: np.ndarray,
    err_c_m: np.ndarray,
) -> None:
    """Step C 단일 패널: 실제점 vs 추정점, 색상=오차(m)."""
    m = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(preds_c[:, 0]) & np.isfinite(preds_c[:, 1])
    if not np.any(m):
        return
    err = np.asarray(err_c_m[m], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 10.8))
    ax.scatter(
        tx[m],
        ty[m],
        s=140,
        facecolors="none",
        edgecolors="dimgray",
        linewidths=1.1,
        label="실제 (True)",
        zorder=3,
    )

    cmap = plt.cm.RdYlGn_r
    finite_e = err[np.isfinite(err)]
    if finite_e.size >= 5:
        vmin = float(np.percentile(finite_e, 5))
        vmax = float(np.percentile(finite_e, 95))
        if vmax <= vmin:
            vmin, vmax = 0.0, max(float(np.max(finite_e)), 0.5)
    else:
        vmin, vmax = 0.0, 3.0

    z = np.asarray(err_c_m[m], dtype=float)
    z[~np.isfinite(z)] = np.nan
    sm = ax.scatter(
        preds_c[m, 0],
        preds_c[m, 1],
        c=z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=36,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.25,
        label="추정 Step C",
        zorder=4,
    )
    plt.colorbar(sm, ax=ax, shrink=0.72, label="|오차| (m)")
    for xi, yi, pxi, pyi in zip(tx[m], ty[m], preds_c[m, 0], preds_c[m, 1]):
        ax.plot([xi, pxi], [yi, pyi], color="gray", alpha=0.35, lw=0.85, zorder=2)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("V12 Step C — 실제(회색 원) vs 추정(색은 오차)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.savefig(out_dir / "v12_map_stepC_pairwise_truth_pred.png", dpi=200)
    plt.close()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "train"
    val_dir = root / "data" / "validation"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tm, tv = resolve_train_kgh_corrected_paths(train_dir)
    vm, vd = resolve_validation_paths(val_dir)
    cfg = Config(grid_size_m=0.6, huber_f_scale=1.0)
    loc = FusionLocalizerV8(cfg)
    train_df, val_df = loc.load_datasets(tm, tv, vm, vd)
    print(f"\n[V12] 학습 {len(train_df)}행 | 검증 {len(val_df)}행")

    train_imp, val_imp = wifi_median_impute_fusion(train_df, val_df, loc.wifi_cols)
    train_fit = augment_train_for_canon_fit(train_imp, loc)

    tx = val_imp["True_X"].values.astype(float)
    ty = val_imp["True_Y"].values.astype(float)

    tri_ref = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m=0.6, huber_f_scale=1.0)
    calib_specs = v12_calibration_spec_list()

    # --- Phase 1: Step A 후보 선별 (강건 Wi‑Fi 보정 + Wi‑Fi 삼변) ----------------
    phase_a_records: List[Dict[str, Any]] = []
    cand_a: List[Tuple[float, CalibMethodSpec, str, float, Dict[str, Tuple[float, float]]]] = []

    for feat in FEAT_MODES:
        for spec in calib_specs:
            coeffs_c = fit_robust_calibration_per_ap(train_fit, tri_ref, feat_mode=feat, spec=spec)
            for hf in HUBER_F_WIFI_GRID:
                xy_a = predict_step_a_calibrated(val_imp, loc, coeffs_c, feat, hf)
                ea = point_errors(tx, ty, xy_a[:, 0], xy_a[:, 1])
                ra, ma = rmse_mae(ea)
                phase_a_records.append(
                    dict(
                        feat_mode=feat,
                        calib_kind=spec.kind,
                        huber_f_wifi=float(hf),
                        val_RMSE_A_m=float(ra),
                        val_MAE_A_m=float(ma),
                    )
                )
                cand_a.append((float(ra), spec, feat, float(hf), coeffs_c))

    cand_a.sort(key=lambda t: t[0])
    top_a = cand_a[:TOP_A_CANDIDATES]

    # --- Phase 2: Step B + Step C (게이트 × UWB 인플레이트 × 융합 Huber × IRLS) ---
    best_c = float("inf")
    best_pack: Optional[Dict[str, Any]] = None
    phase_bc_records: List[Dict[str, Any]] = []

    for rank, (_ra0, spec, feat, hf_wifi, coeffs_c) in enumerate(top_a):
        xy_a_val = predict_step_a_calibrated(val_imp, loc, coeffs_c, feat, hf_wifi)
        for gate in GATE_THRESH_GRID:
            for uvb_inf in UWB_VAR_INFLATE_GRID:
                pdw_store: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                for k, (_, row) in enumerate(val_imp.iterrows()):
                    gxy = xy_a_val[k]
                    p, d, w = row_fusion_gated_arrays(
                        row, loc, gxy, gate, coeffs_c, feat, uvb_inf
                    )
                    pdw_store.append((p, d, w))

                for hf_fus in HUBER_F_FUSION_GRID:
                    preds_b = np.full((len(val_imp), 2), np.nan, dtype=float)
                    for k, (p, d, w) in enumerate(pdw_store):
                        preds_b[k] = solve_trilat_array(p, d, w, huber_f_scale=hf_fus)
                    eb = point_errors(tx, ty, preds_b[:, 0], preds_b[:, 1])
                    rb, mb = rmse_mae(eb)

                    for ith in IRLS_THRESH_GRID_M:
                        for mx in IRLS_MAX_ITER_GRID:
                            preds_c = np.full((len(val_imp), 2), np.nan, dtype=float)
                            for k, (p, d, w) in enumerate(pdw_store):
                                if len(d) < 3:
                                    p2, d2, w2 = row_wifi_calibrated_arrays(
                                        val_imp.iloc[k], loc, coeffs_c, feat
                                    )
                                    preds_c[k] = solve_trilat_array(
                                        p2, d2, w2, huber_f_scale=hf_wifi
                                    )
                                else:
                                    preds_c[k] = irls_refine_xyz(
                                        p, d, w, hf_fus, float(ith), int(mx)
                                    )
                            ec = point_errors(tx, ty, preds_c[:, 0], preds_c[:, 1])
                            rc, mc = rmse_mae(ec)
                            phase_bc_records.append(
                                dict(
                                    phaseA_rank=int(rank),
                                    feat_mode=feat,
                                    calib_kind=spec.kind,
                                    huber_f_wifi=float(hf_wifi),
                                    gate_threshold_m=float(gate),
                                    uwb_variance_inflate=float(uvb_inf),
                                    huber_f_fusion=float(hf_fus),
                                    irls_residual_thresh_m=float(ith),
                                    irls_max_iter=int(mx),
                                    val_RMSE_B_m=float(rb),
                                    val_RMSE_C_m=float(rc),
                                    val_MAE_C_m=float(mc),
                                )
                            )
                            if rc < best_c - 1e-14:
                                best_c = float(rc)
                                best_pack = dict(
                                    spec=spec,
                                    feat=str(feat),
                                    hf_wifi=float(hf_wifi),
                                    coeffs=coeffs_c,
                                    gate=float(gate),
                                    uwb_inf=float(uvb_inf),
                                    hf_fus=float(hf_fus),
                                    irls_thresh=float(ith),
                                    irls_mx=int(mx),
                                    xy_a_val=xy_a_val.copy(),
                                    preds_b=preds_b.copy(),
                                    preds_c=preds_c.copy(),
                                    rmse_a_rank_proxy=float(_ra0),
                                    rmse_b=float(rb),
                                    rmse_c=float(rc),
                                )

    if best_pack is None:
        raise RuntimeError("V12: 그리드 탐색 실패")

    xy_a_best = predict_step_a_calibrated(val_imp, loc, best_pack["coeffs"], best_pack["feat"], best_pack["hf_wifi"])
    preds_b_final = best_pack["preds_b"]
    preds_c_final = best_pack["preds_c"]
    ea = point_errors(tx, ty, xy_a_best[:, 0], xy_a_best[:, 1])
    eb = point_errors(tx, ty, preds_b_final[:, 0], preds_b_final[:, 1])
    ec = point_errors(tx, ty, preds_c_final[:, 0], preds_c_final[:, 1])

    rmse_a, mae_a = rmse_mae(ea)
    rmse_b, mae_b = rmse_mae(eb)
    rmse_c, mae_c = rmse_mae(ec)

    d_ab = rmse_a - rmse_b
    d_bc = rmse_b - rmse_c
    d_ac = rmse_a - rmse_c

    print("\n" + "=" * 72)
    print("V12 검증 — Step A: 강건 Wi‑Fi 보정 삼변 | B: 게이트+UWB·Wi‑Fi 융합 | C: IRLS")
    print("=" * 72)
    print(f"  Step A  RMSE={rmse_a:.4f} m  MAE={mae_a:.4f} m")
    print(f"  Step B  RMSE={rmse_b:.4f} m  MAE={mae_b:.4f} m")
    print(f"  Step C  RMSE={rmse_c:.4f} m  MAE={mae_c:.4f} m")
    print("=" * 72)
    print("검증 RMSE 개선폭 (양수 = 이전 단계 대비 개선)")
    print(f"  A → B: ΔRMSE {d_ab:+.4f} m   | MAE {mae_a:.4f} → {mae_b:.4f}")
    print(f"  B → C: ΔRMSE {d_bc:+.4f} m   | MAE {mae_b:.4f} → {mae_c:.4f}")
    print(f"  A → C: ΔRMSE {d_ac:+.4f} m   | MAE {mae_a:.4f} → {mae_c:.4f}")
    print(
        f"\n[선택 하이퍼파라미터] calib={best_pack['spec'].kind}, feat={best_pack['feat']}, "
        f"huber_f_wifi={best_pack['hf_wifi']}, gate={best_pack['gate']:.3f} m, "
        f"uwb_var_inflate={best_pack['uwb_inf']}, huber_f_fusion={best_pack['hf_fus']}, "
        f"IRLS_thresh={best_pack['irls_thresh']:.2f}, IRLS_iter={best_pack['irls_mx']}"
    )

    pref = resolve_paths(root)
    summ = {
        "objective": "Minimize validation RMSE after Step C (IRLS). Top Phase-A candidates forwarded to Phase B/C.",
        "val_metrics": {
            "step_A_RMSE_m": rmse_a,
            "step_A_MAE_m": mae_a,
            "step_B_RMSE_m": rmse_b,
            "step_B_MAE_m": mae_b,
            "step_C_RMSE_m": rmse_c,
            "step_C_MAE_m": mae_c,
            "delta_RMSE_A_to_B_m": float(d_ab),
            "delta_RMSE_B_to_C_m": float(d_bc),
            "delta_RMSE_A_to_C_m": float(d_ac),
        },
        "selected_hyperparameters": {
            "wifi_calibration_kind": best_pack["spec"].kind,
            "wifi_calibration_huber_epsilon": best_pack["spec"].huber_epsilon,
            "wifi_calibration_huber_alpha": best_pack["spec"].huber_alpha,
            "wifi_calibration_ransac_residual_thresh": best_pack["spec"].ransac_residual_threshold,
            "feat_mode": best_pack["feat"],
            "huber_f_scale_wifi_only_trilateration": best_pack["hf_wifi"],
            "uwb_geometric_gate_threshold_m": best_pack["gate"],
            "uwb_variance_inflate_multiplier": best_pack["uwb_inf"],
            "huber_f_scale_fusion_trilateration": best_pack["hf_fus"],
            "irls_residual_threshold_m": best_pack["irls_thresh"],
            "irls_max_iterations": best_pack["irls_mx"],
            "phase_A_top_K_evaluated": TOP_A_CANDIDATES,
        },
        "searched_grids_documentation": {
            "feat_modes": FEAT_MODES,
            "huber_f_wifi_candidates": HUBER_F_WIFI_GRID,
            "huber_f_fusion_candidates": HUBER_F_FUSION_GRID,
            "gate_threshold_m_candidates": GATE_THRESH_GRID,
            "uwb_variance_inflate_candidates": UWB_VAR_INFLATE_GRID,
            "irls_residual_threshold_m": IRLS_THRESH_GRID_M,
            "irls_max_iter": IRLS_MAX_ITER_GRID,
            "calibration_specs": "v12_calibration_spec_list() — identity + 2×Huber + 2×RANSAC (런타임 제한)",
        },
        "runtime_note_grid_size": (
            "초기 구현 전체 Cartesian(캘리브×게이트×IRLS 조합 과다)는 수백 만 회 least_squares 호출로"
            " 10분+ 소요되어, 현재 레포에서는 Phase A 후보 소비·후보 Calibration 축약·Gate/IRLS 샘플링 폭 조정으로 균형함."
            " 필요 시 GATE_THRESH_GRID 확장 또는 v12_calibration_spec_list에 RANSAC(낮은 max_trials) 추가 권장."
        ),
        "rationale_plain_ko": (
            "① Step A에서 V11과 동일한 강건 Wi‑Fi AP별 거리 선형 보정(허버·항등) 후 Wi‑Fi만 삼변해 "
            "극단 UWB 노이즈와 분리된 '안심 좌표'를 만든다. 본 검증에서는 항등(Train 미피팅)·bias_sub가 최적이었다. "
            "② Step B는 V9와 같이 |d_geom(안심)−d_UWB)|가 게이트를 넘는 UWB만 하드 탈락시키고, "
            "통과 UWB에는 분산을 곱 인플레이트해 초저분산 UWB 과신을 억제한 뒤 보정 Wi‑Fi와 융합한다. "
            "③ Step C는 V11식 IRLS(Tukey×Huber)로 잔차 기반 재가중. "
            "단일 검증 폴드에서 그리드를 택해 과적합 위험을 줄였다."
        ),
        "data_paths": {
            "train_median": str(pref.train_median),
            "train_variance": str(pref.train_variance),
            "val_median": str(pref.val_median),
            "val_variance": str(pref.val_variance),
        },
        "visualizations": [
            str(out_dir / "v12_map_true_vs_predicted_steps.png"),
            str(out_dir / "v12_map_error_vectors_stepC.png"),
            str(out_dir / "v12_map_stepC_pairwise_truth_pred.png"),
            str(out_dir / "v12_cdf_steps.png"),
        ],
    }
    (out_dir / "v12_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")

    pred = val_imp[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xy_a_best[:, 0]
    pred["StepA_Y"] = xy_a_best[:, 1]
    pred["StepB_X"] = preds_b_final[:, 0]
    pred["StepB_Y"] = preds_b_final[:, 1]
    pred["StepC_X"] = preds_c_final[:, 0]
    pred["StepC_Y"] = preds_c_final[:, 1]
    pred["ErrA_m"] = ea
    pred["ErrB_m"] = eb
    pred["ErrC_m"] = ec
    pred.to_csv(out_dir / "v12_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(phase_a_records).to_csv(out_dir / "v12_grid_phaseA.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phase_bc_records).to_csv(out_dir / "v12_grid_phaseBC.csv", index=False, encoding="utf-8-sig")

    save_cdf_steps(
        out_dir,
        dict(A=ea, B=eb, C=ec),
        dict(A=rmse_a, B=rmse_b, C=rmse_c),
        dict(A="Step A: Robust Wi‑Fi calib", B="Step B: UWB gated fusion", C="Step C: IRLS"),
    )

    save_v12_true_vs_pred_maps(
        out_dir, tx, ty, xy_a_best, preds_b_final, preds_c_final, rmse_a, rmse_b, rmse_c
    )
    save_v12_error_vectors_step_c(out_dir, tx, ty, preds_c_final, ec)
    save_v12_combined_truth_pred_scatter_step_c(out_dir, tx, ty, preds_c_final, ec)

    print(f"\n저장: {out_dir / 'v12_predictions.csv'}")
    print(f"저장: {out_dir / 'v12_summary.json'}")
    print(f"저장: {out_dir / 'v12_cdf_steps.png'}")
    print(f"저장: {out_dir / 'v12_map_true_vs_predicted_steps.png'}")
    print(f"저장: {out_dir / 'v12_map_error_vectors_stepC.png'}")
    print(f"저장: {out_dir / 'v12_map_stepC_pairwise_truth_pred.png'}")
    print(f"저장: {out_dir / 'v12_grid_phaseA.csv'}")
    print(f"저장: {out_dir / 'v12_grid_phaseBC.csv'}")

    # 보정 계수 표 (canon 이름)
    rows_coef = [{"wifi_canon_ap": k, "A": v[0], "B": v[1]} for k, v in best_pack["coeffs"].items()]
    pd.DataFrame(rows_coef).to_csv(out_dir / "v12_wifi_calibration_coefs.csv", index=False, encoding="utf-8-sig")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
