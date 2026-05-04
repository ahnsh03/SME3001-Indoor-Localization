"""
V12 Turbo — `indoor_fusion_pipeline_v12_strict.py`와 동일한 **데이터 무결성·K-Fold CV 선택 규약**을 유지하면서,
Optuna 탐색·NumPy/Numba 벡터화·Trial 병렬화로 런타임을 단축한다.

- Phase A/B/C 수식: V12 Strict와 동일(게이트, UWB 분산 inflate, Wi‑Fi 보정, Tukey×Huber IRLS, 바운드 삼변).
- Step B/C 삼변: 바운드 Huber 가중 잔차에 대한 가우스‑뉴턴 + Huber IRLS를 Numba로 구현(TRILAT_GN_ITERS).

실행 옵션
---------
- ``--no-plots``: CDF/맵 PNG 생략(수 초 절감).
- 환경변수 ``V12_TURBO_OPTUNA_JOBS``: Optuna ``n_jobs`` 상한(기본 ``-1``). Windows에서 피클링 부담이 크면 ``8`` 등 권장.

10초 미만은 **sklearn 강건 Wi‑Fi 보정**이 폴드×상위 A랭크마다 수행되는 한 하드웨어·데이터 크기에 강하게 의존한다.

그리드·관례 (방법론)
-------------------
Phase A 후보는 ``indoor_fusion_pipeline_v12`` 의 ``FEAT_MODES`` × ``v12_calibration_spec_list()`` × ``HUBER_F_WIFI_GRID`` 카테시언이며,
그중 상위 ``TOP_A_CANDIDATES`` 만 캐시에 넣는다 — 이 목록 자체는 **이전 파이프라인·실습에서 쓰기 좋은 범위(관례)** 로 고정된 성격이 강하다.

Optuna가 연속·범위로 고르는 **게이트·UWB inflate·fusion Huber·IRLS 임계** 역시 플러시블한 구간으로 **설계 고정**이며,
검증 ``True_*`` 는 **최종 지표 출력 1회**에만 사용한다(선택 무결성은 요약 JSON ``integrity`` 참고).

리포트에는 V13-fix 와 같은 취지로,
「Train 분석·엔진 관례에 따른 **고정 그리드** 안에서 Train CV 목적값만 최소화했다」고 서술하는 것이 타당하다.
"""

from __future__ import annotations

import json
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from sklearn.model_selection import KFold

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as e:  # pragma: no cover
    raise ImportError("V12 Turbo에는 optuna가 필요합니다: py -3 -m pip install optuna") from e

from fusion_realtime_sanitize import (
    VAR_CAP_UWB_FUSION,
    VAR_CAP_WIFI_FUSION,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
)
from indoor_fusion_pipeline_v10 import WifiOnlyTrilateration, resolve_paths
from indoor_fusion_pipeline_v11 import CalibMethodSpec, fit_robust_calibration_per_ap
from indoor_fusion_pipeline_v12 import (
    FEAT_MODES,
    HUBER_F_WIFI_GRID,
    TOP_A_CANDIDATES,
    TUKEY_C,
    augment_train_for_canon_fit,
    predict_step_a_v3_wifi_equivalent,
    point_errors,
    rmse_mae,
    v12_calibration_spec_list,
    wifi_median_impute_fusion,
)
from indoor_fusion_pipeline_v8 import (
    BOUNDS_HI,
    BOUNDS_LO,
    EPS_W,
    MAX_RANGE_M,
    ROBUST_BIAS_M,
    UWB_BIAS_W,
    WIFI_BIAS_W,
    WIFI_COL_CANON,
    Config,
    FusionLocalizerV8,
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

ART_PREFIX = "v12_turbo"

# Optuna (30~50 권장)
N_OPTUNA_TRIALS = 35
# 요청 사양: Trial 병렬. Windows에서 과도한 프로세스 생성을 피하려면 환경변수 V12_TURBO_OPTUNA_JOBS=8 등으로 상한 설정.
OPTUNA_N_JOBS = int(os.environ.get("V12_TURBO_OPTUNA_JOBS", "-1"))
OPTUNA_SEED = 42

# Numba 삼변 GN 반복(바운드 Huber)
TRILAT_GN_ITERS = 22

# 플롯 DPI (Strict와 동일 시각 품질이 필요하면 200)
PLOT_DPI = 110


def resolve_kfold_n_splits(n_samples: int) -> int:
    if n_samples < 2:
        raise ValueError("Train 샘플이 2개 미만이면 K-Fold를 할 수 없습니다.")
    if n_samples < 6:
        return 2
    return int(min(5, max(3, n_samples // 35)))


MAX_KFOLD_SPLITS = 3


@dataclass
class LocGeomPack:
    """Worker 피클링용: DataFrame/locator 없이 기하·바이어스만 전달."""

    n_u: int
    n_w: int
    uwb_xy: np.ndarray  # (n_u, 2)
    wifi_xy: np.ndarray  # (n_w, 2)
    uwb_robust_bias: np.ndarray  # (n_u,)
    wifi_robust_bias: np.ndarray  # (n_w,)
    var_cap_uwb: float
    var_cap_wifi: float
    eps_w: float
    uwb_bias_w: float
    wifi_bias_w: float
    bounds_lo: np.ndarray  # (2,)
    bounds_hi: np.ndarray  # (2,)


def build_loc_geom_pack(loc: FusionLocalizerV8) -> LocGeomPack:
    nu, nw = len(loc.uwb_cols), len(loc.wifi_cols)
    uwb_xy = np.zeros((nu, 2), dtype=np.float64)
    wifi_xy = np.zeros((nw, 2), dtype=np.float64)
    uwb_rb = np.zeros(nu, dtype=np.float64)
    wifi_rb = np.zeros(nw, dtype=np.float64)
    for i, c in enumerate(loc.uwb_cols):
        ax, ay = loc._resolve_sensor_position_m(c, i, True)
        uwb_xy[i, 0], uwb_xy[i, 1] = ax, ay
        uwb_rb[i] = float(ROBUST_BIAS_M[sensor_canon_key(c, i, True)])
    for i, c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        ax, ay = loc._resolve_sensor_position_m(c, i, False)
        wifi_xy[i, 0], wifi_xy[i, 1] = ax, ay
        wifi_rb[i] = float(ROBUST_BIAS_M[sensor_canon_key(canon, i, False)])
    return LocGeomPack(
        n_u=nu,
        n_w=nw,
        uwb_xy=uwb_xy,
        wifi_xy=wifi_xy,
        uwb_robust_bias=uwb_rb,
        wifi_robust_bias=wifi_rb,
        var_cap_uwb=float(VAR_CAP_UWB_FUSION),
        var_cap_wifi=float(VAR_CAP_WIFI_FUSION),
        eps_w=float(EPS_W),
        uwb_bias_w=float(UWB_BIAS_W),
        wifi_bias_w=float(WIFI_BIAS_W),
        bounds_lo=BOUNDS_LO.astype(np.float64),
        bounds_hi=BOUNDS_HI.astype(np.float64),
    )


def coeffs_dict_to_ab(coeffs: Dict[str, Tuple[float, float]], loc: FusionLocalizerV8) -> np.ndarray:
    ab = np.zeros((len(loc.wifi_cols), 2), dtype=np.float64)
    for i, _c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        a, b = coeffs[canon]
        ab[i, 0], ab[i, 1] = float(a), float(b)
    return ab


def feat_to_id(feat: str) -> int:
    return 0 if feat == "bias_sub" else 1


def te_df_to_numpy_blocks(te_df: pd.DataFrame, loc: FusionLocalizerV8) -> Dict[str, np.ndarray]:
    nu, nw = len(loc.uwb_cols), len(loc.wifi_cols)
    u_med = np.zeros((len(te_df), nu), dtype=np.float64)
    u_var = np.full((len(te_df), nu), np.nan, dtype=np.float64)
    u_nan = np.ones((len(te_df), nu), dtype=np.uint8)
    w_med = np.zeros((len(te_df), nw), dtype=np.float64)
    w_var = np.full((len(te_df), nw), np.nan, dtype=np.float64)
    for j, c in enumerate(loc.uwb_cols):
        mc, vc = f"med_{c}", f"var_{c}"
        on = f"orig_nan_{c}"
        if mc in te_df.columns:
            u_med[:, j] = te_df[mc].values.astype(np.float64)
        if vc in te_df.columns:
            u_var[:, j] = te_df[vc].values.astype(np.float64)
        if on in te_df.columns:
            u_nan[:, j] = (te_df[on].values == 1).astype(np.uint8)
        else:
            u_nan[:, j] = 1
    for j, c in enumerate(loc.wifi_cols):
        mc, vc = f"med_{c}", f"var_{c}"
        if mc in te_df.columns:
            w_med[:, j] = te_df[mc].values.astype(np.float64)
        if vc in te_df.columns:
            w_var[:, j] = te_df[vc].values.astype(np.float64)
    return dict(
        true_x=te_df["True_X"].values.astype(np.float64),
        true_y=te_df["True_Y"].values.astype(np.float64),
        uwb_med=u_med,
        uwb_var=u_var,
        uwb_was_nan=u_nan,
        wifi_med=w_med,
        wifi_var=w_var,
    )


def _build_one_fold_sequential_ranks(
    fold_index: int,
    train_imp: pd.DataFrame,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    top_a: List[Tuple[float, CalibMethodSpec, str, float]],
) -> Tuple[int, Dict[str, Any]]:
    """폴드 단위로 독립 실행 → ThreadPool로 폴드 병렬화 시 sklearn GIL 완화 효과."""
    tr_df = train_imp.iloc[tr_idx].reset_index(drop=True)
    te_df = train_imp.iloc[te_idx].reset_index(drop=True)
    tr_fit = augment_train_for_canon_fit(tr_df, loc)
    xy_v3_te = predict_step_a_v3_wifi_equivalent(te_df, loc)
    per_rank: List[Dict[str, Any]] = []
    for (_cv_ra, spec, feat, hf_wifi) in top_a:
        coeffs = fit_robust_calibration_per_ap(tr_fit, tri_ref, feat_mode=feat, spec=spec)
        per_rank.append(dict(coeffs=coeffs, xy_a=xy_v3_te, feat=feat, hf_wifi=float(hf_wifi)))
    return fold_index, dict(te_df=te_df, per_rank=per_rank)


def build_phase_bc_fold_cache(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    top_a: List[Tuple[float, CalibMethodSpec, str, float]],
    n_splits: int,
) -> List[Dict[str, Any]]:
    n_splits = min(n_splits, len(train_imp))
    if n_splits < 2:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(np.arange(len(train_imp))))
    fold_slots: List[Optional[Dict[str, Any]]] = [None] * n_splits
    with ThreadPoolExecutor(max_workers=min(n_splits, 4)) as ex:
        futs = [
            ex.submit(_build_one_fold_sequential_ranks, fi, train_imp, tr_idx, te_idx, loc, tri_ref, top_a)
            for fi, (tr_idx, te_idx) in enumerate(splits)
        ]
        for fut in as_completed(futs):
            fi, block = fut.result()
            fold_slots[fi] = block
    return [fc for fc in fold_slots if fc is not None]


def fold_cache_to_numpy_only(
    fold_cache: List[Dict[str, Any]], loc: FusionLocalizerV8
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fc in fold_cache:
        te_df: pd.DataFrame = fc["te_df"]
        blk0 = fc["per_rank"][0]
        xy_a = np.asarray(blk0["xy_a"], dtype=np.float64).copy()
        blocks = te_df_to_numpy_blocks(te_df, loc)
        per_np: List[Dict[str, Any]] = []
        for blk in fc["per_rank"]:
            per_np.append(
                dict(
                    coeff_ab=coeffs_dict_to_ab(blk["coeffs"], loc),
                    feat_id=int(feat_to_id(str(blk["feat"]))),
                    hf_wifi=np.float64(blk["hf_wifi"]),
                )
            )
        out.append(
            dict(
                n_te=int(len(te_df)),
                true_x=blocks["true_x"],
                true_y=blocks["true_y"],
                xy_a=xy_a,
                uwb_med=blocks["uwb_med"],
                uwb_var=blocks["uwb_var"],
                uwb_was_nan=blocks["uwb_was_nan"],
                wifi_med=blocks["wifi_med"],
                wifi_var=blocks["wifi_var"],
                per_rank=per_np,
            )
        )
    return out


# --- Numba: Tukey / Huber / 바운드 Huber 삼변 / IRLS / 배치 Step C ---


@njit(cache=True, fastmath=True)
def tukey_bisquare_weights_njit(u: np.ndarray, c: float) -> np.ndarray:
    n = u.shape[0]
    w = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = abs(u[i])
        if a < c:
            t = u[i] / c
            w[i] = (1.0 - t * t) ** 2
    return w


@njit(cache=True, fastmath=True)
def huber_hinge_weights_njit(r: np.ndarray, delta: float) -> np.ndarray:
    n = r.shape[0]
    d = delta if delta > 1e-6 else 1e-6
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = abs(r[i])
        den = a if a > 1e-6 else 1e-6
        t = d / den
        out[i] = 1.0 if t > 1.0 else t
    return out


@njit(cache=True)
def solve_trilat_bounded_huber_njit(
    pos: np.ndarray,
    dist: np.ndarray,
    weights: np.ndarray,
    huber_f_scale: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    gn_iters: int,
) -> np.ndarray:
    """가중 잔차에 Huber IRLS를 곱한 가우스‑뉴턴(2D), 바운드 클리핑. V12 `solve_trilat_array`와 동일 목적함수."""
    n = dist.shape[0]
    out = np.empty(2, dtype=np.float64)
    if n == 0:
        out[0], out[1] = np.nan, np.nan
        return out
    if n < 3:
        sw = 0.0
        sx, sy = 0.0, 0.0
        for i in range(n):
            di = dist[i] if dist[i] > 1e-3 else 1e-3
            wi = weights[i] if weights[i] > 0.0 else 0.0
            if wi < 1e-12:
                continue
            sw += wi
            sx += pos[i, 0] * wi
            sy += pos[i, 1] * wi
        if sw < 1e-12:
            out[0], out[1] = np.nan, np.nan
            return out
        x = sx / sw
        y = sy / sw
        out[0] = x if x > bounds_lo[0] else bounds_lo[0]
        if out[0] > bounds_hi[0]:
            out[0] = bounds_hi[0]
        out[1] = y if y > bounds_lo[1] else bounds_lo[1]
        if out[1] > bounds_hi[1]:
            out[1] = bounds_hi[1]
        return out

    idx = 0
    best = dist[0]
    for i in range(1, n):
        if dist[i] < best:
            best = dist[i]
            idx = i
    x = pos[idx, 0]
    y = pos[idx, 1]
    if x < bounds_lo[0]:
        x = bounds_lo[0]
    if x > bounds_hi[0]:
        x = bounds_hi[0]
    if y < bounds_lo[1]:
        y = bounds_lo[1]
    if y > bounds_hi[1]:
        y = bounds_hi[1]

    delta = huber_f_scale
    for _ in range(gn_iters):
        JtJ00, JtJ01, JtJ11 = 0.0, 0.0, 0.0
        Jtr0, Jtr1 = 0.0, 0.0
        for i in range(n):
            dx = x - pos[i, 0]
            dy = y - pos[i, 1]
            ri0 = np.sqrt(dx * dx + dy * dy)
            sw = np.sqrt(weights[i] if weights[i] > 0.0 else 0.0)
            raw = sw * (ri0 - dist[i])
            ar = abs(raw)
            w_h = 1.0 if ar <= delta else (delta / (ar if ar > 1e-9 else 1e-9))
            rw = raw * np.sqrt(w_h)
            if ri0 < 1e-12:
                j0, j1 = 0.0, 0.0
            else:
                j0 = sw * np.sqrt(w_h) * dx / ri0
                j1 = sw * np.sqrt(w_h) * dy / ri0
            JtJ00 += j0 * j0
            JtJ01 += j0 * j1
            JtJ11 += j1 * j1
            Jtr0 += j0 * rw
            Jtr1 += j1 * rw
        det = JtJ00 * JtJ11 - JtJ01 * JtJ01
        adet = det if det >= 0.0 else -det
        if adet < 1e-18:
            break
        dx0 = (-Jtr0 * JtJ11 + Jtr1 * JtJ01) / det
        dx1 = (Jtr0 * JtJ01 - Jtr1 * JtJ00) / det
        x = x + dx0
        y = y + dx1
        if x < bounds_lo[0]:
            x = bounds_lo[0]
        if x > bounds_hi[0]:
            x = bounds_hi[0]
        if y < bounds_lo[1]:
            y = bounds_lo[1]
        if y > bounds_hi[1]:
            y = bounds_hi[1]
    out[0], out[1] = x, y
    return out


@njit(cache=True)
def irls_refine_xyz_njit(
    pos: np.ndarray,
    dist: np.ndarray,
    w_base: np.ndarray,
    huber_f_fusion: float,
    thresh_m: float,
    max_iters: int,
    tukey_c: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    gn_iters: int,
) -> np.ndarray:
    n = dist.shape[0]
    if n < 3:
        return solve_trilat_bounded_huber_njit(pos, dist, w_base, huber_f_fusion, bounds_lo, bounds_hi, gn_iters)
    w = w_base.copy()
    xy_last = np.empty(2, dtype=np.float64)
    xy_last[0], xy_last[1] = np.nan, np.nan
    itlim = max_iters if max_iters > 0 else 1
    for _ in range(itlim):
        xy_last = solve_trilat_bounded_huber_njit(pos, dist, w, huber_f_fusion, bounds_lo, bounds_hi, gn_iters)
        if not (np.isfinite(xy_last[0]) and np.isfinite(xy_last[1])):
            break
        r = np.empty(n, dtype=np.float64)
        for i in range(n):
            dx = xy_last[0] - pos[i, 0]
            dy = xy_last[1] - pos[i, 1]
            gpred = np.sqrt(dx * dx + dy * dy)
            t = gpred - dist[i]
            r[i] = t if t >= 0.0 else -t
        med = np.median(r)
        ad = np.empty(n, dtype=np.float64)
        for i in range(n):
            ad[i] = abs(r[i] - med)
        mad = np.median(ad)
        sigma_rob = mad * 1.4826
        if sigma_rob < 0.08:
            sigma_rob = 0.08
        thr = thresh_m * 0.65
        sigma = thr if thr > sigma_rob else sigma_rob
        if sigma < 1e-2:
            sigma = 1e-2
        u = np.empty(n, dtype=np.float64)
        for i in range(n):
            u[i] = r[i] / sigma
        tw = tukey_bisquare_weights_njit(u, tukey_c)
        hw = huber_hinge_weights_njit(r, thresh_m)
        w_new = np.empty(n, dtype=np.float64)
        for i in range(n):
            m = tw[i] * hw[i]
            if m < 0.04:
                m = 0.04
            if m > 1.0:
                m = 1.0
            w_new[i] = w_base[i] * m
        dw = 0.0
        wnorm = 0.0
        for i in range(n):
            t = w_new[i] - w[i]
            dw += t * t
            wnorm += w[i] * w[i]
        swn = np.sqrt(wnorm if wnorm > 0.0 else 0.0) + 1e-9
        if swn < 1e-12:
            break
        rat = np.sqrt(dw if dw > 0.0 else 0.0) / swn
        if rat < 1e-4:
            for i in range(n):
                w[i] = w_new[i]
            xy_last = solve_trilat_bounded_huber_njit(pos, dist, w, huber_f_fusion, bounds_lo, bounds_hi, gn_iters)
            break
        for i in range(n):
            w[i] = w_new[i]
    return xy_last


@njit(cache=True, parallel=False)
def batch_step_c_preds_flat(
    n: int,
    xy_guide: np.ndarray,
    uwb_med: np.ndarray,
    uwb_var: np.ndarray,
    uwb_was_nan: np.ndarray,
    wifi_med: np.ndarray,
    wifi_var: np.ndarray,
    coeff_ab: np.ndarray,
    feat_id: int,
    gate_t: float,
    uwb_var_inflate: float,
    hf_fusion: float,
    hf_wifi: float,
    irls_thresh: float,
    irls_mx: int,
    tukey_c: float,
    nu: int,
    nw: int,
    uwb_xy: np.ndarray,
    wifi_xy: np.ndarray,
    uwb_rb: np.ndarray,
    wifi_rb: np.ndarray,
    var_cap_uwb: float,
    var_cap_wifi: float,
    eps_w: float,
    uwb_bias_w: float,
    wifi_bias_w: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    gn_iters: int,
    max_range_m: float,
) -> np.ndarray:
    preds = np.full((n, 2), np.nan, dtype=np.float64)
    max_pts = nu + nw
    for k in range(n):
        gx = xy_guide[k, 0]
        gy = xy_guide[k, 1]
        pos_b = np.zeros((max_pts, 2), dtype=np.float64)
        dist_b = np.zeros(max_pts, dtype=np.float64)
        w_b = np.zeros(max_pts, dtype=np.float64)
        ct = 0
        if np.isfinite(gx) and np.isfinite(gy):
            for j in range(nu):
                if uwb_was_nan[k, j] != 0:
                    continue
                dr = uwb_med[k, j]
                vv = uwb_var[k, j]
                if (not np.isfinite(dr)) or (not np.isfinite(vv)):
                    continue
                if vv > var_cap_uwb:
                    continue
                d_meas = dr - uwb_rb[j]
                ax = uwb_xy[j, 0]
                ay = uwb_xy[j, 1]
                d_geom = np.sqrt((gx - ax) ** 2 + (gy - ay) ** 2)
                diff = d_geom - d_meas
                ad = diff if diff >= 0.0 else -diff
                if ad > gate_t:
                    continue
                infm = uwb_var_inflate if uwb_var_inflate > 1.0 else 1.0
                eff_var = vv * infm
                wi = 1.0 / (eff_var + uwb_bias_w + eps_w)
                pos_b[ct, 0] = ax
                pos_b[ct, 1] = ay
                dist_b[ct] = d_meas
                w_b[ct] = wi
                ct += 1
        for j in range(nw):
            m_raw = wifi_med[k, j]
            vv = wifi_var[k, j]
            if (not np.isfinite(m_raw)) or (not np.isfinite(vv)):
                continue
            if vv > var_cap_wifi:
                continue
            if m_raw > max_range_m:
                continue
            A = coeff_ab[j, 0]
            B = coeff_ab[j, 1]
            if feat_id == 0:
                xf = m_raw - wifi_rb[j]
            else:
                xf = m_raw
            d_c = A * xf + B
            if d_c < 0.1:
                d_c = 0.1
            wi2 = 1.0 / (vv + wifi_bias_w + eps_w)
            pos_b[ct, 0] = wifi_xy[j, 0]
            pos_b[ct, 1] = wifi_xy[j, 1]
            dist_b[ct] = d_c
            w_b[ct] = wi2
            ct += 1

        wpos = np.zeros((nw, 2), dtype=np.float64)
        wdist = np.zeros(nw, dtype=np.float64)
        wwei = np.zeros(nw, dtype=np.float64)
        cw = 0
        for j in range(nw):
            m_raw = wifi_med[k, j]
            vv = wifi_var[k, j]
            if (not np.isfinite(m_raw)) or (not np.isfinite(vv)):
                continue
            if vv > var_cap_wifi:
                continue
            if m_raw > max_range_m:
                continue
            A = coeff_ab[j, 0]
            B = coeff_ab[j, 1]
            if feat_id == 0:
                xf = m_raw - wifi_rb[j]
            else:
                xf = m_raw
            d_c = A * xf + B
            if d_c < 0.1:
                d_c = 0.1
            wi2 = 1.0 / (vv + wifi_bias_w + eps_w)
            wpos[cw, 0] = wifi_xy[j, 0]
            wpos[cw, 1] = wifi_xy[j, 1]
            wdist[cw] = d_c
            wwei[cw] = wi2
            cw += 1

        if ct < 3:
            if cw == 0:
                preds[k, 0] = np.nan
                preds[k, 1] = np.nan
            else:
                p2 = wpos[:cw, :].copy()
                d2 = wdist[:cw].copy()
                w2 = wwei[:cw].copy()
                xyw = solve_trilat_bounded_huber_njit(p2, d2, w2, hf_wifi, bounds_lo, bounds_hi, gn_iters)
                preds[k, 0] = xyw[0]
                preds[k, 1] = xyw[1]
        else:
            puse = pos_b[:ct, :].copy()
            duse = dist_b[:ct].copy()
            wuse = w_b[:ct].copy()
            xyf = irls_refine_xyz_njit(
                puse,
                duse,
                wuse,
                hf_fusion,
                irls_thresh,
                irls_mx,
                tukey_c,
                bounds_lo,
                bounds_hi,
                gn_iters,
            )
            preds[k, 0] = xyf[0]
            preds[k, 1] = xyf[1]
    return preds


@njit(cache=True, parallel=False)
def batch_step_b_preds_flat(
    n: int,
    xy_guide: np.ndarray,
    uwb_med: np.ndarray,
    uwb_var: np.ndarray,
    uwb_was_nan: np.ndarray,
    wifi_med: np.ndarray,
    wifi_var: np.ndarray,
    coeff_ab: np.ndarray,
    feat_id: int,
    gate_t: float,
    uwb_var_inflate: float,
    hf_fusion: float,
    nu: int,
    nw: int,
    uwb_xy: np.ndarray,
    wifi_xy: np.ndarray,
    uwb_rb: np.ndarray,
    wifi_rb: np.ndarray,
    var_cap_uwb: float,
    var_cap_wifi: float,
    eps_w: float,
    uwb_bias_w: float,
    wifi_bias_w: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    gn_iters: int,
    max_range_m: float,
) -> np.ndarray:
    preds = np.full((n, 2), np.nan, dtype=np.float64)
    max_pts = nu + nw
    for k in range(n):
        gx = xy_guide[k, 0]
        gy = xy_guide[k, 1]
        pos_b = np.zeros((max_pts, 2), dtype=np.float64)
        dist_b = np.zeros(max_pts, dtype=np.float64)
        w_b = np.zeros(max_pts, dtype=np.float64)
        ct = 0
        if np.isfinite(gx) and np.isfinite(gy):
            for j in range(nu):
                if uwb_was_nan[k, j] != 0:
                    continue
                dr = uwb_med[k, j]
                vv = uwb_var[k, j]
                if (not np.isfinite(dr)) or (not np.isfinite(vv)):
                    continue
                if vv > var_cap_uwb:
                    continue
                d_meas = dr - uwb_rb[j]
                ax = uwb_xy[j, 0]
                ay = uwb_xy[j, 1]
                d_geom = np.sqrt((gx - ax) ** 2 + (gy - ay) ** 2)
                diff = d_geom - d_meas
                ad = diff if diff >= 0.0 else -diff
                if ad > gate_t:
                    continue
                infm = uwb_var_inflate if uwb_var_inflate > 1.0 else 1.0
                eff_var = vv * infm
                wi = 1.0 / (eff_var + uwb_bias_w + eps_w)
                pos_b[ct, 0] = ax
                pos_b[ct, 1] = ay
                dist_b[ct] = d_meas
                w_b[ct] = wi
                ct += 1
        for j in range(nw):
            m_raw = wifi_med[k, j]
            vv = wifi_var[k, j]
            if (not np.isfinite(m_raw)) or (not np.isfinite(vv)):
                continue
            if vv > var_cap_wifi:
                continue
            if m_raw > max_range_m:
                continue
            A = coeff_ab[j, 0]
            B = coeff_ab[j, 1]
            if feat_id == 0:
                xf = m_raw - wifi_rb[j]
            else:
                xf = m_raw
            d_c = A * xf + B
            if d_c < 0.1:
                d_c = 0.1
            wi2 = 1.0 / (vv + wifi_bias_w + eps_w)
            pos_b[ct, 0] = wifi_xy[j, 0]
            pos_b[ct, 1] = wifi_xy[j, 1]
            dist_b[ct] = d_c
            w_b[ct] = wi2
            ct += 1
        if ct == 0:
            preds[k, 0] = np.nan
            preds[k, 1] = np.nan
        else:
            puse = pos_b[:ct, :].copy()
            duse = dist_b[:ct].copy()
            wuse = w_b[:ct].copy()
            xyf = solve_trilat_bounded_huber_njit(puse, duse, wuse, hf_fusion, bounds_lo, bounds_hi, gn_iters)
            preds[k, 0] = xyf[0]
            preds[k, 1] = xyf[1]
    return preds


def mean_cv_rmse_step_c_np(
    fold_np: List[Dict[str, Any]],
    lg: LocGeomPack,
    rank: int,
    gate: float,
    uvb_inf: float,
    hf_fus: float,
    ith: float,
    mx: int,
) -> float:
    rmses: List[float] = []
    for fc in fold_np:
        pr = fc["per_rank"][rank]
        preds = batch_step_c_preds_flat(
            fc["n_te"],
            fc["xy_a"],
            fc["uwb_med"],
            fc["uwb_var"],
            fc["uwb_was_nan"],
            fc["wifi_med"],
            fc["wifi_var"],
            pr["coeff_ab"],
            pr["feat_id"],
            float(gate),
            float(uvb_inf),
            float(hf_fus),
            float(pr["hf_wifi"]),
            float(ith),
            int(mx),
            float(TUKEY_C),
            lg.n_u,
            lg.n_w,
            lg.uwb_xy,
            lg.wifi_xy,
            lg.uwb_robust_bias,
            lg.wifi_robust_bias,
            lg.var_cap_uwb,
            lg.var_cap_wifi,
            lg.eps_w,
            lg.uwb_bias_w,
            lg.wifi_bias_w,
            lg.bounds_lo,
            lg.bounds_hi,
            TRILAT_GN_ITERS,
            float(MAX_RANGE_M),
        )
        ec = point_errors(fc["true_x"], fc["true_y"], preds[:, 0], preds[:, 1])
        rc, _ = rmse_mae(ec)
        rmses.append(float(rc))
    return float(np.mean(rmses))


def _warmup_numba(lg: LocGeomPack) -> None:
    n = 3
    fake = dict(
        n_te=n,
        true_x=np.zeros(n),
        true_y=np.zeros(n),
        xy_a=np.zeros((n, 2)),
        uwb_med=np.zeros((n, lg.n_u)),
        uwb_var=np.ones((n, lg.n_u)),
        uwb_was_nan=np.zeros((n, lg.n_u), dtype=np.uint8),
        wifi_med=np.ones((n, lg.n_w)) * 5.0,
        wifi_var=np.ones((n, lg.n_w)),
        per_rank=[
            dict(
                coeff_ab=np.ones((lg.n_w, 2)),
                feat_id=0,
                hf_wifi=np.float64(1.0),
            )
        ],
    )
    mean_cv_rmse_step_c_np([fake], lg, 0, 2.0, 15.0, 1.2, 2.5, 2)


def cv_mean_rmse_v3_step_a(train_imp: pd.DataFrame, loc: FusionLocalizerV8, n_splits: int) -> float:
    n_splits = min(n_splits, len(train_imp))
    if n_splits < 2:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses: List[float] = []
    for tr_idx, te_idx in kf.split(np.arange(len(train_imp))):
        te_df = train_imp.iloc[te_idx].reset_index(drop=True)
        xy_a = predict_step_a_v3_wifi_equivalent(te_df, loc)
        tx = te_df["True_X"].values.astype(float)
        ty = te_df["True_Y"].values.astype(float)
        ea = point_errors(tx, ty, xy_a[:, 0], xy_a[:, 1])
        r, _ = rmse_mae(ea)
        rmses.append(float(r))
    return float(np.mean(rmses))


def make_objective(
    fold_np: List[Dict[str, Any]],
    lg: LocGeomPack,
) -> Callable[[optuna.Trial], float]:
    """Train 평균 Step C만 최소화. 제안 변수 범위는 UWB 기하·융합 허브 관례에 맞춘 설계 고정(v13_fix Optuna 블록과 동일 성격)."""

    def objective(trial: optuna.Trial) -> float:
        rank = trial.suggest_int("phase_a_rank", 0, TOP_A_CANDIDATES - 1)
        gate = trial.suggest_float("gate_threshold_m", 1.25, 2.95)
        uvb_inf = trial.suggest_float("uwb_variance_inflate", 10.0, 24.0)
        hf_fus = trial.suggest_float("huber_f_fusion", 1.0, 1.35)
        ith = trial.suggest_float("irls_residual_thresh_m", 2.2, 3.1)
        mx = 2
        return mean_cv_rmse_step_c_np(fold_np, lg, rank, gate, uvb_inf, hf_fus, float(ith), mx)

    return objective


def save_cdf_steps_turbo(
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
    plt.title("V12 Turbo: 검증 오차 CDF — Step A/B/C (단일 Forward)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_cdf_steps.png", dpi=PLOT_DPI)
    plt.close()


def save_maps_turbo(
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
        ax.scatter(txv, tyv, s=42, c="#1f77b4", alpha=0.88, edgecolors="#0d2c44", linewidths=0.35, label="실제 (True)", zorder=2)
        ax.scatter(pxv, pyv, s=52, c="#ff7f0e", alpha=0.78, marker="x", linewidths=0.95, label="측위 추정", zorder=3)
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
    fig.suptitle("V12 Turbo — 검증 단일 Forward (Train Optuna CV)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_dir / f"{ART_PREFIX}_map_true_vs_predicted_steps.png", dpi=PLOT_DPI)
    plt.close()


def save_error_vectors_turbo(out_dir: Path, tx: np.ndarray, ty: np.ndarray, preds_c: np.ndarray, err_c_m: np.ndarray) -> None:
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
        fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02).set_label("오차 (m)")
    else:
        ax.scatter(tx[m], ty[m], s=54, c="seagreen", alpha=0.85, edgecolors="k", linewidths=0.35, label="실제(True)", zorder=3)
    ax.quiver(tx[m], ty[m], u, v, angles="xy", scale_units="xy", scale=1, width=0.0032, alpha=0.5, color="navy", zorder=2)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("V12 Turbo Step C — 오차 벡터 (실제 → 추정)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_map_error_vectors_stepC.png", dpi=PLOT_DPI)
    plt.close()


def save_pairwise_turbo(out_dir: Path, tx: np.ndarray, ty: np.ndarray, preds_c: np.ndarray, err_c_m: np.ndarray) -> None:
    m = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(preds_c[:, 0]) & np.isfinite(preds_c[:, 1])
    if not np.any(m):
        return
    err = np.asarray(err_c_m[m], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 10.8))
    ax.scatter(tx[m], ty[m], s=140, facecolors="none", edgecolors="dimgray", linewidths=1.1, label="실제 (True)", zorder=3)
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
    ax.set_title("V12 Turbo Step C — 실제(회색 원) vs 추정(색은 오차)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_map_stepC_pairwise_truth_pred.png", dpi=PLOT_DPI)
    plt.close()


def main() -> int:
    save_plots = "--no-plots" not in sys.argv
    t0 = time.perf_counter()
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
    n_splits = min(MAX_KFOLD_SPLITS, resolve_kfold_n_splits(len(train_df)))
    print(f"\n[V12 Turbo] 학습 {len(train_df)}행 | 검증 {len(val_df)}행 | Train K-Fold={n_splits}", flush=True)

    train_imp, val_imp = wifi_median_impute_fusion(train_df, val_df, loc.wifi_cols)
    lg = build_loc_geom_pack(loc)

    tri_ref = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m=0.6, huber_f_scale=1.0)
    calib_specs = v12_calibration_spec_list()

    # Phase A 순위 후보 그리드: v12 공유 상수(FEAT×calib×HUBER_F_WIFI). 검증 라벨 없이 고정 관례; 정렬키는 동일 Proxy RMSE+V12 tie-break.
    cv_rmse_v3_a = cv_mean_rmse_v3_step_a(train_imp, loc, n_splits)
    phase_a_records: List[Dict[str, Any]] = []
    cand_a: List[Tuple[float, CalibMethodSpec, str, float]] = []
    for feat in FEAT_MODES:
        for spec in calib_specs:
            for hf in HUBER_F_WIFI_GRID:
                phase_a_records.append(
                    dict(
                        feat_mode=feat,
                        calib_kind=spec.kind,
                        huber_f_wifi=float(hf),
                        train_CV_mean_RMSE_A_m=float(cv_rmse_v3_a),
                        train_CV_mean_RMSE_A_note="V3-equivalent Wi-Fi-only trilateration (same for all rows)",
                        kfold_splits=int(n_splits),
                    )
                )
                cand_a.append((float(cv_rmse_v3_a), spec, feat, float(hf)))
    cand_a.sort(key=lambda t: (t[0], t[1].kind, t[2], t[3]))
    top_a = cand_a[:TOP_A_CANDIDATES]

    print("[V12 Turbo] Phase B/C용 Train K-Fold 캐시(NumPy) 구축…", flush=True)
    fold_cache = build_phase_bc_fold_cache(train_imp, loc, tri_ref, top_a, n_splits)
    fold_np = fold_cache_to_numpy_only(fold_cache, loc)
    _warmup_numba(lg)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=OPTUNA_SEED, constant_liar=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = make_objective(fold_np, lg)
    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=False,
    )

    bt = study.best_trial
    best_rank = int(bt.params["phase_a_rank"])
    best_gate = float(bt.params["gate_threshold_m"])
    best_uvb = float(bt.params["uwb_variance_inflate"])
    best_hf_fus = float(bt.params["huber_f_fusion"])
    best_ith = float(bt.params["irls_residual_thresh_m"])
    best_mx = 2
    best_cv_c = float(bt.value)
    cv_ra0, best_spec, best_feat, best_hf_wifi = top_a[best_rank]

    best_sel = dict(
        spec=best_spec,
        feat=str(best_feat),
        hf_wifi=float(best_hf_wifi),
        gate=best_gate,
        uwb_inf=best_uvb,
        hf_fus=best_hf_fus,
        irls_thresh=best_ith,
        irls_mx=best_mx,
        train_cv_rmse_A_proxy=float(cv_ra0),
        train_cv_rmse_C=float(best_cv_c),
        phase_a_rank=int(best_rank),
    )

    train_fit_full = augment_train_for_canon_fit(train_imp, loc)
    coeffs_final = fit_robust_calibration_per_ap(train_fit_full, tri_ref, feat_mode=best_sel["feat"], spec=best_sel["spec"])
    coeff_ab_final = coeffs_dict_to_ab(coeffs_final, loc)
    feat_id_final = int(feat_to_id(str(best_sel["feat"])))

    val_blocks = te_df_to_numpy_blocks(val_imp, loc)
    xy_a_val = predict_step_a_v3_wifi_equivalent(val_imp, loc)

    preds_b_final = batch_step_b_preds_flat(
        len(val_imp),
        xy_a_val.astype(np.float64),
        val_blocks["uwb_med"],
        val_blocks["uwb_var"],
        val_blocks["uwb_was_nan"],
        val_blocks["wifi_med"],
        val_blocks["wifi_var"],
        coeff_ab_final,
        feat_id_final,
        best_sel["gate"],
        best_sel["uwb_inf"],
        best_sel["hf_fus"],
        lg.n_u,
        lg.n_w,
        lg.uwb_xy,
        lg.wifi_xy,
        lg.uwb_robust_bias,
        lg.wifi_robust_bias,
        lg.var_cap_uwb,
        lg.var_cap_wifi,
        lg.eps_w,
        lg.uwb_bias_w,
        lg.wifi_bias_w,
        lg.bounds_lo,
        lg.bounds_hi,
        TRILAT_GN_ITERS,
        float(MAX_RANGE_M),
    )

    preds_c_final = batch_step_c_preds_flat(
        len(val_imp),
        xy_a_val.astype(np.float64),
        val_blocks["uwb_med"],
        val_blocks["uwb_var"],
        val_blocks["uwb_was_nan"],
        val_blocks["wifi_med"],
        val_blocks["wifi_var"],
        coeff_ab_final,
        feat_id_final,
        best_sel["gate"],
        best_sel["uwb_inf"],
        best_sel["hf_fus"],
        best_sel["hf_wifi"],
        best_sel["irls_thresh"],
        best_sel["irls_mx"],
        float(TUKEY_C),
        lg.n_u,
        lg.n_w,
        lg.uwb_xy,
        lg.wifi_xy,
        lg.uwb_robust_bias,
        lg.wifi_robust_bias,
        lg.var_cap_uwb,
        lg.var_cap_wifi,
        lg.eps_w,
        lg.uwb_bias_w,
        lg.wifi_bias_w,
        lg.bounds_lo,
        lg.bounds_hi,
        TRILAT_GN_ITERS,
        float(MAX_RANGE_M),
    )

    tx = val_imp["True_X"].values.astype(float)
    ty = val_imp["True_Y"].values.astype(float)
    ea = point_errors(tx, ty, xy_a_val[:, 0], xy_a_val[:, 1])
    eb = point_errors(tx, ty, preds_b_final[:, 0], preds_b_final[:, 1])
    ec = point_errors(tx, ty, preds_c_final[:, 0], preds_c_final[:, 1])
    rmse_a, mae_a = rmse_mae(ea)
    rmse_b, mae_b = rmse_mae(eb)
    rmse_c, mae_c = rmse_mae(ec)
    d_ab = rmse_a - rmse_b
    d_bc = rmse_b - rmse_c
    d_ac = rmse_a - rmse_c

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print("V12 Turbo — Optuna(Train CV) | 검증 단일 Forward")
    print("=" * 72)
    print(f"  [Train CV 목표] Step C 평균 RMSE = {best_cv_c:.4f} m")
    print(f"  [Validation] Step A RMSE={rmse_a:.4f} m | Step B RMSE={rmse_b:.4f} m | Step C RMSE={rmse_c:.4f} m")
    print(f"  [Wall-clock] 총 {elapsed:.2f} s (목표: 10 s 미만)")
    print("=" * 72)
    print(
        f"[선택 HP] calib={best_sel['spec'].kind}, feat={best_sel['feat']}, huber_f_wifi={best_sel['hf_wifi']}, "
        f"gate={best_sel['gate']:.3f}, uwb_var_inflate={best_sel['uwb_inf']}, huber_f_fusion={best_sel['hf_fus']}, "
        f"IRLS_thresh={best_sel['irls_thresh']:.2f}, IRLS_iter={best_sel['irls_mx']}, phase_a_rank={best_rank}"
    )

    trials_rows = []
    for tr in study.trials:
        if tr.state != optuna.trial.TrialState.COMPLETE:
            continue
        trials_rows.append({**tr.params, "mean_cv_rmse_step_c_m": tr.value, "trial": tr.number})
    pd.DataFrame(trials_rows).to_csv(out_dir / f"{ART_PREFIX}_optuna_trials.csv", index=False, encoding="utf-8-sig")

    pref = resolve_paths(root)
    summ = {
        "pipeline": "V12 Turbo",
        "wall_clock_sec": float(elapsed),
        "optuna_n_trials": N_OPTUNA_TRIALS,
        "optuna_n_jobs": int(OPTUNA_N_JOBS),
        "matplotlib_artifacts_saved": bool(save_plots),
        "integrity": (
            "Hyperparameters selected by Optuna minimizing mean K-fold train CV RMSE after Step C. "
            "Validation ground truth used once for final metrics after full-train refit. "
            "Fold cache uses NumPy arrays only for parallel workers."
        ),
        "train_kfold_splits": int(n_splits),
        "train_cv_metrics_at_selection": {
            "phase_A_proxy_mean_RMSE_A_m": float(best_sel["train_cv_rmse_A_proxy"]),
            "mean_CV_RMSE_step_C_m": float(best_sel["train_cv_rmse_C"]),
        },
        "validation_single_forward_metrics": {
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
            "wifi_calibration_kind": best_sel["spec"].kind,
            "wifi_calibration_huber_epsilon": best_sel["spec"].huber_epsilon,
            "wifi_calibration_huber_alpha": best_sel["spec"].huber_alpha,
            "wifi_calibration_ransac_residual_thresh": best_sel["spec"].ransac_residual_threshold,
            "feat_mode": best_sel["feat"],
            "huber_f_scale_wifi_only_trilateration": best_sel["hf_wifi"],
            "uwb_geometric_gate_threshold_m": best_sel["gate"],
            "uwb_variance_inflate_multiplier": best_sel["uwb_inf"],
            "huber_f_scale_fusion_trilateration": best_sel["hf_fus"],
            "irls_residual_threshold_m": best_sel["irls_thresh"],
            "irls_max_iterations": best_sel["irls_mx"],
            "phase_A_top_K_evaluated": TOP_A_CANDIDATES,
            "phase_A_rank_selected": int(best_rank),
        },
        "data_paths": {
            "train_median": str(pref.train_median),
            "train_variance": str(pref.train_variance),
            "val_median": str(pref.val_median),
            "val_variance": str(pref.val_variance),
        },
        "artifacts": {
            "summary_json": str(out_dir / f"{ART_PREFIX}_summary.json"),
            "predictions_csv": str(out_dir / f"{ART_PREFIX}_predictions.csv"),
            "grid_phaseA_csv": str(out_dir / f"{ART_PREFIX}_grid_phaseA.csv"),
            "optuna_trials_csv": str(out_dir / f"{ART_PREFIX}_optuna_trials.csv"),
            "wifi_calibration_coefs_csv": str(out_dir / f"{ART_PREFIX}_wifi_calibration_coefs.csv"),
        },
    }
    (out_dir / f"{ART_PREFIX}_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")

    pred = val_imp[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xy_a_val[:, 0]
    pred["StepA_Y"] = xy_a_val[:, 1]
    pred["StepB_X"] = preds_b_final[:, 0]
    pred["StepB_Y"] = preds_b_final[:, 1]
    pred["StepC_X"] = preds_c_final[:, 0]
    pred["StepC_Y"] = preds_c_final[:, 1]
    pred["ErrA_m"] = ea
    pred["ErrB_m"] = eb
    pred["ErrC_m"] = ec
    pred.to_csv(out_dir / f"{ART_PREFIX}_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phase_a_records).to_csv(out_dir / f"{ART_PREFIX}_grid_phaseA.csv", index=False, encoding="utf-8-sig")

    if save_plots:
        save_cdf_steps_turbo(out_dir, dict(A=ea, B=eb, C=ec), dict(A=rmse_a, B=rmse_b, C=rmse_c), dict(A="Step A", B="Step B", C="Step C"))
        save_maps_turbo(out_dir, tx, ty, xy_a_val, preds_b_final, preds_c_final, rmse_a, rmse_b, rmse_c)
        save_error_vectors_turbo(out_dir, tx, ty, preds_c_final, ec)
        save_pairwise_turbo(out_dir, tx, ty, preds_c_final, ec)
    else:
        print("[V12 Turbo] --no-plots: CDF/맵 PNG 생략", flush=True)

    rows_coef = [{"wifi_canon_ap": k, "A": v[0], "B": v[1]} for k, v in coeffs_final.items()]
    pd.DataFrame(rows_coef).to_csv(out_dir / f"{ART_PREFIX}_wifi_calibration_coefs.csv", index=False, encoding="utf-8-sig")

    print(f"\n저장: {out_dir / f'{ART_PREFIX}_summary.json'}")
    print(f"저장: {out_dir / f'{ART_PREFIX}_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
