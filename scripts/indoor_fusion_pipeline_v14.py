"""
V14 Turbo + JWT — Step A·UWB는 **v12_turbo와 동일**(V3 Wi‑Fi 안심, 하드 UWB 게이트). Wi‑Fi 융합에만 JWT 반영:

- AP별 optional Isotonic, Train‑only **Score RANSAC**(임계 1.5m)·**삼각부등식(Numba)**.
- 가중 `1/(1 + w_r·RANSAC + w_t·triangle)` 는 JWT `final_outlier_score`와 같은 결합 형태.

무결성: Optuna는 Train K‑Fold Step C만, 검증 정답은 최종 리포트 1회.

CLI: ``--jwt-ablation`` — 최적 HP(gate·fusion 등)를 고정한 채 **Iso / 삼각만 / Score RANSAC만 / JWT 전체**를 단계적으로 켠 검증 Step B·C RMSE 표를 출력한다.

Step B가 Step A보다 나빠지는 흔한 원인: **Wi‑Fi JWT 소프트**(삼각·Score로 일부 앵커 가중을 줄임)와 **Isotonic으로 바뀐 거리**가 Turbo의 균일 `1/var` 가중·선형 보정과 달라 **UWB와의 기하 정합이 악화**되는 경우(특히 삼각 페널티가 클 때).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import KFold

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as e:  # pragma: no cover
    raise ImportError("V14에는 optuna가 필요합니다: py -3 -m pip install optuna") from e

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
    augment_train_for_canon_fit,
    point_errors,
    predict_step_a_v3_wifi_equivalent,
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

ART_PREFIX = "v14"

# 기본 약 50회(환경변수 `V14_OPTUNA_TRIALS`로 40~60 등 조정 권장). 병렬 시 `OPTUNA` 워커가 공유 `cal_cache`에 안전 접근하도록 락 사용.
N_OPTUNA_TRIALS = int(os.environ.get("V14_OPTUNA_TRIALS", "50"))
OPTUNA_N_JOBS = int(os.environ.get("V14_OPTUNA_JOBS", "-1"))
OPTUNA_SEED = 42
TRILAT_GN_ITERS = 22
PLOT_DPI = 110

# JWT 호환 허용오차 및 삼각 점수
JWT_BASE_TOL_DIFF = 1.0
JWT_REL_TOL_DIFF = 0.10
JWT_BASE_TOL_SUM = 0.8
JWT_REL_TOL_SUM = 0.08
JWT_TRIANGLE_NORM = 3.0
JWT_TRIANGLE_SCORE_CAP = 3.0
# JWT `apply_score_ransac_to_eval`: 보정거리→기하거리 선형 RANSAC 잔차 기반 점수
SCORE_RANSAC_THRESHOLD_M = 1.5
SCORE_RANSAC_MIN_SAMPLES = 20
SCORE_RANSAC_MAX_TRIALS = 500

# Isotonic RANSAC(보정 학습 단계만; 검증에서는 피팅된 곡선만 적용해 누수 없음)
ISO_RANSAC_RESID_THRESH = 5.0
ISO_MIN_VALID = 20
ISO_MIN_INLIERS = 10
ISO_RANSAC_MAX_TRIALS = 600

TUKEY_C = 4.685

_CAL_LOCK = threading.Lock()

MAX_ISO_KNOTS = 352
# v12_turbo와 동일: 작은 Phase A 집합으로 검증 일반화 유지
TOP_A_CANDIDATES = 3


MAX_KFOLD_SPLITS = 3


def resolve_kfold_n_splits(n_samples: int) -> int:
    if n_samples < 2:
        raise ValueError("Train 샘플이 2개 미만이면 K-Fold를 할 수 없습니다.")
    if n_samples < 6:
        return 2
    return int(min(5, max(3, n_samples // 35)))


@dataclass
class LocGeomPack:
    n_u: int
    n_w: int
    uwb_xy: np.ndarray
    wifi_xy: np.ndarray
    uwb_robust_bias: np.ndarray
    wifi_robust_bias: np.ndarray
    var_cap_uwb: float
    var_cap_wifi: float
    eps_w: float
    uwb_bias_w: float
    wifi_bias_w: float
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray


def build_loc_geom_pack(loc: FusionLocalizerV8) -> LocGeomPack:
    from indoor_fusion_pipeline_v8 import ROBUST_BIAS_M as _RB

    nu, nw = len(loc.uwb_cols), len(loc.wifi_cols)
    uwb_xy = np.zeros((nu, 2), dtype=np.float64)
    wifi_xy = np.zeros((nw, 2), dtype=np.float64)
    uwb_rb = np.zeros(nu, dtype=np.float64)
    wifi_rb = np.zeros(nw, dtype=np.float64)
    for i, c in enumerate(loc.uwb_cols):
        ax, ay = loc._resolve_sensor_position_m(c, i, True)
        uwb_xy[i, 0], uwb_xy[i, 1] = ax, ay
        uwb_rb[i] = float(_RB[sensor_canon_key(c, i, True)])
    for i, c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        ax, ay = loc._resolve_sensor_position_m(c, i, False)
        wifi_xy[i, 0], wifi_xy[i, 1] = ax, ay
        wifi_rb[i] = float(_RB[sensor_canon_key(canon, i, False)])
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


@dataclass
class WifiCalibNp:
    """Wi‑Fi 보정: 선형 또는 Isotonic 노드(노브가 없으면 해당 AP만 선형)."""

    coeff_ab: np.ndarray  # (nw, 2)
    use_iso: np.ndarray  # (nw,) uint8
    iso_x: np.ndarray  # (nw, MAX_ISO_KNOTS)
    iso_y: np.ndarray
    iso_n: np.ndarray  # (nw,) int32


def coeffs_dict_to_ab(coeffs: Dict[str, Tuple[float, float]], loc: FusionLocalizerV8) -> np.ndarray:
    ab = np.zeros((len(loc.wifi_cols), 2), dtype=np.float64)
    for i, _c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        a, b = coeffs[canon]
        ab[i, 0], ab[i, 1] = float(a), float(b)
    return ab


def feat_to_id(feat: str) -> int:
    return 0 if feat == "bias_sub" else 1


def _make_ransac_iso():
    base = LinearRegression(fit_intercept=False)

    try:
        return RANSACRegressor(
            estimator=base,
            min_samples=0.5,
            residual_threshold=float(ISO_RANSAC_RESID_THRESH),
            max_trials=ISO_RANSAC_MAX_TRIALS,
            loss="absolute_error",
            random_state=42,
        )
    except TypeError:
        return RANSACRegressor(
            base_estimator=base,
            min_samples=0.5,
            residual_threshold=float(ISO_RANSAC_RESID_THRESH),
            max_trials=ISO_RANSAC_MAX_TRIALS,
            loss="absolute_loss",
            random_state=42,
        )


def _fit_iso_knots_for_ap(
    X_raw: np.ndarray,
    y_geom: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(X_raw) < ISO_MIN_VALID:
        return None
    X1 = X_raw.reshape(-1, 1)
    rs = _make_ransac_iso()

    rs.fit(X1, y_geom)
    inliers = getattr(rs, "inlier_mask_", None)
    if inliers is None:
        return None
    Xin = np.asarray(X_raw[inliers], dtype=float)
    yin = np.asarray(y_geom[inliers], dtype=float)
    if Xin.size < ISO_MIN_INLIERS:
        return None
    order = np.argsort(Xin)
    Xin, yin = Xin[order], yin[order]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(Xin.reshape(-1, 1).ravel(), yin)
    xt = np.asarray(iso.X_thresholds_, dtype=np.float64)
    yt = np.asarray(iso.y_thresholds_, dtype=np.float64)
    if xt.size <= 1:
        return None
    if xt.size > MAX_ISO_KNOTS:
        sel = np.linspace(0, xt.size - 1, MAX_ISO_KNOTS).astype(int)
        xt, yt = xt[sel], yt[sel]
    return xt, yt


def fit_wifi_calib_pack_v14(
    train_aug: pd.DataFrame,
    loc: FusionLocalizerV8,
    spec: CalibMethodSpec,
    feat_mode: str,
    *,
    iso_ap_mask: int,
) -> WifiCalibNp:
    base_coeffs = fit_robust_calibration_per_ap(
        train_aug,
        WifiOnlyTrilateration(WIFI_COL_CANON, grid_m=0.6, huber_f_scale=1.0),
        feat_mode=str(feat_mode),
        spec=spec,
    )
    coeff_ab = coeffs_dict_to_ab(base_coeffs, loc)
    use_iso = np.zeros(len(loc.wifi_cols), dtype=np.uint8)
    iso_x = np.full((len(loc.wifi_cols), MAX_ISO_KNOTS), np.nan, dtype=np.float64)
    iso_y = np.full_like(iso_x, np.nan)
    iso_n = np.zeros(len(loc.wifi_cols), dtype=np.int32)

    for wi, c in enumerate(loc.wifi_cols):
        if not ((iso_ap_mask >> wi) & 1):
            continue
        canon = WIFI_COL_CANON[wi]
        ax, ay = loc._resolve_sensor_position_m(c, wi, False)
        feats: List[float] = []
        geoms: List[float] = []
        key = sensor_canon_key(canon, wi, False)
        for _, row in train_aug.iterrows():
            mr = row.get(f"med_{canon}", np.nan)
            vv = row.get(f"var_{canon}", np.nan)
            if not np.isfinite(mr) or not np.isfinite(vv) or float(vv) > VAR_CAP_WIFI_FUSION or float(mr) > MAX_RANGE_M:
                continue
            tx, ty = float(row["True_X"]), float(row["True_Y"])
            geoms.append(float(np.hypot(tx - ax, ty - ay)))
            if feat_mode == "raw":
                feats.append(float(mr))
            else:
                feats.append(float(mr) - float(ROBUST_BIAS_M[key]))
        X_raw = np.asarray(feats, dtype=np.float64)
        y_geom = np.asarray(geoms, dtype=np.float64)
        kn = _fit_iso_knots_for_ap(X_raw, y_geom)
        if kn is None:
            use_iso[wi] = 0
            continue
        xt, yt = kn
        nk = xt.size
        use_iso[wi] = 1
        iso_n[wi] = int(nk)
        iso_x[wi, :nk] = xt
        iso_y[wi, :nk] = yt

    return WifiCalibNp(
        coeff_ab=coeff_ab,
        use_iso=use_iso.astype(np.uint8),
        iso_x=iso_x,
        iso_y=iso_y,
        iso_n=iso_n.astype(np.int32),
    )


CalibScoreBundle = Tuple[WifiCalibNp, List[Optional[Any]]]


def fit_jwt_score_ransac_models(
    train_aug: pd.DataFrame,
    loc: FusionLocalizerV8,
    pack: WifiCalibNp,
    feat: str,
) -> List[Optional[Any]]:
    """JWT `train_score_ransac_models`와 동일: X=보정거리, y=True 기하거리."""
    nw = len(loc.wifi_cols)

    fid = feat_to_id(str(feat))
    out: List[Optional[Any]] = []
    for j, c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[j]
        ax, ay = loc._resolve_sensor_position_m(c, j, False)

        key_u = sensor_canon_key(canon, j, False)

        feats_x: List[List[float]] = []
        geoms: List[float] = []
        for _, row in train_aug.iterrows():
            mr = row.get(f"med_{canon}", np.nan)

            vv_r = row.get(f"var_{canon}", np.nan)
            if not np.isfinite(mr) or not np.isfinite(vv_r):
                continue
            if float(vv_r) > VAR_CAP_WIFI_FUSION or float(mr) > MAX_RANGE_M:
                continue
            tx, ty = float(row["True_X"]), float(row["True_Y"])
            d_geom = float(np.hypot(tx - ax, ty - ay))

            xf = float(mr) - float(ROBUST_BIAS_M[key_u]) if fid == 0 else float(mr)
            ca, cb = pack.coeff_ab[j, 0], pack.coeff_ab[j, 1]
            nk = int(pack.iso_n[j])
            if pack.use_iso[j] != 0 and nk >= 2:
                xx = np.asarray(pack.iso_x[j, :nk], dtype=np.float64)
                yy = np.asarray(pack.iso_y[j, :nk], dtype=np.float64)
                d_cal = float(np.interp(np.float64(xf), xx, yy))

            else:

                d_cal = float(ca * xf + cb)

            d_cal = d_cal if d_cal >= 0.1 else 0.1
            feats_x.append([d_cal])
            geoms.append(d_geom)

        if len(feats_x) < SCORE_RANSAC_MIN_SAMPLES:
            out.append(None)
            continue
        X1 = np.asarray(feats_x, dtype=np.float64)
        y1 = np.asarray(geoms, dtype=np.float64)
        try:
            rs = RANSACRegressor(
                estimator=LinearRegression(fit_intercept=False),
                min_samples=0.5,
                residual_threshold=float(SCORE_RANSAC_THRESHOLD_M),
                max_trials=SCORE_RANSAC_MAX_TRIALS,
                loss="absolute_error",
                random_state=42,
            )
        except TypeError:  # pragma: no cover
            rs = RANSACRegressor(
                base_estimator=LinearRegression(fit_intercept=False),
                min_samples=0.5,
                residual_threshold=float(SCORE_RANSAC_THRESHOLD_M),
                max_trials=SCORE_RANSAC_MAX_TRIALS,
                loss="absolute_loss",
                random_state=42,
            )

        try:
            rs.fit(X1, y1)

            out.append(rs)
        except Exception:
            out.append(None)
    assert len(out) == nw
    return out


def build_jwt_ransac_score_matrix(
    df: pd.DataFrame,
    loc: FusionLocalizerV8,
    pack: WifiCalibNp,
    models: List[Optional[Any]],
    dcal: np.ndarray,
    vt: np.ndarray,
) -> np.ndarray:
    thr = float(SCORE_RANSAC_THRESHOLD_M)

    nw = len(loc.wifi_cols)
    n = len(df)
    out = np.zeros((n, nw), dtype=np.float64)

    tx = df["True_X"].values.astype(np.float64)

    ty = df["True_Y"].values.astype(np.float64)
    for k in range(n):
        for j in range(nw):
            if int(vt[k, j]) == 0:
                continue
            mdl = models[j]

            if mdl is None:
                continue

            corr = float(dcal[k, j])

            if not np.isfinite(corr):
                continue
            try:
                pred_raw = float(mdl.predict(np.array([[corr]], dtype=np.float64))[0])

            except Exception:
                continue

            pred = pred_raw if pred_raw > 0.0 else 0.0

            c = loc.wifi_cols[j]
            ax, ay = loc._resolve_sensor_position_m(c, j, False)
            geom = float(np.hypot(tx[k] - ax, ty[k] - ay))
            abs_resid = abs(geom - pred)
            outlier = abs_resid > thr
            score = abs_resid / thr
            if score > 3.0:

                score = 3.0
            if not outlier:

                score *= 0.5

            out[k, j] = score

    return out


def run_wifi_geom_aux_numba(
    n_te: int,
    nw: int,
    wifi_med: np.ndarray,
    wifi_var: np.ndarray,
    pack: WifiCalibNp,
    feat_id: int,
    lg: LocGeomPack,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_L: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dcal = np.full((n_te, nw), np.nan, dtype=np.float64)
    vt = np.zeros((n_te, nw), dtype=np.uint8)
    tri = np.zeros((n_te, nw), dtype=np.float64)
    batch_wifi_calibrated_dist_triangle(
        n_te,
        nw,
        wifi_med,
        wifi_var,
        pack.coeff_ab,
        pack.use_iso,
        pack.iso_x,
        pack.iso_y,
        pack.iso_n,
        np.int32(feat_id),
        lg.wifi_robust_bias,
        lg.var_cap_wifi,
        float(MAX_RANGE_M),
        pair_i,
        pair_j,
        pair_L,
        JWT_TRIANGLE_NORM,
        JWT_TRIANGLE_SCORE_CAP,
        dcal,
        vt,
        tri,
    )
    return dcal, vt, tri


def precompute_anchor_pair_indices(n_w: int) -> Tuple[np.ndarray, np.ndarray]:
    pis: List[int] = []
    pjs: List[int] = []
    for i in range(n_w):
        for j in range(i + 1, n_w):
            pis.append(i)
            pjs.append(j)
    return np.asarray(pis, dtype=np.int64), np.asarray(pjs, dtype=np.int64)


def anchor_pair_lengths_m(wifi_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_w = wifi_xy.shape[0]
    pis, pjs = precompute_anchor_pair_indices(n_w)
    Ll = []
    for i, j in zip(pis, pjs):
        dx = wifi_xy[int(i), 0] - wifi_xy[int(j), 0]
        dy = wifi_xy[int(i), 1] - wifi_xy[int(j), 1]
        Ll.append(np.sqrt(dx * dx + dy * dy))
    return pis, pjs, np.asarray(Ll, dtype=np.float64)


# ---------- Numba: 삼각부등식, Isotonic 보간, Tukey/Huber/삼변 ----------


@njit(cache=True, fastmath=True)
def interp_iso_sorted_njit(x: float, xt: np.ndarray, yt: np.ndarray, nk: int) -> float:
    """단조 보간(Sklearn 노드 순 증가 가정)."""
    if nk < 2:
        return np.nan
    if x <= xt[0]:
        return yt[0]
    if x >= xt[nk - 1]:
        return yt[nk - 1]
    lo = 0
    hi = nk - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xt[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xt[lo]) / (xt[hi] - xt[lo] + 1e-12)
    return yt[lo] * (1.0 - t) + yt[hi] * t


@njit(cache=True, fastmath=True)
def triangle_violation_pair_njit(
    d_i: float,
    d_j: float,
    L_ij: float,
    tol_diff_const: float,
    tol_rel_diff: float,
    tol_sum_const: float,
    tol_rel_sum: float,
    valid_i: np.uint8,
    valid_j: np.uint8,
):
    if valid_i == 0 or valid_j == 0:
        return 0.0, 0.0
    if d_i <= 0 or d_j <= 0:
        return 0.0, 0.0
    tol_diff = tol_diff_const + tol_rel_diff * L_ij
    tol_sum = tol_sum_const + tol_rel_sum * L_ij
    abs_diff = d_i - d_j
    diff_violation = abs_diff if abs_diff >= 0.0 else -abs_diff
    diff_violation = diff_violation - L_ij - tol_diff
    sum_violation = L_ij - (d_i + d_j) - tol_sum
    v_diff = diff_violation if diff_violation > 0.0 else 0.0
    v_sum = sum_violation if sum_violation > 0.0 else 0.0
    violation = v_diff if v_diff >= v_sum else v_sum
    if violation <= 0.0:
        return 0.0, 0.0
    if v_diff >= v_sum:
        rat = violation / (tol_diff + 1e-9)
    else:
        rat = violation / (tol_sum + 1e-9)
    return violation, rat


@njit(cache=True, parallel=False)
def batch_wifi_calibrated_dist_triangle(
    n: int,
    nw: int,
    wifi_med: np.ndarray,
    wifi_var: np.ndarray,
    coeff_ab: np.ndarray,
    use_iso: np.ndarray,
    iso_x: np.ndarray,
    iso_y: np.ndarray,
    iso_n: np.ndarray,
    feat_id: np.int32,
    wifi_rb: np.ndarray,
    var_cap_wifi: float,
    max_range_m: float,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_L: np.ndarray,
    jwt_norm: float,
    jwt_cap: float,
    out_dcal: np.ndarray,
    out_valid: np.ndarray,
    out_tri: np.ndarray,
) -> None:
    npair = pair_i.shape[0]
    for k in range(n):
        ds = np.empty(nw, dtype=np.float64)
        vv = np.empty(nw, dtype=np.uint8)
        for j in range(nw):
            vv[j] = 0
            raw = wifi_med[k, j]
            vva = wifi_var[k, j]
            if not (np.isfinite(raw) and np.isfinite(vva)):
                continue
            if vva > var_cap_wifi:
                continue
            if raw > max_range_m:
                continue
            vv[j] = 1
            if feat_id == 0:
                xf = raw - wifi_rb[j]
            else:
                xf = raw
            a = coeff_ab[j, 0]
            b = coeff_ab[j, 1]
            if use_iso[j] != 0 and iso_n[j] >= 2:
                di = interp_iso_sorted_njit(
                    xf,
                    iso_x[j, : iso_n[j]],
                    iso_y[j, : iso_n[j]],
                    int(iso_n[j]),
                )
            else:
                di = a * xf + b
            if di < 0.1:
                di = 0.1
            ds[j] = di
            out_dcal[k, j] = di
            out_valid[k, j] = 1

        rs = np.zeros(nw, dtype=np.float64)
        mxr = np.zeros(nw, dtype=np.float64)
        cnt = np.zeros(nw, dtype=np.int32)
        for p in range(npair):
            ii = pair_i[p]
            jj = pair_j[p]
            Lij = pair_L[p]
            vi = vv[ii]
            vj = vv[jj]
            vsc, ratt = triangle_violation_pair_njit(
                ds[ii],
                ds[jj],
                Lij,
                JWT_BASE_TOL_DIFF,
                JWT_REL_TOL_DIFF,
                JWT_BASE_TOL_SUM,
                JWT_REL_TOL_SUM,
                vi,
                vj,
            )
            if vsc > 0.0:
                rs[ii] += vsc
                rs[jj] += vsc
                if ratt > mxr[ii]:
                    mxr[ii] = ratt
                if ratt > mxr[jj]:
                    mxr[jj] = ratt
                cnt[ii] += 1
                cnt[jj] += 1
        for j in range(nw):
            if vv[j] == 0:
                out_tri[k, j] = 0.0
                out_valid[k, j] = 0
                out_dcal[k, j] = np.nan
                continue
            raw_sc = rs[j] / (jwt_norm + 1e-9)
            comb = raw_sc if raw_sc >= mxr[j] else mxr[j]
            if comb > jwt_cap:
                comb = jwt_cap
            out_tri[k, j] = comb


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
def batch_step_b_preds_v14_flat(
    n: int,
    xy_guide: np.ndarray,
    uwb_med: np.ndarray,
    uwb_var: np.ndarray,
    uwb_was_nan: np.ndarray,
    wifi_med: np.ndarray,
    wifi_var: np.ndarray,
    wifi_tri_score: np.ndarray,
    wifi_ransac_score: np.ndarray,
    wifi_valid_mask: np.ndarray,
    coeff_ab: np.ndarray,
    use_iso: np.ndarray,
    iso_x: np.ndarray,
    iso_y: np.ndarray,
    iso_n: np.ndarray,
    feat_id: int,
    gate_t: float,
    w_jwt_ransac: float,
    w_jwt_tri: float,
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
                vv_u = uwb_var[k, j]
                if (not np.isfinite(dr)) or (not np.isfinite(vv_u)):
                    continue
                if vv_u > var_cap_uwb:
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
                eff_var = vv_u * infm
                wi = 1.0 / (eff_var + uwb_bias_w + eps_w)
                pos_b[ct, 0] = ax
                pos_b[ct, 1] = ay
                dist_b[ct] = d_meas
                w_b[ct] = wi
                ct += 1
        for j in range(nw):
            mr = wifi_med[k, j]
            vv_w = wifi_var[k, j]
            if not (np.isfinite(mr) and np.isfinite(vv_w)):
                continue
            if vv_w > var_cap_wifi:
                continue
            if mr > max_range_m:
                continue
            if wifi_valid_mask[k, j] == 0:
                continue
            xf = mr - wifi_rb[j] if feat_id == 0 else mr
            A = coeff_ab[j, 0]
            B = coeff_ab[j, 1]
            if use_iso[j] != 0 and iso_n[j] >= 2:
                d_c = interp_iso_sorted_njit(xf, iso_x[j, : iso_n[j]], iso_y[j, : iso_n[j]], int(iso_n[j]))
            else:
                d_c = A * xf + B
            if d_c < 0.1:
                d_c = 0.1
            jwt_comb = w_jwt_ransac * wifi_ransac_score[k, j] + w_jwt_tri * wifi_tri_score[k, j]
            denom = 1.0 + jwt_comb
            if denom < 1e-12:
                denom = 1e-12
            wi2 = 1.0 / (vv_w + wifi_bias_w + eps_w) / denom
            pos_b[ct, 0] = wifi_xy[j, 0]
            pos_b[ct, 1] = wifi_xy[j, 1]
            dist_b[ct] = d_c
            w_b[ct] = wi2
            ct += 1
        if ct == 0:
            preds[k, 0], preds[k, 1] = np.nan, np.nan
        else:
            preds[k, :] = solve_trilat_bounded_huber_njit(
                pos_b[:ct, :], dist_b[:ct], w_b[:ct], hf_fusion, bounds_lo, bounds_hi, gn_iters
            )
    return preds


@njit(cache=True, parallel=False)
def batch_step_c_preds_v14_flat(
    n: int,
    xy_guide: np.ndarray,
    uwb_med: np.ndarray,
    uwb_var: np.ndarray,
    uwb_was_nan: np.ndarray,
    wifi_med: np.ndarray,
    wifi_var: np.ndarray,
    wifi_tri_score: np.ndarray,
    wifi_ransac_score: np.ndarray,
    wifi_valid_mask: np.ndarray,
    coeff_ab: np.ndarray,
    use_iso: np.ndarray,
    iso_x: np.ndarray,
    iso_y: np.ndarray,
    iso_n: np.ndarray,
    feat_id: int,
    gate_t: float,
    w_jwt_ransac: float,
    w_jwt_tri: float,
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
                vv_u = uwb_var[k, j]
                if (not np.isfinite(dr)) or (not np.isfinite(vv_u)):
                    continue
                if vv_u > var_cap_uwb:
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
                eff_var = vv_u * infm
                wi = 1.0 / (eff_var + uwb_bias_w + eps_w)
                pos_b[ct, 0] = ax
                pos_b[ct, 1] = ay
                dist_b[ct] = d_meas
                w_b[ct] = wi
                ct += 1
        for j in range(nw):
            m_raw = wifi_med[k, j]
            vv_w = wifi_var[k, j]
            if not (np.isfinite(m_raw) and np.isfinite(vv_w)):
                continue
            if vv_w > var_cap_wifi:
                continue
            if m_raw > max_range_m:
                continue
            if wifi_valid_mask[k, j] == 0:
                continue
            xf = m_raw - wifi_rb[j] if feat_id == 0 else m_raw
            A = coeff_ab[j, 0]
            B = coeff_ab[j, 1]
            if use_iso[j] != 0 and iso_n[j] >= 2:
                d_c = interp_iso_sorted_njit(xf, iso_x[j, : iso_n[j]], iso_y[j, : iso_n[j]], int(iso_n[j]))
            else:
                d_c = A * xf + B
            if d_c < 0.1:
                d_c = 0.1
            jwt_comb = w_jwt_ransac * wifi_ransac_score[k, j] + w_jwt_tri * wifi_tri_score[k, j]
            denom_j = 1.0 + jwt_comb
            if denom_j < 1e-12:
                denom_j = 1e-12
            wi2 = 1.0 / (vv_w + wifi_bias_w + eps_w) / denom_j
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
            mr2 = wifi_med[k, j]
            vv2 = wifi_var[k, j]
            if not (np.isfinite(mr2) and np.isfinite(vv2)):
                continue
            if vv2 > var_cap_wifi:
                continue
            if mr2 > max_range_m:
                continue
            if wifi_valid_mask[k, j] == 0:
                continue
            xf2 = mr2 - wifi_rb[j] if feat_id == 0 else mr2
            A2 = coeff_ab[j, 0]
            B2 = coeff_ab[j, 1]
            if use_iso[j] != 0 and iso_n[j] >= 2:
                dd = interp_iso_sorted_njit(xf2, iso_x[j, : iso_n[j]], iso_y[j, : iso_n[j]], int(iso_n[j]))
            else:
                dd = A2 * xf2 + B2
            if dd < 0.1:
                dd = 0.1
            wpos[cw, 0] = wifi_xy[j, 0]
            wpos[cw, 1] = wifi_xy[j, 1]
            wdist[cw] = dd
            wwei[cw] = 1.0 / (vv2 + wifi_bias_w + eps_w)
            cw += 1
        if ct < 3:
            if cw == 0:
                preds[k, 0] = np.nan
                preds[k, 1] = np.nan
            else:
                xyw = solve_trilat_bounded_huber_njit(
                    wpos[:cw, :].copy(), wdist[:cw].copy(), wwei[:cw].copy(), hf_wifi, bounds_lo, bounds_hi, gn_iters
                )
                preds[k, 0] = xyw[0]
                preds[k, 1] = xyw[1]
        else:
            preds[k, :] = irls_refine_xyz_njit(
                pos_b[:ct, :].copy(),
                dist_b[:ct].copy(),
                w_b[:ct].copy(),
                hf_fusion,
                irls_thresh,
                irls_mx,
                tukey_c,
                bounds_lo,
                bounds_hi,
                gn_iters,
            )
    return preds


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


def _build_one_fold_v14(
    fold_index: int,
    train_imp: pd.DataFrame,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    _loc: FusionLocalizerV8,
    top_a: List[Tuple[float, CalibMethodSpec, str, float]],
) -> Tuple[int, Dict[str, Any]]:
    tr_df = train_imp.iloc[tr_idx].reset_index(drop=True)

    te_df = train_imp.iloc[te_idx].reset_index(drop=True)

    per_rank: List[Dict[str, Any]] = []
    for (_cv_ra, spec, feat, hf_wifi) in top_a:
        per_rank.append(dict(spec=spec, feat=feat, hf_wifi=float(hf_wifi)))
    return fold_index, dict(tr_df=tr_df, te_df=te_df, per_rank=per_rank)


def build_phase_bc_fold_cache_v14(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
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
            ex.submit(_build_one_fold_v14, fi, train_imp, tr_idx, te_idx, loc, top_a)
            for fi, (tr_idx, te_idx) in enumerate(splits)
        ]
        for fut in as_completed(futs):
            fi, block = fut.result()
            fold_slots[fi] = block
    return [fc for fc in fold_slots if fc is not None]


def fold_cache_to_numpy_only_v14(
    fold_cache: List[Dict[str, Any]],
    loc: FusionLocalizerV8,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fc in fold_cache:
        te_df: pd.DataFrame = fc["te_df"]
        tr_df: pd.DataFrame = fc["tr_df"]
        blk = te_df_to_numpy_blocks(te_df, loc)
        blk["xy_a_v3"] = predict_step_a_v3_wifi_equivalent(te_df, loc).astype(np.float64)
        per_np: List[Dict[str, Any]] = []
        for pr in fc["per_rank"]:
            per_np.append(dict(spec=pr["spec"], feat=str(pr["feat"]), hf_wifi=np.float64(pr["hf_wifi"])))
        out.append(
            dict(n_te=int(len(te_df)), tr_df=tr_df.copy(), te_df=te_df.copy(), blk=blk, per_rank=per_np)
        )
    return out


def _warmup_numba_v14(lg: LocGeomPack, pair_i: np.ndarray, pair_j: np.ndarray, pair_L: np.ndarray, pack_dummy: WifiCalibNp) -> None:
    nw = lg.n_w
    n = 3
    w_tri = np.zeros((n, nw), dtype=np.float64)
    wd = np.zeros((n, nw), dtype=np.float64)
    wv = np.zeros((n, nw), dtype=np.uint8)
    batch_wifi_calibrated_dist_triangle(
        n,
        nw,
        np.ones((n, nw)),
        np.ones((n, nw)),
        pack_dummy.coeff_ab,
        pack_dummy.use_iso,
        pack_dummy.iso_x,
        pack_dummy.iso_y,
        pack_dummy.iso_n,
        np.int32(0),
        lg.wifi_robust_bias,
        lg.var_cap_wifi,
        MAX_RANGE_M,
        pair_i,
        pair_j,
        pair_L,
        JWT_TRIANGLE_NORM,
        JWT_TRIANGLE_SCORE_CAP,
        wd,
        wv,
        w_tri,
    )
    wf = WifiCalibNp(
        coeff_ab=np.ones((nw, 2)),
        use_iso=np.zeros(nw, dtype=np.uint8),
        iso_x=np.full((nw, MAX_ISO_KNOTS), 0.5),
        iso_y=np.full((nw, MAX_ISO_KNOTS), 1.0),
        iso_n=np.full(nw, 2, dtype=np.int32),
    )
    _ = wf
    wran = np.zeros_like(w_tri)
    batch_step_c_preds_v14_flat(
        n,
        np.zeros((n, 2)),
        np.zeros((n, lg.n_u)),
        np.ones((n, lg.n_u)),
        np.zeros((n, lg.n_u), dtype=np.uint8),
        np.ones((n, nw)),
        np.ones((n, nw)),
        w_tri,
        wran,
        wv,
        pack_dummy.coeff_ab,
        pack_dummy.use_iso,
        pack_dummy.iso_x,
        pack_dummy.iso_y,
        pack_dummy.iso_n,
        0,
        2.0,
        0.0,
        0.0,
        15.0,
        1.2,
        1.05,
        2.2,
        2,
        float(TUKEY_C),
        lg.n_u,
        nw,
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


def cv_mean_rmse_v3_step_a(train_imp: pd.DataFrame, loc: FusionLocalizerV8, n_splits: int) -> float:
    """Phase A 순위 안내용: V12 Turbo와 동일한 V3 Wi‑Fi 안심 좌표 proxy."""

    n_splits = min(n_splits, len(train_imp))
    if n_splits < 2:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses: List[float] = []
    for _tr, te_idx in kf.split(np.arange(len(train_imp))):
        te_df = train_imp.iloc[te_idx].reset_index(drop=True)
        xy_a = predict_step_a_v3_wifi_equivalent(te_df, loc)
        tx = te_df["True_X"].values.astype(float)
        ty = te_df["True_Y"].values.astype(float)
        ea = point_errors(tx, ty, xy_a[:, 0], xy_a[:, 1])
        r, _ = rmse_mae(ea)
        rmses.append(float(r))
    return float(np.mean(rmses))


def mean_cv_rmse_step_c_np_v14(
    fold_objs: List[Dict[str, Any]],
    loc: FusionLocalizerV8,
    lg: LocGeomPack,
    *,
    rank: int,
    iso_mask: int,
    jwt_on: bool,
    hp: Dict[str, float],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_L: np.ndarray,
    cal_cache: Dict[Tuple[int, int, int], CalibScoreBundle],
) -> float:
    rmses: List[float] = []
    mx = int(hp["irls_mx"])
    gate = float(hp["gate_threshold_m"])
    uvb_inf = float(hp["uwb_variance_inflate"])
    hf_fus = float(hp["huber_f_fusion"])
    ith = float(hp["irls_residual_thresh_m"])
    w_jwt_tri = float(hp["w_jwt_triangle"]) if jwt_on else 0.0
    w_jwt_ra = float(hp["w_jwt_ransac"]) if jwt_on else 0.0
    nu, nw = lg.n_u, lg.n_w

    for fi, fo in enumerate(fold_objs):
        pr = fo["per_rank"][rank]
        spec: CalibMethodSpec = pr["spec"]
        feat = str(pr["feat"])
        hf_wifi = float(pr["hf_wifi"])
        key = (fi, rank, iso_mask)
        with _CAL_LOCK:
            if key not in cal_cache:
                tr_aug = augment_train_for_canon_fit(fo["tr_df"], loc)
                pack = fit_wifi_calib_pack_v14(tr_aug, loc, spec, feat, iso_ap_mask=iso_mask)
                score_mdls = fit_jwt_score_ransac_models(tr_aug, loc, pack, feat)
                cal_cache[key] = (pack, score_mdls)

            pack, score_mdls = cal_cache[key]
        te_df: pd.DataFrame = fo["te_df"]
        blk = fo["blk"]
        xy_a = blk["xy_a_v3"].astype(np.float64)
        dcal, vt, tri = run_wifi_geom_aux_numba(
            fo["n_te"],
            nw,
            blk["wifi_med"],
            blk["wifi_var"],
            pack,
            feat_to_id(feat),
            lg,
            pair_i,
            pair_j,
            pair_L,
        )
        rsc = (
            build_jwt_ransac_score_matrix(te_df, loc, pack, score_mdls, dcal, vt)
            if jwt_on
            else np.zeros((fo["n_te"], nw), dtype=np.float64)
        )
        fid = feat_to_id(feat)
        preds = batch_step_c_preds_v14_flat(
            fo["n_te"],
            xy_a,
            blk["uwb_med"],
            blk["uwb_var"],
            blk["uwb_was_nan"],
            blk["wifi_med"],
            blk["wifi_var"],
            tri,
            rsc,
            vt,
            pack.coeff_ab,
            pack.use_iso,
            pack.iso_x,
            pack.iso_y,
            pack.iso_n,
            int(fid),
            gate,
            w_jwt_ra,
            w_jwt_tri,
            uvb_inf,
            hf_fus,
            hf_wifi,
            float(ith),
            int(mx),
            float(TUKEY_C),
            nu,
            nw,
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
        ec = point_errors(blk["true_x"], blk["true_y"], preds[:, 0], preds[:, 1])
        rc, _ = rmse_mae(ec)
        rmses.append(float(rc))
    return float(np.mean(rmses))


def make_objective_v14(
    fold_objs: List[Dict[str, Any]],
    loc: FusionLocalizerV8,
    lg: LocGeomPack,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_L: np.ndarray,
    cal_cache: Dict[Tuple[int, int, int], CalibScoreBundle],
) -> Callable[[optuna.Trial], float]:
    n_rank = len(fold_objs[0]["per_rank"])
    n_w_bits = max(1, (1 << lg.n_w))

    def objective(trial: optuna.Trial) -> float:
        rank = trial.suggest_int("phase_a_rank", 0, n_rank - 1)
        iso_mask = trial.suggest_int("iso_ap_mask", 0, n_w_bits - 1)
        jwt_on = bool(trial.suggest_int("jwt_wifi_soft_enable", 0, 1))
        hp = dict(
            w_jwt_triangle=float(trial.suggest_float("w_jwt_triangle", 0.0, 5.0)),
            w_jwt_ransac=float(trial.suggest_float("w_jwt_ransac", 0.0, 5.5)),
            gate_threshold_m=float(trial.suggest_float("gate_threshold_m", 1.25, 2.95)),
            uwb_variance_inflate=float(trial.suggest_float("uwb_variance_inflate", 10.0, 24.0)),
            huber_f_fusion=float(trial.suggest_float("huber_f_fusion", 1.0, 1.35)),
            irls_residual_thresh_m=float(trial.suggest_float("irls_residual_thresh_m", 2.2, 3.1)),
            irls_mx=2.0,
        )
        return mean_cv_rmse_step_c_np_v14(
            fold_objs,
            loc,
            lg,
            rank=rank,
            iso_mask=int(iso_mask),
            jwt_on=jwt_on,
            hp=hp,
            pair_i=pair_i,
            pair_j=pair_j,
            pair_L=pair_L,
            cal_cache=cal_cache,
        )

    return objective


def val_forward_step_bc_rmse(
    val_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    lg: LocGeomPack,
    *,
    val_blocks: Dict[str, np.ndarray],
    xy_a_val: np.ndarray,
    pack: WifiCalibNp,
    feat_label: str,
    train_fit_full: pd.DataFrame,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_L: np.ndarray,
    gate: float,
    uvb_inf: float,
    hf_fusion: float,
    hf_wifi: float,
    irls_thresh: float,
    irls_mx: int,
    w_jwt_ransac: float,
    w_jwt_triangle: float,
    need_ransac_matrix: bool,
) -> Tuple[float, float, float]:
    """검증 단일 패스 Step B·C RMSE (및 Step A는 이미 계산된 것과 동일하게 참고용)."""

    nw = lg.n_w

    n_v = len(val_imp)

    fid_f = feat_to_id(str(feat_label))

    dcal, vt, tri = run_wifi_geom_aux_numba(
        n_v,
        nw,
        val_blocks["wifi_med"],
        val_blocks["wifi_var"],
        pack,
        int(fid_f),
        lg,
        pair_i,
        pair_j,
        pair_L,
    )

    if need_ransac_matrix and w_jwt_ransac > 1e-12:

        mdls = fit_jwt_score_ransac_models(train_fit_full, loc, pack, str(feat_label))

        rsc = build_jwt_ransac_score_matrix(val_imp, loc, pack, mdls, dcal, vt)

    else:

        rsc = np.zeros((n_v, nw), dtype=np.float64)

    preds_b = batch_step_b_preds_v14_flat(
        n_v,

        xy_a_val,
        val_blocks["uwb_med"],
        val_blocks["uwb_var"],
        val_blocks["uwb_was_nan"],
        val_blocks["wifi_med"],
        val_blocks["wifi_var"],
        tri,

        rsc,

        vt,

        pack.coeff_ab,
        pack.use_iso,
        pack.iso_x,

        pack.iso_y,

        pack.iso_n,
        int(fid_f),
        float(gate),

        float(w_jwt_ransac),

        float(w_jwt_triangle),

        float(uvb_inf),
        float(hf_fusion),

        lg.n_u,

        nw,

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

    preds_c = batch_step_c_preds_v14_flat(
        n_v,

        xy_a_val,

        val_blocks["uwb_med"],

        val_blocks["uwb_var"],

        val_blocks["uwb_was_nan"],

        val_blocks["wifi_med"],

        val_blocks["wifi_var"],

        tri,

        rsc,

        vt,

        pack.coeff_ab,

        pack.use_iso,

        pack.iso_x,

        pack.iso_y,

        pack.iso_n,

        int(fid_f),

        float(gate),

        float(w_jwt_ransac),

        float(w_jwt_triangle),

        float(uvb_inf),

        float(hf_fusion),

        float(hf_wifi),

        float(irls_thresh),

        int(irls_mx),

        float(TUKEY_C),

        lg.n_u,

        nw,

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

    eb = point_errors(tx, ty, preds_b[:, 0], preds_b[:, 1])

    ec = point_errors(tx, ty, preds_c[:, 0], preds_c[:, 1])

    rb, _ = rmse_mae(eb)

    rc, _ = rmse_mae(ec)

    ra, _ = rmse_mae(point_errors(tx, ty, xy_a_val[:, 0], xy_a_val[:, 1]))

    return float(rb), float(rc), float(ra)


def run_jwt_ablation_report(
    *,
    val_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    lg: LocGeomPack,
    train_fit_full: pd.DataFrame,

    val_blocks: Dict[str, np.ndarray],
    xy_a_val: np.ndarray,
    best_spec: CalibMethodSpec,
    best_feat: str,
    best_hf_wifi: float,

    best_iso: int,
    best_hp: Dict[str, Any],

    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_L: np.ndarray,

    out_dir: Path,
) -> List[Dict[str, Any]]:
    """단계 부착: iso0+JWT끔 → 선형에서 삼각/RANSAC만 → iso만 → iso+삼각/ RANSAC만 → JWT 전체 — gate·fusion·IRLS 고정."""

    g = float(best_hp["gate_threshold_m"])

    uv = float(best_hp["uwb_variance_inflate"])

    hf = float(best_hp["huber_f_fusion"])

    ith = float(best_hp["irls_residual_thresh_m"])

    imx = int(best_hp["irls_mx"])

    wt = float(best_hp["w_jwt_triangle"])

    wr = float(best_hp["w_jwt_ransac"])

    pack_linear = fit_wifi_calib_pack_v14(train_fit_full, loc, best_spec, str(best_feat), iso_ap_mask=0)

    pack_iso = fit_wifi_calib_pack_v14(train_fit_full, loc, best_spec, str(best_feat), iso_ap_mask=int(best_iso))

    stages: List[Tuple[str, WifiCalibNp, float, float, bool]] = [
        ("S0 iso=0 JWT off (Wi‑Fi 가중=Turbo형 1/var)", pack_linear, 0.0, 0.0, False),

        (
            f"S0b iso=0 w_tri={wt:.3f}, w_ransac=0 (선형보정 유지·삼각만)",
            pack_linear,
            0.0,
            wt,
            False,
        ),

        (
            f"S0c iso=0 w_ransac={wr:.3f}, w_tri=0 (선형보정 유지·Score RANSAC만)",
            pack_linear,
            wr,
            0.0,
            True,
        ),

        (f"S1 iso_mask={best_iso} JWT off (Isotonic 효과만)", pack_iso, 0.0, 0.0, False),

        (

            f"S2 iso_mask={best_iso} w_tri={wt:.3f}, w_ransac=0 (삼각만)",

            pack_iso,

            0.0,

            wt,

            False,

        ),

        (
            f"S2b iso_mask={best_iso} w_ransac={wr:.3f}, w_tri=0 (Score RANSAC만)",
            pack_iso,
            wr,
            0.0,
            True,
        ),

        (

            f"S3 iso_mask={best_iso} JWT full (w_tri={wt:.3f}, w_ransac={wr:.3f})",

            pack_iso,

            wr,

            wt,

            True,

        ),

    ]

    rows: List[Dict[str, Any]] = []

    print("\n── JWT / Isotonic 단계별 검증 RMSE (Optuna 선택 gate·fusion·IRLS 고정) ──", flush=True)

    colw = 60
    print(f"{'단계':<{colw}} {'StepA':>9} {'StepB':>9} {'StepC':>9}", flush=True)

    for name, pk, rwa, rtw, nr in stages:

        rsb, rsc_, rsa = val_forward_step_bc_rmse(
            val_imp,

            loc,
            lg,

            val_blocks=val_blocks,

            xy_a_val=xy_a_val,
            pack=pk,

            feat_label=str(best_feat),

            train_fit_full=train_fit_full,

            pair_i=pair_i,
            pair_j=pair_j,
            pair_L=pair_L,
            gate=g,

            uvb_inf=uv,

            hf_fusion=hf,

            hf_wifi=float(best_hf_wifi),

            irls_thresh=ith,

            irls_mx=imx,
            w_jwt_ransac=rwa,
            w_jwt_triangle=rtw,
            need_ransac_matrix=nr,

        )

        rows.append(dict(stage=name, RMSE_step_A_m=rsa, RMSE_step_B_m=rsb, RMSE_step_C_m=rsc_))

        disp = name if len(name) <= colw else (name[: colw - 1] + "…")
        print(f"{disp:<{colw}} {rsa:>9.4f} {rsb:>9.4f} {rsc_:>9.4f}", flush=True)

    ab_path = out_dir / f"{ART_PREFIX}_jwt_ablation_validation.csv"

    pd.DataFrame(rows).to_csv(ab_path, index=False, encoding="utf-8-sig")

    print(f"\n저장: {ab_path}", flush=True)

    return rows


def save_cdf_v14(
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
    plt.title("V14: 검증 오차 CDF — Step A/B/C (Train CV Optuna → 단일 검증)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_cdf_steps.png", dpi=PLOT_DPI)
    plt.close()


def save_maps_v14(
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
        (xy_a[:, 0], xy_a[:, 1], "Step A (V12식 V3 Wi‑Fi 안심)", rmse_a),
        (preds_b[:, 0], preds_b[:, 1], "Step B (하드 게이트 UWB + JWT Wi‑Fi 가중)", rmse_b),
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
    fig.suptitle("V14 — 검증 단일 Forward (Train Optuna CV)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_dir / f"{ART_PREFIX}_map_true_vs_predicted_steps.png", dpi=PLOT_DPI)
    plt.close()


def main() -> int:
    save_plots = "--no-plots" not in sys.argv
    run_jwt_ablation = "--jwt-ablation" in sys.argv
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
    print(f"\n[V14] 학습 {len(train_df)}행 | 검증 {len(val_df)}행 | Train K-Fold={n_splits}", flush=True)

    train_imp, val_imp = wifi_median_impute_fusion(train_df, val_df, loc.wifi_cols)
    lg = build_loc_geom_pack(loc)
    pair_i, pair_j, pair_L = anchor_pair_lengths_m(lg.wifi_xy)

    calib_specs = v12_calibration_spec_list()
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
                        train_CV_mean_RMSE_A_note="V3-equivalent Wi-Fi proxy (Phase A ordering only)",
                        kfold_splits=int(n_splits),
                    )
                )
                cand_a.append((float(cv_rmse_v3_a), spec, feat, float(hf)))
    cand_a.sort(key=lambda t: (t[0], t[1].kind, t[2], t[3]))
    top_a = cand_a[:TOP_A_CANDIDATES]

    print("[V14] Train K-Fold 캐시(폴드 DF + NumPy TE 블록) 구축…", flush=True)
    fold_cache = build_phase_bc_fold_cache_v14(train_imp, loc, top_a, n_splits)
    fold_objs = fold_cache_to_numpy_only_v14(fold_cache, loc)

    pack_dummy = fit_wifi_calib_pack_v14(
        augment_train_for_canon_fit(train_imp.iloc[: min(30, len(train_imp))], loc),
        loc,
        top_a[0][1],
        str(top_a[0][2]),
        iso_ap_mask=0,
    )
    _warmup_numba_v14(lg, pair_i, pair_j, pair_L, pack_dummy)

    cal_cache: Dict[Tuple[int, int, int], CalibScoreBundle] = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=OPTUNA_SEED, constant_liar=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        make_objective_v14(fold_objs, loc, lg, pair_i, pair_j, pair_L, cal_cache),
        n_trials=N_OPTUNA_TRIALS,
        n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=False,
    )

    bt = study.best_trial
    best_rank = int(bt.params["phase_a_rank"])
    best_iso = int(bt.params["iso_ap_mask"])
    best_jwt_on = bool(int(bt.params.get("jwt_wifi_soft_enable", 1)))

    best_hp = dict(
        w_jwt_triangle=float(bt.params["w_jwt_triangle"]),
        w_jwt_ransac=float(bt.params["w_jwt_ransac"]),
        gate_threshold_m=float(bt.params["gate_threshold_m"]),
        uwb_variance_inflate=float(bt.params["uwb_variance_inflate"]),
        huber_f_fusion=float(bt.params["huber_f_fusion"]),
        irls_residual_thresh_m=float(bt.params["irls_residual_thresh_m"]),
        irls_mx=2.0,
    )
    best_cv_c = float(bt.value)
    _cv_ra0, best_spec, best_feat, best_hf_wifi = top_a[best_rank]

    train_fit_full = augment_train_for_canon_fit(train_imp, loc)
    pack_final = fit_wifi_calib_pack_v14(train_fit_full, loc, best_spec, str(best_feat), iso_ap_mask=best_iso)
    score_models_final = (
        fit_jwt_score_ransac_models(train_fit_full, loc, pack_final, str(best_feat)) if best_jwt_on else [None] * lg.n_w
    )

    val_blocks = te_df_to_numpy_blocks(val_imp, loc)

    nw = lg.n_w
    xy_a_val = predict_step_a_v3_wifi_equivalent(val_imp, loc).astype(np.float64)

    fid_f = feat_to_id(str(best_feat))
    dcal_v, vt_v, tri_v = run_wifi_geom_aux_numba(
        len(val_imp),
        nw,
        val_blocks["wifi_med"],
        val_blocks["wifi_var"],
        pack_final,
        int(fid_f),
        lg,
        pair_i,
        pair_j,
        pair_L,
    )
    if best_jwt_on:
        rsc_v = build_jwt_ransac_score_matrix(val_imp, loc, pack_final, score_models_final, dcal_v, vt_v)

    else:
        rsc_v = np.zeros((len(val_imp), nw), dtype=np.float64)

    wf_ra = float(best_hp["w_jwt_ransac"]) if best_jwt_on else 0.0
    wf_tri = float(best_hp["w_jwt_triangle"]) if best_jwt_on else 0.0

    preds_b_final = batch_step_b_preds_v14_flat(
        len(val_imp),
        xy_a_val,
        val_blocks["uwb_med"],
        val_blocks["uwb_var"],
        val_blocks["uwb_was_nan"],
        val_blocks["wifi_med"],
        val_blocks["wifi_var"],
        tri_v,
        rsc_v,
        vt_v,
        pack_final.coeff_ab,
        pack_final.use_iso,
        pack_final.iso_x,
        pack_final.iso_y,
        pack_final.iso_n,
        int(fid_f),
        float(best_hp["gate_threshold_m"]),
        wf_ra,
        wf_tri,
        float(best_hp["uwb_variance_inflate"]),
        float(best_hp["huber_f_fusion"]),
        lg.n_u,
        nw,
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

    preds_c_final = batch_step_c_preds_v14_flat(
        len(val_imp),
        xy_a_val,
        val_blocks["uwb_med"],
        val_blocks["uwb_var"],
        val_blocks["uwb_was_nan"],
        val_blocks["wifi_med"],
        val_blocks["wifi_var"],
        tri_v,
        rsc_v,
        vt_v,
        pack_final.coeff_ab,
        pack_final.use_iso,
        pack_final.iso_x,
        pack_final.iso_y,
        pack_final.iso_n,
        int(fid_f),
        float(best_hp["gate_threshold_m"]),
        wf_ra,
        wf_tri,
        float(best_hp["uwb_variance_inflate"]),
        float(best_hp["huber_f_fusion"]),
        float(best_hf_wifi),
        float(best_hp["irls_residual_thresh_m"]),
        int(best_hp["irls_mx"]),
        float(TUKEY_C),
        lg.n_u,
        nw,
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
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("V14 — V12 안심/게이트 + JWT(Isotonic·Score·삼각) Wi‑Fi 가중 | 검증 단일 Forward")
    print("=" * 72)
    print(f"  [Train CV] Step C 평균 RMSE = {best_cv_c:.4f} m")
    print(f"  [Validation] Step A RMSE={rmse_a:.4f} m | Step B RMSE={rmse_b:.4f} m | Step C RMSE={rmse_c:.4f} m")
    print(f"  [Wall-clock] 총 {elapsed:.2f} s")
    print("=" * 72)
    print(
        "[선택 HP] ",
        f"JWT_soft={'ON' if best_jwt_on else 'OFF'}, rank={best_rank}, iso_mask={best_iso}, "
        f"w_JWT_tri={best_hp['w_jwt_triangle']:.3f}, w_JWT_ransac={best_hp['w_jwt_ransac']:.3f}, "
        f"gate={best_hp['gate_threshold_m']:.2f}, calib={best_spec.kind}, feat={best_feat}, huber_f_wifi={best_hf_wifi}",
        flush=True,
    )

    trials_rows: List[Dict[str, Any]] = []
    for tr in study.trials:
        if tr.state != optuna.trial.TrialState.COMPLETE:
            continue
        trials_rows.append({**tr.params, "mean_cv_rmse_step_c_m": tr.value, "trial": tr.number})
    pd.DataFrame(trials_rows).to_csv(out_dir / f"{ART_PREFIX}_optuna_trials.csv", index=False, encoding="utf-8-sig")

    pref = resolve_paths(root)
    summ = {
        "pipeline": "V14 Turbo+JWT",
        "wall_clock_sec": float(elapsed),
        "optuna_n_trials": N_OPTUNA_TRIALS,
        "optuna_n_jobs": int(OPTUNA_N_JOBS),
        "matplotlib_artifacts_saved": bool(save_plots),
        "integrity": (
            "Hyperparameters selected exclusively by minimizing mean K-fold TRAIN CV RMSE (Step C). "
            "Validation ground truth used only once for reporting after full-train calibration refit."
        ),
        "train_kfold_splits": int(n_splits),
        "jwt_triangle": dict(
            base_tol_diff=JWT_BASE_TOL_DIFF,
            rel_tol_diff=JWT_REL_TOL_DIFF,
            base_tol_sum=JWT_BASE_TOL_SUM,
            rel_tol_sum=JWT_REL_TOL_SUM,
            norm=JWT_TRIANGLE_NORM,
            cap=JWT_TRIANGLE_SCORE_CAP,
        ),
        "jwt_score_ransac_residual_threshold_m": float(SCORE_RANSAC_THRESHOLD_M),
        "train_cv_metrics_at_selection": dict(mean_CV_RMSE_step_C_m=float(best_cv_c)),
        "validation_single_forward_metrics": dict(
            step_A_RMSE_m=rmse_a,
            step_B_RMSE_m=rmse_b,
            step_C_RMSE_m=rmse_c,
            step_A_MAE_m=mae_a,
            step_B_MAE_m=mae_b,
            step_C_MAE_m=mae_c,
        ),
        "selected_hyperparameters": {
            **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in best_hp.items()},
            "iso_ap_mask_bits": best_iso,
            "jwt_wifi_soft_enable": bool(best_jwt_on),
            "phase_a_rank": best_rank,
            "wifi_calibration_kind": best_spec.kind,
            "feat_mode": str(best_feat),
            "huber_f_scale_wifi_only_trilateration": float(best_hf_wifi),
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
            "optuna_trials_csv": str(out_dir / f"{ART_PREFIX}_optuna_trials.csv"),
            "wifi_calibration_meta_json": str(out_dir / f"{ART_PREFIX}_wifi_calibration_meta.json"),
        },
    }
    ab_rows: Optional[List[Dict[str, Any]]] = None
    if run_jwt_ablation:
        ab_rows = run_jwt_ablation_report(
            val_imp=val_imp,
            loc=loc,
            lg=lg,
            train_fit_full=train_fit_full,
            val_blocks=val_blocks,
            xy_a_val=xy_a_val,
            best_spec=best_spec,
            best_feat=str(best_feat),
            best_hf_wifi=float(best_hf_wifi),
            best_iso=int(best_iso),
            best_hp=best_hp,
            pair_i=pair_i,
            pair_j=pair_j,
            pair_L=pair_L,
            out_dir=out_dir,
        )
        summ["jwt_ablation_validation"] = ab_rows
        summ["artifacts"]["jwt_ablation_validation_csv"] = str(out_dir / f"{ART_PREFIX}_jwt_ablation_validation.csv")

    meta_pack = dict(
        use_iso_bits=[int(x) for x in pack_final.use_iso.tolist()],
        iso_n_nodes=[int(x) for x in pack_final.iso_n.tolist()],
        coeff_ab_snapshot=pack_final.coeff_ab.round(6).tolist(),
        iso_ap_bitmask=int(best_iso),
    )
    (out_dir / f"{ART_PREFIX}_wifi_calibration_meta.json").write_text(json.dumps(meta_pack, indent=2), encoding="utf-8")

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
        save_cdf_v14(out_dir, dict(A=ea, B=eb, C=ec), dict(A=rmse_a, B=rmse_b, C=rmse_c), dict(A="Step A", B="Step B", C="Step C"))
        save_maps_v14(out_dir, tx, ty, xy_a_val, preds_b_final, preds_c_final, rmse_a, rmse_b, rmse_c)
    else:
        print("[V14] --no-plots: PNG 생략", flush=True)

    print(f"\n저장: {out_dir / f'{ART_PREFIX}_summary.json'}")
    print(f"저장: {out_dir / f'{ART_PREFIX}_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
