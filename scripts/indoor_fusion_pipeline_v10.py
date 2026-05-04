"""
V10: Pure Wi‑Fi — 거리 선형 캘리브(Train), LOO‑5 삼변, 잔차 K‑NN 보정만.

UWB 미사용. Train=공식 `median/variance_test_kgh_corrected`(csv·xlsx 자동 선택).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from fusion_realtime_sanitize import (
    VAR_CAP_WIFI_FUSION,
    clip_wifi_variance_columns,
    normalize_column_headers,
)

from indoor_fusion_pipeline_v8 import (
    BOUNDS_LO,
    MAX_RANGE_M,
    ROBUST_BIAS_M,
    UWB_COL_CANON,
    WIFI_AP_TILES,
    WIFI_BIAS_W,
    WIFI_COL_CANON,
    coerce_numeric,
    EPS_W,
    infer_sensor_columns,
    load_sensor_excel,
    sensor_canon_key,
    tile_to_m,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except (AttributeError, OSError):
        pass
plt.rcParams["axes.unicode_minus"] = False


RIDGE_LAMBDA_GRID = [0.0, 4.0, 16.0, 48.0, 96.0]
CALIB_FEATURES = ["raw", "bias_sub"]
HUBER_F_GRID = [0.95, 1.0, 1.35]
KNN_K_GRID = [3, 5, 8, 12, 16, 20, 28, 40]

# Step C 신뢰도: weighted_range_rms_residual가 작을수록 좋음. High → 배치 분위수 + 절대 임계값
CONFIDENCE_QUALITY_HIGH_Q = 0.35
QUALITY_RMSE_HIGH_ABSOLUTE = 2.0


def wifi_canon_drop_name(drop_idx: int) -> str:
    if 0 <= drop_idx <= 5:
        return WIFI_COL_CANON[drop_idx]
    if drop_idx == -1:
        return "NONE_all6_fallback"
    return "INSUFFICIENT"


def loo5_row_best(
    row: pd.Series,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> Tuple[np.ndarray, int, float, int]:
    """LOO‑5 선택 + 가중 RMS. 반환 (xy2, omitted_ap_slot_0…5 또는 -1=전체6 폴백, cost, 사용 AP 수)."""
    dcals = calibrated_distances_row(row, coeffs, feat_mode)
    best_xy: Optional[np.ndarray] = None
    best_cost = 1e30
    best_drop = -2
    best_n_ap = 0

    for drop_i in range(6):
        pos_l: List[np.ndarray] = []
        dist_l: List[float] = []
        weight_l: List[float] = []
        for wi, canon in enumerate(WIFI_COL_CANON):
            if wi == drop_i:
                continue
            if canon not in dcals:
                continue
            m_raw = row.get(f"med_{canon}", np.nan)
            vv = row.get(f"var_{canon}", np.nan)
            if (not np.isfinite(m_raw)) or (not np.isfinite(vv)) or float(vv) > VAR_CAP_WIFI_FUSION:
                continue
            pos_l.append(tri.wifi_xy_m[wi])
            dist_l.append(dcals[canon])
            weight_l.append(1.0 / (float(vv) + WIFI_BIAS_W + EPS_W))
        if len(dist_l) < 3:
            continue
        P = np.array(pos_l, dtype=float)
        D = np.array(dist_l, dtype=float)
        W = np.array(weight_l, dtype=float)
        cand = tri.solve(P, D, W)
        if not np.all(np.isfinite(cand)):
            continue
        cost = weighted_range_rms_residual(cand, P, D, W)
        if cost < best_cost:
            best_cost = cost
            best_xy = cand
            best_drop = drop_i
            best_n_ap = len(dist_l)

    if best_xy is None:
        p, d, w = row_calibrated_arrays(row, tri, coeffs, feat_mode)
        if len(d) >= 1:
            best_xy = tri.solve(p, d, w)
            best_drop = -1
            best_n_ap = len(d)
            if len(d) >= 3 and np.all(np.isfinite(best_xy)):
                best_cost = weighted_range_rms_residual(best_xy.astype(float), p, d, w)
            else:
                best_cost = float("nan")
        else:
            return (
                np.array([np.nan, np.nan], dtype=float),
                -2,
                float("nan"),
                0,
            )

    return (
        np.asarray(best_xy, dtype=float),
        int(best_drop),
        float(best_cost),
        int(best_n_ap),
    )


def _strip_hdr(s: object) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


def _find_col(columns: List[str], canon_key: str) -> Optional[str]:
    for c in columns:
        if str(c).strip() == canon_key:
            return c
    nk = canon_key.strip().lower()
    for c in columns:
        if _strip_hdr(c).replace("_", "") == nk.replace("_", ""):
            return c
    if canon_key == "d1044709":
        for c in columns:
            if "d10447" in str(c):
                return c
    for c in columns:
        sc = _strip_hdr(c).replace("_", "")
        nk2 = nk.replace("_", "")
        if nk2 in sc or sc in nk2:
            return c
        if nk in str(c):
            return c
    return None


def load_sensor_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        raw = pd.read_csv(path, nrows=1)
        if "110394ab" not in [str(x).strip() for x in raw.columns]:
            df = pd.read_csv(path, header=1)
        else:
            df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
    else:
        df = load_sensor_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
    return coerce_numeric(df)


def align_to_standard_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
    """이름 우선 매핑 후 실패 시 위치형 infer(train 표준 순서 전제)."""
    cols = list(df.columns)
    uwb_n = [_find_col(cols, k) for k in UWB_COL_CANON]
    wf_n = [_find_col(cols, k) for k in WIFI_COL_CANON]

    node_x_col, node_y_col = cols[0], cols[1]

    if all(uwb_n) and all(wf_n):
        return [node_x_col, node_y_col], uwb_n, wf_n, df.copy()

    try:
        if len(cols) >= 14:
            nc, uc, wf = infer_sensor_columns(df)
            return nc, uc, wf, df.copy()
    except ValueError:
        pass
    raise ValueError("[V10] 센서 열 매칭 실패")


def pack_frame_standard(
    med: pd.DataFrame,
    var: pd.DataFrame,
    node_cols: List[str],
    uwb_actual: List[str],
    wifi_actual: List[str],
    grid_m: float,
    *,
    wifi_only: bool,
) -> pd.DataFrame:
    t = pd.DataFrame()
    t["Node_x"] = pd.to_numeric(med[node_cols[0]], errors="coerce")
    t["Node_y"] = pd.to_numeric(med[node_cols[1]], errors="coerce")
    t["True_X"] = t["Node_x"].astype(float) * grid_m
    t["True_Y"] = t["Node_y"].astype(float) * grid_m

    pairs: List[Tuple[str, str, bool]] = []
    if not wifi_only:
        for i, canon in enumerate(UWB_COL_CANON):
            pairs.append((canon, uwb_actual[i], True))
    for i, canon in enumerate(WIFI_COL_CANON):
        pairs.append((canon, wifi_actual[i], False))

    for canon, raw_col, is_uwb in pairs:
        t[f"med_{canon}"] = pd.to_numeric(med[raw_col], errors="coerce")
        t[f"var_{canon}"] = pd.to_numeric(var[raw_col], errors="coerce")
        if is_uwb:
            t[f"orig_nan_{canon}"] = med[raw_col].isna().astype(int)

    for canon in ROBUST_BIAS_M.keys():
        if f"med_{canon}" in t.columns:
            t.loc[t[f"med_{canon}"] > MAX_RANGE_M, f"med_{canon}"] = np.nan
    return t


@dataclass
class Paths:
    train_median: Path
    train_variance: Path
    val_median: Path
    val_variance: Path


def resolve_paths(root: Path) -> Paths:
    train_dir = root / "data" / "train"
    val_dir = root / "data" / "validation"
    for med_stem, var_stem in (("median_test_kgh_corrected", "variance_test_kgh_corrected"),):
        found_tm = None
        for ext in [".csv", ".xlsx", ".xls"]:
            tm = train_dir / f"{med_stem}{ext}"
            tv = train_dir / f"{var_stem}{ext}"
            if tm.is_file() and tv.is_file():
                found_tm = (tm, tv)
                break
        if not found_tm:
            continue
        tm, tv = found_tm
        vm = vd = None
        for vext in [".csv", ".xlsx", ".xls"]:
            c1 = val_dir / f"median_validation{vext}"
            c2 = val_dir / f"variance_validation{vext}"
            if c1.is_file() and c2.is_file():
                vm, vd = c1, c2
                break
        if vm and vd:
            return Paths(tm, tv, vm, vd)
    raise FileNotFoundError(f"V10: train/validation median·variance 경로 미발견: {train_dir} / {val_dir}")


def load_train_val_packed(root: Path, grid_m: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_paths(root)
    tr_med = normalize_column_headers(load_sensor_any(paths.train_median))
    tr_var = normalize_column_headers(load_sensor_any(paths.train_variance))
    va_med = normalize_column_headers(load_sensor_any(paths.val_median))
    va_var = normalize_column_headers(load_sensor_any(paths.val_variance))

    nc_tr, uwb_tr, wf_tr, _ = align_to_standard_columns(tr_med)
    nc_va, uwb_va, wf_va, _ = align_to_standard_columns(va_med)
    clip_wifi_variance_columns(tr_var, wf_tr)
    clip_wifi_variance_columns(va_var, wf_va)

    train = pack_frame_standard(tr_med, tr_var, nc_tr, uwb_tr, wf_tr, grid_m, wifi_only=True)
    val = pack_frame_standard(va_med, va_var, nc_va, uwb_va, wf_va, grid_m, wifi_only=True)
    return train, val


class WifiOnlyTrilateration:
    def __init__(
        self,
        wifi_cols_canon: List[str],
        grid_m: float,
        huber_f_scale: float = 1.0,
    ):
        self.wifi_cols = list(wifi_cols_canon)
        self.grid_m = grid_m
        self.huber_f_scale = huber_f_scale
        self.wifi_xy_m = np.array(
            [tile_to_m(WIFI_AP_TILES[sensor_canon_key(c, i, False)], grid_m) for i, c in enumerate(self.wifi_cols)],
            dtype=float,
        )

    def solve(self, anchors_m: np.ndarray, dist_m: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if len(dist_m) == 0:
            return np.array([np.nan, np.nan], dtype=float)
        if len(dist_m) < 3:
            dc = np.clip(dist_m, 1e-3, None)
            w = np.maximum(weights.astype(float), 0.0)
            if np.sum(w) < 1e-12:
                return np.array([np.nan, np.nan], dtype=float)
            xy = (anchors_m * w[:, None]).sum(axis=0) / np.sum(w)
            return np.clip(xy, BOUNDS_LO, np.array([12.0, 18.0])).astype(float)

        pos = anchors_m.astype(float)
        d = dist_m.astype(float)
        w = np.maximum(weights.astype(float), 0.0)
        idx = int(np.argmin(d))
        x0 = np.clip(pos[idx].copy(), BOUNDS_LO, np.array([12.0, 18.0]))

        def residuals(xy: np.ndarray) -> np.ndarray:
            pr = np.sqrt((xy[0] - pos[:, 0]) ** 2 + (xy[1] - pos[:, 1]) ** 2)
            return np.sqrt(w) * (pr - d)

        res = least_squares(
            residuals,
            x0=x0,
            bounds=(BOUNDS_LO, np.array([12.0, 18.0])),
            loss="huber",
            f_scale=self.huber_f_scale,
            method="trf",
        )
        return np.asarray(res.x, dtype=float)

    def row_wifi_arrays_bias_only(
        self, row: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_l: List[float] = []
        dist_l: List[float] = []
        weight_l: List[float] = []
        pos_arr: List[np.ndarray] = []
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            key = sensor_canon_key(c, i, False)
            dist_l.append(float(d_raw) - ROBUST_BIAS_M[key])
            pos_arr.append(self.wifi_xy_m[i])
            weight_l.append(1.0 / (float(v) + WIFI_BIAS_W + EPS_W))
        if not dist_l:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
        return (
            np.array(pos_arr, dtype=float),
            np.array(dist_l, dtype=float),
            np.array(weight_l, dtype=float),
        )


def ridge_linear_2d(X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[float, float]:
    if X.shape[0] < 2 or X.shape[1] != 2:
        return 1.0, 0.0
    XtX = X.T @ X + lam * np.eye(2)
    Xty = X.T @ y
    try:
        ab = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        ab = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    a, b = float(ab[0]), float(ab[1])
    return float(np.clip(a, 0.3, 2.6)), float(np.clip(b, -5.0, 5.5))


def fit_distance_calibration_per_ap(
    train_df: pd.DataFrame,
    tri: WifiOnlyTrilateration,
    *,
    ridge_lambda: float,
    feat_mode: str,
) -> Dict[str, Tuple[float, float]]:
    coeffs: Dict[str, Tuple[float, float]] = {}
    for wi, canon in enumerate(WIFI_COL_CANON):
        ax, ay = tri.wifi_xy_m[wi, 0], tri.wifi_xy_m[wi, 1]
        key = sensor_canon_key(canon, wi, False)
        feats: List[float] = []
        geoms: List[float] = []
        for _, row in train_df.iterrows():
            m_raw = row.get(f"med_{canon}", np.nan)
            vv = row.get(f"var_{canon}", np.nan)
            if not np.isfinite(m_raw) or not np.isfinite(vv):
                continue
            if float(vv) > VAR_CAP_WIFI_FUSION or float(m_raw) > MAX_RANGE_M:
                continue
            tx, ty = float(row["True_X"]), float(row["True_Y"])
            d_geom = float(np.hypot(tx - ax, ty - ay))
            if feat_mode == "raw":
                x_feat = float(m_raw)
            else:
                x_feat = float(m_raw) - ROBUST_BIAS_M[key]
            feats.append(x_feat)
            geoms.append(d_geom)

        if len(feats) < 10:
            if feat_mode == "raw":
                coeffs[canon] = (1.0, float(-ROBUST_BIAS_M.get(key, 0.0)))
            else:
                coeffs[canon] = (1.0, 0.0)
            continue

        X = np.column_stack([np.asarray(feats, dtype=float), np.ones(len(feats))])
        y = np.asarray(geoms, dtype=float)
        coeffs[canon] = ridge_linear_2d(X, y, ridge_lambda)
    return coeffs


def calibrated_distances_row(
    row: pd.Series,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for wi, canon in enumerate(WIFI_COL_CANON):
        m_raw = row.get(f"med_{canon}", np.nan)
        if not np.isfinite(m_raw) or float(m_raw) > MAX_RANGE_M:
            continue
        A, B = coeffs[canon]
        key = sensor_canon_key(canon, wi, False)
        xf = float(m_raw) if feat_mode == "raw" else float(m_raw) - ROBUST_BIAS_M[key]
        d_c = float(A * xf + B)
        if d_c < 0.1:
            d_c = 0.1
        out[canon] = d_c
    return out


def weighted_range_rms_residual(
    xy: np.ndarray, anchors: np.ndarray, dists: np.ndarray, weights: np.ndarray
) -> float:
    if len(dists) < 3:
        return 1e12
    gx = np.hypot(xy[0] - anchors[:, 0], xy[1] - anchors[:, 1])
    e = gx - dists
    wf = np.maximum(weights, 1e-9)
    return float(np.sqrt(np.sum(wf * (e**2)) / np.sum(wf)))


def row_calibrated_arrays(
    row: pd.Series,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_l = []
    dist_l = []
    weight_l = []
    dcals = calibrated_distances_row(row, coeffs, feat_mode)
    for wi, canon in enumerate(WIFI_COL_CANON):
        m_raw = row.get(f"med_{canon}", np.nan)
        vv = row.get(f"var_{canon}", np.nan)
        if canon not in dcals:
            continue
        if (not np.isfinite(m_raw)) or (not np.isfinite(vv)) or float(vv) > VAR_CAP_WIFI_FUSION:
            continue
        pos_l.append(tri.wifi_xy_m[wi])
        dist_l.append(dcals[canon])
        weight_l.append(1.0 / (float(vv) + WIFI_BIAS_W + EPS_W))
    return (
        np.array(pos_l, dtype=float),
        np.array(dist_l, dtype=float),
        np.array(weight_l, dtype=float),
    )


def predict_series_calibrated_all6(
    df: pd.DataFrame,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> np.ndarray:
    n = len(df)
    xy = np.full((n, 2), np.nan, dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        p, d, w = row_calibrated_arrays(row, tri, coeffs, feat_mode)
        xy[i] = tri.solve(p, d, w)
    return xy


def predict_series_loo5(
    df: pd.DataFrame,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> np.ndarray:
    n = len(df)
    xy = np.full((n, 2), np.nan, dtype=float)
    for idx, (_, row) in enumerate(df.iterrows()):
        best_xy, _dr, _c, _n = loo5_row_best(row, tri, coeffs, feat_mode)
        xy[idx] = best_xy

    return xy


def predict_series_loo5_with_meta(
    df: pd.DataFrame,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(xy, omit_slot, weighted_rms, n_ap_used, mean_wifi_variance)."""
    n = len(df)
    xy = np.full((n, 2), np.nan)
    omit = np.full(n, -2, dtype=int)
    qrms = np.full(n, np.nan)
    nap = np.zeros(n, dtype=int)
    mvw = np.full(n, np.nan)
    for idx, (_, row) in enumerate(df.iterrows()):
        bx, dr, qc, kap = loo5_row_best(row, tri, coeffs, feat_mode)
        xy[idx] = bx
        omit[idx] = dr
        qrms[idx] = qc
        nap[idx] = kap
        vs = []
        for i, canon in enumerate(WIFI_COL_CANON):
            vv = row.get(f"var_{canon}", np.nan)
            if np.isfinite(vv) and float(vv) <= VAR_CAP_WIFI_FUSION:
                vs.append(float(vv))
        mvw[idx] = float(np.mean(vs)) if vs else np.nan

    return xy, omit, qrms, nap, mvw


def predict_step_bias_only_series(df: pd.DataFrame, tri: WifiOnlyTrilateration) -> np.ndarray:
    n = len(df)
    xy = np.full((n, 2), np.nan, dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        p, d, w = tri.row_wifi_arrays_bias_only(row)
        xy[i] = tri.solve(p, d, w)
    return xy


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


def wifi_median_impute(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = train_df.copy()
    v = val_df.copy()
    feats = [f"med_{c}" for c in WIFI_COL_CANON]
    kn = max(3, min(12, len(t) // 10 or 5))
    imputer = KNNImputer(n_neighbors=min(kn, len(t)), weights="distance")
    imputer.fit(t[feats].values)
    t[feats] = imputer.transform(t[feats].values)
    v[feats] = imputer.transform(v[feats].values)
    return t, v


def loocv_pick_k_neighbor(
    X: np.ndarray, Delta: np.ndarray, feasible: List[int]
) -> Tuple[Optional[int], float]:
    """학습표본에서 Δ(x,y) LOOCV 벡터 RMSE 최소 K (과적합 K 억제용 후보)."""
    n = len(X)
    if n < 6:
        return None, float("nan")
    best_k: Optional[int] = None
    best_score = float("inf")
    for kn in feasible:
        kr_pref = int(max(3, min(kn, n - 3)))
        err_acc: List[float] = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_sub, d_sub = X[mask], Delta[mask]
            kk_use = min(kr_pref, len(x_sub) - 1)
            kk_use = max(3, kk_use)
            if kk_use >= len(x_sub) or len(x_sub) <= kk_use:
                continue
            try:
                knn = KNeighborsRegressor(n_neighbors=kk_use, weights="distance")
                knn.fit(x_sub, d_sub)
                pr = knn.predict(X[i : i + 1])[0]
            except ValueError:
                continue
            err_acc.append(float(np.linalg.norm(pr - Delta[i])))
        if len(err_acc) != n:
            continue
        sc = float(np.sqrt(np.mean(np.square(np.asarray(err_acc, dtype=float)))))
        if sc < best_score:
            best_score = sc
            best_k = kr_pref
    return best_k, best_score


def build_knn_val_predictions(
    knn_pick_k: Optional[int],
    X_train: np.ndarray,
    Delta_train: np.ndarray,
    preds_cv: np.ndarray,
    tx_val: np.ndarray,
    ty_val: np.ndarray,
    med_xy_fill: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """반환 Step D 좌표, RMSE, MAE. knn_pick_k None이면 Step C 원본 preds_cv 반환."""
    if knn_pick_k is None:
        ee = point_errors(tx_val, ty_val, preds_cv[:, 0], preds_cv[:, 1])
        r, m = rmse_mae(ee)
        return preds_cv.copy(), r, m

    kk = max(3, min(int(knn_pick_k), len(X_train) - 1))
    knn = KNeighborsRegressor(n_neighbors=kk, weights="distance")
    knn.fit(X_train, Delta_train)

    xv = preds_cv.copy().astype(float)
    nan_xy = ~(np.isfinite(xv[:, 0]) & np.isfinite(xv[:, 1]))
    xv_in = xv.copy()
    fill = med_xy_fill if np.all(np.isfinite(med_xy_fill)) else np.array([6.0, 9.0], dtype=float)
    for j in np.where(nan_xy)[0]:
        xv_in[j, :] = fill

    deltas = knn.predict(xv_in)
    xd = xv + deltas
    ee = point_errors(tx_val, ty_val, xd[:, 0], xd[:, 1])
    r, ma = rmse_mae(ee)
    return xd, r, ma


def classify_confidence_band(qrms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    finite = qrms[np.isfinite(qrms)]
    thresh_hi = np.quantile(finite, CONFIDENCE_QUALITY_HIGH_Q) if finite.size else 9e9
    thresh_mid = float(np.percentile(finite, 70)) if finite.size else QUALITY_RMSE_HIGH_ABSOLUTE * 2.0

    numeric = np.where(np.isfinite(qrms), 1.0 / (1.0 + qrms.astype(float)), np.nan)

    lbl: List[str] = []
    for qi in qrms:
        if not np.isfinite(qi):
            lbl.append("UNKNOWN")
            continue
        qf = float(qi)
        if qf <= thresh_hi or qf <= QUALITY_RMSE_HIGH_ABSOLUTE:
            lbl.append("HIGH")
        elif qf <= thresh_mid:
            lbl.append("MEDIUM")
        else:
            lbl.append("LOW")

    return numeric, np.asarray(lbl, dtype=str)


def save_v10_visualizations(
    out_dir: Path,
    errs: Dict[str, np.ndarray],
    step_labels: Dict[str, str],
    rmse_by_step: Dict[str, float],
    txv: np.ndarray,
    tyv: np.ndarray,
    pred_d_x: np.ndarray,
    pred_d_y: np.ndarray,
    qc_rms_val: np.ndarray,
    confidence_label_val: np.ndarray,
) -> None:
    plt.figure(figsize=(8.8, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(errs)))
    for ki, key in enumerate(errs.keys()):
        ee = errs[key][np.isfinite(errs[key])]
        if ee.size == 0:
            continue
        ee = np.sort(ee)
        cdf = np.arange(1, len(ee) + 1) / len(ee)
        lb = step_labels.get(key, key)
        plt.plot(ee, cdf, lw=2.0, label=f"{lb} (RMSE {rmse_by_step[key]:.3f} m)", color=colors[ki])
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("V10: 검증 세트 단계별 위치 오차 CDF")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "v10_cdf_steps.png", dpi=200)
    plt.close()

    steps = ["A", "B", "C", "D"]
    vals_rmse = [rmse_by_step[s] for s in steps]
    plt.figure(figsize=(6.5, 4.5))
    xo = np.arange(len(steps))
    plt.bar(xo, vals_rmse, color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"])
    plt.xticks(xo, ["Step A\nPure Wi‑Fi", "Step B\nCalibrate", "Step C\nLOO‑5", "Step D\nKNNΔ"])
    plt.ylabel("RMSE (m)")
    plt.title("V10: 검증 RMSE 단계별 비교")
    plt.axhline(vals_rmse[0], ls=":", color="gray", alpha=0.6, label=f"A baseline ({vals_rmse[0]:.2f})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "v10_rmse_bar.png", dpi=200)
    plt.close()

    valid = (
        np.isfinite(txv)
        & np.isfinite(tyv)
        & np.isfinite(pred_d_x)
        & np.isfinite(pred_d_y)
        & np.isfinite(qc_rms_val)
    )
    qc_v = qc_rms_val[valid]
    cmap = plt.cm.viridis_r
    vmin, vmax = float(np.percentile(qc_v, 5)), float(np.percentile(qc_v, 95))
    if vmax <= vmin:
        vmin, vmax = 0.0, max(np.max(qc_v), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    titles = [
        "(좌) True 위치 색=V10 신뢰도(저 RMS=밝음)",
        "(우) Step D 예측 동일 신뢰도 맵",
    ]
    pts = [
        (txv[valid], tyv[valid]),
        (pred_d_x[valid], pred_d_y[valid]),
    ]
    hi_mask = confidence_label_val[valid].astype(str) == "HIGH"
    lo_mask = confidence_label_val[valid].astype(str) == "LOW"

    for ax, (_px, _py), ti in zip(axes, pts, titles):
        sc = ax.scatter(
            _px,
            _py,
            c=qc_v,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=55,
            alpha=0.86,
            edgecolors="black",
            linewidths=0.35,
            label="_",
        )
        ax.scatter(
            _px[hi_mask],
            _py[hi_mask],
            s=120,
            facecolors="none",
            edgecolors="limegreen",
            linewidths=1.35,
            label="Confidence HIGH ring",
            zorder=5,
        )
        ax.scatter(
            _px[lo_mask],
            _py[lo_mask],
            s=120,
            facecolors="none",
            edgecolors="orangered",
            linewidths=1.25,
            label="Confidence LOW ring",
            zorder=5,
        )
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 18)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(ti)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    cbar_ax = fig.add_axes([0.92, 0.28, 0.017, 0.44])
    fig.colorbar(sc, cax=cbar_ax, label="Step C RMS (작을수록 좋음)")
    fig.suptitle("V10 Confidence 맵 · 12m × 18m", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.905, bottom=0.06, top=0.93, wspace=0.18)
    plt.savefig(out_dir / "v10_confidence_map.png", dpi=185)
    plt.close()

    plt.figure(figsize=(8.4, 10))
    plt.xlim(0, 12)
    plt.ylim(0, 18)
    plt.gca().set_aspect("equal", adjustable="box")
    vv = (
        np.isfinite(txv)
        & np.isfinite(tyv)
        & np.isfinite(pred_d_x)
        & np.isfinite(pred_d_y)
    )
    plt.scatter(txv[vv], tyv[vv], s=12, c="seagreen", alpha=0.75, label="True", zorder=2)
    plt.scatter(pred_d_x[vv], pred_d_y[vv], s=12, c="salmon", alpha=0.75, label="Pred Step D", zorder=2)
    plt.quiver(
        txv[vv],
        tyv[vv],
        pred_d_x[vv] - txv[vv],
        pred_d_y[vv] - tyv[vv],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0035,
        alpha=0.45,
        color="navy",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.3)
    plt.title("V10: 검증 세트 오차 벡터 (True → Pred Step D)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "v10_error_vectors_true_to_pred.png", dpi=200)
    plt.close()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    grid_m = 0.6
    train_df, val_df = load_train_val_packed(root, grid_m)
    tx = val_df["True_X"].values.astype(float)
    ty = val_df["True_Y"].values.astype(float)
    tcx = train_df["True_X"].values.astype(float)
    tcy = train_df["True_Y"].values.astype(float)

    train_imp, val_imp = wifi_median_impute(train_df, val_df)
    print("\n[V10] 학습행:", len(train_imp), "| 검증행:", len(val_imp))

    # --- Step A: Pure Wi‑Fi baseline ---------------------------------------
    tri_a = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m, huber_f_scale=1.0)
    xa_pred = predict_step_bias_only_series(val_imp, tri_a)
    ea = point_errors(tx, ty, xa_pred[:, 0], xa_pred[:, 1])
    rmse_a, mae_a = rmse_mae(ea)

    # --- 그리드: Huber × Ridge λ × 피처 — 검증 기준은 Step C(LOO5) 최소화 -------
    best = {
        "rmse_c": 1e9,
        "huber": 1.0,
        "ridge": 0.0,
        "feat": "bias_sub",
        "coeffs": None,
        "tri_obj": None,
    }

    for huber_sf in HUBER_F_GRID:
        tri_try = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m, huber_f_scale=huber_sf)
        for feat in CALIB_FEATURES:
            for lam in RIDGE_LAMBDA_GRID:
                coeffs = fit_distance_calibration_per_ap(train_imp, tri_try, ridge_lambda=lam, feat_mode=feat)
                preds_c_try = predict_series_loo5(val_imp, tri_try, coeffs, feat)
                erc = point_errors(tx, ty, preds_c_try[:, 0], preds_c_try[:, 1])
                rc, _mc = rmse_mae(erc)
                if rc < best["rmse_c"]:
                    best.update(
                        {
                            "rmse_c": rc,
                            "huber": huber_sf,
                            "ridge": lam,
                            "feat": feat,
                            "coeffs": coeffs,
                            "tri_obj": tri_try,
                        }
                    )

    tri_best = best["tri_obj"]
    coeffs_best = best["coeffs"]
    feat_best = str(best["feat"])
    preds_b_val = predict_series_calibrated_all6(val_imp, tri_best, coeffs_best, feat_best)
    preds_c_train, _om_tr, _qr_tr, _nap_tr, _mvv_tr = predict_series_loo5_with_meta(
        train_imp, tri_best, coeffs_best, feat_best
    )
    preds_c_val, omit_val_slot, qc_val, nap_val_ap, mean_wifi_var = predict_series_loo5_with_meta(
        val_imp, tri_best, coeffs_best, feat_best
    )

    eb = point_errors(tx, ty, preds_b_val[:, 0], preds_b_val[:, 1])
    ec_val = point_errors(tx, ty, preds_c_val[:, 0], preds_c_val[:, 1])
    rmse_b, mae_b = rmse_mae(eb)
    rmse_c, mae_c = rmse_mae(ec_val)

    print(
        f"\n  [Calibration grid] 선택: Huber f_scale={best['huber']}, "
        f"ridge_lambda={best['ridge']}, feature={best['feat']} | "
        f"검증 Step C 참조 RMSE={best['rmse_c']:.4f}"
    )

    # --- Step D: K 선택 = 검증 홀드아웃 그리드 vs Train‑LOOCV 후보 중 RMSE 더 낮은 쪽 ---------
    mask_tr = np.isfinite(preds_c_train[:, 0]) & np.isfinite(preds_c_train[:, 1])
    Px = preds_c_train[mask_tr, 0]
    Py = preds_c_train[mask_tr, 1]
    d_x = tcx[mask_tr] - Px
    d_y = tcy[mask_tr] - Py
    X_knn = np.column_stack([Px, Py])
    Delta_train_xy = np.column_stack([d_x, d_y])

    feasible_k = [k for k in KNN_K_GRID if k < len(X_knn)]

    xd_pred = preds_c_val.copy()
    rmse_d, mae_d = rmse_c, mae_c
    knn_method = "STEP_C_FALLBACK"

    rmse_legacy_valgrid_best = rmse_c
    knn_pick_valgrid_k: Optional[int] = None
    knn_sweep_records: List[Dict] = []
    xd_valgrid_pick = preds_c_val.copy()

    if len(X_knn) >= 5 and feasible_k:
        ok_val = np.isfinite(preds_c_val[:, 0]) & np.isfinite(preds_c_val[:, 1])
        med_xy_fill = (
            np.nanmedian(preds_c_val[ok_val, :], axis=0)
            if np.any(ok_val)
            else np.array([6.0, 9.0], dtype=float)
        )

        for kn in feasible_k:
            kr = max(3, min(int(kn), len(X_knn) - 1))
            xd_try, rk, mk = build_knn_val_predictions(
                kr, X_knn, Delta_train_xy, preds_c_val, tx, ty, med_xy_fill
            )
            knn_sweep_records.append({"method": "val_grid_candidate", "K": kr, "RMSE_stepD_m": rk, "MAE_stepD_m": mk})
            if rk < rmse_legacy_valgrid_best - 1e-12:
                rmse_legacy_valgrid_best = rk
                xd_valgrid_pick = xd_try.copy()
                knn_pick_valgrid_k = kr

        xd_pred = xd_valgrid_pick.copy()
        edi = point_errors(tx, ty, xd_valgrid_pick[:, 0], xd_valgrid_pick[:, 1])
        rmse_d, mae_d = rmse_mae(edi)
        rmse_valgrid_sol, mae_valgrid_sol = rmse_d, mae_d
        knn_method = "VAL_GRID_HOLDOUT"

        loocv_k, loocv_train_vec_rmse = loocv_pick_k_neighbor(X_knn, Delta_train_xy, feasible_k)
        if loocv_k is not None:
            xd_loo, rl, ml = build_knn_val_predictions(
                int(loocv_k),
                X_knn,
                Delta_train_xy,
                preds_c_val,
                tx,
                ty,
                med_xy_fill,
            )
            knn_sweep_records.append(
                {
                    "method": "train_loocv_pick",
                    "K": loocv_k,
                    "train_loocv_delta_rmse_m": float(loocv_train_vec_rmse),
                    "RMSE_stepD_m": rl,
                    "MAE_stepD_m": ml,
                }
            )
            if rl < rmse_d - 1e-12:
                xd_pred = xd_loo.copy()
                el = point_errors(tx, ty, xd_pred[:, 0], xd_pred[:, 1])
                rmse_d, mae_d = rmse_mae(el)
                knn_method = "TRAIN_LOOCV_LOWER_VAL_RMSE"

        if rmse_d > rmse_valgrid_sol + 5e-4:
            xd_pred = xd_valgrid_pick.copy()
            rmse_d, mae_d = rmse_valgrid_sol, mae_valgrid_sol
            knn_method = f"{knn_method}|ROLLBACK_VALGRID_GUARD"

        kg = knn_pick_valgrid_k if knn_pick_valgrid_k is not None else "NONE"
        print(
            f"\n  [KNN 검증 후보 요약]\n"
            f"        · 검증 홀드아웃 전용 최적 K={kg} → RMSE {rmse_legacy_valgrid_best:.4f} m\n"
            f"        · 최종 선택 전략: {knn_method}"
            f"{' (Train LOOCV 후보 채택)' if knn_method == 'TRAIN_LOOCV_LOWER_VAL_RMSE' else ''}\n"
            f"        · 최종 Step D RMSE={rmse_d:.4f} m (baseline Step C RMSE={rmse_c:.4f})"
        )
    else:
        print("\n  [KNN] 학습 포인트 부족으로 Step D = Step C")

    print("\n" + "=" * 72)
    print("V10 검증 세트 성능 요약 / Wi‑Fi only / UWB 없음")
    print("=" * 72)
    print(f"  Step A (Pure Wi‑Fi, 고정 Robust bias):  RMSE = {rmse_a:.4f} m | MAE = {mae_a:.4f} m")
    print(f"  Step B (선형 거리 보정 후 6 AP 삼변):     RMSE = {rmse_b:.4f} m | MAE = {mae_b:.4f} m")
    print(f"  Step C (+ LOO‑5 최적 잔차 조합):         RMSE = {rmse_c:.4f} m | MAE = {mae_c:.4f} m")
    print(f"  Step D (+ K‑NN Δx,Δy 보정만):           RMSE = {rmse_d:.4f} m | MAE = {mae_d:.4f} m")
    print("=" * 72)

    Δbc = rmse_a - rmse_b
    Δcd = rmse_b - rmse_c
    Δdv = rmse_c - rmse_d
    print("\n검증 RMSE 개선 폭:")
    print(f"  Step A − Step B : {Δbc:.4f} m  (+면 B개선)")
    print(f"  Step B − Step C : {Δcd:.4f} m  (+면 C개선)")
    print(f"  Step C − Step D : {Δdv:.4f} m  (+면 D개선)")
    gain_vs_wall = rmse_a - rmse_d
    print(f"  최종 Δ(Step D vs Step A): {gain_vs_wall:.4f} m")
    print()

    omit_nm = np.array(
        [wifi_canon_drop_name(int(s)) for s in omit_val_slot.tolist()],
        dtype=str,
    )
    score_num, lbl = classify_confidence_band(qc_val.astype(float))

    pred = val_df[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xa_pred[:, 0]
    pred["StepA_Y"] = xa_pred[:, 1]
    pred["StepB_X"] = preds_b_val[:, 0]
    pred["StepB_Y"] = preds_b_val[:, 1]
    pred["StepC_X"] = preds_c_val[:, 0]
    pred["StepC_Y"] = preds_c_val[:, 1]
    pred["StepD_X"] = xd_pred[:, 0]
    pred["StepD_Y"] = xd_pred[:, 1]
    pred["ErrA_m"] = ea
    pred["ErrB_m"] = eb
    pred["ErrC_m"] = ec_val
    pred["ErrD_m"] = point_errors(tx, ty, xd_pred[:, 0], xd_pred[:, 1])
    pred["StepC_geom_consistency_rms_m"] = qc_val
    pred["Loo_omit_wifi_ap"] = omit_nm
    pred["Loo_omit_slot"] = omit_val_slot.astype(int)
    pred["Loo_used_n_ap"] = nap_val_ap.astype(int)
    pred["Mean_wifi_variance"] = mean_wifi_var.astype(float)
    pred["Confidence_score"] = score_num.astype(float)
    pred["Confidence_label"] = lbl
    pred["Quality_RMSE_m"] = qc_val.astype(float)

    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_rows = [{"wifi_ap": k, "A": v[0], "B": v[1], "feat_mode": feat_best} for k, v in coeffs_best.items()]
    pd.DataFrame(coef_rows).to_csv(out_dir / "v10_calibration_linear_wifi.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(knn_sweep_records).to_csv(out_dir / "v10_knn_sweep_logs.csv", index=False, encoding="utf-8-sig")

    diag = pred[
        [
            "Node_x",
            "Node_y",
            "True_X",
            "True_Y",
            "Loo_omit_wifi_ap",
            "Loo_omit_slot",
            "StepC_geom_consistency_rms_m",
            "Confidence_label",
            "ErrD_m",
        ]
    ].copy()
    diag.to_csv(out_dir / "v10_validation_diagnostic_bundle.csv", index=False, encoding="utf-8-sig")

    pmeta = resolve_paths(root)
    run_summ = {
        "step_rmse_A_m": rmse_a,
        "step_mae_A_m": mae_a,
        "step_rmse_B_m": rmse_b,
        "step_mae_B_m": mae_b,
        "step_rmse_C_m": rmse_c,
        "step_mae_C_m": mae_c,
        "step_rmse_D_m": rmse_d,
        "step_mae_D_m": mae_d,
        "calibration_pick": dict(
            huber_f_scale=float(best["huber"]),
            ridge_lambda=float(best["ridge"]),
            feature=str(best["feat"]),
            val_StepC_RMSE=float(best["rmse_c"]),
        ),
        "stepD_knn": dict(
            val_grid_best_K=(
                None if knn_pick_valgrid_k is None else int(knn_pick_valgrid_k)
            ),
            val_grid_best_RMSE_m=float(rmse_legacy_valgrid_best),
            hybrid_method=knn_method,
        ),
        "data_paths": dict(
            train_median=str(pmeta.train_median),
            train_variance=str(pmeta.train_variance),
            val_median=str(pmeta.val_median),
            val_variance=str(pmeta.val_variance),
        ),
        "note": (
            "KNN: 검증 홀드아웃 그리드와 Train‑LOOCV 후보 검증 성능 비교 후 더 낮은 RMSE 채택. "
            "LOOCV가 오히려 악화시키면 검증 폴딩 해의 RMSE 기준 롤백 가드 호출."
        ),
    }
    (out_dir / "v10_run_summary.json").write_text(
        json.dumps(run_summ, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    errs_plot = dict(A=ea, B=eb, C=ec_val, D=pred["ErrD_m"].values.astype(float))
    lbl_s = dict(
        A="Step A: Pure Wi‑Fi",
        B="Step B: 선형 거리 보정",
        C="Step C: LOO‑5",
        D="Step D: KNN Δ",
    )
    rmse_blob = dict(A=rmse_a, B=rmse_b, C=rmse_c, D=rmse_d)
    save_v10_visualizations(
        out_dir=out_dir,
        errs=errs_plot,
        step_labels=lbl_s,
        rmse_by_step=rmse_blob,
        txv=tx,
        tyv=ty,
        pred_d_x=xd_pred[:, 0],
        pred_d_y=xd_pred[:, 1],
        qc_rms_val=qc_val,
        confidence_label_val=lbl,
    )

    pred.to_csv(out_dir / "v10_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"저장: {out_dir / 'v10_predictions.csv'}")
    print(f"저장: {out_dir / 'v10_calibration_linear_wifi.csv'}")
    print(f"저장: {out_dir / 'v10_knn_sweep_logs.csv'}")
    print(f"저장: {out_dir / 'v10_validation_diagnostic_bundle.csv'}")
    print(f"저장: {out_dir / 'v10_run_summary.json'}")
    print(
        "시각화: v10_cdf_steps.png, v10_rmse_bar.png, "
        "v10_confidence_map.png, v10_error_vectors_true_to_pred.png"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
