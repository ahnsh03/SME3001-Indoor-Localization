"""
V7: V6 Step-C 계열(오프셋·Var cap·Bounds·UWB 0.5 / Wi-Fi 2.0)을 베이스로
Step B(Log-Var 가중), Step C(HBB alpha 그리드), Step D(거리 감쇠 beta 그리드)를 순차 적용.

검증 RMSE 기준으로 alpha·beta를 선택하고, 상세 CSV·CDF·Kill 막대·오차 벡터 맵을 저장.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
from scipy.optimize import least_squares
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fusion_realtime_sanitize import (
    HARDWARE_CALIB_BIAS_M,
    VAR_CAP_UWB_FUSION,
    VAR_CAP_WIFI_FUSION,
    standardize_loaded_median_variance,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
)

# --- 앵커 (V6 동일) ---
UWB_ANCHOR_TILES: Dict[str, Tuple[float, float]] = {
    "110394ab": (1, 4),
    "e63ce2f": (20, 7),
    "8e610981": (5, 15),
    "d10485af": (4, 27),
    "d1044709": (15, 14),
    "4e610206": (14, 24),
}
WIFI_AP_TILES: Dict[str, Tuple[float, float]] = {
    "SW_11": (1, 4),
    "SW_first_team": (20, 7),
    "볼링공": (5, 15),
    "SW_4": (4, 27),
    "SW_6": (15, 14),
    "SW_5": (14, 24),
}

UWB_COL_CANON = ["110394ab", "e63ce2f", "8e610981", "d10485af", "d1044709", "4e610206"]
WIFI_COL_CANON = ["SW_11", "SW_first_team", "볼링공", "SW_4", "SW_6", "SW_5"]

ROBUST_BIAS_M: Dict[str, float] = dict(HARDWARE_CALIB_BIAS_M)

ALL_SENSOR_KEYS: List[str] = UWB_COL_CANON + WIFI_COL_CANON

BOUNDS_LO = np.array([0.0, 0.0], dtype=float)
BOUNDS_HI = np.array([12.0, 18.0], dtype=float)
MAX_RANGE_M = 22.0
EPS_W = 1e-4

UWB_BIAS_W = 0.5
WIFI_BIAS_W = 2.0

ALPHA_GRID = [1.2, 1.5, 1.8, 2.0, 2.5]
BETA_GRID = [0.01, 0.03, 0.05, 0.1, 0.2]


def load_sensor_excel(path: Path) -> pd.DataFrame:
    peek = pd.read_excel(path, header=None, nrows=1)
    first_cell = str(peek.iloc[0, 0]).strip().lower()
    if first_cell in ("node_x", "nodex"):
        raw = pd.read_excel(path, header=None)
        header_row = raw.iloc[0].astype(str).str.strip().tolist()
        df = raw.iloc[1:].copy().reset_index(drop=True)
        df.columns = header_row
    else:
        raw = pd.read_excel(path)
        if "Node_x" in raw.columns:
            df = raw.copy()
        else:
            header_row = raw.iloc[0].astype(str).str.strip().tolist()
            df = raw.iloc[1:].copy().reset_index(drop=True)
            df.columns = header_row
    for c in list(df.columns):
        if "d1044709" in str(c):
            df = df.rename(columns={c: "d1044709"})
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def infer_sensor_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    cols = df.columns.tolist()
    if len(cols) < 14:
        raise ValueError("최소 14열(Node_x, Node_y + 센서 12개)이 필요합니다.")
    return cols[:2], cols[2:8], cols[8:14]


def tile_to_m(xy: Tuple[float, float], grid_m: float) -> Tuple[float, float]:
    return xy[0] * grid_m, xy[1] * grid_m


def sensor_canon_key(col: str, idx: int, is_uwb: bool) -> str:
    if is_uwb:
        if col in UWB_ANCHOR_TILES:
            return col
        return UWB_COL_CANON[idx]
    if col in WIFI_AP_TILES:
        return col
    return WIFI_COL_CANON[idx]


@dataclass
class Config:
    grid_size_m: float = 0.6
    trilateration_loss: str = "huber"
    huber_f_scale: float = 1.0
    uwb_imputer_neighbors: int = 5


class FusionLocalizerV7:
    def __init__(self, config: Config):
        self.config = config
        self.uwb_cols: List[str] = []
        self.wifi_cols: List[str] = []
        self.uwb_imputer: KNNImputer | None = None
        self.uwb_pos_order_m: List[Tuple[float, float]] = []
        self.wifi_pos_order_m: List[Tuple[float, float]] = []

    def _assert_files(self, paths: List[Path]) -> None:
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")

    def _resolve_sensor_position_m(self, col: str, idx: int, is_uwb: bool) -> Tuple[float, float]:
        g = self.config.grid_size_m
        if is_uwb and col in UWB_ANCHOR_TILES:
            return tile_to_m(UWB_ANCHOR_TILES[col], g)
        if (not is_uwb) and col in WIFI_AP_TILES:
            return tile_to_m(WIFI_AP_TILES[col], g)
        return (self.uwb_pos_order_m if is_uwb else self.wifi_pos_order_m)[idx]

    def load_datasets(
        self,
        train_median_path: Path,
        train_variance_path: Path,
        val_median_path: Path,
        val_variance_path: Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._assert_files([train_median_path, train_variance_path, val_median_path, val_variance_path])

        tr_med, tr_var, node_cols, self.uwb_cols, self.wifi_cols = standardize_loaded_median_variance(
            train_median_path, train_variance_path, coerce_numeric, infer_sensor_columns
        )
        va_med, va_var, _nc2, uwb_v2, wifi_v2 = standardize_loaded_median_variance(
            val_median_path, val_variance_path, coerce_numeric, infer_sensor_columns
        )
        if self.uwb_cols != uwb_v2 or self.wifi_cols != wifi_v2:
            raise ValueError("validation 열이 train 과 불일치")

        self.uwb_pos_order_m = [
            tile_to_m(t, self.config.grid_size_m)
            for t in [(1, 4), (20, 7), (5, 15), (4, 27), (15, 14), (14, 24)]
        ]
        self.wifi_pos_order_m = [
            tile_to_m(t, self.config.grid_size_m)
            for t in [(20, 7), (4, 27), (14, 24), (15, 14), (1, 4), (5, 15)]
        ]

        g = self.config.grid_size_m
        sensor_cols = self.uwb_cols + self.wifi_cols

        def pack(med: pd.DataFrame, var: pd.DataFrame) -> pd.DataFrame:
            t = pd.DataFrame()
            t["Node_x"] = med[node_cols[0]]
            t["Node_y"] = med[node_cols[1]]
            t["True_X"] = t["Node_x"].astype(float) * g
            t["True_Y"] = t["Node_y"].astype(float) * g
            for c in self.uwb_cols:
                t[f"med_{c}"] = med[c]
                t[f"var_{c}"] = var[c]
                t[f"orig_nan_{c}"] = med[c].isna().astype(int)
            for c in self.wifi_cols:
                t[f"med_{c}"] = med[c]
                t[f"var_{c}"] = var[c]
            return t

        train = pack(tr_med, tr_var)
        val = pack(va_med, va_var)
        for c in sensor_cols:
            for df in (train, val):
                df.loc[df[f"med_{c}"] > MAX_RANGE_M, f"med_{c}"] = np.nan
        return train, val

    def fit_uwb_imputer(self, train_df: pd.DataFrame) -> None:
        feat_cols = [f"med_{c}" for c in self.wifi_cols] + [f"med_{c}" for c in self.uwb_cols]
        self.uwb_imputer = KNNImputer(n_neighbors=self.config.uwb_imputer_neighbors, weights="distance")
        self.uwb_imputer.fit(train_df[feat_cols])

    def add_imputed_uwb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.uwb_imputer is None:
            raise RuntimeError("UWB imputer not fitted")
        out = df.copy()
        feat_cols = [f"med_{c}" for c in self.wifi_cols] + [f"med_{c}" for c in self.uwb_cols]
        imp_df = pd.DataFrame(self.uwb_imputer.transform(out[feat_cols]), columns=feat_cols, index=out.index)
        for c in self.uwb_cols:
            out[f"med_imp_{c}"] = imp_df[f"med_{c}"]
        return out

    def _smart_initial_guess(self, pos: np.ndarray, dist: np.ndarray) -> np.ndarray:
        if len(dist) == 0:
            return np.array([6.0, 9.0], dtype=float)
        idx = int(np.argmin(dist))
        return np.array(pos[idx], dtype=float)

    def _solve_trilat(
        self,
        pos: np.ndarray,
        dist: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        if len(dist) == 0:
            return np.array([np.nan, np.nan], dtype=float)
        if len(dist) < 3:
            d = np.clip(dist, 1e-3, None)
            w = np.maximum(weights, 0.0)
            if w.sum() < 1e-12:
                return np.array([np.nan, np.nan], dtype=float)
            xy = (pos * w[:, None]).sum(axis=0) / w.sum()
            return np.clip(xy, BOUNDS_LO, BOUNDS_HI).astype(float)

        x0 = self._smart_initial_guess(pos, dist)
        x0 = np.clip(x0, BOUNDS_LO, BOUNDS_HI)

        def residuals(xy: np.ndarray) -> np.ndarray:
            pred = np.sqrt((xy[0] - pos[:, 0]) ** 2 + (xy[1] - pos[:, 1]) ** 2)
            return np.sqrt(np.maximum(weights, 0.0)) * (pred - dist)

        res = least_squares(
            residuals,
            x0=x0,
            bounds=(BOUNDS_LO, BOUNDS_HI),
            loss=self.config.trilateration_loss,
            f_scale=self.config.huber_f_scale,
            method="trf",
        )
        return np.array(res.x, dtype=float)

    def _count_var_cap_kills(self, row: pd.Series, kills: DefaultDict[str, DefaultDict[str, int]]) -> None:
        for i, c in enumerate(self.uwb_cols):
            v = row.get(f"var_{c}", np.nan)
            if np.isfinite(v) and float(v) > VAR_CAP_UWB_FUSION:
                key = sensor_canon_key(c, i, True)
                kills[key]["Var > fusion cap"] += 1
        for i, c in enumerate(self.wifi_cols):
            v = row.get(f"var_{c}", np.nan)
            if np.isfinite(v) and float(v) > VAR_CAP_WIFI_FUSION:
                key = sensor_canon_key(c, i, False)
                kills[key]["Var > fusion cap"] += 1

    def _row_arrays_step_a(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """선형 분산 가중 + 오프셋 + Var cap + (삼변 입력만)."""
        pos_l: List[Tuple[float, float]] = []
        dist_l: List[float] = []
        w_l: List[float] = []
        for i, c in enumerate(self.uwb_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan or (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_UWB_FUSION:
                continue
            key = sensor_canon_key(c, i, True)
            d_use = float(d_raw) - ROBUST_BIAS_M[key]
            pos_l.append(self._resolve_sensor_position_m(c, i, True))
            dist_l.append(d_use)
            w_l.append(1.0 / (float(v) + UWB_BIAS_W + EPS_W))
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            key = sensor_canon_key(c, i, False)
            d_use = float(d_raw) - ROBUST_BIAS_M[key]
            pos_l.append(self._resolve_sensor_position_m(c, i, False))
            dist_l.append(d_use)
            w_l.append(1.0 / (float(v) + WIFI_BIAS_W + EPS_W))
        return (
            np.asarray(pos_l, dtype=float),
            np.asarray(dist_l, dtype=float),
            np.asarray(w_l, dtype=float),
        )

    def _row_arrays_wifi_only(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step A와 동일 선형 가중·오프셋·Var cap이나 Wi-Fi 6개만."""
        pos_l: List[Tuple[float, float]] = []
        dist_l: List[float] = []
        w_l: List[float] = []
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            key = sensor_canon_key(c, i, False)
            d_use = float(d_raw) - ROBUST_BIAS_M[key]
            pos_l.append(self._resolve_sensor_position_m(c, i, False))
            dist_l.append(d_use)
            w_l.append(1.0 / (float(v) + WIFI_BIAS_W + EPS_W))
        return (
            np.asarray(pos_l, dtype=float),
            np.asarray(dist_l, dtype=float),
            np.asarray(w_l, dtype=float),
        )

    def predict_wifi_only(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        xs: List[float] = []
        ys: List[float] = []
        for _, row in df.iterrows():
            p, d, w = self._row_arrays_wifi_only(row)
            xy = self._solve_trilat(p, d, w)
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    def _row_uwb_wifi_calib_for_pair(self, row: pd.Series, idx: int) -> Tuple[float, float, float, float, bool, bool]:
        """동일 인덱스 co-site UWB/Wi-Fi의 보정 거리·분산. (UWB 사용 가능, Wi-Fi 사용 가능)"""
        c_u = self.uwb_cols[idx]
        c_w = self.wifi_cols[idx]
        vu = row.get(f"var_{c_u}", np.nan)
        vw = row.get(f"var_{c_w}", np.nan)
        du_raw = row.get(f"med_{c_u}", np.nan)
        dw_raw = row.get(f"med_{c_w}", np.nan)
        was_nan = row.get(f"orig_nan_{c_u}", 1) == 1
        u_ok = (
            (not was_nan)
            and np.isfinite(du_raw)
            and np.isfinite(vu)
            and float(vu) <= VAR_CAP_UWB_FUSION
        )
        w_ok = np.isfinite(dw_raw) and np.isfinite(vw) and float(vw) <= VAR_CAP_WIFI_FUSION
        if not u_ok:
            return np.nan, np.nan, np.nan, np.nan, False, w_ok
        if not w_ok:
            ku = sensor_canon_key(c_u, idx, True)
            du = float(du_raw) - ROBUST_BIAS_M[ku]
            return du, np.nan, float(vu), np.nan, True, False
        ku = sensor_canon_key(c_u, idx, True)
        kw = sensor_canon_key(c_w, idx, False)
        du = float(du_raw) - ROBUST_BIAS_M[ku]
        dw = float(dw_raw) - ROBUST_BIAS_M[kw]
        return du, dw, float(vu), float(vw), True, True

    def _hbb_mask(self, row: pd.Series, alpha: float) -> List[bool]:
        """인덱스별 UWB를 HBB로 제외할지 (True=제외)."""
        kill_uwb = [False] * 6
        for idx in range(6):
            du, dw, _vu, _vw, u_ok, w_ok = self._row_uwb_wifi_calib_for_pair(row, idx)
            if u_ok and w_ok and np.isfinite(du) and np.isfinite(dw) and dw > 1e-6:
                if du > alpha * dw:
                    kill_uwb[idx] = True
        return kill_uwb

    def _row_arrays_step_b(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log1p 분산 가중."""
        pos_l: List[Tuple[float, float]] = []
        dist_l: List[float] = []
        w_l: List[float] = []
        for i, c in enumerate(self.uwb_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan or (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_UWB_FUSION:
                continue
            key = sensor_canon_key(c, i, True)
            d_use = float(d_raw) - ROBUST_BIAS_M[key]
            pos_l.append(self._resolve_sensor_position_m(c, i, True))
            dist_l.append(d_use)
            w_l.append(1.0 / (np.log1p(float(v)) + UWB_BIAS_W + EPS_W))
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            key = sensor_canon_key(c, i, False)
            d_use = float(d_raw) - ROBUST_BIAS_M[key]
            pos_l.append(self._resolve_sensor_position_m(c, i, False))
            dist_l.append(d_use)
            w_l.append(1.0 / (np.log1p(float(v)) + WIFI_BIAS_W + EPS_W))
        return (
            np.asarray(pos_l, dtype=float),
            np.asarray(dist_l, dtype=float),
            np.asarray(w_l, dtype=float),
        )

    def _row_arrays_step_c(
        self, row: pd.Series, alpha: float, hbb_kills: DefaultDict[str, int] | None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step B 가중 + HBB(UWB만)."""
        hbb = self._hbb_mask(row, alpha)
        pos_l: List[Tuple[float, float]] = []
        dist_l: List[float] = []
        w_l: List[float] = []
        for i, c in enumerate(self.uwb_cols):
            if hbb[i]:
                if hbb_kills is not None:
                    hbb_kills[sensor_canon_key(c, i, True)] += 1
                continue
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan or (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_UWB_FUSION:
                continue
            key = sensor_canon_key(c, i, True)
            d_use = float(d_raw) - ROBUST_BIAS_M[key]
            pos_l.append(self._resolve_sensor_position_m(c, i, True))
            dist_l.append(d_use)
            w_l.append(1.0 / (np.log1p(float(v)) + UWB_BIAS_W + EPS_W))
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            key = sensor_canon_key(c, i, False)
            d_use = float(d_raw) - ROBUST_BIAS_M[key]
            pos_l.append(self._resolve_sensor_position_m(c, i, False))
            dist_l.append(d_use)
            w_l.append(1.0 / (np.log1p(float(v)) + WIFI_BIAS_W + EPS_W))
        return (
            np.asarray(pos_l, dtype=float),
            np.asarray(dist_l, dtype=float),
            np.asarray(w_l, dtype=float),
        )

    def _solve_step_d_row(self, pos: np.ndarray, dist: np.ndarray, w_prev: np.ndarray, beta: float) -> np.ndarray:
        if len(dist) == 0:
            return np.array([np.nan, np.nan], dtype=float)
        _xy_temp = self._solve_trilat(pos, dist, w_prev)
        w_final = w_prev * np.exp(-beta * dist)
        return self._solve_trilat(pos, dist, w_final)

    def predict_frame(
        self,
        df: pd.DataFrame,
        *,
        step: str,
        alpha: float = 2.0,
        beta: float = 0.05,
        hbb_kills: DefaultDict[str, int] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        xs: List[float] = []
        ys: List[float] = []
        for _, row in df.iterrows():
            if step == "A":
                p, d, w = self._row_arrays_step_a(row)
                xy = self._solve_trilat(p, d, w)
            elif step == "B":
                p, d, w = self._row_arrays_step_b(row)
                xy = self._solve_trilat(p, d, w)
            elif step == "C":
                p, d, w = self._row_arrays_step_c(row, alpha, hbb_kills)
                xy = self._solve_trilat(p, d, w)
            else:
                p, d, w = self._row_arrays_step_c(row, alpha, hbb_kills)
                xy = self._solve_step_d_row(p, d, w, beta)
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    @staticmethod
    def point_errors(tx: np.ndarray, ty: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return np.sqrt((px - tx) ** 2 + (py - ty) ** 2)

    @staticmethod
    def rmse_mae(err: np.ndarray) -> Tuple[float, float]:
        err = err[np.isfinite(err)]
        if err.size == 0:
            return float("nan"), float("nan")
        rmse = float(np.sqrt(mean_squared_error(np.zeros_like(err), err)))
        mae = float(mean_absolute_error(np.zeros_like(err), err))
        return rmse, mae


def _configure_stdio_utf8() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass


def main() -> int:
    _configure_stdio_utf8()
    cfg = Config()
    loc = FusionLocalizerV7(cfg)
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "train"
    val_dir = root / "data" / "validation"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_med_p, train_var_p = resolve_train_kgh_corrected_paths(train_dir)
    val_med_p, val_var_p = resolve_validation_paths(val_dir)
    train, val = loc.load_datasets(
        train_median_path=train_med_p,
        train_variance_path=train_var_p,
        val_median_path=val_med_p,
        val_variance_path=val_var_p,
    )
    print(f"  데이터 행 수: Train(all rows)={len(train)} | 검증 len={len(val)}")
    loc.fit_uwb_imputer(train)
    val_p = loc.add_imputed_uwb_features(val)

    tx = val_p["True_X"].values
    ty = val_p["True_Y"].values

    xw, yw = loc.predict_wifi_only(val_p)
    err_wf = loc.point_errors(tx, ty, xw, yw)
    rwf, mwf = loc.rmse_mae(err_wf)
    print("\n" + "=" * 72)
    print("V7 Validation — Wi-Fi only (Step A와 동일 전처리·가중·bounds, UWB 미사용)")
    print("=" * 72)
    print(f"  Wi-Fi-only RMSE = {rwf:.4f} m  |  MAE = {mwf:.4f} m")
    print("=" * 72 + "\n")

    xa, ya = loc.predict_frame(val_p, step="A")
    xb, yb = loc.predict_frame(val_p, step="B")

    # --- Step C: alpha 그리드 ---
    trace_rows: List[Dict] = []
    best_alpha = ALPHA_GRID[0]
    best_rmse_c = 1e9
    for a in ALPHA_GRID:
        xc, yc = loc.predict_frame(val_p, step="C", alpha=a, beta=0.0)
        err = loc.point_errors(tx, ty, xc, yc)
        r, m = loc.rmse_mae(err)
        trace_rows.append({"phase": "C_alpha_sweep", "alpha": a, "beta": np.nan, "RMSE_m": r, "MAE_m": m})
        if r < best_rmse_c:
            best_rmse_c = r
            best_alpha = a

    xc_best, yc_best = loc.predict_frame(val_p, step="C", alpha=best_alpha, beta=0.0)

    # --- Step D: beta 그리드 (alpha = best_alpha 고정) ---
    best_beta = BETA_GRID[0]
    best_rmse_d = 1e9
    for b in BETA_GRID:
        xd, yd = loc.predict_frame(val_p, step="D", alpha=best_alpha, beta=b)
        err = loc.point_errors(tx, ty, xd, yd)
        r, m = loc.rmse_mae(err)
        trace_rows.append({"phase": "D_beta_sweep", "alpha": best_alpha, "beta": b, "RMSE_m": r, "MAE_m": m})
        if r < best_rmse_d:
            best_rmse_d = r
            best_beta = b

    # 전체 (alpha, beta) 조합 탐색 로그 (상호작용 관찰용)
    joint_best_a, joint_best_b = best_alpha, best_beta
    joint_best_rmse = 1e9
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            xd, yd = loc.predict_frame(val_p, step="D", alpha=a, beta=b)
            err = loc.point_errors(tx, ty, xd, yd)
            r, m = loc.rmse_mae(err)
            trace_rows.append({"phase": "D_full_grid", "alpha": a, "beta": b, "RMSE_m": r, "MAE_m": m})
            if r < joint_best_rmse:
                joint_best_rmse = r
                joint_best_a, joint_best_b = a, b

    xd_best, yd_best = loc.predict_frame(val_p, step="D", alpha=best_alpha, beta=best_beta)

    err_a = loc.point_errors(tx, ty, xa, ya)
    err_b = loc.point_errors(tx, ty, xb, yb)
    err_c = loc.point_errors(tx, ty, xc_best, yc_best)
    err_d = loc.point_errors(tx, ty, xd_best, yd_best)

    print("\n" + "=" * 72)
    print("V7 Validation — Step별 RMSE / MAE (m)")
    print("=" * 72)
    for label, err in [("A (V6-style linear W + offset + var cap + bounds)", err_a), ("B (+ Log1p variance weight)", err_b), ("C (+ HBB, optimal alpha)", err_c), ("D (+ distance decay, optimal beta)", err_d)]:
        step_letter = label[0]
        r, m = loc.rmse_mae(err)
        print(f"  Step {step_letter}: RMSE = {r:.4f} m  |  MAE = {m:.4f} m  |  {label}")
    print(f"\n  [Step C] 선택 alpha = {best_alpha}  (alpha 그리드에서 검증 RMSE 최소)")
    print(f"  [Step D] 선택 beta = {best_beta}  (beta 그리드, alpha 고정 시 검증 RMSE 최소)")
    print(
        f"  [참고] D_full_grid 전역 최소: alpha={joint_best_a}, beta={joint_best_b}, RMSE={joint_best_rmse:.4f} m"
    )
    if joint_best_a != best_alpha or joint_best_b != best_beta:
        print(
            "        → 순차 선택(alpha 먼저, 그 다음 beta)과 전역 그리드 최소가 다릅니다. "
            "보고서용으로는 v7_grid_search_trace.csv의 D_full_grid를 비교하세요."
        )
    print("=" * 72 + "\n")

    # --- Kill log (최종 alpha 기준, 검증 행 순회) ---
    kills: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    hbb_only: DefaultDict[str, int] = defaultdict(int)
    for _, row in val_p.iterrows():
        loc._count_var_cap_kills(row, kills)
        loc._row_arrays_step_c(row, best_alpha, hbb_only)
    kill_rows = []
    for key in ALL_SENSOR_KEYS:
        kill_rows.append(
            {
                "sensor": key,
                "Var_cap_exceeded_count": kills[key]["Var > fusion cap"],
                "HBB_cross_check_count": hbb_only[key],
            }
        )
    kill_df = pd.DataFrame(kill_rows)

    pred_df = val_p[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred_df["StepA_X"] = xa
    pred_df["StepA_Y"] = ya
    pred_df["StepB_X"] = xb
    pred_df["StepB_Y"] = yb
    pred_df["StepC_X"] = xc_best
    pred_df["StepC_Y"] = yc_best
    pred_df["StepD_X"] = xd_best
    pred_df["StepD_Y"] = yd_best
    pred_df["Final_X"] = xd_best
    pred_df["Final_Y"] = yd_best
    pred_df["Error_StepA_m"] = err_a
    pred_df["Error_StepB_m"] = err_b
    pred_df["Error_StepC_m"] = err_c
    pred_df["Error_StepD_m"] = err_d

    g_rmse_q: List[float] = []
    gsz = cfg.grid_size_m
    for k in range(len(val_p)):
        px = float(xd_best[k])
        py = float(yd_best[k])
        row = val_p.iloc[k]
        es: List[float] = []
        if np.isfinite(px) and np.isfinite(py):
            for wi, c in enumerate(loc.wifi_cols):
                m_raw = row.get(f"med_{c}", np.nan)
                if not np.isfinite(m_raw):
                    continue
                ap_key = c if c in WIFI_AP_TILES else WIFI_COL_CANON[wi]
                ax, ay = tile_to_m(WIFI_AP_TILES[ap_key], gsz)
                d_cal = float(m_raw) - ROBUST_BIAS_M[ap_key]
                geom = float(np.hypot(px - ax, py - ay))
                es.append((geom - d_cal) ** 2)
        g_rmse_q.append(float(np.sqrt(np.mean(es))) if es else float("nan"))
    pred_df["Quality_RMSE_m"] = np.asarray(g_rmse_q, dtype=float)

    pred_df.to_csv(out_dir / "v7_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trace_rows).to_csv(out_dir / "v7_grid_search_trace.csv", index=False, encoding="utf-8-sig")
    kill_df.to_csv(out_dir / "v7_sensor_kill_log.csv", index=False, encoding="utf-8-sig")

    # --- CDF ---
    plt.figure(figsize=(9, 5.5))
    for errs, lab, col in [
        (err_a, "Step A: linear Var + bounds", "tab:blue"),
        (err_b, "Step B: +Log1p Var", "tab:orange"),
        (err_c, f"Step C: +HBB (α={best_alpha})", "tab:green"),
        (err_d, f"Step D: +decay (β={best_beta})", "tab:red"),
    ]:
        e = errs[np.isfinite(errs)]
        if e.size == 0:
            continue
        se = np.sort(e)
        cdf = np.arange(1, len(se) + 1) / len(se)
        plt.plot(se, cdf, lw=2, label=lab, color=col)
    plt.xlabel("Position error (m)")
    plt.ylabel("Cumulative probability")
    plt.title("V7: Step A–D error CDF (validation)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "v7_step_cdf.png", dpi=200)
    plt.close()

    # --- Kill bar (누적 막대) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sensors = kill_df["sensor"].tolist()
    v1 = kill_df["Var_cap_exceeded_count"].values
    v2 = kill_df["HBB_cross_check_count"].values
    xpos = np.arange(len(sensors))
    ax.bar(xpos, v1, label='Var fusion cap 초과', color="steelblue")
    ax.bar(xpos, v2, bottom=v1, label="HBB Cross-Check", color="darkorange")
    ax.set_xticks(xpos)
    ax.set_xticklabels(sensors, rotation=45, ha="right")
    ax.set_ylabel("Drop count (validation rows)")
    ax.set_title("V7: Sensor weight-drop counts by reason")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v7_kill_log_bar.png", dpi=200)
    plt.close(fig)

    # --- Error vector map 12×18 ---
    valid = pred_df[np.isfinite(pred_df["True_X"]) & np.isfinite(pred_df["StepD_X"])].copy()
    plt.figure(figsize=(8, 10))
    plt.xlim(0, 12)
    plt.ylim(0, 18)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.quiver(
        valid["True_X"].values,
        valid["True_Y"].values,
        valid["StepD_X"].values - valid["True_X"].values,
        valid["StepD_Y"].values - valid["True_Y"].values,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
        alpha=0.65,
        color="navy",
    )
    plt.scatter(valid["True_X"], valid["True_Y"], s=14, c="green", label="True", zorder=5)
    plt.scatter(valid["StepD_X"], valid["StepD_Y"], s=14, c="red", label="Pred (Step D)", zorder=5)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("V7: True → Step D prediction (error vectors)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "v7_error_vector_map.png", dpi=200)
    plt.close()

    worst = pred_df.assign(_e=err_d).nlargest(10, "_e")
    print("Step D 기준 Worst 오차 노드 Top 10")
    print("-" * 72)
    for _, r in worst.iterrows():
        print(
            f"  Node({int(r['Node_x'])}, {int(r['Node_y'])}) | "
            f"True ({r['True_X']:.2f}, {r['True_Y']:.2f}) m | "
            f"error = {r['_e']:.3f} m"
        )
    print()
    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main, output_artifact_include_prefixes=("v7_",))
