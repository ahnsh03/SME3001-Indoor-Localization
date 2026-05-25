"""
V6: V3 삼변·UWB KNN 임퓨터를 베이스로, 프로파일러 캘리브·분산 컷(Step B),
경계 제약 least_squares(Step C), 최악 잔차 앵커 드롭(Step D)을 순차 적용.

검증 세트에 대해 Step A~D RMSE/MAE 출력, CDF, v6_validation_predictions.csv 저장.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    standardize_loaded_median_variance,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
)

# --- 공간 / 앵커 (V3 동일) ---
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

RES_DROP_MIN_M = 2.5
BOUNDS_LO = np.array([0.0, 0.0], dtype=float)
BOUNDS_HI = np.array([12.0, 18.0], dtype=float)
MAX_RANGE_M = 22.0
EPS_W = 1e-4


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
    # Step A (V3): UWB 0.5, Wi-Fi 2.5
    uwb_bias_a: float = 0.5
    wifi_bias_a: float = 2.5
    # Step B~D: W = 1/(Variance + Bias), Wi-Fi Bias = 2.0
    uwb_bias_bcd: float = 0.5
    wifi_bias_bcd: float = 2.0
    uwb_imputer_neighbors: int = 5


class FusionLocalizerV6:
    def __init__(self, config: Config):
        self.config = config
        self.uwb_cols: List[str] = []
        self.wifi_cols: List[str] = []
        self.uwb_imputer: KNNImputer | None = None
        self.uwb_pos_order_m: List[Tuple[float, float]] = []
        self.wifi_pos_order_m: List[Tuple[float, float]] = []

    def _tile_to_meter(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        g = self.config.grid_size_m
        return [(x * g, y * g) for x, y in points]

    def _resolve_sensor_position_m(self, sensor_name: str, sensor_index: int, is_uwb: bool) -> Tuple[float, float]:
        if is_uwb and sensor_name in UWB_ANCHOR_TILES:
            return tile_to_m(UWB_ANCHOR_TILES[sensor_name], self.config.grid_size_m)
        if (not is_uwb) and sensor_name in WIFI_AP_TILES:
            return tile_to_m(WIFI_AP_TILES[sensor_name], self.config.grid_size_m)
        return (self.uwb_pos_order_m if is_uwb else self.wifi_pos_order_m)[sensor_index]

    def _assert_files(self, paths: List[Path]) -> None:
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")

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

        self.uwb_pos_order_m = self._tile_to_meter([(1, 4), (20, 7), (5, 15), (4, 27), (15, 14), (14, 24)])
        self.wifi_pos_order_m = self._tile_to_meter([(20, 7), (4, 27), (14, 24), (15, 14), (1, 4), (5, 15)])

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
        bounds: Tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        if len(dist) == 0:
            return np.array([np.nan, np.nan], dtype=float)
        if len(dist) < 3:
            d = np.clip(dist, 1e-3, None)
            w = 1.0 / d
            xy = (pos * w[:, None]).sum(axis=0) / w.sum()
            if bounds is not None:
                lo, hi = bounds
                xy = np.clip(xy, lo, hi)
            return xy.astype(float)

        x0 = self._smart_initial_guess(pos, dist)
        if bounds is not None:
            lo, hi = bounds
            x0 = np.clip(x0, lo, hi)

        def residuals(xy: np.ndarray) -> np.ndarray:
            pred = np.sqrt((xy[0] - pos[:, 0]) ** 2 + (xy[1] - pos[:, 1]) ** 2)
            return np.sqrt(weights) * (pred - dist)

        kw: Dict = dict(
            x0=x0,
            loss=self.config.trilateration_loss,
            f_scale=self.config.huber_f_scale,
            method="trf",
        )
        if bounds is not None:
            kw["bounds"] = bounds
        res = least_squares(residuals, **kw)
        return np.array(res.x, dtype=float)

    def _build_step_a(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_list: List[Tuple[float, float]] = []
        dist_list: List[float] = []
        weight_list: List[float] = []

        for i, c in enumerate(self.uwb_cols):
            d = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan or (not np.isfinite(d)) or (not np.isfinite(v)):
                continue
            pos_list.append(self._resolve_sensor_position_m(c, i, True))
            dist_list.append(float(d))
            weight_list.append(1.0 / (float(v) + self.config.uwb_bias_a + EPS_W))

        for i, c in enumerate(self.wifi_cols):
            d = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d)) or (not np.isfinite(v)):
                continue
            pos_list.append(self._resolve_sensor_position_m(c, i, False))
            dist_list.append(float(d))
            weight_list.append(1.0 / (float(v) + self.config.wifi_bias_a + EPS_W))

        return (
            np.array(pos_list, dtype=float),
            np.array(dist_list, dtype=float),
            np.array(weight_list, dtype=float),
        )

    def _build_step_bcd(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_list: List[Tuple[float, float]] = []
        dist_list: List[float] = []
        weight_list: List[float] = []

        for i, c in enumerate(self.uwb_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan or (not np.isfinite(d_raw)) or (not np.isfinite(v)):
                continue
            key = sensor_canon_key(c, i, True)
            bias = ROBUST_BIAS_M[key]
            d_use = float(d_raw) - bias
            if float(v) > VAR_CAP_UWB_FUSION:
                continue
            pos_list.append(self._resolve_sensor_position_m(c, i, True))
            dist_list.append(d_use)
            weight_list.append(1.0 / (float(v) + self.config.uwb_bias_bcd + EPS_W))

        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)):
                continue
            key = sensor_canon_key(c, i, False)
            bias = ROBUST_BIAS_M[key]
            d_use = float(d_raw) - bias
            if float(v) > VAR_CAP_WIFI_FUSION:
                continue
            pos_list.append(self._resolve_sensor_position_m(c, i, False))
            dist_list.append(d_use)
            weight_list.append(1.0 / (float(v) + self.config.wifi_bias_bcd + EPS_W))

        return (
            np.array(pos_list, dtype=float),
            np.array(dist_list, dtype=float),
            np.array(weight_list, dtype=float),
        )

    def _build_wifi_only_step_bcd(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step B~D와 동일 프로파일 오프셋·Var cap·가중치이나 Wi-Fi 6개만 사용."""
        pos_list: List[Tuple[float, float]] = []
        dist_list: List[float] = []
        weight_list: List[float] = []
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)):
                continue
            key = sensor_canon_key(c, i, False)
            d_use = float(d_raw) - ROBUST_BIAS_M[key]
            if float(v) > VAR_CAP_WIFI_FUSION:
                continue
            pos_list.append(self._resolve_sensor_position_m(c, i, False))
            dist_list.append(d_use)
            weight_list.append(1.0 / (float(v) + self.config.wifi_bias_bcd + EPS_W))
        return (
            np.array(pos_list, dtype=float),
            np.array(dist_list, dtype=float),
            np.array(weight_list, dtype=float),
        )

    def predict_wifi_only_bounded(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        xs: List[float] = []
        ys: List[float] = []
        for _, row in df.iterrows():
            p, d, w = self._build_wifi_only_step_bcd(row)
            xy = self._solve_trilat(p, d, w, bounds=(BOUNDS_LO, BOUNDS_HI))
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    def _solve_step_d(self, pos: np.ndarray, dist: np.ndarray, weights: np.ndarray) -> np.ndarray:
        xy_c = self._solve_trilat(pos, dist, weights, bounds=(BOUNDS_LO, BOUNDS_HI))
        if len(dist) == 0 or not np.all(np.isfinite(xy_c)):
            return xy_c
        geom = np.sqrt((xy_c[0] - pos[:, 0]) ** 2 + (xy_c[1] - pos[:, 1]) ** 2)
        resids = np.abs(geom - dist)
        j = int(np.argmax(resids))
        if resids[j] < RES_DROP_MIN_M:
            return xy_c
        mask = np.ones(len(dist), dtype=bool)
        mask[j] = False
        pos2 = pos[mask]
        dist2 = dist[mask]
        w2 = weights[mask]
        return self._solve_trilat(pos2, dist2, w2, bounds=(BOUNDS_LO, BOUNDS_HI))

    def run_all_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        xa, ya, xb, yb, xc, yc, xd, yd = [], [], [], [], [], [], [], []
        for _, row in out.iterrows():
            pa, da, wa = self._build_step_a(row)
            xya = self._solve_trilat(pa, da, wa, bounds=None)
            xa.append(float(xya[0]))
            ya.append(float(xya[1]))

            pb, db, wb = self._build_step_bcd(row)
            xyb = self._solve_trilat(pb, db, wb, bounds=None)
            xb.append(float(xyb[0]))
            yb.append(float(xyb[1]))

            xyc = self._solve_trilat(pb, db, wb, bounds=(BOUNDS_LO, BOUNDS_HI))
            xc.append(float(xyc[0]))
            yc.append(float(xyc[1]))

            xyd = self._solve_step_d(pb, db, wb)
            xd.append(float(xyd[0]))
            yd.append(float(xyd[1]))

        out["StepA_X"] = xa
        out["StepA_Y"] = ya
        out["StepB_X"] = xb
        out["StepB_Y"] = yb
        out["StepC_X"] = xc
        out["StepC_Y"] = yc
        out["StepD_X"] = xd
        out["StepD_Y"] = yd
        out["Final_X"] = xd
        out["Final_Y"] = yd
        return out

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

    @staticmethod
    def plot_step_cdf(errors: Dict[str, np.ndarray], out_path: Path) -> None:
        plt.figure(figsize=(9, 5.5))
        colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
        labels = {
            "A": "Step A: V3 base (trilat)",
            "B": "Step B: +Profiler calib & var cap",
            "C": "Step C: +Bounds [0,12]×[0,18]",
            "D": "Step D: +Worst residual drop (≥2.5 m)",
        }
        for key in ("A", "B", "C", "D"):
            e = errors[key]
            e = e[np.isfinite(e)]
            if e.size == 0:
                continue
            se = np.sort(e)
            cdf = np.arange(1, len(se) + 1) / len(se)
            plt.plot(se, cdf, lw=2, label=labels[key], color=colors[key])
        plt.xlabel("Position error (m)")
        plt.ylabel("Cumulative probability")
        plt.title("V6: Step A–D error CDF (validation)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


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
    loc = FusionLocalizerV6(cfg)

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
    train_p = loc.add_imputed_uwb_features(train)
    val_p = loc.add_imputed_uwb_features(val)

    wx, wy = loc.predict_wifi_only_bounded(val_p)
    err_wifi = loc.point_errors(val_p["True_X"].values, val_p["True_Y"].values, wx, wy)
    rw, mw = loc.rmse_mae(err_wifi)
    print("\n=== V6 Validation — Wi-Fi only (프로파일 오프셋 + Var≤6 + bounds, UWB 미사용) ===")
    print(f"  Wi-Fi-only RMSE = {rw:.4f} m  |  MAE = {mw:.4f} m\n")

    val_out = loc.run_all_steps(val_p)

    tx = val_out["True_X"].values
    ty = val_out["True_Y"].values
    errs = {
        "A": loc.point_errors(tx, ty, val_out["StepA_X"].values, val_out["StepA_Y"].values),
        "B": loc.point_errors(tx, ty, val_out["StepB_X"].values, val_out["StepB_Y"].values),
        "C": loc.point_errors(tx, ty, val_out["StepC_X"].values, val_out["StepC_Y"].values),
        "D": loc.point_errors(tx, ty, val_out["StepD_X"].values, val_out["StepD_Y"].values),
    }

    print("\n" + "=" * 72)
    print("V6 Validation — Step별 RMSE / MAE (m)")
    print("=" * 72)
    for step in ("A", "B", "C", "D"):
        r, m = loc.rmse_mae(errs[step])
        print(f"  Step {step}:  RMSE = {r:.4f} m  |  MAE = {m:.4f} m")
    print("=" * 72 + "\n")

    m_ok = (
        np.isfinite(val_out["True_X"].values.astype(float))
        & np.isfinite(val_out["True_Y"].values.astype(float))
        & np.isfinite(val_out["StepD_X"].values.astype(float))
        & np.isfinite(val_out["StepD_Y"].values.astype(float))
    )
    valid_nodes = val_out.loc[m_ok].copy()
    valid_nodes["StepD_Error_m"] = loc.point_errors(
        valid_nodes["True_X"].values,
        valid_nodes["True_Y"].values,
        valid_nodes["StepD_X"].values,
        valid_nodes["StepD_Y"].values,
    )
    worst = valid_nodes.nlargest(10, "StepD_Error_m")
    print("Step D 기준 Worst 오차 노드 Top 10 (그리드 Node_x, Node_y / True X,Y / 오차 m)")
    print("-" * 72)
    for _, r in worst.iterrows():
        print(
            f"  Node({int(r['Node_x'])}, {int(r['Node_y'])}) | "
            f"True ({r['True_X']:.2f}, {r['True_Y']:.2f}) m | "
            f"error = {r['StepD_Error_m']:.3f} m"
        )
    print()

    pred_path = out_dir / "v6_validation_predictions.csv"
    cols = [
        "Node_x",
        "Node_y",
        "True_X",
        "True_Y",
        "StepA_X",
        "StepA_Y",
        "StepB_X",
        "StepB_Y",
        "StepC_X",
        "StepC_Y",
        "StepD_X",
        "StepD_Y",
        "Final_X",
        "Final_Y",
    ]
    vend = val_out[cols].copy()
    edi = np.hypot(
        vend["True_X"].values.astype(float) - vend["Final_X"].values.astype(float),
        vend["True_Y"].values.astype(float) - vend["Final_Y"].values.astype(float),
    )
    sq: List[float] = []
    g = cfg.grid_size_m
    for _, row in val_out.iterrows():
        px = float(row.get("Final_X", np.nan))
        py = float(row.get("Final_Y", np.nan))
        errs_sq: List[float] = []
        if np.isfinite(px) and np.isfinite(py):
            for wi, c in enumerate(loc.wifi_cols):
                m_raw = row.get(f"med_{c}", np.nan)
                if not np.isfinite(m_raw):
                    continue
                ap_key = c if c in WIFI_AP_TILES else WIFI_COL_CANON[wi]
                ax, ay = tile_to_m(WIFI_AP_TILES[ap_key], g)
                d_cal = float(m_raw) - ROBUST_BIAS_M[ap_key]
                geom = float(np.hypot(px - ax, py - ay))
                errs_sq.append((geom - d_cal) ** 2)
        sq.append(float(np.sqrt(np.mean(errs_sq))) if errs_sq else float("nan"))
    vend["Quality_RMSE_m"] = sq
    vend["Final_Error_m"] = edi
    vend.to_csv(pred_path, index=False, encoding="utf-8-sig")

    cdf_path = out_dir / "v6_step_comparison_cdf.png"
    loc.plot_step_cdf(errs, cdf_path)

    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main, output_artifact_include_prefixes=("v6_",))
