"""
V8: Wi-Fi-only / UWB-only / V7식 HBB 하이브리드(α=1.2, 선형 Var 가중) / Top-K 앵커 선택 Ablation.

검증 RMSE로 K를 선택하고, v8_predictions.csv 및 CDF를 저장.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fusion_realtime_sanitize import (
    HARDWARE_CALIB_BIAS_M,
    VAR_CAP_UWB_FUSION,
    VAR_CAP_WIFI_FUSION,
    standardize_loaded_median_variance,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
)

if sys.platform == "win32":
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

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

BOUNDS_LO = np.array([0.0, 0.0], dtype=float)
BOUNDS_HI = np.array([12.0, 18.0], dtype=float)
MAX_RANGE_M = 22.0
EPS_W = 1e-4
UWB_BIAS_W = 0.5
WIFI_BIAS_W = 2.0
HBB_ALPHA = 1.2
K_CANDIDATES = [4, 5, 6, 7]


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


class FusionLocalizerV8:
    def __init__(self, config: Config):
        self.config = config
        self.uwb_cols: List[str] = []
        self.wifi_cols: List[str] = []
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

        g = self.config.grid_size_m
        self.uwb_pos_order_m = [
            tile_to_m(t, g) for t in [(1, 4), (20, 7), (5, 15), (4, 27), (15, 14), (14, 24)]
        ]
        self.wifi_pos_order_m = [
            tile_to_m(t, g) for t in [(20, 7), (4, 27), (14, 24), (15, 14), (1, 4), (5, 15)]
        ]

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

    def _smart_initial_guess(self, pos: np.ndarray, dist: np.ndarray) -> np.ndarray:
        if len(dist) == 0:
            return np.array([6.0, 9.0], dtype=float)
        idx = int(np.argmin(dist))
        return np.array(pos[idx], dtype=float)

    def _solve_trilat(self, pos: np.ndarray, dist: np.ndarray, weights: np.ndarray) -> np.ndarray:
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

    def _row_wifi_only(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_l: List[Tuple[float, float]] = []
        dist_l: List[float] = []
        w_l: List[float] = []
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            key = sensor_canon_key(c, i, False)
            pos_l.append(self._resolve_sensor_position_m(c, i, False))
            dist_l.append(float(d_raw) - ROBUST_BIAS_M[key])
            w_l.append(1.0 / (float(v) + WIFI_BIAS_W + EPS_W))
        return np.asarray(pos_l, float), np.asarray(dist_l, float), np.asarray(w_l, float)

    def _row_uwb_only(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            pos_l.append(self._resolve_sensor_position_m(c, i, True))
            dist_l.append(float(d_raw) - ROBUST_BIAS_M[key])
            w_l.append(1.0 / (float(v) + UWB_BIAS_W + EPS_W))
        return np.asarray(pos_l, float), np.asarray(dist_l, float), np.asarray(w_l, float)

    def _hbb_mask(self, row: pd.Series, alpha: float) -> List[bool]:
        kill_uwb = [False] * 6
        for idx in range(6):
            c_u, c_w = self.uwb_cols[idx], self.wifi_cols[idx]
            vu, vw = row.get(f"var_{c_u}", np.nan), row.get(f"var_{c_w}", np.nan)
            du_raw, dw_raw = row.get(f"med_{c_u}", np.nan), row.get(f"med_{c_w}", np.nan)
            was_nan = row.get(f"orig_nan_{c_u}", 1) == 1
            u_ok = (
                (not was_nan)
                and np.isfinite(du_raw)
                and np.isfinite(vu)
                and float(vu) <= VAR_CAP_UWB_FUSION
            )
            w_ok = np.isfinite(dw_raw) and np.isfinite(vw) and float(vw) <= VAR_CAP_WIFI_FUSION
            if not (u_ok and w_ok):
                continue
            ku, kw = sensor_canon_key(c_u, idx, True), sensor_canon_key(c_w, idx, False)
            du = float(du_raw) - ROBUST_BIAS_M[ku]
            dw = float(dw_raw) - ROBUST_BIAS_M[kw]
            if dw > 1e-6 and du > alpha * dw:
                kill_uwb[idx] = True
        return kill_uwb

    def _row_hybrid_c(self, row: pd.Series, alpha: float = HBB_ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hbb = self._hbb_mask(row, alpha)
        pos_l: List[Tuple[float, float]] = []
        dist_l: List[float] = []
        w_l: List[float] = []
        for i, c in enumerate(self.uwb_cols):
            if hbb[i]:
                continue
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan or (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_UWB_FUSION:
                continue
            key = sensor_canon_key(c, i, True)
            pos_l.append(self._resolve_sensor_position_m(c, i, True))
            dist_l.append(float(d_raw) - ROBUST_BIAS_M[key])
            w_l.append(1.0 / (float(v) + UWB_BIAS_W + EPS_W))
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            key = sensor_canon_key(c, i, False)
            pos_l.append(self._resolve_sensor_position_m(c, i, False))
            dist_l.append(float(d_raw) - ROBUST_BIAS_M[key])
            w_l.append(1.0 / (float(v) + WIFI_BIAS_W + EPS_W))
        return np.asarray(pos_l, float), np.asarray(dist_l, float), np.asarray(w_l, float)

    @staticmethod
    def _apply_topk(weights: np.ndarray, k: int) -> np.ndarray:
        n = len(weights)
        if n == 0:
            return weights
        k_use = min(max(k, 1), n)
        out = np.zeros_like(weights)
        order = np.argsort(-weights)
        keep = order[:k_use]
        out[keep] = weights[keep]
        return out

    def _row_topk_d(self, row: pd.Series, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, d, w = self._row_hybrid_c(row, HBB_ALPHA)
        if len(w) == 0:
            return p, d, w
        w2 = FusionLocalizerV8._apply_topk(w, k)
        n_pos = int(np.sum(w2 > 0))
        if n_pos < 3 and len(w) >= 3:
            w2 = FusionLocalizerV8._apply_topk(w, min(3, len(w)))
        return p, d, w2

    def predict_step(self, df: pd.DataFrame, step: str, topk: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        xs: List[float] = []
        ys: List[float] = []
        for _, row in df.iterrows():
            if step == "A":
                p, d, w = self._row_wifi_only(row)
            elif step == "B":
                p, d, w = self._row_uwb_only(row)
            elif step == "C":
                p, d, w = self._row_hybrid_c(row, HBB_ALPHA)
            else:
                p, d, w = self._row_topk_d(row, topk)
            xy = self._solve_trilat(p, d, w)
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        return np.asarray(xs, float), np.asarray(ys, float)

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
    def plot_cdf(errs: Dict[str, np.ndarray], labels: Dict[str, str], out_path: Path) -> None:
        plt.figure(figsize=(9, 5.5))
        colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
        for k in ("A", "B", "C", "D"):
            e = errs[k][np.isfinite(errs[k])]
            if e.size == 0:
                continue
            se = np.sort(e)
            cdf = np.arange(1, len(se) + 1) / len(se)
            plt.plot(se, cdf, lw=2, label=labels[k], color=colors[k])
        plt.xlabel("Position error (m)")
        plt.ylabel("Cumulative probability")
        plt.title("V8: Step A–D error CDF (validation)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


def pack_val_frame_like_v8(
    med_df: pd.DataFrame,
    var_df: pd.DataFrame,
    node_cols: List[str],
    uwb_cols: List[str],
    wifi_cols: List[str],
    grid_m: float,
) -> pd.DataFrame:
    """외부 로더가 만든 median/var 프레임을 Pure Wi-Fi Step A 평가용(V8 패킹)으로 변환한다."""
    t = pd.DataFrame()
    t["Node_x"] = med_df[node_cols[0]].values
    t["Node_y"] = med_df[node_cols[1]].values
    t["True_X"] = t["Node_x"].astype(float) * grid_m
    t["True_Y"] = t["Node_y"].astype(float) * grid_m
    for c in uwb_cols:
        t[f"med_{c}"] = med_df[c].values
        t[f"var_{c}"] = var_df[c].values
        t[f"orig_nan_{c}"] = pd.isna(med_df[c]).astype(int)
    for c in wifi_cols:
        t[f"med_{c}"] = med_df[c].values
        t[f"var_{c}"] = var_df[c].values
    for c in uwb_cols + wifi_cols:
        t.loc[t[f"med_{c}"] > MAX_RANGE_M, f"med_{c}"] = np.nan
    return t


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
    loc = FusionLocalizerV8(cfg)
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "train"
    val_dir = root / "data" / "validation"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_med, train_var = resolve_train_kgh_corrected_paths(train_dir)
    val_med, val_var = resolve_validation_paths(val_dir)
    train_packed, val = loc.load_datasets(
        train_median_path=train_med,
        train_variance_path=train_var,
        val_median_path=val_med,
        val_variance_path=val_var,
    )
    _ = train_packed  # 학습 포함 파이프라인과 행 정합성 확인용 (전 행 패킹, 드랍 없음)

    tx = val["True_X"].values
    ty = val["True_Y"].values

    xa, ya = loc.predict_step(val, "A")
    xb, yb = loc.predict_step(val, "B")
    xc, yc = loc.predict_step(val, "C")

    best_k = K_CANDIDATES[0]
    best_rmse_d = 1e9
    for kk in K_CANDIDATES:
        xd, yd = loc.predict_step(val, "D", topk=kk)
        err = loc.point_errors(tx, ty, xd, yd)
        r, _m = loc.rmse_mae(err)
        if r < best_rmse_d:
            best_rmse_d = r
            best_k = kk

    xd, yd = loc.predict_step(val, "D", topk=best_k)

    g = cfg.grid_size_m
    qc_d: List[float] = []
    for i, (_, row) in enumerate(val.iterrows()):
        px, py = float(xd[i]), float(yd[i])
        errs_sq: List[float] = []
        if np.isfinite(px) and np.isfinite(py):
            for wi, c in enumerate(loc.wifi_cols):
                m_raw = row.get(f"med_{c}", np.nan)
                if not np.isfinite(m_raw):
                    continue
                key_w = sensor_canon_key(c, wi, False)
                ax, ay = tile_to_m(WIFI_AP_TILES[key_w], g)
                d_cal = float(m_raw) - ROBUST_BIAS_M[key_w]
                geom = float(np.hypot(px - ax, py - ay))
                errs_sq.append((geom - d_cal) ** 2)
        qc_d.append(float(np.sqrt(np.mean(errs_sq))) if errs_sq else float("nan"))

    err_a = loc.point_errors(tx, ty, xa, ya)
    err_b = loc.point_errors(tx, ty, xb, yb)
    err_c = loc.point_errors(tx, ty, xc, yc)
    err_d = loc.point_errors(tx, ty, xd, yd)

    print("\n" + "=" * 72)
    print("V8 Validation — Ablation + Top-K (선형 W=1/(Var+Bias), HBB α=1.2)")
    print(
        f"  데이터 행 수: Train 패킹={len(train_packed)} · 검증={len(val)} (로드 후 행 삭제 없음)"
    )
    print("=" * 72)
    for step, name, err in [
        ("A", "Pure Wi-Fi", err_a),
        ("B", "Pure UWB", err_b),
        ("C", "Hybrid + HBB (V7 baseline)", err_c),
        ("D", f"Top-K (K={best_k}, grid {K_CANDIDATES})", err_d),
    ]:
        r, m = loc.rmse_mae(err)
        print(f"  Step {step} [{name}]: RMSE = {r:.4f} m  |  MAE = {m:.4f} m")
    print(f"\n  [Step D] 검증 RMSE 최소 K = {best_k}")
    print("=" * 72 + "\n")

    pred = val[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xa
    pred["StepA_Y"] = ya
    pred["StepB_X"] = xb
    pred["StepB_Y"] = yb
    pred["StepC_X"] = xc
    pred["StepC_Y"] = yc
    pred["StepD_X"] = xd
    pred["StepD_Y"] = yd
    pred["Final_X"] = xd
    pred["Final_Y"] = yd
    pred["Error_A_m"] = err_a
    pred["Error_B_m"] = err_b
    pred["Error_C_m"] = err_c
    pred["Error_D_m"] = err_d
    pred["Quality_RMSE_m"] = np.asarray(qc_d, dtype=float)
    pred.to_csv(out_dir / "v8_predictions.csv", index=False, encoding="utf-8-sig")

    labels = {
        "A": "A: Pure Wi-Fi",
        "B": "B: Pure UWB",
        "C": "C: Hybrid + HBB (α=1.2)",
        "D": f"D: Top-K (K={best_k})",
    }
    loc.plot_cdf({"A": err_a, "B": err_b, "C": err_c, "D": err_d}, labels, out_dir / "v8_step_cdf.png")

    worst = pred.assign(_e=err_d).nlargest(10, "_e")
    print("Step D 기준 Worst 오차 노드 Top 10")
    print("-" * 72)
    for _, r in worst.iterrows():
        print(
            f"  Node({int(r['Node_x'])}, {int(r['Node_y'])}) | "
            f"True ({r['True_X']:.2f}, {r['True_Y']:.2f}) m | error = {r['_e']:.3f} m"
        )
    print()
    print(f"저장: {out_dir / 'v8_predictions.csv'}")
    print(f"저장: {out_dir / 'v8_step_cdf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
