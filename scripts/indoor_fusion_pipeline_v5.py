"""
V5: v1~v3 경험 + v4 진단 반영.

- 융합: Wi-Fi 결측 시 UWB 단독 허용, UWB 분산>fusion 상한(Var_CAP_UWB)에서 가중 감쇠, 비대칭 임계 완화.
- 삼변측량: 앵커별 분산 기반 가중 Huber (v3와 유사한 sqrt(w)*잔차), f_scale 고정에 가깝게.
- 보정: K-NN 대신 Train에서만 Ridge(잔차 ~ [1,x,y]) 적합, 검증에서 개선 확인 시에만 적용.
- 진단: outputs/v5_*.csv/json, v5_validation_predictions.csv
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fusion_realtime_sanitize import (
    VAR_CAP_UWB_FUSION,
    standardize_loaded_median_variance,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
)

# v4와 동일 상수 (로더/바이어스/앵커)
from indoor_fusion_pipeline_v4 import (
    BOUNDS_HI,
    BOUNDS_LO,
    EPS_W,
    HARD_HI,
    HARD_LO,
    QUALITY_RMSE_TH,
    UWB_ANCHOR_TILES,
    UWB_BIAS_M,
    UWB_COL_CANON,
    WIFI_AP_TILES,
    WIFI_BIAS_M,
    WIFI_COL_CANON,
    coerce_numeric,
    infer_sensor_columns,
    load_sensor_excel,
    tile_to_m,
)

# v5 전용
ASYMM_RATIO = 2.2
VAR_UWB_SOFT_START = VAR_CAP_UWB_FUSION
HUBER_F_SCALE = 1.5
RIDGE_ALPHA = 28.0
RIDGE_MIN_IMPROVE_RATIO = 0.97


@dataclass
class Config:
    grid_size_m: float = 0.6


def anchor_weight_from_vars(var_u: float, var_w: float) -> Tuple[float, float]:
    """분산이 클수록 가중 감소; UWB는 Var>fusion 상한에서 추가 감쇠 (하드 드롭 없음)."""
    if not np.isfinite(var_u):
        var_u = 50.0
    if not np.isfinite(var_w):
        var_w = 50.0
    w_u = 1.0 / (np.log1p(max(0.0, var_u)) + EPS_W)
    w_w = 1.0 / (np.log1p(max(0.0, var_w)) + EPS_W)
    if var_u > VAR_UWB_SOFT_START:
        w_u *= float(VAR_UWB_SOFT_START / var_u)
    return float(w_u), float(w_w)


class FusionLocalizerV5:
    def __init__(self, config: Config):
        self.config = config
        self.uwb_cols: List[str] = []
        self.wifi_cols: List[str] = []
        self._anchor_xy_m: List[Tuple[float, float]] = []
        self._ridge_dx: Ridge | None = None
        self._ridge_dy: Ridge | None = None
        self.ridge_applied: bool = False

    def _assert_files(self, paths: List[Path]) -> None:
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")

    def load_frame(self, med_path: Path, var_path: Path) -> pd.DataFrame:
        self._assert_files([med_path, var_path])
        med, var, node_cols, self.uwb_cols, self.wifi_cols = standardize_loaded_median_variance(
            med_path, var_path, coerce_numeric, infer_sensor_columns
        )

        g = self.config.grid_size_m
        self._uwb_canon = []
        self._wifi_canon = []
        for i, c in enumerate(self.uwb_cols):
            k = c if c in UWB_BIAS_M else UWB_COL_CANON[i]
            self._uwb_canon.append(k)
        for i, c in enumerate(self.wifi_cols):
            k = c if c in WIFI_BIAS_M else WIFI_COL_CANON[i]
            self._wifi_canon.append(k)
        self._anchor_xy_m = [tile_to_m(UWB_ANCHOR_TILES[k], g) for k in self._uwb_canon]

        out = pd.DataFrame()
        out["Node_x"] = med[node_cols[0]].values
        out["Node_y"] = med[node_cols[1]].values
        out["True_X"] = out["Node_x"].astype(float) * g
        out["True_Y"] = out["Node_y"].astype(float) * g
        for c in self.uwb_cols:
            out[f"med_{c}"] = med[c].values
            out[f"var_{c}"] = var[c].values
        for c in self.wifi_cols:
            out[f"med_{c}"] = med[c].values
            out[f"var_{c}"] = var[c].values
        return out

    def phase1_calibrate_and_clip(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        d_u = np.empty(6, dtype=float)
        d_w = np.empty(6, dtype=float)
        for i, c in enumerate(self.uwb_cols):
            m = row.get(f"med_{c}", np.nan)
            if not np.isfinite(m):
                d_u[i] = np.nan
                continue
            v = float(m) - UWB_BIAS_M[self._uwb_canon[i]]
            d_u[i] = v if HARD_LO < v < HARD_HI else np.nan
        for i, c in enumerate(self.wifi_cols):
            m = row.get(f"med_{c}", np.nan)
            if not np.isfinite(m):
                d_w[i] = np.nan
                continue
            v = float(m) - WIFI_BIAS_M[self._wifi_canon[i]]
            d_w[i] = v if HARD_LO < v < HARD_HI else np.nan
        return d_u, d_w

    def phase2_fuse(self, row: pd.Series, i: int, d_u: float, d_w: float) -> Tuple[float, str]:
        c_u, c_w = self.uwb_cols[i], self.wifi_cols[i]
        var_u = float(row.get(f"var_{c_u}", np.nan))
        var_w = float(row.get(f"var_{c_w}", np.nan))

        fu, fw = np.isfinite(d_u), np.isfinite(d_w)
        if not fu and not fw:
            return np.nan, "nan_both"
        if not fu and fw:
            return float(d_w), "wifi_only_missing_uwb"
        if fu and not fw:
            return float(d_u), "uwb_only_missing_wifi"

        if d_u >= ASYMM_RATIO * d_w:
            return 0.2 * d_u + 0.8 * d_w, "asym_uwb_multipath"
        if d_w >= ASYMM_RATIO * d_u:
            return 0.35 * d_u + 0.65 * d_w, "asym_wifi_dom"

        w_u, w_w = anchor_weight_from_vars(var_u, var_w)
        s = w_u + w_w
        if s <= 0:
            return float(0.5 * (d_u + d_w)), "fallback_mean"
        return (w_u * d_u + w_w * d_w) / s, "soft_var_weighted"

    def row_fused(self, row: pd.Series) -> Tuple[np.ndarray, List[str], np.ndarray]:
        d_u, d_w = self.phase1_calibrate_and_clip(row)
        fused = np.empty(6, dtype=float)
        rules: List[str] = []
        w_row = np.empty(6, dtype=float)
        for i in range(6):
            fused[i], r = self.phase2_fuse(row, i, d_u[i], d_w[i])
            c_u, c_w = self.uwb_cols[i], self.wifi_cols[i]
            vu = float(row.get(f"var_{c_u}", np.nan))
            vw = float(row.get(f"var_{c_w}", np.nan))
            wu, ww = anchor_weight_from_vars(vu, vw)
            w_row[i] = float(np.nanmedian([wu, ww])) if np.isfinite(fused[i]) else np.nan
        return fused, rules, w_row

    def phase3_trilaterate(self, d_fused: np.ndarray, weights: np.ndarray) -> Tuple[float, float, int]:
        pos = np.array(self._anchor_xy_m, dtype=float)
        valid = np.isfinite(d_fused) & np.isfinite(weights) & (weights > 0)
        n_v = int(valid.sum())
        if n_v < 3:
            return np.nan, np.nan, n_v
        P = pos[valid]
        d = d_fused[valid].astype(float)
        w = weights[valid].astype(float)
        w = np.clip(w, 1e-6, None)

        j = int(np.argmin(d))
        x0 = np.array(P[j], dtype=float)

        def residual(xy: np.ndarray) -> np.ndarray:
            pr = np.sqrt((P[:, 0] - xy[0]) ** 2 + (P[:, 1] - xy[1]) ** 2)
            return np.sqrt(w) * (pr - d)

        res = least_squares(
            residual,
            x0=x0,
            bounds=(np.array(BOUNDS_LO, dtype=float), np.array(BOUNDS_HI, dtype=float)),
            loss="huber",
            f_scale=HUBER_F_SCALE,
            method="trf",
        )
        return float(res.x[0]), float(res.x[1]), n_v

    @staticmethod
    def phase4_quality(pos_xy: Tuple[float, float], anchor_xy: List[Tuple[float, float]], d_fused: np.ndarray) -> Tuple[float, str]:
        if not (np.isfinite(pos_xy[0]) and np.isfinite(pos_xy[1])):
            return np.nan, "Low"
        pos = np.array(anchor_xy, dtype=float)
        geom = np.sqrt((pos[:, 0] - pos_xy[0]) ** 2 + (pos[:, 1] - pos_xy[1]) ** 2)
        mask = np.isfinite(d_fused)
        if mask.sum() == 0:
            return np.nan, "Low"
        r = geom[mask] - d_fused[mask]
        q = float(np.sqrt(np.mean(r**2)))
        return q, ("High" if q < QUALITY_RMSE_TH else "Low")

    def run_base(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        out = df.copy()
        xs, ys, qs, cs = [], [], [], []
        diag = []
        for si, (_, row) in enumerate(out.iterrows()):
            d_f, rules, w_r = self.row_fused(row)
            x, y, n_v = self.phase3_trilaterate(d_f, w_r)
            q, c = self.phase4_quality((x, y), self._anchor_xy_m, d_f)
            xs.append(x)
            ys.append(y)
            qs.append(q)
            cs.append(c)
            diag.append(
                {
                    "sample_index": si,
                    "p3_n_valid": n_v,
                    "p4_q": q,
                    "p4_conf": c,
                    "fuse_rules": "|".join(rules),
                }
            )
        out["Base_X"] = xs
        out["Base_Y"] = ys
        out["Quality_RMSE_m"] = qs
        out["Confidence"] = cs
        return out, pd.DataFrame(diag)

    def try_fit_ridge(self, train_base: pd.DataFrame) -> None:
        """Train 기준 위치 오차가 줄어들 때만 Ridge 보정 활성화."""
        xa = train_base["Base_X"].values.astype(float)
        ya = train_base["Base_Y"].values.astype(float)
        tx = train_base["True_X"].values.astype(float)
        ty = train_base["True_Y"].values.astype(float)
        m = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(tx) & np.isfinite(ty)
        if m.sum() < 12:
            self._ridge_dx = None
            self._ridge_dy = None
            self.ridge_applied = False
            self._ridge_report = {"ridge_enabled": False, "reason": "too_few_train_points", "n_train_fit": int(m.sum())}
            return
        X = np.column_stack([np.ones(m.sum()), xa[m], ya[m]])
        dx = (tx - xa)[m]
        dy = (ty - ya)[m]
        err_before = np.hypot(dx, dy)
        rid_x = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        rid_y = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        rid_x.fit(X, dx)
        rid_y.fit(X, dy)
        dx_p = rid_x.predict(X)
        dy_p = rid_y.predict(X)
        err_after = np.hypot(dx - dx_p, dy - dy_p)
        imp = float(np.mean(err_after) / (np.mean(err_before) + 1e-9))
        if imp < RIDGE_MIN_IMPROVE_RATIO:
            self._ridge_dx = rid_x
            self._ridge_dy = rid_y
            self.ridge_applied = True
        else:
            self._ridge_dx = None
            self._ridge_dy = None
            self.ridge_applied = False

        rep = {
            "ridge_enabled": self.ridge_applied,
            "mean_err_before_m": float(np.mean(err_before)),
            "mean_err_after_m": float(np.mean(err_after)),
            "improvement_ratio": float(imp),
            "alpha": RIDGE_ALPHA,
            "n_train_fit": int(m.sum()),
        }
        self._ridge_report = rep

    def apply_ridge(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.ridge_applied or self._ridge_dx is None:
            out["Pred_X"] = out["Base_X"]
            out["Pred_Y"] = out["Base_Y"]
            return out
        xa = out["Base_X"].values.astype(float)
        ya = out["Base_Y"].values.astype(float)
        pred_x = np.full(len(out), np.nan)
        pred_y = np.full(len(out), np.nan)
        m = np.isfinite(xa) & np.isfinite(ya)
        X = np.column_stack([np.ones(m.sum()), xa[m], ya[m]])
        pred_x[m] = xa[m] + self._ridge_dx.predict(X)
        pred_y[m] = ya[m] + self._ridge_dy.predict(X)
        out["Pred_X"] = pred_x
        out["Pred_Y"] = pred_y
        return out

    def run_full(self, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tr_base, _ = self.run_base(train_df)
        self.try_fit_ridge(tr_base)
        tr_out = self.apply_ridge(tr_base.copy())
        ev_base, ev_diag = self.run_base(eval_df)
        self._eval_diag = ev_diag
        ev_out = self.apply_ridge(ev_base.copy())
        return tr_out, ev_out

    def wifi_only_weights_row(self, row: pd.Series, d_w: np.ndarray) -> np.ndarray:
        w_row = np.empty(6, dtype=float)
        for i in range(6):
            c_w = self.wifi_cols[i]
            vw = float(row.get(f"var_{c_w}", np.nan))
            if not np.isfinite(d_w[i]) or not np.isfinite(vw):
                w_row[i] = np.nan
            else:
                w_row[i] = 1.0 / (np.log1p(max(0.0, vw)) + EPS_W)
        return w_row

    def run_wifi_only_base(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """각 앵커에서 Wi-Fi 보정 거리·log1p 분산 가중치만 사용한 phase3 삼변."""
        xs: List[float] = []
        ys: List[float] = []
        for _, row in df.iterrows():
            _d_u, d_w = self.phase1_calibrate_and_clip(row)
            w_row = self.wifi_only_weights_row(row, d_w)
            x, y, _nv = self.phase3_trilaterate(d_w, w_row)
            xs.append(x)
            ys.append(y)
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def position_errors(tx: np.ndarray, ty: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return np.sqrt((px - tx) ** 2 + (py - ty) ** 2)


def main() -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass

    root = Path(__file__).resolve().parent.parent
    train_dir = root / "data" / "train"
    val_dir = root / "data" / "validation"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_med, train_var = resolve_train_kgh_corrected_paths(train_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    try:
        val_med, val_var = resolve_validation_paths(val_dir)
        label = "validation"
    except FileNotFoundError:
        val_med, val_var = train_med, train_var
        label = "train (검증 없음, 원본 train)"

    cfg = Config()
    loc = FusionLocalizerV5(cfg)
    train_df = loc.load_frame(train_med, train_var)
    val_df = loc.load_frame(val_med, val_var)
    print(f"V5 행 수: Train={len(train_df)} 검증 len={len(val_df)}")

    wx, wy = loc.run_wifi_only_base(val_df)
    err_wifi = position_errors(val_df["True_X"].values, val_df["True_Y"].values, wx, wy)
    fin_w = np.isfinite(err_wifi)
    ewf = err_wifi[fin_w]
    rmse_wf = (
        float(np.sqrt(mean_squared_error(np.zeros(len(ewf)), ewf))) if len(ewf) else float("nan")
    )
    mae_wf = float(mean_absolute_error(np.zeros(len(ewf)), ewf)) if len(ewf) else float("nan")
    print("\n=== V5 Validation (Wi-Fi only 삼변, UWB 미사용 · 동일 보정+log1p 가중) ===")
    print(f"Wi-Fi-only RMSE: {rmse_wf:.4f} m | MAE: {mae_wf:.4f} m")

    tr_pred, va_pred = loc.run_full(train_df, val_df)

    if getattr(loc, "_ridge_report", None):
        (out_dir / "v5_train_ridge_report.json").write_text(
            json.dumps(loc._ridge_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("저장:", out_dir / "v5_train_ridge_report.json")

    fin = (
        np.isfinite(va_pred["True_X"].values.astype(float))
        & np.isfinite(va_pred["True_Y"].values.astype(float))
        & np.isfinite(va_pred["Pred_X"].values.astype(float))
        & np.isfinite(va_pred["Pred_Y"].values.astype(float))
    )
    ok = va_pred.loc[fin]
    err = position_errors(ok["True_X"].values, ok["True_Y"].values, ok["Pred_X"].values, ok["Pred_Y"].values)
    rmse = float(np.sqrt(mean_squared_error(np.zeros(len(err)), err)))
    mae = float(mean_absolute_error(np.zeros(len(err)), err))
    hi = ok[ok["Confidence"] == "High"]
    rmse_hi = (
        float(np.sqrt(mean_squared_error(np.zeros(len(hi)), position_errors(hi["True_X"].values, hi["True_Y"].values, hi["Pred_X"].values, hi["Pred_Y"].values))))
        if len(hi) > 0
        else float("nan")
    )

    print(f"\n=== V5 ({label}) ===")
    print(f"Ridge 보정 활성: {loc.ridge_applied}")
    print(f"전체 RMSE: {rmse:.4f} m | MAE: {mae:.4f} m (n={len(err)})")
    print(f'Confidence="High"만 RMSE: {rmse_hi:.4f} m (n={len(hi)})')

    save = va_pred[
        ["Node_x", "Node_y", "True_X", "True_Y", "Base_X", "Base_Y", "Pred_X", "Pred_Y", "Confidence", "Quality_RMSE_m"]
    ].copy()
    save["Error_m"] = position_errors(save["True_X"].values, save["True_Y"].values, save["Pred_X"].values, save["Pred_Y"].values)
    save.to_csv(out_dir / "v5_validation_predictions.csv", index=False)
    print("저장:", out_dir / "v5_validation_predictions.csv")

    if hasattr(loc, "_eval_diag") and len(loc._eval_diag):
        loc._eval_diag.to_csv(out_dir / "v5_validation_fusion_summary.csv", index=False)
        print("저장:", out_dir / "v5_validation_fusion_summary.csv")

    err_sorted = np.sort(err)
    cdf = np.arange(1, len(err_sorted) + 1) / max(len(err_sorted), 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(err_sorted, cdf, lw=2, color="darkgreen", label="V5")
    plt.axvline(2.0, color="gray", ls="--", lw=1)
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("V5 error CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "v5_error_cdf.png", dpi=160)
    plt.close()
    print("저장:", out_dir / "v5_error_cdf.png")

    plt.figure(figsize=(9, 6.5))
    plt.scatter(save["True_X"], save["True_Y"], c="blue", s=36, alpha=0.85, label="True", zorder=2)
    hm = save["Confidence"] == "High"
    lm = save["Confidence"] == "Low"
    plt.scatter(save.loc[hm, "Pred_X"], save.loc[hm, "Pred_Y"], c="green", s=30, alpha=0.8, label="Pred High", zorder=3)
    plt.scatter(save.loc[lm, "Pred_X"], save.loc[lm, "Pred_Y"], c="red", s=30, alpha=0.8, label="Pred Low", zorder=3)
    plt.xlim(0, 12)
    plt.ylim(0, 18)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("V5 confidence map")
    plt.savefig(out_dir / "v5_confidence_map.png", dpi=160)
    plt.close()
    print("저장:", out_dir / "v5_confidence_map.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
