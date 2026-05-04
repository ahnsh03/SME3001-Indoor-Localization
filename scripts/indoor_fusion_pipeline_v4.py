"""
V4: K-NN 없이 캘리브레이션 + 비대칭 융합 + 분산 컷 + 경계 Huber 삼변측량 + 품질 필터.

진단: outputs/v4_*.csv/json + v4_validation_predictions.csv, 터미널 요약
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fusion_realtime_sanitize import (
    PHASE1_UWB_BIAS_M,
    PHASE1_WIFI_BIAS_M,
    VAR_CAP_UWB_FUSION,
    VAR_CAP_WIFI_FUSION,
    standardize_loaded_median_variance,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
)

# --- 앵커 (그리드 타일, m 변환은 grid_size_m 배) ---
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

UWB_BIAS_M: Dict[str, float] = dict(PHASE1_UWB_BIAS_M)
WIFI_BIAS_M: Dict[str, float] = dict(PHASE1_WIFI_BIAS_M)

UWB_COL_CANON = ["110394ab", "e63ce2f", "8e610981", "d10485af", "d1044709", "4e610206"]
WIFI_COL_CANON = ["SW_11", "SW_first_team", "볼링공", "SW_4", "SW_6", "SW_5"]

HARD_LO = 0.5
HARD_HI = 22.0
VAR_UWB_CUT = VAR_CAP_UWB_FUSION  # profiler Train 분산 P95 정렬 (~16.56 m²)
ASYMM_UWB_DOM = 2.0
EPS_W = 1e-4
QUALITY_RMSE_TH = 2.5
BOUNDS_LO = (0.0, 0.0)
BOUNDS_HI = (12.0, 18.0)


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


@dataclass
class Config:
    grid_size_m: float = 0.6


class FusionLocalizerV4:
    def __init__(self, config: Config):
        self.config = config
        self.uwb_cols: List[str] = []
        self.wifi_cols: List[str] = []
        self._anchor_xy_m: List[Tuple[float, float]] = []

    def _assert_files(self, paths: List[Path]) -> None:
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}")

    def load_validation_frame(self, med_path: Path, var_path: Path) -> pd.DataFrame:
        self._assert_files([med_path, var_path])
        med, var, node_cols, self.uwb_cols, self.wifi_cols = standardize_loaded_median_variance(
            med_path, var_path, coerce_numeric, infer_sensor_columns
        )

        g = self.config.grid_size_m
        self._uwb_canon: List[str] = []
        self._wifi_canon: List[str] = []
        for i, c in enumerate(self.uwb_cols):
            k = c if c in UWB_BIAS_M else UWB_COL_CANON[i]
            if k not in UWB_ANCHOR_TILES or k not in UWB_BIAS_M:
                raise ValueError(f"UWB 열 또는 Bias 정의 없음: {c!r} -> {k!r}")
            self._uwb_canon.append(k)
        for i, c in enumerate(self.wifi_cols):
            k = c if c in WIFI_BIAS_M else WIFI_COL_CANON[i]
            if k not in WIFI_AP_TILES or k not in WIFI_BIAS_M:
                raise ValueError(f"Wi-Fi 열 또는 Bias 정의 없음: {c!r} -> {k!r}")
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
        d_u = np.empty(len(self.uwb_cols), dtype=float)
        d_w = np.empty(len(self.wifi_cols), dtype=float)
        for i, c in enumerate(self.uwb_cols):
            m = row.get(f"med_{c}", np.nan)
            if not np.isfinite(m):
                d_u[i] = np.nan
                continue
            ku = self._uwb_canon[i]
            v = float(m) - UWB_BIAS_M[ku]
            if HARD_LO < v < HARD_HI:
                d_u[i] = v
            else:
                d_u[i] = np.nan
        for i, c in enumerate(self.wifi_cols):
            m = row.get(f"med_{c}", np.nan)
            if not np.isfinite(m):
                d_w[i] = np.nan
                continue
            kw = self._wifi_canon[i]
            v = float(m) - WIFI_BIAS_M[kw]
            if HARD_LO < v < HARD_HI:
                d_w[i] = v
            else:
                d_w[i] = np.nan
        return d_u, d_w

    @staticmethod
    def _soft_w(variance: float) -> float:
        if not np.isfinite(variance) or variance < 0:
            return np.nan
        return 1.0 / (np.log1p(variance) + EPS_W)

    def phase2_fuse_anchor(
        self,
        row: pd.Series,
        anchor_idx: int,
        d_uwb: float,
        d_wifi: float,
    ) -> Tuple[float, str]:
        c_u = self.uwb_cols[anchor_idx]
        c_w = self.wifi_cols[anchor_idx]
        var_u = float(row.get(f"var_{c_u}", np.nan))
        var_w = float(row.get(f"var_{c_w}", np.nan))

        uwb_dead = (not np.isfinite(d_uwb)) or (np.isfinite(var_u) and var_u > VAR_UWB_CUT)
        if uwb_dead:
            if np.isfinite(d_wifi) and np.isfinite(var_w) and var_w <= VAR_CAP_WIFI_FUSION:
                return float(d_wifi), "wifi_only_uwb_dead_or_var"
            return np.nan, "nan_both_or_no_wifi"

        wifi_dead = (
            (not np.isfinite(d_wifi))
            or (not np.isfinite(var_w))
            or (np.isfinite(var_w) and var_w > VAR_CAP_WIFI_FUSION)
        )
        if wifi_dead:
            return float(d_uwb), "uwb_only_wifi_dead_or_bad_var"

        if d_uwb >= ASYMM_UWB_DOM * d_wifi:
            wu, ww = 0.1, 0.9
            return wu * d_uwb + ww * d_wifi, "asym_uwb_dom_multipath"
        if d_wifi >= ASYMM_UWB_DOM * d_uwb:
            wu, ww = 0.3, 0.7
            return wu * d_uwb + ww * d_wifi, "asym_wifi_dom_underest"

        w_u = self._soft_w(var_u)
        w_w = self._soft_w(var_w)
        if not (np.isfinite(w_u) and np.isfinite(w_w) and (w_u + w_w) > 0):
            return np.nan, "nan_soft_weight"
        return (w_u * d_uwb + w_w * d_wifi) / (w_u + w_w), "soft_logvar"

    def row_d_fused_with_rules(self, row: pd.Series) -> Tuple[np.ndarray, List[str]]:
        d_u, d_w = self.phase1_calibrate_and_clip(row)
        fused = np.empty(6, dtype=float)
        rules: List[str] = []
        for i in range(6):
            fused[i], r = self.phase2_fuse_anchor(row, i, d_u[i], d_w[i])
            rules.append(r)
        return fused, rules

    def phase3_trilaterate(self, d_fused: np.ndarray) -> Tuple[float, float, float, int]:
        pos = np.array(self._anchor_xy_m, dtype=float)
        valid = np.isfinite(d_fused)
        n_v = int(valid.sum())
        if n_v < 3:
            return np.nan, np.nan, np.nan, n_v
        P = pos[valid]
        d = d_fused[valid].astype(float)

        j = int(np.argmin(d))
        x0 = np.array(P[j], dtype=float)

        def residual(xy: np.ndarray) -> np.ndarray:
            pr = np.sqrt((P[:, 0] - xy[0]) ** 2 + (P[:, 1] - xy[1]) ** 2)
            return pr - d

        mad = float(np.median(np.abs(d - np.median(d))))
        f_scale = float(max(0.5, 0.2 * np.median(d), 3.0 * mad if mad > 0 else 0.5))

        res = least_squares(
            residual,
            x0=x0,
            bounds=(np.array(BOUNDS_LO, dtype=float), np.array(BOUNDS_HI, dtype=float)),
            loss="huber",
            f_scale=f_scale,
            method="trf",
        )
        return float(res.x[0]), float(res.x[1]), f_scale, n_v

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
        conf = "High" if q < QUALITY_RMSE_TH else "Low"
        return q, conf

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        xs, ys, qs, confs = [], [], [], []
        diag_rows: List[Dict] = []
        for sample_i, (_, row) in enumerate(out.iterrows()):
            d_u, d_w = self.phase1_calibrate_and_clip(row)
            d_f, rules = self.row_d_fused_with_rules(row)
            x, y, f_sc, n_v = self.phase3_trilaterate(d_f)
            q, c = self.phase4_quality((x, y), self._anchor_xy_m, d_f)
            xs.append(x)
            ys.append(y)
            qs.append(q)
            confs.append(c)
            rec: Dict = {"sample_index": sample_i}
            for i in range(6):
                rec[f"p1_uwb_cal_{i}"] = d_u[i]
                rec[f"p1_wifi_cal_{i}"] = d_w[i]
                rec[f"p2_rule_{i}"] = rules[i]
                rec[f"p2_d_fused_{i}"] = d_f[i]
            rec["p3_n_fused_valid"] = n_v
            rec["p3_f_scale"] = f_sc
            rec["p4_quality_rmse_m"] = q
            rec["p4_confidence"] = c
            diag_rows.append(rec)
        out["Pred_X"] = xs
        out["Pred_Y"] = ys
        out["Quality_RMSE_m"] = qs
        out["Confidence"] = confs
        self._last_diag = pd.DataFrame(diag_rows)
        return out

    def run_wifi_only_phase3(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """UWB를 쓰지 않고 phase1 Wi-Fi 보정 거리만으로 phase3 삼변(동일 앵커 기하)."""
        xs: List[float] = []
        ys: List[float] = []
        for _, row in df.iterrows():
            _du, d_w = self.phase1_calibrate_and_clip(row)
            x, y, _fsc, _nv = self.phase3_trilaterate(d_w)
            xs.append(x)
            ys.append(y)
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def position_errors(true_x: np.ndarray, true_y: np.ndarray, pred_x: np.ndarray, pred_y: np.ndarray) -> np.ndarray:
    return np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)


def print_and_save_v4_diagnostics(loc: FusionLocalizerV4, diag: pd.DataFrame, out_dir: Path) -> None:
    n = len(diag)
    print("\n--- V4 단계별 진단 요약 ---\n")
    # Phase 1: 유효 캘리브 개수 (앵커당)
    for i in range(6):
        nu = np.isfinite(diag[f"p1_uwb_cal_{i}"]).sum()
        nw = np.isfinite(diag[f"p1_wifi_cal_{i}"]).sum()
        print(f"  Phase1 앵커{i}: UWB 유효 {nu}/{n}, Wi-Fi 유효 {nw}/{n}")

    # Phase 2: 규칙 빈도 (전 앵커 long)
    rules_all: List[str] = []
    for i in range(6):
        rules_all.extend(diag[f"p2_rule_{i}"].astype(str).tolist())
    ctr = Counter(rules_all)
    print("\n  Phase2 융합 규칙 빈도 (앵커×샘플):")
    for k, v in ctr.most_common():
        print(f"    {k}: {v}")

    n_valid_dist = diag[[f"p2_d_fused_{i}" for i in range(6)]].apply(np.isfinite, axis=1).sum(axis=1)
    print("\n  샘플당 유효 d_fused 개수 분포:")
    print(n_valid_dist.value_counts().sort_index().to_string())

    nv3 = (diag["p3_n_fused_valid"] >= 3).sum()
    print(f"\n  Phase3: 삼변측량 가능 샘플 (n_fused>=3): {nv3}/{n}")

    agg = {
        "n_samples": n,
        "phase2_rule_counts": dict(ctr),
        "n_fused_per_sample": n_valid_dist.value_counts().to_dict(),
        "phase3_ge3": int(nv3),
    }
    agg_path = out_dir / "v4_phase_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    wide_path = out_dir / "v4_per_sample_diagnostics.csv"
    diag.to_csv(wide_path, index=False)

    # 앵커별 Phase2 규칙 CSV (참고용)
    rows = []
    for i in range(6):
        vc = diag[f"p2_rule_{i}"].astype(str).value_counts()
        for rule, cnt in vc.items():
            rows.append({"anchor_index": i, "fusion_rule": rule, "count": int(cnt)})
    pd.DataFrame(rows).to_csv(out_dir / "v4_fusion_rule_by_anchor.csv", index=False)


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
        train_med_path, train_var_path = resolve_train_kgh_corrected_paths(train_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    try:
        val_m_path, val_v_path = resolve_validation_paths(val_dir)
        med_path, var_path = val_m_path, val_v_path
        data_label = "validation"
    except FileNotFoundError:
        med_path, var_path = train_med_path, train_var_path
        data_label = "train (검증 파일 없음, 원본 train)"

    loc = FusionLocalizerV4(Config())
    df = loc.load_validation_frame(med_path, var_path)
    print(f"V4 입력 행 수(유실 없음): {len(df)}")
    wfx, wfy = loc.run_wifi_only_phase3(df)
    err_wifi = position_errors(df["True_X"].values, df["True_Y"].values, wfx, wfy)
    fin_w = np.isfinite(err_wifi)
    ewf = err_wifi[fin_w]
    rmse_wifi = (
        float(np.sqrt(mean_squared_error(np.zeros(len(ewf)), ewf))) if len(ewf) else float("nan")
    )
    mae_wifi = float(mean_absolute_error(np.zeros(len(ewf)), ewf)) if len(ewf) else float("nan")
    print(f"\n=== V4 Wi-Fi-only (UWB 미사용, phase1 Wi-Fi + phase3) — {data_label} ===")
    print(f"Wi-Fi-only RMSE: {rmse_wifi:.4f} m | MAE: {mae_wifi:.4f} m (n={len(ewf)})\n")

    pred_df = loc.run(df)
    diag = getattr(loc, "_last_diag", pd.DataFrame())
    if len(diag):
        merged = pd.concat([pred_df.reset_index(drop=True), diag.reset_index(drop=True)], axis=1)
        merged.to_csv(out_dir / "v4_predictions_with_phase_columns.csv", index=False)
        print_and_save_v4_diagnostics(loc, diag, out_dir)

    fin = (
        np.isfinite(pred_df["True_X"].values.astype(float))
        & np.isfinite(pred_df["True_Y"].values.astype(float))
        & np.isfinite(pred_df["Pred_X"].values.astype(float))
        & np.isfinite(pred_df["Pred_Y"].values.astype(float))
    )
    ok = pred_df.loc[fin]
    if len(ok) == 0:
        print("유효한 예측 좌표가 없습니다 (모든 샘플 삼변측량 실패).", file=sys.stderr)
        return 1
    err = position_errors(
        ok["True_X"].values, ok["True_Y"].values, ok["Pred_X"].values, ok["Pred_Y"].values
    )
    rmse_all = float(np.sqrt(mean_squared_error(np.zeros(len(err)), err)))
    mae_all = float(mean_absolute_error(np.zeros(len(err)), err))

    hi = ok[ok["Confidence"] == "High"]
    if len(hi) > 0:
        err_h = position_errors(hi["True_X"].values, hi["True_Y"].values, hi["Pred_X"].values, hi["Pred_Y"].values)
        rmse_hi = float(np.sqrt(mean_squared_error(np.zeros(len(err_h)), err_h)))
    else:
        rmse_hi = float("nan")

    print(f"\n=== V4 ({data_label}) ===\n")
    print(f"전체 RMSE: {rmse_all:.4f} m | MAE: {mae_all:.4f} m  (n={len(err)})")
    print(f'Confidence="High"만 RMSE: {rmse_hi:.4f} m  (n={len(hi)})')

    save = pred_df[
        ["Node_x", "Node_y", "True_X", "True_Y", "Pred_X", "Pred_Y", "Confidence", "Quality_RMSE_m"]
    ].copy()
    save["Error_m"] = position_errors(
        save["True_X"].values, save["True_Y"].values, save["Pred_X"].values, save["Pred_Y"].values
    )
    csv_path = out_dir / "v4_validation_predictions.csv"
    save.to_csv(csv_path, index=False)

    err_sorted = np.sort(err)
    cdf = np.arange(1, len(err_sorted) + 1) / len(err_sorted)
    plt.figure(figsize=(7, 4.5))
    plt.plot(err_sorted, cdf, color="tab:blue", lw=2, label="V4 position error")
    plt.axvline(2.0, color="gray", ls="--", lw=1, label="2 m goal")
    plt.xlabel("Position error (m)")
    plt.ylabel("Cumulative probability")
    plt.title("V4 error CDF (validation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    cdf_path = out_dir / "v4_error_cdf.png"
    plt.savefig(cdf_path, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6.5))
    plt.scatter(save["True_X"], save["True_Y"], c="blue", s=36, alpha=0.85, label="True", zorder=2)
    hi_mask = save["Confidence"] == "High"
    lo_mask = save["Confidence"] == "Low"
    plt.scatter(
        save.loc[hi_mask, "Pred_X"],
        save.loc[hi_mask, "Pred_Y"],
        c="green",
        s=32,
        alpha=0.75,
        label="Pred (High)",
        zorder=3,
    )
    plt.scatter(
        save.loc[lo_mask, "Pred_X"],
        save.loc[lo_mask, "Pred_Y"],
        c="red",
        s=32,
        alpha=0.75,
        label="Pred (Low)",
        zorder=3,
    )
    plt.xlim(0, 12)
    plt.ylim(0, 18)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("V4 confidence map (12 m x 18 m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    map_path = out_dir / "v4_confidence_map.png"
    plt.savefig(map_path, dpi=160)
    plt.close()

    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main, output_artifact_include_prefixes=("v4_",))
