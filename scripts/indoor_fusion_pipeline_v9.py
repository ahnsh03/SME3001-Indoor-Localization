"""
V9: Pure Wi-Fi 1차 삼변 → Wi-Fi 안심좌표 기준 UWB 기하 게이트 → 융합 삼변 (Huber + bounds).

검증 세트에서 gate_threshold 그리드를 탐색해 최소 RMSE를 선택하고, CSV·CDF·Kill 로그를 저장.

주의: 위 방식은 검증 정답으로 gate를 고르는 **선택 편향**이 있다.
무결성 평가는 동일 폴더의 `indoor_fusion_pipeline_v9_strict.py`(Train K-Fold로만 gate 선택)를 실행할 것.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fusion_realtime_sanitize import resolve_train_kgh_corrected_paths, resolve_validation_paths
from fusion_realtime_sanitize import VAR_CAP_UWB_FUSION, VAR_CAP_WIFI_FUSION
from indoor_fusion_pipeline_v8 import (
    Config,
    FusionLocalizerV8,
    ROBUST_BIAS_M,
    UWB_COL_CANON,
    UWB_BIAS_W,
    WIFI_BIAS_W,
    WIFI_COL_CANON,
    EPS_W,
    sensor_canon_key,
    tile_to_m,
    WIFI_AP_TILES,
)

if sys.platform == "win32":
    try:
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

ALL_KEYS = UWB_COL_CANON + WIFI_COL_CANON

# Train Clean UWB 잔차(sensor_spatial_profiler 순차 필터 후 |e| 분포의 P90~P95 근처)까지 포함해 탐색.
# V9 게이트는 |d_geom(Wi‑Fi 초기좌표) − d_meas(UWB)| 이므로 ranging MAE와 직역 같지 않지만,
# 그리드를 좁히면 게이트가 과도하게 엄격해지기 쉬움 → 상한을 넉넉히 둠.
_GATE_DENSE = [round(float(x), 2) for x in np.linspace(0.85, 8.50, 30)]
_GATE_LEGACY = [0.85, 1.03, 1.36, 1.54, 2.0, 2.5, 3.0]
GATE_THRESH_GRID = sorted(set(_GATE_LEGACY + _GATE_DENSE))


class FusionLocalizerV9(FusionLocalizerV8):
    def _row_step_b(
        self,
        row: pd.Series,
        wifi_x: float,
        wifi_y: float,
        gate_t: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_l: List[Tuple[float, float]] = []
        dist_l: List[float] = []
        w_l: List[float] = []

        if not np.isfinite(wifi_x) or not np.isfinite(wifi_y):
            return self._row_wifi_only(row)

        anchor_xy_m = dict(
            (sensor_canon_key(c, i, True), self._resolve_sensor_position_m(c, i, True))
            for i, c in enumerate(self.uwb_cols)
        )

        for i, c in enumerate(self.uwb_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan:
                continue
            if float(v) > VAR_CAP_UWB_FUSION:
                continue
            if not np.isfinite(d_raw) or not np.isfinite(v):
                continue
            key_u = sensor_canon_key(c, i, True)
            d_meas = float(d_raw) - ROBUST_BIAS_M[key_u]
            ax_u, ay_u = anchor_xy_m[key_u]
            d_geom = float(np.hypot(wifi_x - ax_u, wifi_y - ay_u))
            if abs(d_geom - d_meas) > gate_t:
                continue
            pos_l.append((ax_u, ay_u))
            dist_l.append(d_meas)
            w_l.append(1.0 / (float(v) + UWB_BIAS_W + EPS_W))

        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d_raw)) or (not np.isfinite(v)) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            key_w = sensor_canon_key(c, i, False)
            pos_l.append(self._resolve_sensor_position_m(c, i, False))
            dist_l.append(float(d_raw) - ROBUST_BIAS_M[key_w])
            w_l.append(1.0 / (float(v) + WIFI_BIAS_W + EPS_W))

        return (
            np.asarray(pos_l, dtype=float),
            np.asarray(dist_l, dtype=float),
            np.asarray(w_l, dtype=float),
        )

    def predict_step_b_series(
        self, df: pd.DataFrame, gate_t: float, xa: np.ndarray, ya: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        xs: List[float] = []
        ys: List[float] = []
        for k, (_, row) in enumerate(df.iterrows()):
            wx, wy = float(xa[k]), float(ya[k])
            p, d, w = self._row_step_b(row, wx, wy, gate_t)
            xy = self._solve_trilat(p, d, w)
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    def accumulate_kill_log(
        self,
        df: pd.DataFrame,
        xa: np.ndarray,
        ya: np.ndarray,
        gate_t: float,
    ) -> Tuple[pd.DataFrame, DefaultDict[str, int], DefaultDict[str, int]]:
        var_drop: DefaultDict[str, int] = defaultdict(int)
        geom_drop: DefaultDict[str, int] = defaultdict(int)
        anchor_xy_key = [(sensor_canon_key(c, i, True), i, c) for i, c in enumerate(self.uwb_cols)]

        for k, (_, row) in enumerate(df.iterrows()):
            wx, wy = float(xa[k]), float(ya[k])
            for key_u, i, c in anchor_xy_key:
                d_raw = row.get(f"med_{c}", np.nan)
                v = row.get(f"var_{c}", np.nan)
                was_nan = row.get(f"orig_nan_{c}", 1) == 1
                if was_nan or not np.isfinite(d_raw):
                    continue
                if not np.isfinite(v):
                    continue
                if float(v) > VAR_CAP_UWB_FUSION:
                    var_drop[key_u] += 1
                    continue
                if not (np.isfinite(wx) and np.isfinite(wy)):
                    geom_drop[key_u] += 1
                    continue
                d_meas = float(d_raw) - ROBUST_BIAS_M[key_u]
                ax_u, ay_u = self._resolve_sensor_position_m(c, i, True)
                d_geom = float(np.hypot(wx - ax_u, wy - ay_u))
                if abs(d_geom - d_meas) > gate_t:
                    geom_drop[key_u] += 1

            for i, c in enumerate(self.wifi_cols):
                vw = row.get(f"var_{c}", np.nan)
                mw = row.get(f"med_{c}", np.nan)
                kw = sensor_canon_key(c, i, False)
                if np.isfinite(mw) and np.isfinite(vw) and float(vw) > VAR_CAP_WIFI_FUSION:
                    var_drop[kw] += 1

        rows = []
        for key in ALL_KEYS:
            rows.append(
                {
                    "sensor": key,
                    "Var_gt_6_drop_count": int(var_drop[key]),
                    "Geometric_gate_drop_count": int(geom_drop[key]),
                }
            )
        return pd.DataFrame(rows), var_drop, geom_drop


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
    loc = FusionLocalizerV9(cfg)
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "train"
    val_dir = root / "data" / "validation"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_med, train_var = resolve_train_kgh_corrected_paths(train_dir)
    val_med, val_var = resolve_validation_paths(val_dir)
    _, val = loc.load_datasets(
        train_median_path=train_med,
        train_variance_path=train_var,
        val_median_path=val_med,
        val_variance_path=val_var,
    )
    print(f"  V9 데이터 행 수: 검증 len={len(val)}")

    tx = val["True_X"].values.astype(float)
    ty = val["True_Y"].values.astype(float)

    xa, ya = loc.predict_step(val, "A")
    err_a = loc.point_errors(tx, ty, xa, ya)
    rmse_a, mae_a = loc.rmse_mae(err_a)

    print("\n" + "=" * 72)
    print("V9 — Step A: Pure Wi-Fi (validation)")
    print("=" * 72)
    print(f"  RMSE = {rmse_a:.4f} m   |   MAE = {mae_a:.4f} m")
    print("=" * 72)

    best_t = GATE_THRESH_GRID[0]
    best_rmse = 1e9
    xf, yf = xa, ya
    for gt in GATE_THRESH_GRID:
        xb, yb = loc.predict_step_b_series(val, gt, xa, ya)
        err = loc.point_errors(tx, ty, xb, yb)
        r, _ = loc.rmse_mae(err)
        if r < best_rmse:
            best_rmse = r
            best_t = gt
            xf, yf = xb, yb

    err_b = loc.point_errors(tx, ty, xf, yf)
    rmse_b, mae_b = loc.rmse_mae(err_b)

    g = cfg.grid_size_m
    qc_b: List[float] = []
    for i, (_, row) in enumerate(val.iterrows()):
        px, py = float(xf[i]), float(yf[i])
        es: List[float] = []
        if np.isfinite(px) and np.isfinite(py):
            for wi, c in enumerate(loc.wifi_cols):
                m_raw = row.get(f"med_{c}", np.nan)
                if not np.isfinite(m_raw):
                    continue
                key_w = sensor_canon_key(c, wi, False)
                ax, ay = tile_to_m(WIFI_AP_TILES[key_w], g)
                d_cal = float(m_raw) - ROBUST_BIAS_M[key_w]
                geom = float(np.hypot(px - ax, py - ay))
                es.append((geom - d_cal) ** 2)
        qc_b.append(float(np.sqrt(np.mean(es))) if es else float("nan"))

    print("\nV9 — Step B: Wi-Fi guided UWB gating + fusion (validation)")
    print(f"  선택 gate_threshold = {best_t} m")
    print(f"  RMSE = {rmse_b:.4f} m   |   MAE = {mae_b:.4f} m")
    print()

    kill_df, _, _ = loc.accumulate_kill_log(val, xa, ya, best_t)

    pred = val[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xa
    pred["StepA_Y"] = ya
    pred["Final_X"] = xf
    pred["Final_Y"] = yf
    pred["Error_StepA_m"] = err_a
    pred["Error_Final_m"] = err_b
    pred["Quality_RMSE_m"] = np.asarray(qc_b, dtype=float)
    pred.to_csv(out_dir / "v9_predictions.csv", index=False, encoding="utf-8-sig")
    kill_df.to_csv(out_dir / "v9_uwb_kill_log.csv", index=False, encoding="utf-8-sig")

    worst = pred.assign(_e=err_b).nlargest(10, "_e")
    print("Step B 기준 Worst 오차 노드 Top 10")
    print("-" * 72)
    for _, r in worst.iterrows():
        print(
            f"  Node({int(r['Node_x'])}, {int(r['Node_y'])}) | "
            f"True ({r['True_X']:.2f}, {r['True_Y']:.2f}) m | "
            f"error = {r['_e']:.3f} m"
        )

    ea = err_a[np.isfinite(err_a)]
    eb = err_b[np.isfinite(err_b)]
    plt.figure(figsize=(8.5, 5))
    for e, lab, sty in [(ea, "Step A: Pure Wi-Fi", "tab:blue"), (eb, f"Step B: V9 (gate={best_t} m)", "tab:red")]:
        if e.size == 0:
            continue
        s = np.sort(e)
        cdf = np.arange(1, len(s) + 1) / len(s)
        plt.plot(s, cdf, lw=2, label=lab)
    plt.xlabel("Position error (m)")
    plt.ylabel("Cumulative probability")
    plt.title("V9: Step A vs Step B error CDF (validation)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "v9_step_cdf.png", dpi=200)
    plt.close()

    print()
    print(f"저장: {out_dir / 'v9_predictions.csv'}")
    print(f"저장: {out_dir / 'v9_uwb_kill_log.csv'}")
    print(f"저장: {out_dir / 'v9_step_cdf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
