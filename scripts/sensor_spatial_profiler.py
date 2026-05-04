"""
실내 UWB/Wi-Fi 통제 실험: (1) 센서 편향·공간 EDA, (2) UWB 순차 3단계 필터 시각화.

입력: 공식 Train `median_test_kgh_corrected.*` + `variance_test_kgh_corrected.*` (data/train/, csv 우선)
실행: py -3 scripts/sensor_spatial_profiler.py [all|spatial|uwb]   (기본 all)

출력: outputs/sensor_profiler_eda_*.png, outputs/uwb_filter_sequential_filtering.png
환경변수: MEDIAN_XLSX, VARIANCE_XLSX, PROFILER_OUT_DIR, UWB_FILTER_PNG
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 경로 ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "train"
from fusion_realtime_sanitize import (
    VAR_CAP_UWB_FUSION,
    clip_wifi_variance_columns,
    load_sensor_tabular,
    normalize_column_headers,
    resolve_train_kgh_corrected_paths,
)

try:
    _DEF_MEDIAN, _DEF_VARIANCE = resolve_train_kgh_corrected_paths(DEFAULT_TRAIN)
except FileNotFoundError:
    _DEF_MEDIAN = DEFAULT_TRAIN / "median_test_kgh_corrected.csv"
    _DEF_VARIANCE = DEFAULT_TRAIN / "variance_test_kgh_corrected.csv"
MEDIAN_PATH = Path(_DEF_MEDIAN)
VARIANCE_PATH = Path(_DEF_VARIANCE)
OUT_DIR = PROJECT_ROOT / "outputs"
UWB_FILTER_OUT_PNG = OUT_DIR / "uwb_filter_sequential_filtering.png"

GRID_M = 0.6
ERROR_ABS_CAP_M = 15.0
TRIM_PROP = 0.10  # 상·하위 각 10% 절사
BOXPLOT_YLIM = (-10.0, 20.0)
ROOM_X_M = 12.0
ROOM_Y_M = 18.0

# --- UWB 순차 필터 (UWB 6앵커만) ---
UWB_FILTER_HARD_LO = 0.5
UWB_FILTER_HARD_HI = 22.0
MAD_MULT = 3.0
MAD_MIN_BAND = 1.5

# 타일 좌표 (그리드 단위) — 0.6m 곱하면 미터
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
WIFI_ANCHOR_ORDER: List[str] = ["SW_11", "SW_first_team", "볼링공", "SW_4", "SW_6", "SW_5"]
UWB_ANCHOR_ORDER: List[str] = ["110394ab", "e63ce2f", "8e610981", "d10485af", "d1044709", "4e610206"]


def load_sensor_excel(path: Path) -> pd.DataFrame:
    """과제용 xlsx는 (1) 첫 행이 문자 헤더이고 header=False 저장, 또는 (2) 퓨전 파이프라인처럼 read_excel 기본 헤더 다음 행에 다시 헤더가 있는 형식이 혼재합니다."""
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
    node_cols = cols[:2]
    uwb_cols = cols[2:8]
    wifi_cols = cols[8:14]
    return node_cols, uwb_cols, wifi_cols


def load_median_variance_sanitized(median_path: Path, variance_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = coerce_numeric(normalize_column_headers(load_sensor_tabular(Path(median_path))))
    var = coerce_numeric(normalize_column_headers(load_sensor_tabular(Path(variance_path))))
    _nc, _uwb, wifi = infer_sensor_columns(med)
    clip_wifi_variance_columns(var, wifi)
    return med, var


def anchor_xy_m(anchor_tiles: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    return {k: (v[0] * GRID_M, v[1] * GRID_M) for k, v in anchor_tiles.items()}


def anchor_xy_for_sensor(
    col: str,
    kind_uwb: bool,
    uwb_cols: List[str],
    wifi_cols: List[str],
) -> Tuple[float, float]:
    uwb_m = anchor_xy_m(UWB_ANCHOR_TILES)
    wifi_m = anchor_xy_m(WIFI_AP_TILES)
    if kind_uwb:
        if col in uwb_m:
            return uwb_m[col]
        idx = uwb_cols.index(col) if col in uwb_cols else 0
        return uwb_m[UWB_ANCHOR_ORDER[idx]]
    if col in wifi_m:
        return wifi_m[col]
    idx = wifi_cols.index(col) if col in wifi_cols else 0
    return wifi_m[WIFI_ANCHOR_ORDER[idx]]


def true_distance_m(nx: np.ndarray, ny: np.ndarray, ax: float, ay: float) -> np.ndarray:
    px, py = nx.astype(float) * GRID_M, ny.astype(float) * GRID_M
    return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)


def clean_errors_for_robust(raw_err: np.ndarray) -> np.ndarray:
    """NaN 제외 후 |e|<=15, 그 다음 상·하위 10% 절사."""
    e = raw_err[np.isfinite(raw_err)]
    if e.size == 0:
        return e
    e = e[np.abs(e) <= ERROR_ABS_CAP_M]
    if e.size == 0:
        return e
    lo = np.quantile(e, TRIM_PROP)
    hi = np.quantile(e, 1.0 - TRIM_PROP)
    return e[(e >= lo) & (e <= hi)]


def catastrophic_mask(med: np.ndarray, true_d: np.ndarray) -> np.ndarray:
    """NaN median 또는 |error| >= 15."""
    err = med - true_d
    bad = ~np.isfinite(med) | (np.abs(err) >= ERROR_ABS_CAP_M)
    return bad


def analysis_a_tables(
    med_df: pd.DataFrame,
    var_df: pd.DataFrame,
    node_cols: List[str],
    uwb_cols: List[str],
    wifi_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    nx = med_df[node_cols[0]].values
    ny = med_df[node_cols[1]].values
    n = len(med_df)

    rows = []
    raw_errors_by_sensor: Dict[str, np.ndarray] = {}

    sensor_specs: List[Tuple[str, str, str]] = []
    for c in uwb_cols:
        sensor_specs.append((c, "UWB", c))
    for c in wifi_cols:
        sensor_specs.append((c, "Wi-Fi", c))

    for col, kind, label in sensor_specs:
        ax, ay = anchor_xy_for_sensor(col, kind == "UWB", uwb_cols, wifi_cols)
        true_d = true_distance_m(nx, ny, ax, ay)
        med = med_df[col].values.astype(float)
        raw_err = med - true_d
        raw_errors_by_sensor[col] = raw_err.copy()

        clean = clean_errors_for_robust(raw_err)
        robust_bias = float(np.median(clean)) if clean.size else np.nan
        trimmed_mean_err = float(np.mean(clean)) if clean.size else np.nan

        bad = catastrophic_mask(med, true_d)
        cat_pct = 100.0 * float(bad.sum()) / max(n, 1)

        v = var_df[col].values.astype(float)
        vfin = v[np.isfinite(v)]
        p50 = float(np.percentile(vfin, 50)) if vfin.size else np.nan
        p90 = float(np.percentile(vfin, 90)) if vfin.size else np.nan
        p99 = float(np.percentile(vfin, 99)) if vfin.size else np.nan

        rows.append(
            {
                "Sensor": label,
                "Type": kind,
                "Robust Bias (m)": robust_bias,
                "Trimmed Mean Err (m)": trimmed_mean_err,
                "Catastrophic %": cat_pct,
                "Var P50": p50,
                "Var P90": p90,
                "Var P99": p99,
            }
        )

    summary = pd.DataFrame(rows)
    return summary, raw_errors_by_sensor


def analysis_b_spatial(
    med_df: pd.DataFrame,
    node_cols: List[str],
    uwb_cols: List[str],
    wifi_cols: List[str],
) -> None:
    nx = med_df[node_cols[0]].values
    ny = med_df[node_cols[1]].values
    keys = np.array([f"{int(x)},{int(y)}" for x, y in zip(nx, ny)])

    uwb_mat = np.column_stack([med_df[c].values.astype(float) for c in uwb_cols])
    uwb_nan_count = np.sum(~np.isfinite(uwb_mat), axis=1)
    df_u = pd.DataFrame({"key": keys, "Node_x": nx, "Node_y": ny, "uwb_nan": uwb_nan_count})
    top_nan = df_u.sort_values("uwb_nan", ascending=False).drop_duplicates("key").head(5)

    ratio_rows = []
    uniq_keys = np.unique(keys)
    for k in uniq_keys:
        m = keys == k
        ratios = []
        for c in uwb_cols + wifi_cols:
            ax, ay = anchor_xy_for_sensor(c, c in uwb_cols, uwb_cols, wifi_cols)
            td = true_distance_m(nx[m], ny[m], ax, ay).ravel()
            med = med_df.loc[m, c].values.astype(float)
            ok = np.isfinite(med) & np.isfinite(td) & (td > 0.2)
            if ok.any():
                ratios.append(float(np.nanmean(med[ok] / td[ok])))
        if ratios:
            ratio_rows.append(
                {
                    "key": k,
                    "Node_x": int(nx[m][0]),
                    "Node_y": int(ny[m][0]),
                    "mean_meas_over_true": float(np.nanmean(ratios)),
                }
            )
    df_r = pd.DataFrame(ratio_rows).sort_values("mean_meas_over_true", ascending=False).head(5)

    print("\n=== Analysis B: 공간(Spatial) ===\n")
    print("[UWB 결측(NaN) 최다 사각지대 Top 5 - 그리드 (Node_x, Node_y), UWB NaN 건수]")
    for _, r in top_nan.iterrows():
        print(f"  ({int(r['Node_x'])}, {int(r['Node_y'])}): UWB NaN count = {int(r['uwb_nan'])}")
    print("\n[측정/실제 비율 평균이 가장 큰 좌표 Top 5 - 다중경로/과대평가 의심]")
    for _, r in df_r.iterrows():
        print(
            f"  ({int(r['Node_x'])}, {int(r['Node_y'])}): "
            f"mean(median/true) over sensors = {r['mean_meas_over_true']:.3f}"
        )


def plot_scatter_calibration(
    med_df: pd.DataFrame,
    node_cols: List[str],
    uwb_cols: List[str],
    wifi_cols: List[str],
    out_path: Path,
) -> None:
    nx = med_df[node_cols[0]].values
    ny = med_df[node_cols[1]].values
    cols = uwb_cols + wifi_cols
    kinds = ["UWB"] * 6 + ["Wi-Fi"] * 6
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()
    for axp, col, kind in zip(axes, cols, kinds):
        axy = anchor_xy_for_sensor(col, kind == "UWB", uwb_cols, wifi_cols)
        td = true_distance_m(nx, ny, axy[0], axy[1])
        md = med_df[col].values.astype(float)
        ok = np.isfinite(td) & np.isfinite(md)
        axp.scatter(td[ok], md[ok], s=8, alpha=0.5, c="steelblue")
        lim_max = float(np.nanpercentile(np.r_[td[ok], md[ok]], 99)) if ok.any() else 1.0
        lim_max = max(lim_max, 1.0)
        axp.plot([0, lim_max], [0, lim_max], "r--", lw=1, label="y=x")
        disp = col
        if kind == "UWB" and col not in UWB_ANCHOR_TILES:
            disp = UWB_ANCHOR_ORDER[uwb_cols.index(col)]
        elif kind == "Wi-Fi" and col not in WIFI_AP_TILES:
            disp = WIFI_ANCHOR_ORDER[wifi_cols.index(col)]
        axp.set_title(f"{kind}: {disp}", fontsize=9)
        axp.set_xlabel("True distance (m)")
        axp.set_ylabel("Measured median (m)")
        axp.set_aspect("auto")
    fig.suptitle("Sensor calibration: measured median vs true distance", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


def plot_error_boxplot(raw_by_sensor: Dict[str, np.ndarray], order: List[str], out_path: Path) -> None:
    data = [raw_by_sensor[c][np.isfinite(raw_by_sensor[c])] for c in order]
    labels = order
    fig, ax = plt.subplots(figsize=(14, 5))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_ylim(BOXPLOT_YLIM)
    ax.set_ylabel("Raw error (m) = median - true")
    ax.set_title("Raw error distribution by sensor (outliers hidden; y clipped for body visibility)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


def plot_spatial_mae_heatmap(
    med_df: pd.DataFrame,
    node_cols: List[str],
    uwb_cols: List[str],
    wifi_cols: List[str],
    out_path: Path,
) -> None:
    nx = med_df[node_cols[0]].values.astype(float)
    ny = med_df[node_cols[1]].values.astype(float)
    xm, ym = nx * GRID_M, ny * GRID_M

    abs_errs = []
    for c in uwb_cols + wifi_cols:
        ax_a, ay_a = anchor_xy_for_sensor(c, c in uwb_cols, uwb_cols, wifi_cols)
        td = true_distance_m(nx, ny, ax_a, ay_a)
        md = med_df[c].values.astype(float)
        abs_errs.append(np.abs(md - td))
    abs_stack = np.column_stack(abs_errs)
    mae = np.nanmean(abs_stack, axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(xm, ym, c=mae, cmap="inferno", s=80, alpha=0.85, edgecolors="k", linewidths=0.2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean |error| over 12 sensors (m)")

    for name, (tx, ty) in UWB_ANCHOR_TILES.items():
        ax.scatter(tx * GRID_M, ty * GRID_M, marker="^", s=120, c="cyan", edgecolors="k", zorder=5, label="_UWB" if name != "110394ab" else "UWB anchor")
    for name, (tx, ty) in WIFI_AP_TILES.items():
        ax.scatter(tx * GRID_M, ty * GRID_M, marker="s", s=100, c="lime", edgecolors="k", zorder=5, label="_WiFi" if name != "SW_11" else "Wi-Fi AP")

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="cyan", markeredgecolor="k", markersize=10, label="UWB anchor"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="lime", markeredgecolor="k", markersize=9, label="Wi-Fi AP"),
    ]
    ax.legend(handles=legend_elems, loc="upper right")
    ax.set_xlim(0, ROOM_X_M)
    ax.set_ylim(0, ROOM_Y_M)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Spatial MAE heatmap ({ROOM_X_M:.0f} m x {ROOM_Y_M:.0f} m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


def print_actionable(summary: pd.DataFrame, uwb_cols: List[str], wifi_cols: List[str], var_df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("Actionable 요약 (알고리즘 V4 설계용)")
    print("=" * 72)

    print("\n1) Calibration Offsets - 삼변측량 전 각 기기 측정값에서 빼야 할 Robust Bias (m):")
    offsets = []
    for _, r in summary.iterrows():
        v = r["Robust Bias (m)"]
        if np.isfinite(v):
            offsets.append(f"  {r['Sensor']}: {v:+.4f}")
        else:
            offsets.append(f"  {r['Sensor']}: NaN (데이터 부족)")
    print("\n".join(offsets))

    uwb_vars = np.concatenate([var_df[c].values.astype(float) for c in uwb_cols])
    wifi_vars = np.concatenate([var_df[c].values.astype(float) for c in wifi_cols])
    uwb_vars = uwb_vars[np.isfinite(uwb_vars)]
    wifi_vars = wifi_vars[np.isfinite(wifi_vars)]

    p95_uwb = float(np.percentile(uwb_vars, 95)) if uwb_vars.size else np.nan
    p95_wifi = float(np.percentile(wifi_vars, 95)) if wifi_vars.size else np.nan

    print("\n2) Variance Cap Threshold 제안 (가중치 0 / Drop 기준):")
    print(f"   UWB 전체 분산 값의 95% 백분위수 ≈ {p95_uwb:.6g}")
    print(f"   Wi-Fi 전체 분산 값의 95% 백분위수 ≈ {p95_wifi:.6g}")
    print(
        "   권장: 융합 시 분산이 위 값을 초과하면 신뢰도가 급격히 떨어진다고 보고 "
        "가중치 0(또는 Huber/게이트) 처리. 필요 시 P99로 더 보수적으로 설정 가능."
    )
    print("=" * 72 + "\n")


# ----- UWB 순차 필터 (melt + 3단계) -----


def _true_dist_scalar(nx: float, ny: float, ax: float, ay: float) -> float:
    px, py = float(nx) * GRID_M, float(ny) * GRID_M
    return float(np.hypot(px - ax, py - ay))


def _mad_threshold(errors: np.ndarray) -> Tuple[float, float]:
    e = errors[np.isfinite(errors)]
    if e.size == 0:
        return np.nan, np.nan
    med = float(np.median(e))
    mad = float(np.median(np.abs(e - med)))
    return med, mad


def _build_long_uwb(med_df: pd.DataFrame, var_df: pd.DataFrame, node_cols: List[str], uwb_cols: List[str]) -> pd.DataFrame:
    rows = []
    uwb_m = anchor_xy_m(UWB_ANCHOR_TILES)
    for i in med_df.index:
        nx = med_df.loc[i, node_cols[0]]
        ny = med_df.loc[i, node_cols[1]]
        for uwb in uwb_cols:
            measured = float(med_df.loc[i, uwb])
            if not np.isfinite(measured):
                continue
            var = float(var_df.loc[i, uwb]) if uwb in var_df.columns else np.nan
            ax, ay = uwb_m[uwb]
            td = _true_dist_scalar(nx, ny, ax, ay)
            rows.append(
                {
                    "row_idx": i,
                    "Node_x": nx,
                    "Node_y": ny,
                    "anchor": uwb,
                    "true_dist": td,
                    "measured": measured,
                    "variance": var,
                    "error": measured - td,
                }
            )
    return pd.DataFrame(rows)


def _apply_stage3_mad(df_s2: pd.DataFrame) -> pd.Series:
    keep = pd.Series(False, index=df_s2.index)
    for _anchor, g in df_s2.groupby("anchor", sort=False):
        err = g["error"].values.astype(float)
        med_e, mad_e = _mad_threshold(err)
        if not np.isfinite(med_e):
            keep.loc[g.index] = False
            continue
        band = max(MAD_MULT * mad_e, MAD_MIN_BAND)
        m = np.abs(g["error"].values - med_e) <= band
        keep.loc[g.index] = m
    return keep


def _plot_uwb_filter_four_panels(
    long0: pd.DataFrame,
    m1: pd.Series,
    m2: pd.Series,
    m3: pd.Series,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    def decorate(ax: plt.Axes, title: str) -> None:
        ax.plot([0, 22], [0, 22], "r--", lw=1.2, label="y=x (ideal)")
        ax.set_xlim(0, 22)
        ax.set_ylim(0, 40)
        ax.set_xlabel("True distance (m)")
        ax.set_ylabel("Measured median (m)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=7.5)
        ax.grid(True, alpha=0.25)

    td, ms = long0["true_dist"].values, long0["measured"].values

    k1 = m1.values
    d1 = ~k1
    n_k1, n_d1 = int(k1.sum()), int(d1.sum())
    ax1.scatter(td[k1], ms[k1], s=10, c="tab:blue", alpha=0.45, label=f"Stage1 생존 (n={n_k1})")
    ax1.scatter(td[d1], ms[d1], s=28, c="tab:red", marker="x", linewidths=0.9, label=f"Hard limit 탈락 (n={n_d1})")
    decorate(ax1, "Stage 1: Hard limits (0.5 < meas < 22 m)")

    k2_full = m1 & m2
    d2_only = m1 & (~m2)
    n_k2 = int(k2_full.sum())
    n_d2 = int(d2_only.sum())
    ax2.scatter(td[k2_full], ms[k2_full], s=10, c="tab:blue", alpha=0.45, label=f"Stage2 생존 (n={n_k2})")
    ax2.scatter(td[d2_only], ms[d2_only], s=28, c="tab:orange", marker="x", linewidths=0.9, label=f"Variance 탈락 (n={n_d2})")
    decorate(ax2, f"Stage 2: Variance < {VAR_CAP_UWB_FUSION}")

    k3_full = m1 & m2 & m3
    d3_only = m1 & m2 & (~m3)
    n_k3 = int(k3_full.sum())
    n_d3 = int(d3_only.sum())
    ax3.scatter(td[k3_full], ms[k3_full], s=10, c="tab:blue", alpha=0.45, label=f"최종 통과 (n={n_k3})")
    ax3.scatter(td[d3_only], ms[d3_only], s=28, c="tab:purple", marker="x", linewidths=0.9, label=f"MAD 탈락 (n={n_d3})")
    decorate(ax3, "Stage 3: MAD per anchor")

    n_f = int(k3_full.sum())
    ax4.scatter(td[k3_full], ms[k3_full], s=12, c="tab:blue", alpha=0.55, label=f"Final clean (n={n_f})")
    decorate(ax4, "Final: Clean data only")

    fig.suptitle("UWB sequential filtering (True vs Measured)", fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"저장: {out_path}")


def run_spatial_profiler(median_path: Path, variance_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not median_path.is_file() or not variance_path.is_file():
        print("필수 파일이 없습니다:", file=sys.stderr)
        print(f"  median:  {median_path}", file=sys.stderr)
        print(f"  variance:{variance_path}", file=sys.stderr)
        return 1

    med_df, var_df = load_median_variance_sanitized(median_path, variance_path)
    node_cols, uwb_cols, wifi_cols = infer_sensor_columns(med_df)
    infer_sensor_columns(var_df)
    _nc2, uwb_v, wifi_v = infer_sensor_columns(var_df)
    if uwb_cols != uwb_v or wifi_cols != wifi_v:
        raise ValueError("median과 variance 파일의 센서 열 순서/이름이 일치하지 않습니다.")

    print("\n=== [spatial] Sensor Spatial Profiler ===")
    print(f"Median:  {median_path}")
    print(f"Variance:{variance_path}")
    print(f"Grid→m: ×{GRID_M}, 절사: 상·하위 각 {TRIM_PROP*100:.0f}%, |error|>{ERROR_ABS_CAP_M}m 제외\n")

    summary, raw_by_sensor = analysis_a_tables(med_df, var_df, node_cols, uwb_cols, wifi_cols)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("=== Analysis A: 기기별 강건 편향 및 분산 백분위 ===\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}" if np.isfinite(x) else "NaN"))

    bias_csv = out_dir / "sensor_profiler_robust_bias_table.csv"
    summary.to_csv(bias_csv, index=False, encoding="utf-8-sig")
    print(f"\n저장: {bias_csv}  (← fusion_realtime_sanitize PHASE1_* 와 동기화 참고용)")

    analysis_b_spatial(med_df, node_cols, uwb_cols, wifi_cols)

    p1 = out_dir / "sensor_profiler_eda_calibration_scatter.png"
    p2 = out_dir / "sensor_profiler_eda_error_boxplot.png"
    p3 = out_dir / "sensor_profiler_eda_spatial_error_heatmap.png"
    plot_scatter_calibration(med_df, node_cols, uwb_cols, wifi_cols, p1)
    plot_error_boxplot(raw_by_sensor, uwb_cols + wifi_cols, p2)
    plot_spatial_mae_heatmap(med_df, node_cols, uwb_cols, wifi_cols, p3)

    print_actionable(summary, uwb_cols, wifi_cols, var_df)
    return 0


def run_uwb_sequential_filter(median_path: Path, variance_path: Path, out_png: Path) -> int:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not median_path.is_file() or not variance_path.is_file():
        print("필수 파일 없음:", median_path, variance_path, file=sys.stderr)
        return 1

    med_df, var_df = load_median_variance_sanitized(median_path, variance_path)
    node_cols, uwb_cols, _wifi = infer_sensor_columns(med_df)
    _nc, uwb_v, _w = infer_sensor_columns(var_df)
    if uwb_cols != uwb_v:
        raise ValueError("median / variance 의 UWB 열 이름이 일치하지 않습니다.")

    for c in uwb_cols:
        if c not in UWB_ANCHOR_TILES:
            raise ValueError(f"알 수 없는 UWB 열: {c}")

    print("\n=== [uwb] UWB Sequential Filter ===")

    long0 = _build_long_uwb(med_df, var_df, node_cols, uwb_cols)
    long0 = long0.reset_index(drop=True)
    n0 = len(long0)

    m1 = (long0["measured"] > UWB_FILTER_HARD_LO) & (long0["measured"] < UWB_FILTER_HARD_HI)
    n1 = int(m1.sum())
    d1 = n0 - n1

    m2 = np.isfinite(long0["variance"]) & (long0["variance"] < VAR_CAP_UWB_FUSION)
    m2_eff = m1 & m2
    n2 = int(m2_eff.sum())
    d2 = n1 - n2

    df_s2 = long0.loc[m1 & m2].copy()
    mad_pass_s2 = _apply_stage3_mad(df_s2)
    m3 = pd.Series(False, index=long0.index)
    m3.loc[df_s2.index] = mad_pass_s2.to_numpy(dtype=bool)

    n3 = int((m1 & m2 & m3).sum())
    d3 = n2 - n3

    print(f"Stage 0 (Raw, UWB median finite): n = {n0}")
    print(f"Stage 1 (Hard limits): kept = {n1}, dropped = {d1}")
    print(f"Stage 2 (Variance < {VAR_CAP_UWB_FUSION}): kept = {n2}, dropped = {d2}")
    print(f"Stage 3 (MAD per anchor): kept = {n3}, dropped = {d3}")
    print()

    final_df = long0.loc[m1 & m2 & m3].copy()
    print("최종 Clean 데이터 기준 Mean Error (Measured - True), 앵커별 (m):")
    clean_anchor_rows: List[Dict[str, object]] = []
    for anchor in uwb_cols:
        sub = final_df.loc[final_df["anchor"] == anchor, "error"]
        if len(sub) == 0:
            print(f"  {anchor}: N/A (no rows)")
            clean_anchor_rows.append(
                {"anchor": anchor, "n_clean": 0, "mean_error_measured_minus_true_m": float("nan")}
            )
        else:
            em = float(sub.mean())
            print(f"  {anchor}: {em:+.6f}  (n={len(sub)})")
            clean_anchor_rows.append({"anchor": anchor, "n_clean": int(len(sub)), "mean_error_measured_minus_true_m": em})

    clean_csv = out_png.parent / "sensor_profiler_uwb_clean_anchor_mean_error.csv"
    pd.DataFrame(clean_anchor_rows).to_csv(clean_csv, index=False, encoding="utf-8-sig")
    print(f"\n저장 (참고·PHASE1 대체 아님): {clean_csv}")

    print()
    print("V9 하이브리드 모델의 gate_threshold 설정을 위한 참고 지표:")
    print("-" * 60)
    n_clean = len(final_df)
    if n_clean == 0:
        print("  최종 Clean UWB 레코드 없음 — 통계를 계산할 수 없습니다.")
    else:
        abs_err = np.abs(final_df["error"].astype(float).values)
        mae = float(np.nanmean(abs_err))
        med_ae = float(np.nanmedian(abs_err))
        p90 = float(np.nanpercentile(abs_err, 90))
        p95 = float(np.nanpercentile(abs_err, 95))
        p99 = float(np.nanpercentile(abs_err, 99))
        mx = float(np.nanmax(abs_err))
        print(f"  전체 Clean 데이터 개수:     {n_clean:,}")
        print(f"  절대 오차 평균 (MAE):       {mae:.6f} m")
        print(f"  절대 오차 중앙값 (Median): {med_ae:.6f} m")
        print(f"  절대 오차 P90:              {p90:.6f} m")
        print(f"  절대 오차 P95:              {p95:.6f} m")
        print(f"  절대 오차 P99:              {p99:.6f} m")
        print(f"  절대 오차 최댓값 (Max):     {mx:.6f} m")
        pool_csv = out_png.parent / "sensor_profiler_uwb_clean_abs_error_pool.csv"
        pd.DataFrame(
            [
                {
                    "n_clean_records": n_clean,
                    "abs_error_mae": mae,
                    "abs_error_median": med_ae,
                    "abs_error_p90": p90,
                    "abs_error_p95": p95,
                    "abs_error_p99": p99,
                    "abs_error_max": mx,
                    "note": "Ranging |measured-true| on sequential-filter Clean UWB; V9 gate is |geom-d_meas| not identical",
                }
            ]
        ).to_csv(pool_csv, index=False, encoding="utf-8-sig")
        print(f"저장: {pool_csv}")
    print("-" * 60)
    print()

    _plot_uwb_filter_four_panels(long0, m1, m2, m3, out_png)
    return 0


def _configure_stdio_utf8() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass


def main() -> int:
    _configure_stdio_utf8()

    ap = argparse.ArgumentParser(
        description="센서 공간 프로파일(spatial) + UWB 순차 필터(uwb). 기본은 둘 다 실행(all)."
    )
    ap.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "spatial", "uwb"],
        help="all: spatial 후 uwb | spatial: EDA만 | uwb: UWB 필터만",
    )
    args = ap.parse_args()

    median_path = Path(os.environ.get("MEDIAN_XLSX", str(MEDIAN_PATH)))
    variance_path = Path(os.environ.get("VARIANCE_XLSX", str(VARIANCE_PATH)))
    out_dir = Path(os.environ.get("PROFILER_OUT_DIR", str(OUT_DIR)))
    uwb_png = Path(os.environ.get("UWB_FILTER_PNG", str(UWB_FILTER_OUT_PNG)))

    if args.mode in ("all", "spatial"):
        rc = run_spatial_profiler(median_path, variance_path, out_dir)
        if rc != 0:
            return rc
    if args.mode in ("all", "uwb"):
        rc = run_uwb_sequential_filter(median_path, variance_path, uwb_png)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
