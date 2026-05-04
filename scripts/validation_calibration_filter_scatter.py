"""
검증(validation) median/variance에 대해 True 거리 vs 측정 거리 산점도를
필터 단계별로 비교 저장한다.

1) 원본 median
2) 센서별 Robust bias 차감 (융합 파이프라인 동일 오프셋)
3) + 분산 Var ≤ 센서별 fusion cap(UWB/Wi‑Fi)
4) + 원본 median 물리 게이트 (0.5 < median < 22 m)

출력: outputs/validation_calibration_filter_scatter.png
실행: py -3 scripts/validation_calibration_filter_scatter.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fusion_realtime_sanitize import (
    HARDWARE_CALIB_BIAS_M,
    VAR_CAP_UWB_FUSION,
    VAR_CAP_WIFI_FUSION,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_PATH = PROJECT_ROOT / "outputs" / "validation_calibration_filter_scatter.png"

if sys.platform == "win32":
    try:
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

GRID_M = 0.6
HARD_LO = 0.5
HARD_HI = 22.0

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
        raise ValueError("최소 14열이 필요합니다.")
    return cols[:2], cols[2:8], cols[8:14]


def anchor_xy_m(tiles: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    return {k: (v[0] * GRID_M, v[1] * GRID_M) for k, v in tiles.items()}


def sensor_canon_key(col: str, idx: int, is_uwb: bool) -> str:
    if is_uwb:
        if col in UWB_ANCHOR_TILES:
            return col
        return UWB_COL_CANON[idx]
    if col in WIFI_AP_TILES:
        return col
    return WIFI_COL_CANON[idx]


def anchor_xy_for_sensor(
    col: str, is_uwb: bool, idx: int, uwb_cols: List[str], wifi_cols: List[str]
) -> Tuple[float, float]:
    uwb_m = anchor_xy_m(UWB_ANCHOR_TILES)
    wifi_m = anchor_xy_m(WIFI_AP_TILES)
    if is_uwb:
        if col in uwb_m:
            return uwb_m[col]
        j = uwb_cols.index(col) if col in uwb_cols else idx
        return uwb_m[UWB_COL_CANON[j]]
    if col in wifi_m:
        return wifi_m[col]
    j = wifi_cols.index(col) if col in wifi_cols else idx
    return wifi_m[WIFI_COL_CANON[j]]


def true_distance_m(nx: np.ndarray, ny: np.ndarray, ax: float, ay: float) -> np.ndarray:
    px, py = nx.astype(float) * GRID_M, ny.astype(float) * GRID_M
    return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)


def collect_with_bias(
    med_df: pd.DataFrame,
    var_df: pd.DataFrame,
    node_cols: List[str],
    uwb_cols: List[str],
    wifi_cols: List[str],
    uwb_only: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx = med_df[node_cols[0]].values
    ny = med_df[node_cols[1]].values
    td_a, mr_a, vr_a, bias_a = [], [], [], []
    for i, col in enumerate(uwb_cols):
        if not uwb_only:
            continue
        ax, ay = anchor_xy_for_sensor(col, True, i, uwb_cols, wifi_cols)
        td_a.append(true_distance_m(nx, ny, ax, ay))
        mr_a.append(med_df[col].values.astype(float))
        vr_a.append(var_df[col].values.astype(float))
        bias_a.append(np.full(len(med_df), ROBUST_BIAS_M[sensor_canon_key(col, i, True)]))
    for i, col in enumerate(wifi_cols):
        if uwb_only:
            continue
        ax, ay = anchor_xy_for_sensor(col, False, i, uwb_cols, wifi_cols)
        td_a.append(true_distance_m(nx, ny, ax, ay))
        mr_a.append(med_df[col].values.astype(float))
        vr_a.append(var_df[col].values.astype(float))
        bias_a.append(np.full(len(med_df), ROBUST_BIAS_M[sensor_canon_key(col, i, False)]))
    if not td_a:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    return (
        np.concatenate(td_a),
        np.concatenate(mr_a),
        np.concatenate(vr_a),
        np.concatenate(bias_a),
    )


def axis_lim(td: np.ndarray, yplots: List[np.ndarray], fin: np.ndarray) -> float:
    parts = []
    for y in yplots:
        parts.append(td[fin])
        parts.append(y[fin])
    stack = np.concatenate([p[np.isfinite(p)] for p in parts if p.size])
    if stack.size == 0:
        return 22.0
    hi = float(np.percentile(stack, 99.5))
    return max(hi, 5.0)


def decorate_ax(ax: plt.Axes, title: str, xmax: float) -> None:
    ax.plot([0, xmax], [0, xmax], "r--", lw=1.0, alpha=0.9, label="y=x")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax * 1.06)
    ax.set_xlabel("True distance (m)")
    ax.set_ylabel("Median / calibrated (m)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def run_plot() -> None:
    val_dir = PROJECT_ROOT / "data" / "validation"
    med_df = coerce_numeric(load_sensor_excel(val_dir / "median_validation.xlsx"))
    var_df = coerce_numeric(load_sensor_excel(val_dir / "variance_validation.xlsx"))
    node_cols, uwb_cols, wifi_cols = infer_sensor_columns(med_df)
    _nc, uwb_v, wifi_v = infer_sensor_columns(var_df)
    if uwb_cols != uwb_v or wifi_cols != wifi_v:
        raise ValueError("median / variance 열 불일치")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    for row_idx, (uwb_only, row_label) in enumerate([(True, "UWB"), (False, "Wi-Fi")]):
        var_cap_row = VAR_CAP_UWB_FUSION if uwb_only else VAR_CAP_WIFI_FUSION
        titles = [
            "1) 원본 median",
            "2) Bias 차감",
            f"3) + Var ≤ {var_cap_row}",
            f"4) + {HARD_LO} < 원본 < {HARD_HI} m",
        ]
        td, mr, vr, bias = collect_with_bias(
            med_df, var_df, node_cols, uwb_cols, wifi_cols, uwb_only
        )
        if td.size == 0:
            continue
        y_cal = mr - bias
        s0 = np.isfinite(td) & np.isfinite(mr)
        s1 = s0 & np.isfinite(vr) & (vr <= var_cap_row)
        s2 = s1 & (mr > HARD_LO) & (mr < HARD_HI)

        xmax = axis_lim(td, [mr, y_cal], s0)

        ax0 = axes[row_idx, 0]
        if np.any(s0):
            ax0.scatter(td[s0], mr[s0], s=9, alpha=0.42, c="steelblue" if uwb_only else "darkgreen", rasterized=True)
        decorate_ax(ax0, f"{row_label}\n{titles[0]}", xmax)

        ax1 = axes[row_idx, 1]
        if np.any(s0):
            ax1.scatter(td[s0], y_cal[s0], s=9, alpha=0.42, c="steelblue" if uwb_only else "darkgreen", rasterized=True)
        decorate_ax(ax1, f"{row_label}\n{titles[1]}", xmax)

        ax2 = axes[row_idx, 2]
        dropped_var = s0 & (~s1)
        if np.any(dropped_var):
            ax2.scatter(
                td[dropped_var],
                y_cal[dropped_var],
                s=8,
                c="lightgray",
                alpha=0.35,
                label="분산 cap 초과 또는 결측",
                rasterized=True,
            )
        if np.any(s1):
            ax2.scatter(
                td[s1],
                y_cal[s1],
                s=11,
                alpha=0.5,
                c="steelblue" if uwb_only else "darkgreen",
                label="통과",
                rasterized=True,
            )
        decorate_ax(ax2, f"{row_label}\n{titles[2]}", xmax)
        if row_idx == 0 and uwb_only:
            ax2.legend(loc="upper left", fontsize=7)

        ax3 = axes[row_idx, 3]
        dropped_gate = s1 & (~s2)
        if np.any(dropped_var):
            ax3.scatter(
                td[dropped_var],
                y_cal[dropped_var],
                s=6,
                c="lightgray",
                alpha=0.25,
                rasterized=True,
            )
        if np.any(dropped_gate):
            ax3.scatter(
                td[dropped_gate],
                y_cal[dropped_gate],
                s=18,
                c="tab:orange",
                alpha=0.55,
                marker="x",
                linewidths=0.6,
                label="거리 게이트 탈락",
                rasterized=True,
            )
        if np.any(s2):
            ax3.scatter(
                td[s2],
                y_cal[s2],
                s=11,
                alpha=0.55,
                c="steelblue" if uwb_only else "darkgreen",
                label="최종 통과",
                rasterized=True,
            )
        decorate_ax(ax3, f"{row_label}\n{titles[3]}", xmax)
        if row_idx == 1 and not uwb_only:
            ax3.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "Validation: 필터 단계별 True vs 측정 거리 (train과 동일 오프셋·Var cap·거리 게이트)",
        fontsize=12,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)
    print(f"저장: {OUT_PATH}")


def main() -> int:
    run_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
