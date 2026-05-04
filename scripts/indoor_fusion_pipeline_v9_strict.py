"""
V9 Strict — V9와 동일한 Step A(순수 Wi‑Fi) + Step B(UWB 기하 게이트 융합) 수식.

`gate_threshold` 선택만 **Train K-Fold CV 평균 RMSE(Step B)**로 수행하고,
검증 세트의 `True_X`/`True_Y`는 **최종 단일 Forward 후 평가·저장**에만 사용한다.

원본 `indoor_fusion_pipeline_v9.py`는 검증 RMSE로 게이트를 고르므로 선택 편향이 있다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from fusion_realtime_sanitize import resolve_train_kgh_corrected_paths, resolve_validation_paths
from indoor_fusion_pipeline_v8 import (
    Config,
    ROBUST_BIAS_M,
    WIFI_AP_TILES,
    sensor_canon_key,
    tile_to_m,
)
from indoor_fusion_pipeline_v9 import GATE_THRESH_GRID, FusionLocalizerV9

ART_PREFIX = "v9_strict"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except (AttributeError, OSError):
        pass
plt.rcParams["axes.unicode_minus"] = False


def resolve_kfold_n_splits(n_samples: int) -> int:
    if n_samples < 2:
        raise ValueError("Train 샘플이 2개 미만이면 K-Fold를 할 수 없습니다.")
    if n_samples < 6:
        return 2
    return int(min(5, max(3, n_samples // 35)))


MAX_KFOLD_SPLITS = 3


def cv_mean_rmse_step_b_gate(
    train_df: pd.DataFrame,
    loc: FusionLocalizerV9,
    gate_t: float,
    n_splits: int,
) -> float:
    """각 폴드 홀드아웃 train에서 Step A → Step B 후 RMSE; 폴드 평균."""
    n_splits = min(n_splits, len(train_df))
    if n_splits < 2:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses: List[float] = []
    for _, te_idx in kf.split(np.arange(len(train_df))):
        te_df = train_df.iloc[te_idx].reset_index(drop=True)
        xa, ya = loc.predict_step(te_df, "A")
        xb, yb = loc.predict_step_b_series(te_df, float(gate_t), xa, ya)
        tx = te_df["True_X"].values.astype(float)
        ty = te_df["True_Y"].values.astype(float)
        err = loc.point_errors(tx, ty, xb, yb)
        r, _ = loc.rmse_mae(err)
        rmses.append(float(r))
    return float(np.mean(rmses))


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
    train_df, val_df = loc.load_datasets(
        train_median_path=train_med,
        train_variance_path=train_var,
        val_median_path=val_med,
        val_variance_path=val_var,
    )
    n_splits = min(MAX_KFOLD_SPLITS, resolve_kfold_n_splits(len(train_df)))
    print(f"\n[V9 Strict] 학습 {len(train_df)}행 | 검증 {len(val_df)}행 | Train K-Fold={n_splits}", flush=True)

    # --- gate: Train CV만 (검증 정답 미사용) --------------------------------
    print("[V9 Strict] Train CV로 gate_threshold 탐색 중…", flush=True)
    grid_records: List[Dict[str, Any]] = []
    best_t = float(GATE_THRESH_GRID[0])
    best_cv = float("inf")
    for gt in GATE_THRESH_GRID:
        cv_r = cv_mean_rmse_step_b_gate(train_df, loc, float(gt), n_splits)
        grid_records.append(dict(gate_threshold_m=float(gt), train_CV_mean_RMSE_step_B_m=float(cv_r), kfold_splits=n_splits))
        if cv_r < best_cv - 1e-14:
            best_cv = float(cv_r)
            best_t = float(gt)

    # --- 검증: 단일 Forward (선택된 gate만) ---------------------------------
    tx = val_df["True_X"].values.astype(float)
    ty = val_df["True_Y"].values.astype(float)
    xa, ya = loc.predict_step(val_df, "A")
    err_a = loc.point_errors(tx, ty, xa, ya)
    rmse_a, mae_a = loc.rmse_mae(err_a)

    xf, yf = loc.predict_step_b_series(val_df, best_t, xa, ya)
    err_b = loc.point_errors(tx, ty, xf, yf)
    rmse_b, mae_b = loc.rmse_mae(err_b)

    g = cfg.grid_size_m
    qc_b: List[float] = []
    for i, (_, row) in enumerate(val_df.iterrows()):
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

    kill_df, _, _ = loc.accumulate_kill_log(val_df, xa, ya, best_t)

    pred = val_df[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xa
    pred["StepA_Y"] = ya
    pred["Final_X"] = xf
    pred["Final_Y"] = yf
    pred["Error_StepA_m"] = err_a
    pred["Error_Final_m"] = err_b
    pred["Quality_RMSE_m"] = np.asarray(qc_b, dtype=float)
    pred.to_csv(out_dir / f"{ART_PREFIX}_predictions.csv", index=False, encoding="utf-8-sig")
    kill_df.to_csv(out_dir / f"{ART_PREFIX}_uwb_kill_log.csv", index=False, encoding="utf-8-sig")

    summ = {
        "pipeline": "V9 Strict",
        "integrity": (
            "gate_threshold selected by mean K-fold train CV RMSE (Step B only). "
            "Validation labels used once for final metrics and exports."
        ),
        "train_kfold_splits": int(n_splits),
        "selected_gate_threshold_m": float(best_t),
        "train_cv_at_selection": {"mean_CV_RMSE_step_B_m": float(best_cv)},
        "validation_single_forward_metrics": {
            "step_A_RMSE_m": float(rmse_a),
            "step_A_MAE_m": float(mae_a),
            "step_B_RMSE_m": float(rmse_b),
            "step_B_MAE_m": float(mae_b),
        },
        "gate_grid_size": len(GATE_THRESH_GRID),
    }
    (out_dir / f"{ART_PREFIX}_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(grid_records).to_csv(out_dir / f"{ART_PREFIX}_gate_grid_trainCV.csv", index=False, encoding="utf-8-sig")

    ea = err_a[np.isfinite(err_a)]
    eb = err_b[np.isfinite(err_b)]
    plt.figure(figsize=(8.5, 5))
    for e, lab, sty in [(ea, "Step A: Pure Wi-Fi", "tab:blue"), (eb, f"Step B: V9 Strict (gate={best_t:.3f} m)", "tab:red")]:
        if e.size == 0:
            continue
        s = np.sort(e)
        cdf = np.arange(1, len(s) + 1) / len(s)
        plt.plot(s, cdf, lw=2, label=lab)
    plt.xlabel("Position error (m)")
    plt.ylabel("Cumulative probability")
    plt.title("V9 Strict: Step A vs Step B error CDF (validation, single pass)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_step_cdf.png", dpi=200)
    plt.close()

    print("\n" + "=" * 72)
    print("V9 Strict — gate: Train K-Fold CV | 검증: 단일 Forward")
    print("=" * 72)
    print(f"  [Train CV] Step B 평균 RMSE (게이트 선택 기준) = {best_cv:.4f} m @ gate={best_t:.4f} m")
    print(f"  [Validation] Step A  RMSE={rmse_a:.4f} m  MAE={mae_a:.4f} m")
    print(f"  [Validation] Step B  RMSE={rmse_b:.4f} m  MAE={mae_b:.4f} m")
    print("=" * 72)
    print(f"\n저장: {out_dir / f'{ART_PREFIX}_summary.json'}")
    print(f"저장: {out_dir / f'{ART_PREFIX}_predictions.csv'}")
    print(f"저장: {out_dir / f'{ART_PREFIX}_uwb_kill_log.csv'}")
    print(f"저장: {out_dir / f'{ART_PREFIX}_gate_grid_trainCV.csv'}")
    print(f"저장: {out_dir / f'{ART_PREFIX}_step_cdf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
