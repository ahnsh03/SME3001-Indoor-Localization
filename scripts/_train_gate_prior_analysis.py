"""One-off: train-only OOF distribution for UWB geometric gate residuals (no validation)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fusion_realtime_sanitize import VAR_CAP_UWB_FUSION, resolve_train_kgh_corrected_paths
from indoor_fusion_pipeline_v12 import wifi_median_impute_fusion
from indoor_fusion_pipeline_v8 import Config, FusionLocalizerV8, ROBUST_BIAS_M, sensor_canon_key


def main() -> None:
    train_dir = ROOT / "data" / "train"
    tm, tv = resolve_train_kgh_corrected_paths(train_dir)
    cfg = Config(grid_size_m=0.6, huber_f_scale=1.0)
    loc = FusionLocalizerV8(cfg)
    train_df, _ = loc.load_datasets(tm, tv, tm, tv)

    n_splits = min(3, max(2, len(train_df) // 35))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    wifi_err: list[float] = []
    true_err: list[float] = []
    row_med_wifi: list[float] = []

    for tr_idx, te_idx in kf.split(np.arange(len(train_df))):
        tr_raw = train_df.iloc[tr_idx].reset_index(drop=True)
        te_raw = train_df.iloc[te_idx].reset_index(drop=True)
        _, te_imp = wifi_median_impute_fusion(tr_raw, te_raw, loc.wifi_cols)
        xa, ya = loc.predict_step(te_imp, "A")
        for k, (_, row) in enumerate(te_imp.iterrows()):
            wx, wy = float(xa[k]), float(ya[k])
            tx = float(row["True_X"])
            ty = float(row["True_Y"])
            row_w: list[float] = []
            for i, c in enumerate(loc.uwb_cols):
                d_raw = row.get(f"med_{c}", np.nan)
                v = row.get(f"var_{c}", np.nan)
                was_nan = row.get(f"orig_nan_{c}", 1) == 1
                if was_nan or not np.isfinite(d_raw) or not np.isfinite(v):
                    continue
                if float(v) > VAR_CAP_UWB_FUSION:
                    continue
                key_u = sensor_canon_key(c, i, True)
                ax, ay = loc._resolve_sensor_position_m(c, i, True)
                d_meas = float(d_raw) - ROBUST_BIAS_M[key_u]
                if np.isfinite(wx) and np.isfinite(wy):
                    d_geom_w = float(np.hypot(wx - ax, wy - ay))
                    e = abs(d_geom_w - d_meas)
                    wifi_err.append(e)
                    row_w.append(e)
                if np.isfinite(tx) and np.isfinite(ty):
                    d_geom_t = float(np.hypot(tx - ax, ty - ay))
                    true_err.append(abs(d_geom_t - d_meas))
            if row_w:
                row_med_wifi.append(float(np.median(row_w)))

    g = np.asarray(wifi_err, dtype=float)
    t = np.asarray(true_err, dtype=float)
    rw = np.asarray(row_med_wifi, dtype=float)

    def q(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p))

    print(f"[train rows={len(train_df)} folds={n_splits}]")
    print("WiFi-only StepA anchor |d_geom - d_meas| (m), all (row,anchor) pairs")
    print(f"  n={g.size} p50={q(g,50):.3f} p90={q(g,90):.3f} p95={q(g,95):.3f} p99={q(g,99):.3f}")
    print("Per-row median of the above (one scalar per OOF row)")
    print(f"  n={rw.size} p50={q(rw,50):.3f} p90={q(rw,90):.3f} p95={q(rw,95):.3f} p99={q(rw,99):.3f}")
    print("True position same metric (oracle lower bound on gate residual)")
    print(f"  n={t.size} p50={q(t,50):.3f} p90={q(t,90):.3f} p95={q(t,95):.3f} p99={q(t,99):.3f}")
    lo = max(0.5, round(q(rw, 25) * 0.9, 2))
    hi = min(8.5, round(q(rw, 97.5) * 1.1, 2))
    print(f"Heuristic Optuna gate range from row-median train OOF: [{lo}, {hi}] m")


if __name__ == "__main__":
    main()
