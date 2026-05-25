"""Midterm train data: Wi-Fi vs UWB distance error vs truth (quick audit)."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
train_med = ROOT / "data" / "train" / "median_test_kgh_corrected.csv"
df = pd.read_csv(train_med)
# columns: Node_x, Node_y, 6 UWB, 6 WiFi (per README)
cols = df.columns.tolist()
uwb_cols = [c for c in cols if c not in ("Node_x", "Node_y") and not str(c).startswith("SW")]
wifi_cols = [c for c in cols if str(c).startswith("SW") or "SW" in str(c)]
if len(wifi_cols) < 6:
    # fallback: last 6 cols wifi
    wifi_cols = cols[-6:]
    uwb_cols = cols[2:-6]

tx = df["Node_x"].to_numpy(float) * 0.6
ty = df["Node_y"].to_numpy(float) * 0.6

# anchor positions from v12 script constants (tile * 0.6) — approximate via unique node grid
# For error vs truth we only need measured - true distance per anchor at each row.
# True distance requires anchor coords — load from fusion sanitize or hardcode from README

ANCHOR_XY = {
    "110394ab": (0.0, 0.0),
    "e63ce2f": (12.0, 0.0),
    "8e610981": (12.0, 18.0),
    "d10485af": (0.0, 18.0),
    "d1044709": (6.0, 9.0),
    "4e610206": (6.0, 0.0),
}
SW_MAP = {
    "SW_11": (0.0, 0.0),
    "SW_first_team": (12.0, 0.0),
    "볼링공": (12.0, 18.0),
    "SW_4": (0.0, 18.0),
    "SW_6": (6.0, 9.0),
    "SW_5": (6.0, 0.0),
}


def pool_errors(cols_map):
    errs = []
    for name, (ax, ay) in cols_map.items():
        if name not in df.columns:
            continue
        d_meas = df[name].to_numpy(float)
        d_true = np.hypot(tx - ax, ty - ay)
        e = d_meas - d_true
        errs.append(e[np.isfinite(e)])
    return np.concatenate(errs) if errs else np.array([])


uwb_e = pool_errors(ANCHOR_XY)
wifi_e = pool_errors(SW_MAP)
print("Train rows:", len(df))
print("UWB cols used:", [c for c in ANCHOR_XY if c in df.columns])
print("WiFi cols used:", [c for c in SW_MAP if c in df.columns])
for label, arr in [("UWB", uwb_e), ("WiFi", wifi_e)]:
    ae = np.abs(arr)
    print(f"\n[{label}] n={len(arr)}")
    print(f"  bias(mean err)={arr.mean():.3f} m, MAE={ae.mean():.3f}, MedAE={np.median(ae):.3f}, P90={np.percentile(ae,90):.3f}")
    print(f"  frac |err|>3m: {(ae>3).mean()*100:.1f}%")
    print(f"  frac |err|>5m: {(ae>5).mean()*100:.1f}%")
