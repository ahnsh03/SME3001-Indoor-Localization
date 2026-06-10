"""
Structural experiments v25-v27 vs v17 baseline.

v25: 2-pass position affine
v26: far-only 2nd isotonic
v27: quantile tau sweep + MLP
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.cv import run_fold_cv, save_version_outputs, tune_quantile_tau
from lib.io_mat import load_dataset, project_root
from lib.pipeline import PipelineConfig, VERSION_REGISTRY


def main() -> None:
    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    out_dir = project_root() / "outputs"

    baseline = replace(VERSION_REGISTRY["v17"], weight_gamma=1.0)
    rows = []

    print("=== v17 baseline ===")
    r17 = run_fold_cv(d_hat, p, bs, baseline)
    rows.append(
        {
            "version": "v17",
            "cv_rmse_m": r17["oof_position"]["rmse_m"],
            "cv_median_m": r17["oof_position"]["median_m"],
            "cv_p90_m": r17["oof_position"]["p90_m"],
            "extra": None,
        }
    )
    print(f"  RMSE={rows[-1]['cv_rmse_m']:.3f} m")

    print("\n=== v25: 2-pass position affine ===")
    cfg25 = VERSION_REGISTRY["v25"]
    r25 = run_fold_cv(d_hat, p, bs, cfg25)
    save_version_outputs(out_dir, "v25", r25)
    rows.append(
        {
            "version": "v25",
            "cv_rmse_m": r25["oof_position"]["rmse_m"],
            "cv_median_m": r25["oof_position"]["median_m"],
            "cv_p90_m": r25["oof_position"]["p90_m"],
            "extra": r25.get("oof_stage_position", {}).get("after_pos_affine"),
        }
    )
    print(f"  RMSE={rows[-1]['cv_rmse_m']:.3f} m")

    print("\n=== v26: far 2nd isotonic ===")
    cfg26 = VERSION_REGISTRY["v26"]
    r26 = run_fold_cv(d_hat, p, bs, cfg26)
    save_version_outputs(out_dir, "v26", r26)
    rows.append(
        {
            "version": "v26",
            "cv_rmse_m": r26["oof_position"]["rmse_m"],
            "cv_median_m": r26["oof_position"]["median_m"],
            "cv_p90_m": r26["oof_position"]["p90_m"],
            "extra": None,
        }
    )
    print(f"  RMSE={rows[-1]['cv_rmse_m']:.3f} m")

    print("\n=== v27: quantile tau CV ===")
    tune = tune_quantile_tau(d_hat, p, bs, VERSION_REGISTRY["v27"])
    cfg27 = PipelineConfig(
        version="v27",
        loss="huber",
        calib="quantile_mlp",
        weight_gamma=1.0,
        quantile_tau=tune["best_quantile_tau"],
        description=f"quantile_mlp tau={tune['best_quantile_tau']}",
    )
    r27 = run_fold_cv(d_hat, p, bs, cfg27)
    r27["tau_tuning"] = tune
    save_version_outputs(out_dir, "v27", r27)
    rows.append(
        {
            "version": "v27",
            "cv_rmse_m": r27["oof_position"]["rmse_m"],
            "cv_median_m": r27["oof_position"]["median_m"],
            "cv_p90_m": r27["oof_position"]["p90_m"],
            "extra": {"best_tau": tune["best_quantile_tau"], "grid": tune["grid"]},
        }
    )
    print(f"  best tau={tune['best_quantile_tau']} RMSE={rows[-1]['cv_rmse_m']:.3f} m")

    b0 = rows[0]["cv_rmse_m"]
    for row in rows:
        row["delta_vs_v17_m"] = round(row["cv_rmse_m"] - b0, 4)

    payload = {"baseline": "v17", "results": rows, "best": min(rows, key=lambda r: r["cv_rmse_m"])}
    path = out_dir / "structural_v25_v27.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {path}")
    for row in rows:
        print(
            f"  {row['version']}: RMSE={row['cv_rmse_m']:.3f} "
            f"(delta {row['delta_vs_v17_m']:+.3f})"
        )


if __name__ == "__main__":
    main()
