"""
Single-factor ablations vs v17 (each tests one hypothesis).

Writes outputs/ablation_results.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.cv import run_fold_cv, save_version_outputs
from lib.io_mat import load_dataset, project_root
from lib.pipeline import VERSION_REGISTRY
from lib.spatial import space_summary


def main() -> None:
    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    out_dir = project_root() / "outputs"

    baseline = replace(VERSION_REGISTRY["v17"], weight_gamma=1.0)
    versions = [
        ("v17", baseline, "baseline"),
        ("v21", VERSION_REGISTRY["v21"], "far: min-dhat zone Isotonic"),
        ("v22", VERSION_REGISTRY["v22"], "quantile tau=0.35 + MLP"),
        ("v23", VERSION_REGISTRY["v23"], "near/mid trilat weight boost"),
        ("v24", VERSION_REGISTRY["v24"], "MLP noise aug std=0.03"),
    ]

    rows = []
    for ver, cfg, note in versions:
        print(f"\n=== {ver}: {note} ===")
        res = run_fold_cv(d_hat, p, bs, cfg)
        rmse = res["oof_position"]["rmse_m"]
        med = res["oof_position"]["median_m"]
        p90 = res["oof_position"]["p90_m"]
        delta = rmse - rows[0]["cv_rmse_m"] if rows else 0.0
        row = {
            "version": ver,
            "hypothesis": note,
            "cv_rmse_m": rmse,
            "cv_median_m": med,
            "cv_p90_m": p90,
            "delta_vs_v17_m": round(delta, 4) if rows else 0.0,
        }
        rows.append(row)
        print(f"  RMSE={rmse:.3f} median={med:.3f} p90={p90:.3f} delta={row['delta_vs_v17_m']:+.3f}")
        save_version_outputs(out_dir, ver, res)

    best = min(rows, key=lambda r: r["cv_rmse_m"])
    payload = {
        "space": space_summary(p, bs),
        "baseline_version": "v17",
        "results": rows,
        "best": best,
    }
    path = out_dir / "ablation_results.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBest: {best['version']} RMSE={best['cv_rmse_m']:.3f} m")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
