"""
Run candidate versions until spatial goal tier met (CV OOF RMSE).

Usage:
  py -3 scripts/run_experiment_loop.py
  py -3 scripts/run_experiment_loop.py --goal G2_good_indoor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.cv import (
    run_fold_cv,
    save_version_outputs,
    tune_pos_blend_weight,
    tune_weight_gamma,
)
from lib.io_mat import load_dataset, project_root
from lib.pipeline import VERSION_REGISTRY, PipelineConfig


def _goal_rmse(goal_key: str) -> float:
    path = project_root() / "outputs" / "spatial_goals.json"
    if path.exists():
        g = json.loads(path.read_text(encoding="utf-8"))
        return float(g["tiers"][goal_key]["rmse_m"])
    defaults = {"G1_course_pass": 10.0, "G2_good_indoor": 7.0, "G3_stretch": 5.0}
    return defaults.get(goal_key, 10.0)


def _resolve_cfg(ver: str, d_hat, p, bs) -> tuple[PipelineConfig, dict]:
    cfg = VERSION_REGISTRY[ver]
    extra = {}
    if ver == "v17":
        tune = tune_weight_gamma(d_hat, p, bs, cfg)
        cfg = PipelineConfig(
            version=ver,
            calib=cfg.calib,
            weight_gamma=tune["best_weight_gamma"],
            description=cfg.description,
        )
        extra["weight_tuning"] = tune
    elif ver == "v20":
        tune = tune_pos_blend_weight(d_hat, p, bs, cfg)
        cfg = PipelineConfig(
            version=ver,
            calib=cfg.calib,
            pos_blend_weight=tune["best_pos_blend_weight"],
            description=cfg.description,
        )
        extra["blend_tuning"] = tune
    return cfg, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="G2_good_indoor")
    ap.add_argument(
        "--versions",
        nargs="*",
        default=["v16", "v17", "v18", "v19", "v20"],
    )
    args = ap.parse_args()

    target = _goal_rmse(args.goal)
    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    out_dir = project_root() / "outputs"
    catalog = []

    print(f"Goal {args.goal}: CV OOF RMSE < {target:.2f} m\n")

    for ver in args.versions:
        if ver not in VERSION_REGISTRY:
            continue
        cfg, extra = _resolve_cfg(ver, d_hat, p, bs)
        print(f"=== {ver} ===")
        res = run_fold_cv(d_hat, p, bs, cfg)
        if extra:
            res.update(extra)
        rmse = res["oof_position"]["rmse_m"]
        met = rmse < target
        print(f"OOF RMSE = {rmse:.3f} m  {'OK' if met else 'not yet'}")
        save_version_outputs(out_dir, ver, res)
        catalog.append({"version": ver, "cv_rmse_m": rmse, "goal_met": met})

    best = min(catalog, key=lambda r: r["cv_rmse_m"])
    payload = {
        "goal": args.goal,
        "target_rmse_m": target,
        "best": best,
        "versions": catalog,
    }
    (out_dir / "experiment_loop.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nBest: {best['version']} {best['cv_rmse_m']:.3f} m")
    if best["goal_met"]:
        print(f"Goal {args.goal} achieved.")
    else:
        print(f"Goal {args.goal} not reached yet (need < {target:.2f} m).")


if __name__ == "__main__":
    main()
