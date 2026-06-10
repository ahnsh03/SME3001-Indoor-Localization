"""
Run incremental pipeline versions with 5-Fold CV and stage logging.

Default: v00-v03, v05, v06, v09 (HardGate v04 excluded).

Usage (from final_project/):
  py -3 scripts/run_all_versions.py
  py -3 scripts/run_all_versions.py --versions v05 v06
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
    tune_asym_pos_weight,
    tune_huber_f_scale,
    tune_pos_blend_weight,
    tune_top_k,
    tune_weight_gamma,
)
from lib.io_mat import load_dataset, project_root
from lib.pipeline import ACTIVE_VERSIONS, VERSION_REGISTRY, PipelineConfig
from lib.run_io import RunSession


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--versions",
        nargs="*",
        default=ACTIVE_VERSIONS,
        help=f"default: {ACTIVE_VERSIONS}",
    )
    ap.add_argument("--mat", type=str, default=None)
    ap.add_argument("--include-deprecated", action="store_true", help="run v04 HardGate")
    args = ap.parse_args()

    versions = list(args.versions)
    if args.include_deprecated and "v04" not in versions:
        versions.append("v04")

    out_dir = project_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunSession("run_all_versions"):
        data = load_dataset(args.mat)
        d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
        catalog = []

        for ver in versions:
            if ver not in VERSION_REGISTRY:
                print(f"skip unknown version {ver}")
                continue
            cfg = VERSION_REGISTRY[ver]
            print(f"\n=== {ver}: {cfg.description} ===")

            extra = {}
            if ver == "v09":
                tune = tune_huber_f_scale(d_hat, p, bs, cfg)
                cfg = PipelineConfig(
                    version="v09",
                    loss=cfg.loss,
                    huber_f_scale=tune["best_huber_f_scale"],
                    calib=cfg.calib,
                    gate=False,
                    description=cfg.description
                    + f" (f_scale={tune['best_huber_f_scale']:.2f})",
                )
                extra["f_scale_tuning"] = tune
            elif ver == "v10":
                tune = tune_huber_f_scale(d_hat, p, bs, cfg)
                cfg = PipelineConfig(
                    version="v10",
                    loss=cfg.loss,
                    huber_f_scale=tune["best_huber_f_scale"],
                    calib=cfg.calib,
                    gate=False,
                    description=cfg.description
                    + f" (f_scale={tune['best_huber_f_scale']:.2f})",
                )
                extra["f_scale_tuning"] = tune
            elif ver == "v11":
                tune = tune_weight_gamma(d_hat, p, bs, cfg)
                cfg = PipelineConfig(
                    version="v11",
                    loss=cfg.loss,
                    huber_f_scale=cfg.huber_f_scale,
                    calib=cfg.calib,
                    gate=False,
                    weight_gamma=tune["best_weight_gamma"],
                    description=cfg.description
                    + f" (gamma={tune['best_weight_gamma']})",
                )
                extra["weight_tuning"] = tune
            elif ver == "v14":
                tune = tune_asym_pos_weight(d_hat, p, bs, cfg)
                cfg = PipelineConfig(
                    version="v14",
                    loss=cfg.loss,
                    calib=cfg.calib,
                    asym_pos_weight=tune["best_asym_pos_weight"],
                    description=cfg.description
                    + f" (asym_w={tune['best_asym_pos_weight']})",
                )
                extra["asym_tuning"] = tune
            elif ver == "v15":
                tune = tune_top_k(d_hat, p, bs, cfg)
                cfg = PipelineConfig(
                    version="v15",
                    loss=cfg.loss,
                    calib=cfg.calib,
                    top_k=tune["best_top_k"],
                    description=cfg.description
                    + f" (top_k={tune['best_top_k'] or 18})",
                )
                extra["top_k_tuning"] = tune
            elif ver == "v17":
                tune = tune_weight_gamma(d_hat, p, bs, cfg)
                cfg = PipelineConfig(
                    version="v17",
                    loss=cfg.loss,
                    calib=cfg.calib,
                    weight_gamma=tune["best_weight_gamma"],
                    description=cfg.description
                    + f" (gamma={tune['best_weight_gamma']})",
                )
                extra["weight_tuning"] = tune
            elif ver == "v20":
                tune = tune_pos_blend_weight(d_hat, p, bs, cfg)
                cfg = PipelineConfig(
                    version="v20",
                    loss=cfg.loss,
                    calib=cfg.calib,
                    pos_blend_weight=tune["best_pos_blend_weight"],
                    description=cfg.description
                    + f" (blend_w={tune['best_pos_blend_weight']})",
                )
                extra["blend_tuning"] = tune

            cv_result = run_fold_cv(d_hat, p, bs, cfg)
            if extra:
                cv_result.update(extra)

            rmse = cv_result["oof_position"]["rmse_m"]
            print(f"OOF RMSE = {rmse:.3f} m")
            if cv_result.get("oof_stage_position"):
                for stage, st in cv_result["oof_stage_position"].items():
                    print(f"  stage {stage}: RMSE {st['rmse_m']:.3f} m")

            save_version_outputs(out_dir, ver, cv_result)
            catalog.append(
                {
                    "version": ver,
                    "cv_rmse_m": rmse,
                    "calib": cfg.calib,
                    "huber_f_scale": cfg.huber_f_scale,
                    "weight_gamma": cfg.weight_gamma,
                    "gate": cfg.gate,
                }
            )

        # delta vs previous in run order
        prev_rmse = None
        for row in catalog:
            if prev_rmse is not None:
                row["delta_vs_prev_m"] = round(row["cv_rmse_m"] - prev_rmse, 4)
            prev_rmse = row["cv_rmse_m"]

        best = min(catalog, key=lambda r: r["cv_rmse_m"])
        cat_path = out_dir / "version_catalog.json"
        payload = {"versions": catalog, "best": best}
        cat_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nBest: {best['version']} RMSE={best['cv_rmse_m']:.3f} m")
        print(f"Wrote {cat_path}")


if __name__ == "__main__":
    main()
