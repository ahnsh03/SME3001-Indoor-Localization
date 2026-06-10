"""
Retrospective review + combination experiments (v25 base).

Fixes under-tested singles by pairing with full v25 stack and proper CV tuning.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.cv import (
    clone_pipeline_cfg,
    run_fold_cv,
    save_version_outputs,
    tune_asym_pos_weight,
    tune_huber_f_scale,
    tune_top_k,
    tune_weight_gamma,
)
from lib.io_mat import load_dataset, project_root
from lib.pipeline import VERSION_REGISTRY, PipelineConfig

V25_BASE = PipelineConfig(
    version="v25",
    calib="isotonic_mlp",
    weight_gamma=1.0,
    pos_refine_affine=True,
)


def _row(ver: str, note: str, res: dict, extra=None) -> dict:
    o = res["oof_position"]
    return {
        "version": ver,
        "note": note,
        "cv_rmse_m": o["rmse_m"],
        "cv_median_m": o["median_m"],
        "cv_p90_m": o["p90_m"],
        "extra": extra,
    }


def main() -> None:
    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    out = project_root() / "outputs"

    retrospective = {
        "under_tested": [
            {
                "item": "gamma=1.5 (v11)",
                "issue": "v11는 Isotonic만; v17/25는 gamma CV→1.0. MLP+아핀과 미조합",
            },
            {
                "item": "f_scale=0.8 (v10)",
                "issue": "v06에만 적용; isotonic_mlp+v25 미적용",
            },
            {
                "item": "asym Huber (v14)",
                "issue": "Isotonic 단독 10.3m; v17/25와 미조합",
            },
            {
                "item": "top_k (v15)",
                "issue": "Isotonic 단독; 보정 후 k는 v25와 미조합",
            },
            {
                "item": "zone weight (v23)",
                "issue": "UE단위 가중은 삼변 해 불변; 채널별은 악화",
            },
            {
                "item": "mindist Iso (v21)",
                "issue": "v17 대비 +0.01만 확인; +아핀 미조합",
            },
        ],
        "tune_bug_fixed": "clone_pipeline_cfg로 튜닝 시 pos_refine_affine 유지",
    }

    rows = []
    print("=== v25 baseline ===")
    r25 = run_fold_cv(d_hat, p, bs, V25_BASE)
    rows.append(_row("v25", "baseline", r25))
    print(f"  RMSE={rows[-1]['cv_rmse_m']:.3f}")

    experiments = [
        ("v28", "gamma CV on v25", lambda: _exp_gamma(d_hat, p, bs)),
        ("v29", "f_scale CV on v25", lambda: _exp_fscale(d_hat, p, bs)),
        ("v30", "asym CV on v25", lambda: _exp_asym(d_hat, p, bs)),
        ("v31", "top_k CV on v25", lambda: _exp_topk(d_hat, p, bs)),
        ("v32", "gamma=1.5 fixed", lambda: _exp_fixed(d_hat, p, bs, "v32")),
        ("v33", "gamma1.5+f0.8", lambda: _exp_fixed(d_hat, p, bs, "v33")),
        ("v34", "far 2nd affine", lambda: _exp_fixed(d_hat, p, bs, "v34")),
        ("v35", "mindist+affine", lambda: _exp_fixed(d_hat, p, bs, "v35")),
    ]

    for ver, note, fn in experiments:
        print(f"\n=== {ver}: {note} ===")
        res, extra = fn()
        save_version_outputs(out, ver, res)
        rows.append(_row(ver, note, res, extra))
        d = rows[-1]["cv_rmse_m"] - rows[0]["cv_rmse_m"]
        print(f"  RMSE={rows[-1]['cv_rmse_m']:.3f} delta={d:+.3f}")

    # Best combo: v25 + best gamma from v28 + f0.8
    g_best = rows[[r["version"] for r in rows].index("v28")].get("extra", {})
    if g_best and "best_weight_gamma" in g_best:
        wg = g_best["best_weight_gamma"]
        cfg36 = clone_pipeline_cfg(
            V25_BASE, weight_gamma=wg, huber_f_scale=0.8, version="v36"
        )
        print("\n=== v36: v25 + best_gamma + f0.8 ===")
        r36 = run_fold_cv(d_hat, p, bs, cfg36)
        save_version_outputs(out, "v36", r36)
        rows.append(
            _row(
                "v36",
                f"combo gamma={wg} f=0.8",
                r36,
                {"weight_gamma": wg, "huber_f_scale": 0.8},
            )
        )
        print(f"  RMSE={rows[-1]['cv_rmse_m']:.3f}")

    b0 = rows[0]["cv_rmse_m"]
    for r in rows:
        r["delta_vs_v25_m"] = round(r["cv_rmse_m"] - b0, 4)

    payload = {
        "retrospective": retrospective,
        "baseline": "v25",
        "results": rows,
        "best": min(rows, key=lambda x: x["cv_rmse_m"]),
    }
    path = out / "combination_review.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBest: {payload['best']['version']} RMSE={payload['best']['cv_rmse_m']:.3f}")
    print(f"Wrote {path}")


def _exp_gamma(d_hat, p, bs):
    tune = tune_weight_gamma(d_hat, p, bs, V25_BASE)
    cfg = clone_pipeline_cfg(
        V25_BASE,
        version="v28",
        weight_gamma=tune["best_weight_gamma"],
    )
    res = run_fold_cv(d_hat, p, bs, cfg)
    res["gamma_tuning"] = tune
    return res, tune


def _exp_fscale(d_hat, p, bs):
    tune = tune_huber_f_scale(d_hat, p, bs, V25_BASE)
    cfg = clone_pipeline_cfg(
        V25_BASE, version="v29", huber_f_scale=tune["best_huber_f_scale"]
    )
    res = run_fold_cv(d_hat, p, bs, cfg)
    res["f_scale_tuning"] = tune
    return res, tune


def _exp_asym(d_hat, p, bs):
    tune = tune_asym_pos_weight(d_hat, p, bs, V25_BASE)
    cfg = clone_pipeline_cfg(
        V25_BASE, version="v30", asym_pos_weight=tune["best_asym_pos_weight"]
    )
    res = run_fold_cv(d_hat, p, bs, cfg)
    res["asym_tuning"] = tune
    return res, tune


def _exp_topk(d_hat, p, bs):
    tune = tune_top_k(d_hat, p, bs, V25_BASE)
    cfg = clone_pipeline_cfg(V25_BASE, version="v31", top_k=tune["best_top_k"])
    res = run_fold_cv(d_hat, p, bs, cfg)
    res["top_k_tuning"] = tune
    return res, tune


def _exp_fixed(d_hat, p, bs, ver: str):
    cfg = VERSION_REGISTRY[ver]
    res = run_fold_cv(d_hat, p, bs, cfg)
    return res, None


if __name__ == "__main__":
    main()
