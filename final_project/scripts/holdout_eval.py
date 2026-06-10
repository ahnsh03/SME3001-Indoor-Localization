"""Hold-out 560/140 evaluation (no hyperparameter peeking on holdout)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.cv import tune_pos_blend_weight, tune_weight_gamma
from lib.holdout import eval_holdout, holdout_split, save_holdout_report
from lib.io_mat import load_dataset, project_root
from lib.pipeline import VERSION_REGISTRY, PipelineConfig


def _cfg_for_version(ver: str, d_hat, p, bs) -> PipelineConfig:
    cfg = VERSION_REGISTRY[ver]
    if ver == "v11":
        t = tune_weight_gamma(d_hat, p, bs, cfg)
        return PipelineConfig(
            version=ver,
            loss=cfg.loss,
            calib=cfg.calib,
            weight_gamma=t["best_weight_gamma"],
        )
    if ver == "v17":
        t = tune_weight_gamma(d_hat[:, :], p, bs, cfg)
        return PipelineConfig(
            version=ver,
            loss=cfg.loss,
            calib=cfg.calib,
            weight_gamma=t["best_weight_gamma"],
        )
    if ver == "v20":
        t = tune_pos_blend_weight(d_hat, p, bs, cfg)
        return PipelineConfig(
            version=ver,
            loss=cfg.loss,
            calib=cfg.calib,
            pos_blend_weight=t["best_pos_blend_weight"],
        )
    return cfg


def main() -> None:
    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    tr, ho = holdout_split(n)

    versions = ["v16", "v17", "v18", "v19", "v20"]
    rows = []
    for ver in versions:
        if ver not in VERSION_REGISTRY:
            continue
        cfg = _cfg_for_version(ver, d_hat, p, bs)
        # tune only on train subset
        if ver in ("v17", "v20", "v11"):
            sub = tr
            if ver == "v17":
                t = tune_weight_gamma(d_hat[:, sub], p[:, sub], bs, VERSION_REGISTRY["v17"])
                cfg = PipelineConfig(
                    version=ver,
                    calib="isotonic_mlp",
                    weight_gamma=t["best_weight_gamma"],
                )
            elif ver == "v20":
                t = tune_pos_blend_weight(d_hat[:, sub], p[:, sub], bs, VERSION_REGISTRY["v20"])
                cfg = PipelineConfig(
                    version=ver,
                    calib="isotonic_mlp",
                    pos_blend_weight=t["best_pos_blend_weight"],
                )
        res = eval_holdout(d_hat, p, bs, cfg, tr, ho)
        rmse = res["holdout_position"]["rmse_m"]
        rows.append({"version": ver, "holdout_rmse_m": rmse})
        print(f"{ver} holdout RMSE = {rmse:.3f} m")

    payload = {"n_train": int(tr.size), "n_holdout": int(ho.size), "versions": rows}
    path = project_root() / "outputs" / "holdout_eval.json"
    save_holdout_report(path, payload)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
