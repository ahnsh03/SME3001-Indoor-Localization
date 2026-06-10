"""
v17 OOF error breakdown by zone (GT radius vs min-d_hat proxy).

Writes outputs/tail_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.cv import CV_SEED, N_SPLITS
from lib.io_mat import load_dataset, project_root
from lib.metrics import position_errors, summarize_errors
from lib.pipeline import VERSION_REGISTRY, localize_user
from lib.spatial import bounds_from_positions, zone_from_min_dist, zone_from_true_radius


def main() -> None:
    from dataclasses import replace

    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    cfg = replace(VERSION_REGISTRY["v17"], weight_gamma=1.0)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    oof_err = np.full(n, np.nan)
    zones_gt = np.full(n, -1, dtype=np.int64)
    zones_md = np.full(n, -1, dtype=np.int64)
    centroid = np.array([p[0].mean(), p[1].mean()], dtype=np.float64)

    for tr, va in kf.split(np.arange(n)):
        train_idx = np.asarray(tr, dtype=np.int64)
        calib = fit_calib(d_hat, p, bs, train_idx, cfg.calib)
        xy_bounds = bounds_from_positions(p[:, train_idx])
        for u in va:
            xy, _ = localize_user(
                d_hat[:, u], bs, calib, cfg, xy_bounds=xy_bounds
            )
            oof_err[u] = float(np.hypot(xy[0] - p[0, u], xy[1] - p[1, u]))
            r = float(np.hypot(p[0, u] - centroid[0], p[1, u] - centroid[1]))
            zones_gt[u] = zone_from_true_radius(r)
            zones_md[u] = zone_from_min_dist(d_hat[:, u])

    overall = summarize_errors(oof_err)
    by_gt = {}
    by_md = {}
    for z, name in enumerate(["near", "mid", "far"]):
        m = zones_gt == z
        if m.sum() > 0:
            by_gt[name] = {**summarize_errors(oof_err[m]), "n": int(m.sum())}
        m2 = zones_md == z
        if m2.sum() > 0:
            by_md[name] = {**summarize_errors(oof_err[m2]), "n": int(m2.sum())}

    payload = {
        "baseline": "v17",
        "overall": overall,
        "by_gt_radius_zone": by_gt,
        "by_min_dhat_zone": by_md,
        "interpretation": "far tail drives RMSE; min-dhat zone is inference-safe proxy",
    }
    out = project_root() / "outputs" / "tail_analysis.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
