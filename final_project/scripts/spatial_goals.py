"""
Indoor positioning goals for ~120m x 60m UE span (InF_DH_FR1).

Writes outputs/spatial_goals.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import load_dataset, project_root
from lib.spatial import space_summary


def main() -> None:
    data = load_dataset()
    p, bs = data["p"], data["bs"]
    sp = space_summary(p, bs)
    diag = float(
        np.hypot(sp["total_span_m"]["x"], sp["total_span_m"]["y"])
    )

    goals = {
        "space": sp,
        "note": "Design: ±60m x ±30m (total 120x60m). Bounds: lib/spatial.py",
        "tiers": {
            "G0_baseline": {
                "rmse_m": 15.0,
                "description": "Huber raw 수준 이하",
            },
            "G1_course_pass": {
                "rmse_m": 10.0,
                "p90_m": 18.0,
                "description": "기말 합격선 — CV OOF",
            },
            "G2_good_indoor": {
                "rmse_m": 7.0,
                "median_m": 5.5,
                "description": "대각선 ~5% 오차 (합성 NLOS 데이터 현실적 상한)",
            },
            "G3_stretch": {
                "rmse_m": 5.0,
                "description": "UWB급 — 본 데이터(+16m bias)로는 매우 어려움",
            },
        },
        "current_best": {
            "version": "v16",
            "cv_oof_rmse_m": None,
        },
    }

    cat = project_root() / "outputs" / "version_catalog.json"
    if cat.exists():
        c = json.loads(cat.read_text(encoding="utf-8"))
        if "best" in c:
            goals["current_best"]["version"] = c["best"]["version"]
            goals["current_best"]["cv_oof_rmse_m"] = c["best"]["cv_rmse_m"]

    out = project_root() / "outputs" / "spatial_goals.json"
    out.write_text(json.dumps(goals, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(goals, indent=2, ensure_ascii=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
