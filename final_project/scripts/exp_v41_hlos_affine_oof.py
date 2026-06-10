"""v41 OOF: HLOS-Rwgh (raw+abs) + 2-pass position affine only — no v30, no per-BS affine."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hlos_pipeline import run_hlos_oof_cv
from lib.hlos_rwgh import HlosConfig
from lib.io_mat import load_dataset


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    cfg = HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=True)
    t0 = time.perf_counter()
    r = run_hlos_oof_cv(
        d_hat, p, bs,
        hlos_cfg=cfg,
        use_per_bs_affine=False,
        use_residual=False,
        use_pos_affine=True,
        seed=42,
    )
    print(f"v41_hlos_affine_oof rmse={r['rmse_m']:.3f} median={r['median_m']:.3f} p90={r['p90_m']:.3f}")
    print(f"fold_rmse={r['fold_rmse']}")
    out = ROOT / "outputs" / "v41_hlos_affine_summary.json"
    out.write_text(
        json.dumps({"elapsed_s": time.perf_counter() - t0, **r, "config": "hlos_abs+pos_affine"}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
