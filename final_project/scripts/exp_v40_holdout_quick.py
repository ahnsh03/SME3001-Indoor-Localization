"""Quick holdout 560/140 for v40 full pipeline."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hlos_pipeline import fit_hlos_fold_model
from lib.hlos_rwgh import HlosConfig
from lib.io_mat import load_dataset


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    tr, va = idx[:560], idx[560:]
    cfg = HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=True)

    t0 = time.perf_counter()
    m = fit_hlos_fold_model(
        d_hat, p, bs, tr,
        hlos_cfg=cfg,
        use_per_bs_affine=True,
        use_residual=True,
        use_pos_affine=True,
        seed=42,
    )
    pred = np.zeros((2, len(va)))
    for j, u in enumerate(va):
        pred[:, j] = m.predict_hlos(d_hat[:, u], bs)
    e = np.hypot(p[0, va] - pred[0], p[1, va] - pred[1])
    print(
        f"v40_full holdout rmse={float(np.sqrt(np.mean(e**2))):.3f} "
        f"median={float(np.median(e)):.3f} p90={float(np.percentile(e,90)):.3f} "
        f"({time.perf_counter()-t0:.1f}s)"
    )


if __name__ == "__main__":
    main()
