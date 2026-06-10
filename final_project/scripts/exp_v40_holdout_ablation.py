"""Holdout ablation for v40 components (fast triage)."""
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


def run_case(name: str, tr, va, d_hat, p, bs, cfg, **opts) -> None:
    t0 = time.perf_counter()
    m = fit_hlos_fold_model(d_hat, p, bs, tr, hlos_cfg=cfg, seed=42, **opts)
    pred = np.zeros((2, len(va)))
    for j, u in enumerate(va):
        pred[:, j] = m.predict_hlos(d_hat[:, u], bs)
    e = np.hypot(p[0, va] - pred[0], p[1, va] - pred[1])
    print(
        f"{name}: rmse={float(np.sqrt(np.mean(e**2))):.3f} "
        f"median={float(np.median(e)):.3f} ({time.perf_counter()-t0:.0f}s)"
    )


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    tr, va = idx[:560], idx[560:]
    cfg = HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=True)

    run_case("hlos_only", tr, va, d_hat, p, bs, cfg, use_per_bs_affine=False, use_residual=False, use_pos_affine=False)
    run_case("hlos+affine_bs", tr, va, d_hat, p, bs, cfg, use_per_bs_affine=True, use_residual=False, use_pos_affine=False)
    run_case("hlos+residual", tr, va, d_hat, p, bs, cfg, use_per_bs_affine=False, use_residual=True, use_pos_affine=False)
    run_case("hlos+pos_affine", tr, va, d_hat, p, bs, cfg, use_per_bs_affine=False, use_residual=False, use_pos_affine=True)
    run_case("hlos+res+pos", tr, va, d_hat, p, bs, cfg, use_per_bs_affine=False, use_residual=True, use_pos_affine=True)


if __name__ == "__main__":
    main()
