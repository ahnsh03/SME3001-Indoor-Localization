"""Train v41 on full 700 users and save model_v41.npz (for dev; not submit)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hlos_pipeline import fit_hlos_fold_model
from lib.hlos_rwgh import HlosConfig
from lib.io_mat import load_dataset


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    cfg = HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=True)
    train_idx = np.arange(n, dtype=np.int64)
    model = fit_hlos_fold_model(
        d_hat, p, bs, train_idx,
        hlos_cfg=cfg,
        use_per_bs_affine=False,
        use_residual=False,
        use_pos_affine=True,
        seed=42,
    )
    out = ROOT / "model_v41.npz"
    np.savez_compressed(
        out,
        hlos_cfg=np.array([cfg.top_k, cfg.subset_k, cfg.gamma_w, cfg.gamma_r, float(cfg.los_use_abs)]),
        per_bs_scale=model.per_bs.scale if model.per_bs else np.ones(18),
        per_bs_bias=model.per_bs.bias if model.per_bs else np.zeros(18),
        pos_affine=model.pos_affine if model.pos_affine is not None else np.eye(2, 3),
    )
    # sklearn models need pickle — save separately
    import pickle

    with open(ROOT / "model_v41_los.pkl", "wb") as f:
        pickle.dump(model.los_clf, f)
    print(f"saved {out} and model_v41_los.pkl (retrain los_clf for inference)")


if __name__ == "__main__":
    main()
