"""Per-sample oracle RMSE lower bound over OOF candidate predictors (no leakage)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.cv import clone_pipeline_cfg
from lib.io_mat import load_dataset
from lib.pipeline import VERSION_REGISTRY, localize_user
from lib.post_refine import fit_pos_affine
from lib.spatial import bounds_from_positions

_spec = importlib.util.spec_from_file_location(
    "_oof_best",
    ROOT / "scripts" / "exp_end2end_anchor_wls_oof_best.py",
)
_oof = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_oof)
train_one_fold = _oof.train_one_fold


def _rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e**2)))


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    cfg = VERSION_REGISTRY["v30"]
    X = d_hat.T.astype(np.float32)
    Y = p.T.astype(np.float32)
    BS = torch.from_numpy(bs.astype(np.float32))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    errs: dict[str, np.ndarray] = {}

    oof_v30 = np.zeros((2, n))
    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        calib = fit_calib(d_hat, p, bs, tr, cfg.calib, quantile_tau=cfg.quantile_tau, mlp_noise_std=cfg.mlp_noise_std)
        cfg0 = clone_pipeline_cfg(cfg, pos_refine_affine=False)
        xy_pre = bounds_from_positions(p[:, tr])
        p_tr = np.zeros((2, len(tr)))
        for j, u in enumerate(tr):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg0, xy_bounds=xy_pre)
            p_tr[:, j] = xy
        calib.pos_affine = fit_pos_affine(p_tr, p[:, tr])
        xy_b = bounds_from_positions(p[:, tr])
        for u in va:
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg, xy_bounds=xy_b)
            oof_v30[:, u] = xy
    e_v30 = np.hypot(p[0] - oof_v30[0], p[1] - oof_v30[1])
    errs["v30"] = e_v30
    print(f"v30_oof: rmse={_rmse(e_v30):.3f}")

    oof_anchor = np.zeros((n, 2), dtype=np.float32)
    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        pred, _ = train_one_fold(
            torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]),
            torch.from_numpy(X[va]), torch.from_numpy(Y[va]),
            BS, seed=100 + fold,
        )
        oof_anchor[va] = pred
    e_a = np.sqrt(np.sum((oof_anchor.T - p) ** 2, axis=0))
    errs["anchor"] = e_a
    print(f"anchor_oof: rmse={_rmse(e_a):.3f}")

    E = np.stack(list(errs.values()))
    e_oracle = np.min(E, axis=0)
    print(f"oracle_min_over_{len(errs)}_oof: rmse={_rmse(e_oracle):.3f} median={np.median(e_oracle):.3f} p90={np.percentile(e_oracle,90):.3f}")


if __name__ == "__main__":
    main()
