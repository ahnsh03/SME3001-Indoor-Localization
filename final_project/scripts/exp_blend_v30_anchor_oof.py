"""OOF blend: v30 pipeline + best anchor-WLS (grid alpha + optional HGB stack)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.cv import clone_pipeline_cfg
from lib.io_mat import load_dataset
from lib.pipeline import VERSION_REGISTRY, localize_user
from lib.post_refine import fit_pos_affine
from lib.spatial import bounds_from_positions

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_oof_best",
    ROOT / "scripts" / "exp_end2end_anchor_wls_oof_best.py",
)
_oof = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_oof)
train_one_fold = _oof.train_one_fold


def _rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    base_cfg = VERSION_REGISTRY["v30"]
    X = d_hat.T.astype(np.float32)
    Y = p.T.astype(np.float32)
    BS = torch.from_numpy(bs.astype(np.float32))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_v30 = np.zeros((2, n), dtype=np.float64)
    oof_anchor = np.zeros((n, 2), dtype=np.float32)

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)

        calib = fit_calib(
            d_hat, p, bs, tr, base_cfg.calib,
            quantile_tau=base_cfg.quantile_tau,
            mlp_noise_std=base_cfg.mlp_noise_std,
        )
        cfg_no_aff = clone_pipeline_cfg(base_cfg, pos_refine_affine=False)
        xy_bounds_pre = bounds_from_positions(p[:, tr])
        p_tr_raw = np.zeros((2, len(tr)), dtype=np.float64)
        for j, u in enumerate(tr):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg_no_aff, xy_bounds=xy_bounds_pre)
            p_tr_raw[:, j] = xy
        calib.pos_affine = fit_pos_affine(p_tr_raw, p[:, tr])
        xy_bounds = bounds_from_positions(p[:, tr])
        for u in va:
            xy, _ = localize_user(d_hat[:, u], bs, calib, base_cfg, xy_bounds=xy_bounds)
            oof_v30[:, u] = xy

        Xtr = torch.from_numpy(X[tr])
        Ytr = torch.from_numpy(Y[tr])
        Xva = torch.from_numpy(X[va])
        Yva = torch.from_numpy(Y[va])
        pred, _ = train_one_fold(Xtr, Ytr, Xva, Yva, BS, seed=100 + fold)
        oof_anchor[va] = pred
        print(f"fold={fold} done")

    print(f"v30_oof rmse={_rmse(p, oof_v30):.3f}")
    anchor_t = oof_anchor.T
    print(f"anchor_oof rmse={_rmse(p, anchor_t):.3f}")

    best_alpha, best_rmse = 0.5, 1e9
    for alpha in np.linspace(0.0, 1.0, 21):
        blend = alpha * oof_v30 + (1.0 - alpha) * anchor_t
        rm = _rmse(p, blend)
        if rm < best_rmse:
            best_rmse, best_alpha = rm, float(alpha)
    print(f"grid_blend best_alpha={best_alpha:.2f} rmse={best_rmse:.3f}")

    # meta stacker on OOF preds + raw distances
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    meta_x = np.hstack([oof_v30.T, oof_anchor, mins, maxs, means, stds])
    meta_oof = np.zeros((n, 2), dtype=np.float64)
    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        m = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=6, random_state=42 + fold)
        )
        m.fit(meta_x[tr], Y[tr])
        meta_oof[va] = m.predict(meta_x[va])
    print(f"meta_stack rmse={_rmse(p, meta_oof.T):.3f}")


if __name__ == "__main__":
    main()
