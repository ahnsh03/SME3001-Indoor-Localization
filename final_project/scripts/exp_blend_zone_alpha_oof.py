"""Per-fold zone-wise blend of v30 + anchor OOF (alpha tuned on train only)."""
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

_spec = importlib.util.spec_from_file_location("_oof", ROOT / "scripts" / "exp_end2end_anchor_wls_oof_best.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
train_one_fold = _mod.train_one_fold


def zone_id_xy(x: float, y: float, x_mid: float, y_mid: float) -> int:
    z = 0
    if x >= x_mid:
        z += 1
    if y >= y_mid:
        z += 2
    return z


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    cfg = VERSION_REGISTRY["v30"]
    X = d_hat.T.astype(np.float32)
    Y = p.T.astype(np.float32)
    BS = torch.from_numpy(bs.astype(np.float32))
    x_mid, y_mid = float(np.median(p[0])), float(np.median(p[1]))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_blend = np.zeros((2, n))

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)

        calib = fit_calib(d_hat, p, bs, tr, cfg.calib, quantile_tau=cfg.quantile_tau, mlp_noise_std=cfg.mlp_noise_std)
        cfg0 = clone_pipeline_cfg(cfg, pos_refine_affine=False)
        xy_pre = bounds_from_positions(p[:, tr])
        p_tr_raw = np.zeros((2, len(tr)))
        pred_v30_tr = np.zeros((2, len(tr)))
        pred_v30_va = np.zeros((2, len(va)))
        for j, u in enumerate(tr):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg0, xy_bounds=xy_pre)
            p_tr_raw[:, j] = xy
        calib.pos_affine = fit_pos_affine(p_tr_raw, p[:, tr])
        xy_b = bounds_from_positions(p[:, tr])
        for j, u in enumerate(tr):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg, xy_bounds=xy_b)
            pred_v30_tr[:, j] = xy
        for j, u in enumerate(va):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg, xy_bounds=xy_b)
            pred_v30_va[:, j] = xy

        pred_a_tr, _ = train_one_fold(
            torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]),
            torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]),
            BS, seed=100 + fold,
        )
        pred_a_va, _ = train_one_fold(
            torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]),
            torch.from_numpy(X[va]), torch.from_numpy(Y[va]),
            BS, seed=100 + fold,
        )

        alpha_z = {}
        for z in range(4):
            tr_z = [j for j, u in enumerate(tr) if zone_id_xy(p[0, u], p[1, u], x_mid, y_mid) == z]
            if len(tr_z) < 15:
                alpha_z[z] = 0.2
                continue
            best_rmse, best_a = 1e9, 0.2
            for alpha in np.linspace(0.0, 1.0, 21):
                b0 = alpha * pred_v30_tr[0, tr_z] + (1 - alpha) * pred_a_tr[tr_z, 0]
                b1 = alpha * pred_v30_tr[1, tr_z] + (1 - alpha) * pred_a_tr[tr_z, 1]
                e = np.hypot(p[0, tr[tr_z]] - b0, p[1, tr[tr_z]] - b1)
                rm = float(np.sqrt(np.mean(e**2)))
                if rm < best_rmse:
                    best_rmse, best_a = rm, float(alpha)
            alpha_z[z] = best_a

        for j, u in enumerate(va):
            z = zone_id_xy(p[0, u], p[1, u], x_mid, y_mid)
            a = alpha_z.get(z, 0.2)
            oof_blend[0, u] = a * pred_v30_va[0, j] + (1 - a) * pred_a_va[j, 0]
            oof_blend[1, u] = a * pred_v30_va[1, j] + (1 - a) * pred_a_va[j, 1]

        e_va = np.hypot(p[0, va] - oof_blend[0, va], p[1, va] - oof_blend[1, va])
        print(f"fold={fold} rmse={float(np.sqrt(np.mean(e_va**2))):.3f} alphas={alpha_z}")

    e = np.hypot(p[0] - oof_blend[0], p[1] - oof_blend[1])
    print(
        f"zone_blend_oof rmse={float(np.sqrt(np.mean(e**2))):.3f} "
        f"median={float(np.median(e)):.3f} p90={float(np.percentile(e, 90)):.3f}"
    )


if __name__ == "__main__":
    main()
