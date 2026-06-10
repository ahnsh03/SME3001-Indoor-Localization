"""5-fold OOF: v30 calib + learned WLS + 2-pass position affine (v30-style)."""
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
from lib.io_mat import load_dataset
from lib.pipeline import VERSION_REGISTRY
from lib.post_refine import apply_pos_affine, fit_pos_affine
from lib.spatial import bounds_from_positions

_spec = importlib.util.spec_from_file_location(
    "_anchor_v30",
    ROOT / "scripts" / "exp_anchor_on_v30calib_oof.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
AnchorRelWLSOnly = _mod.AnchorRelWLSOnly
train_fold = _mod.train_fold


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    cfg = VERSION_REGISTRY["v30"]
    n = d_hat.shape[1]
    BS = torch.from_numpy(bs.astype(np.float32))
    Y = p.T.astype(np.float32)
    oof = np.zeros((2, n), dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        calib = fit_calib(
            d_hat, p, bs, tr, cfg.calib,
            quantile_tau=cfg.quantile_tau,
            mlp_noise_std=cfg.mlp_noise_std,
        )
        Xtr = torch.from_numpy(np.stack([calib.apply(d_hat[:, u]) for u in tr]).astype(np.float32))
        Xva = torch.from_numpy(np.stack([calib.apply(d_hat[:, u]) for u in va]).astype(np.float32))
        Ytr = torch.from_numpy(Y[tr])
        Yva = torch.from_numpy(Y[va])
        pred_va, _, state = train_fold(Xtr, Ytr, Xva, Yva, BS, seed=400 + fold)

        model = AnchorRelWLSOnly()
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            _, p_tr_t = model(Xtr, BS)
        p_tr = p_tr_t.cpu().numpy().T  # (2, n_tr)
        M = fit_pos_affine(p_tr, p[:, tr])

        for j, u in enumerate(va):
            oof[:, u] = apply_pos_affine(pred_va[j], M)

        e = np.hypot(oof[0, va] - p[0, va], oof[1, va] - p[1, va])
        print(f"fold={fold} rmse={float(np.sqrt(np.mean(e**2))):.3f}")

    e = np.hypot(oof[0] - p[0], oof[1] - p[1])
    print(
        f"physics_stack_oof rmse={float(np.sqrt(np.mean(e**2))):.3f} "
        f"median={float(np.median(e)):.3f} p90={float(np.percentile(e, 90)):.3f}"
    )


if __name__ == "__main__":
    main()
