"""v40: leakage-safe HLOS pipeline ablations + optional anchor blend (no v30)."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.hlos_pipeline import fit_hlos_fold_model, run_hlos_oof_cv
from lib.hlos_rwgh import HlosConfig
from lib.io_mat import load_dataset

_spec = importlib.util.spec_from_file_location("_anchor", ROOT / "scripts" / "exp_end2end_anchor_wls_oof_best.py")
_anchor = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_anchor)
train_one_fold = _anchor.train_one_fold


def _blend_oof(
    d_hat: np.ndarray,
    p: np.ndarray,
    bs: np.ndarray,
    hlos_cfg: HlosConfig,
    seed: int = 42,
) -> dict:
    n = d_hat.shape[1]
    X = d_hat.T.astype(np.float32)
    Y = p.T.astype(np.float32)
    BS = torch.from_numpy(bs.astype(np.float32))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_blend = np.zeros((2, n))
    alphas = []

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        m = fit_hlos_fold_model(
            d_hat, p, bs, tr,
            hlos_cfg=hlos_cfg,
            use_per_bs_affine=True,
            use_residual=True,
            use_pos_affine=True,
            seed=seed + fold,
        )
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
        p_h_tr = np.zeros((2, len(tr)))
        p_h_va = np.zeros((2, len(va)))
        for j, u in enumerate(tr):
            p_h_tr[:, j] = m.predict_hlos(d_hat[:, u], bs)
        for j, u in enumerate(va):
            p_h_va[:, j] = m.predict_hlos(d_hat[:, u], bs)

        best_a, best_rm = 0.0, 1e9
        for alpha in np.linspace(0.0, 1.0, 21):
            b0 = alpha * p_h_tr[0] + (1.0 - alpha) * pred_a_tr[:, 0]
            b1 = alpha * p_h_tr[1] + (1.0 - alpha) * pred_a_tr[:, 1]
            e = np.hypot(p[0, tr] - b0, p[1, tr] - b1)
            rm = float(np.sqrt(np.mean(e**2)))
            if rm < best_rm:
                best_rm, best_a = rm, float(alpha)
        alphas.append(best_a)
        oof_blend[0, va] = best_a * p_h_va[0] + (1.0 - best_a) * pred_a_va[:, 0]
        oof_blend[1, va] = best_a * p_h_va[1] + (1.0 - best_a) * pred_a_va[:, 1]

    e = np.hypot(p[0] - oof_blend[0], p[1] - oof_blend[1])
    return {
        "name": "v40_full+anchor_blend",
        "rmse_m": float(np.sqrt(np.mean(e**2))),
        "median_m": float(np.median(e)),
        "p90_m": float(np.percentile(e, 90)),
        "blend_alpha_hlos_mean": float(np.mean(alphas)),
    }


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    cfg = HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=True)

    results = []
    t0 = time.perf_counter()

    cases = [
        ("v40_hlos_raw", dict(use_per_bs_affine=False, use_residual=False, use_pos_affine=False)),
        ("v40_hlos+per_bs", dict(use_per_bs_affine=True, use_residual=False, use_pos_affine=False)),
        ("v40_hlos+residual", dict(use_per_bs_affine=True, use_residual=True, use_pos_affine=False)),
        ("v40_full", dict(use_per_bs_affine=True, use_residual=True, use_pos_affine=True)),
    ]
    for name, opts in cases:
        r = run_hlos_oof_cv(d_hat, p, bs, hlos_cfg=cfg, seed=42, **opts)
        row = {"name": name, **{k: r[k] for k in ("rmse_m", "median_m", "p90_m", "fold_rmse")}}
        results.append(row)
        print(f"{name}: rmse={row['rmse_m']:.3f} median={row['median_m']:.3f} folds={row['fold_rmse']}")

    print("anchor blend (slow)...")
    results.append(_blend_oof(d_hat, p, bs, cfg))
    print(f"{results[-1]['name']}: rmse={results[-1]['rmse_m']:.3f} alpha={results[-1].get('blend_alpha_hlos')}")

    out = ROOT / "outputs" / "v40_hlos_ablation.json"
    out.write_text(json.dumps({"elapsed_s": time.perf_counter() - t0, "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    best = min(results, key=lambda x: x["rmse_m"])
    print(f"BEST: {best['name']} rmse={best['rmse_m']:.3f}")


if __name__ == "__main__":
    main()
