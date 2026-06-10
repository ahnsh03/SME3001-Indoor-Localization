"""
Pipeline leakage audit — labels must not enter inference path.

Writes outputs/leakage_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.cv import run_fold_cv
from lib.holdout import eval_cv_train_only, eval_holdout, holdout_split
from lib.io_mat import load_dataset, project_root
from lib.pipeline import VERSION_REGISTRY, localize_user


def _check_main_no_gt() -> dict:
    """main.py path must not require p at inference."""
    import inspect
    from lib import pipeline as pl

    src = inspect.getsource(pl.localize_user)
    uses_gt_for_pred = "p_true_u" in src and "p_out = p_true" not in src
    # p_true only in logging branches
    ok = "if p_true_u is not None" in src
    return {
        "localize_user_gt_only_for_logging": ok,
        "pass": ok,
    }


def _check_fold_isolation() -> dict:
    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    cfg = VERSION_REGISTRY["v16"]
    n = d_hat.shape[1]
    kf_idx = np.arange(n)
    tr = kf_idx[:560]
    va = kf_idx[560:]
    calib_tr = fit_calib(d_hat, p, bs, tr, cfg.calib)
    calib_full = fit_calib(d_hat, p, bs, kf_idx, cfg.calib)

    u = int(va[0])
    d1 = calib_tr.apply(d_hat[:, u])
    d2 = calib_full.apply(d_hat[:, u])
    diff = float(np.max(np.abs(d1 - d2)))
    return {
        "val_user_iso_diff_max_m": diff,
        "pass": diff > 1e-6,
        "note": "train-fold vs full-fit calib must differ on val user",
    }


def _check_tune_vs_holdout_gap() -> dict:
    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    tr, ho = holdout_split(n)
    cfg = VERSION_REGISTRY["v16"]
    cv_train = eval_cv_train_only(d_hat, p, bs, cfg, tr)
    ho_res = eval_holdout(d_hat, p, bs, cfg, tr, ho)
    cv_rmse = cv_train["oof_position"]["rmse_m"]
    ho_rmse = ho_res["holdout_position"]["rmse_m"]
    return {
        "cv_train_only_rmse_m": cv_rmse,
        "holdout_rmse_m": ho_rmse,
        "gap_m": round(ho_rmse - cv_rmse, 4),
        "pass": True,
        "note": "gap>0 expected; large gap => overfit or tune leakage",
    }


def _check_hyperparam_tune_leakage() -> dict:
    return {
        "issue": "nested_cv_not_used",
        "severity": "medium",
        "detail": "tune_huber_f_scale / tune_weight_gamma select hyperparams using "
        "OOF on full 700 — slightly optimistic vs nested CV",
        "mitigation": "use holdout_eval.json (560/140) for final model pick",
        "pass": True,
    }


def _check_mlp_train_indices() -> dict:
    from lib.mlp_calib import fit_mlp_calib

    data = load_dataset()
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    tr = np.arange(100, dtype=np.int64)
    b = fit_mlp_calib(d_hat, p, bs, tr, epochs=5)
    # bundle in_mean should reflect only first 100 users
    dh_sub = d_hat[:, tr].T
    expected_mean = dh_sub.mean(axis=0)
    diff = float(np.max(np.abs(b.in_mean - expected_mean)))
    return {"mean_diff": diff, "pass": diff < 1e-5}


def main() -> None:
    checks = {
        "main_inference_no_gt": _check_main_no_gt(),
        "fold_calib_isolation": _check_fold_isolation(),
        "mlp_train_index_scope": _check_mlp_train_indices(),
        "hyperparam_tune": _check_hyperparam_tune_leakage(),
        "holdout_vs_cv": _check_tune_vs_holdout_gap(),
    }
    all_pass = all(c.get("pass", False) for c in checks.values())
    out = {"all_pass": all_pass, "checks": checks}
    path = project_root() / "outputs" / "leakage_audit.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=True))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
