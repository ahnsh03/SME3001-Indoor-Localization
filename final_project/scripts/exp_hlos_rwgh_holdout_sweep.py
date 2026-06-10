"""Quick holdout sweep for HLOS-Rwgh variants (to triage before OOF)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.hlos_rwgh import HlosConfig, build_bs_features, fit_los_classifier, localize_user_hlos, los_labels, ls_init_weighted
from lib.io_mat import load_dataset
from lib.pipeline import VERSION_REGISTRY


def _rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def run_case(name: str, d_hat: np.ndarray, p: np.ndarray, bs: np.ndarray, tr: np.ndarray, va: np.ndarray, cfg: HlosConfig, use_v30_calib: bool) -> float:
    if use_v30_calib:
        v30 = VERSION_REGISTRY["v30"]
        calib = fit_calib(d_hat, p, bs, tr, v30.calib, quantile_tau=v30.quantile_tau, mlp_noise_std=v30.mlp_noise_std)
        get_d = lambda u: calib.apply(d_hat[:, u])
    else:
        get_d = lambda u: d_hat[:, u]

    X_list, y_list = [], []
    for u in tr:
        du = get_d(u)
        p0 = ls_init_weighted(du, bs)
        X_list.append(build_bs_features(du, bs, p0))
        y_list.append(los_labels(du, p[:, u], bs, cfg.los_thresh_m, use_abs=cfg.los_use_abs))
    los_clf = fit_los_classifier(np.vstack(X_list), np.concatenate(y_list), cfg, seed=42)

    pred = np.zeros((2, len(va)))
    for j, u in enumerate(va):
        pred[:, j] = localize_user_hlos(get_d(u), bs, los_clf, cfg)
    rm = _rmse(p[:, va], pred)
    print(f"{name}: holdout_rmse={rm:.3f}")
    return rm


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    tr, va = idx[:560], idx[560:]

    cases: list[tuple[str, HlosConfig, bool]] = [
        ("raw_signed_t8k4", HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=False), False),
        ("raw_abs_t8k4", HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=True), False),
        ("raw_signed_t10k4", HlosConfig(top_k=10, subset_k=4, gamma_w=3.0, los_use_abs=False), False),
        ("calib_signed_t8k4", HlosConfig(top_k=8, subset_k=4, gamma_w=4.0, los_use_abs=False), True),
    ]
    best = ("", 1e9)
    for name, cfg, use_calib in cases:
        rm = run_case(name, d_hat, p, bs, tr, va, cfg, use_calib)
        if rm < best[1]:
            best = (name, rm)
    print(f"best={best[0]} rmse={best[1]:.3f}")


if __name__ == "__main__":
    main()
