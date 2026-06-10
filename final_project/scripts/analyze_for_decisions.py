"""
Regenerate observation metrics for DEV_DECISION_LOG.md.

Writes outputs/decision_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import load_dataset, geometric_distances


def huber_rmse(dh, p, bs):
    px, py = p[0], p[1]
    bx, by = bs[0], bs[1]
    errs = []
    for u in range(dh.shape[1]):
        du = dh[:, u]

        def fun(xy):
            pred = np.sqrt((bx - xy[0]) ** 2 + (by - xy[1]) ** 2)
            return pred - du

        r = least_squares(
            fun,
            np.array([bx.mean(), by.mean()]),
            loss="huber",
            f_scale=1.0,
            bounds=([-65, -35], [65, 35]),
        )
        errs.append(np.hypot(r.x[0] - px[u], r.x[1] - py[u]))
    e = np.array(errs)
    return float(np.sqrt((e**2).mean()))


def main() -> None:
    data = load_dataset()
    p, dh, bs = data["p"], data["d_hat"], data["bs"]
    N = dh.shape[1]
    d_true = geometric_distances(p, bs)
    resid = dh - d_true

    # global affine
    A = np.column_stack([dh.ravel(), np.ones(dh.size)])
    ab, *_ = np.linalg.lstsq(A, d_true.ravel(), rcond=None)
    dh_g = ab[0] * dh + ab[1]

    # per-BS quad
    dh_q = dh.copy()
    for k in range(18):
        x, y = dh[k], d_true[k]
        A2 = np.column_stack([x, x**2, np.ones(N)])
        c, *_ = np.linalg.lstsq(A2, y, rcond=None)
        dh_q[k] = A2 @ c

    sp = [spearmanr(d_true[:, u], dh[:, u]).correlation for u in range(N)]

    px, py = p[0], p[1]
    r_ue = np.hypot(px - px.mean(), py - py.mean())
    regional = []
    for lo, hi, lab in [(0, 15, "near"), (15, 30, "mid"), (30, 999, "far")]:
        m = (r_ue >= lo) & (r_ue < hi)
        if m.sum() < 5:
            continue
        regional.append(
            {
                "zone": lab,
                "n": int(m.sum()),
                "huber_raw_rmse_m": huber_rmse(dh[:, m], p[:, m], bs),
                "huber_global_calib_rmse_m": huber_rmse(dh_g[:, m], p[:, m], bs),
            }
        )

    out = {
        "n_user": N,
        "corr_dtrue_dhat": float(np.corrcoef(d_true.ravel(), dh.ravel())[0, 1]),
        "spearman_mean": float(np.nanmean(sp)),
        "resid_mean_m": float(resid.mean()),
        "resid_median_m": float(np.median(resid)),
        "ratio_median_dhat_dtrue": float(np.median(dh / np.maximum(d_true, 1e-6))),
        "global_affine": {"alpha": float(ab[0]), "beta": float(ab[1])},
        "huber_rmse_raw_m": huber_rmse(dh, p, bs),
        "huber_rmse_global_affine_m": huber_rmse(dh_g, p, bs),
        "huber_rmse_per_bs_quad_fullfit_m": huber_rmse(dh_q, p, bs),
        "regional_rmse": regional,
    }

    path = ROOT / "outputs" / "decision_analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("Wrote", path)


if __name__ == "__main__":
    main()
