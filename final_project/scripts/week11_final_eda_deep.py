"""Deep EDA for InF_DH_FR1.mat -> docs/"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MAT = ROOT / "data" / "InF_DH_FR1.mat"

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load():
    d = sio.loadmat(MAT, squeeze_me=True)
    p = np.asarray(d["p"], float)
    dh = np.asarray(d["d_hat"], float)
    bs = np.asarray(d.get("p_bs", d["BS_positions"]), float)
    if dh.shape[0] != 18:
        dh = dh.T
    if bs.shape[0] != 2:
        bs = bs.T
    return p, dh, bs


def huber_all(dh, bs, p_gt):
    n = dh.shape[1]
    est = np.zeros((2, n))
    x0 = bs.mean(axis=1)
    errs = []
    for u in range(n):
        du = dh[:, u]

        def fun(xy):
            pred = np.sqrt((bs[0] - xy[0]) ** 2 + (bs[1] - xy[1]) ** 2)
            return pred - du

        r = least_squares(fun, x0, loss="huber", f_scale=1.0)
        est[:, u] = r.x
        errs.append(np.hypot(r.x[0] - p_gt[0, u], r.x[1] - p_gt[1, u]))
    return np.array(errs), est


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    p, dh, bs = load()
    n = dh.shape[1]
    px, py = p[0], p[1]
    bx, by = bs[0], bs[1]
    d_true = np.sqrt((px[None, :] - bx[:, None]) ** 2 + (py[None, :] - by[:, None]) ** 2)
    resid = dh - d_true

    stats = {
        "N_train": int(n),
        "p_x_range": [float(p[0].min()), float(p[0].max())],
        "p_y_range": [float(p[1].min()), float(p[1].max())],
        "bs_x_range": [float(bx.min()), float(bx.max())],
        "bs_y_range": [float(by.min()), float(by.max())],
        "corr_dtrue_dhat": float(np.corrcoef(d_true.ravel(), dh.ravel())[0, 1]),
        "resid_mean_m": float(resid.mean()),
        "resid_std_m": float(resid.std()),
        "resid_median_m": float(np.median(resid)),
        "ratio_dhat_over_dtrue_median": float(np.median(dh.ravel() / (d_true.ravel() + 1e-9))),
    }

    per_bs = []
    for k in range(18):
        r = resid[k]
        per_bs.append(
            {
                "bs_id": int(k),
                "bs_x": float(bx[k]),
                "bs_y": float(by[k]),
                "resid_mean": float(r.mean()),
                "resid_std": float(r.std()),
                "d_true_mean": float(d_true[k].mean()),
                "d_hat_mean": float(dh[k].mean()),
            }
        )
    stats["per_bs"] = per_bs

  # per-user residual RMS
    user_rms = np.sqrt((resid**2).mean(axis=0))
    stats["user_residual_rms_mean"] = float(user_rms.mean())
    stats["user_residual_rms_p90"] = float(np.percentile(user_rms, 90))

    errs, _ = huber_all(dh, bs, p)
    stats["huber_trilat_rmse_m"] = float(np.sqrt((errs**2).mean()))
    stats["huber_trilat_median_m"] = float(np.median(errs))
    stats["huber_trilat_p90_m"] = float(np.percentile(errs, 90))

    # LS linear (no robust)
    errs_ls = []
    for u in range(n):
        def fun(xy):
            pred = np.sqrt((bs[0] - xy[0]) ** 2 + (bs[1] - xy[1]) ** 2)
            return pred - dh[:, u]

        r = least_squares(fun, bs.mean(axis=1), loss="linear")
        errs_ls.append(np.hypot(r.x[0] - p[0, u], r.x[1] - p[1, u]))
    errs_ls = np.array(errs_ls)
    stats["ls_trilat_rmse_m"] = float(np.sqrt((errs_ls**2).mean()))

    (DOCS / "final_eda_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # fig: per-BS mean residual
    fig, ax = plt.subplots(figsize=(10, 4))
    means = [b["resid_mean"] for b in per_bs]
    ax.bar(range(18), means, color="#6366f1", edgecolor="white")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("기지국 인덱스 k")
    ax.set_ylabel("평균 잔차 d_hat - d_true (m)")
    ax.set_title("기지국별 RTT 체계적 bias (700명 평균)")
    fig.tight_layout()
    fig.savefig(DOCS / "fig07_final_per_bs_residual_mean.png", dpi=150)
    plt.close(fig)

    # fig: positioning error CDF
    fig, ax = plt.subplots(figsize=(7, 4.5))
    se = np.sort(errs)
    cdf = np.arange(1, n + 1) / n
    ax.plot(se, cdf, lw=2, color="#2563eb", label=f"Huber 삼변 (RMSE={stats['huber_trilat_rmse_m']:.1f}m)")
    se2 = np.sort(errs_ls)
    ax.plot(se2, cdf, lw=2, color="#94a3b8", ls="--", label=f"LS 삼변 (RMSE={stats['ls_trilat_rmse_m']:.1f}m)")
    ax.axvline(1.55, color="#16a34a", ls=":", label="중간 Pure Wi-Fi (~1.55m)")
    ax.set_xlabel("위치 오차 (m)")
    ax.set_ylabel("CDF")
    ax.set_title("기말 데이터: naive 삼변측량 오차 분포")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "fig08_final_trilat_error_cdf.png", dpi=150)
    plt.close(fig)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
