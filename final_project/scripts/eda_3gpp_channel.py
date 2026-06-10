"""
3GPP InF-DH-FR1 가설 검증 EDA: 잔차(d_hat - d_true) 분포·편향.

Outputs:
  outputs/eda_3gpp.json
  docs/fig_eda_residual_hist.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import geometric_distances, load_dataset, project_root


def main() -> None:
    data = load_dataset()
    p, bs, d_hat = data["p"], data["bs"], data["d_hat"]
    d_true = geometric_distances(p, bs)
    resid = (d_hat - d_true).ravel()

    pos_frac = float(np.mean(resid > 0))
    summary = {
        "n_pairs": int(resid.size),
        "mean_m": float(np.mean(resid)),
        "median_m": float(np.median(resid)),
        "std_m": float(np.std(resid)),
        "skew": float(stats.skew(resid)),
        "kurtosis": float(stats.kurtosis(resid)),
        "fraction_positive": pos_frac,
        "p90_m": float(np.percentile(resid, 90)),
        "p99_m": float(np.percentile(resid, 99)),
        "ratio_median_dhat_over_dtrue": float(
            np.median(d_hat.ravel() / np.maximum(d_true.ravel(), 1e-6))
        ),
        "hypothesis": "InF-DH NLOS positive bias (d_hat > d_true)",
        "hypothesis_supported": bool(np.mean(resid) > 5.0 and pos_frac > 0.55),
    }

    out_dir = project_root() / "outputs"
    docs = project_root() / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    (out_dir / "eda_3gpp.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(resid, bins=100, density=True, alpha=0.75, color="#2563eb")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="zero error")
    ax.axvline(summary["mean_m"], color="#16a34a", linestyle="-", label=f"mean={summary['mean_m']:.1f}m")
    ax.set_title("Residual: d_hat - d_true (InF_DH_FR1 train)")
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig_path = docs / "fig_eda_residual_hist.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out_dir / 'eda_3gpp.json'}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
