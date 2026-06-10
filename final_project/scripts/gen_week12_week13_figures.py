"""Generate figures for 12주차 / 13주차 development reports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT12 = ROOT / "docs" / "12주차"
OUT13 = ROOT / "docs" / "13주차"
OUT12.mkdir(parents=True, exist_ok=True)
OUT13.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)


def _load_rmse(version: str) -> float | None:
    p = ROOT / "outputs" / f"{version}_summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return float(d["oof_position"]["rmse_m"])


def fig12_version_evolution() -> None:
    versions = ["v00", "v01", "v02", "v06", "v11", "v13", "v14", "v16", "v17", "v18"]
    labels = [
        "LS",
        "Huber",
        "전역아핀",
        "Isotonic",
        "γ=1.5",
        "MLP만",
        "asym단독",
        "Iso+MLP",
        "v17★",
        "PosMLP",
    ]
    rmses = [_load_rmse(v) for v in versions]
    colors = ["#94a3b8" if v != "v17" else "#2563eb" for v in versions]
    colors[-1] = "#ef4444"  # v18 fail

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(versions))
    bars = ax.bar(x, rmses, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(10.0, color="#f59e0b", ls="--", lw=1.2, label="기말 합격 목표 10 m")
    ax.axhline(15.0, color="#9ca3af", ls=":", lw=1.0, label="baseline 15 m")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("CV OOF RMSE (m)")
    ax.set_title("12주차: 버전 진화 (v00 → v17, 실패 사례 포함)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(r for r in rmses if r) * 1.05)
    for b, r in zip(bars, rmses):
        if r is not None:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.3,
                f"{r:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.savefig(OUT12 / "fig01_version_evolution.png")
    plt.close(fig)


def fig12_eda_residual() -> None:
    stats = json.loads((ROOT / "docs" / "final_eda_stats.json").read_text(encoding="utf-8"))
    per_bs = stats["per_bs"]
    bs_ids = [r["bs_id"] for r in per_bs]
    means = [r["resid_mean"] for r in per_bs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(bs_ids, means, color="#6366f1", alpha=0.85)
    axes[0].axhline(stats["resid_mean_m"], color="#dc2626", ls="--", label=f"전체 mean={stats['resid_mean_m']:.1f}m")
    axes[0].set_xlabel("BS id")
    axes[0].set_ylabel("d̂ − d_true mean (m)")
    axes[0].set_title("BS별 RTT 편향 (양의 NLOS)")
    axes[0].legend()

    axes[1].axis("off")
    text = (
        f"UE N = {stats['N_train']}\n"
        f"corr(d_true, d̂) = {stats['corr_dtrue_dhat']:.3f}\n"
        f"잔차 mean = {stats['resid_mean_m']:.1f} m\n"
        f"잔차 median = {stats['resid_median_m']:.1f} m\n"
        f"공간 span ≈ {stats['p_x_range'][1]-stats['p_x_range'][0]:.0f} m × "
        f"{stats['p_y_range'][1]-stats['p_y_range'][0]:.0f} m"
    )
    axes[1].text(0.1, 0.5, text, fontsize=12, va="center", family="monospace")
    axes[1].set_title("EDA 요약")
    fig.suptitle("12주차: 기말 데이터 EDA", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT12 / "fig02_eda_bias.png")
    plt.close(fig)


def fig12_cv_folds() -> None:
    d = json.loads((ROOT / "outputs" / "v17_summary.json").read_text(encoding="utf-8"))
    folds = d["folds"]
    fid = [f["fold"] for f in folds]
    rmse = [f["val_rmse_m"] for f in folds]
    oof = d["oof_position"]["rmse_m"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(fid, rmse, color="#0ea5e9", label="fold val RMSE")
    ax.axhline(oof, color="#dc2626", ls="--", lw=2, label=f"OOF={oof:.2f} m")
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("12주차: v17 5-Fold CV (seed=42)")
    ax.legend()
    fig.savefig(OUT12 / "fig03_cv_fold_v17.png")
    plt.close(fig)


def fig12_spatial_zones() -> None:
    tail = json.loads((ROOT / "outputs" / "tail_analysis.json").read_text(encoding="utf-8"))
    zones = tail["by_gt_radius_zone"]
    names = list(zones.keys())
    rmses = [zones[k]["rmse_m"] for k in names]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(names, rmses, color=["#22c55e", "#eab308", "#ef4444"])
    ax.axhline(8.90, color="#2563eb", ls="--", label="전체 v17 OOF")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("12주차: GT 반경별 RMSE (v17)")
    ax.legend()
    fig.savefig(OUT12 / "fig04_spatial_zone_v17.png")
    plt.close(fig)


def fig13_structural() -> None:
    versions = ["v17", "v25", "v26", "v27"]
    labels = ["v17\n(MLP+γ)", "v25\n(+아핀)", "v26\n(far Iso)", "v27\n(quantile)"]
    rmses = [_load_rmse(v) for v in versions]
    colors = ["#64748b", "#2563eb", "#94a3b8", "#94a3b8"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    ax.bar(x, rmses, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CV OOF RMSE (m)")
    ax.set_title("13주차: 구조 변경 실험 (v25 채택)")
    for i, r in enumerate(rmses):
        if r:
            ax.text(i, r + 0.1, f"{r:.2f}", ha="center", fontsize=9)
    fig.savefig(OUT13 / "fig01_structural_v25_v27.png")
    plt.close(fig)


def fig13_combination() -> None:
    cr = json.loads((ROOT / "outputs" / "combination_review.json").read_text(encoding="utf-8"))
    rows = [r for r in cr["results"] if r["version"] in ("v25", "v28", "v29", "v30", "v31", "v34")]
    labels = [r["version"] for r in rows]
    rmses = [r["cv_rmse_m"] for r in rows]
    colors = ["#94a3b8" if v != "v30" else "#16a34a" for v in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, rmses, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CV OOF RMSE (m)")
    ax.set_title("13주차: v25 대비 조합 재검토 (★ v30)")
    ax.axhline(8.762, color="#64748b", ls=":", label="v25=8.76")
    for i, r in enumerate(rmses):
        ax.text(i, r + 0.05, f"{r:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.savefig(OUT13 / "fig02_combination_review.png")
    plt.close(fig)


def fig13_asym_grid() -> None:
    d = json.loads((ROOT / "outputs" / "v30_summary.json").read_text(encoding="utf-8"))
    grid = d["asym_tuning"]["grid"]
    ws = [g["asym_pos_weight"] for g in grid]
    rmses = [g["cv_rmse_m"] for g in grid]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, rmses, "o-", color="#7c3aed", lw=2, markersize=8)
    ax.axvline(5.0, color="#16a34a", ls="--", label="선택 w=5")
    ax.set_xlabel("asym_pos_weight")
    ax.set_ylabel("CV OOF RMSE (m)")
    ax.set_title("13주차: 비대칭 Huber 가중 튜닝 → v30")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(OUT13 / "fig03_asym_grid_v30.png")
    plt.close(fig)


def fig13_stage_v30() -> None:
    d = json.loads((ROOT / "outputs" / "v30_summary.json").read_text(encoding="utf-8"))
    st = d["oof_stage_position"]
    names = ["raw trilat", "main trilat", "최종(+아핀+asym)"]
    keys = ["after_raw_trilat", "after_main_trilat", "after_pos_affine"]
    rmses = [st[k]["rmse_m"] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, rmses, color=["#cbd5e1", "#f97316", "#16a34a"])
    ax.set_ylabel("RMSE (m)")
    ax.set_title("13주차: v30 스테이지별 OOF RMSE")
    ax.tick_params(axis="x", rotation=15)
    for i, r in enumerate(rmses):
        ax.text(i, r + 0.15, f"{r:.2f}", ha="center", fontsize=9)
    fig.savefig(OUT13 / "fig04_stage_decomposition_v30.png")
    plt.close(fig)


def fig13_timeline() -> None:
    milestones = [
        ("v06", 10.51, "Isotonic"),
        ("v16", 9.28, "MLP"),
        ("v17", 8.90, "γ=1"),
        ("v25", 8.76, "아핀"),
        ("v30", 7.996, "asym★"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (ver, rmse, note) in enumerate(milestones):
        ax.plot(i, rmse, "o", ms=12, color="#2563eb" if ver != "v30" else "#16a34a")
        ax.annotate(f"{ver}\n{note}\n{rmse:.2f}m", (i, rmse), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
    ax.plot(range(len(milestones)), [m[1] for m in milestones], "-", color="#94a3b8", alpha=0.6)
    ax.axhline(10, color="#f59e0b", ls="--", alpha=0.7)
    ax.axhline(7, color="#22c55e", ls="--", alpha=0.5, label="G2 목표 7m")
    ax.set_xticks([])
    ax.set_ylabel("CV OOF RMSE (m)")
    ax.set_title("13주차: 최종 개발 마일스톤")
    ax.legend()
    ax.invert_yaxis()
    fig.savefig(OUT13 / "fig05_milestone_timeline.png")
    plt.close(fig)


def main() -> None:
    fig12_version_evolution()
    fig12_eda_residual()
    fig12_cv_folds()
    fig12_spatial_zones()
    fig13_structural()
    fig13_combination()
    fig13_asym_grid()
    fig13_stage_v30()
    fig13_timeline()
    print(f"Saved figures to {OUT12} and {OUT13}")


if __name__ == "__main__":
    main()
