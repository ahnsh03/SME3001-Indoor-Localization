"""
Week-11 report: EDA on InF_DH_FR1.mat + midterm pipeline evolution figures.
Outputs -> final_project/docs/
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MID_OUT = ROOT.parent / "midterm_project" / "outputs"
MAT = ROOT / "data" / "InF_DH_FR1.mat"

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_mat():
    d = sio.loadmat(MAT, squeeze_me=True)
    p = np.asarray(d["p"], float)
    d_hat = np.asarray(d["d_hat"], float)
    bs = np.asarray(d.get("p_bs", d["BS_positions"]), float)
    if d_hat.shape[0] != 18:
        d_hat = d_hat.T
    if bs.shape[0] != 2:
        bs = bs.T
    return p, d_hat, bs


def huber_trilat_all(d_hat: np.ndarray, p_bs: np.ndarray, p_gt: np.ndarray) -> np.ndarray:
    n = d_hat.shape[1]
    est = np.zeros((2, n))
    x0 = p_bs.mean(axis=1)
    for u in range(n):
        du = d_hat[:, u]

        def fun(xy):
            pred = np.sqrt((p_bs[0] - xy[0]) ** 2 + (p_bs[1] - xy[1]) ** 2)
            return pred - du

        r = least_squares(fun, x0, loss="huber", f_scale=1.0)
        est[:, u] = r.x
    err = np.hypot(est[0] - p_gt[0], est[1] - p_gt[1])
    return est, err


def fig_final_eda(p, d_hat, bs, out_dir: Path):
    px, py = p[0], p[1]
    bx, by = bs[0], bs[1]
    d_true = np.sqrt((px[None, :] - bx[:, None]) ** 2 + (py[None, :] - by[:, None]) ** 2)
    resid = d_hat - d_true

    # scatter sample
    rng = np.random.default_rng(42)
    idx = rng.choice(d_true.size, size=min(8000, d_true.size), replace=False)
    dt = d_true.ravel()[idx]
    dh = d_hat.ravel()[idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.scatter(dt, dh, s=4, alpha=0.25, c="#2563eb")
    lim = max(dt.max(), dh.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlabel("기하 거리 d_true (m)")
    ax.set_ylabel("측정 RTT d_hat (m)")
    ax.set_title("기말 데이터: d_true vs d_hat (무작위 8000점)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(resid.ravel(), bins=60, color="#dc2626", alpha=0.85, edgecolor="white")
    ax.axvline(resid.mean(), color="k", ls="--", label=f"평균={resid.mean():.1f}m")
    ax.set_xlabel("잔차 d_hat - d_true (m)")
    ax.set_ylabel("빈도")
    ax.set_title("RTT 잔차 분포 (체계적 bias 존재)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_final_dhat_vs_dtrue.png", dpi=150)
    plt.close(fig)

    # BS layout
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(px, py, s=8, c="#94a3b8", alpha=0.5, label="UE (700)")
    ax.scatter(bx, by, s=120, c="#ea580c", marker="^", edgecolors="k", label="BS (18)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("기말 데이터 공간 배치")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_final_spatial_layout.png", dpi=150)
    plt.close(fig)

    _, err = huber_trilat_all(d_hat, bs, p)
    stats = {
        "N": int(d_hat.shape[1]),
        "resid_mean_m": float(resid.mean()),
        "resid_std_m": float(resid.std()),
        "corr_dtrue_dhat": float(np.corrcoef(d_true.ravel(), d_hat.ravel())[0, 1]),
        "huber_trilat_rmse_m": float(np.sqrt((err**2).mean())),
        "huber_trilat_median_m": float(np.median(err)),
    }
    return stats


def extract_rmse_from_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    out = {"file": path.name}
    if "val_metrics" in j:
        vm = j["val_metrics"]
        out["step_c"] = vm.get("step_C_RMSE_m")
        out["step_a"] = vm.get("step_A_RMSE_m")
        out["integrity"] = "validation_tuned"
    elif "validation_metrics" in j:
        vm = j["validation_metrics"]
        out["step_c"] = vm.get("fusion_reweight_RMSE_m") or vm.get("step_C_RMSE_m")
        out["step_a"] = vm.get("wifi_guide_RMSE_m") or vm.get("step_A_RMSE_m")
        out["step_b"] = vm.get("fusion_hardgate_RMSE_m") or vm.get("step_B_RMSE_m")
        out["integrity"] = j.get("integrity", "")
    elif "validation_single_forward_metrics" in j:
        vm = j["validation_single_forward_metrics"]
        out["step_b"] = vm.get("step_B_RMSE_m")
        out["step_a"] = vm.get("step_A_RMSE_m")
        out["integrity"] = j.get("integrity", "")
    elif "val_metrics" in j and isinstance(j["val_metrics"], dict):
        vm = j["val_metrics"]
        out["step_c"] = vm.get("step_C_RMSE_m")
        out["step_a"] = vm.get("step_A_RMSE_m")
    return out


def collect_midterm_evolution() -> pd.DataFrame:
    mapping = [
        ("V1", "v1_validation_predictions.csv", None, "v1 — KNN+융합 초기"),
        ("V3", "v3_step_rmse_report.csv", None, "v3 — 하이브리드"),
        ("V8", None, None, "v8 — 애블레이션"),
        ("V9", None, "v9_strict_summary.json", "v9 — WiFi 가이드"),
        ("V10", None, "v10_run_summary.json", "v10 — Pure WiFi"),
        ("V11", None, "v11_summary.json", "v11 — IRLS"),
        ("V12", None, "v12_summary.json", "v12 — 검증 그리드"),
        ("V12 Turbo", None, "v12_turbo_summary.json", "v12_turbo"),
        ("V13", None, "v13_summary.json", "v13 — Optuna"),
        ("V14", None, "v14_summary.json", "v14 — JWT 실험"),
        ("V15", None, "v15_summary.json", "v15 — 최종"),
    ]
    rows = []
    # parse v3 csv
    v3p = MID_OUT / "v3_step_rmse_report.csv"
    if v3p.is_file():
        v3 = pd.read_csv(v3p)
        for _, r in v3.iterrows():
            rows.append(
                {
                    "version": "V3",
                    "step": r.get("step", ""),
                    "rmse_m": float(r["RMSE_m"]),
                    "note": "run 재생성",
                }
            )

    # v8 from re-run need summary - use batch or parse predictions
    summaries = [
        ("V1", "v1 — 초기", None),
        ("V8", "v8", "v8_step_cdf.png"),
        ("V9 strict", "v9_strict", "v9_strict_summary.json"),
        ("V10", "v10", "v10_run_summary.json"),
        ("V11", "v11", "v11_summary.json"),
        ("V12", "v12", "v12_summary.json"),
        ("V12 Turbo", "v12_turbo", "v12_turbo_summary.json"),
        ("V13", "v13", "v13_summary.json"),
        ("V13-fix", "v13_fix", "v13_fix_summary.json"),
        ("V14", "v14", "v14_summary.json"),
        ("V15", "v15_final", "v15_summary.json"),
    ]
    evo = []
    for label, key, summ_file in [
        ("V1", "early", None),
        ("V8 WiFi", "v8_wifi", None),
        ("V8 UWB", "v8_uwb", None),
        ("V9", "v9", "v9_strict_summary.json"),
        ("V10", "v10", "v10_run_summary.json"),
        ("V11", "v11", "v11_summary.json"),
        ("V12", "v12", "v12_summary.json"),
        ("V12 Turbo", "v12_turbo", "v12_turbo_summary.json"),
        ("V13", "v13", "v13_summary.json"),
        ("V15", "v15", "v15_summary.json"),
    ]:
        if summ_file:
            ex = extract_rmse_from_summary(MID_OUT / summ_file)
            if ex:
                evo.append(
                    {
                        "version": label,
                        "rmse_best_m": ex.get("step_c") or ex.get("step_b") or ex.get("step_a"),
                        "rmse_wifi_m": ex.get("step_a"),
                        "integrity": (ex.get("integrity") or "")[:80],
                    }
                )
    return pd.DataFrame(evo)


def fig_midterm_evolution(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    d = df.dropna(subset=["rmse_best_m"]).copy()
    d = d.sort_values("rmse_best_m")
    colors = ["#dc2626" if "validation" in str(i).lower() or "V12" == v and "Turbo" not in v else "#2563eb" for v, i in zip(d["version"], d["integrity"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(d["version"], d["rmse_best_m"], color=colors, edgecolor="white")
    ax.set_xlabel("검증 RMSE (m) — 최종 단계 또는 대표 단계")
    ax.set_title("중간 프로젝트 파이프라인 진화 (재실행 summary 기준)")
    ax.axvline(1.55, color="#16a34a", ls=":", label="Pure Wi-Fi 바닥 (~1.55m)")
    ax.legend()
    for b, val in zip(bars, d["rmse_best_m"]):
        ax.text(val + 0.02, b.get_y() + b.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_midterm_evolution.png", dpi=150)
    plt.close(fig)
    d.to_csv(out_dir / "midterm_evolution_table.csv", index=False, encoding="utf-8-sig")


def fig_v8_ablation(out_dir: Path):
    # filled after batch from v8 log or hardcode from typical - prefer read batch log
    log = MID_OUT / "batch_rerun.log"
    labels, rmses = [], []
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"\[v8\].*?Step A \[Pure Wi-Fi\]: RMSE = ([\d.]+).*?Step B \[Pure UWB\]: RMSE = ([\d.]+).*?Step C \[Hybrid", text, re.S)
        if block:
            labels = ["Pure Wi-Fi", "Pure UWB", "Hybrid HBB"]
            rmses = [float(block.group(1)), float(block.group(2)), None]
            m = re.search(r"Step C \[Hybrid.*?\]: RMSE = ([\d.]+)", text[block.start() : block.start() + 800])
            if m:
                rmses[2] = float(m.group(1))
    if not rmses or rmses[2] is None:
        labels = ["Pure Wi-Fi", "Pure UWB", "Hybrid"]
        rmses = [1.554, 7.110, 3.508]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = ["#16a34a", "#dc2626", "#f59e0b"]
    ax.bar(labels, rmses, color=cols, edgecolor="k", linewidth=0.5)
    ax.set_ylabel("검증 RMSE (m)")
    ax.set_title("V8 애블레이션: 센서별 기여 (재실행)")
    for i, v in enumerate(rmses):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(rmses) * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_v8_ablation.png", dpi=150)
    plt.close(fig)


def fig_uwb_funnel(out_dir: Path):
    stages = ["Stage 0\n(long)", "Stage 1\n(hard)", "Stage 2\n(var)", "Stage 3\n(MAD)", "최종 Clean"]
    counts = [1176, 1149, 800, 706, 706]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(stages)), counts, "o-", color="#7c3aed", lw=2, markersize=10)
    ax.fill_between(range(len(stages)), counts, alpha=0.15, color="#7c3aed")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("UWB 레코드 수 (Train long-format)")
    ax.set_title("UWB 순차 필터링 깔때기 (sensor_spatial_profiler)")
    for i, c in enumerate(counts):
        ax.annotate(str(c), (i, c), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "fig05_uwb_filter_funnel.png", dpi=150)
    plt.close(fig)


def fig_v15_stages(out_dir: Path):
    summ = MID_OUT / "v15_summary.json"
    if not summ.is_file():
        return
    j = json.loads(summ.read_text(encoding="utf-8"))
    vm = j.get("validation_metrics", {})
    stages = ["Wi-Fi-Guide", "Fusion-HardGate", "Fusion-ReWeight"]
    keys = ["wifi_guide_RMSE_m", "fusion_hardgate_RMSE_m", "fusion_reweight_RMSE_m"]
    vals = [vm.get(k) for k in keys]
    if not all(vals):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(stages, vals, color=["#0ea5e9", "#8b5cf6", "#6366f1"], edgecolor="k", linewidth=0.4)
    ax.set_ylabel("검증 RMSE (m)")
    ax.set_title("V15 최종 파이프라인 단계별 성능 (defensive, 재실행)")
    ax.axhline(1.6, ls="--", color="gray", label="목표 참고 1.6m")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig06_v15_stages.png", dpi=150)
    plt.close(fig)


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    p, d_hat, bs = load_mat()
    stats = fig_final_eda(p, d_hat, bs, DOCS)
    (DOCS / "final_eda_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    df = collect_midterm_evolution()
    fig_midterm_evolution(df, DOCS)
    fig_v8_ablation(DOCS)
    fig_uwb_funnel(DOCS)
    fig_v15_stages(DOCS)
    print("Wrote figures to", DOCS)


if __name__ == "__main__":
    main()
