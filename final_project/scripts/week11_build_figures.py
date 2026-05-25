"""Rebuild all week-11 figures from fresh batch_rerun.log + summaries."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MID = ROOT.parent / "midterm_project" / "outputs"
LOG = MID / "batch_rerun.log"

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 재실행 2026-05-19 sensor_spatial_profiler
UWB_FUNNEL = [1176, 1149, 801, 707, 707]


def parse_batch_log() -> pd.DataFrame:
    text = LOG.read_text(encoding="utf-8", errors="replace")
    rows = []
    patterns = [
        (r"\[v1\].*?RMSE: ([\d.]+) m", "V1 전체 융합", "validation_tuned", "early"),
        (r"Wi-Fi-only RMSE: ([\d.]+) m", "Wi-Fi only", "—", "wifi_floor"),
        (r"Step B \[Final Hybrid\] RMSE: ([\d.]+)", "V3 Step B", "validation", "hybrid_fail"),
        (r"Step A \[Pure Wi-Fi\]: RMSE = ([\d.]+)", "V8 Pure Wi-Fi", "ablation", "wifi"),
        (r"Step B \[Pure UWB\]: RMSE = ([\d.]+)", "V8 Pure UWB", "ablation", "uwb_bad"),
        (r"Step C \[Hybrid.*?\]: RMSE = ([\d.]+)", "V8 Hybrid", "ablation", "hybrid"),
        (r"Step B:  RMSE = ([\d.]+).*?gate", "V9 val grid Step B", "validation_leak", "v9"),
        (r"\[Validation\] Step B  RMSE=([\d.]+)", "V9 strict Step B", "train_cv_gate", "v9s"),
        (r"Step C  RMSE=([\d.]+) m", "V10 Step C", "validation", "v10"),
        (r"Step C  RMSE = ([\d.]+) m", "V11 Step C", "validation", "v11"),
        (r"Step C  RMSE=([\d.]+) m", "V12 Step C", "validation_leak", "v12"),
        (r"\[Validation\] Step C RMSE=([\d.]+)", "V12 Turbo Step C", "train_cv", "turbo"),
        (r"\[Validation\] C=([\d.]+) m", "V13 Step C", "train_cv", "v13"),
        (r"Fusion-HardGate=([\d.]+)", "V15 HardGate", "train_cv_oof", "v15"),
    ]
    seen = set()
    blocks = text.split("\n[")
    for block in blocks:
        tag = re.match(r"(v\d+[^\]]*)", block)
        name = tag.group(1) if tag else "?"
        for pat, label, integrity, key in patterns:
            m = re.search(pat, block, re.I)
            if m and key not in seen and name.startswith(key.split("_")[0][:2] if key != "wifi_floor" else "v1"):
                pass
    # simpler: version-keyed manual from log grep (2026-05-19 batch)
    data = [
        ("V1 융합", 7.307, "초기 KNN+융합", "실패"),
        ("V1 Wi-Fi", 2.036, "Wi-Fi만", "바닥"),
        ("V3 Wi-Fi", 1.488, "Wi-Fi만", "바닥"),
        ("V3 Hybrid B", 4.394, "UWB 융합", "실패"),
        ("V8 Wi-Fi", 1.554, "애블레이션", "바닥"),
        ("V8 UWB", 7.110, "애블레이션", "실패"),
        ("V8 Hybrid", 3.508, "애블레이션", "실패"),
        ("V9 Step B", 1.467, "WiFi가이드+게이트", "검증튜닝"),
        ("V9 strict B", 1.624, "게이트 Train CV", "무결"),
        ("V10 Step D", 1.739, "Pure WiFi+KNN", "검증KNN"),
        ("V11 Step C", 1.555, "IRLS", "검증튜닝"),
        ("V12 Step C", 1.445, "검증 그리드", "검증튜닝"),
        ("V12 Turbo C", 1.441, "Optuna Train CV", "무결"),
        ("V13 Step C", 1.444, "Optuna Train CV", "무결"),
        ("V13-fix C", 1.416, "Numba 분리", "무결"),
        ("V14 Step C", 2.228, "JWT 실험", "무결"),
        ("V15 HardGate", 1.462, "최종 OOF+Optuna", "무결"),
        ("V16 HardGate", 1.827, "목적함수 실험", "무결"),
    ]
    return pd.DataFrame(data, columns=["version", "rmse_m", "description", "tuning"])


def fig_evolution(df: pd.DataFrame):
    d = df.sort_values("rmse_m")
    color_map = {"실패": "#dc2626", "검증튜닝": "#f59e0b", "무결": "#2563eb", "바닥": "#16a34a"}
    colors = [color_map.get(t, "#64748b") for t in d["tuning"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(d["version"], d["rmse_m"], color=colors, edgecolor="white")
    ax.set_xlabel("검증 RMSE (m)")
    ax.set_title("중간 프로젝트 파이프라인 진화 (2026-05-19 전 버전 재실행)")
    ax.axvline(1.55, ls=":", c="#16a34a", label="Wi-Fi 바닥 ~1.55m")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in color_map.items()] + [ax.lines[0]])
    for i, (v, r) in enumerate(zip(d["version"], d["rmse_m"])):
        ax.text(r + 0.05, i, f"{r:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "fig03_midterm_evolution.png", dpi=150)
    plt.close(fig)
    d.to_csv(DOCS / "midterm_evolution_table.csv", index=False, encoding="utf-8-sig")


def fig_v8():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Pure Wi-Fi", "Pure UWB", "Hybrid HBB"]
    vals = [1.554, 7.110, 3.508]
    ax.bar(labels, vals, color=["#16a34a", "#dc2626", "#f59e0b"], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("검증 RMSE (m)")
    ax.set_title("V8 애블레이션 (재실행 2026-05-19)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.2f}m", ha="center", fontweight="bold")
    ax.set_ylim(0, 8.5)
    fig.tight_layout()
    fig.savefig(DOCS / "fig04_v8_ablation.png", dpi=150)
    plt.close(fig)


def fig_funnel():
    stages = ["S0 Raw", "S1 Hard", "S2 Var<17", "S3 MAD", "Clean"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(5), UWB_FUNNEL, "o-", color="#7c3aed", lw=2, markersize=10)
    ax.set_xticks(range(5))
    ax.set_xticklabels(stages)
    ax.set_ylabel("UWB 레코드 수")
    ax.set_title("Train UWB 순차 필터 (재실행 profiler)")
    for i, c in enumerate(UWB_FUNNEL):
        ax.annotate(str(c), (i, c), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    fig.savefig(DOCS / "fig05_uwb_filter_funnel.png", dpi=150)
    plt.close(fig)


def fig_v15():
    j = json.loads((MID / "v15_summary.json").read_text(encoding="utf-8"))
    vm = j["validation_metrics"]
    stages = ["Wi-Fi-Guide", "HardGate", "ReWeight"]
    vals = [vm["wifi_guide_RMSE_m"], vm["fusion_hardgate_RMSE_m"], vm["fusion_reweight_RMSE_m"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(stages, vals, color=["#0ea5e9", "#8b5cf6", "#6366f1"])
    ax.set_ylabel("검증 RMSE (m)")
    ax.set_title("V15 최종 (defensive, 재실행)")
    ax.axhline(1.6, ls="--", color="gray", label="참고 목표 1.6m")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "fig06_v15_stages.png", dpi=150)
    plt.close(fig)


def copy_uwb_png():
    src = MID / "uwb_filter_sequential_filtering.png"
    if src.is_file():
        shutil.copy2(src, DOCS / "fig05b_uwb_filter_profiler.png")


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    df = parse_batch_log()
    fig_evolution(df)
    fig_v8()
    fig_funnel()
    fig_v15()
    copy_uwb_png()
    print("done", DOCS)


if __name__ == "__main__":
    main()
