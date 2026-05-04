"""
V12 Strict (Parallel) — `indoor_fusion_pipeline_v12_strict.py`와 동일한 수식·무결성 규칙에 더해:

- **K_FOLD_SPLITS** 글로벌로 K-Fold 폴드 수를 제어. `1`이면 KFold 없이 전체 Train에 피팅 후 동일 Train에서 자기 채점(빠른 실험용, 낙관적 편향).
- **Phase B/C** 하이퍼 그리드는 `ProcessPoolExecutor`로 병렬 평가(Windows spawn + top-level worker).

실행: `py -3 scripts/indoor_fusion_pipeline_v12_strict_parallel.py`
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from fusion_realtime_sanitize import resolve_train_kgh_corrected_paths, resolve_validation_paths
from indoor_fusion_pipeline_v10 import WifiOnlyTrilateration, resolve_paths
from indoor_fusion_pipeline_v11 import CalibMethodSpec, fit_robust_calibration_per_ap
from indoor_fusion_pipeline_v12 import (
    FEAT_MODES,
    GATE_THRESH_GRID,
    HUBER_F_FUSION_GRID,
    HUBER_F_WIFI_GRID,
    IRLS_MAX_ITER_GRID,
    IRLS_THRESH_GRID_M,
    TOP_A_CANDIDATES,
    UWB_VAR_INFLATE_GRID,
    augment_train_for_canon_fit,
    irls_refine_xyz,
    point_errors,
    predict_step_a_v3_wifi_equivalent,
    rmse_mae,
    row_fusion_gated_arrays,
    row_wifi_calibrated_arrays,
    solve_trilat_array,
    v12_calibration_spec_list,
    wifi_median_impute_fusion,
)
from indoor_fusion_pipeline_v8 import WIFI_COL_CANON, Config, FusionLocalizerV8

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except (AttributeError, OSError):
        pass
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 실험 제어: None 또는 0 = 자동 폴드 수, 1 = KFold 생략·전체 Train 자기채점, 정수>=2 = 명시 폴드
# ---------------------------------------------------------------------------
K_FOLD_SPLITS: Optional[int] = None

MAX_KFOLD_SPLITS = 3
ART_PREFIX = "v12_strict_parallel"


def resolve_kfold_n_splits(n_samples: int) -> int:
    if n_samples < 2:
        raise ValueError("Train 샘플이 2개 미만이면 K-Fold를 할 수 없습니다.")
    if n_samples < 6:
        return 2
    return int(min(5, max(3, n_samples // 35)))


def resolve_effective_cv_splits(
    n_train: int,
    k_fold_splits: Optional[int],
    *,
    max_kfold_splits: int,
) -> Tuple[int, str]:
    """
    반환: (effective_splits, mode_tag)
    - 1: KFold 미사용, 전체 Train 피팅 후 동일 Train에서 RMSE (낙관적 편향 허용).
    - >=2: KFold(n_splits=effective_splits).
    """
    kfs: Optional[int] = k_fold_splits
    if kfs == 0:
        kfs = None
    if kfs == 1:
        if n_train < 1:
            raise ValueError("Train이 비어 있습니다.")
        return 1, "K1_no_fold_self_score_on_train"

    if kfs is None:
        k_auto = min(max_kfold_splits, resolve_kfold_n_splits(n_train))
        k_eff = max(2, min(k_auto, n_train))
        return k_eff, "kfold_auto"

    k_req = int(kfs)
    if k_req < 2:
        return 1, "K1_no_fold_self_score_on_train"
    k_eff = min(k_req, n_train)
    if k_eff < 2:
        raise ValueError(f"K_FOLD_SPLITS={k_fold_splits!r}는 train 행 수 {n_train}와 맞지 않습니다(최소 2 폴드 필요).")
    # 명시 정수 K는 MAX_KFOLD_SPLITS로 추가 캡하지 않음(실험 유연성).
    return k_eff, "kfold_explicit"


def cv_mean_rmse_v3_step_a(train_imp: pd.DataFrame, loc: FusionLocalizerV8, n_splits: int) -> float:
    """V3 동일 Wi-Fi 전용 Step A의 Train 측 RMSE: n_splits==1이면 자기채점 1회, 아니면 K-Fold 평균."""
    if n_splits == 1:
        te_df = train_imp.reset_index(drop=True)
        xy_a = predict_step_a_v3_wifi_equivalent(te_df, loc)
        tx = te_df["True_X"].values.astype(float)
        ty = te_df["True_Y"].values.astype(float)
        ea = point_errors(tx, ty, xy_a[:, 0], xy_a[:, 1])
        r, _ = rmse_mae(ea)
        return float(r)

    n_splits = min(n_splits, len(train_imp))
    if n_splits < 2:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses: List[float] = []
    for _, te_idx in kf.split(np.arange(len(train_imp))):
        te_df = train_imp.iloc[te_idx].reset_index(drop=True)
        xy_a = predict_step_a_v3_wifi_equivalent(te_df, loc)
        tx = te_df["True_X"].values.astype(float)
        ty = te_df["True_Y"].values.astype(float)
        ea = point_errors(tx, ty, xy_a[:, 0], xy_a[:, 1])
        r, _ = rmse_mae(ea)
        rmses.append(float(r))
    return float(np.mean(rmses))


def build_phase_bc_fold_cache(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    top_a: List[Tuple[float, CalibMethodSpec, str, float]],
    n_splits: int,
) -> List[Dict[str, Any]]:
    """n_splits==1: KFold 없이 전체 Train 1블록. n_splits>=2: 기존 K-Fold 캐시."""
    if n_splits == 1:
        train_df = train_imp.reset_index(drop=True)
        tr_fit = augment_train_for_canon_fit(train_df, loc)
        xy_v3_all = predict_step_a_v3_wifi_equivalent(train_df, loc)
        per_rank: List[Dict[str, Any]] = []
        for (_cv_ra, spec, feat, hf_wifi) in top_a:
            coeffs = fit_robust_calibration_per_ap(tr_fit, tri_ref, feat_mode=feat, spec=spec)
            per_rank.append(dict(coeffs=coeffs, xy_a=xy_v3_all, feat=feat, hf_wifi=float(hf_wifi)))
        return [dict(te_df=train_df, per_rank=per_rank)]

    n_splits = min(n_splits, len(train_imp))
    if n_splits < 2:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_cache: List[Dict[str, Any]] = []
    for tr_idx, te_idx in kf.split(np.arange(len(train_imp))):
        tr_df = train_imp.iloc[tr_idx].reset_index(drop=True)
        te_df = train_imp.iloc[te_idx].reset_index(drop=True)
        tr_fit = augment_train_for_canon_fit(tr_df, loc)
        xy_v3_te = predict_step_a_v3_wifi_equivalent(te_df, loc)
        per_rank = []
        for (_cv_ra, spec, feat, hf_wifi) in top_a:
            coeffs = fit_robust_calibration_per_ap(tr_fit, tri_ref, feat_mode=feat, spec=spec)
            per_rank.append(dict(coeffs=coeffs, xy_a=xy_v3_te, feat=feat, hf_wifi=float(hf_wifi)))
        fold_cache.append(dict(te_df=te_df, per_rank=per_rank))
    return fold_cache


def mean_cv_rmse_step_c_from_cache(
    fold_cache: List[Dict[str, Any]],
    loc: FusionLocalizerV8,
    rank: int,
    gate: float,
    uvb_inf: float,
    hf_fus: float,
    ith: float,
    mx: int,
) -> float:
    rmses: List[float] = []
    for fc in fold_cache:
        te_df: pd.DataFrame = fc["te_df"]
        blk = fc["per_rank"][rank]
        coeffs = blk["coeffs"]
        xy_a = blk["xy_a"]
        feat = blk["feat"]
        hf_wifi = blk["hf_wifi"]
        pdw_store: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for k, (_, row) in enumerate(te_df.iterrows()):
            p, d, w = row_fusion_gated_arrays(row, loc, xy_a[k], gate, coeffs, feat, uvb_inf)
            pdw_store.append((p, d, w))
        preds_c = np.full((len(te_df), 2), np.nan, dtype=float)
        for k, (p, d, w) in enumerate(pdw_store):
            if len(d) < 3:
                p2, d2, w2 = row_wifi_calibrated_arrays(te_df.iloc[k], loc, coeffs, feat)
                preds_c[k] = solve_trilat_array(p2, d2, w2, huber_f_scale=hf_wifi)
            else:
                preds_c[k] = irls_refine_xyz(p, d, w, hf_fus, float(ith), int(mx))
        tx = te_df["True_X"].values.astype(float)
        ty = te_df["True_Y"].values.astype(float)
        ec = point_errors(tx, ty, preds_c[:, 0], preds_c[:, 1])
        rc, _ = rmse_mae(ec)
        rmses.append(float(rc))
    return float(np.mean(rmses))


# --- Windows spawn 호환: 모듈 최상위 Worker (pickle 가능) -------------------


def _phase_bc_grid_worker(
    args: Tuple[List[Dict[str, Any]], FusionLocalizerV8, int, float, float, float, float, int],
) -> Tuple[int, float, float, float, float, int, float]:
    """
    단일 (rank, gate, uwb_inflate, hf_fusion, irls_thresh, irls_max_iter) 조합에 대한
    Train CV Step C 평균 RMSE를 계산한다.

    반환: (rank, gate, uvb_inf, hf_fus, ith, mx, train_CV_mean_RMSE_C_m)
    """
    fold_cache, loc, rank, gate, uvb_inf, hf_fus, ith, mx = args
    cv_rc = mean_cv_rmse_step_c_from_cache(
        fold_cache,
        loc,
        int(rank),
        float(gate),
        float(uvb_inf),
        float(hf_fus),
        float(ith),
        int(mx),
    )
    return (int(rank), float(gate), float(uvb_inf), float(hf_fus), float(ith), int(mx), float(cv_rc))


def run_phase_bc_grid_parallel(
    fold_cache: List[Dict[str, Any]],
    loc: FusionLocalizerV8,
    top_a: List[Tuple[float, CalibMethodSpec, str, float]],
    *,
    max_workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Phase B/C 그리드 전체를 병렬 평가하고, (phase_bc_records, best_sel)을 반환한다.
    best_sel은 strict 본문과 동일 키를 유지한다.
    """
    tasks: List[Tuple[List[Dict[str, Any]], FusionLocalizerV8, int, float, float, float, float, int]] = []
    for rank, (_cv_ra0, _spec, _feat, _hf) in enumerate(top_a):
        for gate in GATE_THRESH_GRID:
            for uvb_inf in UWB_VAR_INFLATE_GRID:
                for hf_fus in HUBER_F_FUSION_GRID:
                    for ith in IRLS_THRESH_GRID_M:
                        for mx in IRLS_MAX_ITER_GRID:
                            tasks.append(
                                (fold_cache, loc, int(rank), float(gate), float(uvb_inf), float(hf_fus), float(ith), int(mx))
                            )

    n_workers = max(1, int(max_workers))
    phase_bc_records: List[Dict[str, Any]] = []
    best_cv_c = float("inf")
    best_sel: Optional[Dict[str, Any]] = None

    chunksize = max(1, len(tasks) // (n_workers * 4)) if len(tasks) >= n_workers * 4 else 1

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        raw_results = list(ex.map(_phase_bc_grid_worker, tasks, chunksize=chunksize))

    for rank, gate, uvb_inf, hf_fus, ith, mx, cv_rc in raw_results:
        cv_ra0, spec, feat, hf_wifi = top_a[rank]
        row = dict(
            phaseA_rank=int(rank),
            feat_mode=str(feat),
            calib_kind=spec.kind,
            huber_f_wifi=float(hf_wifi),
            gate_threshold_m=float(gate),
            uwb_variance_inflate=float(uvb_inf),
            huber_f_fusion=float(hf_fus),
            irls_residual_thresh_m=float(ith),
            irls_max_iter=int(mx),
            train_CV_mean_RMSE_C_m=float(cv_rc),
        )
        phase_bc_records.append(row)
        if cv_rc < best_cv_c - 1e-14:
            best_cv_c = float(cv_rc)
            best_sel = dict(
                spec=spec,
                feat=str(feat),
                hf_wifi=float(hf_wifi),
                gate=float(gate),
                uwb_inf=float(uvb_inf),
                hf_fus=float(hf_fus),
                irls_thresh=float(ith),
                irls_mx=int(mx),
                train_cv_rmse_A_proxy=float(cv_ra0),
                train_cv_rmse_C=float(cv_rc),
                phase_a_rank=int(rank),
            )

    if best_sel is None:
        raise RuntimeError("Phase B/C 병렬 그리드: 결과 없음")
    phase_bc_records.sort(
        key=lambda r: (
            r["train_CV_mean_RMSE_C_m"],
            r["phaseA_rank"],
            r["gate_threshold_m"],
            r["uwb_variance_inflate"],
        )
    )
    return phase_bc_records, best_sel


def save_cdf_steps_parallel(
    out_dir: Path,
    errs: Dict[str, np.ndarray],
    rmse_by: Dict[str, float],
    labels: Dict[str, str],
) -> None:
    plt.figure(figsize=(8.2, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(errs)))
    for ki, key in enumerate(sorted(errs.keys())):
        e = errs[key][np.isfinite(errs[key])]
        if e.size == 0:
            continue
        e = np.sort(e)
        cdf = np.arange(1, len(e) + 1) / len(e)
        plt.plot(e, cdf, lw=2.0, color=colors[ki], label=f"{labels[key]} (RMSE {rmse_by[key]:.3f} m)")
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("V12 Strict Parallel: 검증 오차 CDF — Step A/B/C")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_cdf_steps.png", dpi=200)
    plt.close()


def save_maps_parallel(
    out_dir: Path,
    tx: np.ndarray,
    ty: np.ndarray,
    xy_a: np.ndarray,
    preds_b: np.ndarray,
    preds_c: np.ndarray,
    rmse_a: float,
    rmse_b: float,
    rmse_c: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)
    triples = [
        (xy_a[:, 0], xy_a[:, 1], "Step A (Wi‑Fi)", rmse_a),
        (preds_b[:, 0], preds_b[:, 1], "Step B (+ UWB 게이트)", rmse_b),
        (preds_c[:, 0], preds_c[:, 1], "Step C (+ IRLS)", rmse_c),
    ]
    for ax, (px, py, ttl, rm) in zip(axes, triples):
        m = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(px) & np.isfinite(py)
        txv, tyv = tx[m], ty[m]
        pxv, pyv = px[m], py[m]
        ax.scatter(txv, tyv, s=42, c="#1f77b4", alpha=0.88, edgecolors="#0d2c44", linewidths=0.35, label="실제", zorder=2)
        ax.scatter(pxv, pyv, s=52, c="#ff7f0e", alpha=0.78, marker="x", linewidths=0.95, label="추정", zorder=3)
        for xi, yi, pxi, pyi in zip(txv, tyv, pxv, pyv):
            ax.plot([xi, pxi], [yi, pyi], color="#7f7f7f", alpha=0.22, lw=0.85, zorder=1)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 18)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{ttl}\n(RMSE {rm:.3f} m)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("V12 Strict Parallel — 검증 단일 Forward", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_dir / f"{ART_PREFIX}_map_true_vs_predicted_steps.png", dpi=200)
    plt.close()


def save_error_vectors_parallel(
    out_dir: Path,
    tx: np.ndarray,
    ty: np.ndarray,
    preds_c: np.ndarray,
    err_c_m: np.ndarray,
) -> None:
    m = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(preds_c[:, 0]) & np.isfinite(preds_c[:, 1])
    if not np.any(m):
        return
    fig, ax = plt.subplots(figsize=(8.0, 10.8))
    u = preds_c[m, 0] - tx[m]
    v = preds_c[m, 1] - ty[m]
    err = err_c_m[m]
    if np.any(np.isfinite(err)):
        er = np.where(np.isfinite(err), err, 0.0)
        sc = ax.scatter(tx[m], ty[m], c=er, cmap="viridis", s=54, alpha=0.92, edgecolors="k", linewidths=0.35, zorder=3)
        fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02).set_label("오차 (m)")
    ax.quiver(tx[m], ty[m], u, v, angles="xy", scale_units="xy", scale=1, width=0.0032, alpha=0.5, color="navy", zorder=2)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Step C 오차 벡터")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_map_error_vectors_stepC.png", dpi=200)
    plt.close()


def save_pairwise_parallel(
    out_dir: Path,
    tx: np.ndarray,
    ty: np.ndarray,
    preds_c: np.ndarray,
    err_c_m: np.ndarray,
) -> None:
    m = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(preds_c[:, 0]) & np.isfinite(preds_c[:, 1])
    if not np.any(m):
        return
    err = np.asarray(err_c_m[m], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 10.8))
    ax.scatter(tx[m], ty[m], s=140, facecolors="none", edgecolors="dimgray", linewidths=1.1, label="실제", zorder=3)
    cmap = plt.cm.RdYlGn_r
    finite_e = err[np.isfinite(err)]
    if finite_e.size >= 5:
        vmin = float(np.percentile(finite_e, 5))
        vmax = float(np.percentile(finite_e, 95))
        if vmax <= vmin:
            vmin, vmax = 0.0, max(float(np.max(finite_e)), 0.5)
    else:
        vmin, vmax = 0.0, 3.0
    z = np.asarray(err_c_m[m], dtype=float)
    z[~np.isfinite(z)] = np.nan
    sm = ax.scatter(
        preds_c[m, 0],
        preds_c[m, 1],
        c=z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=36,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.25,
        label="Step C",
        zorder=4,
    )
    plt.colorbar(sm, ax=ax, shrink=0.72, label="|오차| (m)")
    for xi, yi, pxi, pyi in zip(tx[m], ty[m], preds_c[m, 0], preds_c[m, 1]):
        ax.plot([xi, pxi], [yi, pyi], color="gray", alpha=0.35, lw=0.85, zorder=2)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("실제 vs 추정 (Step C)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_map_stepC_pairwise_truth_pred.png", dpi=200)
    plt.close()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "train"
    val_dir = root / "data" / "validation"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tm, tv = resolve_train_kgh_corrected_paths(train_dir)
    vm, vd = resolve_validation_paths(val_dir)
    cfg = Config(grid_size_m=0.6, huber_f_scale=1.0)
    loc = FusionLocalizerV8(cfg)
    train_df, val_df = loc.load_datasets(tm, tv, vm, vd)

    n_splits, split_mode = resolve_effective_cv_splits(
        len(train_df),
        K_FOLD_SPLITS,
        max_kfold_splits=MAX_KFOLD_SPLITS,
    )
    cpu_n = os.cpu_count() or 1
    print(
        f"\n[V12 Strict Parallel] 학습 {len(train_df)}행 | 검증 {len(val_df)}행 | "
        f"effective_splits={n_splits} ({split_mode}) | K_FOLD_SPLITS={K_FOLD_SPLITS!r} | workers={cpu_n}",
        flush=True,
    )

    train_imp, val_imp = wifi_median_impute_fusion(train_df, val_df, loc.wifi_cols)
    tri_ref = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m=0.6, huber_f_scale=1.0)
    calib_specs = v12_calibration_spec_list()

    cv_rmse_v3_a = cv_mean_rmse_v3_step_a(train_imp, loc, n_splits)
    phase_a_records: List[Dict[str, Any]] = []
    cand_a: List[Tuple[float, CalibMethodSpec, str, float]] = []

    for feat in FEAT_MODES:
        for spec in calib_specs:
            for hf in HUBER_F_WIFI_GRID:
                phase_a_records.append(
                    dict(
                        feat_mode=feat,
                        calib_kind=spec.kind,
                        huber_f_wifi=float(hf),
                        train_CV_mean_RMSE_A_m=float(cv_rmse_v3_a),
                        train_CV_mean_RMSE_A_note=(
                            "V3-equivalent Wi-Fi-only (K>=2: K-fold mean; K=1: self-score on full train)"
                        ),
                        kfold_splits=int(n_splits),
                        cv_split_mode=split_mode,
                    )
                )
                cand_a.append((float(cv_rmse_v3_a), spec, feat, float(hf)))

    cand_a.sort(key=lambda t: (t[0], t[1].kind, t[2], t[3]))
    top_a = cand_a[:TOP_A_CANDIDATES]

    print("[V12 Strict Parallel] Phase B/C 폴드 캐시 구축 중…", flush=True)
    fold_cache = build_phase_bc_fold_cache(train_imp, loc, tri_ref, top_a, n_splits)

    print(f"[V12 Strict Parallel] Phase B/C 그리드 병렬 평가 (workers={cpu_n})…", flush=True)
    phase_bc_records, best_sel = run_phase_bc_grid_parallel(fold_cache, loc, top_a, max_workers=cpu_n)

    train_fit_full = augment_train_for_canon_fit(train_imp, loc)
    coeffs_final = fit_robust_calibration_per_ap(
        train_fit_full, tri_ref, feat_mode=best_sel["feat"], spec=best_sel["spec"]
    )

    xy_a_val = predict_step_a_v3_wifi_equivalent(val_imp, loc)
    pdw_store: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for k, (_, row) in enumerate(val_imp.iterrows()):
        p, d, w = row_fusion_gated_arrays(
            row,
            loc,
            xy_a_val[k],
            best_sel["gate"],
            coeffs_final,
            best_sel["feat"],
            best_sel["uwb_inf"],
        )
        pdw_store.append((p, d, w))

    preds_b_final = np.full((len(val_imp), 2), np.nan, dtype=float)
    for k, (p, d, w) in enumerate(pdw_store):
        preds_b_final[k] = solve_trilat_array(p, d, w, huber_f_scale=best_sel["hf_fus"])

    preds_c_final = np.full((len(val_imp), 2), np.nan, dtype=float)
    for k, (p, d, w) in enumerate(pdw_store):
        if len(d) < 3:
            p2, d2, w2 = row_wifi_calibrated_arrays(val_imp.iloc[k], loc, coeffs_final, best_sel["feat"])
            preds_c_final[k] = solve_trilat_array(p2, d2, w2, huber_f_scale=best_sel["hf_wifi"])
        else:
            preds_c_final[k] = irls_refine_xyz(
                p, d, w, best_sel["hf_fus"], best_sel["irls_thresh"], best_sel["irls_mx"]
            )

    tx = val_imp["True_X"].values.astype(float)
    ty = val_imp["True_Y"].values.astype(float)
    ea = point_errors(tx, ty, xy_a_val[:, 0], xy_a_val[:, 1])
    eb = point_errors(tx, ty, preds_b_final[:, 0], preds_b_final[:, 1])
    ec = point_errors(tx, ty, preds_c_final[:, 0], preds_c_final[:, 1])
    rmse_a, mae_a = rmse_mae(ea)
    rmse_b, mae_b = rmse_mae(eb)
    rmse_c, mae_c = rmse_mae(ec)
    d_ab = rmse_a - rmse_b
    d_bc = rmse_b - rmse_c
    d_ac = rmse_a - rmse_c

    print("\n" + "=" * 72)
    print("V12 Strict Parallel — Train CV(또는 K=1 자기채점) | 검증 단일 Forward")
    print("=" * 72)
    print(f"  [Train 목표] Step C 평균 RMSE = {best_sel['train_cv_rmse_C']:.4f} m  |  모드: {split_mode}")
    print(f"  [Validation] Step A  RMSE={rmse_a:.4f} m  MAE={mae_a:.4f} m")
    print(f"  [Validation] Step B  RMSE={rmse_b:.4f} m  MAE={mae_b:.4f} m")
    print(f"  [Validation] Step C  RMSE={rmse_c:.4f} m  MAE={mae_c:.4f} m")
    print("=" * 72)

    pref = resolve_paths(root)
    summ = {
        "pipeline": "V12 Strict Parallel",
        "K_FOLD_SPLITS_global": K_FOLD_SPLITS,
        "effective_train_cv_splits": int(n_splits),
        "cv_split_mode": split_mode,
        "parallel_workers": int(cpu_n),
        "train_cv_metrics_at_selection": {
            "phase_A_proxy_mean_RMSE_A_m": float(best_sel["train_cv_rmse_A_proxy"]),
            "mean_CV_RMSE_step_C_m": float(best_sel["train_cv_rmse_C"]),
        },
        "validation_single_forward_metrics": {
            "step_A_RMSE_m": rmse_a,
            "step_A_MAE_m": mae_a,
            "step_B_RMSE_m": rmse_b,
            "step_B_MAE_m": mae_b,
            "step_C_RMSE_m": rmse_c,
            "step_C_MAE_m": mae_c,
            "delta_RMSE_A_to_B_m": float(d_ab),
            "delta_RMSE_B_to_C_m": float(d_bc),
            "delta_RMSE_A_to_C_m": float(d_ac),
        },
        "selected_hyperparameters": {
            "wifi_calibration_kind": best_sel["spec"].kind,
            "feat_mode": best_sel["feat"],
            "huber_f_scale_wifi": best_sel["hf_wifi"],
            "gate_m": best_sel["gate"],
            "uwb_variance_inflate": best_sel["uwb_inf"],
            "huber_f_fusion": best_sel["hf_fus"],
            "irls_residual_thresh_m": best_sel["irls_thresh"],
            "irls_max_iter": best_sel["irls_mx"],
            "phase_A_rank": int(best_sel["phase_a_rank"]),
        },
        "data_paths": {
            "train_median": str(pref.train_median),
            "train_variance": str(pref.train_variance),
            "val_median": str(pref.val_median),
            "val_variance": str(pref.val_variance),
        },
    }
    (out_dir / f"{ART_PREFIX}_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")

    pred = val_imp[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xy_a_val[:, 0]
    pred["StepA_Y"] = xy_a_val[:, 1]
    pred["StepB_X"] = preds_b_final[:, 0]
    pred["StepB_Y"] = preds_b_final[:, 1]
    pred["StepC_X"] = preds_c_final[:, 0]
    pred["StepC_Y"] = preds_c_final[:, 1]
    pred["ErrA_m"] = ea
    pred["ErrB_m"] = eb
    pred["ErrC_m"] = ec
    pred.to_csv(out_dir / f"{ART_PREFIX}_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phase_a_records).to_csv(out_dir / f"{ART_PREFIX}_grid_phaseA.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phase_bc_records).to_csv(out_dir / f"{ART_PREFIX}_grid_phaseBC.csv", index=False, encoding="utf-8-sig")
    rows_coef = [{"wifi_canon_ap": k, "A": v[0], "B": v[1]} for k, v in coeffs_final.items()]
    pd.DataFrame(rows_coef).to_csv(out_dir / f"{ART_PREFIX}_wifi_calibration_coefs.csv", index=False, encoding="utf-8-sig")

    save_cdf_steps_parallel(out_dir, dict(A=ea, B=eb, C=ec), dict(A=rmse_a, B=rmse_b, C=rmse_c), dict(A="Step A", B="Step B", C="Step C"))
    save_maps_parallel(out_dir, tx, ty, xy_a_val, preds_b_final, preds_c_final, rmse_a, rmse_b, rmse_c)
    save_error_vectors_parallel(out_dir, tx, ty, preds_c_final, ec)
    save_pairwise_parallel(out_dir, tx, ty, preds_c_final, ec)

    print(f"\n저장: {out_dir / f'{ART_PREFIX}_summary.json'}")
    print(f"저장: {out_dir / f'{ART_PREFIX}_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
