"""
V12 Fast2 — 검증 정답 없이(Train CV만) 하이퍼파라미터를 고르고, v12가 검증 튜닝으로 얻은
성능에 가깝게 UWB 융합을 끌어올리기 위한 개선판.

개선 요지 (데이터 누수 없음)
---------------------------
1. **Step A**: 기본은 **Train에서만 피팅한 AP 보정 + 바운드 Huber 삼변**(`predict_step_a_calibrated`).
   추가로 **V3식 Wi‑Fi 삼변 XY**(무경계 Huber, `predict_step_a_v3_wifi_equivalent`)를
   “안심 좌표” 후보로 두되, 융합용 AP 보정 계수는 동일 폴드의 `spec`으로만 피팅(검증 라벨 미사용).
   **블렌드**: 동일 폴드에서 피팅한 보정 거리 삼변과 V3식 삼변 XY를 컨벡스 결합(`blend_alpha`).
2. **Joint UWB 선택**: Train CV Step A 순위 상위 K개 각각에 대해 UWB 게이트·inflate·fusion·IRLS
   그리드를 돌린 뒤 **Train CV Step C**(동률 시 B→A→캘리브 단순도) 최소만 채택한다.
   (Step A만 최적화하면 게이트 기준점과 맞지 않아 B/C가 흔들릴 수 있음.)
3. **KNN 잔차**: 후보에 포함 가능하나 기본은 k=0(V12 Fast2 그리드에서는 k=0만 사용).
4. **검증**: 전체 train 재학습 후 단일 forward; 정답(`True_X/Y`)은 지표·CSV용.

참고: `outputs/v12_summary.json`은 validation RMSE로 그리드를 고른 결과(선택 편향).
본 스크립트는 동일 수식 계열이나 튜닝에 검증 라벨을 쓰지 않는다.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from fusion_realtime_sanitize import resolve_train_kgh_corrected_paths, resolve_validation_paths
from indoor_fusion_pipeline_v10 import WifiOnlyTrilateration, resolve_paths
from indoor_fusion_pipeline_v11 import CalibMethodSpec, fit_robust_calibration_per_ap
from indoor_fusion_pipeline_v12 import (
    augment_train_for_canon_fit,
    irls_refine_xyz,
    point_errors,
    predict_step_a_calibrated,
    predict_step_a_v3_wifi_equivalent,
    rmse_mae,
    row_fusion_gated_arrays,
    row_wifi_calibrated_arrays,
    solve_trilat_array,
    wifi_median_impute_fusion,
)
from indoor_fusion_pipeline_v12_strict import resolve_kfold_n_splits
from indoor_fusion_pipeline_v8 import WIFI_COL_CANON, Config, FusionLocalizerV8

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ART_PREFIX = "v12_fast2"


@dataclass(frozen=True)
class StepAConfig:
    """Step A XY 산출 + 융합용 AP 보정(spec). blend_alpha 가 있으면 보정 삼변과 V3 삼변을 컨벡스 결합."""

    spec: CalibMethodSpec
    feat: str
    hf_wifi_calib: float
    knn_k: int = 0
    use_v3_xy: bool = False
    hf_wifi_v3: float = 1.35
    wifi_bias_v3: float = 2.5
    blend_alpha: Optional[float] = None  # 보정 분 가중치; None 이면 블렌드 없음
    blend_hf_v3: float = 1.35
    blend_wifi_bias_v3: float = 2.5


def _blend_calib_v3_xy(
    xy_c: np.ndarray,
    xy_v: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """두 삼변 좌표 컨벡스 결합. 한쪽만 유한하면 그쪽만 사용."""
    a = float(alpha)
    b = 1.0 - a
    out = np.full_like(xy_c, np.nan, dtype=float)
    for j in range(2):
        c = xy_c[:, j]
        v = xy_v[:, j]
        m2 = np.isfinite(c) & np.isfinite(v)
        out[m2, j] = a * c[m2] + b * v[m2]
        m0 = np.isfinite(c) & ~np.isfinite(v)
        out[m0, j] = c[m0]
        m1 = ~np.isfinite(c) & np.isfinite(v)
        out[m1, j] = v[m1]
    return out


def _spec_simplicity_rank(sp: CalibMethodSpec) -> int:
    if sp.kind == "identity":
        return 0
    if sp.kind == "huber":
        return 1
    return 2


def _predict_step_a_xy(
    df: pd.DataFrame,
    loc: FusionLocalizerV8,
    coeffs: Dict[str, Tuple[float, float]],
    cfg: StepAConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.blend_alpha is not None:
        xy_c = predict_step_a_calibrated(df, loc, coeffs, cfg.feat, float(cfg.hf_wifi_calib))
        xy_v = predict_step_a_v3_wifi_equivalent(
            df,
            loc,
            wifi_bias=float(cfg.blend_wifi_bias_v3),
            huber_f_scale=float(cfg.blend_hf_v3),
        )
        xy = _blend_calib_v3_xy(xy_c, xy_v, float(cfg.blend_alpha))
        return xy[:, 0], xy[:, 1]
    if cfg.use_v3_xy:
        xy = predict_step_a_v3_wifi_equivalent(
            df,
            loc,
            wifi_bias=float(cfg.wifi_bias_v3),
            huber_f_scale=float(cfg.hf_wifi_v3),
        )
        return xy[:, 0].astype(float), xy[:, 1].astype(float)
    xy = predict_step_a_calibrated(df, loc, coeffs, cfg.feat, float(cfg.hf_wifi_calib))
    return xy[:, 0].astype(float), xy[:, 1].astype(float)


# Step A 탐색 (Train CV만): 보정 삼변 + (동일 Train 피팅 계수로) V3식 XY 안심좌표
def iter_step_a_candidates() -> List[StepAConfig]:
    h = CalibMethodSpec("huber", huber_epsilon=1.35, huber_alpha=5e-4)
    h_lo = CalibMethodSpec("huber", huber_epsilon=1.2, huber_alpha=5e-4)
    id_sp = CalibMethodSpec("identity")
    calib_rows: List[StepAConfig] = [
        StepAConfig(id_sp, "bias_sub", 1.0, 0),
        StepAConfig(id_sp, "bias_sub", 1.15, 0),
        StepAConfig(id_sp, "bias_sub", 1.35, 0),
        StepAConfig(h_lo, "bias_sub", 1.2, 0),
        StepAConfig(h, "bias_sub", 1.0, 0),
        StepAConfig(h, "bias_sub", 1.1, 0),
        StepAConfig(h, "bias_sub", 1.2, 0),
        StepAConfig(h, "bias_sub", 1.28, 0),
        StepAConfig(h, "bias_sub", 1.32, 0),
        StepAConfig(h, "bias_sub", 1.35, 0),
        StepAConfig(CalibMethodSpec("ransac", ransac_residual_threshold=3.5), "bias_sub", 1.35, 0),
        StepAConfig(CalibMethodSpec("ransac", ransac_residual_threshold=5.5), "bias_sub", 1.35, 0),
    ]
    blend_rows: List[StepAConfig] = []
    for sp_i, hf_c in ((id_sp, 1.35), (h, 1.35), (h_lo, 1.2)):
        for alp in (0.2, 0.35, 0.5, 0.65, 0.8):
            blend_rows.append(
                StepAConfig(
                    sp_i,
                    "bias_sub",
                    float(hf_c),
                    0,
                    blend_alpha=float(alp),
                    blend_hf_v3=1.35,
                    blend_wifi_bias_v3=2.5,
                )
            )
    v3_rows: List[StepAConfig] = []
    for sp in (id_sp, h):
        for wb in (2.0, 2.5, 3.0):
            v3_rows.append(
                StepAConfig(
                    sp,
                    "bias_sub",
                    hf_wifi_calib=1.35,
                    knn_k=0,
                    use_v3_xy=True,
                    hf_wifi_v3=1.35,
                    wifi_bias_v3=float(wb),
                )
            )
    return calib_rows + blend_rows + v3_rows

# Phase B/C — v12 검증튜닝 근방 + Train CV만으로 선택 (PDW 캐시로 (gate,inflate)당 1회 구축)
# Train CV는 Step B RMSE로만 조합 선택(IRLS 제외 → 대폭 단축). 검증은 여전히 Step C까지 계산.
GATE_GRID = [1.25, 1.4, 1.55, 1.7, 1.85, 2.0]
UWB_INFLATE_GRID = [8.0, 10.0, 16.0, 24.0]
HUBER_F_FUSION_GRID = [1.0, 1.2, 1.35]
# IRLS는 Train CV에 쓰지 않고, Step B로 고른 (gate,inflate,hf_fus)에 대해 소수만 검증 전 재점검
IRLS_REFINE_GRID = [(2.2, 2), (2.75, 2), (3.1, 2)]

MAX_KFOLD = 2
# Train CV Step A만 최소화하면 게이트 기준점과 어긋나 Step B/C가 악화될 수 있어,
# Step A 상위 K개 각각에 대해 Phase B→C 전체를 돌린 뒤 **Train CV Step C**가 최소인 조합을 고른다.
TOP_STEP_A_JOINT_BY_TRAIN_CV_STEP_C = 5


def _knn_residual_refine_xy(
    xa_tr: np.ndarray,
    ya_tr: np.ndarray,
    tx_tr: np.ndarray,
    ty_tr: np.ndarray,
    xa_te: np.ndarray,
    ya_te: np.ndarray,
    knn_k: int,
) -> np.ndarray:
    """Train 좌표 기준 KNN 잔차로 검증/홀드아웃 Step A 보정. knn_k==0 이면 (xa_te, ya_te) 그대로."""
    out_x = np.asarray(xa_te, dtype=float).copy()
    out_y = np.asarray(ya_te, dtype=float).copy()
    if knn_k <= 0:
        return np.column_stack([out_x, out_y])
    m_tr = np.isfinite(xa_tr) & np.isfinite(ya_tr) & np.isfinite(tx_tr) & np.isfinite(ty_tr)
    m_te = np.isfinite(xa_te) & np.isfinite(ya_te)
    n_valid = int(np.sum(m_tr))
    k_eff = min(int(knn_k), max(2, n_valid - 1))
    if n_valid < k_eff + 2 or int(np.sum(m_te)) == 0:
        return np.column_stack([out_x, out_y])
    X = np.column_stack([xa_tr[m_tr], ya_tr[m_tr]])
    dx = tx_tr[m_tr] - xa_tr[m_tr]
    dy = ty_tr[m_tr] - ya_tr[m_tr]
    knn = KNeighborsRegressor(n_neighbors=k_eff, weights="distance", metric="euclidean")
    knn.fit(X, np.column_stack([dx, dy]))
    idx_te = np.flatnonzero(m_te)
    Xe = np.column_stack([xa_te[idx_te], ya_te[idx_te]])
    row_ok = np.isfinite(Xe).all(axis=1)
    if np.any(row_ok):
        pred = knn.predict(Xe[row_ok])
        jj = idx_te[row_ok]
        out_x[jj] = out_x[jj] + pred[:, 0]
        out_y[jj] = out_y[jj] + pred[:, 1]
    return np.column_stack([out_x, out_y])


def mean_cv_rmse_step_a_bundle(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    cfg: StepAConfig,
    n_splits: int,
) -> float:
    """Step A (+선택적 KNN 잔차)의 Train K-Fold 평균 RMSE."""
    n_splits = min(n_splits, len(train_imp))
    if n_splits < 2:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses: List[float] = []
    for tr_idx, te_idx in kf.split(np.arange(len(train_imp))):
        tr_df = train_imp.iloc[tr_idx].reset_index(drop=True)
        te_df = train_imp.iloc[te_idx].reset_index(drop=True)
        tr_fit = augment_train_for_canon_fit(tr_df, loc)
        coeffs = fit_robust_calibration_per_ap(tr_fit, tri_ref, feat_mode=cfg.feat, spec=cfg.spec)
        xa_tr, ya_tr = _predict_step_a_xy(tr_df, loc, coeffs, cfg)
        xa_te, ya_te = _predict_step_a_xy(te_df, loc, coeffs, cfg)
        tx_tr = tr_df["True_X"].values.astype(float)
        ty_tr = tr_df["True_Y"].values.astype(float)
        tx_te = te_df["True_X"].values.astype(float)
        ty_te = te_df["True_Y"].values.astype(float)
        xy_ref = _knn_residual_refine_xy(xa_tr, ya_tr, tx_tr, ty_tr, xa_te, ya_te, int(cfg.knn_k))
        err = point_errors(tx_te, ty_te, xy_ref[:, 0], xy_ref[:, 1])
        r, _ = rmse_mae(err)
        rmses.append(float(r))
    return float(np.mean(rmses))


def build_fold_cache_single_rank(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    cfg: StepAConfig,
    n_splits: int,
) -> List[Dict[str, Any]]:
    """단일 StepAConfig에 대해 폴드별 coeffs + 잔차 보정된 Step A 좌표."""
    n_splits = min(n_splits, len(train_imp))
    if n_splits < 2:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_cache: List[Dict[str, Any]] = []
    for tr_idx, te_idx in kf.split(np.arange(len(train_imp))):
        tr_df = train_imp.iloc[tr_idx].reset_index(drop=True)
        te_df = train_imp.iloc[te_idx].reset_index(drop=True)
        tr_fit = augment_train_for_canon_fit(tr_df, loc)
        coeffs = fit_robust_calibration_per_ap(tr_fit, tri_ref, feat_mode=cfg.feat, spec=cfg.spec)
        xa_tr, ya_tr = _predict_step_a_xy(tr_df, loc, coeffs, cfg)
        xa_te, ya_te = _predict_step_a_xy(te_df, loc, coeffs, cfg)
        tx_tr = tr_df["True_X"].values.astype(float)
        ty_tr = tr_df["True_Y"].values.astype(float)
        xy_ref = _knn_residual_refine_xy(xa_tr, ya_tr, tx_tr, ty_tr, xa_te, ya_te, int(cfg.knn_k))
        per_rank = [dict(coeffs=coeffs, xy_a=xy_ref, feat=cfg.feat, hf_wifi=float(cfg.hf_wifi_calib))]
        fold_cache.append(dict(te_df=te_df, per_rank=per_rank))
    return fold_cache


def rmse_step_b_one_fold(
    te_df: pd.DataFrame,
    pdw_store: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    hf_fus: float,
) -> float:
    """Train CV용: IRLS 없이 융합 Huber 삼변만(속도)."""
    preds_b = np.full((len(te_df), 2), np.nan, dtype=float)
    for k, (p, d, w) in enumerate(pdw_store):
        preds_b[k] = solve_trilat_array(p, d, w, huber_f_scale=hf_fus)
    tx = te_df["True_X"].values.astype(float)
    ty = te_df["True_Y"].values.astype(float)
    eb = point_errors(tx, ty, preds_b[:, 0], preds_b[:, 1])
    r, _ = rmse_mae(eb)
    return float(r)


def mean_cv_step_b_from_pdws(
    fold_pdws: List[Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Any, str, float]],
    hf_fus: float,
) -> float:
    return float(np.mean([rmse_step_b_one_fold(te_df, pdw, hf_fus) for te_df, pdw, _, _, _ in fold_pdws]))


def rmse_step_c_one_fold(
    te_df: pd.DataFrame,
    pdw_store: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    loc: FusionLocalizerV8,
    coeffs: Any,
    feat: str,
    hf_wifi: float,
    hf_fus: float,
    ith: float,
    mx: int,
) -> float:
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
    r, _ = rmse_mae(ec)
    return float(r)


def build_fold_pdws_gate_inflate(
    fold_cache: List[Dict[str, Any]],
    loc: FusionLocalizerV8,
    gate: float,
    uinf: float,
) -> List[Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Any, str, float]]:
    """(gate, uinf) 고정 시 폴드별 PDW 1회."""
    fold_pdws: List[Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Any, str, float]] = []
    for fc in fold_cache:
        te_df: pd.DataFrame = fc["te_df"]
        blk = fc["per_rank"][0]
        coeffs = blk["coeffs"]
        xy_a = blk["xy_a"]
        feat = blk["feat"]
        hf_wifi = float(blk["hf_wifi"])
        pdw_store: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for k, (_, row) in enumerate(te_df.iterrows()):
            p, d, w = row_fusion_gated_arrays(row, loc, xy_a[k], gate, coeffs, feat, uinf)
            pdw_store.append((p, d, w))
        fold_pdws.append((te_df, pdw_store, coeffs, feat, hf_wifi))
    return fold_pdws


def mean_cv_step_c_from_pdws(
    fold_pdws: List[Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Any, str, float]],
    loc: FusionLocalizerV8,
    hf_fus: float,
    ith: float,
    mx: int,
) -> float:
    rmses = [
        rmse_step_c_one_fold(te_df, pdw, loc, coeffs, feat, hf_wifi, hf_fus, ith, mx)
        for te_df, pdw, coeffs, feat, hf_wifi in fold_pdws
    ]
    return float(np.mean(rmses))


def main() -> int:
    t0 = time.perf_counter()
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
    n_splits = min(MAX_KFOLD, resolve_kfold_n_splits(len(train_df)))
    print(f"\n[V12 Fast2] 학습 {len(train_df)}행 | 검증 {len(val_df)}행 | K-Fold={n_splits} (무결성)", flush=True)

    train_imp, val_imp = wifi_median_impute_fusion(train_df, val_df, loc.wifi_cols)
    tri_ref = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m=0.6, huber_f_scale=1.0)

    # --- 1) Train CV: Step A (보정 삼변 ± V3식 XY) --------------------------------
    print("[V12 Fast2] Train CV — Step A (보정 Wi‑Fi 삼변 / V3 XY 후보) …", flush=True)
    stage_a_records: List[Dict[str, Any]] = []
    step_a_cands = iter_step_a_candidates()
    cv_scores: List[float] = []
    for i, acfg in enumerate(step_a_cands):
        sc = mean_cv_rmse_step_a_bundle(train_imp, loc, tri_ref, acfg, n_splits)
        cv_scores.append(float(sc))
        stage_a_records.append(
            dict(
                candidate_index=int(i),
                calib_kind=acfg.spec.kind,
                feat_mode=acfg.feat,
                huber_f_wifi_calib=float(acfg.hf_wifi_calib),
                use_v3_xy=bool(acfg.use_v3_xy),
                hf_wifi_v3=float(acfg.hf_wifi_v3),
                wifi_bias_v3=float(acfg.wifi_bias_v3),
                blend_calib_alpha=acfg.blend_alpha,
                blend_hf_v3=float(acfg.blend_hf_v3),
                blend_wifi_bias_v3=float(acfg.blend_wifi_bias_v3),
                knn_residual_neighbors=int(acfg.knn_k),
                train_CV_mean_RMSE_step_A_m=float(sc),
            )
        )
        print(f"    … Step A 후보 {i + 1}/{len(step_a_cands)} CV_RMSE={sc:.4f}", flush=True)

    best_pool_cv = min(cv_scores)
    order_by_step_a = sorted(
        range(len(cv_scores)),
        key=lambda ii: (float(cv_scores[ii]), _spec_simplicity_rank(step_a_cands[ii].spec)),
    )
    k_joint = min(TOP_STEP_A_JOINT_BY_TRAIN_CV_STEP_C, len(order_by_step_a))
    top_a_idx = order_by_step_a[:k_joint]

    # --- 2) Train CV: Phase B/C × Step A 상위 K (목표: Train CV Step C 최소) -----
    print(
        f"[V12 Fast2] Train CV — Phase B/C joint with top-{k_joint} Step-A by train CV …",
        flush=True,
    )
    best_joint_key: Optional[Tuple[float, float, float, int]] = None
    best_post: Optional[Dict[str, Any]] = None
    n_gate_inf = len(GATE_GRID) * len(UWB_INFLATE_GRID)
    n_inner = len(HUBER_F_FUSION_GRID)
    n_combo = n_gate_inf * n_inner

    for rank_j, ai in enumerate(top_a_idx):
        acfg = step_a_cands[ai]
        fold_cache = build_fold_cache_single_rank(train_imp, loc, tri_ref, acfg, n_splits)
        stage_bc_local: List[Dict[str, Any]] = []
        done = 0
        best_cv_b = float("inf")
        best_bc_b: Optional[Dict[str, Any]] = None
        for gate in GATE_GRID:
            for uinf in UWB_INFLATE_GRID:
                fold_pdws = build_fold_pdws_gate_inflate(fold_cache, loc, float(gate), float(uinf))
                for hf_fus in HUBER_F_FUSION_GRID:
                    cv_b = mean_cv_step_b_from_pdws(fold_pdws, float(hf_fus))
                    done += 1
                    stage_bc_local.append(
                        dict(
                            step_A_candidate_index=int(ai),
                            gate_m=float(gate),
                            uwb_inflate=float(uinf),
                            huber_f_fusion=float(hf_fus),
                            train_CV_mean_RMSE_step_B_m=float(cv_b),
                            selection_objective="min_train_CV_RMSE_step_B_no_IRLS",
                        )
                    )
                    if cv_b < best_cv_b - 1e-14:
                        best_cv_b = float(cv_b)
                        best_bc_b = dict(gate=float(gate), uinf=float(uinf), hf_fus=float(hf_fus))
        if best_bc_b is None:
            raise RuntimeError("V12 Fast2: Phase B 탐색 실패")

        fold_pdws_irls = build_fold_pdws_gate_inflate(
            fold_cache, loc, best_bc_b["gate"], best_bc_b["uinf"]
        )
        best_cv_c = float("inf")
        best_bc: Optional[Dict[str, Any]] = None
        for ith, mx in IRLS_REFINE_GRID:
            cv_c = mean_cv_step_c_from_pdws(
                fold_pdws_irls, loc, best_bc_b["hf_fus"], float(ith), int(mx)
            )
            stage_bc_local.append(
                dict(
                    step_A_candidate_index=int(ai),
                    gate_m=float(best_bc_b["gate"]),
                    uwb_inflate=float(best_bc_b["uinf"]),
                    huber_f_fusion=float(best_bc_b["hf_fus"]),
                    irls_thresh=float(ith),
                    irls_max_iter=int(mx),
                    train_CV_mean_RMSE_step_C_m=float(cv_c),
                    selection_objective="min_train_CV_RMSE_step_C_given_best_B",
                )
            )
            if cv_c < best_cv_c - 1e-14:
                best_cv_c = float(cv_c)
                best_bc = dict(
                    gate=float(best_bc_b["gate"]),
                    uinf=float(best_bc_b["uinf"]),
                    hf_fus=float(best_bc_b["hf_fus"]),
                    irls_thresh=float(ith),
                    irls_mx=int(mx),
                )
        if best_bc is None:
            raise RuntimeError("V12 Fast2: Phase B/C 탐색 실패")

        jkey = (
            float(best_cv_c),
            float(best_cv_b),
            float(cv_scores[ai]),
            _spec_simplicity_rank(acfg.spec),
        )
        print(
            f"    … joint {rank_j + 1}/{k_joint}  StepA_idx={ai}  "
            f"train_CV_A={cv_scores[ai]:.4f}  B={best_cv_b:.4f}  C={best_cv_c:.4f}",
            flush=True,
        )
        if best_joint_key is None or jkey < best_joint_key:
            best_joint_key = jkey
            best_post = dict(
                ai=int(ai),
                acfg=acfg,
                best_bc=best_bc,
                best_cv_b=float(best_cv_b),
                best_cv_c=float(best_cv_c),
                stage_bc=stage_bc_local,
            )

    if best_post is None:
        raise RuntimeError("V12 Fast2: joint Phase B/C 실패")

    best_idx = int(best_post["ai"])
    cfg_a: StepAConfig = best_post["acfg"]
    best_a = float(cv_scores[best_idx])
    best_cv_b = float(best_post["best_cv_b"])
    best_cv_c = float(best_post["best_cv_c"])
    best_bc = best_post["best_bc"]
    stage_bc: List[Dict[str, Any]] = list(best_post["stage_bc"])

    train_fit_full0 = augment_train_for_canon_fit(train_imp, loc)
    tx_all = train_imp["True_X"].values.astype(float)
    ty_all = train_imp["True_Y"].values.astype(float)
    coeffs_resub = fit_robust_calibration_per_ap(
        train_fit_full0, tri_ref, feat_mode=cfg_a.feat, spec=cfg_a.spec
    )
    xa0, ya0 = _predict_step_a_xy(train_imp, loc, coeffs_resub, cfg_a)
    xy0 = _knn_residual_refine_xy(xa0, ya0, tx_all, ty_all, xa0, ya0, int(cfg_a.knn_k))
    er0 = point_errors(tx_all, ty_all, xy0[:, 0], xy0[:, 1])
    best_resub, _ = rmse_mae(er0)
    best_resub = float(best_resub)
    stage_a_records[best_idx]["train_resub_RMSE_step_A_m"] = best_resub

    # --- 3) 전체 train 재피팅 → 검증 단일 Forward -------------------------------
    train_fit = augment_train_for_canon_fit(train_imp, loc)
    coeffs_f = fit_robust_calibration_per_ap(train_fit, tri_ref, feat_mode=cfg_a.feat, spec=cfg_a.spec)
    xa_tr_f, ya_tr_f = _predict_step_a_xy(train_imp, loc, coeffs_f, cfg_a)
    tx_tr_f = train_imp["True_X"].values.astype(float)
    ty_tr_f = train_imp["True_Y"].values.astype(float)
    xa_val, ya_val = _predict_step_a_xy(val_imp, loc, coeffs_f, cfg_a)
    xy_a_val = _knn_residual_refine_xy(
        xa_tr_f, ya_tr_f, tx_tr_f, ty_tr_f, xa_val, ya_val, int(cfg_a.knn_k)
    )

    pdw_store: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for k, (_, row) in enumerate(val_imp.iterrows()):
        p, d, w = row_fusion_gated_arrays(
            row,
            loc,
            xy_a_val[k],
            best_bc["gate"],
            coeffs_f,
            cfg_a.feat,
            best_bc["uinf"],
        )
        pdw_store.append((p, d, w))

    preds_b = np.full((len(val_imp), 2), np.nan, dtype=float)
    for k, (p, d, w) in enumerate(pdw_store):
        preds_b[k] = solve_trilat_array(p, d, w, huber_f_scale=best_bc["hf_fus"])

    preds_c = np.full((len(val_imp), 2), np.nan, dtype=float)
    for k, (p, d, w) in enumerate(pdw_store):
        if len(d) < 3:
            p2, d2, w2 = row_wifi_calibrated_arrays(val_imp.iloc[k], loc, coeffs_f, cfg_a.feat)
            preds_c[k] = solve_trilat_array(p2, d2, w2, huber_f_scale=float(cfg_a.hf_wifi_calib))
        else:
            preds_c[k] = irls_refine_xyz(
                p, d, w, best_bc["hf_fus"], best_bc["irls_thresh"], best_bc["irls_mx"]
            )

    tx = val_imp["True_X"].values.astype(float)
    ty = val_imp["True_Y"].values.astype(float)
    ea = point_errors(tx, ty, xy_a_val[:, 0], xy_a_val[:, 1])
    eb = point_errors(tx, ty, preds_b[:, 0], preds_b[:, 1])
    ec = point_errors(tx, ty, preds_c[:, 0], preds_c[:, 1])
    rmse_a, mae_a = rmse_mae(ea)
    rmse_b, mae_b = rmse_mae(eb)
    rmse_c, mae_c = rmse_mae(ec)
    elapsed = time.perf_counter() - t0

    pref = resolve_paths(root)
    summ = {
        "pipeline": "V12 Fast2",
        "reference_cheating_baseline": (
            "outputs/v12_summary.json tuned hyperparameters on validation labels (biased). "
            "Fast2 never uses validation True_X/Y for selection."
        ),
        "integrity": "Step A bundle + Phase B/C selected only by train K-fold CV; validation used once.",
        "runtime_seconds": round(float(elapsed), 3),
        "train_kfold_splits": int(n_splits),
        "train_cv_step_A_mean_RMSE_m": float(best_a),
        "train_cv_step_A_pool_min_RMSE_m": float(best_pool_cv),
        "step_A_phase_BC_joint_selection": (
            f"Among top-{k_joint} Step-A candidates by (train_CV_step_A_RMSE, calib_simplicity), "
            "each runs full Phase B grid then IRLS subgrid; pick lexicographic minimum "
            "(train_CV_step_C_RMSE, train_CV_step_B_RMSE, train_CV_step_A_RMSE, calib_simplicity). "
            "Validation labels never used for selection."
        ),
        "step_A_top_indices_by_train_CV_A": [int(x) for x in top_a_idx],
        "train_resub_step_A_RMSE_m_chosen_candidate": float(best_resub),
        "train_cv_step_B_mean_RMSE_m": float(best_cv_b),
        "train_cv_step_C_mean_RMSE_m": float(best_cv_c),
        "validation_metrics": {
            "step_A_RMSE_m": float(rmse_a),
            "step_A_MAE_m": float(mae_a),
            "step_B_RMSE_m": float(rmse_b),
            "step_B_MAE_m": float(mae_b),
            "step_C_RMSE_m": float(rmse_c),
            "step_C_MAE_m": float(mae_c),
            "delta_RMSE_A_to_B_m": float(rmse_a - rmse_b),
            "delta_RMSE_B_to_C_m": float(rmse_b - rmse_c),
        },
        "selected": {
            "wifi_calibration_kind": cfg_a.spec.kind,
            "feat_mode": cfg_a.feat,
            "huber_f_wifi_calibrated_trilat": float(cfg_a.hf_wifi_calib),
            "step_A_use_v3_xy": bool(cfg_a.use_v3_xy),
            "step_A_hf_wifi_v3": float(cfg_a.hf_wifi_v3),
            "step_A_wifi_bias_v3": float(cfg_a.wifi_bias_v3),
            "step_A_blend_calib_alpha": cfg_a.blend_alpha,
            "step_A_blend_hf_v3": float(cfg_a.blend_hf_v3),
            "step_A_blend_wifi_bias_v3": float(cfg_a.blend_wifi_bias_v3),
            "knn_residual_neighbors_train_fit": int(cfg_a.knn_k),
            "gate_m": best_bc["gate"],
            "uwb_variance_inflate": best_bc["uinf"],
            "huber_f_fusion": best_bc["hf_fus"],
            "irls_thresh_m": best_bc["irls_thresh"],
            "irls_max_iter": best_bc["irls_mx"],
        },
        "grids_searched": {
            "step_a_candidates": "iter_step_a_candidates()",
            "gate_m": GATE_GRID,
            "uwb_inflate": UWB_INFLATE_GRID,
            "huber_f_fusion": HUBER_F_FUSION_GRID,
            "irls_refine_grid": [[float(a), int(b)] for a, b in IRLS_REFINE_GRID],
        },
        "v12_validation_tuned_reference_from_outputs_v12_summary": {
            "note": "검증 라벨로 선택된 하이퍼 — 편향된 지표",
            "step_A_RMSE_m": 1.5716031362024965,
            "step_B_RMSE_m": 1.4466804779563518,
            "step_C_RMSE_m": 1.4453084716212174,
        },
        "data_paths": {
            "train_median": str(pref.train_median),
            "train_variance": str(pref.train_variance),
            "val_median": str(pref.val_median),
            "val_variance": str(pref.val_variance),
        },
    }
    (out_dir / f"{ART_PREFIX}_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(stage_a_records).to_csv(out_dir / f"{ART_PREFIX}_grid_stepA.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stage_bc).to_csv(out_dir / f"{ART_PREFIX}_grid_phaseBC.csv", index=False, encoding="utf-8-sig")

    pred = val_imp[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xy_a_val[:, 0]
    pred["StepA_Y"] = xy_a_val[:, 1]
    pred["StepB_X"] = preds_b[:, 0]
    pred["StepB_Y"] = preds_b[:, 1]
    pred["StepC_X"] = preds_c[:, 0]
    pred["StepC_Y"] = preds_c[:, 1]
    pred["ErrA_m"] = ea
    pred["ErrB_m"] = eb
    pred["ErrC_m"] = ec
    pred.to_csv(out_dir / f"{ART_PREFIX}_predictions.csv", index=False, encoding="utf-8-sig")
    rows_coef = [{"wifi_canon_ap": k, "A": v[0], "B": v[1]} for k, v in coeffs_f.items()]
    pd.DataFrame(rows_coef).to_csv(out_dir / f"{ART_PREFIX}_wifi_calibration_coefs.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print(f"V12 Fast2 — 완료 {elapsed:.1f}s | Train CV만 사용 | 검증 단일 평가")
    print("=" * 72)
    print(f"  [Train CV] Step A 평균 RMSE (선택 기준) = {best_a:.4f} m")
    print(f"  [Train CV] Step B 평균 RMSE (1차 선택) = {best_cv_b:.4f} m")
    print(f"  [Train CV] Step C 평균 RMSE (IRLS 소그리드) = {best_cv_c:.4f} m")
    print(f"  [Validation] Step A  RMSE={rmse_a:.4f} m  MAE={mae_a:.4f} m")
    print(f"  [Validation] Step B  RMSE={rmse_b:.4f} m  MAE={mae_b:.4f} m")
    print(f"  [Validation] Step C  RMSE={rmse_c:.4f} m  MAE={mae_c:.4f} m")
    print("=" * 72)
    v3tag = (
        f", v3_xy=1 hf_v3={cfg_a.hf_wifi_v3} wb={cfg_a.wifi_bias_v3}"
        if cfg_a.use_v3_xy
        else (
            f", blend_alpha={cfg_a.blend_alpha} (calib+V3)"
            if cfg_a.blend_alpha is not None
            else ", v3_xy=0 blend=—"
        )
    )
    print(
        f"[선택] calib={cfg_a.spec.kind}, hf_wifi_calib={cfg_a.hf_wifi_calib}{v3tag}, "
        f"KNN_k={cfg_a.knn_k}, gate={best_bc['gate']:.3f}, inflate={best_bc['uinf']}, "
        f"hf_fus={best_bc['hf_fus']}, IRLS={best_bc['irls_thresh']:.2f}"
    )
    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main, output_artifact_include_prefixes=("v12_fast2_",))
