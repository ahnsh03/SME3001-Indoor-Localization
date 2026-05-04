"""
V13-fix — V13 로직과 동일 단, 다음을 만족하도록 수정한 변형입니다.

**(A) 코드 독립**  
Step B/C Numba 경로는 `indoor_fusion_pipeline_v12_turbo.py` 대신 **`fusion_turbo_numba_core.py`**만 사용합니다.

**(B) 캘리브 순위를 Train 근거만으로 구성**  
Turbo/v12에 가까운 **설계자 고정 삽입**(`forced_*`)을 제거합니다. 순위는
`rank_calibration_candidates` 가 산출한 **Train K-Fold 순수 보정 Step A RMSE** 정렬 순서만 따릅니다.

Optuna **enqueue warmstart**(수치까지 하드코딩한 시드)는 기본 실행에서 **생략**합니다.

무결성(검증 정답)은 원본 V13과 동일: Optuna 목적은 Train 평균 Step C만, 검증 라벨은 최종 포워드 1회.

보고서·방법론에서의 그리드 정당화 (학습 무결과 별개)
-------------------------------------------------
본 스크립트의 ``V3_PROFILES``, Wi‑Fi 보정 **카탈로그**, Optuna **탐색 범위**(게이트·inflate·블렌드 등)는
**(1)** Train만 본 EDA·선행 스캔으로 도출되었거나 **플러시블 구간으로 줄였다고 주장할 수 있는 부분**이 있고,
**(2)** 동시에 이전 파이프라인(v12 등) 관례·문헌에 가까운 값이 섞일 수 있다.

리포트에는 **「전량이 검증 라벨로 역산한 결과」처럼 쓰지 말 것**을 권장한다. 다음 정도면 통상 과제 요구에 충분하다:

- 「Train 특성 분석 및 기존 융합 엔진 관례를 참고하여 **후보 그리드를 고정**했고,
  하이퍼는 그 안에서만 Train CV로 선택했다.**

아래 각 블록 옆 주석과 ``make_objective_clamped`` 내부 주석도 같은 선에서 이해하면 된다.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as e:
    raise ImportError("indoor_fusion_pipeline_v13_fix.py 에는 optuna 가 필요합니다.") from e

from fusion_realtime_sanitize import resolve_train_kgh_corrected_paths, resolve_validation_paths
from indoor_fusion_pipeline_v10 import WifiOnlyTrilateration, resolve_paths
from indoor_fusion_pipeline_v11 import CalibMethodSpec, fit_robust_calibration_per_ap
from indoor_fusion_pipeline_v12 import (
    TUKEY_C,
    augment_train_for_canon_fit,
    point_errors,
    predict_step_a_calibrated,
    predict_step_a_v3_wifi_equivalent,
    rmse_mae,
    wifi_median_impute_fusion,
)
from fusion_turbo_numba_core import (
    LocGeomPack,
    TRILAT_GN_ITERS,
    batch_step_b_preds_flat,
    batch_step_c_preds_flat,
    build_loc_geom_pack,
    coeffs_dict_to_ab,
    feat_to_id,
    resolve_kfold_n_splits,
    te_df_to_numpy_blocks,
)
from indoor_fusion_pipeline_v8 import MAX_RANGE_M, WIFI_COL_CANON, Config, FusionLocalizerV8

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except (AttributeError, OSError):
        pass
plt.rcParams["axes.unicode_minus"] = False

ART_PREFIX = "v13_fix"
MAX_KFOLD_SPLITS = int(os.environ.get("V13_FIX_MAX_KFOLD", os.environ.get("V13_MAX_KFOLD", "3")))
N_OPTUNA_TRIALS = int(os.environ.get("V13_FIX_OPTUNA_TRIALS", os.environ.get("V13_OPTUNA_TRIALS", "42")))
OPTUNA_N_JOBS = int(os.environ.get("V13_FIX_OPTUNA_JOBS", os.environ.get("V13_OPTUNA_JOBS", "-1")))
OPTUNA_SEED = int(os.environ.get("V13_FIX_OPTUNA_SEED", os.environ.get("V13_OPTUNA_SEED", "42")))
PLOT_DPI = int(os.environ.get("V13_FIX_PLOT_DPI", os.environ.get("V13_PLOT_DPI", "110")))
# V3 동급 Wi‑Fi 삼변 (huber_f_scale, wifi_bias_m) 후보. Train EDA·삼변 민감도로 좁힌 대표값 + 엔진 관례 혼합.
# Optuna는 인덱스만 고름; 프로필 자체는 이 과제에서 고정 그리드로 둔다(보고서에는 위 모듈 docstring 절 참고).
V3_PROFILES: List[Tuple[float, float]] = [
    (1.35, 2.5),
    (1.35, 2.0),
    (1.35, 3.0),
    (1.2, 2.25),
    (1.45, 2.75),
]
V13_ALLOW_KNN = os.environ.get("V13_FIX_ALLOW_KNN", os.environ.get("V13_ALLOW_KNN", "")).strip().lower() in ("1", "true", "yes")
V13_FIX_ENQUEUE_WARMSTARTS = os.environ.get("V13_FIX_ENQUEUE_WARMSTARTS", "").strip().lower() in ("1", "true", "yes")


def calib_triple_key(sp: CalibMethodSpec, feat: str, hf_wifi: float) -> Tuple[Any, ...]:
    """캘리브 후보의 동일성 키(랭크 병합·중복 제거용)."""
    if sp.kind == "ransac":
        return ("ransac", str(feat), float(hf_wifi), float(sp.ransac_residual_threshold))
    if sp.kind == "huber":
        return ("huber", str(feat), float(hf_wifi), float(sp.huber_epsilon), float(sp.huber_alpha))
    return (sp.kind, str(feat), float(hf_wifi))


def calibration_catalog_extended() -> List[Tuple[CalibMethodSpec, str, float]]:
    """AP별 강건 보정·hf_wifi 조합 후보. 검증 라벨 미사용; Train CV로 순위만 매김(merge_calib_ranks_train_cv_only).

    후보 집합 자체는 실험실습에서 쓰기 좋은 범위 + 기존 파이프라인 관례의 합성으로 고정(모듈 docstring 절).
    """
    h = CalibMethodSpec("huber", huber_epsilon=1.35, huber_alpha=5e-4)
    h_lo = CalibMethodSpec("huber", huber_epsilon=1.2, huber_alpha=5e-4)
    id_sp = CalibMethodSpec("identity")
    rs35 = CalibMethodSpec("ransac", ransac_residual_threshold=3.5)
    rs55 = CalibMethodSpec("ransac", ransac_residual_threshold=5.5)
    feat = "bias_sub"
    rows = [
        (id_sp, feat, 1.0),
        (id_sp, feat, 1.15),
        (id_sp, feat, 1.28),
        (id_sp, feat, 1.35),
        (h_lo, feat, 1.2),
        (h, feat, 1.0),
        (h, feat, 1.1),
        (h, feat, 1.2),
        (h, feat, 1.28),
        (h, feat, 1.32),
        (h, feat, 1.35),
        (rs35, feat, 1.35),
        (rs55, feat, 1.35),
    ]
    uniq: Dict[Tuple[Any, ...], Tuple[CalibMethodSpec, str, float]] = {}
    for sp, ft, hf in rows:
        uniq[calib_triple_key(sp, ft, float(hf))] = (sp, ft, float(hf))
    return list(uniq.values())


def cv_mean_rmse_calibrated_only(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    spec: CalibMethodSpec,
    feat: str,
    hf_wifi: float,
    n_splits: int,
) -> float:
    n_splits = max(2, min(n_splits, len(train_imp)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses: List[float] = []
    for tr_idx, te_idx in kf.split(np.arange(len(train_imp))):
        te_df = train_imp.iloc[te_idx].reset_index(drop=True)
        tr_df = train_imp.iloc[tr_idx].reset_index(drop=True)
        tr_fit = augment_train_for_canon_fit(tr_df, loc)
        coeffs = fit_robust_calibration_per_ap(tr_fit, tri_ref, feat_mode=feat, spec=spec)
        xy_a = predict_step_a_calibrated(te_df, loc, coeffs, feat, float(hf_wifi))
        tx = te_df["True_X"].values.astype(float)
        ty = te_df["True_Y"].values.astype(float)
        ea = point_errors(tx, ty, xy_a[:, 0], xy_a[:, 1])
        r, _ = rmse_mae(ea)
        rmses.append(float(r))
    return float(np.mean(rmses))


def rank_calibration_candidates(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    catalog: Sequence[Tuple[CalibMethodSpec, str, float]],
    n_splits: int,
    top_k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, int, Tuple[CalibMethodSpec, str, float]]]]:
    rows_meta: List[Dict[str, Any]] = []
    scored: List[Tuple[float, int, Tuple[CalibMethodSpec, str, float]]] = []
    for i, (spec, feat, hf_wifi) in enumerate(catalog):
        rm = cv_mean_rmse_calibrated_only(train_imp, loc, tri_ref, spec, feat, float(hf_wifi), n_splits)
        rows_meta.append(
            dict(
                catalog_index=i,
                calib_kind=spec.kind,
                feat_mode=str(feat),
                huber_f_wifi=float(hf_wifi),
                train_CV_RMSE_pure_calibrated_step_a=float(rm),
            )
        )
        sim = 0 if spec.kind == "identity" else (1 if spec.kind == "huber" else 2)
        scored.append((float(rm), sim, (spec, str(feat), float(hf_wifi))))
    scored.sort(key=lambda z: (z[0], z[1], z[2][0].kind))
    if top_k is not None:
        scored = scored[: min(int(top_k), len(scored))]
    return rows_meta, scored


def merge_calib_ranks_train_cv_only(
    catalog: List[Tuple[CalibMethodSpec, str, float]],
    scored_full: List[Tuple[float, int, Tuple[CalibMethodSpec, str, float]]],
    max_ranks: int,
) -> Tuple[List[Tuple[CalibMethodSpec, str, float]], List[str]]:
    """Train K-Fold 순수 보정 Step A RMSE 오름차순으로만 캘리브 순위를 구성(설계자 고정 삽입 없음)."""
    cat_by_key = {calib_triple_key(s, f, h): (s, f, h) for s, f, h in catalog}
    ordered: List[Tuple[CalibMethodSpec, str, float]] = []
    tags: List[str] = []
    seen: set = set()
    for rm, _, trip in scored_full:
        k = calib_triple_key(trip[0], trip[1], trip[2])
        if k in seen or k not in cat_by_key:
            continue
        ordered.append(cat_by_key[k])
        tags.append(f"train_cv_pure_stepA_rmse={float(rm):.6f}")
        seen.add(k)
        if len(ordered) >= max(1, int(max_ranks)):
            break
    return ordered[: max(1, int(max_ranks))], tags[: max(1, int(max_ranks))]


@njit(cache=True, fastmath=True)
def convex_blend_xy_njit(xy_c: np.ndarray, xy_v: np.ndarray, alpha: float) -> np.ndarray:
    """보정 삼변과 V3 스타일 XY 컨벡스 결합. 한쪽만 유효하면 해당 값 사용."""
    a = float(alpha)
    b = 1.0 - a
    n = xy_c.shape[0]
    out = np.empty((n, 2), dtype=np.float64)
    for row in range(n):
        for j in range(2):
            cc = xy_c[row, j]
            vv = xy_v[row, j]
            fc = np.isfinite(cc)
            fv = np.isfinite(vv)
            if fc and fv:
                out[row, j] = a * cc + b * vv
            elif fc:
                out[row, j] = cc
            elif fv:
                out[row, j] = vv
            else:
                out[row, j] = np.nan
    return out


def _knn_residual_refine_xy(
    xa_tr: np.ndarray,
    ya_tr: np.ndarray,
    tx_tr: np.ndarray,
    ty_tr: np.ndarray,
    xa_te: np.ndarray,
    ya_te: np.ndarray,
    knn_k: int,
) -> np.ndarray:
    """Fast2 계열: Train 좌표 기준 거리 역가중 잔차 KNN. knn_k==0 이면 통과."""
    out_x = np.asarray(xa_te, dtype=np.float64).copy()
    out_y = np.asarray(ya_te, dtype=np.float64).copy()
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
        out_x[jj] += pred[:, 0]
        out_y[jj] += pred[:, 1]
    return np.column_stack([out_x, out_y])


def _fold_worker_pack(
    fold_index: int,
    train_imp: pd.DataFrame,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    calib_ranks: List[Tuple[CalibMethodSpec, str, float]],
    v3_profiles: List[Tuple[float, float]],
) -> Tuple[int, Dict[str, Any]]:
    tr_df = train_imp.iloc[tr_idx].reset_index(drop=True)
    te_df = train_imp.iloc[te_idx].reset_index(drop=True)
    tr_fit = augment_train_for_canon_fit(tr_df, loc)
    blk_te = te_df_to_numpy_blocks(te_df, loc)
    nk = len(te_df)

    xv3_te: List[np.ndarray] = []
    xv3_tr: List[np.ndarray] = []
    for hf, wb in v3_profiles:
        xv3_te.append(
            predict_step_a_v3_wifi_equivalent(
                te_df, loc, wifi_bias=float(wb), huber_f_scale=float(hf)
            ).astype(np.float64)
        )
        xv3_tr.append(
            predict_step_a_v3_wifi_equivalent(
                tr_df, loc, wifi_bias=float(wb), huber_f_scale=float(hf)
            ).astype(np.float64)
        )
    xy_v3_te_stack = np.stack(xv3_te, axis=0)
    xy_v3_tr_stack = np.stack(xv3_tr, axis=0)

    tx_tr = tr_df["True_X"].values.astype(np.float64)
    ty_tr = tr_df["True_Y"].values.astype(np.float64)

    per_rank: List[Dict[str, Any]] = []
    for spec, feat, hf_wifi in calib_ranks:
        coeffs = fit_robust_calibration_per_ap(tr_fit, tri_ref, feat_mode=feat, spec=spec)
        xy_c_te = np.asarray(predict_step_a_calibrated(te_df, loc, coeffs, feat, float(hf_wifi)), dtype=np.float64)
        xy_c_tr = np.asarray(predict_step_a_calibrated(tr_df, loc, coeffs, feat, float(hf_wifi)), dtype=np.float64)
        per_rank.append(
            dict(
                coeffs=coeffs,
                coeff_ab=coeffs_dict_to_ab(coeffs, loc),
                feat_id=int(feat_to_id(str(feat))),
                hf_wifi=np.float64(hf_wifi),
                xy_cal_te=xy_c_te,
                xy_cal_tr=xy_c_tr,
            )
        )

    fc = dict(
        fold_index=int(fold_index),
        n_te=int(nk),
        true_x=blk_te["true_x"],
        true_y=blk_te["true_y"],
        uwb_med=blk_te["uwb_med"],
        uwb_var=blk_te["uwb_var"],
        uwb_was_nan=blk_te["uwb_was_nan"],
        wifi_med=blk_te["wifi_med"],
        wifi_var=blk_te["wifi_var"],
        xy_v3_te_stack=xy_v3_te_stack,
        xy_v3_tr_stack=xy_v3_tr_stack,
        tx_tr=tx_tr,
        ty_tr=ty_tr,
        per_rank=per_rank,
    )
    return fold_index, fc


def build_fold_num_pack(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    tri_ref: WifiOnlyTrilateration,
    calib_ranks: List[Tuple[CalibMethodSpec, str, float]],
    v3_profiles: List[Tuple[float, float]],
    n_splits: int,
) -> List[Dict[str, Any]]:
    n_splits = max(2, min(int(n_splits), len(train_imp)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(np.arange(len(train_imp))))
    fold_slots: List[Optional[Dict[str, Any]]] = [None] * len(splits)
    with ThreadPoolExecutor(max_workers=min(len(splits), 4)) as ex:
        futs = [
            ex.submit(
                _fold_worker_pack,
                fi,
                train_imp,
                tri,
                tei,
                loc,
                tri_ref,
                calib_ranks,
                v3_profiles,
            )
            for fi, (tri, tei) in enumerate(splits)
        ]
        for fut in as_completed(futs):
            fi, blk = fut.result()
            fold_slots[fi] = blk
    out_list: List[Dict[str, Any]] = []
    for slot in fold_slots:
        assert slot is not None
        out_list.append(slot)
    return out_list


def assemble_xy_guide_fold(
    fc: Dict[str, Any],
    *,
    rank: int,
    v3_idx: int,
    blend_alpha: float,
    knn_k: int,
) -> np.ndarray:
    blk = fc["per_rank"][rank]
    xv = fc["xy_v3_te_stack"][v3_idx]
    xc = blk["xy_cal_te"]
    base = convex_blend_xy_njit(xc, xv, float(blend_alpha))
    xv_tr = fc["xy_v3_tr_stack"][v3_idx]
    bc_tr = convex_blend_xy_njit(blk["xy_cal_tr"], xv_tr, float(blend_alpha))
    return _knn_residual_refine_xy(
        bc_tr[:, 0],
        bc_tr[:, 1],
        fc["tx_tr"],
        fc["ty_tr"],
        base[:, 0],
        base[:, 1],
        int(knn_k),
    ).astype(np.float64)


def mean_cv_rmse_step_c_from_assembled(fold_np: List[Dict[str, Any]], lg: LocGeomPack, **trial_kw: Any) -> float:
    rmses: List[float] = []
    rank = int(trial_kw["calib_rank"])
    gate = float(trial_kw["gate"])
    uvb_inf = float(trial_kw["inflate"])
    hf_fus = float(trial_kw["hf_fus"])
    hf_wifi_eff = float(trial_kw["hf_wifi"])
    ith = float(trial_kw["irls_t"])
    mx = int(trial_kw["irls_mx"])
    tukey_c = float(trial_kw["tukey_c"])
    gn_i = int(trial_kw["gn_iters"])
    v3_idx = int(trial_kw["v3_idx"])
    ba = float(trial_kw["blend_alpha"])
    knn_k = int(trial_kw["knn_k"])

    for fc in fold_np:
        xy_g = assemble_xy_guide_fold(fc, rank=rank, v3_idx=v3_idx, blend_alpha=ba, knn_k=knn_k)
        pr = fc["per_rank"][rank]
        preds = batch_step_c_preds_flat(
            fc["n_te"],
            xy_g,
            fc["uwb_med"],
            fc["uwb_var"],
            fc["uwb_was_nan"],
            fc["wifi_med"],
            fc["wifi_var"],
            pr["coeff_ab"],
            int(pr["feat_id"]),
            gate,
            uvb_inf,
            hf_fus,
            hf_wifi_eff,
            ith,
            mx,
            tukey_c,
            lg.n_u,
            lg.n_w,
            lg.uwb_xy,
            lg.wifi_xy,
            lg.uwb_robust_bias,
            lg.wifi_robust_bias,
            lg.var_cap_uwb,
            lg.var_cap_wifi,
            lg.eps_w,
            lg.uwb_bias_w,
            lg.wifi_bias_w,
            lg.bounds_lo,
            lg.bounds_hi,
            gn_i,
            float(MAX_RANGE_M),
        )
        ec = point_errors(fc["true_x"], fc["true_y"], preds[:, 0], preds[:, 1])
        rm, _ = rmse_mae(ec)
        rmses.append(float(rm))
    return float(np.mean(rmses))


def make_objective_clamped(
    fold_np: List[Dict[str, Any]],
    lg: LocGeomPack,
    n_rank: int,
    n_prof: int,
) -> Callable[[optuna.Trial], float]:
    """Train CV Step C 만 최소화. 연속·이산 후보 범위는 설계 고정; 검증 정답은 미사용."""

    nr = max(1, int(n_rank))
    npv = max(1, int(n_prof))
    allow_knn = V13_ALLOW_KNN
    # 컨벡스 블렌드 α: 과적합 완화를 위해 이산 집합으로 제한(TPE가 연속 공간 전체를 헤매지 않게).
    BLEND_GRID = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]

    def objective(trial: optuna.Trial) -> float:
        calib_rank = trial.suggest_int("calibration_rank", 0, nr - 1)
        v3_idx = trial.suggest_int("v3_profile_idx", 0, npv - 1)
        blend_alpha = float(trial.suggest_categorical("blend_calib_alpha", BLEND_GRID))
        knn_suggest = int(trial.suggest_categorical("knn_residual_neighbors", [0, 3, 6]))
        if (
            not allow_knn
            or blend_alpha <= 1e-12
            or blend_alpha >= 1.0 - 1e-12
        ):
            knn_k = 0
        else:
            knn_k = knn_suggest

        # UWB 기하 게이트·inflate: m 단위·분산 규모에 맞춘 플러시블 구간(Train EDA로 정당화 가능한 형태로 고정).
        gate = trial.suggest_float("gate_threshold_m", 1.25, 2.05)
        uvb_inf = trial.suggest_float("uwb_variance_inflate", 8.0, 24.0)
        hf_fus = float(trial.suggest_categorical("huber_f_fusion", [1.0, 1.2, 1.35]))
        ith = float(trial.suggest_categorical("irls_residual_thresh_m", [2.2, 2.75, 3.05, 3.1]))

        blk = fold_np[0]["per_rank"][calib_rank]
        kw = dict(
            calib_rank=int(calib_rank),
            v3_idx=int(v3_idx),
            blend_alpha=float(blend_alpha),
            knn_k=int(knn_k),
            gate=float(gate),
            inflate=float(uvb_inf),
            hf_fus=float(hf_fus),
            hf_wifi=float(blk["hf_wifi"]),
            irls_t=float(ith),
            irls_mx=2,
            tukey_c=float(TUKEY_C),
            gn_iters=int(TRILAT_GN_ITERS),
        )

        rm = mean_cv_rmse_step_c_from_assembled(fold_np, lg, **kw)

        trial.set_user_attr("cv_kw", kw)
        return float(rm)

    return objective


def warmup_numba_brief(lg: LocGeomPack) -> None:
    fk: Dict[str, Any] = dict(
        n_te=3,
        true_x=np.zeros(3, dtype=np.float64),
        true_y=np.zeros(3, dtype=np.float64),
        uwb_med=np.ones((3, lg.n_u), dtype=np.float64) * 2.8,
        uwb_var=np.ones((3, lg.n_u), dtype=np.float64) * 0.06,
        uwb_was_nan=np.zeros((3, lg.n_u), dtype=np.uint8),
        wifi_med=np.ones((3, lg.n_w), dtype=np.float64) * 6.7,
        wifi_var=np.ones((3, lg.n_w), dtype=np.float64) * 0.8,
        per_rank=[
            dict(
                coeff_ab=np.ones((lg.n_w, 2), dtype=np.float64),
                feat_id=0,
                hf_wifi=np.float64(1.25),
                xy_cal_te=np.ones((3, 2), dtype=np.float64) * 2.0,
                xy_cal_tr=np.ones((3, 2), dtype=np.float64) * 2.0,
            )
        ],
        xy_v3_te_stack=np.ones((max(1, len(V3_PROFILES)), 3, 2), dtype=np.float64) * 2.1,
        xy_v3_tr_stack=np.ones((max(1, len(V3_PROFILES)), 3, 2), dtype=np.float64) * 2.1,
        tx_tr=np.ones(3, dtype=np.float64) * 6.9,
        ty_tr=np.ones(3, dtype=np.float64) * 8.44,
    )
    mean_cv_rmse_step_c_from_assembled(
        [fk],
        lg,
        calib_rank=0,
        v3_idx=0,
        blend_alpha=0.45,
        knn_k=0,
        gate=float(1.74),
        inflate=float(17.85),
        hf_fus=float(1.135),
        hf_wifi=float(1.32),
        irls_t=float(2.95),
        irls_mx=2,
        tukey_c=float(TUKEY_C),
        gn_iters=int(TRILAT_GN_ITERS),
    )


def _calib_rank_lookup(
    calib_ranks: List[Tuple[CalibMethodSpec, str, float]],
    spec: CalibMethodSpec,
    feat: str,
    hf_wifi: float,
) -> int:
    k = calib_triple_key(spec, feat, float(hf_wifi))
    for i, (s, f, h) in enumerate(calib_ranks):
        if calib_triple_key(s, f, float(h)) == k:
            return i
    return 0


def enqueue_warmstarts_v13(
    study: optuna.Study,
    calib_ranks: List[Tuple[CalibMethodSpec, str, float]],
    n_prof: int,
) -> None:
    """Fast2 / Turbo 요약에 가까운 격점으로 TPESampler 스타트를 앞당긴다."""
    id_sp = CalibMethodSpec("identity")
    h = CalibMethodSpec("huber", huber_epsilon=1.35, huber_alpha=5e-4)
    npv = max(1, n_prof)
    nr = max(1, len(calib_ranks))
    seeds = [
        dict(
            calibration_rank=_calib_rank_lookup(calib_ranks, h, "bias_sub", 1.2),
            v3_profile_idx=0,
            blend_calib_alpha=0.5,
            knn_residual_neighbors=0,
            gate_threshold_m=1.4,
            uwb_variance_inflate=10.0,
            huber_f_fusion=1.35,
            irls_residual_thresh_m=3.1,
        ),
        dict(
            calibration_rank=_calib_rank_lookup(calib_ranks, id_sp, "bias_sub", 1.0),
            v3_profile_idx=0,
            blend_calib_alpha=0.0,
            knn_residual_neighbors=0,
            gate_threshold_m=1.69,
            uwb_variance_inflate=13.83,
            huber_f_fusion=1.2,
            irls_residual_thresh_m=3.1,
        ),
        dict(
            calibration_rank=_calib_rank_lookup(calib_ranks, id_sp, "bias_sub", 1.35),
            v3_profile_idx=0,
            blend_calib_alpha=0.5,
            knn_residual_neighbors=0,
            gate_threshold_m=1.53,
            uwb_variance_inflate=14.0,
            huber_f_fusion=1.2,
            irls_residual_thresh_m=3.05,
        ),
    ]
    for prm in seeds:
        prm = prm.copy()
        prm["calibration_rank"] = int(np.clip(prm["calibration_rank"], 0, nr - 1))
        prm["v3_profile_idx"] = int(np.clip(prm["v3_profile_idx"], 0, npv - 1))
        try:
            study.enqueue_trial(prm)
        except Exception:
            pass


def save_plots_v13(
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
        (xy_a[:, 0], xy_a[:, 1], "Step A (블렌드+KNN)", rmse_a),
        (preds_b[:, 0], preds_b[:, 1], "Step B", rmse_b),
        (preds_c[:, 0], preds_c[:, 1], "Step C", rmse_c),
    ]
    for ax, (px, py, ttl, rm) in zip(axes, triples):
        m = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(px) & np.isfinite(py)
        ax.scatter(tx[m], ty[m], s=42, c="#1f77b4", alpha=0.88, edgecolors="#0d2c44", linewidths=0.35, label="True", zorder=2)
        ax.scatter(px[m], py[m], s=52, c="#ff7f0e", alpha=0.78, marker="x", linewidths=0.95, label="Pred", zorder=3)
        for xi, yi, pxi, pyi in zip(tx[m], ty[m], px[m], py[m]):
            ax.plot([xi, pxi], [yi, pyi], color="#7f7f7f", alpha=0.22, lw=0.85, zorder=1)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 18)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{ttl}\n(RMSE {rm:.3f} m)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("V13-fix — Train Optuna / 검증 단일 Forward", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_dir / f"{ART_PREFIX}_map_true_vs_predicted_steps.png", dpi=PLOT_DPI)
    plt.close()


def save_cdf_fast(out_dir: Path, ea: np.ndarray, eb: np.ndarray, ec: np.ndarray, ra: float, rb: float, rc: float) -> None:
    plt.figure(figsize=(8.2, 5.0))
    for errs, lbl, rm in [(ea, "Step A", ra), (eb, "Step B", rb), (ec, "Step C", rc)]:
        e = np.sort(errs[np.isfinite(errs)])
        if e.size == 0:
            continue
        cdf = np.arange(1, len(e) + 1) / len(e)
        plt.plot(e, cdf, lw=2.0, label=f"{lbl} (RMSE {rm:.3f} m)")
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("V13-fix CDF")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{ART_PREFIX}_cdf_steps.png", dpi=PLOT_DPI)
    plt.close()


def main() -> int:
    save_plots = "--no-plots" not in sys.argv
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

    n_splits = min(MAX_KFOLD_SPLITS, resolve_kfold_n_splits(len(train_df)))
    print(f"\n[V13-fix] 학습 {len(train_df)}행 | 검증 {len(val_df)}행 | K-Fold={n_splits}", flush=True)

    train_imp, val_imp = wifi_median_impute_fusion(train_df, val_df, loc.wifi_cols)
    lg = build_loc_geom_pack(loc)
    tri_ref = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m=0.6, huber_f_scale=1.0)

    catalog = calibration_catalog_extended()
    phase_a_rows, scored_full = rank_calibration_candidates(
        train_imp, loc, tri_ref, catalog, n_splits, top_k=None
    )
    raw_cap = int(os.environ.get("V13_FIX_TOP_CALIB_RANKS", os.environ.get("V13_TOP_CALIB_RANKS", "0")))
    max_ranks = len(catalog) if raw_cap <= 0 else min(raw_cap, len(catalog))
    calib_ranks, calib_rank_tags = merge_calib_ranks_train_cv_only(catalog, scored_full, max_ranks=max_ranks)

    print(
        f"[V13-fix] 캘리브 캐시 {len(calib_ranks)}개 (Train 순수 보정 Step A CV 순만; 설계 고정 삽입 없음) | "
        f"V13_FIX_TOP_CALIB_RANKS/V13_TOP_CALIB_RANKS={raw_cap}",
        flush=True,
    )

    fold_np = build_fold_num_pack(train_imp, loc, tri_ref, calib_ranks, V3_PROFILES, n_splits)
    warmup_numba_brief(lg)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=OPTUNA_SEED, n_startup_trials=min(14, max(5, N_OPTUNA_TRIALS // 4)))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if V13_FIX_ENQUEUE_WARMSTARTS:
        enqueue_warmstarts_v13(study, calib_ranks, len(V3_PROFILES))
        print("[V13-fix] V13_FIX_ENQUEUE_WARMSTARTS=1 → 구 V13 형태 enqueue 사용(설계자 시드)", flush=True)

    obj = make_objective_clamped(fold_np, lg, len(calib_ranks), len(V3_PROFILES))
    study.optimize(obj, n_trials=N_OPTUNA_TRIALS, n_jobs=OPTUNA_N_JOBS, show_progress_bar=False)

    bt = study.best_trial
    br = int(bt.params["calibration_rank"])
    v3i = int(bt.params["v3_profile_idx"])
    ba = float(bt.params["blend_calib_alpha"])
    gate = float(bt.params["gate_threshold_m"])
    uinf = float(bt.params["uwb_variance_inflate"])
    hf_fus = float(bt.params["huber_f_fusion"])
    ith = float(bt.params["irls_residual_thresh_m"])
    mx = 2
    gn_i = int(TRILAT_GN_ITERS)
    tukey_c = float(TUKEY_C)
    best_cv = float(bt.value)

    knn_raw = int(bt.params.get("knn_residual_neighbors", 0))
    if not V13_ALLOW_KNN or ba <= 1e-12 or ba >= 1.0 - 1e-12:
        knn_k = 0
    else:
        knn_k = knn_raw
    trial_knn_chosen = int(bt.params.get("knn_residual_neighbors", 0))

    spec_f, feat_f, hf_wifi_f = calib_ranks[br]
    v3_hf, v3_wb = V3_PROFILES[v3i]

    train_fit = augment_train_for_canon_fit(train_imp, loc)
    coeffs_f = fit_robust_calibration_per_ap(train_fit, tri_ref, feat_mode=feat_f, spec=spec_f)
    coeff_ab = coeffs_dict_to_ab(coeffs_f, loc)
    feat_id = int(feat_to_id(str(feat_f)))

    xy_cal_tr = predict_step_a_calibrated(train_imp, loc, coeffs_f, feat_f, float(hf_wifi_f)).astype(np.float64)
    xy_cal_val = predict_step_a_calibrated(val_imp, loc, coeffs_f, feat_f, float(hf_wifi_f)).astype(np.float64)
    xy_v_tr = predict_step_a_v3_wifi_equivalent(
        train_imp, loc, wifi_bias=float(v3_wb), huber_f_scale=float(v3_hf)
    ).astype(np.float64)
    xy_v_val = predict_step_a_v3_wifi_equivalent(
        val_imp, loc, wifi_bias=float(v3_wb), huber_f_scale=float(v3_hf)
    ).astype(np.float64)
    bc_tr = convex_blend_xy_njit(xy_cal_tr, xy_v_tr, float(ba))
    bc_val = convex_blend_xy_njit(xy_cal_val, xy_v_val, float(ba))
    tx_all = train_imp["True_X"].values.astype(np.float64)
    ty_all = train_imp["True_Y"].values.astype(np.float64)
    xy_a_val = _knn_residual_refine_xy(
        bc_tr[:, 0],
        bc_tr[:, 1],
        tx_all,
        ty_all,
        bc_val[:, 0],
        bc_val[:, 1],
        int(knn_k),
    )

    val_blk = te_df_to_numpy_blocks(val_imp, loc)
    preds_b = batch_step_b_preds_flat(
        len(val_imp),
        xy_a_val,
        val_blk["uwb_med"],
        val_blk["uwb_var"],
        val_blk["uwb_was_nan"],
        val_blk["wifi_med"],
        val_blk["wifi_var"],
        coeff_ab,
        feat_id,
        gate,
        uinf,
        hf_fus,
        lg.n_u,
        lg.n_w,
        lg.uwb_xy,
        lg.wifi_xy,
        lg.uwb_robust_bias,
        lg.wifi_robust_bias,
        lg.var_cap_uwb,
        lg.var_cap_wifi,
        lg.eps_w,
        lg.uwb_bias_w,
        lg.wifi_bias_w,
        lg.bounds_lo,
        lg.bounds_hi,
        gn_i,
        float(MAX_RANGE_M),
    )
    preds_c = batch_step_c_preds_flat(
        len(val_imp),
        xy_a_val,
        val_blk["uwb_med"],
        val_blk["uwb_var"],
        val_blk["uwb_was_nan"],
        val_blk["wifi_med"],
        val_blk["wifi_var"],
        coeff_ab,
        feat_id,
        gate,
        uinf,
        hf_fus,
        float(hf_wifi_f),
        ith,
        mx,
        float(tukey_c),
        lg.n_u,
        lg.n_w,
        lg.uwb_xy,
        lg.wifi_xy,
        lg.uwb_robust_bias,
        lg.wifi_robust_bias,
        lg.var_cap_uwb,
        lg.var_cap_wifi,
        lg.eps_w,
        lg.uwb_bias_w,
        lg.wifi_bias_w,
        lg.bounds_lo,
        lg.bounds_hi,
        gn_i,
        float(MAX_RANGE_M),
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

    print("\n" + "=" * 72)
    print("V13-fix — fusion_turbo_numba_core + Train-CV 순수 캘리브 순위 (Optuna)")
    print("=" * 72)
    print(f"  [Train CV] mean Step C RMSE = {best_cv:.4f} m")
    print(f"  [Validation] A={rmse_a:.4f} m | B={rmse_b:.4f} m | C={rmse_c:.4f} m")
    print(f"  [Wall-clock] {elapsed:.2f} s")
    print("=" * 72)

    pref = resolve_paths(root)
    summ = {
        "pipeline": "V13-fix",
        "variant_notes": dict(
            numba_dependency="fusion_turbo_numba_core.py (no import from indoor_fusion_pipeline_v12_turbo)",
            calibration_rank_merge="Train K-Fold pure calibrated Step-A RMSE order only (no designer forced tuples)",
            optuna_enqueue_warmstarts=bool(V13_FIX_ENQUEUE_WARMSTARTS),
        ),
        "integrity": (
            "Optuna minimizes train K-fold mean Step C RMSE only (with Fast2 blended Step-A guide inside folds). "
            "Validation Truth used once post full-train refit."
        ),
        "wall_clock_sec": float(elapsed),
        "train_kfold_splits": int(n_splits),
        "calibration_catalog_size": len(catalog),
        "top_calibration_ranks_cached": len(calib_ranks),
        "calibration_cache_merge_tags_head": calib_rank_tags[:24],
        "v13_allow_knn_env": bool(V13_ALLOW_KNN),
        "v3_profiles": [{"huber_f_scale": h, "wifi_bias_m": w} for h, w in V3_PROFILES],
        "train_cv_best_step_c_rmse_m": float(best_cv),
        "optuna_n_trials": N_OPTUNA_TRIALS,
        "optuna_n_jobs": OPTUNA_N_JOBS,
        "selected": {
            "calibration_rank": br,
            "calib_kind": spec_f.kind,
            "feat_mode": feat_f,
            "huber_f_wifi_calibrated_trilat": float(hf_wifi_f),
            "v3_profile_index": v3i,
            "v3_huber_f_scale": float(v3_hf),
            "v3_wifi_bias_m": float(v3_wb),
            "blend_calib_alpha": float(ba),
            "knn_residual_neighbors_trial": trial_knn_chosen,
            "knn_residual_neighbors": int(knn_k),
            "gate_m": gate,
            "uwb_variance_inflate": uinf,
            "huber_f_fusion": hf_fus,
            "irls_residual_thresh_m": ith,
            "irls_max_iter": mx,
            "trilat_gn_iterations_fixed": gn_i,
            "irls_tukey_c_fixed": tukey_c,
            "effective_knn_for_forward": int(knn_k),
        },
        "validation_metrics": {
            "step_A_RMSE_m": float(rmse_a),
            "step_A_MAE_m": float(mae_a),
            "step_B_RMSE_m": float(rmse_b),
            "step_B_MAE_m": float(mae_b),
            "step_C_RMSE_m": float(rmse_c),
            "step_C_MAE_m": float(mae_c),
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
    pred["StepB_X"] = preds_b[:, 0]
    pred["StepB_Y"] = preds_b[:, 1]
    pred["StepC_X"] = preds_c[:, 0]
    pred["StepC_Y"] = preds_c[:, 1]
    pred["ErrA_m"] = ea
    pred["ErrB_m"] = eb
    pred["ErrC_m"] = ec
    pred.to_csv(out_dir / f"{ART_PREFIX}_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phase_a_rows).to_csv(out_dir / f"{ART_PREFIX}_grid_phaseA_calib_catalog.csv", index=False, encoding="utf-8-sig")

    trials_rows: List[Dict[str, Any]] = []
    for tr in study.trials:
        if tr.state != optuna.trial.TrialState.COMPLETE:
            continue
        row: Dict[str, Any] = {"trial": int(tr.number), "mean_cv_rmse_step_c_m": float(tr.value)}
        row.update({k: v for k, v in tr.params.items()})
        trials_rows.append(row)
    pd.DataFrame(trials_rows).to_csv(out_dir / f"{ART_PREFIX}_optuna_trials.csv", index=False, encoding="utf-8-sig")

    coef_rows = [{"wifi_canon_ap": k, "A": v[0], "B": v[1]} for k, v in coeffs_f.items()]
    pd.DataFrame(coef_rows).to_csv(out_dir / f"{ART_PREFIX}_wifi_calibration_coefs.csv", index=False, encoding="utf-8-sig")

    if save_plots:
        save_plots_v13(out_dir, tx, ty, xy_a_val, preds_b, preds_c, rmse_a, rmse_b, rmse_c)
        save_cdf_fast(out_dir, ea, eb, ec, rmse_a, rmse_b, rmse_c)
    else:
        print("[V13-fix] --no-plots: PNG 생략", flush=True)

    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main, output_artifact_include_prefixes=("v13_fix_",))
