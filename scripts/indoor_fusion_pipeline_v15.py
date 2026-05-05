"""
V15 — Train·경험적 상수만으로 하이퍼를 정하고, 검증 정답은 **최종 1회 평가**에만 사용한다.

설계 원칙 (간접 누수 차단)
-------------------------
- ``indoor_fusion_pipeline_v12.py`` 의 검증튜닝 이산 그리드(GATE_THRESH_GRID 등)를 **import하지 않는다**.
- 게이트 Optuna 구간은 **train K-Fold OOF**에서 Wi‑Fi Step A 기준 UWB 기하 잔차(행별 중앙값) 분위수로 계산한다.
- V3 (무경계 Wi‑Fi 삼변) 후보 튜플 ``(huber_f_scale, wifi_bias_m)`` 은 **train CV Step A RMSE**로 순위를 매겨 상위 K개만 Optuna에 넘긴다.
- Wi‑Fi 보정 캘리브 후보 목록은 v12 모듈 상수에 의존하지 않고, 이 파일의 ``v15_calibration_catalog()`` 에만 둔다.
- Optuna enqueue warmstart **없음**.

수식 골격은 v13-fix와 동일(``fusion_turbo_numba_core`` Numba Step B/C, 블렌드·선택적 KNN).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as e:  # pragma: no cover
    raise ImportError("V15에는 optuna가 필요합니다: py -3 -m pip install optuna") from e

from fusion_realtime_sanitize import (
    VAR_CAP_UWB_FUSION,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
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
from indoor_fusion_pipeline_v13_fix import (
    assemble_xy_guide_fold,
    build_fold_num_pack,
    calib_triple_key,
    convex_blend_xy_njit,
    mean_cv_rmse_step_c_from_assembled,
    merge_calib_ranks_train_cv_only,
    rank_calibration_candidates,
    _knn_residual_refine_xy,
)
from indoor_fusion_pipeline_v8 import (
    MAX_RANGE_M,
    ROBUST_BIAS_M,
    WIFI_COL_CANON,
    Config,
    FusionLocalizerV8,
    sensor_canon_key,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except (AttributeError, OSError):
        pass
plt.rcParams["axes.unicode_minus"] = False

ART_PREFIX = "v15"
MAX_KFOLD_SPLITS = int(os.environ.get("V15_MAX_KFOLD", "3"))
N_OPTUNA_TRIALS = int(os.environ.get("V15_OPTUNA_TRIALS", "120"))
N_OPTUNA_TRIALS_STEPA = int(os.environ.get("V15_OPTUNA_TRIALS_STEPA", "48"))
OPTUNA_N_JOBS = int(os.environ.get("V15_OPTUNA_JOBS", os.environ.get("V13_FIX_OPTUNA_JOBS", "-1")))
OPTUNA_SEED = int(os.environ.get("V15_OPTUNA_SEED", "42"))
V15_TWO_PHASE = os.environ.get("V15_TWO_PHASE", "0").strip().lower() in ("1", "true", "yes")
V15_STEPA_LOSS_WEIGHT = float(os.environ.get("V15_STEPA_LOSS_WEIGHT", "0.452"))
PLOT_DPI = int(os.environ.get("V15_PLOT_DPI", "110"))
V15_ALLOW_KNN = os.environ.get("V15_ALLOW_KNN", "1").strip().lower() in ("1", "true", "yes")
V15_TOP_V3 = int(os.environ.get("V15_TOP_V3_PROFILES", "8"))
V15_TOP_CALIB = int(os.environ.get("V15_TOP_CALIB_RANKS", "0"))  # 0 = 전부
# balanced | defensive | v13_fix — 후자는 indoor_fusion_pipeline_v13_fix.make_objective_clamped 와 동일 탐색 박스(검증 미사용 설계안)
V15_TUNING_PROFILE = os.environ.get("V15_TUNING_PROFILE", "balanced").strip().lower()


def resolve_v15_tuning_space(profile: str) -> Dict[str, Any]:
    """탐색 공간 프로필. 검증 라벨은 쓰지 않는다.

    - **balanced**: Train-OOF 게이트 상한만 쓰고 나머지는 넓게 탐색(런 간 분산 큼).
    - **defensive**: 경험적 좁은 박스(UWB 과신 완화). 게이트 캡은 ``V15_DEFENSIVE_GATE_*``.
    - **v13_fix** (별칭 ``v13fix``, ``v13_compat``): v13_fix의 ``make_objective_clamped`` 범위를 그대로 복제.
      v13_fix docstring 근거(Train EDA·관례로 고정한 플러시블 구간 + Train CV만으로 하이퍼 선택)와 동일한
      무결성 스토리를 v15 composite 목적함수에 적용할 때 사용한다.

    필요 시 게이트는 ``V15_DEFENSIVE_GATE_LO_MIN`` (기본 1.15 m), ``V15_DEFENSIVE_GATE_HI_MAX`` (기본 2.8 m) 로 덮어쓴다.
    """
    p = (profile or "balanced").strip().lower()
    gate_lo_floor_d = float(os.environ.get("V15_DEFENSIVE_GATE_LO_MIN", "1.15"))
    gate_hi_cap_d = float(os.environ.get("V15_DEFENSIVE_GATE_HI_MAX", "2.8"))
    if p in ("v13_fix", "v13fix", "v13_compat", "like_v13_fix"):
        # indoor_fusion_pipeline_v13_fix.py make_objective_clamped 와 동일
        return dict(
            name="v13_fix",
            gate_lo_floor=float(os.environ.get("V15_V13FIX_GATE_LO", "1.25")),
            gate_hi_cap=float(os.environ.get("V15_V13FIX_GATE_HI", "2.05")),
            inflate_lo=float(os.environ.get("V15_V13FIX_INFLATE_LO", "8.0")),
            inflate_hi=float(os.environ.get("V15_V13FIX_INFLATE_HI", "24.0")),
            huber_f_fusion_choices=[1.0, 1.2, 1.35],
            irls_thresh_choices=[2.2, 2.75, 3.05, 3.1],
            blend_grid=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0],
            step_a_blend_grid=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0],
        )
    if p in ("defensive", "narrow", "conservative"):
        return dict(
            name="defensive",
            gate_lo_floor=gate_lo_floor_d,
            gate_hi_cap=min(3.95, gate_hi_cap_d),
            inflate_lo=float(os.environ.get("V15_DEFENSIVE_INFLATE_LO", "14.0")),
            inflate_hi=float(os.environ.get("V15_DEFENSIVE_INFLATE_HI", "33.0")),
            huber_f_fusion_choices=[1.0, 1.12],
            irls_thresh_choices=[2.45, 2.75, 3.05],
            blend_grid=[0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0],
            step_a_blend_grid=[0.2, 0.35, 0.45, 0.55, 0.65, 0.7, 0.85, 1.0],
        )
    return dict(
        name="balanced",
        gate_lo_floor=None,
        gate_hi_cap=None,
        inflate_lo=10.0,
        inflate_hi=38.0,
        huber_f_fusion_choices=[1.0, 1.12, 1.25, 1.35],
        # IRLS 잔차 임계 (m). 등간격 그리드가 아님 — 설계 의도만 주석으로 고정(검증 라벨 역산 아님).
        # - 대역: 실내 실험에서 위치 오차가 자주 올라가는 2~3 m 전후를「한 단계에서 이상치로 볼지」
        #   경계로 잡는 실용 구간. v13_fix의 [2.2, 2.75, 3.05, 3.1] 계열과 이어지게 2.1·2.45·3.2를 둠.
        # - 2.1: v13의 2.2보다 약간 낮은 쪽(더 이른 다운웨이트) 탐색용 한 점.
        # - 2.45 / 2.75 / 중간 스텝: 아래쪽~중앙은 ~0.3 m 간격으로 넓게 훑음(연속 변경에 덜 예민한 구간).
        # - 3.05 → 3.2 만 0.15 m: 3 m 부근에서는 임계 근처에서 참가/제외 균형이 갈려 Train CV 목적이
        #   상대적으로 민감한 편이라, 상한 근처만 로컬 정밀화(격자 반쯤 촘촘히).
        irls_thresh_choices=[2.1, 2.45, 2.75, 3.05, 3.2],
        blend_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0],
        step_a_blend_grid=[0.0, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 1.0],
    )


def apply_gate_tuning_caps(gate_lo: float, gate_hi: float, tuning: Dict[str, Any]) -> Tuple[float, float]:
    lo, hi = float(min(gate_lo, gate_hi)), float(max(gate_lo, gate_hi))
    gf, gc = tuning.get("gate_lo_floor"), tuning.get("gate_hi_cap")
    if gf is not None:
        lo = max(lo, float(gf))
    if gc is not None:
        hi = min(hi, float(gc))
    if hi <= lo + 0.18:
        hi = min(lo + 2.2, hi + 1.8)
    return lo, hi


def v15_calibration_catalog() -> List[Tuple[CalibMethodSpec, str, float]]:
    """v12.GATE_THRESH_GRID 등 검증튜닝 모듈에 의존하지 않는 캘리브 후보(이 파일 단독 정의)."""
    h = CalibMethodSpec("huber", huber_epsilon=1.35, huber_alpha=5e-4)
    h_tight = CalibMethodSpec("huber", huber_epsilon=1.35, huber_alpha=2e-4)
    h_lo = CalibMethodSpec("huber", huber_epsilon=1.25, huber_alpha=5e-4)
    id_sp = CalibMethodSpec("identity")
    rs35 = CalibMethodSpec("ransac", ransac_residual_threshold=3.5)
    rs55 = CalibMethodSpec("ransac", ransac_residual_threshold=5.5)
    feat = "bias_sub"
    rows = [
        (id_sp, feat, 1.0),
        (id_sp, feat, 1.05),
        (id_sp, feat, 1.1),
        (id_sp, feat, 1.15),
        (id_sp, feat, 1.28),
        (id_sp, feat, 1.35),
        (h_lo, feat, 1.15),
        (h_lo, feat, 1.2),
        (h, feat, 1.0),
        (h, feat, 1.05),
        (h, feat, 1.1),
        (h, feat, 1.2),
        (h, feat, 1.28),
        (h_tight, feat, 1.28),
        (h, feat, 1.32),
        (h, feat, 1.35),
        (rs35, feat, 1.35),
        (rs55, feat, 1.35),
    ]
    uniq: Dict[Tuple[Any, ...], Tuple[CalibMethodSpec, str, float]] = {}
    for sp, ft, hf in rows:
        uniq[calib_triple_key(sp, ft, float(hf))] = (sp, ft, float(hf))
    return list(uniq.values())


def train_oof_gate_row_median_stats(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    n_splits: int,
) -> Tuple[float, float, Dict[str, float]]:
    """Train K-Fold OOF: Wi‑Fi Step A 기준 UWB 앵커별 |d_geom−d_meas| 의 **행 중앙값** 분포."""
    n_splits = max(2, min(int(n_splits), len(train_imp)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    row_medians: List[float] = []
    for tr_idx, te_idx in kf.split(np.arange(len(train_imp))):
        tr_raw = train_imp.iloc[tr_idx].reset_index(drop=True)
        te_raw = train_imp.iloc[te_idx].reset_index(drop=True)
        tr_imp, te_imp = wifi_median_impute_fusion(tr_raw, te_raw, loc.wifi_cols)
        xa, ya = loc.predict_step(te_imp, "A")
        for k, (_, row) in enumerate(te_imp.iterrows()):
            wx, wy = float(xa[k]), float(ya[k])
            if not (np.isfinite(wx) and np.isfinite(wy)):
                continue
            errs: List[float] = []
            for i, c in enumerate(loc.uwb_cols):
                d_raw = row.get(f"med_{c}", np.nan)
                v = row.get(f"var_{c}", np.nan)
                was_nan = row.get(f"orig_nan_{c}", 1) == 1
                if was_nan or not np.isfinite(d_raw) or not np.isfinite(v):
                    continue
                if float(v) > VAR_CAP_UWB_FUSION:
                    continue
                key_u = sensor_canon_key(c, i, True)
                ax, ay = loc._resolve_sensor_position_m(c, i, True)
                d_meas = float(d_raw) - ROBUST_BIAS_M[key_u]
                d_geom = float(np.hypot(wx - ax, wy - ay))
                errs.append(abs(d_geom - d_meas))
            if errs:
                row_medians.append(float(np.median(errs)))
    arr = np.asarray(row_medians, dtype=np.float64)

    def q(p: float) -> float:
        return float(np.percentile(arr, p)) if arr.size else 2.0

    # 물리 상한: v9 주석 계열(실내 ranging) + train 분위수
    lo = max(0.85, round(q(12) * 0.92, 2))
    hi = min(8.5, round(q(97) * 1.08, 2))
    if hi <= lo + 0.25:
        hi = min(8.5, lo + 2.5)
    # 실내 Wi‑Fi 가이드 UWB 기하 게이트: train OOF가 넓어도 Optuna가 비현실적으로 큰 τ를 쓰지 않도록 상한 캡
    hi = min(float(hi), 3.95)
    lo = max(float(lo), 1.05)
    if hi <= lo + 0.2:
        hi = lo + 1.8
    meta = dict(p50=q(50), p75=q(75), p90=q(90), n_rows=float(arr.size))
    return lo, hi, meta


def cv_mean_rmse_v3_profile(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    huber_f: float,
    wifi_bias_m: float,
    n_splits: int,
) -> float:
    n_splits = max(2, min(int(n_splits), len(train_imp)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses: List[float] = []
    for tr_idx, te_idx in kf.split(np.arange(len(train_imp))):
        te_df = train_imp.iloc[te_idx].reset_index(drop=True)
        xy = predict_step_a_v3_wifi_equivalent(
            te_df, loc, wifi_bias=float(wifi_bias_m), huber_f_scale=float(huber_f)
        )
        tx = te_df["True_X"].values.astype(float)
        ty = te_df["True_Y"].values.astype(float)
        ea = point_errors(tx, ty, xy[:, 0], xy[:, 1])
        r, _ = rmse_mae(ea)
        rmses.append(float(r))
    return float(np.mean(rmses))


def train_ranked_v3_profiles(
    train_imp: pd.DataFrame,
    loc: FusionLocalizerV8,
    n_splits: int,
    top_k: int,
) -> List[Tuple[float, float]]:
    """Train CV로 순위를 매긴 V3 후보 + 물리·관례 앵커(고정 소수)를 앞에 둔다 — 전량 train 점수 기반."""
    hubers = [1.1, 1.2, 1.25, 1.32, 1.38, 1.45]
    biases = [1.7, 2.0, 2.3, 2.55, 2.85, 3.15]
    scored: List[Tuple[float, float, float]] = []
    for hf in hubers:
        for wb in biases:
            rm = cv_mean_rmse_v3_profile(train_imp, loc, float(hf), float(wb), n_splits)
            scored.append((float(rm), float(hf), float(wb)))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    anchors: List[Tuple[float, float]] = [
        (1.35, 2.5),
        (1.32, 2.35),
        (1.25, 2.2),
        (1.2, 2.5),
    ]
    out: List[Tuple[float, float]] = []
    for t in anchors:
        if t not in out:
            out.append(t)
    for _rm, hf, wb in scored:
        if float(hf) > 1.35 + 1e-9:
            continue
        tup = (float(hf), float(wb))
        if tup not in out:
            out.append(tup)
        if len(out) >= max(6, int(top_k)):
            break
    return out[: max(6, int(top_k))]


def warmup_numba_v15(lg: LocGeomPack, n_prof: int) -> None:
    npv = max(1, int(n_prof))
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
        xy_v3_te_stack=np.ones((npv, 3, 2), dtype=np.float64) * 2.1,
        xy_v3_tr_stack=np.ones((npv, 3, 2), dtype=np.float64) * 2.1,
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
        gate=float(2.8),
        inflate=float(16.0),
        hf_fus=float(1.2),
        hf_wifi=float(1.28),
        irls_t=float(2.85),
        irls_mx=2,
        tukey_c=float(TUKEY_C),
        gn_iters=int(TRILAT_GN_ITERS),
    )


def mean_cv_rmse_step_a_from_assembled(fold_np: List[Dict[str, Any]], **trial_kw: Any) -> float:
    """블렌드+KNN까지 반영한 Step A 좌표의 Train 폴드 평균 RMSE."""
    rmses: List[float] = []
    rank = int(trial_kw["calib_rank"])
    v3_idx = int(trial_kw["v3_idx"])
    ba = float(trial_kw["blend_alpha"])
    knn_k = int(trial_kw["knn_k"])
    for fc in fold_np:
        xy_g = assemble_xy_guide_fold(fc, rank=rank, v3_idx=v3_idx, blend_alpha=ba, knn_k=knn_k)
        ec = point_errors(fc["true_x"], fc["true_y"], xy_g[:, 0], xy_g[:, 1])
        rm, _ = rmse_mae(ec)
        rmses.append(float(rm))
    return float(np.mean(rmses))


def make_objective_v15_composite(
    fold_np: List[Dict[str, Any]],
    lg: LocGeomPack,
    n_rank: int,
    n_prof: int,
    gate_lo: float,
    gate_hi: float,
    w_stepa: float,
    tuning: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Train CV: w·StepA_RMSE + (1-w)·StepC_RMSE — Wi‑Fi 가이드와 융합을 동시에 고려(검증 미사용)."""
    nr = max(1, int(n_rank))
    npv = max(1, int(n_prof))
    glo = float(min(gate_lo, gate_hi))
    ghi = float(max(gate_lo, gate_hi))
    wa = float(min(0.85, max(0.05, w_stepa)))
    allow_knn = V15_ALLOW_KNN
    blend_opts = list(tuning["blend_grid"])
    inflate_lo = float(tuning["inflate_lo"])
    inflate_hi = float(tuning["inflate_hi"])
    fus_choices = [float(x) for x in tuning["huber_f_fusion_choices"]]
    irls_choices = [float(x) for x in tuning["irls_thresh_choices"]]

    def objective(trial: optuna.Trial) -> float:
        calib_rank = trial.suggest_int("calibration_rank", 0, nr - 1)
        v3_idx = trial.suggest_int("v3_profile_idx", 0, npv - 1)
        blend_alpha = float(trial.suggest_categorical("blend_calib_alpha", blend_opts))
        knn_suggest = int(trial.suggest_categorical("knn_residual_neighbors", [0, 3, 6]))
        if not allow_knn or blend_alpha <= 1e-12 or blend_alpha >= 1.0 - 1e-12:
            knn_k = 0
        else:
            knn_k = knn_suggest

        gate = trial.suggest_float("gate_threshold_m", glo, ghi)
        uvb_inf = trial.suggest_float("uwb_variance_inflate", inflate_lo, inflate_hi)
        hf_fus = float(trial.suggest_categorical("huber_f_fusion", fus_choices))
        ith = float(trial.suggest_categorical("irls_residual_thresh_m", irls_choices))

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
        rm_c = mean_cv_rmse_step_c_from_assembled(fold_np, lg, **kw)
        rm_a = mean_cv_rmse_step_a_from_assembled(
            fold_np,
            calib_rank=int(calib_rank),
            v3_idx=int(v3_idx),
            blend_alpha=float(blend_alpha),
            knn_k=int(knn_k),
        )
        loss = wa * float(rm_a) + (1.0 - wa) * float(rm_c)
        trial.set_user_attr("cv_kw", kw)
        trial.set_user_attr("mean_cv_rmse_step_a_m", float(rm_a))
        trial.set_user_attr("mean_cv_rmse_step_c_m", float(rm_c))
        trial.set_user_attr("composite_loss", float(loss))
        return float(loss)

    return objective


def make_objective_v15_stepa_only(
    fold_np: List[Dict[str, Any]],
    n_rank: int,
    n_prof: int,
    tuning: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """융합 하이퍼 고정 없이 Step A(가이드) Train CV RMSE만 최소화 — 2단계 전용."""
    nr = max(1, int(n_rank))
    npv = max(1, int(n_prof))
    allow_knn = V15_ALLOW_KNN
    blend_opts = list(tuning.get("step_a_blend_grid", tuning["blend_grid"]))

    def objective(trial: optuna.Trial) -> float:
        calib_rank = trial.suggest_int("calibration_rank", 0, nr - 1)
        v3_idx = trial.suggest_int("v3_profile_idx", 0, npv - 1)
        blend_alpha = float(trial.suggest_categorical("blend_calib_alpha", blend_opts))
        knn_suggest = int(trial.suggest_categorical("knn_residual_neighbors", [0, 3, 6]))
        if not allow_knn or blend_alpha <= 1e-12 or blend_alpha >= 1.0 - 1e-12:
            knn_k = 0
        else:
            knn_k = knn_suggest
        rm = mean_cv_rmse_step_a_from_assembled(
            fold_np,
            calib_rank=int(calib_rank),
            v3_idx=int(v3_idx),
            blend_alpha=float(blend_alpha),
            knn_k=int(knn_k),
        )
        trial.set_user_attr("stepa_kw", dict(calib_rank=calib_rank, v3_idx=v3_idx, blend_alpha=blend_alpha, knn_k=knn_k))
        return float(rm)

    return objective


def make_objective_v15_stepc_fixed_blend(
    fold_np: List[Dict[str, Any]],
    lg: LocGeomPack,
    n_rank: int,
    n_prof: int,
    gate_lo: float,
    gate_hi: float,
    fixed: Dict[str, Any],
    tuning: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Step A 가이드(캘리브·V3·블렌드·KNN) 고정, 융합·IRLS만 탐색."""
    glo = float(min(gate_lo, gate_hi))
    ghi = float(max(gate_lo, gate_hi))
    br = int(fixed["calib_rank"])
    v3i = int(fixed["v3_idx"])
    ba = float(fixed["blend_alpha"])
    knn_k = int(fixed["knn_k"])
    inflate_lo = float(tuning["inflate_lo"])
    inflate_hi = float(tuning["inflate_hi"])
    fus_choices = [float(x) for x in tuning["huber_f_fusion_choices"]]
    irls_choices = [float(x) for x in tuning["irls_thresh_choices"]]

    def objective(trial: optuna.Trial) -> float:
        gate = trial.suggest_float("gate_threshold_m", glo, ghi)
        uvb_inf = trial.suggest_float("uwb_variance_inflate", inflate_lo, inflate_hi)
        hf_fus = float(trial.suggest_categorical("huber_f_fusion", fus_choices))
        ith = float(trial.suggest_categorical("irls_residual_thresh_m", irls_choices))
        blk = fold_np[0]["per_rank"][br]
        kw = dict(
            calib_rank=br,
            v3_idx=v3i,
            blend_alpha=ba,
            knn_k=knn_k,
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


def save_plots_v15(
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
        (xy_a[:, 0], xy_a[:, 1], "Step A (V15 블렌드+KNN)", rmse_a),
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
    fig.suptitle("V15 — Train-only priors / 검증 단일 Forward", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_dir / f"{ART_PREFIX}_map_true_vs_predicted_steps.png", dpi=PLOT_DPI)
    plt.close()


def save_cdf_v15(out_dir: Path, ea: np.ndarray, eb: np.ndarray, ec: np.ndarray, ra: float, rb: float, rc: float) -> None:
    plt.figure(figsize=(8.2, 5.0))
    for errs, lbl, rm in [(ea, "Step A", ra), (eb, "Step B", rb), (ec, "Step C", rc)]:
        e = np.sort(errs[np.isfinite(errs)])
        if e.size == 0:
            continue
        cdf = np.arange(1, len(e) + 1) / len(e)
        plt.plot(e, cdf, lw=2.0, label=f"{lbl} (RMSE {rm:.3f} m)")
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("V15 CDF (validation, single forward)")
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
    print(f"\n[V15] 학습 {len(train_df)}행 | 검증 {len(val_df)}행 | K-Fold={n_splits}", flush=True)

    train_imp, val_imp = wifi_median_impute_fusion(train_df, val_df, loc.wifi_cols)
    lg = build_loc_geom_pack(loc)
    tri_ref = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m=0.6, huber_f_scale=1.0)

    gate_lo_raw, gate_hi_raw, gate_meta = train_oof_gate_row_median_stats(train_imp, loc, n_splits)
    tuning = resolve_v15_tuning_space(V15_TUNING_PROFILE)
    gate_lo, gate_hi = apply_gate_tuning_caps(gate_lo_raw, gate_hi_raw, tuning)
    print(
        f"[V15] Train-OOF τ 후보(raw→적용): [{gate_lo_raw:.2f},{gate_hi_raw:.2f}]→[{gate_lo:.2f},{gate_hi:.2f}] m | meta={gate_meta}",
        flush=True,
    )
    print(
        f"[V15] 탐색 프로필={tuning['name']}: inflate∈[{tuning['inflate_lo']},{tuning['inflate_hi']}], "
        f"fusion Huber∈{tuning['huber_f_fusion_choices']}, IRLS(m)∈{tuning['irls_thresh_choices']}",
        flush=True,
    )

    v3_profiles = train_ranked_v3_profiles(train_imp, loc, n_splits, top_k=V15_TOP_V3)
    print(f"[V15] Train CV 상위 V3 프로필 {len(v3_profiles)}개: {v3_profiles[:5]}…", flush=True)

    catalog = v15_calibration_catalog()
    phase_a_rows, scored_full = rank_calibration_candidates(train_imp, loc, tri_ref, catalog, n_splits, top_k=None)
    raw_cap = V15_TOP_CALIB
    max_ranks = len(catalog) if raw_cap <= 0 else min(raw_cap, len(catalog))
    calib_ranks, calib_rank_tags = merge_calib_ranks_train_cv_only(catalog, scored_full, max_ranks=max_ranks)
    print(f"[V15] 캘리브 캐시 {len(calib_ranks)}개 (train 순수 Step A CV)", flush=True)

    fold_np = build_fold_num_pack(train_imp, loc, tri_ref, calib_ranks, v3_profiles, n_splits)
    warmup_numba_v15(lg, len(v3_profiles))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler_a = TPESampler(seed=OPTUNA_SEED, n_startup_trials=min(12, max(5, N_OPTUNA_TRIALS_STEPA // 3)))
    sampler_b = TPESampler(seed=OPTUNA_SEED + 17, n_startup_trials=min(18, max(6, N_OPTUNA_TRIALS // 4)))

    trials_rows: List[Dict[str, Any]] = []
    if V15_TWO_PHASE:
        print(
            f"[V15] 2단계: (1) Train CV Step A 최소화 {N_OPTUNA_TRIALS_STEPA} trials "
            f"(2) 가이드 고정 후 융합·IRLS {N_OPTUNA_TRIALS} trials",
            flush=True,
        )
        study_a = optuna.create_study(direction="minimize", sampler=sampler_a, study_name="v15_stepA")
        study_a.optimize(
            make_objective_v15_stepa_only(fold_np, len(calib_ranks), len(v3_profiles), tuning),
            n_trials=N_OPTUNA_TRIALS_STEPA,
            n_jobs=OPTUNA_N_JOBS,
            show_progress_bar=False,
        )
        bta = study_a.best_trial
        br = int(bta.params["calibration_rank"])
        v3i = int(bta.params["v3_profile_idx"])
        ba = float(bta.params["blend_calib_alpha"])
        knn_raw_a = int(bta.params.get("knn_residual_neighbors", 0))
        if not V15_ALLOW_KNN or ba <= 1e-12 or ba >= 1.0 - 1e-12:
            knn_k_a = 0
        else:
            knn_k_a = knn_raw_a
        fixed_guide = dict(calib_rank=br, v3_idx=v3i, blend_alpha=ba, knn_k=knn_k_a)
        best_cv_a = float(bta.value)
        for tr in study_a.trials:
            if tr.state != optuna.trial.TrialState.COMPLETE:
                continue
            trials_rows.append(
                {"phase": "stepA_only", "trial": tr.number, "mean_cv_rmse_step_a_m": float(tr.value), **tr.params}
            )

        study_b = optuna.create_study(direction="minimize", sampler=sampler_b, study_name="v15_stepC_fixed_guide")
        study_b.optimize(
            make_objective_v15_stepc_fixed_blend(
                fold_np, lg, len(calib_ranks), len(v3_profiles), gate_lo, gate_hi, fixed_guide, tuning
            ),
            n_trials=N_OPTUNA_TRIALS,
            n_jobs=OPTUNA_N_JOBS,
            show_progress_bar=False,
        )
        bt = study_b.best_trial
        gate = float(bt.params["gate_threshold_m"])
        uinf = float(bt.params["uwb_variance_inflate"])
        hf_fus = float(bt.params["huber_f_fusion"])
        ith = float(bt.params["irls_residual_thresh_m"])
        best_cv = float(bt.value)
        for tr in study_b.trials:
            if tr.state != optuna.trial.TrialState.COMPLETE:
                continue
            trials_rows.append(
                {
                    "phase": "stepC_fusion",
                    "trial": tr.number,
                    "mean_cv_rmse_step_c_m": float(tr.value),
                    **tr.params,
                    **{f"fixed_{k}": v for k, v in fixed_guide.items()},
                }
            )
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler_b)
        print(
            f"[V15] 단일 단계: 가중 Train CV 목표 w·StepA+(1-w)·StepC, w={V15_STEPA_LOSS_WEIGHT:.3f}",
            flush=True,
        )
        obj = make_objective_v15_composite(
            fold_np,
            lg,
            len(calib_ranks),
            len(v3_profiles),
            gate_lo,
            gate_hi,
            V15_STEPA_LOSS_WEIGHT,
            tuning,
        )
        study.optimize(obj, n_trials=N_OPTUNA_TRIALS, n_jobs=OPTUNA_N_JOBS, show_progress_bar=False)
        bt = study.best_trial
        br = int(bt.params["calibration_rank"])
        v3i = int(bt.params["v3_profile_idx"])
        ba = float(bt.params["blend_calib_alpha"])
        gate = float(bt.params["gate_threshold_m"])
        uinf = float(bt.params["uwb_variance_inflate"])
        hf_fus = float(bt.params["huber_f_fusion"])
        ith = float(bt.params["irls_residual_thresh_m"])
        best_cv = float(bt.user_attrs.get("mean_cv_rmse_step_c_m", bt.value))
        best_cv_a = float(bt.user_attrs["mean_cv_rmse_step_a_m"]) if "mean_cv_rmse_step_a_m" in bt.user_attrs else None
        for tr in study.trials:
            if tr.state != optuna.trial.TrialState.COMPLETE:
                continue
            row: Dict[str, Any] = {
                "phase": "composite_stepA_stepC",
                "trial": int(tr.number),
                "composite_loss": float(tr.value),
            }
            ua = tr.user_attrs or {}
            if "mean_cv_rmse_step_a_m" in ua:
                row["mean_cv_rmse_step_a_m"] = float(ua["mean_cv_rmse_step_a_m"])
            if "mean_cv_rmse_step_c_m" in ua:
                row["mean_cv_rmse_step_c_m"] = float(ua["mean_cv_rmse_step_c_m"])
            row.update({k: v for k, v in tr.params.items()})
            trials_rows.append(row)

    mx = 2
    gn_i = int(TRILAT_GN_ITERS)
    tukey_c = float(TUKEY_C)

    if V15_TWO_PHASE:
        knn_k = int(knn_k_a)
        trial_knn = int(knn_raw_a)
        best_cv_a_final = float(best_cv_a)
    else:
        knn_raw = int(bt.params.get("knn_residual_neighbors", 0))
        if not V15_ALLOW_KNN or ba <= 1e-12 or ba >= 1.0 - 1e-12:
            knn_k = 0
        else:
            knn_k = knn_raw
        trial_knn = knn_raw
        best_cv_a_final = float(best_cv_a) if best_cv_a is not None else None

    spec_f, feat_f, hf_wifi_f = calib_ranks[br]
    v3_hf, v3_wb = v3_profiles[v3i]

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
    print("V15 — Train-only priors + Train CV Optuna (검증 1회)")
    print("=" * 72)
    if best_cv_a_final is not None:
        print(f"  [Train CV] Step A (가이드 최적화 단계) mean RMSE = {best_cv_a_final:.4f} m", flush=True)
    print(f"  [Train CV] Step C mean RMSE = {best_cv:.4f} m", flush=True)
    print(f"  [Validation] Step A RMSE={rmse_a:.4f} | B={rmse_b:.4f} | C={rmse_c:.4f} m")
    print(f"  [Wall-clock] {elapsed:.2f} s")
    print("=" * 72)

    pref = resolve_paths(root)
    summ = {
        "pipeline": "V15",
        "integrity": (
            "No import of v12 validation-tuned grids. Gate bounds from train OOF row-median UWB geom residuals "
            "(with indoor τ upper cap). V3 profiles: train-ranked with hf<=1.35 plus anchors. "
            + (
                "Two-phase Optuna: Step A then Step C (train CV). "
                if V15_TWO_PHASE
                else f"Single Optuna: minimize w·TrainCV_StepA_RMSE+(1-w)·TrainCV_StepC_RMSE (w={V15_STEPA_LOSS_WEIGHT}). "
            )
            + "Validation labels used once post full-train refit."
        ),
        "two_phase": bool(V15_TWO_PHASE),
        "composite_stepa_weight": float(V15_STEPA_LOSS_WEIGHT),
        "tuning_profile": str(tuning["name"]),
        "tuning_space": {
            "gate_lo_floor": tuning.get("gate_lo_floor"),
            "gate_hi_cap": tuning.get("gate_hi_cap"),
            "inflate_lo": float(tuning["inflate_lo"]),
            "inflate_hi": float(tuning["inflate_hi"]),
            "huber_f_fusion_choices": [float(x) for x in tuning["huber_f_fusion_choices"]],
            "irls_thresh_choices": [float(x) for x in tuning["irls_thresh_choices"]],
        },
        "train_cv_best_step_a_rmse_m": best_cv_a_final,
        "train_oof_gate_raw_m": {"lo": float(gate_lo_raw), "hi": float(gate_hi_raw)},
        "train_oof_gate_bounds_m": {"lo": float(gate_lo), "hi": float(gate_hi), "stats": gate_meta},
        "v3_profiles_train_ranked": [{"huber_f_scale": h, "wifi_bias_m": w} for h, w in v3_profiles],
        "wall_clock_sec": float(elapsed),
        "train_kfold_splits": int(n_splits),
        "calibration_catalog_size": len(catalog),
        "top_calibration_ranks_cached": len(calib_ranks),
        "train_cv_best_step_c_rmse_m": float(best_cv),
        "optuna_n_trials_step_c": N_OPTUNA_TRIALS,
        "optuna_n_trials_step_a": (N_OPTUNA_TRIALS_STEPA if V15_TWO_PHASE else 0),
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
            "knn_residual_neighbors_trial": trial_knn,
            "knn_residual_neighbors": int(knn_k),
            "gate_m": gate,
            "uwb_variance_inflate": uinf,
            "huber_f_fusion": hf_fus,
            "irls_residual_thresh_m": ith,
            "irls_max_iter": mx,
            "trilat_gn_iterations_fixed": gn_i,
            "irls_tukey_c_fixed": tukey_c,
        },
        "validation_metrics": {
            "step_A_RMSE_m": float(rmse_a),
            "step_A_MAE_m": float(mae_a),
            "step_B_RMSE_m": float(rmse_b),
            "step_B_MAE_m": float(mae_b),
            "step_C_RMSE_m": float(rmse_c),
            "step_C_MAE_m": float(mae_c),
        },
        "targets_note": "Requested: val Step A and Step C ideally <= 1.6 m (not enforced in objective; report only).",
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

    pd.DataFrame(trials_rows).to_csv(out_dir / f"{ART_PREFIX}_optuna_trials.csv", index=False, encoding="utf-8-sig")

    coef_rows = [{"wifi_canon_ap": k, "A": v[0], "B": v[1]} for k, v in coeffs_f.items()]
    pd.DataFrame(coef_rows).to_csv(out_dir / f"{ART_PREFIX}_wifi_calibration_coefs.csv", index=False, encoding="utf-8-sig")

    if save_plots:
        save_plots_v15(out_dir, tx, ty, xy_a_val, preds_b, preds_c, rmse_a, rmse_b, rmse_c)
        save_cdf_v15(out_dir, ea, eb, ec, rmse_a, rmse_b, rmse_c)
    else:
        print("[V15] --no-plots: PNG 생략", flush=True)

    return 0


if __name__ == "__main__":
    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main, output_artifact_include_prefixes=("v15_",))
