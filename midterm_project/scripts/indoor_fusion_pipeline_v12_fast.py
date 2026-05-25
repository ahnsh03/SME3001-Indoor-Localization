"""
V12 Fast — V12/Strict와 동일한 **무결성**(Train K-Fold로만 하이퍼파라미터 선택,
검증 정답은 최종 단일 평가)을 유지하면서 **그리드·폴드·산출물**을 줄여 실행 시간을 단축한다.

- 원본 `indoor_fusion_pipeline_v12.py`: 검증 RMSE로 그리드 선택 → 데이터 누수.
- `indoor_fusion_pipeline_v12_strict.py`: 동일 수식·무결성이나 그리드가 커 느림.
- 본 스크립트: 축소 그리드 + K≤2 + Phase A 후보 2개 + 기본 **PNG 생략**(CSV/JSON 위주).

환경 변수 `V12_FAST_PLOTS=1` 이면 검증 오차 CDF PNG만 추가 저장한다(맵 3종은 생략).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fusion_realtime_sanitize import resolve_train_kgh_corrected_paths, resolve_validation_paths
from indoor_fusion_pipeline_v10 import WifiOnlyTrilateration, resolve_paths
from indoor_fusion_pipeline_v11 import CalibMethodSpec, fit_robust_calibration_per_ap
from indoor_fusion_pipeline_v12 import (
    augment_train_for_canon_fit,
    irls_refine_xyz,
    point_errors,
    predict_step_a_v3_wifi_equivalent,
    rmse_mae,
    row_fusion_gated_arrays,
    row_wifi_calibrated_arrays,
    solve_trilat_array,
    wifi_median_impute_fusion,
)
from indoor_fusion_pipeline_v12_strict import (
    build_phase_bc_fold_cache,
    cv_mean_rmse_v3_step_a,
    mean_cv_rmse_step_c_from_cache,
    resolve_kfold_n_splits,
)
from indoor_fusion_pipeline_v8 import WIFI_COL_CANON, Config, FusionLocalizerV8

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ART_PREFIX = "v12_fast"

# --- 축소 그리드 (strict 대비 조합 수 대폭 감소) --------------------------------
FEAT_MODES = ["bias_sub"]
HUBER_F_WIFI_FAST = [1.35]
HUBER_F_FUSION_FAST = [1.35]
GATE_THRESH_FAST = [1.55, 1.9, 2.35]
UWB_VAR_INFLATE_FAST = [24.0]
IRLS_THRESH_FAST = [2.2]
IRLS_MAX_ITER_FAST = [2]
TOP_A_FAST = 2
MAX_KFOLD_FAST = 2


def v12_fast_calibration_specs() -> List[CalibMethodSpec]:
    return [
        CalibMethodSpec("identity"),
        CalibMethodSpec("huber", huber_epsilon=1.35, huber_alpha=5e-4),
    ]


def main() -> int:
    t0 = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    train_dir = root / "data" / "train"
    val_dir = root / "data" / "validation"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    want_plots = os.environ.get("V12_FAST_PLOTS", "").strip() in ("1", "true", "True", "yes", "YES")

    tm, tv = resolve_train_kgh_corrected_paths(train_dir)
    vm, vd = resolve_validation_paths(val_dir)
    cfg = Config(grid_size_m=0.6, huber_f_scale=1.0)
    loc = FusionLocalizerV8(cfg)
    train_df, val_df = loc.load_datasets(tm, tv, vm, vd)
    n_splits = min(MAX_KFOLD_FAST, resolve_kfold_n_splits(len(train_df)))
    print(f"\n[V12 Fast] 학습 {len(train_df)}행 | 검증 {len(val_df)}행 | K-Fold={n_splits} (무결성)", flush=True)

    train_imp, val_imp = wifi_median_impute_fusion(train_df, val_df, loc.wifi_cols)
    tri_ref = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m=0.6, huber_f_scale=1.0)
    calib_specs = v12_fast_calibration_specs()

    cv_rmse_v3_a = cv_mean_rmse_v3_step_a(train_imp, loc, n_splits)
    phase_a_records: List[Dict[str, Any]] = []
    cand_a: List[Tuple[float, CalibMethodSpec, str, float]] = []
    for feat in FEAT_MODES:
        for spec in calib_specs:
            for hf in HUBER_F_WIFI_FAST:
                phase_a_records.append(
                    dict(
                        feat_mode=feat,
                        calib_kind=spec.kind,
                        huber_f_wifi=float(hf),
                        train_CV_mean_RMSE_A_m=float(cv_rmse_v3_a),
                        kfold_splits=int(n_splits),
                    )
                )
                cand_a.append((float(cv_rmse_v3_a), spec, feat, float(hf)))
    cand_a.sort(key=lambda t: (t[0], t[1].kind, t[2], t[3]))
    top_a = cand_a[:TOP_A_FAST]

    print("[V12 Fast] Train CV 캐시 + 축소 그리드로 Phase B/C 탐색…", flush=True)
    fold_cache = build_phase_bc_fold_cache(train_imp, loc, tri_ref, top_a, n_splits)

    best_cv_c = float("inf")
    best_sel: Optional[Dict[str, Any]] = None
    phase_bc_records: List[Dict[str, Any]] = []

    for rank, (cv_ra0, spec, feat, hf_wifi) in enumerate(top_a):
        for gate in GATE_THRESH_FAST:
            for uvb_inf in UWB_VAR_INFLATE_FAST:
                for hf_fus in HUBER_F_FUSION_FAST:
                    for ith in IRLS_THRESH_FAST:
                        for mx in IRLS_MAX_ITER_FAST:
                            cv_rc = mean_cv_rmse_step_c_from_cache(
                                fold_cache, loc, rank, float(gate), float(uvb_inf), float(hf_fus), float(ith), int(mx)
                            )
                            phase_bc_records.append(
                                dict(
                                    phaseA_rank=int(rank),
                                    feat_mode=feat,
                                    calib_kind=spec.kind,
                                    huber_f_wifi=float(hf_wifi),
                                    gate_threshold_m=float(gate),
                                    uwb_variance_inflate=float(uvb_inf),
                                    huber_f_fusion=float(hf_fus),
                                    irls_residual_thresh_m=float(ith),
                                    irls_max_iter=int(mx),
                                    train_CV_mean_RMSE_C_m=float(cv_rc),
                                )
                            )
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
        raise RuntimeError("V12 Fast: 그리드 탐색 실패")

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

    elapsed = time.perf_counter() - t0
    pref = resolve_paths(root)
    summ = {
        "pipeline": "V12 Fast",
        "integrity": (
            "Same as V12 Strict: HP from train K-fold CV only; validation labels once at end. "
            "Smaller grids and K<=2 for speed; plots optional via V12_FAST_PLOTS=1."
        ),
        "runtime_seconds": round(float(elapsed), 3),
        "train_kfold_splits": int(n_splits),
        "fast_grid": {
            "gate_threshold_m": GATE_THRESH_FAST,
            "uwb_variance_inflate": UWB_VAR_INFLATE_FAST,
            "huber_f_fusion": HUBER_F_FUSION_FAST,
            "irls_residual_threshold_m": IRLS_THRESH_FAST,
            "irls_max_iter": IRLS_MAX_ITER_FAST,
            "huber_f_wifi": HUBER_F_WIFI_FAST,
            "top_phase_a": TOP_A_FAST,
        },
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
            "huber_f_scale_wifi_only_trilateration": best_sel["hf_wifi"],
            "uwb_geometric_gate_threshold_m": best_sel["gate"],
            "uwb_variance_inflate_multiplier": best_sel["uwb_inf"],
            "huber_f_scale_fusion_trilateration": best_sel["hf_fus"],
            "irls_residual_threshold_m": best_sel["irls_thresh"],
            "irls_max_iterations": best_sel["irls_mx"],
            "phase_A_rank_selected": int(best_sel["phase_a_rank"]),
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

    if want_plots:
        import matplotlib.pyplot as plt

        plt.rcParams["axes.unicode_minus"] = False
        if sys.platform == "win32":
            try:
                plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
            except Exception:
                pass
        plt.figure(figsize=(8.2, 5.0))
        for lab, err, rm in [("Step A", ea, rmse_a), ("Step B", eb, rmse_b), ("Step C", ec, rmse_c)]:
            e = err[np.isfinite(err)]
            if e.size == 0:
                continue
            s = np.sort(e)
            cdf = np.arange(1, len(s) + 1) / len(s)
            plt.plot(s, cdf, lw=2.0, label=f"{lab} (RMSE {rm:.3f} m)")
        plt.xlabel("Position error (m)")
        plt.ylabel("CDF")
        plt.title("V12 Fast: validation CDF")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{ART_PREFIX}_cdf_steps.png", dpi=150)
        plt.close()

    print("\n" + "=" * 72)
    print(f"V12 Fast — 완료 ({elapsed:.1f}s) | Train CV만으로 HP | 검증 단일 평가")
    print("=" * 72)
    print(f"  [Train CV] Step C 평균 RMSE (선택) = {best_sel['train_cv_rmse_C']:.4f} m")
    print(f"  [Validation] Step A  RMSE={rmse_a:.4f} m  MAE={mae_a:.4f} m")
    print(f"  [Validation] Step B  RMSE={rmse_b:.4f} m  MAE={mae_b:.4f} m")
    print(f"  [Validation] Step C  RMSE={rmse_c:.4f} m  MAE={mae_c:.4f} m")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(
        Path(__file__),
        main,
        output_artifact_include_prefixes=("v12_fast_",),
        output_artifact_exclude_prefixes=("v12_fast2_",),
    )
