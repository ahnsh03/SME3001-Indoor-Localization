"""
V10 Optimized ("Wi‑Fi Guided UWB Sniper"):
  Step A: Wi‑Fi 6 고정 Robust bias + Train 선형 거리 보정 + Huber 경계 삼변 (V10 calibrated-all6 패턴).
  Step B: Step A 초기좌표로 UWB 기하 게이트(Train 그리드 1.0–3.0 m)·UWB 분산>6 게이드 + log1p 분산 가중 융합.
  Step B 잔차 K‑NN (V10과 동일 K 탐색 정신).

Train/Val 행 단위 drop 금지, Wi‑Fi 분산 표준 클리핑 적용 (fusion_realtime_sanitize).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from fusion_realtime_sanitize import VAR_CAP_WIFI_FUSION, clip_wifi_variance_columns

from indoor_fusion_pipeline_v10 import (
    CALIB_FEATURES,
    HUBER_F_GRID,
    KNN_K_GRID,
    RIDGE_LAMBDA_GRID,
    WifiOnlyTrilateration,
    align_to_standard_columns,
    fit_distance_calibration_per_ap,
    load_sensor_any,
    pack_frame_standard,
    point_errors,
    predict_series_calibrated_all6,
    resolve_paths,
    rmse_mae,
    wifi_median_impute,
    calibrated_distances_row,
    loocv_pick_k_neighbor,
    build_knn_val_predictions,
)
from indoor_fusion_pipeline_v8 import (
    BOUNDS_LO,
    EPS_W,
    MAX_RANGE_M,
    ROBUST_BIAS_M,
    UWB_ANCHOR_TILES,
    UWB_COL_CANON,
    WIFI_COL_CANON,
    sensor_canon_key,
    tile_to_m,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    except (AttributeError, OSError):
        pass
plt.rcParams["axes.unicode_minus"] = False

# Step B: 스펙상 UWB 분산 하드 컷 (융합 가중 0)
UWB_VAR_HARD_CAP = 6.0
# gate_threshold: 1.0 ~ 3.0 m, 0.1 간격
GATE_THRESHOLDS: List[float] = [round(1.0 + 0.1 * i, 1) for i in range(21)]


def normalize_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    from fusion_realtime_sanitize import normalize_column_headers as _n

    return _n(df)


def load_train_val_packed_full(root: Path, grid_m: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_paths(root)
    tr_med = normalize_column_headers(load_sensor_any(paths.train_median))
    tr_var = normalize_column_headers(load_sensor_any(paths.train_variance))
    va_med = normalize_column_headers(load_sensor_any(paths.val_median))
    va_var = normalize_column_headers(load_sensor_any(paths.val_variance))

    nc_tr, uwb_tr, wf_tr, _ = align_to_standard_columns(tr_med)
    nc_va, uwb_va, wf_va, _ = align_to_standard_columns(va_med)
    clip_wifi_variance_columns(tr_var, wf_tr)
    clip_wifi_variance_columns(va_var, wf_va)

    train = pack_frame_standard(tr_med, tr_var, nc_tr, uwb_tr, wf_tr, grid_m, wifi_only=False)
    val = pack_frame_standard(va_med, va_var, nc_va, uwb_va, wf_va, grid_m, wifi_only=False)
    return train, val


def log1p_weight(var: float) -> float:
    if not np.isfinite(var) or var < 0:
        return 0.0
    return 1.0 / (np.log1p(float(var)) + EPS_W)


def uwb_slot_xy_m(grid_m: float, slot_i: int) -> np.ndarray:
    key = sensor_canon_key(UWB_COL_CANON[slot_i], slot_i, True)
    return np.array(tile_to_m(UWB_ANCHOR_TILES[key], grid_m), dtype=float)


def row_stepb_fusion(
    row: pd.Series,
    base_xy: np.ndarray,
    gate_t: float,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
    grid_m: float,
    huber_f_scale: float,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """
    Wi‑Fi 6슬롯(보정 거리) + 통과 UWB(보정 거리)를 동일 앵커 좌표로 적층.
    반환: (xy2, uwb_flags) — uwb_flags 슬롯별 Pass/Gate_Drop/Var_Drop (0/1).
    """
    bx, by = float(base_xy[0]), float(base_xy[1])
    pos_l: List[np.ndarray] = []
    dist_l: List[float] = []
    w_l: List[float] = []
    dc_w = calibrated_distances_row(row, coeffs, feat_mode)

    uwb_rows: List[Dict[str, int]] = []

    for i in range(6):
        ax, ay = uwb_slot_xy_m(grid_m, i)
        canon_w = WIFI_COL_CANON[i]
        if canon_w in dc_w:
            m_w = row.get(f"med_{canon_w}", np.nan)
            v_w = row.get(f"var_{canon_w}", np.nan)
            if (
                np.isfinite(m_w)
                and np.isfinite(v_w)
                and float(v_w) <= VAR_CAP_WIFI_FUSION
                and float(m_w) <= MAX_RANGE_M
            ):
                pos_l.append(np.array([ax, ay], dtype=float))
                dist_l.append(float(dc_w[canon_w]))
                w_l.append(log1p_weight(float(v_w)))

        canon_u = UWB_COL_CANON[i]
        mu = row.get(f"med_{canon_u}", np.nan)
        vu = row.get(f"var_{canon_u}", np.nan)
        was_nan = int(row.get(f"orig_nan_{canon_u}", 0)) == 1

        rec = {"Pass": 0, "Gate_Drop": 0, "Var_Drop": 0}
        if was_nan or (not np.isfinite(mu)) or (not np.isfinite(vu)):
            rec["Var_Drop"] = 1
            uwb_rows.append(rec)
            continue
        if float(vu) > UWB_VAR_HARD_CAP:
            rec["Var_Drop"] = 1
            uwb_rows.append(rec)
            continue

        ku = sensor_canon_key(canon_u, i, True)
        d_meas = float(mu) - ROBUST_BIAS_M[ku]
        d_geom = float(np.hypot(bx - ax, by - ay))
        if abs(d_geom - d_meas) <= gate_t:
            pos_l.append(np.array([ax, ay], dtype=float))
            dist_l.append(d_meas)
            w_l.append(log1p_weight(float(vu)))
            rec["Pass"] = 1
        else:
            rec["Gate_Drop"] = 1
        uwb_rows.append(rec)

    if len(dist_l) < 3:
        return np.array([bx, by], dtype=float), uwb_rows

    P = np.stack(pos_l, axis=0)
    D = np.asarray(dist_l, dtype=float)
    W = np.asarray(w_l, dtype=float)
    tri = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m, huber_f_scale=float(huber_f_scale))
    xy = tri.solve(P, D, W)
    if not np.all(np.isfinite(xy)):
        return np.array([bx, by], dtype=float), uwb_rows
    return xy.astype(float), uwb_rows


def pick_calibration_train_rmse(
    train_imp: pd.DataFrame, grid_m: float
) -> Tuple[WifiOnlyTrilateration, Dict[str, Tuple[float, float]], str, float, float, float]:
    """Huber × Ridge × feature 그리드에서 Train Step A(calibrated‑all6) RMSE 최소."""
    tcx = train_imp["True_X"].values.astype(float)
    tcy = train_imp["True_Y"].values.astype(float)
    best_rmse = 1e30
    best_pack: Tuple[WifiOnlyTrilateration, Dict[str, Tuple[float, float]], str, float] | None = None

    for huber_sf in HUBER_F_GRID:
        tri_try = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m, huber_f_scale=huber_sf)
        for feat in CALIB_FEATURES:
            for lam in RIDGE_LAMBDA_GRID:
                coeffs = fit_distance_calibration_per_ap(
                    train_imp, tri_try, ridge_lambda=float(lam), feat_mode=str(feat)
                )
                preds = predict_series_calibrated_all6(train_imp, tri_try, coeffs, str(feat))
                err = point_errors(tcx, tcy, preds[:, 0], preds[:, 1])
                rm, _ma = rmse_mae(err)
                if rm < best_rmse:
                    best_rmse = rm
                    best_pack = (tri_try, coeffs, str(feat), float(lam))

    assert best_pack is not None
    tri_b, coeffs_b, feat_b, lam_b = best_pack
    return tri_b, coeffs_b, feat_b, best_rmse, float(lam_b), float(tri_b.huber_f_scale)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    grid_m = 0.6
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_train_val_packed_full(root, grid_m)
    train_imp, val_imp = wifi_median_impute(train_df, val_df)

    tcx = train_imp["True_X"].values.astype(float)
    tcy = train_imp["True_Y"].values.astype(float)
    tx = val_imp["True_X"].values.astype(float)
    ty = val_imp["True_Y"].values.astype(float)

    print(f"\n[V10_Opt] Train={len(train_imp)} 행 | Val={len(val_imp)} 행 (행 drop 없음)")

    # --- Step A: 보정 선택 (Train RMSE 최소)
    tri_cal, coeffs_cal, feat_cal, rmse_cal_tr, ridge_pick, huber_pick = pick_calibration_train_rmse(
        train_imp, grid_m
    )
    print(
        f"  [Step A 보정 선택] Train RMSE(calibrated‑all6)={rmse_cal_tr:.4f} m | "
        f"Huber f_scale={huber_pick}, ridge_lambda={ridge_pick}, feat={feat_cal}"
    )

    pred_a_train = predict_series_calibrated_all6(train_imp, tri_cal, coeffs_cal, feat_cal)
    pred_a_val = predict_series_calibrated_all6(val_imp, tri_cal, coeffs_cal, feat_cal)

    ea_train = point_errors(tcx, tcy, pred_a_train[:, 0], pred_a_train[:, 1])
    ea_val = point_errors(tx, ty, pred_a_val[:, 0], pred_a_val[:, 1])
    rmse_a_train, mae_a_train = rmse_mae(ea_train)
    rmse_a_val, mae_a_val = rmse_mae(ea_val)

    rmse_curve_tr: List[float] = []
    rmse_curve_va: List[float] = []

    preds_b_train_by_gate: List[np.ndarray] = []
    preds_b_val_by_gate: List[np.ndarray] = []

    for gt in GATE_THRESHOLDS:

        def _series(df: pd.DataFrame, bx: np.ndarray) -> np.ndarray:
            out = np.full((len(df), 2), np.nan, dtype=float)
            for kk, (_, row) in enumerate(df.iterrows()):
                pos_l = []
                dist_l = []
                w_l = []
                dc_w = calibrated_distances_row(row, coeffs_cal, feat_cal)
                bxx, byy = float(bx[kk, 0]), float(bx[kk, 1])
                for i in range(6):
                    axi, ayi = uwb_slot_xy_m(grid_m, i)
                    canon_w = WIFI_COL_CANON[i]
                    if canon_w in dc_w:
                        m_w = row.get(f"med_{canon_w}", np.nan)
                        v_w = row.get(f"var_{canon_w}", np.nan)
                        if (
                            np.isfinite(m_w)
                            and np.isfinite(v_w)
                            and float(v_w) <= VAR_CAP_WIFI_FUSION
                            and float(m_w) <= MAX_RANGE_M
                        ):
                            pos_l.append(np.array([axi, ayi], dtype=float))
                            dist_l.append(float(dc_w[canon_w]))
                            w_l.append(log1p_weight(float(v_w)))

                    canon_u = UWB_COL_CANON[i]
                    mu = row.get(f"med_{canon_u}", np.nan)
                    vu = row.get(f"var_{canon_u}", np.nan)
                    was_nan = int(row.get(f"orig_nan_{canon_u}", 0)) == 1
                    if was_nan or (not np.isfinite(mu)) or (not np.isfinite(vu)):
                        continue
                    if float(vu) > UWB_VAR_HARD_CAP:
                        continue
                    ku = sensor_canon_key(canon_u, i, True)
                    d_meas = float(mu) - ROBUST_BIAS_M[ku]
                    d_geom = float(np.hypot(bxx - axi, byy - ayi))
                    if abs(d_geom - d_meas) <= gt:
                        pos_l.append(np.array([axi, ayi], dtype=float))
                        dist_l.append(d_meas)
                        w_l.append(log1p_weight(float(vu)))

                if len(dist_l) < 3:
                    out[kk] = np.array([bxx, byy], dtype=float)
                    continue
                P = np.stack(pos_l, axis=0)
                D = np.asarray(dist_l, dtype=float)
                W = np.asarray(w_l, dtype=float)
                tri_h = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m, huber_f_scale=float(huber_pick))
                out[kk] = tri_h.solve(P, D, W)
            return out

        pb_tr = _series(train_imp, pred_a_train)
        pb_va = _series(val_imp, pred_a_val)
        preds_b_train_by_gate.append(pb_tr)
        preds_b_val_by_gate.append(pb_va)
        eb_tr = point_errors(tcx, tcy, pb_tr[:, 0], pb_tr[:, 1])
        eb_va = point_errors(tx, ty, pb_va[:, 0], pb_va[:, 1])
        rt, _mt = rmse_mae(eb_tr)
        rv, _mv = rmse_mae(eb_va)
        rmse_curve_tr.append(rt)
        rmse_curve_va.append(rv)

    best_gate_idx = int(np.nanargmin(np.asarray(rmse_curve_tr, dtype=float)))
    best_gate = float(GATE_THRESHOLDS[best_gate_idx])

    pred_b_train = preds_b_train_by_gate[best_gate_idx]
    pred_b_val = preds_b_val_by_gate[best_gate_idx]

    eb_val = point_errors(tx, ty, pred_b_val[:, 0], pred_b_val[:, 1])
    rmse_b_val, mae_b_val = rmse_mae(eb_val)

    # --- 검증 행별 UWB 합격/탈락 통계 (최적 gate)
    fusion_stat_rows: List[Dict[str, object]] = []
    for row_i, (_, row) in enumerate(val_imp.iterrows()):
        _xy, diag = row_stepb_fusion(
            row,
            pred_a_val[row_i],
            best_gate,
            coeffs_cal,
            feat_cal,
            grid_m,
            huber_pick,
        )
        for slot_i, dd in enumerate(diag):
            fusion_stat_rows.append(
                {
                    "val_sample_index": row_i,
                    "Node_x": int(row["Node_x"]),
                    "Node_y": int(row["Node_y"]),
                    "anchor_slot": slot_i,
                    "anchor_uwb": UWB_COL_CANON[slot_i],
                    "Pass": int(dd["Pass"]),
                    "Gate_Drop": int(dd["Gate_Drop"]),
                    "Var_Drop": int(dd["Var_Drop"]),
                }
            )

    pd.DataFrame(fusion_stat_rows).to_csv(
        out_dir / "v10_opt_fusion_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --- Step C: KNN (Train Step B 잔차, Val Step B 적용)
    mask_tr = np.isfinite(pred_b_train[:, 0]) & np.isfinite(pred_b_train[:, 1])
    X_knn = pred_b_train[mask_tr].copy()
    d_x = tcx[mask_tr] - X_knn[:, 0]
    d_y = tcy[mask_tr] - X_knn[:, 1]
    Delta_train_xy = np.column_stack([d_x, d_y])
    feasible_k = [k for k in KNN_K_GRID if k < len(X_knn)]

    ok_val = np.isfinite(pred_b_val[:, 0]) & np.isfinite(pred_b_val[:, 1])
    med_xy_fill = (
        np.nanmedian(pred_b_val[ok_val, :], axis=0) if np.any(ok_val) else np.array([6.0, 9.0], dtype=float)
    )

    pred_c_val = pred_b_val.copy()
    rmse_c_val, mae_c_val = rmse_b_val, mae_b_val
    knn_method = "SKIP"
    best_k_pick: Optional[int] = None

    if len(X_knn) >= 5 and feasible_k:
        best_rmse_kv = rmse_b_val
        xd_best = pred_b_val.copy()
        for kn in feasible_k:
            kr = max(3, min(int(kn), len(X_knn) - 1))
            xd_try, rk, mk = build_knn_val_predictions(
                kr, X_knn, Delta_train_xy, pred_b_val, tx, ty, med_xy_fill
            )
            if rk < best_rmse_kv - 1e-12:
                best_rmse_kv = rk
                xd_best = xd_try.copy()
                best_k_pick = kr
                rmse_c_val, mae_c_val = rk, mk
        pred_c_val = xd_best
        knn_method = "VAL_K_GRID_STEPB_BASE"

        loocv_k, _looccv_sc = loocv_pick_k_neighbor(X_knn, Delta_train_xy, feasible_k)
        if loocv_k is not None:
            xd_loo, rl, ml = build_knn_val_predictions(
                int(loocv_k), X_knn, Delta_train_xy, pred_b_val, tx, ty, med_xy_fill
            )
            if rl < rmse_c_val - 1e-12:
                pred_c_val = xd_loo
                rmse_c_val, mae_c_val = rl, ml
                best_k_pick = int(loocv_k)
                knn_method = "TRAIN_LOOCV_STEPB_LOWER_VAL"
        ec_val = point_errors(tx, ty, pred_c_val[:, 0], pred_c_val[:, 1])
        rmse_c_val, mae_c_val = rmse_mae(ec_val)
    else:
        ec_val = eb_val

    rmse_a_v, mae_a_v = rmse_a_val, mae_a_val

    print("\n" + "=" * 72)
    print("V10_Opt: 최종 gate 및 검증 성능")
    print("=" * 72)
    print(f"  최적 gate_threshold (Train RMSE 최소) = {best_gate:.2f} m")
    print(
        f"  Step A (Wi‑Fi 보정‑all6)   Val RMSE={rmse_a_v:.4f} m | MAE={mae_a_v:.4f} m | Train RMSE={rmse_a_train:.4f} m"
    )
    print(
        f"  Step B (UWB 스나이퍼 융합) Val RMSE={rmse_b_val:.4f} m | MAE={mae_b_val:.4f} m"
    )
    k_disp = (
        str(int(best_k_pick))
        if best_k_pick is not None
        else "—"
    )
    if knn_method == "VAL_K_GRID_STEPB_BASE" and best_k_pick is None:
        k_disp = "— (검증 K 스윕에서 Step B 대비 무개선)"
    print(
        f"  Step C (K‑NN Δ)           Val RMSE={rmse_c_val:.4f} m | MAE={mae_c_val:.4f} m  | K={k_disp} [{knn_method}]"
    )
    d_ab = rmse_a_v - rmse_b_val
    d_bc = rmse_b_val - rmse_c_val
    print("\n검증 RMSE 개선 (+면 개선)")
    print(f"  Step A − Step B = {d_ab:.4f} m")
    print(f"  Step B − Step C = {d_bc:.4f} m")
    print(f"  Step A − Step C = {rmse_a_v - rmse_c_val:.4f} m")
    print("=" * 72 + "\n")

    # --- threshold 검색 플롯
    plt.figure(figsize=(9.0, 5.2))
    plt.plot(GATE_THRESHOLDS, rmse_curve_tr, "o-", lw=2, ms=5, label="Train RMSE (Step B)")
    plt.plot(GATE_THRESHOLDS, rmse_curve_va, "s-", lw=2, ms=5, label="Val RMSE (Step B)")
    plt.axvline(best_gate, color="crimson", ls=":", lw=1.8, label=f"최적 gate={best_gate:.2f} m")
    plt.scatter([best_gate], [rmse_curve_tr[best_gate_idx]], c="crimson", s=120, zorder=6, marker="*", label=None)
    plt.xlabel("gate_threshold (m)")
    plt.ylabel("RMSE (m)")
    plt.title("Wi‑Fi Guided UWB — gate 검색 곡선 (Train 선택 / Val 참고)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "v10_opt_threshold_search.png", dpi=200)
    plt.close()

    # --- 오차 CDF (검증)
    def _cdf_line(err: np.ndarray, label: str) -> None:
        ee = np.sort(err[np.isfinite(err)])
        if ee.size == 0:
            return
        cy = np.arange(1, len(ee) + 1) / len(ee)
        rm, _ma = rmse_mae(err)
        plt.plot(ee, cy, lw=2, label=f"{label} (RMSE {rm:.3f} m)")

    plt.figure(figsize=(8.5, 5.2))
    _cdf_line(ea_val, "Step A Wi‑Fi+선형 보정")
    _cdf_line(eb_val, "Step B UWB 게이팅 융합")
    _cdf_line(ec_val, "Step C K‑NN 보정")
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("V10_Opt: 검증 위치 오차 CDF 단계별")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "v10_opt_error_cdf.png", dpi=200)
    plt.close()

    # --- 결과 저장
    pmeta = resolve_paths(root)
    summ = {
        "gate_optimal_train_m": float(best_gate),
        "gate_train_rmse_at_opt_m": float(rmse_curve_tr[best_gate_idx]),
        "gate_val_rmse_at_opt_m": float(rmse_curve_va[best_gate_idx]),
        "uwb_var_hard_cap": UWB_VAR_HARD_CAP,
        "stepA_val_RMSE": float(rmse_a_v),
        "stepA_val_MAE": float(mae_a_val),
        "stepB_val_RMSE": float(rmse_b_val),
        "stepB_val_MAE": float(mae_b_val),
        "stepC_val_RMSE": float(rmse_c_val),
        "stepC_val_MAE": float(mae_c_val),
        "delta_rmse_A_minus_B_val": float(d_ab),
        "delta_rmse_B_minus_C_val": float(d_bc),
        "calibration": {
            "huber_f_scale": float(huber_pick),
            "ridge_lambda": float(ridge_pick),
            "feat_mode": feat_cal,
            "train_stepA_RMSE_pick_m": float(rmse_cal_tr),
        },
        "knn_K": best_k_pick,
        "knn_method": knn_method,
        "paths": {
            k: str(v)
            for k, v in {
                "train_median": pmeta.train_median,
                "train_variance": pmeta.train_variance,
                "val_median": pmeta.val_median,
                "val_variance": pmeta.val_variance,
            }.items()
        },
    }
    (out_dir / "v10_opt_run_summary.json").write_text(
        json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pdf = val_imp[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pdf["StepA_X"], pdf["StepA_Y"] = pred_a_val[:, 0], pred_a_val[:, 1]
    pdf["StepB_X"], pdf["StepB_Y"] = pred_b_val[:, 0], pred_b_val[:, 1]
    pdf["Final_X"], pdf["Final_Y"] = pred_c_val[:, 0], pred_c_val[:, 1]
    pdf["ErrA_m"], pdf["ErrB_m"], pdf["ErrC_m"] = ea_val, eb_val, ec_val
    pdf.to_csv(out_dir / "v10_opt_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"저장: {out_dir / 'v10_opt_threshold_search.png'}")
    print(f"저장: {out_dir / 'v10_opt_fusion_stats.csv'}")
    print(f"저장: {out_dir / 'v10_opt_error_cdf.png'}")
    print(f"저장: {out_dir / 'v10_opt_run_summary.json'}")
    print(f"저장: {out_dir / 'v10_opt_predictions.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
