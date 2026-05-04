"""
V11: Pure Wi‑Fi — 강건 거리 보정(Huber/RANSAC) + IRLS 가중 삼변 + 다중피처 K‑NN.

UWB 미사용. Train=`median_test_kgh_corrected` / 검증=validation. 그리드는 검증 RMSE 기준.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from fusion_realtime_sanitize import VAR_CAP_WIFI_FUSION

from indoor_fusion_pipeline_v10 import (
    CALIB_FEATURES,
    KNN_K_GRID,
    WifiOnlyTrilateration,
    calibrated_distances_row,
    load_train_val_packed,
    point_errors,
    predict_step_bias_only_series,
    resolve_paths,
    ridge_linear_2d,
    rmse_mae,
    row_calibrated_arrays,
    weighted_range_rms_residual,
    wifi_median_impute,
)
from indoor_fusion_pipeline_v8 import (
    MAX_RANGE_M,
    ROBUST_BIAS_M,
    WIFI_COL_CANON,
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

# --- Grids (검증 홀드아웃 · iterrows 삼변 비용 고려 소형 세트)
HUBER_F_GRID = [0.95, 1.0, 1.35]
HUBER_CALIB_EPS_GRID = [1.2, 1.45]
HUBER_CALIB_ALPHA_GRID = [1e-5, 5e-4]
RANSAC_RESID_THRESH_GRID = [3.5, 6.0]

IRLS_THRESH_GRID_M = [1.4, 1.75, 2.05, 2.4, 2.85, 3.35, 3.95]
IRLS_MAX_ITER_GRID = [2, 3]

TUKEY_C = 4.685


@dataclass
class CalibMethodSpec:
    kind: str
    huber_epsilon: float = 1.35
    huber_alpha: float = 1e-4
    ransac_residual_threshold: float = 4.0


def _identity_calibration_coeffs(*, feat_mode: str) -> Dict[str, Tuple[float, float]]:
    """보정 호모그래피 없음(A=1, B 선택) — Robust bias 피처와 동일 거리 매핑으로 Step A 대비 과도 악화 방지."""
    out: Dict[str, Tuple[float, float]] = {}
    for wi, canon in enumerate(WIFI_COL_CANON):
        key = sensor_canon_key(canon, wi, False)
        if feat_mode == "raw":
            out[canon] = (1.0, float(-ROBUST_BIAS_M.get(key, 0.0)))
        else:
            out[canon] = (1.0, 0.0)
    return out


def _clip_ab(a: float, b: float) -> Tuple[float, float]:
    return float(np.clip(a, 0.3, 2.6)), float(np.clip(b, -5.0, 5.5))


def iter_calib_method_specs() -> List[CalibMethodSpec]:
    out: List[CalibMethodSpec] = []
    out.append(CalibMethodSpec("identity"))
    for e in HUBER_CALIB_EPS_GRID:
        for a in HUBER_CALIB_ALPHA_GRID:
            out.append(CalibMethodSpec("huber", huber_epsilon=e, huber_alpha=a))
    for rt in RANSAC_RESID_THRESH_GRID:
        out.append(CalibMethodSpec("ransac", ransac_residual_threshold=rt))
    return out


from indoor_fusion_pipeline_v8 import EPS_W, WIFI_BIAS_W  # noqa: E402


def row_calibrated_pos_dist_weight_var_v10(
    row: pd.Series,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_l: List[np.ndarray] = []
    dist_l: List[float] = []
    weight_l: List[float] = []
    var_l: List[float] = []
    dcals = calibrated_distances_row(row, coeffs, feat_mode)
    for wi, canon in enumerate(WIFI_COL_CANON):
        m_raw = row.get(f"med_{canon}", np.nan)
        vv = row.get(f"var_{canon}", np.nan)
        if canon not in dcals:
            continue
        if (not np.isfinite(m_raw)) or (not np.isfinite(vv)) or float(vv) > VAR_CAP_WIFI_FUSION:
            continue
        pos_l.append(tri.wifi_xy_m[wi])
        dist_l.append(dcals[canon])
        weight_l.append(1.0 / (float(vv) + WIFI_BIAS_W + EPS_W))
        var_l.append(float(vv))
    if not dist_l:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0)
    return (
        np.array(pos_l, dtype=float),
        np.array(dist_l, dtype=float),
        np.array(weight_l, dtype=float),
        np.array(var_l, dtype=float),
    )


def fit_robust_calibration_per_ap(
    train_df: pd.DataFrame,
    tri: WifiOnlyTrilateration,
    *,
    feat_mode: str,
    spec: CalibMethodSpec,
) -> Dict[str, Tuple[float, float]]:
    if spec.kind == "identity":
        return _identity_calibration_coeffs(feat_mode=str(feat_mode))

    coeffs: Dict[str, Tuple[float, float]] = {}
    for wi, canon in enumerate(WIFI_COL_CANON):
        ax, ay = tri.wifi_xy_m[wi, 0], tri.wifi_xy_m[wi, 1]
        key = sensor_canon_key(canon, wi, False)
        feats: List[float] = []
        geoms: List[float] = []
        for _, row in train_df.iterrows():
            m_raw = row.get(f"med_{canon}", np.nan)
            vv = row.get(f"var_{canon}", np.nan)
            if not np.isfinite(m_raw) or not np.isfinite(vv):
                continue
            if float(vv) > VAR_CAP_WIFI_FUSION or float(m_raw) > MAX_RANGE_M:
                continue
            tx, ty = float(row["True_X"]), float(row["True_Y"])
            d_geom = float(np.hypot(tx - ax, ty - ay))
            if feat_mode == "raw":
                x_feat = float(m_raw)
            else:
                x_feat = float(m_raw) - ROBUST_BIAS_M[key]
            feats.append(x_feat)
            geoms.append(d_geom)

        if len(feats) < 10:
            if feat_mode == "raw":
                coeffs[canon] = (1.0, float(-ROBUST_BIAS_M.get(key, 0.0)))
            else:
                coeffs[canon] = (1.0, 0.0)
            continue

        X1 = np.asarray(feats, dtype=float).reshape(-1, 1)
        y = np.asarray(geoms, dtype=float)

        if spec.kind == "huber":
            hr = HuberRegressor(
                epsilon=float(spec.huber_epsilon),
                alpha=float(spec.huber_alpha),
                fit_intercept=True,
                max_iter=300,
            )
            try:
                hr.fit(X1, y)
                a, b = float(hr.coef_[0]), float(hr.intercept_)
            except ValueError:
                X2 = np.column_stack([X1.ravel(), np.ones(len(X1))])
                a, b = ridge_linear_2d(X2, y, lam=8.0)
            coeffs[canon] = _clip_ab(a, b)
        else:
            # RANSAC + LinearRegression
            min_s = max(2, min(len(feats) // 2, len(feats) - 1))
            try:
                base = LinearRegression()
                try:
                    rs = RANSACRegressor(
                        estimator=base,
                        min_samples=min_s,
                        residual_threshold=float(spec.ransac_residual_threshold),
                        max_trials=min(800, 50 * len(feats)),
                        random_state=0,
                    )
                except TypeError:
                    rs = RANSACRegressor(
                        base_estimator=base,
                        min_samples=min_s,
                        residual_threshold=float(spec.ransac_residual_threshold),
                        max_trials=min(800, 50 * len(feats)),
                        random_state=0,
                    )
                rs.fit(X1, y)
                est = rs.estimator_
                a, b = float(est.coef_[0]), float(est.intercept_)
            except Exception:
                X2 = np.column_stack([X1.ravel(), np.ones(len(X1))])
                a, b = ridge_linear_2d(X2, y, lam=16.0)
            coeffs[canon] = _clip_ab(a, b)
    return coeffs


def predict_series_calibrated_all6(
    df: pd.DataFrame,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
) -> np.ndarray:
    n = len(df)
    xy = np.full((n, 2), np.nan, dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        p, d, w = row_calibrated_arrays(row, tri, coeffs, feat_mode)
        xy[i] = tri.solve(p, d, w)
    return xy


def tukey_bisquare_weights(u: np.ndarray, c: float = TUKEY_C) -> np.ndarray:
    uu = np.abs(u.astype(float))
    w = np.zeros_like(uu, dtype=float)
    m = uu < c
    w[m] = (1.0 - (uu[m] / c) ** 2) ** 2
    return w


def huber_hinge_weights(r: np.ndarray, delta: float) -> np.ndarray:
    rr = np.abs(r.astype(float))
    d = max(float(delta), 1e-6)
    return np.minimum(1.0, d / np.maximum(rr, 1e-6))


def irls_solve_row(
    row: pd.Series,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
    thresh_m: float,
    max_iters: int,
) -> Tuple[np.ndarray, np.ndarray, float, int, float]:
    """
    반환: (xy, w_final, geom_weighted_rms, n_ap, mean_var_used)
    """
    p, d, w_base, vars_u = row_calibrated_pos_dist_weight_var_v10(row, tri, coeffs, feat_mode)
    n_ap = len(d)
    mean_v = float(np.mean(vars_u)) if n_ap else float("nan")
    if n_ap < 3:
        xy = tri.solve(p, d, w_base)
        if n_ap >= 1 and np.all(np.isfinite(xy)):
            qr = weighted_range_rms_residual(xy.astype(float), p, d, w_base)
        else:
            qr = float("nan")
        return xy.astype(float), w_base, qr, n_ap, mean_v

    w = np.asarray(w_base, dtype=float).copy()
    xy_last = np.full(2, np.nan)

    for _ in range(max(1, int(max_iters))):
        xy_last = tri.solve(p, d, w)
        if not np.all(np.isfinite(xy_last)):
            break
        gpred = np.hypot(xy_last[0] - p[:, 0], xy_last[1] - p[:, 1])
        r = np.abs(gpred - d)
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        sigma_rob = max(mad * 1.4826, 0.08)
        sigma = float(max(thresh_m * 0.65, sigma_rob, 1e-2))
        u = r / sigma
        tw = tukey_bisquare_weights(u, c=TUKEY_C)
        hw = huber_hinge_weights(r, float(thresh_m))
        mul = np.clip(tw * hw, 0.04, 1.0)
        w_new = np.asarray(w_base, dtype=float) * mul
        if float(np.linalg.norm(w_new - w)) / (float(np.linalg.norm(w)) + 1e-9) < 1e-4:
            w = w_new
            xy_last = tri.solve(p, d, w)
            break
        w = w_new

    qr = weighted_range_rms_residual(xy_last.astype(float), p, d, w) if np.all(
        np.isfinite(xy_last)
    ) else float("nan")
    return xy_last.astype(float), w, float(qr), n_ap, mean_v


def predict_series_irls_with_meta(
    df: pd.DataFrame,
    tri: WifiOnlyTrilateration,
    coeffs: Dict[str, Tuple[float, float]],
    feat_mode: str,
    thresh_m: float,
    max_iters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(xy_stepC, geom_rms_residual, mean_wifi_variance_row, nap)."""
    n = len(df)
    xy = np.full((n, 2), np.nan)
    qrms = np.full(n, np.nan)
    mvw = np.full(n, np.nan)
    nap = np.zeros(n, dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        xyi, _wf, qr, na, mv = irls_solve_row(row, tri, coeffs, feat_mode, thresh_m, max_iters)
        xy[i] = xyi
        qrms[i] = qr
        nap[i] = na
        mvw[i] = mv
    return xy, qrms, mvw, nap


def build_multi_feature_matrix(
    xy_stepc: np.ndarray,
    geom_rms: np.ndarray,
    mean_var: np.ndarray,
    nap: np.ndarray,
    *,
    use_log_var: bool,
) -> np.ndarray:
    """[X, Y, geom_rms, mean_var or log1p(mean_var), n_ap_norm] — 최소 4차원, 옵션 5~6."""
    xv = xy_stepc[:, 0]
    yv = xy_stepc[:, 1]
    gr = geom_rms.copy()
    gr[~np.isfinite(gr)] = np.nanmedian(gr[np.isfinite(gr)]) if np.any(np.isfinite(gr)) else 2.5
    mv = mean_var.copy()
    mv[~np.isfinite(mv)] = np.nanmedian(mv[np.isfinite(mv)]) if np.any(np.isfinite(mv)) else 1.0
    if use_log_var:
        mvf = np.log1p(np.clip(mv, 0.0, None))
    else:
        mvf = mv
    nfrac = nap.astype(float) / 6.0
    return np.column_stack([xv, yv, gr, mvf, nfrac])


def build_knn_predictions_multi_scaled(
    K_use: Optional[int],
    X_train_feat: np.ndarray,
    Delta_train: np.ndarray,
    X_val_feat: np.ndarray,
    preds_c_val_xy: np.ndarray,
    tx_val: np.ndarray,
    ty_val: np.ndarray,
    fill_xy: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    if K_use is None:
        e = point_errors(tx_val, ty_val, preds_c_val_xy[:, 0], preds_c_val_xy[:, 1])
        return preds_c_val_xy.copy(), *rmse_mae(e)

    kk = max(3, min(int(K_use), len(X_train_feat) - 1))
    sc = StandardScaler()
    Xtr = sc.fit_transform(np.nan_to_num(X_train_feat, nan=0.0))
    Xva = sc.transform(np.nan_to_num(X_val_feat, nan=0.0))

    knn = KNeighborsRegressor(n_neighbors=kk, weights="distance")
    knn.fit(Xtr, Delta_train)

    xv = preds_c_val_xy.copy().astype(float)
    nan_xy = ~(np.isfinite(xv[:, 0]) & np.isfinite(xv[:, 1]))
    for j in np.where(nan_xy)[0]:
        xv[j, :] = fill_xy

    dlt = knn.predict(Xva)
    xd = xv + dlt
    ee = point_errors(tx_val, ty_val, xd[:, 0], xd[:, 1])
    rr, mm = rmse_mae(ee)
    return xd, rr, mm


def loocv_pick_k_multi(
    X: np.ndarray, Delta: np.ndarray, feasible: List[int]
) -> Tuple[Optional[int], float]:
    """다차원 피처 + 스케일러 LOOCV."""
    n = len(X)
    if n < 6:
        return None, float("nan")
    best_k: Optional[int] = None
    best_score = float("inf")
    for kn in feasible:
        kr = int(max(3, min(kn, n - 3)))
        err_acc: List[float] = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            xs, ds = X[mask], Delta[mask]
            kk_use = max(3, min(kr, len(xs) - 1))
            if kk_use >= len(xs):
                continue
            sc = StandardScaler()
            xtr = sc.fit_transform(np.nan_to_num(xs, nan=0.0))
            x_i = sc.transform(np.nan_to_num(X[i : i + 1], nan=0.0))
            try:
                knn = KNeighborsRegressor(n_neighbors=kk_use, weights="distance")
                knn.fit(xtr, ds)
                pr = knn.predict(x_i)[0]
                err_acc.append(float(np.linalg.norm(pr - Delta[i])))
            except ValueError:
                continue
        if len(err_acc) != n:
            continue
        sc_rmse = float(np.sqrt(np.mean(np.square(np.asarray(err_acc, dtype=float)))))
        if sc_rmse < best_score:
            best_score = sc_rmse
            best_k = kr
    return best_k, best_score


def save_v11_cdf_steps(
    out_dir: Path,
    errs_by_step: Dict[str, np.ndarray],
    lbl: Dict[str, str],
    rmse_by_step: Dict[str, float],
) -> None:
    plt.figure(figsize=(8.8, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(errs_by_step)))
    keys = sorted(errs_by_step.keys())
    for ki, key in enumerate(keys):
        ee = errs_by_step[key][np.isfinite(errs_by_step[key])]
        if ee.size == 0:
            continue
        ee = np.sort(ee)
        cdf = np.arange(1, len(ee) + 1) / len(ee)
        plt.plot(ee, cdf, lw=2.0, label=f"{lbl.get(key, key)} (RMSE {rmse_by_step[key]:.3f} m)", color=colors[ki])
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("V11: 검증 세트 단계별 위치 오차 CDF (Robust calib · IRLS · Multi‑KNN)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "v11_cdf_steps.png", dpi=200)
    plt.close()


def predict_series_bias_xy(df: pd.DataFrame, tri: WifiOnlyTrilateration) -> np.ndarray:
    return predict_step_bias_only_series(df, tri)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    grid_m = 0.6
    train_df, val_df = load_train_val_packed(root, grid_m)
    tx = val_df["True_X"].values.astype(float)
    ty = val_df["True_Y"].values.astype(float)
    tcx = train_df["True_X"].values.astype(float)
    tcy = train_df["True_Y"].values.astype(float)

    train_imp, val_imp = wifi_median_impute(train_df, val_df)
    print("\n[V11] 학습행:", len(train_imp), "| 검증행:", len(val_imp))

    # Step A baseline
    tri_a = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m, huber_f_scale=1.0)
    xa_pred = predict_series_bias_xy(val_imp, tri_a)
    ea = point_errors(tx, ty, xa_pred[:, 0], xa_pred[:, 1])
    rmse_a, mae_a = rmse_mae(ea)

    # --- Stage B: 검증 세트에서「강건 보정 후 6AP 삼변」RMSE 단독 최소화 --------
    best_b: Dict[str, Any] = {
        "rmse_b": float("inf"),
        "huber_f": 1.0,
        "feat": "bias_sub",
        "calib_spec": CalibMethodSpec("huber"),
        "coeffs": None,
        "tri": None,
    }

    specs = iter_calib_method_specs()
    cand_b_log: List[Dict[str, Any]] = []

    for huber_sf in HUBER_F_GRID:
        tri_try = WifiOnlyTrilateration(WIFI_COL_CANON, grid_m, huber_f_scale=float(huber_sf))
        for feat in CALIB_FEATURES:
            for sp in specs:
                coeffs_try = fit_robust_calibration_per_ap(train_imp, tri_try, feat_mode=str(feat), spec=sp)
                pb = predict_series_calibrated_all6(val_imp, tri_try, coeffs_try, str(feat))
                eb_try = point_errors(tx, ty, pb[:, 0], pb[:, 1])
                rb, mb = rmse_mae(eb_try)
                _clab = (
                    sp.kind if sp.kind != "huber" else f"huber_eps{sp.huber_epsilon:g}_a{sp.huber_alpha:g}"
                )
                cand_b_log.append(
                    dict(
                        huber_f_scale=float(huber_sf),
                        calibration_kind=str(sp.kind),
                        calibration_detail=str(_clab),
                        huber_eps=float(sp.huber_epsilon) if sp.kind == "huber" else np.nan,
                        huber_alpha=float(sp.huber_alpha) if sp.kind == "huber" else np.nan,
                        ransac_rt=float(sp.ransac_residual_threshold) if sp.kind == "ransac" else np.nan,
                        feat_mode=str(feat),
                        val_StepB_RMSE_m=float(rb),
                        val_StepB_MAE_m=float(mb),
                    )
                )
                if rb < best_b["rmse_b"] - 1e-14:
                    best_b.update(
                        rmse_b=float(rb),
                        huber_f=huber_sf,
                        feat=str(feat),
                        calib_spec=sp,
                        coeffs=coeffs_try,
                        tri=tri_try,
                    )

    tri_star = best_b["tri"]
    coeffs_star: Dict[str, Tuple[float, float]] = best_b["coeffs"]
    feat_star = str(best_b["feat"])
    sp_star: CalibMethodSpec = best_b["calib_spec"]

    preds_b_val = predict_series_calibrated_all6(val_imp, tri_star, coeffs_star, feat_star)
    eb = point_errors(tx, ty, preds_b_val[:, 0], preds_b_val[:, 1])
    rmse_b, mae_b = rmse_mae(eb)

    # --- Stage C: 고정 보정에서 IRLS만 검증 그리드 --------------------------------
    cand_c_log: List[Dict[str, Any]] = []
    best_c = dict(rmse=float("inf"), thresh=2.0, max_iter=2)

    for th in IRLS_THRESH_GRID_M:
        for mx in IRLS_MAX_ITER_GRID:
            pc = predict_series_irls_with_meta(
                val_imp,
                tri_star,
                coeffs_star,
                feat_star,
                thresh_m=float(th),
                max_iters=int(mx),
            )[0]
            ec = point_errors(tx, ty, pc[:, 0], pc[:, 1])
            rc, mc = rmse_mae(ec)
            cand_c_log.append(
                dict(
                    irls_residual_thresh_m=float(th),
                    irls_max_iter=int(mx),
                    val_StepC_RMSE_m=float(rc),
                    val_StepC_MAE_m=float(mc),
                )
            )
            if rc < best_c["rmse"] - 1e-14:
                best_c.update(rmse=float(rc), thresh=float(th), max_iter=int(mx))

    th_best = float(best_c["thresh"])
    mx_best = int(best_c["max_iter"])

    preds_c_train, qc_tr, mvw_tr, nap_tr = predict_series_irls_with_meta(
        train_imp, tri_star, coeffs_star, feat_star, th_best, mx_best
    )
    preds_c_val, qc_va, mvw_va, nap_va = predict_series_irls_with_meta(
        val_imp, tri_star, coeffs_star, feat_star, th_best, mx_best
    )
    ec_val = point_errors(tx, ty, preds_c_val[:, 0], preds_c_val[:, 1])
    rmse_c, mae_c = rmse_mae(ec_val)

    # --- Stage 3: 다중 피처 K‑NN (scaled) + 검증 K 그리드, Train LOOCV 완화 채택
    mask_tr = np.isfinite(preds_c_train[:, 0]) & np.isfinite(preds_c_train[:, 1])
    d_x = tcx[mask_tr] - preds_c_train[mask_tr, 0]
    d_y = tcy[mask_tr] - preds_c_train[mask_tr, 1]
    Delta_tr = np.column_stack([d_x, d_y])

    feat_variants: List[bool] = [False, True]  # 평균 분산 vs log1p(평균 분산)

    ok_val_xy = np.isfinite(preds_c_val[:, 0]) & np.isfinite(preds_c_val[:, 1])
    fill_xy = (
        np.nanmedian(preds_c_val[ok_val_xy, :], axis=0)
        if np.any(ok_val_xy)
        else np.array([6.0, 9.0], dtype=float)
    )

    feasible_k = [k for k in KNN_K_GRID if k < int(np.sum(mask_tr))]

    rmse_d = rmse_c
    mae_d = mae_c
    xd_val = preds_c_val.copy()
    knn_pick: Optional[int] = None
    use_log_pick = False
    knn_sweep_records: List[Dict[str, Any]] = []

    rmse_best_val_grid = rmse_c
    xd_best_val = preds_c_val.copy()

    if int(np.sum(mask_tr)) >= 6 and feasible_k:
        for use_log_var in feat_variants:
            X_train_m = build_multi_feature_matrix(preds_c_train[mask_tr], qc_tr[mask_tr], mvw_tr[mask_tr], nap_tr[mask_tr], use_log_var=use_log_var)
            X_val_m = build_multi_feature_matrix(preds_c_val, qc_va, mvw_va, nap_va, use_log_var=use_log_var)

            for kn in feasible_k:
                xd_try, rk, mk = build_knn_predictions_multi_scaled(
                    kn,
                    X_train_m,
                    Delta_tr,
                    X_val_m,
                    preds_c_val,
                    tx,
                    ty,
                    fill_xy,
                )
                knn_sweep_records.append(
                    dict(
                        use_log1p_mean_var=bool(use_log_var),
                        K_neighbors=int(max(3, min(int(kn), len(X_train_m) - 1))),
                        RMSE_stepD_val_m=float(rk),
                        MAE_stepD_val_m=float(mk),
                    )
                )
                if rk < rmse_best_val_grid - 1e-14:
                    rmse_best_val_grid = rk
                    xd_best_val = xd_try.copy()
                    knn_pick = int(max(3, min(int(kn), len(X_train_m) - 1)))
                    use_log_pick = use_log_var

        xd_val = xd_best_val.copy()
        edm = point_errors(tx, ty, xd_val[:, 0], xd_val[:, 1])
        rmse_d, mae_d = rmse_mae(edm)

        rmse_guard_val = rmse_d
        # Train LOOCV 보조 선택 (더 낮을 때만)
        for use_log_var in feat_variants:
            X_train_m = build_multi_feature_matrix(preds_c_train[mask_tr], qc_tr[mask_tr], mvw_tr[mask_tr], nap_tr[mask_tr], use_log_var=use_log_var)
            X_val_m = build_multi_feature_matrix(preds_c_val, qc_va, mvw_va, nap_va, use_log_var=use_log_var)
            lok, _ = loocv_pick_k_multi(X_train_m, Delta_tr, feasible_k)
            if lok is None:
                continue
            xd_loo, rl, ml = build_knn_predictions_multi_scaled(
                int(lok),
                X_train_m,
                Delta_tr,
                X_val_m,
                preds_c_val,
                tx,
                ty,
                fill_xy,
            )
            knn_sweep_records.append(
                dict(
                    method="train_loocv_candidate",
                    use_log1p_mean_var=bool(use_log_var),
                    K_neighbors=int(lok),
                    RMSE_stepD_val_m=float(rl),
                    MAE_stepD_val_m=float(ml),
                )
            )
            if rl < rmse_d - 1e-12:
                xd_val = xd_loo
                rmse_d, mae_d = rl, ml
                knn_pick = int(lok)
                use_log_pick = use_log_var

        if rmse_d > rmse_guard_val + 5e-4:
            xd_val = xd_best_val.copy()
            edm = point_errors(tx, ty, xd_val[:, 0], xd_val[:, 1])
            rmse_d, mae_d = rmse_mae(edm)
    else:
        print("\n  [V11 KNN] 학습 포인트 부족 → Step D = Step C")

    ed_final = point_errors(tx, ty, xd_val[:, 0], xd_val[:, 1])

    # --- Console report
    print("\n" + "=" * 72)
    print("V11 검증 세트 성능 (Pure Wi‑Fi · Huber/RANSAC 보정 · IRLS · Multi‑feature KNN)")
    print("=" * 72)
    print(f"  Step A (Pure Wi‑Fi + 고정 bias):           RMSE = {rmse_a:.4f} m | MAE = {mae_a:.4f} m")
    print(f"  Step B (강건 선형 거리 보정 + 6AP 삼변):   RMSE = {rmse_b:.4f} m | MAE = {mae_b:.4f} m")
    print(f"  Step C (IRLS 가중 Bounded Huber 삼변):    RMSE = {rmse_c:.4f} m | MAE = {mae_c:.4f} m")
    print(f"  Step D (다중피처 K‑NN Δx,Δy):             RMSE = {rmse_d:.4f} m | MAE = {mae_d:.4f} m")
    print("=" * 72)
    d_ab = rmse_a - rmse_b
    d_bc = rmse_b - rmse_c
    d_cd = rmse_c - rmse_d
    d_ad = rmse_a - rmse_d
    print("\n검증 RMSE/MAE 개선 폭 (양수 = 해당 스텝이 이전 대비 개선)")
    print(f"  Step A → B  : ΔRMSE = {d_ab:+.4f} m | MAE {mae_a:.4f} → {mae_b:.4f}")
    print(f"  Step B → C  : ΔRMSE = {d_bc:+.4f} m | MAE {mae_b:.4f} → {mae_c:.4f}")
    print(f"  Step C → D  : ΔRMSE = {d_cd:+.4f} m | MAE {mae_c:.4f} → {mae_d:.4f}")
    print(f"  Step A → D  : ΔRMSE = {d_ad:+.4f} m | MAE {mae_a:.4f} → {mae_d:.4f}")
    _cal_txt = (
        "항등 거리 매핑 (A=1, Train 미적합)"
        if sp_star.kind == "identity"
        else (
            f"eps={sp_star.huber_epsilon}, α={sp_star.huber_alpha}"
            if sp_star.kind == "huber"
            else f"RANSAC_residual_thresh={sp_star.ransac_residual_threshold}"
        )
    )
    print(
        f"\n  [그리드 선택 요약] Step B=검증 RMSE(6AP 보정 삼변) 최소 → Step C=IRLS 임계·반복 최소 → Step D=K·피처\n"
        f"        Huber f_scale={best_b['huber_f']}, "
        f"calib={sp_star.kind} ({_cal_txt}) | feat={feat_star} | val StepB RMSE={best_b['rmse_b']:.4f} m\n"
        f"        IRLS_thresh={th_best} m, iters={mx_best} | val StepC RMSE(ref)={best_c['rmse']:.4f} m | "
        f"KNN_K={knn_pick if knn_pick is not None else '—'} log1p_var={use_log_pick}"
    )

    # --- Outputs
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = val_df[["Node_x", "Node_y", "True_X", "True_Y"]].copy()
    pred["StepA_X"] = xa_pred[:, 0]
    pred["StepA_Y"] = xa_pred[:, 1]
    pred["StepB_X"] = preds_b_val[:, 0]
    pred["StepB_Y"] = preds_b_val[:, 1]
    pred["StepC_X"] = preds_c_val[:, 0]
    pred["StepC_Y"] = preds_c_val[:, 1]
    pred["StepD_X"] = xd_val[:, 0]
    pred["StepD_Y"] = xd_val[:, 1]
    pred["ErrA_m"] = ea
    pred["ErrB_m"] = eb
    pred["ErrC_m"] = ec_val
    pred["ErrD_m"] = ed_final
    pred["StepC_geom_weighted_rms_m"] = qc_va.astype(float)
    pred["Mean_wifi_variance_used_m"] = mvw_va.astype(float)
    pred["N_ap_used"] = nap_va.astype(int)

    pd.DataFrame(knn_sweep_records).to_csv(out_dir / "v11_knn_grid_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cand_b_log).to_csv(out_dir / "v11_calibration_grid_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cand_c_log).to_csv(out_dir / "v11_irls_grid_log.csv", index=False, encoding="utf-8-sig")

    coef_rows = [
        {
            "wifi_ap": k,
            "A": v[0],
            "B": v[1],
            "feat_mode": feat_star,
            "calib": sp_star.kind,
        }
        for k, v in coeffs_star.items()
    ]
    pd.DataFrame(coef_rows).to_csv(out_dir / "v11_robust_calibration_wifi.csv", index=False, encoding="utf-8-sig")

    pmeta = resolve_paths(root)
    summ = {
        "step_rmse_A_m": rmse_a,
        "step_mae_A_m": mae_a,
        "step_rmse_B_m": rmse_b,
        "step_mae_B_m": mae_b,
        "step_rmse_C_m": rmse_c,
        "step_mae_C_m": mae_c,
        "step_rmse_D_m": rmse_d,
        "step_mae_D_m": mae_d,
        "delta_rmse_A_to_B_m": float(d_ab),
        "delta_rmse_B_to_C_m": float(d_bc),
        "delta_rmse_C_to_D_m": float(d_cd),
        "delta_rmse_A_to_D_m": float(d_ad),
        "grid_selection": {
            "trilateration_huber_f_scale": float(best_b["huber_f"]),
            "val_StepB_RMSE_gate_m": float(best_b["rmse_b"]),
            "val_StepC_RMSE_gate_m": float(best_c["rmse"]),
            "calibration": {
                "method": sp_star.kind,
                "huber_epsilon": float(sp_star.huber_epsilon),
                "huber_alpha": float(sp_star.huber_alpha),
                "ransac_residual_threshold_m": float(sp_star.ransac_residual_threshold),
            },
            "feat_mode": feat_star,
            "irls_residual_threshold_m": th_best,
            "irls_max_iterations": mx_best,
            "knn_neighbors": knn_pick,
            "knn_use_log1p_mean_var": bool(use_log_pick),
        },
        "data_paths": {
            "train_median": str(pmeta.train_median),
            "train_variance": str(pmeta.train_variance),
            "val_median": str(pmeta.val_median),
            "val_variance": str(pmeta.val_variance),
        },
        "note": (
            "V11: sklearn HuberRegressor/RANSAC per‑AP distance map, IRLS Tukey×Huber "
            "reweighting on top of 1/Var weights, multi‑feature KNN with StandardScaler. "
            "Grids tuned on validation only."
        ),
    }
    (out_dir / "v11_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")

    errs_plot = dict(A=ea, B=eb, C=ec_val, D=ed_final)
    lbl_s = dict(
        A="Step A: Pure Wi‑Fi",
        B="Step B: Robust calib",
        C="Step C: IRLS tri",
        D="Step D: Multi‑KNN",
    )
    rmse_blob = dict(A=rmse_a, B=rmse_b, C=rmse_c, D=rmse_d)
    save_v11_cdf_steps(out_dir, errs_plot, lbl_s, rmse_blob)

    pred.to_csv(out_dir / "v11_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"\n저장: {out_dir / 'v11_predictions.csv'}")
    print(f"저장: {out_dir / 'v11_summary.json'}")
    print(f"저장: {out_dir / 'v11_cdf_steps.png'}")
    print(f"저장: {out_dir / 'v11_robust_calibration_wifi.csv'}")
    print(f"저장: {out_dir / 'v11_calibration_grid_log.csv'}")
    print(f"저장: {out_dir / 'v11_irls_grid_log.csv'}")
    print(f"저장: {out_dir / 'v11_knn_grid_log.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
