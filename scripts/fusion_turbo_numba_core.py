"""
V12 Turbo와 동치인 Step B/C Numba 배치 경로 및 LocGeom 패킹.

`indoor_fusion_pipeline_v12_turbo.py`에 있던 순수 연산 블록을 분리하여
그 스크립트에 의존하지 않는 파이프라인(예: v13_fix)에서 재사용한다.
원본 로직 변경 시 turbo와 함께 동기화할 것.

TRILAT_GN_ITERS: 삼변 Gauss‑Newton 반복.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from numba import njit

from fusion_realtime_sanitize import VAR_CAP_UWB_FUSION, VAR_CAP_WIFI_FUSION
from indoor_fusion_pipeline_v8 import (
    BOUNDS_HI,
    BOUNDS_LO,
    EPS_W,
    ROBUST_BIAS_M,
    UWB_BIAS_W,
    WIFI_BIAS_W,
    WIFI_COL_CANON,
    FusionLocalizerV8,
    sensor_canon_key,
)

TRILAT_GN_ITERS = 22


def resolve_kfold_n_splits(n_samples: int) -> int:
    if n_samples < 2:
        raise ValueError("Train 샘플이 2개 미만이면 K-Fold를 할 수 없습니다.")
    if n_samples < 6:
        return 2
    return int(min(5, max(3, n_samples // 35)))


@dataclass
class LocGeomPack:
    """Worker 피클링용: DataFrame/locator 없이 기하·바이어스만 전달."""

    n_u: int
    n_w: int
    uwb_xy: np.ndarray  # (n_u, 2)
    wifi_xy: np.ndarray  # (n_w, 2)
    uwb_robust_bias: np.ndarray  # (n_u,)
    wifi_robust_bias: np.ndarray  # (n_w,)
    var_cap_uwb: float
    var_cap_wifi: float
    eps_w: float
    uwb_bias_w: float
    wifi_bias_w: float
    bounds_lo: np.ndarray  # (2,)
    bounds_hi: np.ndarray  # (2,)


def build_loc_geom_pack(loc: FusionLocalizerV8) -> LocGeomPack:
    nu, nw = len(loc.uwb_cols), len(loc.wifi_cols)
    uwb_xy = np.zeros((nu, 2), dtype=np.float64)
    wifi_xy = np.zeros((nw, 2), dtype=np.float64)
    uwb_rb = np.zeros(nu, dtype=np.float64)
    wifi_rb = np.zeros(nw, dtype=np.float64)
    for i, c in enumerate(loc.uwb_cols):
        ax, ay = loc._resolve_sensor_position_m(c, i, True)
        uwb_xy[i, 0], uwb_xy[i, 1] = ax, ay
        uwb_rb[i] = float(ROBUST_BIAS_M[sensor_canon_key(c, i, True)])
    for i, c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        ax, ay = loc._resolve_sensor_position_m(c, i, False)
        wifi_xy[i, 0], wifi_xy[i, 1] = ax, ay
        wifi_rb[i] = float(ROBUST_BIAS_M[sensor_canon_key(canon, i, False)])
    return LocGeomPack(
        n_u=nu,
        n_w=nw,
        uwb_xy=uwb_xy,
        wifi_xy=wifi_xy,
        uwb_robust_bias=uwb_rb,
        wifi_robust_bias=wifi_rb,
        var_cap_uwb=float(VAR_CAP_UWB_FUSION),
        var_cap_wifi=float(VAR_CAP_WIFI_FUSION),
        eps_w=float(EPS_W),
        uwb_bias_w=float(UWB_BIAS_W),
        wifi_bias_w=float(WIFI_BIAS_W),
        bounds_lo=BOUNDS_LO.astype(np.float64),
        bounds_hi=BOUNDS_HI.astype(np.float64),
    )


def coeffs_dict_to_ab(coeffs: Dict[str, Tuple[float, float]], loc: FusionLocalizerV8) -> np.ndarray:
    ab = np.zeros((len(loc.wifi_cols), 2), dtype=np.float64)
    for i, _c in enumerate(loc.wifi_cols):
        canon = WIFI_COL_CANON[i]
        a, b = coeffs[canon]
        ab[i, 0], ab[i, 1] = float(a), float(b)
    return ab


def feat_to_id(feat: str) -> int:
    return 0 if feat == "bias_sub" else 1


def te_df_to_numpy_blocks(te_df: pd.DataFrame, loc: FusionLocalizerV8) -> Dict[str, np.ndarray]:
    nu, nw = len(loc.uwb_cols), len(loc.wifi_cols)
    u_med = np.zeros((len(te_df), nu), dtype=np.float64)
    u_var = np.full((len(te_df), nu), np.nan, dtype=np.float64)
    u_nan = np.ones((len(te_df), nu), dtype=np.uint8)
    w_med = np.zeros((len(te_df), nw), dtype=np.float64)
    w_var = np.full((len(te_df), nw), np.nan, dtype=np.float64)
    for j, c in enumerate(loc.uwb_cols):
        mc, vc = f"med_{c}", f"var_{c}"
        on = f"orig_nan_{c}"
        if mc in te_df.columns:
            u_med[:, j] = te_df[mc].values.astype(np.float64)
        if vc in te_df.columns:
            u_var[:, j] = te_df[vc].values.astype(np.float64)
        if on in te_df.columns:
            u_nan[:, j] = (te_df[on].values == 1).astype(np.uint8)
        else:
            u_nan[:, j] = 1
    for j, c in enumerate(loc.wifi_cols):
        mc, vc = f"med_{c}", f"var_{c}"
        if mc in te_df.columns:
            w_med[:, j] = te_df[mc].values.astype(np.float64)
        if vc in te_df.columns:
            w_var[:, j] = te_df[vc].values.astype(np.float64)
    return dict(
        true_x=te_df["True_X"].values.astype(np.float64),
        true_y=te_df["True_Y"].values.astype(np.float64),
        uwb_med=u_med,
        uwb_var=u_var,
        uwb_was_nan=u_nan,
        wifi_med=w_med,
        wifi_var=w_var,
    )


# --- Numba: Tukey / Huber / 바운드 Huber 삼변 / IRLS / 배치 Step B·C ---
# indoor_fusion_pipeline_v12_turbo.py 와 동일.


@njit(cache=True, fastmath=True)
def tukey_bisquare_weights_njit(u: np.ndarray, c: float) -> np.ndarray:
    n = u.shape[0]
    w = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = abs(u[i])
        if a < c:
            t = u[i] / c
            w[i] = (1.0 - t * t) ** 2
    return w


@njit(cache=True, fastmath=True)
def huber_hinge_weights_njit(r: np.ndarray, delta: float) -> np.ndarray:
    n = r.shape[0]
    d = delta if delta > 1e-6 else 1e-6
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = abs(r[i])
        den = a if a > 1e-6 else 1e-6
        t = d / den
        out[i] = 1.0 if t > 1.0 else t
    return out


@njit(cache=True)
def solve_trilat_bounded_huber_njit(
    pos: np.ndarray,
    dist: np.ndarray,
    weights: np.ndarray,
    huber_f_scale: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    gn_iters: int,
) -> np.ndarray:
    n = dist.shape[0]
    out = np.empty(2, dtype=np.float64)
    if n == 0:
        out[0], out[1] = np.nan, np.nan
        return out
    if n < 3:
        sw = 0.0
        sx, sy = 0.0, 0.0
        for i in range(n):
            di = dist[i] if dist[i] > 1e-3 else 1e-3
            wi = weights[i] if weights[i] > 0.0 else 0.0
            if wi < 1e-12:
                continue
            sw += wi
            sx += pos[i, 0] * wi
            sy += pos[i, 1] * wi
        if sw < 1e-12:
            out[0], out[1] = np.nan, np.nan
            return out
        x = sx / sw
        y = sy / sw
        out[0] = x if x > bounds_lo[0] else bounds_lo[0]
        if out[0] > bounds_hi[0]:
            out[0] = bounds_hi[0]
        out[1] = y if y > bounds_lo[1] else bounds_lo[1]
        if out[1] > bounds_hi[1]:
            out[1] = bounds_hi[1]
        return out

    idx = 0
    best = dist[0]
    for i in range(1, n):
        if dist[i] < best:
            best = dist[i]
            idx = i
    x = pos[idx, 0]
    y = pos[idx, 1]
    if x < bounds_lo[0]:
        x = bounds_lo[0]
    if x > bounds_hi[0]:
        x = bounds_hi[0]
    if y < bounds_lo[1]:
        y = bounds_lo[1]
    if y > bounds_hi[1]:
        y = bounds_hi[1]

    delta = huber_f_scale
    for _ in range(gn_iters):
        JtJ00, JtJ01, JtJ11 = 0.0, 0.0, 0.0
        Jtr0, Jtr1 = 0.0, 0.0
        for i in range(n):
            dx = x - pos[i, 0]
            dy = y - pos[i, 1]
            ri0 = np.sqrt(dx * dx + dy * dy)
            sw = np.sqrt(weights[i] if weights[i] > 0.0 else 0.0)
            raw = sw * (ri0 - dist[i])
            ar = abs(raw)
            w_h = 1.0 if ar <= delta else (delta / (ar if ar > 1e-9 else 1e-9))
            rw = raw * np.sqrt(w_h)
            if ri0 < 1e-12:
                j0, j1 = 0.0, 0.0
            else:
                j0 = sw * np.sqrt(w_h) * dx / ri0
                j1 = sw * np.sqrt(w_h) * dy / ri0
            JtJ00 += j0 * j0
            JtJ01 += j0 * j1
            JtJ11 += j1 * j1
            Jtr0 += j0 * rw
            Jtr1 += j1 * rw
        det = JtJ00 * JtJ11 - JtJ01 * JtJ01
        adet = det if det >= 0.0 else -det
        if adet < 1e-18:
            break
        dx0 = (-Jtr0 * JtJ11 + Jtr1 * JtJ01) / det
        dx1 = (Jtr0 * JtJ01 - Jtr1 * JtJ00) / det
        x = x + dx0
        y = y + dx1
        if x < bounds_lo[0]:
            x = bounds_lo[0]
        if x > bounds_hi[0]:
            x = bounds_hi[0]
        if y < bounds_lo[1]:
            y = bounds_lo[1]
        if y > bounds_hi[1]:
            y = bounds_hi[1]
    out[0], out[1] = x, y
    return out


@njit(cache=True)
def irls_refine_xyz_njit(
    pos: np.ndarray,
    dist: np.ndarray,
    w_base: np.ndarray,
    huber_f_fusion: float,
    thresh_m: float,
    max_iters: int,
    tukey_c: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    gn_iters: int,
) -> np.ndarray:
    n = dist.shape[0]
    if n < 3:
        return solve_trilat_bounded_huber_njit(pos, dist, w_base, huber_f_fusion, bounds_lo, bounds_hi, gn_iters)
    w = w_base.copy()
    xy_last = np.empty(2, dtype=np.float64)
    xy_last[0], xy_last[1] = np.nan, np.nan
    itlim = max_iters if max_iters > 0 else 1
    for _ in range(itlim):
        xy_last = solve_trilat_bounded_huber_njit(pos, dist, w, huber_f_fusion, bounds_lo, bounds_hi, gn_iters)
        if not (np.isfinite(xy_last[0]) and np.isfinite(xy_last[1])):
            break
        r = np.empty(n, dtype=np.float64)
        for i in range(n):
            dx = xy_last[0] - pos[i, 0]
            dy = xy_last[1] - pos[i, 1]
            gpred = np.sqrt(dx * dx + dy * dy)
            t = gpred - dist[i]
            r[i] = t if t >= 0.0 else -t
        med = np.median(r)
        ad = np.empty(n, dtype=np.float64)
        for i in range(n):
            ad[i] = abs(r[i] - med)
        mad = np.median(ad)
        sigma_rob = mad * 1.4826
        if sigma_rob < 0.08:
            sigma_rob = 0.08
        thr = thresh_m * 0.65
        sigma = thr if thr > sigma_rob else sigma_rob
        if sigma < 1e-2:
            sigma = 1e-2
        u = np.empty(n, dtype=np.float64)
        for i in range(n):
            u[i] = r[i] / sigma
        tw = tukey_bisquare_weights_njit(u, tukey_c)
        hw = huber_hinge_weights_njit(r, thresh_m)
        w_new = np.empty(n, dtype=np.float64)
        for i in range(n):
            m = tw[i] * hw[i]
            if m < 0.04:
                m = 0.04
            if m > 1.0:
                m = 1.0
            w_new[i] = w_base[i] * m
        dw = 0.0
        wnorm = 0.0
        for i in range(n):
            t = w_new[i] - w[i]
            dw += t * t
            wnorm += w[i] * w[i]
        swn = np.sqrt(wnorm if wnorm > 0.0 else 0.0) + 1e-9
        if swn < 1e-12:
            break
        rat = np.sqrt(dw if dw > 0.0 else 0.0) / swn
        if rat < 1e-4:
            for i in range(n):
                w[i] = w_new[i]
            xy_last = solve_trilat_bounded_huber_njit(pos, dist, w, huber_f_fusion, bounds_lo, bounds_hi, gn_iters)
            break
        for i in range(n):
            w[i] = w_new[i]
    return xy_last


@njit(cache=True, parallel=False)
def batch_step_c_preds_flat(
    n: int,
    xy_guide: np.ndarray,
    uwb_med: np.ndarray,
    uwb_var: np.ndarray,
    uwb_was_nan: np.ndarray,
    wifi_med: np.ndarray,
    wifi_var: np.ndarray,
    coeff_ab: np.ndarray,
    feat_id: int,
    gate_t: float,
    uwb_var_inflate: float,
    hf_fusion: float,
    hf_wifi: float,
    irls_thresh: float,
    irls_mx: int,
    tukey_c: float,
    nu: int,
    nw: int,
    uwb_xy: np.ndarray,
    wifi_xy: np.ndarray,
    uwb_rb: np.ndarray,
    wifi_rb: np.ndarray,
    var_cap_uwb: float,
    var_cap_wifi: float,
    eps_w: float,
    uwb_bias_w: float,
    wifi_bias_w: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    gn_iters: int,
    max_range_m: float,
) -> np.ndarray:
    preds = np.full((n, 2), np.nan, dtype=np.float64)
    max_pts = nu + nw
    for k in range(n):
        gx = xy_guide[k, 0]
        gy = xy_guide[k, 1]
        pos_b = np.zeros((max_pts, 2), dtype=np.float64)
        dist_b = np.zeros(max_pts, dtype=np.float64)
        w_b = np.zeros(max_pts, dtype=np.float64)
        ct = 0
        if np.isfinite(gx) and np.isfinite(gy):
            for j in range(nu):
                if uwb_was_nan[k, j] != 0:
                    continue
                dr = uwb_med[k, j]
                vv = uwb_var[k, j]
                if (not np.isfinite(dr)) or (not np.isfinite(vv)):
                    continue
                if vv > var_cap_uwb:
                    continue
                d_meas = dr - uwb_rb[j]
                ax = uwb_xy[j, 0]
                ay = uwb_xy[j, 1]
                d_geom = np.sqrt((gx - ax) ** 2 + (gy - ay) ** 2)
                diff = d_geom - d_meas
                ad = diff if diff >= 0.0 else -diff
                if ad > gate_t:
                    continue
                infm = uwb_var_inflate if uwb_var_inflate > 1.0 else 1.0
                eff_var = vv * infm
                wi = 1.0 / (eff_var + uwb_bias_w + eps_w)
                pos_b[ct, 0] = ax
                pos_b[ct, 1] = ay
                dist_b[ct] = d_meas
                w_b[ct] = wi
                ct += 1
        for j in range(nw):
            m_raw = wifi_med[k, j]
            vv = wifi_var[k, j]
            if (not np.isfinite(m_raw)) or (not np.isfinite(vv)):
                continue
            if vv > var_cap_wifi:
                continue
            if m_raw > max_range_m:
                continue
            A = coeff_ab[j, 0]
            B = coeff_ab[j, 1]
            if feat_id == 0:
                xf = m_raw - wifi_rb[j]
            else:
                xf = m_raw
            d_c = A * xf + B
            if d_c < 0.1:
                d_c = 0.1
            wi2 = 1.0 / (vv + wifi_bias_w + eps_w)
            pos_b[ct, 0] = wifi_xy[j, 0]
            pos_b[ct, 1] = wifi_xy[j, 1]
            dist_b[ct] = d_c
            w_b[ct] = wi2
            ct += 1

        wpos = np.zeros((nw, 2), dtype=np.float64)
        wdist = np.zeros(nw, dtype=np.float64)
        wwei = np.zeros(nw, dtype=np.float64)
        cw = 0
        for j in range(nw):
            m_raw = wifi_med[k, j]
            vv = wifi_var[k, j]
            if (not np.isfinite(m_raw)) or (not np.isfinite(vv)):
                continue
            if vv > var_cap_wifi:
                continue
            if m_raw > max_range_m:
                continue
            A = coeff_ab[j, 0]
            B = coeff_ab[j, 1]
            if feat_id == 0:
                xf = m_raw - wifi_rb[j]
            else:
                xf = m_raw
            d_c = A * xf + B
            if d_c < 0.1:
                d_c = 0.1
            wi2 = 1.0 / (vv + wifi_bias_w + eps_w)
            wpos[cw, 0] = wifi_xy[j, 0]
            wpos[cw, 1] = wifi_xy[j, 1]
            wdist[cw] = d_c
            wwei[cw] = wi2
            cw += 1

        if ct < 3:
            if cw == 0:
                preds[k, 0] = np.nan
                preds[k, 1] = np.nan
            else:
                p2 = wpos[:cw, :].copy()
                d2 = wdist[:cw].copy()
                w2 = wwei[:cw].copy()
                xyw = solve_trilat_bounded_huber_njit(p2, d2, w2, hf_wifi, bounds_lo, bounds_hi, gn_iters)
                preds[k, 0] = xyw[0]
                preds[k, 1] = xyw[1]
        else:
            puse = pos_b[:ct, :].copy()
            duse = dist_b[:ct].copy()
            wuse = w_b[:ct].copy()
            xyf = irls_refine_xyz_njit(
                puse,
                duse,
                wuse,
                hf_fusion,
                irls_thresh,
                irls_mx,
                tukey_c,
                bounds_lo,
                bounds_hi,
                gn_iters,
            )
            preds[k, 0] = xyf[0]
            preds[k, 1] = xyf[1]
    return preds


@njit(cache=True, parallel=False)
def batch_step_b_preds_flat(
    n: int,
    xy_guide: np.ndarray,
    uwb_med: np.ndarray,
    uwb_var: np.ndarray,
    uwb_was_nan: np.ndarray,
    wifi_med: np.ndarray,
    wifi_var: np.ndarray,
    coeff_ab: np.ndarray,
    feat_id: int,
    gate_t: float,
    uwb_var_inflate: float,
    hf_fusion: float,
    nu: int,
    nw: int,
    uwb_xy: np.ndarray,
    wifi_xy: np.ndarray,
    uwb_rb: np.ndarray,
    wifi_rb: np.ndarray,
    var_cap_uwb: float,
    var_cap_wifi: float,
    eps_w: float,
    uwb_bias_w: float,
    wifi_bias_w: float,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    gn_iters: int,
    max_range_m: float,
) -> np.ndarray:
    preds = np.full((n, 2), np.nan, dtype=np.float64)
    max_pts = nu + nw
    for k in range(n):
        gx = xy_guide[k, 0]
        gy = xy_guide[k, 1]
        pos_b = np.zeros((max_pts, 2), dtype=np.float64)
        dist_b = np.zeros(max_pts, dtype=np.float64)
        w_b = np.zeros(max_pts, dtype=np.float64)
        ct = 0
        if np.isfinite(gx) and np.isfinite(gy):
            for j in range(nu):
                if uwb_was_nan[k, j] != 0:
                    continue
                dr = uwb_med[k, j]
                vv = uwb_var[k, j]
                if (not np.isfinite(dr)) or (not np.isfinite(vv)):
                    continue
                if vv > var_cap_uwb:
                    continue
                d_meas = dr - uwb_rb[j]
                ax = uwb_xy[j, 0]
                ay = uwb_xy[j, 1]
                d_geom = np.sqrt((gx - ax) ** 2 + (gy - ay) ** 2)
                diff = d_geom - d_meas
                ad = diff if diff >= 0.0 else -diff
                if ad > gate_t:
                    continue
                infm = uwb_var_inflate if uwb_var_inflate > 1.0 else 1.0
                eff_var = vv * infm
                wi = 1.0 / (eff_var + uwb_bias_w + eps_w)
                pos_b[ct, 0] = ax
                pos_b[ct, 1] = ay
                dist_b[ct] = d_meas
                w_b[ct] = wi
                ct += 1
        for j in range(nw):
            m_raw = wifi_med[k, j]
            vv = wifi_var[k, j]
            if (not np.isfinite(m_raw)) or (not np.isfinite(vv)):
                continue
            if vv > var_cap_wifi:
                continue
            if m_raw > max_range_m:
                continue
            A = coeff_ab[j, 0]
            B = coeff_ab[j, 1]
            if feat_id == 0:
                xf = m_raw - wifi_rb[j]
            else:
                xf = m_raw
            d_c = A * xf + B
            if d_c < 0.1:
                d_c = 0.1
            wi2 = 1.0 / (vv + wifi_bias_w + eps_w)
            pos_b[ct, 0] = wifi_xy[j, 0]
            pos_b[ct, 1] = wifi_xy[j, 1]
            dist_b[ct] = d_c
            w_b[ct] = wi2
            ct += 1
        if ct == 0:
            preds[k, 0] = np.nan
            preds[k, 1] = np.nan
        else:
            puse = pos_b[:ct, :].copy()
            duse = dist_b[:ct].copy()
            wuse = w_b[:ct].copy()
            xyf = solve_trilat_bounded_huber_njit(puse, duse, wuse, hf_fusion, bounds_lo, bounds_hi, gn_iters)
            preds[k, 0] = xyf[0]
            preds[k, 1] = xyf[1]
    return preds


def _smoke_main() -> int:
    """라이브러리 모듈: K-Fold 분할 수 규칙만 기록(융합 수치 경로 미실행)."""
    root = Path(__file__).resolve().parent.parent
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"n_samples": n, "n_splits": resolve_kfold_n_splits(n)} for n in (2, 5, 6, 34, 35, 36, 70, 100, 200)]
    payload = {"module": "fusion_turbo_numba_core", "resolve_kfold_n_splits": rows}
    path = out / "fusion_turbo_numba_core_smoke.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("fusion_turbo_numba_core: smoke JSON 기록 완료 (resolve_kfold_n_splits 표만).")
    return 0


if __name__ == "__main__":
    from script_run_io import cli_entrypoint

    cli_entrypoint(
        Path(__file__),
        _smoke_main,
        output_artifact_include_prefixes=("fusion_turbo_numba_core_",),
    )
