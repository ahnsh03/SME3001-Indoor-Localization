"""
실시간 데이터 정제: 원본 Train(median/variance_test_kgh_corrected) 로드 후
헤더 정규화·Wi‑Fi 분산 상한 클리핑. 행(dropna) 삭제 없이 테이블 전 행 유지.

Phase 1 Bias(m): Train 원본 CSV에 대해 `sensor_spatial_profiler.py` **Analysis A**의
Robust Bias(trimmed median of measured−true, |e|≤15m·10% 양끝 절사)와 동일한 수치를 사용한다.
이는 **필터 전 전체 행·전 샘플**에 대한 장치별 체계적 편향 보정이다.

**UWB 순차 필터(하드컷→Var<MAD…) 후 Clean 잔차의 앵커별 평균**은
`sensor_profiler_uwb_clean_anchor_mean_error.csv`로 별도 저장된다.
그 값은 「신뢰할 만한 측정만 모았을 때 남은 오차 평균」이므로 곧바로 PHASE1을
대체하지 않는다(조건부 표본 + 이미 다른 단계와 역할 분담).
게이트/분산 cap/그리드 탐색 범위를 잡을 때 **참고 지표**로 쓴다.

`PHASE1_UWB_BIAS_M` 교체 후보 연구 시:
- 같은 Train에서 Analysis A 재실행 결과와 우선 숫자 맞춤.
- 실험적으로 Clean 평균을 쓸 경우는 브랜치/플래그로 분리하여 검증 세트 성능 확인.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

WIFI_VARIANCE_CLIP_MAX = 100.0

# Phase 1 — sensor_spatial_profiler Analysis A on median_test_kgh_corrected.csv (원본 Train)
PHASE1_UWB_BIAS_M: Dict[str, float] = {
    "110394ab": -1.9267,
    "e63ce2f": -3.6430,
    "8e610981": -0.5155,
    "d10485af": -2.0511,
    "d1044709": -0.9961,
    "4e610206": -0.6074,
}
PHASE1_WIFI_BIAS_M: Dict[str, float] = {
    "SW_11": 0.4973,
    "SW_first_team": 0.4400,
    "볼링공": 0.3074,
    "SW_4": 0.4482,
    "SW_6": 0.2626,
    "SW_5": 0.2768,
}
HARDWARE_CALIB_BIAS_M: Dict[str, float] = {**PHASE1_UWB_BIAS_M, **PHASE1_WIFI_BIAS_M}

# sensor_spatial_profiler (Train 원본): Wi‑Fi 분산 전체 P95≈6.52, UWB P95≈16.56
# 가중치/게이트에서 “분산 과대 ⇒ 신뢰 하락” 기준으로 사용 (Wi‑Fi 표 상한 클립 100과 별개)
VAR_CAP_WIFI_FUSION = 7.0
VAR_CAP_UWB_FUSION = 17.0


def load_sensor_excel_legacy(path: Path) -> pd.DataFrame:
    """엑셀: 기존 v4 스타일 — 첫 줄이 문자열 헤더인 경우와 표준 헤더 혼용."""
    path = Path(path)
    peek = pd.read_excel(path, header=None, nrows=1)
    first_cell = str(peek.iloc[0, 0]).strip().lower()
    if first_cell in ("node_x", "nodex"):
        raw = pd.read_excel(path, header=None)
        header_row = raw.iloc[0].astype(str).str.strip().tolist()
        df = raw.iloc[1:].copy().reset_index(drop=True)
        df.columns = header_row
    else:
        raw = pd.read_excel(path)
        if "Node_x" in raw.columns:
            df = raw.copy()
        else:
            header_row = raw.iloc[0].astype(str).str.strip().tolist()
            df = raw.iloc[1:].copy().reset_index(drop=True)
            df.columns = header_row
    return df


def load_sensor_tabular(path: Path) -> pd.DataFrame:
    """CSV 또는 엑셀 로드 후 컬럼명 문자열 스트립만 적용(normalize 전 단계)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        hdr = pd.read_csv(path, nrows=1)
        first_names = [str(x).strip() for x in hdr.columns]
        has_uwb_anchor = any("110394ab" == str(x).strip() for x in hdr.columns)
        low0 = first_names[0].lower().replace("_", "") if first_names else ""
        if has_uwb_anchor or low0 == "nodex":
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, header=1)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    if suf in (".xlsx", ".xls"):
        df = load_sensor_excel_legacy(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    raise ValueError(f"지원 확장자가 아님: {path}")


def normalize_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명 strip, d1044709(식별잘안됨) 등 변형 → d1044709."""
    out = df.copy()
    new_cols: List[str] = []
    seen_d104 = False
    for c in out.columns:
        s = str(c).strip()
        low = s.lower().replace(" ", "")
        if "d1044709" in low or ("d10447" in low and "식별" in s):
            if not seen_d104:
                s = "d1044709"
                seen_d104 = True
        new_cols.append(s)
    out.columns = new_cols
    return out


def infer_sensor_columns_ordered(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """과제 표준: 앞 2열 노드 좌표, 다음 6 UWB, 다음 6 Wi‑Fi."""
    cols = df.columns.tolist()
    if len(cols) < 14:
        raise ValueError("열이 14개 미만입니다: Node_x, Node_y 및 센서 12개 필요")
    return cols[:2], cols[2:8], cols[8:14]


def clip_wifi_variance_columns(
    var_df: pd.DataFrame,
    wifi_column_names: List[str],
    cap: float = WIFI_VARIANCE_CLIP_MAX,
) -> None:
    """Wi‑Fi 분산 폭발(예: 1e8)을 cap으로 억제. 제자리 수정."""
    for c in wifi_column_names:
        if c not in var_df.columns:
            continue
        ser = pd.to_numeric(var_df[c], errors="coerce")
        clipped = np.where(np.isfinite(ser), np.minimum(ser.astype(float), float(cap)), ser)
        var_df[c] = clipped


def standardize_loaded_median_variance(
    med_path: Path,
    var_path: Path,
    coerce_numeric: Callable[[pd.DataFrame], pd.DataFrame],
    infer_sensor_columns: Callable[[pd.DataFrame], Tuple[List[str], List[str], List[str]]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """로드 → 수치화 → 헤더 정규화 → 분산표 Wi‑Fi 클립. 행 삭제 없음."""
    med = coerce_numeric(load_sensor_tabular(med_path))
    var = coerce_numeric(load_sensor_tabular(var_path))
    med = normalize_column_headers(med)
    var = normalize_column_headers(var)
    nc, uwb, wifi = infer_sensor_columns(med)
    _nc_v, uwb_v, wifi_v = infer_sensor_columns(var)
    if uwb != uwb_v or wifi != wifi_v:
        raise ValueError("median / variance 열 구성이 일치하지 않습니다.")
    clip_wifi_variance_columns(var, wifi)
    return med, var, nc, uwb, wifi


def resolve_train_kgh_corrected_paths(train_dir: Path) -> Tuple[Path, Path]:
    """공식 테스트(학습) 데이터: `median/variance_test_kgh_corrected`만 사용. 우선순위 `.csv` → `.xlsx`/`.xls`."""
    train_dir = Path(train_dir)
    for stem_m, stem_v in [
        ("median_test_kgh_corrected", "variance_test_kgh_corrected"),
    ]:
        for ext in (".csv", ".xlsx", ".xls"):
            m = train_dir / f"{stem_m}{ext}"
            v = train_dir / f"{stem_v}{ext}"
            if m.is_file() and v.is_file():
                return m, v
    raise FileNotFoundError(
        f"Train 원본 필요: {train_dir}/median_test_kgh_corrected.* + variance_test_kgh_corrected.*"
    )


def resolve_validation_paths(val_dir: Path) -> Tuple[Path, Path]:
    val_dir = Path(val_dir)
    for ext in (".csv", ".xlsx", ".xls"):
        m = val_dir / f"median_validation{ext}"
        v = val_dir / f"variance_validation{ext}"
        if m.is_file() and v.is_file():
            return m, v
    raise FileNotFoundError(f"검증 쌍 필요: {val_dir}/median_validation.* + variance_validation.*")
