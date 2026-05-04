# 공통 모듈 명세: `fusion_*`

[← 색인](SPEC_INDEX.md)

---

## 1. `fusion_realtime_sanitize.py`

### 1.1 목적·진화 배경

파이프라인 v4 이후 및 v1 리팩터링 과정에서 **median/variance 로드 방식이 파일마다 제각각**이어서, **단일 진입점**으로 모았다. 동시에 **Train 원본에 대한 공간 프로파일러(`sensor_spatial_profiler.py`) Analysis A**와 동일한 **장치별 체계 편향(Phase 1 bias)** 을 모든 융합 스크립트가 공유하도록 고정하였다.

### 1.2 적용 기법·수식

**참 거리(Train에서만 정의 가능한 기준)**  
격자 노드 \((n_x, n_y)\)에 대해 참 위치 \((X,Y) = 0.6\,(n_x, n_y)\) m. 앵커 \(a\)의 참 거리:

\[
d^{\mathrm{true}}_{i,a} = \big\| (X_i, Y_i) - (a_x, a_y) \big\|_2 .
\]

**Analysis A — Robust Bias (장치 \(a\))**  
측정 중앙값 \(m_{i,a}\), 잔차 \(e_{i,a} = m_{i,a} - d^{\mathrm{true}}_{i,a}\).

1. 유한한 \(e_{i,a}\)만 사용.
2. \(|e_{i,a}| \le 15\) m 인 표본만 유지.
3. 잔차 분포에서 **하위 10%·상위 10% 절사**(quantile trim) 후 남은 집합 \(\mathcal{C}_a\).

\[
b^{\mathrm{robust}}_a = \mathrm{median}\{ e_{i,a} : i \in \mathcal{C}_a \}.
\]

이 값이 `PHASE1_UWB_BIAS_M` / `PHASE1_WIFI_BIAS_M`에 하드코딩되어 있으며, 런타임에서는 `HARDWARE_CALIB_BIAS_M`으로 병합되어 쓰인다. (프로파일러를 다시 돌리면 **숫자 동기화**가 필요하다는 주석이 모듈 docstring에 있다.)

**분산 융합 상한 (`VAR_CAP_*`)**  
Train 분산 분포의 **대표 분위(문서화: Wi‑Fi P95≈6.52, UWB P95≈16.56 m² 근처)**를 참고해, 융합·게이트에서 **과대 분산을 신뢰 하향**시키기 위한 상한으로 `VAR_CAP_WIFI_FUSION = 7.0`, `VAR_CAP_UWB_FUSION = 17.0`을 둔다. Wi‑Fi 표 전역 클립 `WIFI_VARIANCE_CLIP_MAX = 100`과는 역할이 다르다.

### 1.3 주요 API

- `load_sensor_tabular` / `load_sensor_excel_legacy`: CSV·xlsx, 헤더 1행/2행 혼재 대응.
- `normalize_column_headers`: `d1044709` 변형 통일.
- `clip_wifi_variance_columns`: 상한 클립.
- `standardize_loaded_median_variance(med_path, var_path, coerce_numeric, infer_cols)`: **행 삭제 없이** 정렬된 median·variance 프레임 반환.
- `resolve_train_kgh_corrected_paths`, `resolve_validation_paths`: 프로젝트 루트 기준 경로 해석.

### 1.4 산출·해석

코드 자체는 PNG를 쓰지 않는다. **바이어스 표의 수치**가 `sensor_spatial_profiler` 출력과 일치하는지가 데이터 파이프라인 일관성의 척도이다.

---

## 2. `fusion_data_audit.py`

### 2.1 목적

CI·수동 점검용으로, `standardize_loaded_median_variance`를 통과한 **Train·Validation 행 수**를 한 번에 출력한다.

### 2.2 기법

외부 수식 없음. `fusion_realtime_sanitize`의 resolver와 동일 경로를 사용하므로 **“파이프라인이 실제로 읽는 행 수”**와 동일하다.

### 2.3 실행

`py -3 scripts/fusion_data_audit.py` (프로젝트 루트)

---

## 3. `fusion_turbo_numba_core.py`

### 3.1 목적·진화 배경

`indoor_fusion_pipeline_v12_turbo.py`에 있던 **Step B/C Numba 배치**와 `LocGeomPack` 패킹이 커지면서, **`v13_fix` 등이 turbo 소스 파일에 의존하지 않도록** 순수 연산 블록을 분리하였다. 주석에 “turbo와 동기화” 요구가 있다.

### 3.2 적용 기법·수식

**공통 삼변 목표 (요지)**  
미지점 \(\mathbf{x}=(x,y)\), 앵커 \(\mathbf{a}_j\), 보정·게이트된 거리 \(\hat{d}_j\), 가중 \(w_j\). 가우스‑뉴턴은 **Huber 손실이 붙은 잔차 벡터**에 대해 반복 선형화한다. 구현은 `solve_trilat_bounded_huber_njit`, `irls_refine_xyz_njit` 등 `@njit` 함수에 캡슐화되어 있다.

**IRLS (Turbo/V12와 동계열)**  
잔차 \(r_j = \|\mathbf{x}-\mathbf{a}_j\| - \hat{d}_j\)에 대해 **Tukey bisquare**와 **Huber hinge**를 조합한 가중을 곱해 \(w_j\)를 갱신하고, 고정 회수(코드 상 2회 등) 반복한다. Tukey 상수는 파이프라인 쪽과 맞추어 `TUKEY_C ≈ 4.685` 계열을 사용한다.

**바운드**  
`BOUNDS_LO`, `BOUNDS_HI`(v8 계열과 동일)로 \(\mathbf{x}\)를 클립한다.

### 3.3 상수

- `TRILAT_GN_ITERS = 22`: GN 반복.
- `resolve_kfold_n_splits(n)`: 샘플 수에 따른 폴드 수 휴리스틱.

### 3.4 의존성

`FusionLocalizerV8`, `ROBUST_BIAS_M`, `VAR_CAP_*` 등 — 기하·캡만 패킹하고 DataFrame은 워커에 넘기지 않는다.
