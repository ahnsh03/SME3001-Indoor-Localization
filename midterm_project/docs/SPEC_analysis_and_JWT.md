# 데이터 분석·보조 파이프라인 명세

[← 색인](SPEC_INDEX.md)

---

## 1. `sensor_spatial_profiler.py`

### 1.1 목적

Train 공식 데이터(`median_test_kgh_corrected` + variance)에 대해:

1. **Analysis A**: 기기별 **강건 편향(Robust Bias)** 및 분산 백분위.
2. **Analysis B**: 공간적 **UWB NaN 다발 격자**, **측정/참거리 비율**이 큰 격자(과대평가 의심).
3. **시각화**: 측정 vs 참거리 산점도, 오차 박스플롯, **UWB 순차 3단계 필터** 전후 비교 PNG.

### 1.2 수식·알고리즘

**참 거리**  
\(d^{\mathrm{true}}_{i,a} = \| (0.6 n_{x,i}, 0.6 n_{y,i}) - \mathbf{a}_a \|_2\).

**Robust Bias (Analysis A, 기기 \(a\))**  
잔차 \(e_{i,a} = m_{i,a} - d^{\mathrm{true}}_{i,a}\).

1. \(|e_{i,a}| \le 15\) m (`ERROR_ABS_CAP_M`) 필터.
2. 하위·상위 각 **10% 절사** (`TRIM_PROP = 0.1`) → 집합 \(\mathcal{C}_a\).

\[
b_a = \mathrm{median}(\mathcal{C}_a), \qquad
\bar{e}^{\mathrm{trim}}_a = \mathrm{mean}(\mathcal{C}_a).
\]

출력 테이블에 **Robust Bias**, **Trimmed Mean Err**, **Catastrophic %**, **Var P50/P90/P99**가 포함된다.

**UWB 순차 필터 (요지)**  
하드 거리 구간 \([0.5, 22]\) m → MAD 기반 이상치 (`MAD_MULT`, `MAD_MIN_BAND`) 등 **단계별 마스크**를 시각화한다(`uwb_filter_sequential_filtering.png`).

### 1.3 우리가 얻은 결과·해석 (프로젝트 관점)

- **장치별 바이어스 표**는 `fusion_realtime_sanitize`의 **`PHASE1_*_BIAS_M`과 동일 정의**로 동기화되어야 한다. → 전체 파이프라인의 **거리 오프셋 일관성**의 근거.
- **분산 P95 근처 값**이 `VAR_CAP_WIFI_FUSION`, `VAR_CAP_UWB_FUSION` 주석과 연결된다 — “융합에서 분산이 너무 크면 신뢰 하향” 기준.
- **Clean UWB 잔차 평균**(순차 필터 후)은 별도 CSV로 남을 수 있으나, docstring에 명시된 대로 **PHASE1을 대체하지 않는다**(조건부 표본). 게이트 그리드 **참고**용.
- **Analysis B** 콘솔 출력: 특정 노드에서 UWB NaN이 몰리거나 `median/true` 비율이 크면 **해당 구역 다경로·센서 배치**를 의심하고, 이후 **게이트 \(\tau\)** 탐색 상한 등에 반영했다(v9 주석의 dense grid 등).

### 1.4 실행·산출

```text
py -3 scripts/sensor_spatial_profiler.py [all|spatial|uwb]
```

기본 `outputs/sensor_profiler_eda_*.png`, `outputs/uwb_filter_sequential_filtering.png`. 환경변수: `MEDIAN_XLSX`, `VARIANCE_XLSX`, `PROFILER_OUT_DIR`, `UWB_FILTER_PNG`.

---

## 2. `validation_calibration_filter_scatter.py`

### 2.1 목적

**검증** median에 대해, 각 센서·각 행에서 **참 거리 vs 측정 median** 산점도를 **필터 단계별**로 겹쳐 그린다.

### 2.2 단계 (코드 주석과 동일)

1. **원본** median.  
2. **`HARDWARE_CALIB_BIAS_M` 차감**(융합과 동일 오프셋).  
3. **분산 cap**: `VAR_CAP_UWB_FUSION` / `VAR_CAP_WIFI_FUSION` 이하만 유효로 간주하는 단계 시각화.  
4. **물리 게이트**: \(0.5 < m < 22\) m.

### 2.3 우리가 얻은 결과·해석

- 단계가 진행될수록 **y=x 대각선 주변으로 점군이 모이는지**로, 바이어스·분산 cap이 **거리 척도를 얼마나 정렬하는지**를 육안 검증한다.
- UWB/Wi‑Fi 중 **특정 기기만 대각선에서 벗어나면** 해당 AP/앵커의 **별도 캘리브 또는 게이트** 필요성을 뒷받침한다.

### 2.4 산출

`outputs/validation_calibration_filter_scatter.png`

---

## 3. `indoor_fusion_pipeline_JWT.py`

### 3.1 목적

팀원 **장원태** 알고리즘을 레포 데이터 규격에 맞춰 **Train 피팅 → Validation 평가**한다.

### 3.2 기법 (모듈 docstring 요지)

- AP별 **RANSAC + Isotonic** 등 **Wi‑Fi 거리 보정**  
- **이상치 점수**: RANSAC 점수, **삼각부등식** 위반 등을 결합  
- 위치는 **Huber `least_squares` 삼변** 등으로 산출

### 3.3 우리가 얻은 결과·해석 (실행 로그 기준 예시)

콘솔에 **학습 in-sample 거리 RMSE**, **검증 거리 RMSE**, **검증 위치 RMSE**가 표 형태로 출력된다. 일반적으로:

- **보정거리_soft_weight** 경로가 **원본거리_Huber** 대비 **위치 RMSE·P90을 개선**하는 패턴이 관측될 수 있다.
- **hard_remove**는 일부 실행에서 **위치 RMSE가 크게 악화**되어, “강한 제거”가 **삼변 가용 앵커 부족**으로 역효과 날 수 있음을 시사한다.

### 3.4 산출

기본 `outputs/jwt_wifi_corrected_outlier_score_position_rmse.xlsx` (스크립트 내 경로 상수 확인).

---

## 4. 문헌·재현

분석 스크립트는 **검증 `True_X/Y`**를 산점도 기준선으로 쓰므로, **EDA 목적**과 **최종 무결 리포트**를 구분해 기술한다.
