# Wi-Fi & UWB 융합 실내 측위 프로젝트

`Variance-Weighted Huber Trilateration`을 중심으로, K-NN 잔차 보정(v1/v2/v3)과 **순수 수학 기반 V4**, **데이터 분석·필터 시각화 스크립트**를 함께 제공합니다.

---

## 1) 프로젝트 구조

```text
midterm_project/
├─ data/
│  ├─ train/
│  │  ├─ median_test_kgh_corrected.(csv|xlsx)   # 모든 파이프라인 공통 학습·개발 입력
│  │  └─ variance_test_kgh_corrected.(csv|xlsx)
│  ├─ validation/
│  │  ├─ median_validation.xlsx
│  │  └─ variance_validation.xlsx
│  ├─ archive_unused/
│  │  └─ (mean/원본 등)
│  └─ reference/
│     ├─ fixed_median_true_distance_error_rate.csv
│     └─ theoretical_distances.xlsx
├─ scripts/
│  ├─ indoor_fusion_pipeline.py           # v1: KNN 잔차 보정
│  ├─ indoor_fusion_pipeline_v2.py        # v2: IRLS + KNN
│  ├─ indoor_fusion_pipeline_v3.py        # v3: Step A/B, KNN
│  ├─ indoor_fusion_pipeline_v4.py        # v4: 캘리브·비대칭 융합·경계 Huber (KNN 없음) + 단계 진단 CSV
│  ├─ indoor_fusion_pipeline_v5.py        # v5: 융합 완화·가중 삼변·조건부 Ridge (KNN 미사용)
│  └─ sensor_spatial_profiler.py          # 공간 EDA + UWB 순차 필터 (통합)
├─ outputs/
│  ├─ v1_validation_predictions.csv
│  ├─ v1_validation_error_cdf.png
│  ├─ v1_validation_error_vector_map.png
│  ├─ v2_validation_predictions.csv
│  ├─ v2_step_rmse_report.csv
│  ├─ v2_validation_error_cdf_steps_abc.png
│  ├─ v3_validation_predictions.csv
│  ├─ v3_step_rmse_report.csv
│  ├─ v3_node_error_analysis.csv
│  ├─ v3_validation_error_cdf_stepA_stepB.png
│  ├─ v3_validation_multistage_map.png
│  ├─ v4_validation_predictions.csv
│  ├─ v4_per_sample_diagnostics.csv
│  ├─ v4_phase_aggregate.json
│  ├─ v4_fusion_rule_by_anchor.csv
│  ├─ v4_predictions_with_phase_columns.csv
│  ├─ v4_error_cdf.png
│  ├─ v4_confidence_map.png
│  ├─ v5_validation_predictions.csv
│  ├─ v5_train_ridge_report.json
│  ├─ v5_validation_fusion_summary.csv
│  ├─ v5_error_cdf.png
│  ├─ v5_confidence_map.png
│  ├─ sensor_profiler_eda_calibration_scatter.png
│  ├─ sensor_profiler_eda_error_boxplot.png
│  ├─ sensor_profiler_eda_spatial_error_heatmap.png
│  └─ uwb_filter_sequential_filtering.png
└─ README.md
```

모든 파이프라인·분석 스크립트의 **산출물(PNG, CSV)은 기본적으로 `outputs/`** 에 저장됩니다.

---

## 2) 데이터 파일 매핑

### 학습(Train) — 테스트(원본) 데이터
- **전 버전 공통**(v1~v10 및 `sensor_spatial_profiler`): `data/train/median_test_kgh_corrected` + `variance_test_kgh_corrected`
- 우선 확장자: **`.csv` → `.xlsx`/`.xls`** (같은 쌍 이름이 있어야 함). `_2` 변형표는 과제 규격에 포함하지 않습니다.

### 검증(Validation)
- `data/validation/median_validation` + `variance_validation` (동일하게 **csv 우선·xlsx 대체**)  
  (일부 파이프라인은 검증 폴더가 없으면 Train 원본으로 대체 출력만 수행합니다.)

### 기타
- `data/archive_unused/*`: 파이프라인에서 사용하지 않는 보관 데이터
- `data/reference/*`: 참고용

### 엑셀 헤더
- 일부 파일은 **첫 행이 컬럼명 문자열**인 형식(`header=False` 저장)과, pandas 기본 헤더 형식이 혼재합니다. `sensor_spatial_profiler`, `indoor_fusion_pipeline_v4`, `indoor_fusion_pipeline_v5`는 자동 판별 로더를 사용합니다.
- `d1044709` 컬럼명 변형은 코드에서 `d1044709`로 통일합니다.

---

## 3) 알고리즘 개요 (v1: `indoor_fusion_pipeline.py`)

1. **전처리**: `Node_x`, `Node_y` × 0.6 m → `True_X`, `True_Y`  
2. **UWB 결측**: `KNNImputer`(Wi-Fi+UWB median 맥락)  
3. **분산 가중 Huber 삼변측량** (`least_squares`, `loss='huber'`)  
4. **K-NN 잔차 보정** → `Final_X`, `Final_Y`  
5. **평가**: RMSE, MAE + CDF, 오차 벡터 맵 (`outputs/v1_*.csv|png`)

---

## 4) v2 (`indoor_fusion_pipeline_v2.py`)

- **Step A**: Smart Guess 삼변측량  
- **Step B**: IRLS(잔차 3 m 초과 센서 가중 0 후 재추정)  
- **Step C**: K-NN 잔차 보정  
- 단계별 RMSE/MAE → `outputs/v2_step_rmse_report.csv`  
- CDF 오버레이 → `outputs/v2_validation_error_cdf_steps_abc.png`

---

## 5) v3 (`indoor_fusion_pipeline_v3.py`)

- KNN으로 UWB 대치 피처 생성 후 **Step A(순수 삼변측량)** → **Step B(KNN 2D 좌표 잔차 보정)**  
- `outputs/v3_validation_predictions.csv`, `v3_step_rmse_report.csv`, `v3_node_error_analysis.csv` 등

---

## 6) v4 (`indoor_fusion_pipeline_v4.py`) — K-NN 없음

- **Phase 1**: 고정 Bias 보정 + `0.5 < d < 22` m 하드 클립  
- **Phase 2**: 앵커별 UWB/Wi-Fi 비대칭 융합 + 분산 `> 6` 시 UWB 폐기  
- **Phase 3**: `d_fused` 6개로 **경계 제약** `[0,12]×[0,18]` Huber 삼변측량  
- **Phase 4**: 기하 잔차 RMSE로 `Confidence` High/Low  
- **진단 산출물** (원인 분석용): `v4_per_sample_diagnostics.csv`, `v4_phase_aggregate.json`, `v4_fusion_rule_by_anchor.csv`, `v4_predictions_with_phase_columns.csv`  
- 예측·그림: `v4_validation_predictions.csv`, `v4_error_cdf.png`, `v4_confidence_map.png`

---

## 6-1) v5 (`indoor_fusion_pipeline_v5.py`) — v4 진단·v1~v3 교훈 반영

- **융합**: Wi-Fi만 없을 때 **UWB 단독** 허용(v4는 `nan`); UWB 분산 `>6` 은 **가중 감쇠**로 하드 폐기 완화.  
- **비대칭**: 비율 임계 `2.2`, 가중 `0.2/0.8`, `0.35/0.65` 로 과도한 UWB 억제 완화.  
- **삼변측량**: 앵커별 분산 기반 가중 + `sqrt(w)*잔차` Huber, `f_scale` 고정(1.5).  
- **보정**: K-NN 대신 Train에서만 **Ridge(`[1,x,y]` → 잔차)** 적합; Train 평균 위치 오차가 **3% 이상** 줄어들 때만 검증에 적용.  
- 산출: `v5_validation_predictions.csv`, `v5_train_ridge_report.json`, `v5_validation_fusion_summary.csv`, `v5_error_cdf.png`, `v5_confidence_map.png`  
- Train은 `median/variance_test_kgh_corrected` 원본만 사용; 검증은 `validation/` (없으면 train으로 대체).

---

## 7) 분석 스크립트 (`sensor_spatial_profiler.py`)

- 입력: Train 원본과 동일 (`median_test_kgh_corrected.*`, `variance_test_kgh_corrected.*`, `fusion_realtime_sanitize` 규칙)
- 실행 후 **`outputs/sensor_profiler_robust_bias_table.csv`**에 Analysis A 표가 저장됩니다. 모든 파이프라인 공통 거리 편향(`fusion_realtime_sanitize.py`의 `PHASE1_*_BIAS_M`)은 이 Robust Bias 정의와 **동일한 수치로 맞춰 둠**(Train 변경 시 프로파일러 재실행 → CSV와 코드 상수 재동기화).  
- **모드** (첫 인자, 생략 시 `all`):  
  - `all`: **spatial**(강건 편향·공간 EDA·`sensor_profiler_eda_*.png`) 후 **uwb**(UWB long melt → Hard→Var→MAD, `uwb_filter_sequential_filtering.png`)  
  - `spatial`: EDA만  
  - `uwb`: UWB 순차 필터만  
- 예: `py -3 scripts/sensor_spatial_profiler.py spatial`  
- 환경 변수: `MEDIAN_XLSX`, `VARIANCE_XLSX`, `PROFILER_OUT_DIR`, `UWB_FILTER_PNG`

---

## 8) 센서/앵커 좌표 (타일 × 0.6 m)

### UWB
- `110394ab`: (1, 4), `e63ce2f`: (20, 7), `8e610981`: (5, 15)  
- `d10485af`: (4, 27), `d1044709`: (15, 14), `4e610206`: (14, 24)

### Wi-Fi (동일 위치 쌍)
- `SW_11`, `SW_first_team`, `볼링공`, `SW_4`, `SW_6`, `SW_5`

---

## 9) 실행 방법

프로젝트 루트에서:

```bash
py -3 scripts/indoor_fusion_pipeline.py
py -3 scripts/indoor_fusion_pipeline_v2.py
py -3 scripts/indoor_fusion_pipeline_v3.py
py -3 scripts/indoor_fusion_pipeline_v4.py
py -3 scripts/indoor_fusion_pipeline_v5.py
py -3 scripts/sensor_spatial_profiler.py          # 기본: all
py -3 scripts/sensor_spatial_profiler.py spatial
py -3 scripts/sensor_spatial_profiler.py uwb
```

필수 패키지(예시):

```bash
py -3 -m pip install pandas numpy scipy scikit-learn matplotlib openpyxl
```

---

## 10) 실행 결과 요약 (참고 수치)

환경·데이터 버전에 따라 달라질 수 있습니다.

| 스크립트 | 비고 |
|----------|------|
| v1 | RMSE ≈ 7.89 m (참고) · 산출 `v1_validation_predictions.csv` 등 |
| v2 | Step A/B/C 단계별 지표 → `v2_step_rmse_report.csv` |
| v3 | Step A vs B 비교 → `v3_step_rmse_report.csv` |
| v4 | 터미널 RMSE/MAE + 단계 진단 CSV·JSON (`v4_*`) |
| v5 | v4 대비 완화 융합·가중 삼변·조건부 Ridge; `v5_train_ridge_report.json` 등 |
| sensor_spatial_profiler | 모드별 EDA PNG + UWB 필터 PNG·터미널 요약 |

---

## 11) 향후 개선 제안

- Huber `f_scale`, 분산 컷(6.0), 비대칭 2배 규칙, V4 품질 임계(2.5 m) 튜닝  
- `loss='soft_l1'` 비교, 구역별 히트맵 자동화  
- CI용 최소 샘플 스모크 테스트(Train csv 고정 행 검증 등)

