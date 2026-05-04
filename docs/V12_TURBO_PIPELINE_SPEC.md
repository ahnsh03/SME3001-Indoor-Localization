# V12 Turbo 실내 융합 파이프라인 — 상세 명세서

본 문서는 `scripts/indoor_fusion_pipeline_v12_turbo.py`의 설계·동작을 팀워크스페이스용으로 정리한 것입니다. **최종 선정 코드는 V13**이며, V12 Turbo는 **같은 실험 무결성·수식 계열을 유지한 고속 베이스라인**으로 참고하면 됩니다. (V13이 본 파일의 Step B/C Numba 함수를 재사용합니다.)

원문 요지는 스크립트 상단 모듈 docstring과 동일합니다.

---

## 1. 설계 목적

- **`indoor_fusion_pipeline_v12_strict.py`와 동일한** 데이터 무결성·K‑Fold 교차검증·Phase A/B/C **수식(게이트, UWB 분산 inflate, Wi‑Fi 보정, Tukey×Huber IRLS, 바운드 제약 삼변)** 을 목표로 둠.
- **Optuna**(하이퍼 탐색) **+ NumPy/Numba 벡터화** **+ Trial 병렬화** 로 런타임을 줄임.
- Step B/C의 삼변은 **바운드 Huber 가중 잔차 기반 가우스‑뉴턴 + Huber IRLS**를 `@njit`로 구현하며, 반복 수는 `TRILAT_GN_ITERS`(기본 22)로 고정.

---

## 2. 한 줄 요약

Train에서 **Wi‑Fi median 보간** 후, **Phase A 상위 K개**(보정 종류 조합 메타만 다름)에 대해 K‑Fold 캐시를 만들고, **Optuna가 Train 평균 Step C RMSE**를 최소화하는 `phase_a_rank`·게이트·inflate·fusion Huber·IRLS 임계를 선택한다. 검증에서는 **동일하게 V3 동급 Wi‑Fi 안심 Step A** 한 종류만 쓰고, 선택된 보정 계수로 B/C만 수행하여 **단일 포워드**로 지표를 낸다.

---

## 3. 실행 방법

프로젝트 루트에서:

```powershell
py -3 .\scripts\indoor_fusion_pipeline_v12_turbo.py
```

CDF/맵 등 PNG 생략:

```powershell
py -3 .\scripts\indoor_fusion_pipeline_v12_turbo.py --no-plots
```

**필수 패키지**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `numba`, `optuna`  
**프로젝트 모듈**: `fusion_realtime_sanitize`, `indoor_fusion_pipeline_v8`~`v12`, `fusion_realtime_sanitize`(경로 분해)·`WifiOnlyTrilateration` 등.

---

## 4. Phase A 처리의 실제 의미 (중요)

메인에서 `FEAT_MODES` × `v12_calibration_spec_list()` × `HUBER_F_WIFI_GRID`의 **카테시안 그리드**를 만들되, 각 조합 행에는 **공통값**으로 `cv_mean_rmse_v3_step_a(...)` 결과(폴드 평균 **V3 동급 Wi‑Fi‑only 삼변** Step A RMSE)가 기록된다. 정렬키는 `(그 공통값, calib 종류, feat, hf_wifi)` 순이므로, **실제로 서열을 가르는 것은 보정 종류 등 메타**(동일 프로시일 때 값이 같으면)이다.

폴드 캐시 `_build_one_fold_sequential_ranks`에서는 각 상위 순위마다 해당 폴드 Train으로만 **`fit_robust_calibration_per_ap`** 하여 `coeff_ab` 등을 만들지만, **가이드 XY `xy_a`는 모든 순위 동일하게** `predict_step_a_v3_wifi_equivalent(te_df, loc)` 결과를 가리킨다. 따라서 Train CV 목적값에서 바뀌는 것은 **융합(Step B/C)에 들어가는 선형 보정 계수와 `feat`·`huber_f_wifi`** 이지, 폴드 내 Step A 좌표 궤적 자체가 순위별로 달라지지는 않는다.

**검증 포워드**에서도 Step A는 `predict_step_a_v3_wifi_equivalent(val_imp, loc)` 단일 경로이다. 요약하면: **표기상 “Wi‑Fi 보정 삼변”과 관련된 그리드는 B/C 입력 가중 거리 계산용 보정 선택**에 가깝고, 검증 산출되는 **Step A RMSE 플롯 컬럼은 V3 동급 경로 하나**이다.

---

## 5. 단계별 정의 (B/C)

공통 입력: 각 시점별 UWB 중앙값/분산/원 NaN 플래그, Wi‑Fi 중앙값/분산, AP 좌표, 강건 바이어스, `FusionLocalizerV8`에서 온 바운드 및 `EPS_W` 등.

### Step B

- 가이드 `xy_guide` 근처에서 **UWB:** 측정 거리와 기하 거리 차가 `gate_threshold_m` 초과면 **제외(하드 게이트)**.
- UWB 분산은 `uwb_variance_inflate`(≥1)로 스케일 후 `1/(var+bias)` 가중.
- **Wi‑Fi:** `feat_id`에 따라 raw에서 바이어스 감산 여부 반영 후 `d_c = A*xf + B` 선형 보정 거리 사용, 분산 역가중.
- 앵커 수에 따라 바운드 **Huber** GN 한 번 또는 Wi‑Fi만으로 축약.

### Step C

- Step B와 동일하게 앵커를 모은 뒤 **`irls_refine_xyz_njit`** 로 **Tukey bisquare × Huber hinge** 형태로 가중치를 업데이트하며 고정 회수 **`irls_max_iterations`(코드 상 2)** 반복.

삼변 핵은 `solve_trilat_bounded_huber_njit` — V12 `solve_trilat_array`와 **같은 목적함수**를 의도한다고 주석에 명시되어 있다.

---

## 6. 데이터 무결성

| 항목 | 내용 |
|------|------|
| Optuna 목표 | 각 Trial마다 Train K‑Fold **평균 Step C RMSE** 최소화 |
| 검증 `True_X/Y` | 선택·목적값에 사용하지 않음. 전체 학습 재피트 후 **최종 1회** 보고 |
| 폴드 내 보정 | 각 폴드의 학습 행만으로 AP 보정 학습 후, 해당 폴드 테스트 행에 적용 |

`fold_cache_to_numpy_only` 로 워커에 넘기는 것은 **DataFrame 없이 NumPy 패킹**이라 Optuna 병렬 시 피클 부담을 줄이기 위함이다.

---

## 7. 상수 및 환경 변수

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `ART_PREFIX` | `v12_turbo` | 산출물 접두사 |
| `N_OPTUNA_TRIALS` | `35` | 시행 수 |
| `OPTUNA_N_JOBS` | `$V12_TURBO_OPTUNA_JOBS` 또는 `-1` | Windows에서는 `8` 등 상한 권장 |
| `OPTUNA_SEED` | `42` | TPE 샘플러 시드 등 |
| `TRILAT_GN_ITERS` | `22` | 삼변 GN 반복 |
| `PLOT_DPI` | `110` | PNG DPI |
| `TOP_A_CANDIDATES` | `3`(v12 모듈) | Phase A 그리드에서 상위 순위 후보 개수 |

K‑Fold 폴수: `resolve_kfold_n_splits` 결과와 `MAX_KFOLD_SPLITS`(3) 중 작은 값.

---

## 8. Optuna 탐색 공간 (`make_objective`)

- `phase_a_rank`: 정수 `0 … TOP_A_CANDIDATES−1`
- `gate_threshold_m`: `1.25`–`2.95` (연속)
- `uwb_variance_inflate`: `10.0`–`24.0` (연속)
- `huber_f_fusion`: `1.0`–`1.35` (연속)
- `irls_residual_thresh_m`: `2.2`–`3.1` (연속)
- `irls_max_iterations`: **항상 `2`**(Trial에서 고정)

TPESampler에 `constant_liar=True`(실험적 옵션) 사용 — 실행 시 ExperimentalWarning 가능.

---

## 9. 산출물 (`outputs/` 접두사 `v12_turbo_`)

| 파일 | 내용 |
|------|------|
| `v12_turbo_summary.json` | 무결성 문구, Train CV 지표, 검증 단일 포워드 Step A/B/C RMSE·MAE, 선택 HP |
| `v12_turbo_predictions.csv` | 검증별 Step A/B/C 좌표·오차 |
| `v12_turbo_grid_phaseA.csv` | Phase A 그리드 각 행(공통 proxy RMSE 포함) |
| `v12_turbo_optuna_trials.csv` | 완료 Trial별 파라미터·목적값 |
| `v12_turbo_wifi_calibration_coefs.csv` | 최종 전체 학습 기준 보정 계수 |

**플롯**(기본): `v12_turbo_cdf_steps.png`, `v12_turbo_map_true_vs_predicted_steps.png`, `v12_turbo_map_error_vectors_stepC.png`, `v12_turbo_map_stepC_pairwise_truth_pred.png`

---

## 10. 참고 재현 예시 수치 (`outputs/v12_turbo_summary.json`)

한 번의 런 기준(데이터 변경 시 달라짐):

- Train CV mean Step C RMSE ≈ **2.53 m**
- 검증 Step A / B / C RMSE ≈ **1.51 / 1.44 / 1.45 m**
- Wall-clock(그 런) ≈ **28 s**

같은 데이터에서 팀 최종안 **V13**은 검증이 더 우수했다. 세부 표는 [`V13_FINAL_PIPELINE_SPEC.md`](V13_FINAL_PIPELINE_SPEC.md)를 본다.

---

## 11. V13과의 관계

| 항목 | V12 Turbo | V13 |
|------|-----------|-----|
| Step A 표현 | 검증 단일 포워드는 **V3 동급** 고정 | **보정 삼변**과 **V3 프로필** **컨벡스 블렌드**(+선택 KNN) |
| Step B/C 구현 | 본 파일 내 Numba | **동일 함수 import** 재사용 |
| 탐색 차원 | 게이트·inflate·fusion·IRLS·`phase_a_rank` | 블렌드·V3 프로필·캘리브 순위 추가 등 더 넓음 |
| 속도傾向 | 비교적 짧음 | 캘리브 탐색·폴드 작업 많아 더 김 |

---

## 12. 운영 시 주의

- **목표_wall_clock “10 s 미만”** 은 코드 주석대로 데이터 크기·sklearn 보정 빈도·머신에 강하게 의존.**보장 KPI가 아니다.**
- `constant_liar` 인터페이스는 앞으로 바뀔 수 있는 **실험적 Optuna 기능**이다.
- Phase A 순위 선택이 **실질적으로 B/C 계수 선택**이라, 보고서에는 “Step A 블렌드”보다는 **Wi‑Fi 융합 계수 패키지 선택** 관점으로 풀이하는 것이 정확하다.

---

*작성 기준: `indoor_fusion_pipeline_v12_turbo.py`, `outputs/v12_turbo_summary.json`.*
