# V13 상세 명세 (V13 + V13-fix)

[← 색인](SPEC_INDEX.md) · [V13 집중본](V13_FINAL_PIPELINE_SPEC.md)

---

## 1) 버전 목표와 V12 대비 변경

V13의 핵심은 V12 Turbo의 빠른 Step B/C 엔진을 유지하면서, Step A 후보 공간을 확장해 **Train-CV 기준 최종 Step C**를 더 낮추는 것이다.

V12 대비 주요 변화:

1. 캘리브 카탈로그 확장(`calibration_catalog_size=13`).
2. V3 프로필 다중 후보(`v3_profiles`).
3. Step A 블렌드 \(\alpha\)와 (조건부) KNN 잔차 후보를 Optuna에 포함.
4. 목적함수는 일관되게 **Train KFold mean Step C RMSE**.

---

## 2) 공통 수식 요약

### Step A (혼합 가이드)

\[
\mathbf{p}_A = (1-\alpha)\,\mathbf{p}_{A}^{V3profile} + \alpha\,\mathbf{p}_{A}^{calib}
\]

조건에 따라 KNN 잔차 \((\Delta x,\Delta y)\)를 더해 보정한다.

### Step B/C

V12 계열과 동일한 게이팅·분산 인플레이트·IRLS 구조를 사용한다:

\[
\left| \|\mathbf{p}_A-\mathbf{a}^{UWB}\|-\tilde d^{UWB}\right|\le\tau
\]

만 통과시키고, Step C는 Tukey×Huber IRLS로 재가중한다.

---

## 3) `outputs/v13*` 파일 전체 분석

총 12개 파일:

- `v13_summary.json`
- `v13_predictions.csv`
- `v13_optuna_trials.csv`
- `v13_grid_phaseA.csv`
- `v13_grid_phaseBC.csv`
- `v13_grid_phaseA_calib_catalog.csv`
- `v13_wifi_calibration_coefs.csv`
- `v13_fix_summary.json`
- `v13_fix_predictions.csv`
- `v13_fix_optuna_trials.csv`
- `v13_fix_grid_phaseA_calib_catalog.csv`
- `v13_fix_wifi_calibration_coefs.csv`

### 3.1 V13 본선 (`v13_summary.json`)

- `integrity`: Train KFold mean Step C RMSE만 최소화, Validation truth는 최종 1회.
- `train_cv_best_step_c_rmse_m = 2.5306718926`
- 검증:
  - Step A 1.4617
  - Step B 1.4194
  - Step C 1.4252
- 선택 HP:
  - `calibration_rank=1`, `calib=identity`, `feat=bias_sub`
  - `v3_profile_index=0`, `blend_calib_alpha=0.5`
  - `gate=1.4562`, `inflate=12.3001`, `huber_f_fusion=1.2`, `irls=3.1/2`

### 3.2 Optuna 시도 분포 (`v13_optuna_trials.csv`)

- 42 trials
- best `mean_cv_rmse_step_c_m = 2.5306718926` (trial 37)
- 분위수(요약): min 2.531 / median 2.611 / max 2.758  
  → 좁은 최저대역과 넓은 후미를 가지는 탐색 분포.

### 3.3 그리드 CSV 해석

- `v13_grid_phaseA.csv`: pure calibrated Step A의 CV 지표.
  - 최저 `train_CV_mean_RMSE_step_A_calibrated_m = 2.571483` (huber, bias_sub, hf=1.35)
- `v13_grid_phaseBC.csv`: B/C 조합 256행.
  - 최저 `train_CV_mean_RMSE_C_m = 2.513895` (phaseA_rank=0, gate=1.65, inflate=12, hf_fusion=1.35, irls=3.1/2)
- `v13_grid_phaseA_calib_catalog.csv`: 카탈로그 13개 순위표.

### 3.4 예측 CSV (`v13_predictions.csv`)

100행 중 유효 99행:

- Step A: RMSE 1.4617, P90 2.2715
- Step B: RMSE 1.4194, P90 2.3115
- Step C: RMSE 1.4252, P90 2.3756

Step B가 최저 RMSE, Step C는 평균은 소폭 상승하나 MAE/P90 측면의 균형을 볼 여지가 있다.

---

## 4) V13-fix (동일 목적, 구현/정책 고정 해제)

### 4.1 코드 의도 (`v13_fix_summary.json` `variant_notes`)

1. Numba 의존을 `fusion_turbo_numba_core.py`로 단일화.
2. `forced_*` 삽입 없이 **Train pure StepA CV 순위만** 사용.
3. Optuna enqueue warmstart 비활성.

### 4.2 결과 비교

- Train CV best: **2.526847** (V13보다 약간 낮음)
- 검증:
  - A 1.4982
  - B 1.4340
  - C 1.4411

즉, Train 목적값은 개선됐지만 검증 A/B/C는 본선 V13보다 불리하다.

`v13_fix_optuna_trials.csv`도 42 trials이며 best row:

- `calibration_rank=10`
- `v3_profile_idx=1`
- `blend_calib_alpha=0.0`
- `gate=1.5424`, `inflate=15.1222`, `huber_f_fusion=1.2`.

---

## 5) 해석 요약

1. V13은 “Train 목적 최소화 + 검증 순전파 성능”을 동시에 달성한 대표 실행이다.
2. V13-fix는 구현 독립성과 선택 정책의 순수성을 강화했으나, 현재 산출물 기준 검증 성능은 본선 V13보다 낮다.
3. 따라서 보고서에서는 **최종 제출 모델: V13**, **정책/구현 대조 실험: V13-fix**로 역할 분리하는 것이 가장 명확하다.

---

## 6) 관련 스크립트

- `scripts/indoor_fusion_pipeline_v13.py`
- `scripts/indoor_fusion_pipeline_v13_fix.py`

---

## 7) 후속 파이프라인 V15 (참고)

**Train-OOF 게이트**, **composite/2단계 Optuna**, **`V15_TUNING_PROFILE`(balanced/defensive/v13_fix)** 로 v13-fix 골격을 확장한 실험용 파이프라인이다. 세부 무결성·환경변수는 **[`V15_PIPELINE_SPEC.md`](V15_PIPELINE_SPEC.md)** 를 본다.
