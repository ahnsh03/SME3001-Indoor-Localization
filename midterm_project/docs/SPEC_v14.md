# V14 상세 명세 (Turbo + JWT Wi-Fi 가중 실험)

[← 색인](SPEC_INDEX.md)

---

## 1) 버전 목표와 V13/V12 대비 변경

`indoor_fusion_pipeline_v14.py`의 모듈 docstring 요지:

- **Step A·UWB 처리**는 v12_turbo 계열과 동일 구조를 유지.
- 차이는 Wi-Fi 융합 경로에 **JWT 계열 가중/보정**(Isotonic, Score RANSAC, 삼각부등식)을 넣는 점.

즉, 목적은 “Turbo 엔진 위에서 JWT Wi-Fi 신뢰도 모델을 결합하면 B/C가 더 좋아지는가”를 검증하는 것이다.

---

## 2) 핵심 기법·수식

### 2.1 JWT soft weight

Wi-Fi 가중은 Turbo형 \(1/\mathrm{var}\)에 JWT 점수를 추가로 적용:

\[
w^{jwt} = \frac{1}{1 + w_t \cdot s_{triangle} + w_r \cdot s_{ransac}}
\]

(\(w_t, w_r\)는 Optuna 탐색).

### 2.2 Isotonic/AP 마스크

`iso_ap_mask_bits`로 AP별 Isotonic 보정을 선택적으로 켠다.

### 2.3 Step B/C

게이트·인플레이트·Huber·IRLS는 V12/Turbo 계열과 동일 형태:

\[
\left| \|\mathbf{p}_A-\mathbf{a}^{UWB}\|-\tilde d^{UWB}\right| \le \tau,\quad
w^{UWB}\propto \frac{1}{\lambda\sigma^2+b_u+\epsilon}
\]

---

## 3) `outputs/v14*` 전체 분석

총 6개 파일:

- `v14_summary.json`
- `v14_predictions.csv`
- `v14_optuna_trials.csv`
- `v14_grid_phaseA.csv`
- `v14_wifi_calibration_meta.json`
- `v14_jwt_ablation_validation.csv`

### 3.1 `v14_summary.json`

- `integrity`: Train KFold Step C 목적 최소화, Validation truth 최종 1회.
- `optuna_n_trials=55`, `optuna_n_jobs=1`, `wall_clock_sec=342.21`.
- Train CV best: `mean_CV_RMSE_step_C_m = 2.5280`.
- 검증:
  - Step A 1.5104
  - Step B 1.8562
  - Step C 1.5583
- 선택 HP:
  - `w_jwt_triangle=4.1622`, `w_jwt_ransac=1.1679`
  - `gate=1.5591`, `inflate=12.5677`, `huber_f_fusion=1.1065`, `irls=2.6723/2`
  - `iso_ap_mask_bits=1`, `jwt_wifi_soft_enable=true`

### 3.2 `v14_optuna_trials.csv`

- 55 trials
- best `mean_cv_rmse_step_c_m = 2.5280089` (trial 1)
- 분포 폭이 큼: median 약 3.037, max 4.408  
  → JWT/iso 조합이 잘못 맞으면 성능이 크게 악화되는 trial이 많다는 뜻.

### 3.3 `v14_predictions.csv`

100행 중 유효 99행:

- Step A: RMSE 1.5104, P90 2.3631
- Step B: RMSE 1.8562, P90 2.6106, max 9.8162
- Step C: RMSE 1.5583, P90 2.6152

Step C가 B를 완화하지만 A보다 여전히 높다.

### 3.4 `v14_grid_phaseA.csv`

- 4행, Phase A proxy 후보.
- 최저 `train_CV_mean_RMSE_A_m = 2.651844...` (identity, bias_sub, hf=1.0).

### 3.5 `v14_wifi_calibration_meta.json`

- `iso_ap_bitmask=1`
- `use_iso_bits=[1,0,0,0,0,0]`
- `iso_n_nodes=[25,0,0,0,0,0]`
- `coeff_ab_snapshot`는 전 AP (1.0, 0.0)

즉, 현재 최적점은 “1번 AP에만 isotonic 적용, 선형 보정계수는 항등” 구조다.

### 3.6 `v14_jwt_ablation_validation.csv` (핵심)

7개 stage 비교:

- **S0 (iso=0, JWT off)**: B 1.4330 / C 1.4435  ← 이 조합이 사실상 최상
- S0b (삼각만): B 1.4537 / C 1.4731
- S0c (RANSAC만): B 1.6918 / C 1.6882
- S1 (iso만): B 1.6928 / C 1.7024
- S2 (iso+삼각): B 1.7035 / C 1.7202
- S2b (iso+RANSAC): B 1.8372 / C 1.5272
- **S3 (iso+JWT full = 최종 선택)**: B 1.8562 / C 1.5583

해석: 현재 데이터에서는 JWT soft 가중과 isotonic이 Step B를 크게 악화시키며, Step C가 일부 복구해도 baseline(S0)보다 못하다.

---

## 4) 모델링 해석

1. Train CV 목적은 V13/V12-Turbo와 비슷한 수준(2.53대)이지만, 검증 순전파는 크게 나빠진다.
2. 특히 `jwt_ablation_validation.csv`가 “성능 저하의 주범은 JWT/iso 조합”임을 직접 보여 준다.
3. 따라서 V14는 제출 최종본이라기보다 **모듈 기여도 분해 실험**으로 분류하는 것이 타당하다.

---

## 5) 관련 스크립트

- `scripts/indoor_fusion_pipeline_v14.py`

