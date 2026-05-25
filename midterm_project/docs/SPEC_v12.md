# V12 상세 명세 (V12 기본 + Strict/Fast/Fast2/Turbo/Strict Parallel)

[← 색인](SPEC_INDEX.md) · [V12 Turbo 집중본](V12_TURBO_PIPELINE_SPEC.md)

---

## 1) 버전 목표와 v11 대비 변경점

`indoor_fusion_pipeline_v12.py`의 모듈 docstring대로 V12는 다음 3개를 결합한 통합안이다.

1. **V11 계열 Step A**: Wi-Fi 강건 보정 기반 삼변(안심 좌표).
2. **V9 계열 Step B**: Wi-Fi 가이드에서 UWB 기하 게이트.
3. **V11 계열 Step C**: Tukey×Huber IRLS 재가중.

핵심 문제의식은 “UWB 저분산·대오차(조용한 암살자)로 단순 분산가중 융합이 붕괴”이며, Pure Wi-Fi 바닥(약 1.5 m대)을 기준으로 **선별적 UWB 사용**을 설계한 것이다.

---

## 2) 공통 수식 (V12 계열)

### Step A (Wi-Fi 가이드)

Wi-Fi 거리 보정 후 삼변으로 \(\mathbf{p}_A=(x_A,y_A)\) 추정.

### Step B (게이팅 융합)

UWB 앵커 \(k\)에 대해:

\[
\left| \|\mathbf{p}_A-\mathbf{a}^{UWB}_k\| - \tilde d^{UWB}_k \right| > \tau
\]

이면 해당 앵커 제외. 통과 UWB는 분산 인플레이트 \(\lambda\):

\[
w_k^{UWB}\propto \frac{1}{\lambda\sigma_k^2+b_u+\epsilon},\quad
w_j^{WiFi}\propto \frac{1}{\sigma_j^2+b_w+\epsilon}.
\]

### Step C (IRLS)

Step B 결과를 초기값으로 Tukey bisquare \(\times\) Huber hinge 가중을 반복 갱신:

\[
\mathbf{x}^{(t+1)}=\arg\min_{\mathbf{x}}
\sum_j \rho\!\left(\sqrt{w_j^{(t)}}(\|\mathbf{x}-\mathbf{a}_j\|-\tilde d_j)\right)
\]

(\(t\) 반복 수는 대부분 2회, `TRILAT_GN_ITERS`는 Turbo 계열에서 22 고정).

---

## 3) 무결성 분기 (매우 중요)

- **V12 기본 (`v12.py`)**: 하이퍼를 **검증 RMSE**로 선택(누수/선택 편향 가능).
- **V12 Strict / Fast / Fast2 / Turbo / Strict Parallel**: 요약 JSON `integrity` 또는 목적함수 문구상 Train 기반 선택(단, Parallel은 `cv_split_mode` 확인 필요).

---

## 4) 사용자 제공 터미널 재실행 로그 반영

아래 값은 사용자가 전달한 콘솔 출력과 `outputs/v12_summary.json`이 일치한다.

- V12 기본:
  - Step A RMSE 1.5716, Step B 1.4467, Step C 1.4453
  - 선택 HP: `calib=identity`, `feat=bias_sub`, `gate=1.550`, `uwb_var_inflate=10`, `huber_f_fusion=1.35`, `IRLS_thresh=3.10`, `IRLS_iter=2`
- V12 strict_parallel (두 번 실행):
  - `effective_splits=3` 모드 실행과 `K_FOLD_SPLITS=1` 모드 실행 모두 검증 A/B/C는 동일 (A 1.5104, B 1.6082, C 1.6086)
  - Train 목표값만 2.5092 ↔ 2.5495로 변경
- V12 turbo:
  - Train CV 목표 2.5336, 검증 A/B/C = 1.5104 / 1.4449 / 1.4525
  - wall-clock 27.69 s

---

## 5) `outputs/v12*` 파일 전체 분석 요약

### 5.1 요약 JSON (6개)

| 파일 | 튜닝 방식 | 검증 Step C RMSE | 해석 |
|---|---|---:|---|
| `v12_summary.json` | **검증 최소화** | **1.4453** | 수치 최상이나 누수 가능 |
| `v12_turbo_summary.json` | Train KFold + Optuna | 1.4525 | 무결성 유지 + 성능 양호 |
| `v12_strict_summary.json` | Train KFold(3) | 1.6086 | Step A보다 B/C 악화 |
| `v12_fast_summary.json` | Train KFold(2) 축소 그리드 | 1.6099 | strict와 유사 |
| `v12_fast2_summary.json` | Train KFold(2) + StepA bundle | 1.6233 | Step A(1.4280)는 우수하나 B/C 악화 |
| `v12_strict_parallel_summary.json` | 병렬 + `cv_split_mode` 기록 | 1.6086 | K1 모드 시 엄격 KFold와 다름 |

### 5.2 예측 CSV (6개)

공통적으로 100행 중 유효 오차는 99행(1행 NaN).  
`v12_predictions.csv`(기본) 기준 P90/최대:

- Step A: P90 2.328 m, max 7.200 m
- Step B: P90 2.459 m, max 4.459 m
- Step C: P90 2.457 m, max 4.496 m

`v12_turbo_predictions.csv`는 Step B P90이 더 낮고(2.304 m) Step C에서 소폭 증가(2.400 m).

### 5.3 그리드/튜닝 CSV (16개)

- `v12_grid_phaseA.csv`, `v12_grid_phaseBC.csv`: **검증 RMSE 기준** 최적 조합이 기록됨.
- `v12_strict_grid_phaseA.csv`, `v12_strict_grid_phaseBC.csv`: **Train CV 기준** 최적 조합.
  - 최적 Step C CV는 `gate=1.55`, `inflate=24`, `huber_f_fusion=1.35`, `irls=3.1/2`.
- `v12_fast_*`: 축소 탐색 흔적(phaseBC 6행).
- `v12_fast2_grid_stepA.csv`(33행), `v12_fast2_grid_phaseBC.csv`(75행): StepA 후보 풀 + Joint 선택 로그.
- `v12_turbo_optuna_trials.csv`(35행):
  - best trial `mean_cv_rmse_step_c_m = 2.533550`
  - best HP: `phase_a_rank=2`, `gate=1.6855`, `inflate=13.8280`, `huber_f_fusion=1.2265`, `irls=3.099`.
- `v12_*_wifi_calibration_coefs.csv`(6개): 각 변형의 최종 Wi-Fi 보정 계수(A,B).

---

## 6) 변형별 기술적 해석

1. **기본 V12**는 검증 선택이어서 B/C 개선폭이 매우 좋게 나온다.
2. **Strict/Fast/Parallel**은 Train 선택으로 바꾸면 동일 데이터에서 Step B/C가 Step A를 이기지 못하는 경우가 생긴다.
3. **Turbo**는 무결성 유지하면서 strict 계열보다 검증 성능이 크게 개선된다.
4. **Fast2**는 Step A 자체는 가장 낮은 편이지만, B/C에서 손실이 커서 최종 C가 높다.

즉 V12 계열의 핵심은 “게이트·인플레이트·IRLS 조합을 **어디서 선택하느냐**”이며, 보고서는 반드시 **누수/무결 구분**을 병기해야 한다.

---

## 7) 관련 스크립트 파일

- `scripts/indoor_fusion_pipeline_v12.py`
- `scripts/indoor_fusion_pipeline_v12_strict.py`
- `scripts/indoor_fusion_pipeline_v12_fast.py`
- `scripts/indoor_fusion_pipeline_v12_fast2.py`
- `scripts/indoor_fusion_pipeline_v12_turbo.py`
- `scripts/indoor_fusion_pipeline_v12_strict_parallel.py`

