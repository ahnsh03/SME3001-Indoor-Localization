# 파이프라인 진화 명세: V1 ~ V7

[← 색인](SPEC_INDEX.md)

공통 기호: 앵커 좌표 \(\mathbf{a}_j \in \mathbb{R}^2\), 보정된 측정 거리 \(\tilde{d}_j\), 가중 \(w_j\). **가중 비선형 최소제곱**은 `scipy.optimize.least_squares`에 **Huber 또는 soft\_L1** 손실로 넣는 형태가 대부분이다.

---

## 1. `indoor_fusion_pipeline_v1.py`

### 1.1 이전 단계·문제의식

프로젝트 **초기 베이스라인**. 별도 “이전 버전” 없이, 과제 스펙(중앙값·분산 12채널)에 맞춘 **첫 통합 융합**이다.

### 1.2 기법

1. **UWB 결측**: `KNNImputer`로 Train에 피트 후 Train·Val에 적용(Wi‑Fi·UWB 컨텍스트 포함).
2. **삼변측량**: UWB 위주로 \(\sqrt{w_j}\,\big(\|\mathbf{x}-\mathbf{a}_j\| - \tilde{d}_j\big)\) 형 잔차에 Huber `f_scale` 적용.
3. **잔차 학습**: 12 median + 12 variance 등 **24차원** 특징으로 `KNeighborsRegressor`가 \((\Delta x, \Delta y)\) 예측(Train 피트).

가중은 통상 \(w_j \propto 1/(v_j + \epsilon)\) (분산 \(v_j\)) 형태를 사용한다.

### 1.3 결과로 확인된 한계

동일 검증 세트에서 **융합+K‑NN 최종 RMSE가 Wi‑Fi 단독 삼변보다 크게 악화**되는 현상이 관측되었다. 원인 후보: UWB 기하 불안정, K‑NN 특징·분포 불일치.

### 1.4 산출물

`outputs/v1_validation_predictions.csv`, CDF·벡터맵 PNG 등.

---

## 2. `indoor_fusion_pipeline_v2.py`

### 2.1 V1 대비 동기·동기

**문제**: V1 삼변 초기값이 불안정할 수 있음.  
**대응**: **Smart 초기값** 후보로 Step A 삼변 → **잔차 큰 앵커에 재가중(IRLS 유사)** → Step B 재삼변 → **K‑NN 잔차**(이웃 12 등).

### 2.2 기법·수식

단계별로 `least_squares` 잔차 벡터를 갱신한다. 임계(예: 3 m)를 넘는 앵커에 대해 가중을 줄이거나 제외하는 **반복적 강건화**가 핵심이다.

### 2.3 결과

Step C까지 포함하면 V1 대비 검증 RMSE가 **소폭 개선**되는 패턴이 보고되었다(`v2_step_rmse_report.csv`).

---

## 3. `indoor_fusion_pipeline_v3.py`

### 3.1 V2 대비 동기

**문제**: 거리 스파이크·원본 UWB 결측과 임퓨트 UWB를 구분하지 않으면 삼변 입력이 오염됨.  
**대응**: **22 m 초과 거리 무효화**, **원 NaN이었던 UWB는 삼변에서 배제**, Step A에서 **Wi‑Fi+UWB 동시** 역분산 가중 Huber 삼변(바이어스 항 \(b_u, b_w\) 포함).

### 3.2 기법·수식

Step A에서 통과한 센서 집합 \(\mathcal{S}\)에 대해:

\[
\min_{\mathbf{x}} \sum_{j \in \mathcal{S}} \rho_{\mathrm{Huber}}\Big(
\sqrt{w_j}\,\big(\|\mathbf{x}-\mathbf{a}_j\| - (\tilde{d}_j - b_{\cdot})\big)
\Big)
\]

Step B: Step A 좌표 \((x_A,y_A)\)만 특징으로 한 **2D K‑NN 잔차** 보정.

### 3.3 결과·한계

Step A는 V2 대비 **검증 RMSE가 크게 감소**하였다. 그러나 Step B에서 **동일 검증에서 RMSE가 Step A보다 증가**하는 경우가 있어, **잔차 K‑NN의 피처·학습 범위** 한계가 드러났다.

---

## 4. `indoor_fusion_pipeline_v4.py`

### 4.1 V3 대비 동기

**문제**: K‑NN에 기대는 방식이 검증에서 불안정. **해석 가능한 규칙 기반**과 **품질 게이트**가 필요하다는 진단.  
**대응**: **K‑NN 제거**. PHASE1 바이어스·`VAR_CAP` 정렬 분산 컷·**비대칭 융합**(Wi‑Fi/UWB 우세 패턴)·**경계 Huber 삼변**·`QUALITY_RMSE_TH` 품질 라벨.

### 4.2 기법

- 거리 하드 클립: \([0.5,\,22]\) m.
- UWB 분산 상한 `VAR_UWB_CUT = VAR_CAP_UWB_FUSION`.
- `ASYMM_UWB_DOM` 등으로 Wi‑Fi vs UWB 불일치 시 가중 비대칭.
- `least_squares` + bounds \([0,12]\times[0,18]\) m.

### 4.3 결과·산출

`v4_phase_aggregate.json`에 **융합 규칙 카운트**가 쌓여, 이후 v5 완화 설계의 근거가 되었다.

---

## 5. `indoor_fusion_pipeline_v5.py`

### 5.1 V4 대비 동기

**문제**: v4가 지나치게 보수적이거나 극단 좌표가 발생.  
**대응**: Wi‑Fi 결측 시 UWB 단독 허용, UWB 분산 상한 초과 시 **하드 드롭 대신 가중 감쇠**, 비대칭 비율 완화(`ASYMM_RATIO`). K‑NN 대신 **Train Ridge** \((\Delta x,\Delta y) \sim [1, x, y]\) 조건부 적용(`RIDGE_MIN_IMPROVE_RATIO`).

### 5.2 기법

앵커별 가중:

\[
w_u \propto \frac{1}{\log(1+\max(0,\sigma^2_u))+\epsilon},\quad
w_w \propto \frac{1}{\log(1+\max(0,\sigma^2_w))+\epsilon}
\]

\(\sigma^2_u > \texttt{VAR\_UWB\_SOFT\_START}\)이면 추가 감쇠 인자 곱함.

### 5.3 결과

동일 검증 CSV 기준으로 v4 대비 **평균 위치 오차가 감소**하는 방향으로 조정되었다.

---

## 6. `indoor_fusion_pipeline_v6.py`

### 6.1 V3 대비 동기 (v5와 병행 실험 축)

**문제**: v3의 KNNImputer·융합은 좋으나 **프로파일러 캘리브·경계·앵커 드롭**이 없어 극단 오차가 남음.  
**대응**: V3식 Step A를 베이스로 **Step B 분산 컷**, **Step C 경계 least_squares**, **Step D 최악 잔차 앵커 드롭** 순차 적용.

### 6.2 기법

`HARDWARE_CALIB_BIAS_M`(당시 명명) 기반 바이어스, `RES_DROP_MIN_M` 등으로 **기하적 이상 앵커**를 제거한다.

### 6.3 결과

단계별 RMSE가 `v6_validation_predictions.csv`에 저장된다. 일부 실행에서 **Final이 Step A보다 RMSE가 소폭 악화**하기도 하여, “다단계가 항상 이득은 아님”이 확인되었다.

---

## 7. `indoor_fusion_pipeline_v7.py`

### 7.1 V6 대비 동기

**문제**: 하이퍼가 고정이라 검증에 맞지 않음.  
**대응**: **검증 RMSE**로 **HBB \(\alpha\)** 그리드, **거리 감쇠 \(\beta\)** 그리드를 탐색(데이터 누수 가능 — 보고서에서 V12 기본과 동일 계열 주의).

### 7.2 기법

Step B: Log‑Var 가중. Step C/D: \(\alpha \in \{\texttt{ALPHA\_GRID}\}\), \(\beta \in \{\texttt{BETA\_GRID}\}\) 순차 적용 후 검증에서 최소 RMSE 선택.

### 7.3 결과

V6 대비 검증 Final RMSE가 **개선**되는 실행이 보고되었다(`v7_predictions.csv`, `v7_grid_search_trace.csv`).

---

## 8. 요약 표

| 버전 | 핵심 추가·변경 | 주요 수학 객체 |
|------|----------------|----------------|
| V1 | KNNImputer + Huber LS + KNN 잔차 | Huber \(\rho\), KNN |
| V2 | Smart init + IRLS류 재피팅 | 반복 가중 LS |
| V3 | 22 m 컷, 임퓨트 UWB 배제, 동시 융합 | 가중 Huber LS |
| V4 | 규칙·비대칭·품질 | bounds + Huber |
| V5 | Ridge 잔차, 연화 융합 | log‑var 가중, Ridge |
| V6 | 다단계 B/C/D | 컷·bounds·드롭 |
| V7 | \(\alpha,\beta\) 검증 그리드 | 가중·감쇠 그리드 |

다음 단계: [SPEC_evolution_v08_v11.md](SPEC_evolution_v08_v11.md).
