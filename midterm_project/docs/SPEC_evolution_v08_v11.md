# 파이프라인 진화 명세: V8 ~ V11 (+ v9_strict, v10_optimized)

[← 색인](SPEC_INDEX.md) · [이전: V1~V7](SPEC_evolution_v01_v07.md)

---

## 1. `indoor_fusion_pipeline_v8.py` — 공통 로컬라이저 허브

### 1.1 V7 대비 동기

**문제**: Wi‑Fi만으로도 위치가 나쁘지 않다는 가설과, UWB를 무조건 넣으면 **오히려 악화**된다는 실험 결과.  
**대응**: **한 스크립트에서** Wi‑Fi only / UWB only / 하이브리드(HBB \(\alpha=1.2\)) / **Top‑K 앵커** 선택을 **애블레이션**하고, 검증 RMSE로 \(K\) 선택.

### 1.2 기법·수식

`FusionLocalizerV8`이 이후 **v9, v12, v13** 등에서 재사용된다.

- **거리 보정**: `ROBUST_BIAS_M` (= `HARDWARE_CALIB_BIAS_M` 계열)으로 \(\tilde{d} = m - b\).
- **삼변**: `least_squares` + Huber, 바운드 `BOUNDS_LO/HI`, `MAX_RANGE_M` 클립.
- **가중**: Wi‑Fi/UWB 각각 \(w \propto 1/(v + b_{\mathrm{wifi/uwb}} + \epsilon)\) (`EPS_W`).

### 1.3 결과·해석

검증에서 **Wi‑Fi 단독 Step A 수준의 RMSE가 약 1.55 m 전후**로 안정적이고, 일부 하이브리드·최종 선택 경로는 **7 m대**까지 악화되는 케이스가 재현되었다. → **“Wi‑Fi 바닥 + 선택적 UWB”**가 이후 v9·v12 설계의 근거가 된다.

---

## 2. `indoor_fusion_pipeline_v9.py`

### 2.1 V8 대비 동기

**문제**: UWB를 전부 넣으면 멀티패스·스파이크가 삼변을 망친다.  
**대응**: **Step A**: Pure Wi‑Fi 삼변으로 안심 좌표 \((x_w,y_w)\) 획득. **Step B**: 각 UWB 앵커에 대해

\[
\big| \|\mathbf{p}_w - \mathbf{a}^{\mathrm{UWB}}_k\| - \tilde{d}^{\mathrm{UWB}}_k \big| \le \tau
\]

를 만족하는 앵커만 남기고(**기하 게이트**), 나머지는 **하드 제외**. 통과 UWB + Wi‑Fi로 다시 Huber 융합 삼변.

\(\tau\)는 **`GATE_THRESH_GRID` 전체를 검증 RMSE로 스캔**하여 최소점 선택 → **검증 라벨 누수(편향)**.

### 2.2 기법

- `VAR_CAP_UWB_FUSION`, `VAR_CAP_WIFI_FUSION` 초과 시 해당 센서 스킵.
- 원 NaN UWB는 게이트 전에 제외.

### 2.3 산출·무결성 대응

`v9_predictions.csv`, `v9_uwb_kill_log.csv`. 무결 평가는 **`indoor_fusion_pipeline_v9_strict.py`** (Train K‑Fold로만 \(\tau\) 선택).

---

## 3. `indoor_fusion_pipeline_v9_strict.py`

### 3.1 v9 대비 동기

v9의 **선택 편향** 제거. 목적값은 **Train 폴드 내 Step B RMSE 평균**만.

### 3.2 결과

`v9_strict_summary.json`: 검증 순전파 Step B는 v9(검증 튜닝)보다 **RMSE가 높아질 수 있음** — 이것이 “진짜 일반화 난이도”에 가깝다.

---

## 4. `indoor_fusion_pipeline_v10.py` — Pure Wi‑Fi

### 4.1 V8~V9 대비 동기

**문제**: UWB 없이도 실무적으로 쓸 만한 바닥이 있는지 확인.  
**대응**: **UWB 미사용**. Train에서 **AP별 선형 거리 보정** \(d_{\mathrm{cal}} = A f + B\) (피처 \(f\)는 CALIB_FEATURES), **LOO‑5 스타일** 삼변, **잔차 K‑NN**.

### 4.2 기법

- Wi‑Fi median 결측: `KNNImputer` 등 파이프라인 내 유틸.
- 삼변: `WifiOnlyTrilateration` + Huber `f_scale` 그리드.
- K: 검증 홀드아웃·LOOCV 롤백 가드(`v10_run_summary.json`의 `note` 참고).

### 4.3 결과

Step A RMSE **약 1.55 m** 전후. Step B~D는 그리드·K에 따라 **일시 악화** 후 일부 **회복** 패턴(`v10_run_summary.json`).

---

## 5. `indoor_fusion_pipeline_v10_optimized.py` (“Wi‑Fi Guided UWB Sniper”)

### 5.1 v10 대비 동기

**문제**: Pure Wi‑Fi만으로는 물리적으로 UWB 정보를 버림.  
**대응**: Step A는 v10 스타일 **보정 Wi‑Fi 삼변**, Step B는 **Wi‑Fi 가이드 + UWB 기하 게이트**(Train에서 게이트 그리드) + log1p 분산 가중 융합 + **잔차 K‑NN**.

### 5.2 기법

v9와 유사한 **기하 불일치 게이트**이나, 게이트 폭 등이 **Train 그리드**에 맞춰져 있다(요약·코드 주석 확인).

---

## 6. `indoor_fusion_pipeline_v11.py` — Pure Wi‑Fi 강건화

### 6.1 v10 대비 동기

**문제**: 선형 보정만으로는 AP별 비선형 왜곡이 남을 수 있음.  
**대응**: AP별 **`HuberRegressor` / `RANSACRegressor` / 항등(identity)** 캘리브 후보, **IRLS**: Tukey bisquare 가중(`TUKEY_C = 4.685`) × Huber hinge 가중을 곱해 반복, **다중 피처 + `StandardScaler` K‑NN**.

### 6.2 수식 요지

- **Tukey**: \(w^T(r) = (1 - (r/c)^2)^2_+\) (표준 bisquare).
- **Huber**: 큰 잔차에 대해 가중 절편.
- 삼변 후 잔차 벡터에 대해 가중을 갱신하는 **IRLS 고정 회수**.

### 6.3 결과·한계

`v11_summary.json`: 그리드가 **검증 RMSE 기준**이면 편향 가능. 본 실행에서는 Step A~D RMSE 변화가 **수 cm 수준**에 머무는 경우도 있었다.

---

## 7. 요약

| 스크립트 | 이전 대비 핵심 | 누수·무결 |
|-----------|----------------|-----------|
| v8 | 애블레이션·공통 클래스 | 검증 RMSE로 K |
| v9 | Wi‑Fi 가이드 UWB 게이트 | **검증으로 \(\tau\)** → 편향 |
| v9_strict | 동일 수식, \(\tau\)만 Train CV | 무결 |
| v10 | Pure Wi‑Fi 바닥 | 일부 검증 그리드 |
| v10_opt | v9류 게이트 + KNN | Train 게이트 그리드 |
| v11 | 강건 보정 + IRLS + KNN | 검증 그리드 명시 |

다음: [SPEC_v12_v13_family.md](SPEC_v12_v13_family.md).
