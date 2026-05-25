# V12 계열 · V13 · V13-fix · V14 명세 (요약본)

[← 색인](SPEC_INDEX.md) · [V12 단독 상세](SPEC_v12.md) · [V13 단독 상세](SPEC_v13.md) · [V15](V15_PIPELINE_SPEC.md) · [V14 단독 상세](SPEC_v14.md) · [V12 Turbo 집중본](V12_TURBO_PIPELINE_SPEC.md) · [V13 최종본](V13_FINAL_PIPELINE_SPEC.md)

---

## 공통 수식 골격 (V12 이후)

### Step A (가이드)

V11 계열 **AP별 거리 보정** 후 Wi‑Fi만으로 `WifiOnlyTrilateration` 등으로 \(\mathbf{p}_A\) 추정. V12 Turbo 검증 포워드는 **V3 동급 Wi‑Fi 삼변 한 경로**로 고정되는 부분이 있다(V12 Turbo 명세 §4 참고).

### Step B (게이팅 융합)

가이드 \(\mathbf{p}_w\)에 대해 UWB \(k\)가

\[
\big| \|\mathbf{p}_w - \mathbf{a}_k\| - \tilde{d}_k \big| > \tau
\]

이면 **제거**. 생존 UWB는 분산 \(\sigma^2_k\)에 인플레이트 \(\lambda\)를 곱해 **과소분산 과신**을 억제:

\[
w_k^{\mathrm{UWB}} \propto \frac{1}{\lambda \sigma^2_k + b_u + \epsilon}.
\]

Wi‑Fi는 보정 거리 \(d^{\mathrm{cal}}\)와 \(w^{\mathrm{WiFi}} \propto 1/(\sigma^2 + b_w + \epsilon)\)로 같이 **바운드 Huber GN**에 넣는다.

### Step C (IRLS)

Step B 출력을 초기값으로, 앵커·거리·가중 묶음에 대해 **Tukey × Huber** 반복(`irls_refine_xyz` 또는 Numba `irls_refine_xyz_njit`). 구현 상 **최대 2회** 등 고정이 많다.

---

## 1. `indoor_fusion_pipeline_v12.py`

### 1.1 v11 대비 동기

- **문제 1**: UWB 저분산·대오차가 단순 역분산 융합을 망가뜨림.  
- **문제 2**: Pure Wi‑Fi는 바닥이 안정적(≈1.55 m).  
- **문제 3**: v9 패턴의 **기하 게이트**가 스파이크 UWB 제거에 효과적이었다.  
**대응**: V11 Step A + **v9 게이트 Step B** + **Tukey×Huber IRLS Step C**를 한 파이프라인으로 통합.

### 1.2 하이퍼 선택 — **데이터 누수**

`FEAT_MODES` × `v12_calibration_spec_list()` × 게이트·inflate·IRLS 그리드의 **카테시안 탐색 후 검증 Step C RMSE 최소** 선택. → **검증 라벨이 선택에 직접 개입**하므로 보고·과제 제출의 **무결 지표로 사용하면 안 된다.**

### 1.3 산출

`v12_summary.json`, `v12_predictions.csv`, `v12_grid_phaseA.csv`, `v12_grid_phaseBC.csv`, 맵·CDF PNG.

---

## 2. `indoor_fusion_pipeline_v12_strict.py`

### 2.1 v12 대비 동기

동일 Step A/B/C **수식**을 유지하되, 하이퍼는 **Train K‑Fold 평균 Step C RMSE**만으로 선택. 검증 `True_X/Y`는 **최종 단일 포워드**에만 사용.

### 2.2 결과 현상

요약 JSON 기준으로 **검증 순전파에서 Step B·C가 Step A보다 RMSE가 악화**되는 경우가 있다. → Train 목적과 검증 분포의 **갭**을 보여 준다.

---

## 3. `indoor_fusion_pipeline_v12_fast.py`

### 3.1 Strict 대비 동기

**실행 시간**: 폴드 수 2, 그리드 축소, `V12_FAST_PLOTS=1`일 때만 CDF 등. 무결성 문구는 요약에 Strict와 동일 계열로 기술.

---

## 4. `indoor_fusion_pipeline_v12_fast2.py`

### 4.1 Turbo/Strict 대비 동기

- Step A 후보를 **여러 개**(보정 삼변, V3 동급 삼변, **컨벡스 블렌드** \(\alpha\)) 생성.  
- **상위 Step A Train CV 후보**마다 Phase B/C 그리드를 돌리고, **사전 정의된 사전순**(Train CV Step C, B, A, 캘리브 단순성 등)으로 **하나를 선택**.  
- **검증 라벨은 선택에 사용하지 않음**(`integrity` 필드).

### 4.2 결과

`v12_fast2_summary.json`에 **검증 튜닝 v12 대비 참조용 편향 베이스라인** 수치가 함께 기록될 수 있다.

---

## 5. `indoor_fusion_pipeline_v12_turbo.py`

### 5.1 Strict 대비 동기

그리드 전수가 느리므로 **Optuna(TPE)**로 연속·이산 혼합 탐색 + **Numba 배치** + Trial 병렬.

### 5.2 상세

**반드시 [V12_TURBO_PIPELINE_SPEC.md](V12_TURBO_PIPELINE_SPEC.md)** 를 본다. `fusion_turbo_numba_core.py`와 동치 Step B/C.

---

## 6. `indoor_fusion_pipeline_v12_strict_parallel.py`

### 6.1 동기

Phase B/C 그리드 **프로세스 병렬화**, `K_FOLD_SPLITS` 조절. 요약 JSON에 **`K1_no_fold_self_score_on_train`** 등이 찍힐 수 있어 **엄격한 K‑폴드와 동일하지 않을 수 있음** — 메타를 반드시 읽는다.

---

## 7. `indoor_fusion_pipeline_v13.py` — **팀 최종 선정**

### 7.1 v12 Turbo 대비 동기

- **문제**: Turbo 검증 Step A는 V3 고정 경로라 **캘리브·V3 프로파일·블렌드** 탐색이 제한적.  
- **대응**: **확장 캘리브 카탈로그** + **V3 프로필 그리드** + **보정/V3 컨벡스 블렌드** + (옵션) KNN 잔차. Step B/C는 **Turbo Numba 배치** 재사용. **Optuna 목적 = Train 폴드 평균 Step C RMSE**.

### 7.2 상세·산출

**[V13_FINAL_PIPELINE_SPEC.md](V13_FINAL_PIPELINE_SPEC.md)** 및 `outputs/v13_summary.json`.

---

## 8. `indoor_fusion_pipeline_v13_fix.py`

### 8.1 v13 대비 동기

1. Numba 경로를 **`fusion_turbo_numba_core`만** 사용 (turbo `.py` 의존 제거).  
2. 캘리브 순위에서 **`forced_*` 운용점 삽입 제거** — Train 근거 순만.  
3. Optuna **enqueue warmstart** 기본 생략.

### 8.2 보고서 주의

모듈 docstring대로, **탐색 공간 자체**는 선행 EDA·관례로 고정한 부분이 있으므로 “전부 Train에서 역산”처럼 쓰지 말 것.

---

## 9. `indoor_fusion_pipeline_v14.py` — 실험 브랜치

### 9.1 v12 Turbo 대비 동기

**Step A·UWB 취급은 Turbo와 동일**하되, Wi‑Fi 융합에 **JWT**류 요소를 가한다:

- AP별 **Isotonic** (optional)  
- **Score RANSAC**, **삼각부등식(Numba)** 기반 가중  
- 가중 형태: `1/(1 + w_r RANSAC + w_t triangle)` 등 JWT `final_outlier_score`와 유사한 결합

### 9.2 무결성·CLI

Optuna는 Train Step C만. `--jwt-ablation`으로 부분 켜기 검증 표 출력.  
**주의**: Wi‑Fi 가중·비선형 보정이 **UWB 기하 정합을 깨뜨리면** Step B가 Step A보다 나빠질 수 있다(모듈 docstring).

---

## 10. 파일별 산출 접두 요약

| 스크립트 | JSON 요약 예 |
|-----------|----------------|
| v12 | `v12_summary.json` |
| v12_strict | `v12_strict_summary.json` |
| v12_fast | `v12_fast_summary.json` |
| v12_fast2 | `v12_fast2_summary.json` |
| v12_turbo | `v12_turbo_summary.json` |
| v12_strict_parallel | `v12_strict_parallel_summary.json` |
| v13 | `v13_summary.json` |
| v13_fix | `v13_fix_summary.json` |
| v14 | `v14_summary.json` |

분석 도구: [SPEC_analysis_and_JWT.md](SPEC_analysis_and_JWT.md).
