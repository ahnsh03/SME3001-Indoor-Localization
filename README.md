# Wi‑Fi & UWB 융합 실내 측위 (Midterm Project)

UWB 6채널·Wi‑Fi 6채널의 **거리 중앙값·분산**으로부터 실내 **2D 위치**를 추정하는 실험 코드 모음이다. 초기에는 **분산 가중 Huber 삼변측량**과 **K‑NN 잔차 보정**에서 출발하였고, 이후 **경계 제약·다단계 그리드·Pure Wi‑Fi·Wi‑Fi 가이드 UWB 기하 게이팅·IRLS·Train K‑Fold / Optuna 무결 튜닝**까지 단계적으로 확장하였다.

**팀 명세상 최종 제출·보고용 권장 파이프라인은 V13**이다. (`docs/V13_FINAL_PIPELINE_SPEC.md`, `outputs/v13_summary.json`)

- **V12 (`indoor_fusion_pipeline_v12.py`)**: 하이퍼파라미터를 **검증 RMSE**로 고른다. **검증 라벨이 튜닝에 직접 들어가므로 데이터 누수(낙관적 편향)**가 있다. **레거시·비교용**으로만 쓰는 것이 좋다.
- **V12 Turbo (`indoor_fusion_pipeline_v12_turbo.py`)**: 하이퍼는 **Train 교차검증과 Optuna(Train 목적)**만으로 선택하고, 검증 정답은 **최종 순전파·지표 1회**에만 쓰는 구성이 `outputs/v12_turbo_summary.json`의 **`integrity`** 필드에 명시되어 있다. (동일 계열의 Strict·Fast·Fast2 등은 각 스크립트·요약 JSON을 따로 확인할 것.)
- **V14**: Turbo 골격에 팀원 **JWT** Wi‑Fi 보정·이상치 점수 등을 얹은 **실험 브랜치**이다. (`--jwt-ablation`, `--no-plots` 등)

---

## 1. 디렉터리 구조

```text
midterm_project/
├── data/
│   ├── train/              # median_test_kgh_corrected, variance_test_kgh_corrected (.csv 우선, 없으면 xlsx)
│   ├── validation/         # median_validation, variance_validation
│   ├── archive_unused/
│   └── reference/
├── docs/
│   ├── V12_TURBO_PIPELINE_SPEC.md   # V12 Turbo: Phase A/B/C, Numba, Optuna, 무결성
│   └── V13_FINAL_PIPELINE_SPEC.md   # 최종 V13: 흐름, 실행, 선정 사유
├── scripts/
│   ├── fusion_realtime_sanitize.py    # 로드·헤더 정규화·Wi‑Fi 분산 캡·Phase1 바이어스·경로 해석
│   ├── fusion_turbo_numba_core.py     # V12 Turbo와 동치 Step B/C Numba 배치 (v13_fix 등에서 재사용)
│   ├── fusion_data_audit.py           # Train/Val sanitize 후 행 수 등 간단 감사
│   ├── indoor_fusion_pipeline_v1.py … v3.py   # 학습형 잔차(K‑NN 등) 초기 라인
│   ├── indoor_fusion_pipeline_v4.py … v7.py   # K‑NN 비의존·Ridge·다단계 기하·그리드
│   ├── indoor_fusion_pipeline_v8.py           # Wi‑Fi/UWB 애블레이션·공통 설정 허브
│   ├── indoor_fusion_pipeline_v9.py           # Wi‑Fi 1차 → UWB 기하 게이트 융합 (검증 그리드 주의)
│   ├── indoor_fusion_pipeline_v9_strict.py    # 게이트만 Train K‑Fold로 선택
│   ├── indoor_fusion_pipeline_v10.py          # Pure Wi‑Fi (선형 캘리브·LOO 스타일·K‑NN)
│   ├── indoor_fusion_pipeline_v10_optimized.py # Wi‑Fi 가이드 + UWB 게이트 + K‑NN 변형
│   ├── indoor_fusion_pipeline_v11.py          # Pure Wi‑Fi: Huber/RANSAC + IRLS + 다중피처 K‑NN
│   ├── indoor_fusion_pipeline_v12.py          # V11 Step A + V9 게이트 Step B + IRLS Step C (검증 튜닝)
│   ├── indoor_fusion_pipeline_v12_strict.py
│   ├── indoor_fusion_pipeline_v12_strict_parallel.py  # 병렬·K1 모드 등 (요약 JSON의 cv 모드 참고)
│   ├── indoor_fusion_pipeline_v12_fast.py
│   ├── indoor_fusion_pipeline_v12_fast2.py      # Step A 블렌드 + Joint Train CV
│   ├── indoor_fusion_pipeline_v12_turbo.py    # Optuna + Numba Step B/C
│   ├── indoor_fusion_pipeline_v13.py          # 최종 선정: Fast2류 Step A + Turbo B/C + Optuna
│   ├── indoor_fusion_pipeline_v13_fix.py       # Numba는 fusion_turbo_numba_core만 사용하는 변형
│   ├── indoor_fusion_pipeline_v14.py          # Turbo + JWT(Wi‑Fi)·실험 플래그
│   ├── indoor_fusion_pipeline_JWT.py          # JWT 단독 Train 피팅·Val 평가
│   ├── sensor_spatial_profiler.py             # 공간 EDA, UWB 순차 필터, 바이어스 표
│   └── validation_calibration_filter_scatter.py
├── outputs/                # v*_predictions.csv, v*_summary.json, 그리드 로그, PNG(실행 옵션에 따라)
└── README.md
```

대부분의 파이프라인은 산출물을 **`outputs/`** 에 둔다.

---

## 2. 버전 축 요약

| 구간 | 스크립트 | 한 줄 설명 |
|------|----------|------------|
| V1~V3 | `v1`~`v3` | KNNImputer·분산 가중 삼변·K‑NN 잔차; V3는 Wi‑Fi/UWB 동시 융합 Step A 등 |
| V4~V5 | `v4`, `v5` | K‑NN 없는 규칙·Huber 융합(v4); v4 완화 + Train Ridge 조건부(v5) |
| V6~V7 | `v6`, `v7` | v3 베이스 다단계(분산 컷·경계·앵커 드롭 등); v7은 α·β **검증 그리드** 선택 |
| V8 | `v8` | Wi‑Fi only / UWB only / 하이브리드 / Top‑K **애블레이션** |
| V9 | `v9`, `v9_strict` | Wi‑Fi 1차 후 UWB 기하 게이트 융합; strict는 **게이트만 Train CV** |
| V10~V11 | `v10`, `v10_optimized`, `v11` | Pure Wi‑Fi 심화; v10_opt는 UWB 융합 변형 |
| V12 계열 | `v12`, `v12_strict`, `v12_fast`, `v12_fast2`, `v12_turbo`, `v12_strict_parallel` | 통합 융합; **`v12`는 검증 튜닝으로 데이터 누수**; **무결 튜닝 설명의 표준 예시는 `v12_turbo`**(`v12_turbo_summary.json`); 그 외 변형은 각 `*_summary.json`·CV 모드 확인 |
| V13 | `v13`, `v13_fix` | **최종 권장**; Optuna + Train K‑Fold Step C 목적; `v13_fix`는 Numba 코어 분리 |
| 실험 | `v14`, `JWT` | V14: Turbo+JWT; `JWT.py`: 동일 데이터 규격 단독 평가 |

---

## 3. 공통·보조 모듈

| 파일 | 역할 |
|------|------|
| `fusion_realtime_sanitize.py` | median/variance 로드, 컬럼 정규화, Wi‑Fi 분산 상한, **PHASE1 / HARDWARE_CALIB** 바이어스, `resolve_*_paths`. 행 단위 `drop` 없이 전 행 유지하는 설계가 기본이다. |
| `fusion_turbo_numba_core.py` | Turbo Step B/C와 동치 **Numba** 배치 삼변·패킹. `v13_fix` 등에서 `v12_turbo` 소스 의존 없이 사용한다. |
| `fusion_data_audit.py` | 표준 경로로 읽은 뒤 sanitize 후 **행 수** 등을 출력한다. |
| `sensor_spatial_profiler.py` | Train 공간 EDA PNG, UWB 순차 필터, Robust bias CSV. 인자: `all` / `spatial` / `uwb` 등(스크립트 헬프 참고). |
| `validation_calibration_filter_scatter.py` | 검증 median 기준 True vs 측정 거리 산점도(필터 단계별 비교 PNG). |
| `indoor_fusion_pipeline_JWT.py` | RANSAC·Isotonic 등 **JWT Wi‑Fi 보정**을 레포 데이터 규격에 맞춘 단독 평가. |

---

## 4. 데이터 매핑

| 용도 | 경로(확장자는 resolver가 `.csv` 우선) |
|------|----------------------------------------|
| Train | `data/train/median_test_kgh_corrected` + `data/train/variance_test_kgh_corrected` |
| Validation | `data/validation/median_validation` + `data/validation/variance_validation` |

표준 레이아웃: 앞 2열 **노드 좌표**, 다음 6열 **UWB**, 다음 6열 **Wi‑Fi**. 참 위치는 일반적으로 **`True_X = Node_x * 0.6`**, **`True_Y = Node_y * 0.6`** (m).

`data/archive_unused/`, `data/reference/`는 파이프라인 기본 입력에서 제외된 보관·참고용이다.

---

## 5. 센서·앵커 (타일 좌표 × `grid_size_m`, 기본 0.6 m)

스크립트 내 상수와 동일하게 유지한다.

**UWB (예):** `110394ab`, `e63ce2f`, `8e610981`, `d10485af`, `d1044709`, `4e610206`  
**Wi‑Fi (동일 위치 쌍):** `SW_11`, `SW_first_team`, `볼링공`, `SW_4`, `SW_6`, `SW_5`

`d1044709` 등 컬럼명 변형은 `fusion_realtime_sanitize` 및 각 로더에서 통일한다.

---

## 6. 실행 방법

프로젝트 **루트**에서 실행한다. **Windows PowerShell**에서는 `&&` 대신 **`Set-Location` 후 세미콜론**을 쓰거나 한 줄씩 실행하는 것이 안전하다.

```powershell
Set-Location "C:\path\to\midterm_project"

py -3 scripts\fusion_data_audit.py

py -3 scripts\indoor_fusion_pipeline_v1.py
py -3 scripts\indoor_fusion_pipeline_v2.py
py -3 scripts\indoor_fusion_pipeline_v3.py
py -3 scripts\indoor_fusion_pipeline_v4.py
py -3 scripts\indoor_fusion_pipeline_v5.py
py -3 scripts\indoor_fusion_pipeline_v6.py
py -3 scripts\indoor_fusion_pipeline_v7.py
py -3 scripts\indoor_fusion_pipeline_v8.py
py -3 scripts\indoor_fusion_pipeline_v9.py
py -3 scripts\indoor_fusion_pipeline_v9_strict.py
py -3 scripts\indoor_fusion_pipeline_v10.py
py -3 scripts\indoor_fusion_pipeline_v10_optimized.py
py -3 scripts\indoor_fusion_pipeline_v11.py

py -3 scripts\indoor_fusion_pipeline_v12.py
py -3 scripts\indoor_fusion_pipeline_v12_strict.py
py -3 scripts\indoor_fusion_pipeline_v12_strict_parallel.py
py -3 scripts\indoor_fusion_pipeline_v12_fast.py
py -3 scripts\indoor_fusion_pipeline_v12_fast2.py
py -3 scripts\indoor_fusion_pipeline_v12_turbo.py
py -3 scripts\indoor_fusion_pipeline_v12_turbo.py --no-plots

py -3 scripts\indoor_fusion_pipeline_v13.py
py -3 scripts\indoor_fusion_pipeline_v13.py --no-plots
py -3 scripts\indoor_fusion_pipeline_v13_fix.py

py -3 scripts\indoor_fusion_pipeline_v14.py
py -3 scripts\indoor_fusion_pipeline_v14.py --no-plots
py -3 scripts\indoor_fusion_pipeline_v14.py --jwt-ablation

py -3 scripts\indoor_fusion_pipeline_JWT.py
py -3 scripts\sensor_spatial_profiler.py
py -3 scripts\sensor_spatial_profiler.py spatial
py -3 scripts\sensor_spatial_profiler.py uwb
py -3 scripts\validation_calibration_filter_scatter.py
```

**Turbo·V13·V14** 계열은 **`numba`**, **`optuna`** 가 필요하다. Turbo·V13에서 병렬도는 환경변수(예: `V12_TURBO_OPTUNA_JOBS`, `V13_OPTUNA_JOBS`)로 제한할 수 있다(각 스크립트 상단·문서 참고).

---

## 7. 의존성 (예시)

```powershell
py -3 -m pip install pandas numpy scipy scikit-learn matplotlib openpyxl numba optuna
```

`openpyxl`은 xlsx 로드 시 필요하다.

---

## 8. 산출물 규칙

- **예측 CSV:** `outputs/v*_predictions.csv`, `v*_validation_predictions.csv` 등(버전별 상이).
- **요약 JSON:** `outputs/v*_summary.json`, `v*_run_summary.json`, `v13_fix_summary.json` 등. **`integrity` 또는 `objective` 필드**로 튜닝에 검증 라벨이 쓰였는지 구분한다.
- **그리드·로그:** `v12_*_grid_phaseA.csv`, `v12_turbo_optuna_trials.csv`, `v9_uwb_kill_log.csv` 등.
- **PNG:** CDF·오차맵 등은 스크립트·플래그에 따라 생성된다(예: `v12_cdf_steps.png`, `v9_step_cdf.png`, `v13_*_map_*.png`). 실행 후 `outputs/` 목록을 확인한다.

최종 수치·무결성 문구는 **`outputs/v13_summary.json`** 및 **`docs/V13_FINAL_PIPELINE_SPEC.md`** 를 우선한다.

---

## 9. 지표 해석 시 주의 (무결성)

1. **`indoor_fusion_pipeline_v12.py`**, **`v9.py`**, **`v11.py`**, **`v10.py`(일부 단계)** 등은 요약에 **검증 RMSE로 그리드·K를 고른다**고 명시된 경우가 있다. 이 경우 수치는 **낙관적 편향**이 있을 수 있다.
2. **`v9_strict`**, **`v12_strict`**, **`v12_fast`**, **`v12_fast2`**, **`v12_turbo`**, **`v13`**, **`v13_fix`** 는 JSON에 **Train K‑Fold 또는 Optuna가 Train 목적만 최소화**한다고 기술된 구성이 있다. **동일 표에서 v12 기본과 v13을 직접 비교할 때는 반드시 주석으로 구분**한다.
3. **`v12_strict_parallel_summary.json`** 의 `cv_split_mode` 등은 **엄격한 K‑폴드와 다를 수 있는 모드**가 기록될 수 있으므로, 수치만 복사하지 말고 요약 메타를 함께 읽는다.

---

## 10. 대표 검증 수치 (현재 `outputs/*_summary.json` 기준, 재실행 시 변동 가능)

동일 **validation** 세트, **서로 다른 튜닝 무결성**이므로 **한 표 안에서도 열 의미를 구분**한다.

| 파이프라인 | 튜닝 | 검증 Step C RMSE (m) | 비고 |
|------------|------|----------------------|------|
| V12 기본 | 검증 최소 | **1.445** | `v12_summary.json` `step_C_RMSE_m` |
| V12 Turbo | Train CV + Optuna | **1.452** | `v12_turbo_summary.json` |
| V12 Strict / Fast / Parallel | Train CV | **1.609** 전후 | `v12_strict_summary.json` 등 |
| V12 Fast2 | Train CV | **1.623** | `v12_fast2_summary.json` |
| V13 | Train CV + Optuna | **1.425** | `v13_summary.json` (팀 최종 권장) |
| V9 Strict | Train CV (게이트만) | Step B **1.624** | `v9_strict_summary.json` |

Pure Wi‑Fi 바닥은 V8·V9 Step A 등에서 **약 1.55 m** 전후가 반복된다. 초기 V1~V7은 버전별 CSV·콘솔로 확인한다.

---

## 11. 설계 문서 (`docs/`)

| 문서 | 내용 |
|------|------|
| [docs/SPEC_INDEX.md](docs/SPEC_INDEX.md) | **전 스크립트 명세 색인** — 파이프라인 진화(V1~V14)·수식·분석 도구 해석 |
| [docs/SPEC_v12.md](docs/SPEC_v12.md) | **V12 단독 상세** — 기본/Strict/Fast/Fast2/Turbo/Parallel + outputs 전수 해석 |
| [docs/SPEC_v13.md](docs/SPEC_v13.md) | **V13 단독 상세** — V13/V13-fix, Optuna·grid·predictions 해석 |
| [docs/SPEC_v14.md](docs/SPEC_v14.md) | **V14 단독 상세** — JWT/Isotonic 실험, ablation CSV 해석 |
| [docs/V13_FINAL_PIPELINE_SPEC.md](docs/V13_FINAL_PIPELINE_SPEC.md) | V13 최종 흐름, 실행 옵션, 선정 근거 |
| [docs/V12_TURBO_PIPELINE_SPEC.md](docs/V12_TURBO_PIPELINE_SPEC.md) | V12 Turbo Phase A/B/C, Numba, Optuna |

상세 분할: `docs/SPEC_fusion_core.md`, `SPEC_evolution_v01_v07.md`, `SPEC_evolution_v08_v11.md`, `SPEC_v12_v13_family.md`, `SPEC_analysis_and_JWT.md`.

---

## 12. 향후·운영 제안

- **회귀 방지:** `fusion_data_audit.py` + `--no-plots` 짧은 스모크로 행 수·상한 검사.
- **그리드 자동화:** 게이트·IRLS 임계·캘리브 카탈로그의 체계적 스윕과 표 자동 생성.
- **CI:** 플롯 없이 Turbo/V13 한 번만 돌리는 워크플로 검토.

---

## 13. 라이선스·출처

본 저장소는 **과제용 실험 코드**이다. 데이터·앵커 좌표는 과제 제공 또는 실험실 기준을 따른다.
