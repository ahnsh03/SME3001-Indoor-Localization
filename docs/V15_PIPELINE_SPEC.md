# V15 실내 융합 파이프라인 — 상세 명세서

[← 명세 색인](SPEC_INDEX.md) · [V13 최종 명세](V13_FINAL_PIPELINE_SPEC.md) · [v13-fix와 수식 공통 요약](SPEC_v12_v13_family.md)

본 문서는 `scripts/indoor_fusion_pipeline_v15.py`의 **설계 원칙**, **무결성**, **환경 변수**, **탐색 프로필**(balanced / defensive / v13_fix), **산출물**을 코드와 일치하도록 정리한 것이다.

---

## 1. 한 줄 요약

**v13-fix와 동일한 Numba Step B/C 골격** 위에서, **검증 라벨 없이** Train K-Fold OOF로 **UWB 기하 게이트 τ 구간**을 만들고, **Train CV로 순위를 매긴 V3·로컬 캘리브 카탈로그**로 가이드를 구성한 뒤, Optuna가 **Train CV 목적**만 최소화한다. 검증 `True_X/Y`는 **전체 Train 재피팅 후 단일 포워드**에만 사용한다.  
**v12의 검증튜닝 이산 그리드(`GATE_THRESH_GRID` 등)를 import하지 않는다.**

---

## 2. 팀 권장 파이프라인과의 관계

- 팀 README 기준 **최종 제출·보고 권장**은 여전히 **V13**(`docs/V13_FINAL_PIPELINE_SPEC.md`)이다.
- **V15**는 **간접 누수를 더 엄격히 피하려는 실험/보조 파이프라인**으로 추가되었다. 보고서에서는 V13과 **목적 함수·게이트 출처·탐색 공간**이 다름을 명시하고 비교한다.

---

## 3. 무결성(튜닝에 검증 미사용)

| 항목 | 내용 |
|------|------|
| 게이트 τ 탐색 구간 | Train K-Fold **OOF**에서, Wi‑Fi **Step A** 가이드 기준 UWB 앵커별 \(\|d_{\mathrm{geom}}-d_{\mathrm{meas}}\|\)의 **행 중앙값** 분포로부터 `lo_raw`, `hi_raw` 산출 후 실내 상한 캡·하한 보정(`train_oof_gate_row_median_stats`). 프로필에 따라 `apply_gate_tuning_caps`로 추가 클램프. |
| V3 `(huber_f, wifi_bias)` | 소그리드에 대해 **Train CV Step A RMSE**로 점수화·정렬 후 상위 K개만 사용(`train_ranked_v3_profiles`). `huber_f > 1.35` 후보 제외 및 앵커 튜플 선행 삽입. |
| Wi‑Fi 캘리브 | `v15_calibration_catalog()` 단독 정의. 순위는 **`merge_calib_ranks_train_cv_only`**(순수 보정 Step A CV). |
| Optuna 목적 | **기본**: `w · mean_cv_RMSE_StepA + (1-w) · mean_cv_RMSE_StepC` (`w`=`V15_STEPA_LOSS_WEIGHT`). **선택**: `V15_TWO_PHASE=1` 시 1단계 Step A만, 2단계 가이드 고정 후 Step C만. |
| 검증 라벨 | Optuna·그리드 선택에 **미사용**. 최종 검증 지표만 보고용. |

`outputs/v15_summary.json`의 `integrity`, `train_oof_gate_raw_m`, `train_oof_gate_bounds_m`, `tuning_profile`, `tuning_space`로 재현·감사 가능하다.

---

## 4. 처리 흐름(개요)

1. 데이터 로드: `fusion_realtime_sanitize.resolve_train_kgh_corrected_paths`, `resolve_validation_paths` → `FusionLocalizerV8.load_datasets`(v13-fix와 동일 경로 규격).
2. Wi‑Fi median 보간(`wifi_median_impute_fusion`), `build_loc_geom_pack`.
3. `train_oof_gate_row_median_stats` → `(gate_lo_raw, gate_hi_raw, meta)`.
4. `resolve_v15_tuning_space(V15_TUNING_PROFILE)` → `apply_gate_tuning_caps` → Optuna에서 사용할 `(gate_lo, gate_hi)`.
5. `train_ranked_v3_profiles`로 V3 목록 확정.
6. `rank_calibration_candidates` + `merge_calib_ranks_train_cv_only`로 캘리브 순위 캐시.
7. `build_fold_num_pack` → Numba warmup → Optuna (**단일 composite** 또는 **2단계**).
8. 전 Train 재피팅 후 검증 포워드 Step A/B/C → `v15_*` 산출물.

내부 융합: `indoor_fusion_pipeline_v13_fix`에서 재사용하는 `build_fold_num_pack`, `assemble_xy_guide_fold`, `mean_cv_rmse_step_c_from_assembled`, `merge_calib_ranks_train_cv_only` 등 + **`fusion_turbo_numba_core`** 배치 호출.

---

## 5. Train OOF 게이트 (`train_oof_gate_row_median_stats`)

- 폴드마다 학습 접선에서 보간한 뒤, 검증 접선 각 행에 대해 살아 있는 UWB 채널에 한해 \(\| \|p_{wifi}-a_k\| - \tilde d_k \|\)를 모으고 **행별 median** 리스트 구축.
- 분위수 `q(p)`에서 `lo`, `hi`를 둠(12th/97th 퍼센타일 계열 스케일·라운딩, 상한 **`min(..., 3.95)` m**, 하한 **`max(..., 1.05)` m** 등 코드 내 물리·실내 상한 포함).
- `meta`: `p50`, `p75`, `p90`, `n_rows` 등 로그 및 JSON용.

분석 보조 스크립트: `scripts/_train_gate_prior_analysis.py`(프로젝트에 존재할 경우 Train OOF 분위 진단 재현용).

---

## 6. 탐색 프로필 (`V15_TUNING_PROFILE`)

환경 변수 `V15_TUNING_PROFILE`: **`balanced`(기본)**, **`defensive`**, **`v13_fix`**(별칭 `v13fix`, `v13_compat`, `like_v13_fix`).

코드 함수: `resolve_v15_tuning_space` + `apply_gate_tuning_caps`.

### 6.1 `balanced`

| 항목 | 값 |
|------|-----|
| 게이트 클램프 | 없음 (`gate_lo_floor` / `gate_hi_cap` = `None`). OOF 적용 구간 그대로. |
| `uwb_variance_inflate` | 연속 **10 ~ 38** |
| `huber_f_fusion` | categorical **1.0, 1.12, 1.25, 1.35** |
| `irls_residual_thresh_m` | categorical **2.1, 2.45, 2.75, 3.05, 3.2** (등간격 그리드 아님; 상단 3 m 부근 로컬 촘촘 의도는 소스 주석 참고). |
| `blend_calib_alpha` 등 | 소스 내 `blend_grid` / `step_a_blend_grid` |

### 6.2 `defensive`

UWB 과신·넓은 τ 조합 완화용 **설계 좁은 박스**(검증 라벨 역산 아님).

| 항목 | 값 |
|------|-----|
| 게이트 추가 클램프 | `≥ V15_DEFENSIVE_GATE_LO_MIN`(기본 **1.15** m), `≤ min(3.95, V15_DEFENSIVE_GATE_HI_MAX)`(기본 **2.8** m) |
| inflate | 연속 **`V15_DEFENSIVE_INFLATE_LO`~HI**(기본 **14 ~ 33**) |
| `huber_f_fusion` | **1.0, 1.12** 만 |
| IRLS | **2.45, 2.75, 3.05** |
| 블렌드 그리드 | **0.2 ~ 1.0** 근처(순 보정 극단 0 제외) |

### 6.3 `v13_fix`

`indoor_fusion_pipeline_v13_fix.make_objective_clamped`와 **동일한 탐색 박스**(게이트 **1.25~2.05** 연속, inflate **8~24**, fusion **1.0/1.2/1.35**, IRLS **2.2, 2.75, 3.05, 3.1** 등). 세부 클램프는 `V15_V13FIX_*` 환경 변수로 덮어쓸 수 있다.

---

## 7. 환경 변수 요약

| 변수 | 기본 | 설명 |
|------|------|------|
| `V15_TUNING_PROFILE` | `balanced` | `balanced` / `defensive` / `v13_fix` (+별칭) |
| `V15_OPTUNA_TRIALS` | `120` | 단일 단계 또는 2단계 중 Step C(융합) 트라이얼 수 |
| `V15_OPTUNA_TRIALS_STEPA` | `48` | `V15_TWO_PHASE=1`일 때 Step A 전용 |
| `V15_OPTUNA_SEED` | `42` | TPE 시드(2단계 시 Step C용 시드는 +17 오프셋) |
| `V15_OPTUNA_JOBS` | `V13_FIX_OPTUNA_JOBS` 또는 `-1` | 병렬 워커 |
| `V15_MAX_KFOLD` | `3` | K-Fold 상한 |
| `V15_TWO_PHASE` | `0` | `1`이면 Step A 최적화 후 가이드 고정·Step C만 탐색 |
| `V15_STEPA_LOSS_WEIGHT` | `0.452` | composite 목적에서 Step A 가중치 `w` |
| `V15_ALLOW_KNN` | `1` | KNN 잔차 정제 허용 |
| `V15_TOP_V3_PROFILES` | `8` | Train 순위 V3 최대 개수(최소 6과 max 조합은 코드 참고) |
| `V15_TOP_CALIB_RANKS` | `0` | `0`이면 캘리브 후보 전부 캐시 |
| `V15_PLOT_DPI` | `110` | PNG 해상도 |
| `V15_DEFENSIVE_GATE_LO_MIN` | `1.15` | defensive 하한 |
| `V15_DEFENSIVE_GATE_HI_MAX` | `2.8` | defensive 상한(상한 `min(3.95, …)`와 결합) |
| `V15_DEFENSIVE_INFLATE_LO` / `HI` | `14` / `33` | defensive inflate |
| `V15_V13FIX_GATE_LO` / `HI` 등 | v13_fix 박스와 동기 | v13_fix 프로필 덮어쓰기 |

---

## 8. 실행 예 (PowerShell)

프로젝트 루트에서:

```powershell
py -3 .\scripts\indoor_fusion_pipeline_v15.py
py -3 .\scripts\indoor_fusion_pipeline_v15.py --no-plots
```

프로필·트라이얼 예:

```powershell
$env:V15_TUNING_PROFILE="defensive"
$env:V15_OPTUNA_TRIALS="12"
py -3 .\scripts\indoor_fusion_pipeline_v15.py
```

2단계·가중치 예:

```powershell
$env:V15_TWO_PHASE="1"
$env:V15_STEPA_LOSS_WEIGHT="0.2"
py -3 .\scripts\indoor_fusion_pipeline_v15.py --no-plots
```

**필수 패키지**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `numba`, `optuna`

---

## 9. 산출물

| 파일 | 설명 |
|------|------|
| `outputs/v15_summary.json` | 무결성 문구, `tuning_profile`, `tuning_space`, OOF 게이트 raw/적용, 선정 하이퍼, 검증 RMSE/MAE |
| `outputs/v15_predictions.csv` | 검증 샘플별 Step A/B/C 좌표·오차 |
| `outputs/v15_optuna_trials.csv` | 트라이얼 로그(위상·composite_loss 등) |
| `outputs/v15_grid_phaseA_calib_catalog.csv` | 캘리브 후보 Train CV 메타 |
| `outputs/v15_wifi_calibration_coefs.csv` | 최종 Train 재피팅 Wi‑Fi 보정 계수 |
| `outputs/v15_*_steps.png`, `v15_cdf_steps.png` | `--no-plots` 시 생략 |
| `outputs/run_reports/indoor_fusion_pipeline_v15_<UTC>/` | `run.log`, `manifest.json` |

접두어 상수: `ART_PREFIX = "v15"`.

---

## 10. V13-fix와의 차이(요약)

| 항목 | V13-fix | V15(기본) |
|------|---------|-----------|
| 게이트 τ | 고정 연속구간 **1.25 ~ 2.05** m | **Train OOF** + 프로필별 클램프 |
| Optuna 목적 | Train CV **Step C만** | **Composite** 또는 **2단계** |
| V3 | 고정 5개 리스트 | **Train 순위형**(+앵커, hf≤1.35 필터) |
| 캘리브 카탈로그 | `calibration_catalog_extended()` | **`v15_calibration_catalog()`** |
| 탐색 박스 | `make_objective_clamped` 단일 정의 | **프로필별** (`balanced` / `defensive` / `v13_fix`) |

`V15_TUNING_PROFILE=v13_fix`이면 탐색 **박스**는 v13-fix와 맞추되, 목적 함수·V3·캘리브 목록 등은 여전히 **V15 파일의 구현**이다.

---

## 11. 보고 시 주의

- Composite 목적으로 인해 **Train CV 최적점과 검증 Step A 불일치**가 날 수 있다.
- **`V15_OPTUNA_TRIALS`가 작으면** 분산·우연 의존도가 커진다.
- **`OPTUNA_N_JOBS=-1`(병렬)**이면 재현성이 실행마다 완전히 동일하지 않을 수 있다.
