# 스크립트 명세서 색인

`scripts/` 아래 **모든 Python 모듈**에 대한 상세 명세를 아래 파일로 나누어 두었다. 파이프라인은 **이전 버전 대비 동기·결과·수식**을 중심으로 서술하였다.

| 문서 | 대상 |
|------|------|
| [SPEC_fusion_core.md](SPEC_fusion_core.md) | `fusion_realtime_sanitize.py`, `fusion_data_audit.py`, `fusion_turbo_numba_core.py` |
| [SPEC_evolution_v01_v07.md](SPEC_evolution_v01_v07.md) | `indoor_fusion_pipeline_v1.py` ~ `v7.py` |
| [SPEC_evolution_v08_v11.md](SPEC_evolution_v08_v11.md) | `v8` ~ `v11`, `v9_strict`, `v10_optimized` |
| [SPEC_v12.md](SPEC_v12.md) | `v12.py`, `v12_strict`, `v12_fast`, `v12_fast2`, `v12_turbo`, `v12_strict_parallel` (v12 계열 단독 상세) |
| [SPEC_v13.md](SPEC_v13.md) | `v13.py`, `v13_fix.py` (v13 계열 단독 상세) |
| [SPEC_v14.md](SPEC_v14.md) | `v14.py` 단독 상세 + JWT ablation 해석 |
| [SPEC_v12_v13_family.md](SPEC_v12_v13_family.md) | V12~V14 공통 수식/관계 요약(요약본) |
| [SPEC_analysis_and_JWT.md](SPEC_analysis_and_JWT.md) | `sensor_spatial_profiler.py`, `validation_calibration_filter_scatter.py`, `indoor_fusion_pipeline_JWT.py` |

**기존 집중 명세(유지·보완)**

| 문서 | 내용 |
|------|------|
| [V12_TURBO_PIPELINE_SPEC.md](V12_TURBO_PIPELINE_SPEC.md) | V12 Turbo 전용: Phase A 의미, Optuna, Numba, 산출물 |
| [V13_FINAL_PIPELINE_SPEC.md](V13_FINAL_PIPELINE_SPEC.md) | 최종 선정 V13: 실행, 환경변수, Optuna 공간, 산출물 |

위 두 파일 상단에는 본 색인으로 오는 링크를 추가하였다.

---

## 스크립트 ↔ 문서 매핑 (빠른 찾기)

| 파일 | 주된 명세 위치 |
|------|----------------|
| `fusion_realtime_sanitize.py` | SPEC_fusion_core.md |
| `fusion_data_audit.py` | SPEC_fusion_core.md |
| `fusion_turbo_numba_core.py` | SPEC_fusion_core.md + V12_TURBO_PIPELINE_SPEC.md |
| `indoor_fusion_pipeline_v1.py` ~ `v7.py` | SPEC_evolution_v01_v07.md |
| `indoor_fusion_pipeline_v8.py` ~ `v11.py` | SPEC_evolution_v08_v11.md |
| `indoor_fusion_pipeline_v9_strict.py` | SPEC_evolution_v08_v11.md |
| `indoor_fusion_pipeline_v10_optimized.py` | SPEC_evolution_v08_v11.md |
| `indoor_fusion_pipeline_v12*.py` | SPEC_v12.md, V12_TURBO_PIPELINE_SPEC.md |
| `indoor_fusion_pipeline_v13*.py` | SPEC_v13.md, V13_FINAL_PIPELINE_SPEC.md |
| `indoor_fusion_pipeline_v14.py` | SPEC_v14.md |
| `sensor_spatial_profiler.py` | SPEC_analysis_and_JWT.md |
| `validation_calibration_filter_scatter.py` | SPEC_analysis_and_JWT.md |
| `indoor_fusion_pipeline_JWT.py` | SPEC_analysis_and_JWT.md |
