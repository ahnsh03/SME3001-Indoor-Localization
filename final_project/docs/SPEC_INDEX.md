# 기말 측위 파이프라인 명세 색인

## 버전표 (2026-05-25)

| ver | 기법 | CV RMSE (m) | Production |
|-----|------|-------------|------------|
| [v00](SPEC_v00.md) | LS | 22.42 | |
| [v01](SPEC_v01.md) | Huber | 16.71 | |
| [v02](SPEC_v02.md) | +전역 아핀 | 12.40 | |
| [v03](SPEC_v03.md) | +BS 아핀 | 12.52 | |
| [v04](SPEC_v04.md) | HardGate | 12.78 | **deprecated** |
| [v05](SPEC_v05.md) | +2차 보정 | 11.56 | |
| [v06](SPEC_v06.md) | +Isotonic | 10.51 | |
| [v10](SPEC_v10.md) | +f_scale CV | 10.15 | |
| [v11](SPEC_v11.md) | +soft weight γ=1.5 | 9.77 | |
| [v12](SPEC_v12.md) | +3구간 Iso | 10.23 | |
| [v13](SPEC_v13.md) | MLP 거리만 | 10.81 | |
| [v14](SPEC_v14.md) | 비대칭 Huber | 10.33 | |
| [v15](SPEC_v15.md) | shortest-k | 10.51 | |
| [v16](SPEC_v16.md) | Isotonic+MLP | 9.28† | |
| [v17](SPEC_v17.md) | v16+γ=1.0 | 8.90† | |
| [v25](SPEC_v25.md) | 2-pass 위치 아핀 | 8.76 | |
| **[v30](SPEC_v30.md)** | **v25+asym Huber w=5** | **7.996** | **★** |
| [v26](SPEC_v26.md) | far 2차 Iso | 9.28 | ✗ |
| [v27](SPEC_v27.md) | quantile τ CV | 9.61 | ✗ |
| [v18](SPEC_v18.md) | PosMLP 직접 | 38.06 | ✗ |
| [v19](SPEC_v19.md) | GBR 거리 | 11.10 | ✗ |
| [v20](SPEC_v20.md) | v16+pos blend | 10.19 | ✗ |

† UE bounds 조정 후 재측정

판단·누수: [`DEV_DECISION_LOG.md`](DEV_DECISION_LOG.md) · EDA: `docs/fig_eda_residual_hist.png`

## 실행

```powershell
py -3 scripts/audit_leakage.py
py -3 scripts/spatial_goals.py
py -3 scripts/run_experiment_loop.py
py -3 scripts/holdout_eval.py
py -3 scripts/run_combination_review.py
py -3 train.py --version v30
```
