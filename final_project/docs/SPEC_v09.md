# SPEC v09 — v05 + Huber f_scale CV

## 추가 기법 (v05 대비)

**단 1가지:** `scipy` Huber `f_scale` 그리드 탐색

- 후보: `{0.5, 0.7, 0.8, 1.0, 1.2, 1.35, 1.5, 2.0}`
- 목적: 5-Fold **OOF RMSE** 최소
- 보정: v05와 동일 `per_bs_quadratic`

## CV 결과 (본 실행)

| 항목 | 값 |
|------|-----|
| best `huber_f_scale` | **0.8** |
| OOF RMSE (m) | **11.48** |
| v05 OOF RMSE | 11.56 |
| Δ vs v05 | **−0.08** (소폭 개선) |

→ **v06(10.51 m)보다 낮지 않음** — production에는 v06 유지.

## grid

`outputs/v09_summary.json` → `f_scale_tuning.grid`

## 실행

```powershell
py -3 scripts/run_all_versions.py --versions v09
```

## 비고

Isotonic(v06)이 우수하므로, f_scale 튜닝은 **v06 위에서 재실험**할 여지 있음 (향후 v10).
