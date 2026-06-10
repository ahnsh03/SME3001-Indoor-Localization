# SPEC v00 — B0 LS 삼변 (보정 없음)

## 목적

교과서 하한 baseline. 이후 모든 개선의 기준점.

## 설정

| 항목 | 값 |
|------|-----|
| `calib` | `none` |
| `loss` | `linear` (최소자승) |
| `gate` | off |

## 알고리즘

1. UE \(u\)의 \(\hat{\mathbf{d}}_{:,u}\) (18,) 그대로 사용.
2. \(\mathbf{p}_u = \arg\min_{\mathbf{p}} \sum_k \big(\|\mathbf{p}-\mathbf{a}_k\| - \hat{d}_{k,u}\big)^2\) (bounds 적용).

## CV 결과 (5-Fold OOF)

| 지표 | 값 (m) |
|------|--------|
| RMSE | **22.42** |
| MAE | 18.05 |
| median | 15.21 |
| P90 | 32.48 |

스테이지: `after_raw_trilat` = `after_main_trilat` (보정 없음).

## 산출물

- `outputs/v00_summary.json`
- `outputs/v00_cv_folds.csv`

## 실행

```powershell
py -3 scripts/run_all_versions.py --versions v00
```

## 다음 버전과의 차이

v01에서 **Huber loss**만 추가 → RMSE **−5.71 m**.
