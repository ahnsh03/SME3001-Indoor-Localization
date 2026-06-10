# SPEC v05 — Huber + BS별 2차 보정

## 추가 기법 (v02 대비)

**단 1가지:** 채널 \(k\)마다 2차 다항 보정

\[
\tilde{d}_k = a_k\,\hat{d}_k + b_k\,\hat{d}_k^2 + c_k
\]

- 파라미터 **54개** (18×3).
- fold **train** UE만으로 OLS, val에 적용.

## 설정

| 항목 | 값 |
|------|-----|
| `calib` | `per_bs_quadratic` |
| `loss` | huber, `f_scale=1.0` |
| `gate` | **off** |

## CV 결과 (5-Fold OOF, seed=42)

| 지표 | v02 | v05 | Δ |
|------|-----|-----|---|
| OOF RMSE (m) | 12.40 | **11.56** | **−0.84** |

### 스테이지 분해

| 스테이지 | RMSE (m) |
|----------|----------|
| after_raw_trilat | 16.71 |
| after_main_trilat | **11.56** |

`oof_dist_mae_after_calib_mean_m` ≈ 10.2 m (거리域 diagnostic).

## 해석

- 전역 아핀(v02)보다 **거리 비선형** 왜곡을 채널별로 흡수.
- v06(Isotonic)보다는 CV **+1.05 m** — 단조 제약이 이 데이터에 더 맞음.

## 산출물

- `outputs/v05_summary.json`, `v05_cv_folds.csv`

## 실행

```powershell
py -3 scripts/run_all_versions.py --versions v05
```
