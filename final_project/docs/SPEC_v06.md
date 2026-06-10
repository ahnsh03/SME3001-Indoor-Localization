# SPEC v06 — Huber + BS별 Isotonic 보정 ★ Production

## 추가 기법 (v05 대비)

**단 1가지:** BS \(k\)마다 **단조** 회귀 \(\hat{d}_k \rightarrow d_{\mathrm{true},k}\)

- `sklearn.isotonic.IsotonicRegression`, `out_of_bounds='clip'`.
- EDA Spearman ≈ 0.83 → **순위는 유지, 스케일만 보정** 가정과 일치.
- HardGate **미사용**.

## 설정

| 항목 | 값 |
|------|-----|
| `calib` | `per_bs_isotonic` |
| `loss` | huber, `f_scale=1.0` |
| `gate` | off |

## CV 결과

| 지표 | v02 | v05 | v06 | Δ (vs v05) |
|------|-----|-----|-----|------------|
| OOF RMSE (m) | 12.40 | 11.56 | **10.51** | **−1.05** |
| vs v01 (Huber만) | — | — | — | **−6.20** (누적) |

### 스테이지 분해

| 스테이지 | RMSE (m) |
|----------|----------|
| after_raw_trilat | 16.71 |
| after_main_trilat | **10.51** |

| 지표 | 값 |
|------|-----|
| MAE | 8.67 m |
| median | 7.34 m |
| P90 | 15.55 m |
| dist MAE after calib | 9.85 m |

## train.py / main.py

- `config.json`: `calib.mode = per_bs_isotonic`, `iso_x`/`iso_y` per BS.
- `model_calib.npz`: isotonic knot arrays (`allow_pickle`).
- train-fit RMSE **10.40 m** (700명, CV 10.51 m과 근접).

## Fold별 val RMSE (m)

| fold | 0 | 1 | 2 | 3 | 4 |
|------|---|---|---|---|---|
| RMSE | 9.52 | 10.89 | 10.12 | 11.42 | 10.58 |

## 산출물

- `outputs/v06_summary.json`
- `config.json`, `model_calib.npz` (production)

## 실행

```powershell
py -3 scripts/run_all_versions.py --versions v06
py -3 train.py --version v06
py -3 main.py
```

## 보고서 메시지

> 가공 RTT는 **채널별 단조 보정 + Huber 삼변**으로 v01 대비 약 **6.2 m**, v02 대비 **1.9 m** CV 개선. 중간 프로젝트 HardGate는 실측 UWB용이며 기말에는 **적용하지 않음**.
