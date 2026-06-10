# SPEC v02 — Huber + 전역 아핀 보정 ★ Production

## 추가 기법 (v01 대비)

**단 1가지:** Train fold에서 **전역** 아핀 보정

\[
\tilde{d}_{k,u} = \alpha\,\hat{d}_{k,u} + \beta
\]

- \(\alpha,\beta\): fold train 560명 × 18채널을 풀어 OLS 1회 (파라미터 **2개**).
- Val fold에는 train에서 추정한 \((\alpha,\beta)\)만 적용 → **누수 없음**.

## 설정

| 항목 | 값 |
|------|-----|
| `calib` | `global_affine` |
| `loss` | `huber`, `f_scale=1.0` |
| `gate` | off |

## CV 결과

| 지표 | v01 | v02 | Δ |
|------|-----|-----|---|
| OOF RMSE (m) | 16.71 | **12.40** | **−4.30** |

### 스테이지 분해 (효과 분리)

| 스테이지 | RMSE (m) | 설명 |
|----------|----------|------|
| after_raw_trilat | 16.71 | 보정 **전** Huber (v01과 동일) |
| after_main_trilat | **12.40** | 보정 **후** Huber |
| dist MAE after calib | 10.88 | 거리域 diagnostic (튜닝 미사용) |

→ **4.3 m 개선은 전역 아핀 한 단계에서 발생** (게이트·BS별 파라미터 없음).

## train.py / main.py

- `train.py`가 catalog에서 최소 RMSE 버전 선택 → **v02**.
- `config.json`, `model_calib.npz`에 \(\alpha,\beta\) 저장.
- `main.py`는 `DH_FR1.mat` 로드 후 동일 파이프라인.

전체 700명 train-fit RMSE: **12.29 m** (CV 12.40 m과 근접 → 과적합 징후 약함).

## Fold별 RMSE (요약)

| fold | val RMSE (m) |
|------|----------------|
| 0 | 11.40 |
| 1 | 12.88 |
| 2 | 11.86 |
| 3 | 13.52 |
| 4 | 12.26 |

## 산출물

- `outputs/v02_summary.json`
- `outputs/v02_cv_folds.csv`
- `config.json` (production)

## 다음 버전과의 관계

v03(BS별 아핀)은 **+0.11 m 악화** — 파라미터 36개가 700명에 과한 가능성. 보고서에서 “단순 모델 선호” 근거로 사용.
