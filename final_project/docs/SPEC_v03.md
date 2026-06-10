# SPEC v03 — Huber + BS별 아핀 보정

## 추가 기법 (v02 대비)

**단 1가지:** BS \(k\)마다 별도 \((\alpha_k,\beta_k)\)

\[
\tilde{d}_{k,u} = \alpha_k\,\hat{d}_{k,u} + \beta_k
\]

- 파라미터 **36개** (18×2).
- fold train UE만으로 채널별 OLS.

## 설정

| 항목 | 값 |
|------|-----|
| `calib` | `per_bs_affine` |
| `loss` | `huber` |
| `gate` | off |

## CV 결과

| 지표 | v02 | v03 | Δ |
|------|-----|-----|---|
| OOF RMSE (m) | 12.40 | **12.52** | **+0.11** (악화) |

스테이지:

| 스테이지 | RMSE (m) |
|----------|----------|
| after_raw_trilat | 16.71 |
| after_main_trilat | 12.52 |

## 해석

- EDA에서 BS별 bias가 달라 보였으나, **CV 기준**으로는 전역 아핀 2파라미터가 더 일반화됨.
- 700명·18채널에서 채널별 36파라미터는 **분산 증가** 가능.
- **Production은 v02 유지**; v03은 ablation·보고서용.

## 산출물

- `outputs/v03_summary.json`

## 보고서 표 예시

| 모델 | 파라미터 수 | CV RMSE (m) |
|------|-------------|-------------|
| v02 global | 2 | 12.40 |
| v03 per-BS | 36 | 12.52 |
