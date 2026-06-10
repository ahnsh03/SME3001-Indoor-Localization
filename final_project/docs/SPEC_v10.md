# SPEC v10 — v06 + Huber f_scale CV

## 관찰 (D-)

v09에서 **quadratic** 보정에 `f_scale=0.8`이 소폭 이득 → **Isotonic(v06)에도 적용** 가치.

## 추가 기법

Huber `f_scale` 그리드 CV (v06 동일 `per_bs_isotonic`).

## CV 결과

| | v06 | v10 | Δ |
|---|-----|-----|---|
| OOF RMSE (m) | 10.51 | **10.15** | **−0.36** |
| best f_scale | 1.0 | **0.8** | |

## 판단

삼변 robustness만 조정해도 **0.36 m** 이득. v11보다는 낮음 → production 후보 2순위.

## 산출물

`outputs/v10_summary.json`, `f_scale_tuning` 필드.
