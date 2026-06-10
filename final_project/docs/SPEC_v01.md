# SPEC v01 — B1 Huber 삼변 (보정 없음)

## 추가 기법 (v00 대비)

**단 1가지:** `scipy.optimize.least_squares(..., loss='huber', f_scale=1.0)`

## 설정

| 항목 | 값 |
|------|-----|
| `calib` | `none` |
| `loss` | `huber` |
| `gate` | off |

## CV 결과

| 지표 | v00 | v01 | Δ |
|------|-----|-----|---|
| OOF RMSE (m) | 22.42 | **16.71** | **−5.71** |

스테이지 진단 (`oof_stage_position`):

| 스테이지 | RMSE (m) |
|----------|----------|
| after_raw_trilat | 16.71 |
| after_main_trilat | 16.71 |

## 해석

- 합성 RTT에 **이상치·heavy-tail** 성분이 있어 LS( v00 )보다 Huber가 유리.
- 아직 **+16 m급 체계 bias**가 남아 있어 보정 단계(v02)가 필요.

## 산출물

- `outputs/v01_summary.json`

## 다음 버전

v02: **전역 아핀 보정 1개** (α, β) 추가 → **−4.30 m**.
