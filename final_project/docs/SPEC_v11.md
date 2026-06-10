# SPEC v11 — v06 + soft 거리 가중 삼변 ★ Production

## 관찰 (D6)

`decision_analysis.json` **regional_rmse**: 원거리 UE일수록 오차 큼.  
멀수록 RTT 불확실 → **가까운 BS에 더 큰 가중** (HardGate 아님).

## 추가 기법

보정 후 Huber 삼변 잔차에 가중 \(w_k \propto 1/\tilde{d}_k^{\gamma}\), \(\gamma\)는 CV grid.

| gamma | CV RMSE (m) |
|-------|-------------|
| 0 | 10.51 |
| 1.0 | ~10.0 |
| **1.5** | **9.77** (best) |

## CV 결과

| | v06 | v11 | Δ |
|---|-----|-----|---|
| OOF RMSE (m) | 10.51 | **9.77** | **−0.74** |
| vs v01 | 16.71 | **9.77** | **−6.94** |

### 스테이지

| 스테이지 | RMSE (m) |
|----------|----------|
| after_raw_trilat | 13.53 (가중 적용 Huber) |
| after_main_trilat | **9.77** |

## production

- `weight_gamma: 1.5`
- `calib: per_bs_isotonic`
- `config.json`, `model_calib.npz`

## 실행

```powershell
py -3 train.py --version v11
```
