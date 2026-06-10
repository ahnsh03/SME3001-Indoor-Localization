# v25 — 2-pass 위치 아핀 (Production ★)

- **base:** v17 (`isotonic_mlp` + `weight_gamma=1.0`)
- **2-pass:** fold train UE에 대해 1차 삼변 → `(2×3)` 아핀 `p' = M @ [x,y,1]` (GT 미사용 추론)
- **CV OOF RMSE:** **8.76 m** (v17 8.90 m, **−0.14 m**)
- **p90:** 11.71 m (v17 12.08 m)
- **재현:** `py -3 train.py --version v25`
