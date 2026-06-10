# v18 — PosMLP 직접 좌표 회귀 (미채택)

- **calib:** `pos_mlp` (d̂ 18차원 → x,y)
- **CV OOF RMSE:** 38.06 m
- **판단:** 700 UE에서 end-to-end 위치 회귀는 fold마다 불안정·과적합. geometry(삼변) 경로 유지.
