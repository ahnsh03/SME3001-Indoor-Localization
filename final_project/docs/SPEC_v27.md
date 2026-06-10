# v27 — quantile τ 스윕 + MLP (미채택)

- **calib:** `quantile_mlp`, τ ∈ {0.25, 0.30, 0.35, 0.40, 0.45} CV
- **best τ:** 0.35 (v22와 동일)
- **CV OOF RMSE:** 9.61 m (+0.71 vs v17)
- Isotonic+MLP가 quantile보다 우수
