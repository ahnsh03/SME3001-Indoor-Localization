# v30 — v25 + 비대칭 Huber CV (Production ★)

- **base:** `isotonic_mlp` + `weight_gamma=1.0` + `pos_refine_affine`
- **추가:** `asym_pos_weight=5.0` (pred > d_obs 잔차 강화, NLOS 일관)
- **CV OOF RMSE:** **7.996 m** (v25 8.76 m, **−0.77 m**)
- **median:** 6.21 m · **p90:** 11.18 m
- **재현:** `py -3 train.py --version v30`

**통찰:** v14는 Isotonic 단독에서만 시험해 10.3 m. **MLP+아핀 스택과 결합**하니 처음으로 효과가 남.
