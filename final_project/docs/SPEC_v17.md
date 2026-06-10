# v17 — Isotonic + MLP + soft weight (Production ★)

- **calib:** `isotonic_mlp`
- **삼변:** Huber + `weight_gamma=1.0` (CV, v11과 동일 축)
- **bounds:** fold train UE span + 2 m
- **CV OOF RMSE:** **8.90 m** (median **6.88 m**, p90 **12.1 m**)
- **Hold-out 140:** **9.31 m** (560 train)
- **재현:** `py -3 train.py --version v17`
