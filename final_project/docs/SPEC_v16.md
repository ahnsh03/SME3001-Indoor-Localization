# v16 — Isotonic + MLP (Production ★)

- **calib:** `isotonic_mlp` — fold별 `per_bs_isotonic` 학습 후 MLP 잔차 보정
- **삼변:** Huber
- **CV OOF RMSE:** **9.38 m** (v11 9.77 m 대비 −0.39 m)
- **train-fit RMSE:** 8.60 m (참고, 과적합 주의)
- **산출:** `config.json`, `model_calib.npz`, `model_mlp.pt`
- **재현:** `py -3 train.py --version v16`
