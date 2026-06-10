# v14 — NLOS 비대칭 Huber

- **calib:** `per_bs_isotonic`
- **삼변:** Huber + `asym_pos_weight` CV (pred > d_obs 잔차 강화)
- **CV OOF RMSE:** 10.33 m
- **판단:** Gemini 제안 방향은 타당하나 γ=1.5 soft weight(v11)보다 나쁨; **미채택**
