# v26 — far UE 2차 Isotonic (미채택)

- **base:** v17 + `isotonic_mlp_far2`
- far(상위 33% `min(d_hat)`) train UE만, pass1 보정 거리→GT로 2차 Isotonic
- **CV OOF RMSE:** 9.28 m (+0.38 vs v17)
