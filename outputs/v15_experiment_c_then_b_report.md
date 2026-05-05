# V15 Experiment Report (Reverse + Conditional C)

- enabled: `True`
- conditional C enabled: `True`
- compared pipelines:
  - `A`
  - `A_to_B`
  - `A_to_B_to_C`
  - `A_to_C_first`
  - `A_to_C_then_B`
  - `A_to_B_conditional_C`

- best RMSE pipeline: `A_to_B` (1.6798 m)

## Metrics
- A: RMSE=1.7231 m, MAE=1.4484 m
- A_to_B: RMSE=1.6798 m, MAE=1.3837 m
- A_to_B_to_C: RMSE=1.6985 m, MAE=1.4005 m
- A_to_C_first: RMSE=1.6985 m, MAE=1.4005 m
- A_to_C_then_B: RMSE=1.9850 m, MAE=1.5106 m
- A_to_B_conditional_C: RMSE=1.6836 m, MAE=1.3861 m
