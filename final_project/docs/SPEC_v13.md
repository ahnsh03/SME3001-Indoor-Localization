# v13 — MLP 거리 보정

- **calib:** `mlp_dist` (PyTorch `DistMLP`, 18→64→64→18 residual)
- **삼변:** Huber
- **CV OOF RMSE:** 10.81 m (v06 대비 악화)
- **판단:** Isotonic 없이 MLP만으로는 700 UE에서 과소·과적합; **미채택**
