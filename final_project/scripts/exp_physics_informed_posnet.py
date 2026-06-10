from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import load_dataset
from lib.calib import CalibParams
from lib.mlp_calib import bundle_from_dict as mlp_bundle_from_dict
from lib.pipeline import PipelineConfig
from lib.spatial import X_BOUNDS, Y_BOUNDS


class PosNet(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, d_corr: torch.Tensor) -> torch.Tensor:
        # output in normalized coords then map to bounds via sigmoid
        z = self.net(d_corr)
        a = torch.sigmoid(z)
        x = X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * a[:, 0]
        y = Y_BOUNDS[0] + (Y_BOUNDS[1] - Y_BOUNDS[0]) * a[:, 1]
        return torch.stack([x, y], dim=1)


def _rmse(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2)))


def _load_model_bundle(root: Path) -> CalibParams:
    import torch

    z = torch.load(root / "model_mlp.pt", map_location="cpu", weights_only=False)
    calib = CalibParams.from_dict(z["calib"])
    calib.mlp_bundle = mlp_bundle_from_dict(z["meta"], z["state_dict"])
    return calib


def huber_loss(r: torch.Tensor, f_scale: float) -> torch.Tensor:
    abs_r = torch.abs(r)
    quad = torch.minimum(abs_r, torch.tensor(f_scale, device=r.device))
    lin = abs_r - quad
    # Huber: 0.5*(r^2) for |r|<=f_scale, f_scale*(|r|-0.5*f_scale) otherwise
    return 0.5 * quad**2 + f_scale * lin


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]  # p: (2,N), d_hat: (18,N), bs:(2,18)
    N = d_hat.shape[1]

    calib = _load_model_bundle(ROOT)

    # precompute calibrated distances d_corr (N,18)
    d_corr = np.zeros((N, 18), dtype=np.float32)
    for u in range(N):
        d_corr[u] = calib.apply(d_hat[:, u]).astype(np.float32)

    bs_t = torch.from_numpy(bs.astype(np.float32))  # (2,18)
    d_t = torch.from_numpy(d_corr)  # (N,18)

    p_t = torch.from_numpy(p.T.astype(np.float32))  # (N,2)

    # hyperparams (aligned with v30)
    asym_pos_weight = 5.0
    f_scale = 1.0

    model = PosNet(hidden=128, dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    model.train()
    batch = 64
    epochs = 200

    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch):
            idx = perm[i : i + batch]
            d_b = d_t[idx]  # (B,18)
            xy = model(d_b)  # (B,2)

            x = xy[:, 0].unsqueeze(1)  # (B,1)
            y = xy[:, 1].unsqueeze(1)
            bx = bs_t[0].unsqueeze(0)  # (1,18)
            by = bs_t[1].unsqueeze(0)
            pred_dist = torch.sqrt((bx - x) ** 2 + (by - y) ** 2)  # (B,18)
            r = pred_dist - d_b  # (B,18)

            # asymmetric weighting for positive residual only
            w_pos = torch.where(r > 0, torch.tensor(asym_pos_weight, device=r.device), torch.tensor(1.0, device=r.device))
            r_w = r * torch.sqrt(w_pos)

            loss = huber_loss(r_w, f_scale=f_scale).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                xy_all = model(d_t)
                rmse = float(torch.sqrt(torch.mean(torch.sum((xy_all - p_t) ** 2, dim=1))))
            print(f"epoch={ep+1} phys-loss rmse(train)={rmse:.3f}")

    model.eval()
    with torch.no_grad():
        xy_all = model(d_t).numpy().T  # (2,N)
    rmse = _rmse(p, xy_all)
    print(f"physics-informed PosNet trainfit rmse={rmse:.3f}")


if __name__ == "__main__":
    main()

