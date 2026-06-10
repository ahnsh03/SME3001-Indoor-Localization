"""Mixture Density Network for (x,y) from calibrated distances + anchor geometry."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calib import fit_calib
from lib.io_mat import load_dataset
from lib.pipeline import VERSION_REGISTRY


class MDNHead(nn.Module):
    def __init__(self, in_dim: int, n_mix: int = 5, hidden: int = 128) -> None:
        super().__init__()
        self.n_mix = n_mix
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_mix)
        self.mu = nn.Linear(hidden, n_mix * 2)
        self.log_sigma = nn.Linear(hidden, n_mix)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        pi = F.softmax(self.pi(h), dim=-1)
        mu_sigma = self.mu(h)
        mu = mu_sigma.view(-1, self.n_mix, 2)
        log_sigma = self.log_sigma(h).view(-1, self.n_mix, 1)
        sigma = torch.exp(log_sigma).clamp(min=1e-3, max=50.0)
        return pi, mu, sigma


def nll_gaussian_mixture(y: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # y: (B,2), pi: (B,K), mu: (B,K,2), sigma: (B,K,1) or (B,K,2)
    y_exp = y.unsqueeze(1)
    if sigma.shape[-1] == 1:
        sigma = sigma.expand(-1, -1, mu.shape[-1])
    var = sigma**2
    log_prob = -0.5 * (
        torch.log(var)
        + ((y_exp - mu) ** 2) / var
        + 2 * torch.log(sigma)
        + np.log(2 * np.pi)
    )
    log_prob = log_prob.sum(dim=-1)  # (B, K)
    log_mix = torch.log(pi + 1e-8) + log_prob
    return -torch.logsumexp(log_mix, dim=1).mean()


def build_features(d_hat_u: np.ndarray, bs: np.ndarray, d_cal: np.ndarray) -> np.ndarray:
    if bs.shape[0] != 2:
        bs = bs.T
    bx, by = bs[0], bs[1]
    d = d_cal.ravel()
    return np.hstack(
        [
            d,
            bx,
            by,
            [d.min(), d.max(), d.mean(), d.std()],
            [np.min(bx), np.max(bx), np.mean(bx)],
            [np.min(by), np.max(by), np.mean(by)],
        ]
    ).astype(np.float32)


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    n = d_hat.shape[1]
    cfg = VERSION_REGISTRY["v30"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((2, n), dtype=np.float64)
    device = torch.device("cpu")

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        calib = fit_calib(d_hat, p, bs, tr, cfg.calib)

        X_list, Y_list = [], []
        for u in tr:
            dc = calib.apply(d_hat[:, u])
            X_list.append(build_features(d_hat[:, u], bs, dc))
            Y_list.append(p[:, u])
        Xtr = torch.from_numpy(np.stack(X_list))
        Ytr = torch.from_numpy(np.stack(Y_list)).float()  # (N, 2)

        model = MDNHead(in_dim=Xtr.shape[1], n_mix=6, hidden=160).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        best_loss = 1e18
        best_state = None

        for ep in range(200):
            model.train()
            perm = torch.randperm(Xtr.shape[0])
            loss_sum = 0.0
            for i in range(0, Xtr.shape[0], 64):
                idx = perm[i : i + 64]
                xb = Xtr[idx]
                yb = Ytr[idx]
                pi, mu, sigma = model(xb)
                loss = nll_gaussian_mixture(yb, pi, mu, sigma)
                loss_sum = loss_sum + loss
            loss_sum = loss_sum / max(1, (Xtr.shape[0] + 63) // 64)
            opt.zero_grad()
            loss_sum.backward()
            opt.step()
            if loss_sum.item() < best_loss:
                best_loss = loss_sum.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for u in va:
                dc = calib.apply(d_hat[:, u])
                x = torch.from_numpy(build_features(d_hat[:, u], bs, dc)[None, :]).float()
                pi, mu, sigma = model(x)
                # mode of mixture (discrete pi) for inference
                k = torch.argmax(pi, dim=-1)
                xy = mu[torch.arange(1), k].squeeze().cpu().numpy()
                oof[:, u] = xy

        e = np.hypot(oof[0] - p[0], oof[1] - p[1])
        print(f"fold={fold} rmse={float(np.sqrt(np.mean(e[va]**2))):.3f}")

    print(
        f"mdn_xy_oof rmse={float(np.sqrt(np.mean(e**2))):.3f} "
        f"median={float(np.median(e)):.3f} p90={float(np.percentile(e,90)):.3f}"
    )


if __name__ == "__main__":
    main()
