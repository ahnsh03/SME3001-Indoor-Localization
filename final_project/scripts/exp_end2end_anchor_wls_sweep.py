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


class AnchorReliabilityWLS(nn.Module):
    def __init__(self, n_anchor: int = 18, rel_hidden: int = 32, res_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_anchor = n_anchor
        self.a_raw = nn.Parameter(torch.zeros(n_anchor))
        self.b = nn.Parameter(torch.zeros(n_anchor))
        self.rel_net = nn.Sequential(
            nn.Linear(4, rel_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rel_hidden, rel_hidden),
            nn.ReLU(),
            nn.Linear(rel_hidden, 1),
        )
        self.res_net = nn.Sequential(
            nn.Linear(n_anchor + 2, res_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden),
            nn.ReLU(),
            nn.Linear(res_hidden, 2),
        )
        self.lambda_ridge = 1e-3

    def forward(self, d_hat: torch.Tensor, bs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = d_hat.shape[0]
        bx, by = bs[0], bs[1]
        scale = torch.nn.functional.softplus(self.a_raw).unsqueeze(0)
        d_cal = torch.clamp(scale * d_hat + self.b.unsqueeze(0), min=1.0)
        mn = d_cal.min(dim=1, keepdim=True).values
        mu = d_cal.mean(dim=1, keepdim=True)
        sd = d_cal.std(dim=1, keepdim=True) + 1e-6
        z = (d_cal - mu) / sd
        feats = torch.stack([d_cal, z, mn.repeat(1, self.n_anchor), mu.repeat(1, self.n_anchor)], dim=2)
        w_anchor = torch.nn.functional.softplus(self.rel_net(feats).squeeze(-1)) + 1e-4

        pairs_i, pairs_j = [], []
        for i in range(self.n_anchor):
            for j in range(i + 1, self.n_anchor):
                pairs_i.append(i)
                pairs_j.append(j)
        ii = torch.tensor(pairs_i, device=d_hat.device, dtype=torch.long)
        jj = torch.tensor(pairs_j, device=d_hat.device, dtype=torch.long)
        xi, yi = bx[ii], by[ii]
        xj, yj = bx[jj], by[jj]
        A1 = 2.0 * (xj - xi)
        A2 = 2.0 * (yj - yi)
        di2 = d_cal[:, ii] ** 2
        dj2 = d_cal[:, jj] ** 2
        rhs = di2 - dj2 - (xi**2 - xj**2).unsqueeze(0) - (yi**2 - yj**2).unsqueeze(0)
        wp = torch.sqrt(w_anchor[:, ii] * w_anchor[:, jj])
        Aw = torch.stack([A1.unsqueeze(0).repeat(B, 1), A2.unsqueeze(0).repeat(B, 1)], dim=2) * wp.unsqueeze(2)
        bw = rhs * wp
        At = Aw.transpose(1, 2)
        AtA = torch.bmm(At, Aw)
        Atb = torch.bmm(At, bw.unsqueeze(2)).squeeze(2)
        eye = torch.eye(2, device=d_hat.device).unsqueeze(0).repeat(B, 1, 1)
        p_wls = torch.linalg.solve(AtA + self.lambda_ridge * eye, Atb.unsqueeze(2)).squeeze(2)

        dp = self.res_net(torch.cat([d_cal, p_wls], dim=1))
        p_fin = p_wls + dp
        return p_wls, p_fin


def eval_holdout(cfg: dict, Xtr: torch.Tensor, Ytr: torch.Tensor, Xva: torch.Tensor, Yva: torch.Tensor, BS: torch.Tensor) -> float:
    torch.manual_seed(42)
    model = AnchorReliabilityWLS(
        n_anchor=18,
        rel_hidden=cfg["rel_hidden"],
        res_hidden=cfg["res_hidden"],
        dropout=cfg["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_rmse = 1e9
    best_state = None
    bs_size = cfg["batch"]

    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], bs_size):
            idx = perm[i : i + bs_size]
            xb, yb = Xtr[idx], Ytr[idx]
            p_wls, p_fin = model(xb, BS)
            e_fin = torch.sqrt(torch.sum((p_fin - yb) ** 2, dim=1))
            e_wls = torch.sqrt(torch.sum((p_wls - yb) ** 2, dim=1))
            loss = (
                torch.mean(e_fin)
                + cfg["aux_wls"] * torch.mean(e_wls)
                + cfg["reg_dp"] * torch.mean(torch.sum((p_fin - p_wls) ** 2, dim=1))
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, pva = model(Xva, BS)
                eva = torch.sqrt(torch.sum((pva - Yva) ** 2, dim=1)).cpu().numpy()
            rmse = float(np.sqrt(np.mean(eva**2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_rmse


def main() -> None:
    data = load_dataset(None)
    d_hat = data["d_hat"].T.astype(np.float32)
    p = data["p"].T.astype(np.float32)
    N = d_hat.shape[0]
    rng = np.random.default_rng(42)
    idx = np.arange(N)
    rng.shuffle(idx)
    tr = idx[: int(0.8 * N)]
    va = idx[int(0.8 * N) :]

    Xtr = torch.from_numpy(d_hat[tr])
    Ytr = torch.from_numpy(p[tr])
    Xva = torch.from_numpy(d_hat[va])
    Yva = torch.from_numpy(p[va])
    BS = torch.from_numpy(data["bs"].astype(np.float32))

    configs = [
        dict(rel_hidden=32, res_hidden=64, dropout=0.10, lr=2e-3, wd=1e-4, batch=64, epochs=220, aux_wls=0.2, reg_dp=0.02),
        dict(rel_hidden=64, res_hidden=128, dropout=0.10, lr=1.5e-3, wd=1e-4, batch=64, epochs=260, aux_wls=0.2, reg_dp=0.02),
        dict(rel_hidden=64, res_hidden=128, dropout=0.05, lr=1e-3, wd=5e-5, batch=64, epochs=300, aux_wls=0.1, reg_dp=0.01),
        dict(rel_hidden=96, res_hidden=160, dropout=0.10, lr=1e-3, wd=1e-4, batch=64, epochs=320, aux_wls=0.15, reg_dp=0.01),
        dict(rel_hidden=64, res_hidden=128, dropout=0.20, lr=1.2e-3, wd=2e-4, batch=64, epochs=280, aux_wls=0.2, reg_dp=0.03),
    ]

    best = (None, 1e9)
    for i, cfg in enumerate(configs):
        r = eval_holdout(cfg, Xtr, Ytr, Xva, Yva, BS)
        print(f"cfg{i}: holdout_rmse={r:.3f} cfg={cfg}")
        if r < best[1]:
            best = (cfg, r)
    print(f"best_holdout_rmse={best[1]:.3f}")
    print(f"best_cfg={best[0]}")


if __name__ == "__main__":
    main()

