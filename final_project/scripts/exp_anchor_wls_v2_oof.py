"""Anchor-WLS v2: focal position loss + mixup + stronger regularization (5-fold OOF)."""
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

from lib.io_mat import load_dataset


class AnchorReliabilityWLS(nn.Module):
    def __init__(self, rel_hidden: int = 48, res_hidden: int = 96, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_anchor = 18
        self.a_raw = nn.Parameter(torch.zeros(18))
        self.b = nn.Parameter(torch.zeros(18))
        self.rel_net = nn.Sequential(
            nn.Linear(4, rel_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rel_hidden, rel_hidden),
            nn.ReLU(),
            nn.Linear(rel_hidden, 1),
        )
        self.res_net = nn.Sequential(
            nn.Linear(20, res_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, 2),
        )
        self.lambda_ridge = 2e-3

    def forward(self, d_hat: torch.Tensor, bs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = d_hat.shape[0]
        bx, by = bs[0], bs[1]
        scale = F.softplus(self.a_raw).unsqueeze(0)
        d_cal = torch.clamp(scale * d_hat + self.b.unsqueeze(0), min=1.0)
        mn = d_cal.min(dim=1, keepdim=True).values
        mu = d_cal.mean(dim=1, keepdim=True)
        sd = d_cal.std(dim=1, keepdim=True) + 1e-6
        z = (d_cal - mu) / sd
        feats = torch.stack([d_cal, z, mn.repeat(1, 18), mu.repeat(1, 18)], dim=2)
        w_anchor = F.softplus(self.rel_net(feats).squeeze(-1)) + 1e-3

        pairs_i, pairs_j = [], []
        for i in range(18):
            for j in range(i + 1, 18):
                pairs_i.append(i)
                pairs_j.append(j)
        ii = torch.tensor(pairs_i, dtype=torch.long, device=d_hat.device)
        jj = torch.tensor(pairs_j, dtype=torch.long, device=d_hat.device)
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
        return p_wls, p_wls + dp


def focal_l1(pred: torch.Tensor, target: torch.Tensor, gamma: float = 1.5) -> torch.Tensor:
    diff = (pred - target).abs().sum(dim=1)
    w = (1.0 + diff) ** gamma
    return (w * diff).mean()


def mixup_xy(y: torch.Tensor, alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lam = torch.distributions.Beta(alpha, alpha).sample((y.shape[0], 1)).to(y.device)
    idx = torch.randperm(y.shape[0], device=y.device)
    y2 = y[idx]
    y_mix = lam * y + (1 - lam) * y2
    return y, y_mix, lam.squeeze(1)


def train_fold(Xtr, Ytr, Xva, Yva, BS, seed: int) -> tuple[float, dict]:
    torch.manual_seed(seed)
    model = AnchorReliabilityWLS(rel_hidden=48, res_hidden=96, dropout=0.2).to(Xtr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=5e-4)
    best_rmse = 1e9
    best_state = None
    for ep in range(180):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], 48):
            idx = perm[i : i + 48]
            xb, yb = Xtr[idx], Ytr[idx]
            y_a, y_b, lam = mixup_xy(yb, alpha=0.25)
            _, p_a = model(xb, BS)
            _, p_b = model(xb, BS)
            p_mix = lam.unsqueeze(1) * p_a + (1 - lam.unsqueeze(1)) * p_b
            loss = focal_l1(p_mix, y_a, gamma=1.2) + focal_l1(p_mix, y_b, gamma=1.2)
            loss = loss + 0.02 * (p_mix - model(xb, BS)[1]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, pva = model(Xva, BS)
                eva = torch.sqrt(torch.sum((pva - Yva) ** 2, dim=1)).cpu().numpy()
            rm = float(np.sqrt(np.mean(eva**2)))
            if rm < best_rmse:
                best_rmse = rm
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_rmse, best_state or {}


def main() -> None:
    data = load_dataset(None)
    X = data["d_hat"].T.astype(np.float32)
    Y = data["p"].T.astype(np.float32)
    BS = torch.from_numpy(data["bs"].astype(np.float32))
    n = X.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((n, 2), dtype=np.float32)
    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        Xtr = torch.from_numpy(X[tr]).to(BS.device)
        Ytr = torch.from_numpy(Y[tr]).to(BS.device)
        Xva = torch.from_numpy(X[va]).to(BS.device)
        Yva = torch.from_numpy(Y[va]).to(BS.device)
        rmse, state = train_fold(Xtr, Ytr, Xva, Yva, BS, seed=200 + fold)
        if state:
            m = AnchorReliabilityWLS().to(BS.device)
            m.load_state_dict(state)
            m.eval()
            with torch.no_grad():
                _, pva = m(Xva, BS)
            oof[va] = pva.cpu().numpy()
        e = np.sqrt(np.sum((oof[va] - Y[va]) ** 2, axis=1))
        print(f"fold={fold} rmse={float(np.sqrt(np.mean(e**2))):.3f}")

    pred = oof.T
    e = np.hypot(pred[0] - data["p"][0], pred[1] - data["p"][1])
    print(f"anchor_wls_v2_oof rmse={float(np.sqrt(np.mean(e**2))):.3f} median={float(np.median(e)):.3f} p90={float(np.percentile(e,90)):.3f}")


if __name__ == "__main__":
    main()
