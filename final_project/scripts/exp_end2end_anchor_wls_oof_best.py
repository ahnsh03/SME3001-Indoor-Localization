from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.io_mat import load_dataset


class AnchorReliabilityWLS(nn.Module):
    def __init__(self, n_anchor: int = 18, rel_hidden: int = 64, res_hidden: int = 192, dropout: float = 0.15):
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


def rmse_from_xy(p_true_2n: np.ndarray, p_pred_2n: np.ndarray) -> float:
    e = np.hypot(p_true_2n[0] - p_pred_2n[0], p_true_2n[1] - p_pred_2n[1])
    return float(np.sqrt(np.mean(e**2)))


def train_one_fold(Xtr, Ytr, Xva, Yva, BS, seed: int) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    model = AnchorReliabilityWLS(rel_hidden=64, res_hidden=192, dropout=0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    best_rmse = 1e9
    best_state = None
    aux_wls = 0.15
    reg_dp = 0.005
    batch = 48
    epochs = 220  # shortened from 360 for runtime

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], batch):
            idx = perm[i : i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            p_wls, p_fin = model(xb, BS)
            e_fin = torch.sqrt(torch.sum((p_fin - yb) ** 2, dim=1))
            e_wls = torch.sqrt(torch.sum((p_wls - yb) ** 2, dim=1))
            loss = torch.mean(e_fin) + aux_wls * torch.mean(e_wls) + reg_dp * torch.mean(torch.sum((p_fin - p_wls) ** 2, dim=1))
            opt.zero_grad()
            loss.backward()
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

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, pva = model(Xva, BS)
    pred = pva.cpu().numpy()  # (n_va,2)
    return pred, best_rmse


def main() -> None:
    data = load_dataset(None)
    X = data["d_hat"].T.astype(np.float32)  # (N,18)
    Y = data["p"].T.astype(np.float32)  # (N,2)
    BS = torch.from_numpy(data["bs"].astype(np.float32))
    n = X.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((n, 2), dtype=np.float32)

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        Xtr = torch.from_numpy(X[tr])
        Ytr = torch.from_numpy(Y[tr])
        Xva = torch.from_numpy(X[va])
        Yva = torch.from_numpy(Y[va])
        pred, best_hold = train_one_fold(Xtr, Ytr, Xva, Yva, BS, seed=100 + fold)
        oof[va] = pred
        e = np.sqrt(np.sum((pred - Y[va]) ** 2, axis=1))
        print(f"fold={fold} best_hold={best_hold:.3f} rmse={float(np.sqrt(np.mean(e**2))):.3f}")

    p_true = Y.T
    p_pred = oof.T
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    print(f"OOF rmse={float(np.sqrt(np.mean(e**2))):.3f} median={float(np.median(e)):.3f} p90={float(np.percentile(e,90)):.3f}")


if __name__ == "__main__":
    main()

