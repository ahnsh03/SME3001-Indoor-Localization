"""5-fold OOF: fold-wise v30 distance calib + anchor reliability WLS (no extra distance scale)."""
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

from lib.calib import fit_calib
from lib.io_mat import load_dataset
from lib.pipeline import VERSION_REGISTRY


class AnchorRelWLSOnly(nn.Module):
    def __init__(self, n_anchor: int = 18, rel_hidden: int = 64, res_hidden: int = 192, dropout: float = 0.15):
        super().__init__()
        self.n_anchor = n_anchor
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

    def forward(self, d_cal: torch.Tensor, bs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = d_cal.shape[0]
        bx, by = bs[0], bs[1]
        d_cal = torch.clamp(d_cal, min=1.0)
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
        ii = torch.tensor(pairs_i, device=d_cal.device, dtype=torch.long)
        jj = torch.tensor(pairs_j, device=d_cal.device, dtype=torch.long)
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
        eye = torch.eye(2, device=d_cal.device).unsqueeze(0).repeat(B, 1, 1)
        p_wls = torch.linalg.solve(AtA + self.lambda_ridge * eye, Atb.unsqueeze(2)).squeeze(2)
        dp = self.res_net(torch.cat([d_cal, p_wls], dim=1))
        return p_wls, p_wls + dp


def train_fold(Xtr, Ytr, Xva, Yva, BS, seed: int) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    model = AnchorRelWLSOnly(rel_hidden=64, res_hidden=192, dropout=0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    best_rmse = 1e9
    best_state = None
    aux_wls, reg_dp, batch, epochs = 0.15, 0.005, 48, 240

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], batch):
            idx = perm[i : i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            p_wls, p_fin = model(xb, BS)
            e_fin = torch.sqrt(torch.sum((p_fin - yb) ** 2, dim=1))
            e_wls = torch.sqrt(torch.sum((p_wls - yb) ** 2, dim=1))
            loss = torch.mean(e_fin) + aux_wls * torch.mean(e_wls) + reg_dp * torch.mean(
                torch.sum((p_fin - p_wls) ** 2, dim=1)
            )
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

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, pva = model(Xva, BS)
    return pva.cpu().numpy(), best_rmse, best_state


def main() -> None:
    data = load_dataset(None)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    cfg = VERSION_REGISTRY["v30"]
    n = d_hat.shape[1]
    BS = torch.from_numpy(bs.astype(np.float32))
    Y = p.T.astype(np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((n, 2), dtype=np.float32)

    for fold, (tr, va) in enumerate(kf.split(np.arange(n))):
        tr = np.asarray(tr, dtype=np.int64)
        va = np.asarray(va, dtype=np.int64)
        calib = fit_calib(
            d_hat, p, bs, tr, cfg.calib,
            quantile_tau=cfg.quantile_tau,
            mlp_noise_std=cfg.mlp_noise_std,
        )
        Xtr = torch.from_numpy(np.stack([calib.apply(d_hat[:, u]) for u in tr]).astype(np.float32))
        Xva = torch.from_numpy(np.stack([calib.apply(d_hat[:, u]) for u in va]).astype(np.float32))
        Ytr = torch.from_numpy(Y[tr])
        Yva = torch.from_numpy(Y[va])
        pred, best_hold, _ = train_fold(Xtr, Ytr, Xva, Yva, BS, seed=300 + fold)
        oof[va] = pred
        e = np.sqrt(np.sum((pred - Y[va]) ** 2, axis=1))
        print(f"fold={fold} best_hold={best_hold:.3f} rmse={float(np.sqrt(np.mean(e**2))):.3f}")

    pred_t = oof.T
    e = np.hypot(p[0] - pred_t[0], p[1] - pred_t[1])
    print(
        f"anchor_v30calib_oof rmse={float(np.sqrt(np.mean(e**2))):.3f} "
        f"median={float(np.median(e)):.3f} p90={float(np.percentile(e, 90)):.3f}"
    )


if __name__ == "__main__":
    main()
