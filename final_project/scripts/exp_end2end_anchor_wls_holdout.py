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


def _err_stats(p_true: np.ndarray, p_pred: np.ndarray) -> tuple[float, float, float]:
    e = np.hypot(p_true[0] - p_pred[0], p_true[1] - p_pred[1])
    return float(np.sqrt(np.mean(e**2))), float(np.median(e)), float(np.percentile(e, 90))


class AnchorReliabilityWLS(nn.Module):
    def __init__(self, n_anchor: int = 18) -> None:
        super().__init__()
        self.n_anchor = n_anchor
        # anchor-specific distance calibration params
        self.a_raw = nn.Parameter(torch.zeros(n_anchor))
        self.b = nn.Parameter(torch.zeros(n_anchor))

        # reliability network per anchor
        self.rel_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.res_net = nn.Sequential(
            nn.Linear(n_anchor + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.lambda_ridge = 1e-3

    def forward(self, d_hat: torch.Tensor, bs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        d_hat: (B,18), bs: (2,18)
        returns: p_wls (B,2), p_final (B,2)
        """
        B = d_hat.shape[0]
        bx = bs[0]  # (18,)
        by = bs[1]

        scale = torch.nn.functional.softplus(self.a_raw).unsqueeze(0)  # (1,18)
        d_cal = scale * d_hat + self.b.unsqueeze(0)
        d_cal = torch.clamp(d_cal, min=1.0)

        # reliability features per anchor
        mn = d_cal.min(dim=1, keepdim=True).values
        mu = d_cal.mean(dim=1, keepdim=True)
        sd = d_cal.std(dim=1, keepdim=True) + 1e-6
        z = (d_cal - mu) / sd
        rel_feats = torch.stack(
            [
                d_cal,
                z,
                mn.repeat(1, self.n_anchor),
                mu.repeat(1, self.n_anchor),
            ],
            dim=2,
        )  # (B,18,4)
        w_anchor = torch.nn.functional.softplus(self.rel_net(rel_feats).squeeze(-1)) + 1e-4  # (B,18)

        # build pairwise linear system A p = b for each batch sample
        pairs_i = []
        pairs_j = []
        for i in range(self.n_anchor):
            for j in range(i + 1, self.n_anchor):
                pairs_i.append(i)
                pairs_j.append(j)
        ii = torch.tensor(pairs_i, device=d_hat.device, dtype=torch.long)
        jj = torch.tensor(pairs_j, device=d_hat.device, dtype=torch.long)
        M = ii.numel()  # 153

        xi, yi = bx[ii], by[ii]  # (M,)
        xj, yj = bx[jj], by[jj]
        A1 = 2.0 * (xj - xi)  # (M,)
        A2 = 2.0 * (yj - yi)

        di2 = d_cal[:, ii] ** 2  # (B,M)
        dj2 = d_cal[:, jj] ** 2
        rhs = di2 - dj2 - (xi**2 - xj**2).unsqueeze(0) - (yi**2 - yj**2).unsqueeze(0)  # (B,M)

        wp = torch.sqrt(w_anchor[:, ii] * w_anchor[:, jj])  # (B,M)
        Aw = torch.stack([A1.unsqueeze(0).repeat(B, 1), A2.unsqueeze(0).repeat(B, 1)], dim=2)  # (B,M,2)
        Aw = Aw * wp.unsqueeze(2)
        bw = rhs * wp

        # solve (A^T A + λI)p = A^T b
        At = Aw.transpose(1, 2)  # (B,2,M)
        AtA = torch.bmm(At, Aw)  # (B,2,2)
        Atb = torch.bmm(At, bw.unsqueeze(2)).squeeze(2)  # (B,2)
        eye = torch.eye(2, device=d_hat.device).unsqueeze(0).repeat(B, 1, 1)
        p_wls = torch.linalg.solve(AtA + self.lambda_ridge * eye, Atb.unsqueeze(2)).squeeze(2)  # (B,2)

        # residual correction
        feat_res = torch.cat([d_cal, p_wls], dim=1)  # (B,20)
        dp = self.res_net(feat_res)
        p_final = p_wls + dp
        return p_wls, p_final


def main() -> None:
    data = load_dataset(None)
    d_hat = data["d_hat"].T.astype(np.float32)  # (N,18)
    p = data["p"].T.astype(np.float32)  # (N,2)
    bs = data["bs"].astype(np.float32)  # (2,18)
    N = d_hat.shape[0]

    # fixed holdout split
    rng = np.random.default_rng(42)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_tr = int(N * 0.8)  # 560
    tr = idx[:n_tr]
    va = idx[n_tr:]

    Xtr = torch.from_numpy(d_hat[tr])
    Ytr = torch.from_numpy(p[tr])
    Xva = torch.from_numpy(d_hat[va])
    Yva = torch.from_numpy(p[va])
    BS = torch.from_numpy(bs)

    model = AnchorReliabilityWLS(n_anchor=18)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best_rmse = 1e9
    best_state = None

    bs_size = 64
    for ep in range(250):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], bs_size):
            bidx = perm[i : i + bs_size]
            xb = Xtr[bidx]
            yb = Ytr[bidx]
            p_wls, p_fin = model(xb, BS)
            e_fin = torch.sqrt(torch.sum((p_fin - yb) ** 2, dim=1))
            e_wls = torch.sqrt(torch.sum((p_wls - yb) ** 2, dim=1))
            # final + aux + regularize residual magnitude
            loss = (
                torch.mean(e_fin)
                + 0.2 * torch.mean(e_wls)
                + 0.02 * torch.mean(torch.sum((p_fin - p_wls) ** 2, dim=1))
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
            if (ep + 1) % 50 == 0:
                print(f"epoch={ep+1} holdout_rmse={rmse:.3f} best={best_rmse:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        _, pva = model(Xva, BS)
        _, ptr = model(Xtr, BS)
    pva_np = pva.cpu().numpy().T
    ptr_np = ptr.cpu().numpy().T
    Yva_np = Yva.cpu().numpy().T
    Ytr_np = Ytr.cpu().numpy().T
    tr_rmse, tr_med, tr_p90 = _err_stats(Ytr_np, ptr_np)
    va_rmse, va_med, va_p90 = _err_stats(Yva_np, pva_np)
    print(f"train rmse={tr_rmse:.3f} med={tr_med:.3f} p90={tr_p90:.3f}")
    print(f"holdout rmse={va_rmse:.3f} med={va_med:.3f} p90={va_p90:.3f}")


if __name__ == "__main__":
    main()

