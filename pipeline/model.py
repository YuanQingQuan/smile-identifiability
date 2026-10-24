# -*- coding: utf-8 -*-

import numpy as np
import torch
from atmosphere import VNIR_LINES, build_T_torch


_O2_IS_ANCHOR = np.array([True, False, True, False, False])


def _interp_matrix(ctrl, grid):

    ctrl = np.asarray(ctrl); P = np.zeros((len(grid), len(ctrl)))
    for i, x in enumerate(grid):
        j = np.clip(np.searchsorted(ctrl, x) - 1, 0, len(ctrl) - 2)
        t = (x - ctrl[j]) / (ctrl[j + 1] - ctrl[j])
        P[i, j] = 1 - t; P[i, j + 1] = t
    return P


class BLSPIC:
    def __init__(self, wl, fwhm, good_mask=None, lines=VNIR_LINES,
                 fine_step=0.5, ctrl_spacing=22.0, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wl = np.asarray(wl, float); self.fwhm = np.asarray(fwhm, float)
        self.B = len(self.wl)
        self.good = np.ones(self.B, bool) if good_mask is None else np.asarray(good_mask, bool)
        self.lines = lines
        lo, hi = self.wl.min() - 40, self.wl.max() + 40
        self.fine = np.arange(lo, hi + 1e-9, fine_step)
        self.ctrl = np.arange(lo, hi + 1e-9, ctrl_spacing)
        self.P = torch.tensor(_interp_matrix(self.ctrl, self.fine), dtype=torch.float32, device=self.device)
        self.Fm = torch.tensor(self.fine, dtype=torch.float32, device=self.device)
        self.wl_t = torch.tensor(self.wl, dtype=torch.float32, device=self.device)
        self.sig = torch.tensor(self.fwhm / 2.3548, dtype=torch.float32, device=self.device)
        self.gmask = torch.tensor(self.good, dtype=torch.bool, device=self.device)

        wl_good = self.wl[self.good]
        self.Pb2c = torch.tensor(_interp_matrix(wl_good, self.ctrl),
                                 dtype=torch.float32, device=self.device)
        self.good_idx = torch.tensor(np.where(self.good)[0], dtype=torch.long, device=self.device)

    def _normalize(self, M):
        m = (M * self.gmask).sum(1, keepdim=True) / self.gmask.sum()
        return M / (m + 1e-8)

    def _data_driven_basis(self, colmean, K):

        cm = np.asarray(colmean, float)
        cm_g = cm[:, self.good]

        cm_g = cm_g / (cm_g.mean(1, keepdims=True) + 1e-8)
        Rbar_band = cm_g.mean(0)
        Psi_band = np.zeros((K, cm_g.shape[1]))
        if K > 0:
            dcm = cm_g - Rbar_band[None, :]
            U, S, Vt = np.linalg.svd(dcm, full_matrices=False)
            for k in range(min(K, Vt.shape[0])):
                Psi_band[k] = Vt[k] * S[k] / max(cm_g.shape[0], 1)

        Rbar_ctrl = self.Pb2c.cpu().numpy() @ Rbar_band
        Psi_ctrl = np.stack([self.Pb2c.cpu().numpy() @ Psi_band[k] for k in range(K)], 0)
        return Rbar_ctrl.astype(np.float32), Psi_ctrl.astype(np.float32)

    def fit(self, colmean, iters=8000, lr_d=3e-3, lr_r=1e-2, lr_a=5e-3,
            lam_r_lam=3e-3, lam_r_x=3e-3, lam_d=1e-3, lam_atm=1e-2,
            learn_atm=True, shared_reflectance=False, verbose=False,
            low_rank=2, o2_only_anchor=True):

        X = colmean.shape[0]; Nc = len(self.ctrl)
        u = torch.linspace(-1, 1, X, device=self.device).unsqueeze(-1)
        Mt = torch.tensor(np.asarray(colmean, np.float32), device=self.device)
        ncm = self._normalize(Mt)

        dnet = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.Tanh(),
                                   torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                   torch.nn.Linear(64, 1)).to(self.device)
        torch.nn.init.zeros_(dnet[-1].weight); torch.nn.init.zeros_(dnet[-1].bias)


        mode = "shared" if shared_reflectance else ("lowrank" if low_rank > 0 else "full")
        if mode == "shared":
            Rbar = torch.nn.Parameter(0.5 * torch.ones(Nc, device=self.device))
            a_coefs = None; Psi_t = None
        elif mode == "lowrank":
            Rbar_np, Psi_np = self._data_driven_basis(colmean, low_rank)

            Rbar = torch.nn.Parameter(torch.tensor(0.3 + 0.2 * (Rbar_np - Rbar_np.min()) /
                                        max(Rbar_np.max() - Rbar_np.min(), 1e-6),
                                        device=self.device))

            Psi_t = torch.tensor(Psi_np, device=self.device)
            a_coefs = torch.nn.Parameter(0.05 * torch.randn(X, low_rank, device=self.device))
        else:
            Rbar = torch.nn.Parameter(0.5 * torch.ones(X, Nc, device=self.device))
            a_coefs = None; Psi_t = None

        s_atm = torch.nn.Parameter(torch.ones(len(self.lines), device=self.device))
        tilt = torch.nn.Parameter(torch.zeros((), device=self.device))


        anchor_w = torch.ones(len(self.lines), device=self.device)
        if o2_only_anchor and len(self.lines) == len(_O2_IS_ANCHOR):
            anchor_w = torch.tensor(_O2_IS_ANCHOR.astype(np.float32), device=self.device)
            anchor_w = torch.where(anchor_w > 0.5, torch.ones_like(anchor_w),
                                   torch.full_like(anchor_w, 0.2))

        band_w = torch.ones(self.B, device=self.device)
        if o2_only_anchor:


            band_w = torch.full((self.B,), 0.2, device=self.device)
            for k, (c, _d, w) in enumerate(self.lines):
                width = max(float(w) * 2.5, float(np.nanmedian(self.fwhm)))
                influence = torch.exp(-0.5 * ((self.wl_t - float(c)) / width) ** 2)
                band_w = torch.maximum(band_w, 0.2 + anchor_w[k] * influence)
            good_weight_mean = band_w[self.gmask].mean().clamp_min(1e-8)
            band_w = band_w / good_weight_mean

        groups = [{"params": dnet.parameters(), "lr": lr_d}, {"params": [Rbar], "lr": lr_r}]
        if a_coefs is not None:
            groups.append({"params": [a_coefs], "lr": lr_r})
        if learn_atm:
            groups.append({"params": [s_atm, tilt], "lr": lr_a})
        opt = torch.optim.Adam(groups)

        hist = []
        for it in range(iters):
            opt.zero_grad()
            d = dnet(u).squeeze(-1)
            T = build_T_torch(self.Fm, self.lines,
                              s_atm if learn_atm else torch.ones(len(self.lines), device=self.device),
                              tilt if learn_atm else torch.zeros((), device=self.device), self.device)
            if mode == "shared":
                Rfine = (self.P @ Rbar.unsqueeze(-1)).squeeze(-1)
                Rfine = Rfine.unsqueeze(0).expand(X, -1)
            elif mode == "lowrank":
                Rbar_f = (self.P @ Rbar).unsqueeze(0)
                Psi_f = self.P @ Psi_t.t()

                Rfine = Rbar_f + a_coefs @ Psi_f.t()
                Rfine = torch.clamp(Rfine, 1e-3, 2.0)
            else:
                Rfine = (self.P @ Rbar.t()).t()
            AE = Rfine * T[None, :]
            centers = self.wl_t[None, :] + d[:, None]
            z = (self.Fm[None, None, :] - centers[:, :, None]) / self.sig[None, :, None]
            G = torch.exp(-0.5 * z * z); G = G / (G.sum(-1, keepdim=True) + 1e-8)
            pred = torch.einsum("xf,xbf->xb", AE, G)
            pred = self._normalize(pred)
            res = (pred - ncm) * self.gmask

            loss = ((res ** 2) * band_w[None, :]).sum() / self.gmask.sum() / X

            loss = loss + lam_r_lam * ((Rfine[:, 2:] - 2 * Rfine[:, 1:-1] + Rfine[:, :-2]) ** 2).mean()
            if mode == "full":
                loss = loss + lam_r_x * ((Rbar[2:] - 2 * Rbar[1:-1] + Rbar[:-2]) ** 2).mean()
            elif mode == "lowrank":

                loss = loss + lam_r_x * ((a_coefs[2:] - 2 * a_coefs[1:-1] + a_coefs[:-2]) ** 2).mean()
            loss = loss + lam_d * ((d[2:] - 2 * d[1:-1] + d[:-2]) ** 2).mean()
            if learn_atm:
                loss = loss + lam_atm * ((s_atm - 1.0) ** 2).mean()
            loss.backward(); opt.step()
            if verbose and it % max(1, iters // 40) == 0:
                hist.append((it, float(loss)))
                print(f"  it {it:5d}  loss {float(loss):.3e}", flush=True)
        with torch.no_grad():
            d = dnet(u).squeeze(-1).cpu().numpy()
        R_out = Rbar.detach().cpu().numpy()
        return dict(dlam=d, depth_scale=s_atm.detach().cpu().numpy(),
                    tilt=float(tilt), reflectance=R_out,
                    a_coefs=(a_coefs.detach().cpu().numpy() if a_coefs is not None else None),
                    ctrl=self.ctrl, loss=float(loss), hist=hist,
                    band_weight=band_w.detach().cpu().numpy(),
                    line_anchor_weight=anchor_w.detach().cpu().numpy(),
                    shared_reflectance=shared_reflectance, mode=mode,
                    low_rank=low_rank, o2_only_anchor=o2_only_anchor)
