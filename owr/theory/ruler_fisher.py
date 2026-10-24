# -*- coding: utf-8 -*-

import numpy as np


_SELLMEIER = dict(B1=0.6961663, B2=0.4079426, B3=0.8974794,
                  C1=0.00467914826, C2=0.0135120631, C3=97.9340025)


def index_fused_silica(lam_nm):

    um = np.asarray(lam_nm, float) / 1000.0
    B, C = _SELLMEIER, _SELLMEIER
    n2 = (1.0
          + B["B1"] * um ** 2 / (um ** 2 - C["C1"])
          + B["B2"] * um ** 2 / (um ** 2 - C["C2"])
          + B["B3"] * um ** 2 / (um ** 2 - C["C3"]))
    return np.sqrt(n2)


def group_index_fused_silica(lam_nm):

    um = np.asarray(lam_nm, float) / 1000.0
    h = 1e-5
    n0 = index_fused_silica(um * 1000.0)
    np_ = index_fused_silica((um + h) * 1000.0)
    nm_ = index_fused_silica((um - h) * 1000.0)
    dn = (np_ - nm_) / (2 * h)
    return n0 - um * dn


def reflectivity_from_contrast(c):

    c = float(np.clip(c, 1e-6, 0.9))
    m = 2.0 * c / (2.0 - c)
    R = 1.0 - 2.0 / (np.sqrt(1.0 + m) + 1.0)
    return float(R)


def etalon_transmission(lam_nm, L_nm, R, n=None, theta=0.0):

    lam_nm = np.asarray(lam_nm, float)
    if n is None:
        n = index_fused_silica(lam_nm)
    n = np.asarray(n, float)
    delta = 4.0 * np.pi * n * L_nm * np.cos(theta) / lam_nm
    m = 4.0 * R / (1.0 - R) ** 2
    return 1.0 / (1.0 + m * np.sin(delta / 2.0) ** 2)


def fringe_phase(lam_nm, L_nm, n=None):

    lam_nm = np.asarray(lam_nm, float)
    if n is None:
        n = index_fused_silica(lam_nm)
    return 4.0 * np.pi * np.asarray(n, float) * L_nm / lam_nm


def fringe_period(lam_nm, L_nm, n=None, group=False):

    lam_nm = np.asarray(lam_nm, float)
    if n is None:
        n = group_index_fused_silica(lam_nm) if group else index_fused_silica(lam_nm)
    return lam_nm ** 2 / (2.0 * np.asarray(n, float) * L_nm)


def srf_attenuation(fwhm_srf_nm, period_nm):

    sigma = fwhm_srf_nm / 2.3548
    return float(np.exp(-2.0 * np.pi ** 2 * sigma ** 2 / period_nm ** 2))


def closed_form_sigma_lambda(fwhm_srf_nm, contrast, snr, n_bands, period_nm=None,
                             sigma_srf_nm=None):

    sigma_srf = fwhm_srf_nm / 2.3548 if fwhm_srf_nm else 0.0
    if period_nm is None:
        period_nm = 2.0 * np.pi * max(sigma_srf, 1e-6)
    A = srf_attenuation(fwhm_srf_nm, period_nm) if fwhm_srf_nm else 1.0
    c_eff = contrast * A
    snr_eff = float(np.mean(snr))
    return float(np.sqrt(2.0) * period_nm / (2.0 * np.pi * c_eff * snr_eff * np.sqrt(n_bands)))


def optimal_fsr(fwhm_srf_nm):

    sigma_srf = fwhm_srf_nm / 2.3548
    return 2.0 * np.pi * sigma_srf


def sigma_lambda_min(fwhm_srf_nm, contrast, snr, n_bands):

    sigma_srf = fwhm_srf_nm / 2.3548
    snr_eff = float(np.mean(snr))
    return float(np.sqrt(2.0 * np.e) * sigma_srf / (contrast * snr_eff * np.sqrt(n_bands)))


def numerical_fisher_phase(lam_b, radiance_b, snr_b, L_nm, contrast,
                           fwhm_srf_nm=None, n=None):

    lam_b = np.asarray(lam_b, float)
    S = np.asarray(radiance_b, float)
    snr = np.asarray(snr_b, float)
    if n is None:
        n = index_fused_silica(lam_b)
        dn_disc = True
    else:
        n = np.broadcast_to(np.asarray(n, float), lam_b.shape).copy()
        dn_disc = False
    beta_b = fringe_phase(lam_b, L_nm, n=n)
    h = 1e-4
    if dn_disc:
        db = (fringe_phase(lam_b + h, L_nm, n=index_fused_silica(lam_b + h))
              - fringe_phase(lam_b - h, L_nm, n=index_fused_silica(lam_b - h))) / (2 * h)
    else:
        db = -4.0 * np.pi * n * L_nm / lam_b ** 2
    period_g = 2.0 * np.pi / np.abs(db)
    if fwhm_srf_nm is None:
        c_eff = np.full(lam_b.shape, float(contrast))
    else:
        c_eff = contrast * np.array([srf_attenuation(fwhm_srf_nm, p) for p in period_g])
    I_beta_band = (c_eff * snr) ** 2 * 0.5
    I_phase = float(np.sum(I_beta_band))
    I_lambda = float(np.sum(I_beta_band * db ** 2))
    sigma_lambda_band = 1.0 / np.sqrt(np.maximum(I_beta_band * db ** 2, 1e-30))
    return {
        "I_lambda": I_lambda,
        "sigma_lambda_nm": float(1.0 / np.sqrt(max(I_lambda, 1e-30))),
        "I_phase": I_phase,
        "sigma_phase_rad": float(1.0 / np.sqrt(max(I_phase, 1e-30))),
        "dbeta_dlambda_per_nm": db,
        "period_group_nm": period_g,
        "c_eff_band": c_eff,
        "sigma_lambda_band_nm": sigma_lambda_band,
    }


def closed_form_sigma_lambda_broadband(lam_b, contrast, snr, L_nm,
                                       fwhm_srf_nm=None, n=None):

    lam_b = np.asarray(lam_b, float)
    snr = np.full(lam_b.shape, float(snr)) if np.isscalar(snr) else np.asarray(snr, float)
    P_g = fringe_period(lam_b, L_nm, n=n, group=(n is None))
    A = (np.array([srf_attenuation(fwhm_srf_nm, p) for p in P_g])
         if fwhm_srf_nm else np.ones_like(P_g))
    terms = (contrast * A * snr * 2.0 * np.pi / P_g) ** 2
    return float(np.sqrt(2.0) / np.sqrt(np.sum(terms)))


def _fringe_templates(lam_b, L_nm, n=None):

    beta = fringe_phase(lam_b, L_nm, n=n)
    t = np.stack([np.cos(beta), np.sin(beta)], 0)
    t = t - t.mean(1, keepdims=True)
    return t


def confound_immunity(lam_b, snr_b, L_nm, contrast, fwhm_srf_nm=None, n=None,
                      nuisance="gain", n_poly=4, line_list=None):

    lam_b = np.asarray(lam_b, float)
    snr = np.asarray(snr_b, float)
    if n is None:
        n = index_fused_silica(lam_b)
    tc, ts = _fringe_templates(lam_b, L_nm, n=n)
    dbeta = np.gradient(fringe_phase(lam_b, L_nm, n=n), lam_b)
    if fwhm_srf_nm is not None:
        A = np.array([srf_attenuation(fwhm_srf_nm, 2 * np.pi / abs(d)) for d in dbeta])
        tc, ts = tc * A, ts * A
    G = np.stack([tc, ts], 0)
    W = snr ** 2

    def wip(a, b):
        return float(np.sum(W * a * b))


    B_range = lam_b.max() - lam_b.min()
    u = (lam_b - lam_b.mean()) / (B_range / 2.0)
    nus = []
    if nuisance == "gain":
        nus.append(np.ones_like(lam_b))
    elif nuisance == "ramp":
        for k in range(1, n_poly + 1):
            nus.append(u ** k)
    elif nuisance == "lines":
        if line_list is None:
            line_list = [(760.0, 1.8), (688.0, 2.0), (720.0, 6.0), (820.0, 8.0), (940.0, 15.0)]
        for c0, sig in line_list:
            dT = -(lam_b - c0) / sig ** 2 * np.exp(-0.5 * ((lam_b - c0) / sig) ** 2)
            dT = dT / (np.sqrt(np.sum(W * dT ** 2)) + 1e-30)
            nus.append(dT)
    else:
        raise ValueError(nuisance)
    Nm = np.stack([v - np.sum(W * v) / np.sum(W) for v in nus], 0)


    cross = G @ (W * Nm).T
    bias_rad = float(np.linalg.norm(cross, ord=2) / max(wip(G[0], G[0]) + wip(G[1], G[1]), 1e-30) * np.sqrt(2.0))
    dbeta_w = float(np.sum(W * np.abs(dbeta)) / np.sum(W))
    bias_nm = bias_rad / max(abs(dbeta_w), 1e-30)

    proj = []
    g_norm = np.sqrt(wip(G[0], G[0]) + wip(G[1], G[1]))
    for i in range(Nm.shape[0]):
        v = Nm[i]
        p = np.linalg.norm(cross[:, i]) / max(g_norm * np.sqrt(wip(v, v)), 1e-30)
        proj.append(float(p))


    K = 2 + Nm.shape[0]
    F = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            a = G[i] if i < 2 else Nm[i - 2]
            b = G[j] if j < 2 else Nm[j - 2]
            F[i, j] = wip(a, b)
    F *= contrast ** 2
    F_BB, F_BN, F_NN = F[:2, :2], F[:2, 2:], F[2:, 2:]
    if F_NN.size:
        F_NN_inv = np.linalg.pinv(F_NN, rcond=1e-10)
        I_eff = float(np.linalg.eigvalsh(F_BB - F_BN @ F_NN_inv @ F_BN.T).min())
    else:
        I_eff = float(np.linalg.eigvalsh(F_BB).min())
    I_main = float(np.linalg.eigvalsh(F_BB).min())
    schur = float(np.sqrt(I_main / max(I_eff, 1e-30)))
    return dict(
        projection_coeffs=proj,
        bias_rad_per_unit=bias_rad,
        bias_nm_per_unit=bias_nm,
        dbeta_dlambda_per_nm=dbeta_w,
        immunity_factor_schur=schur,
    )


def design_spec(sigma_target_nm, fwhm_srf_nm, snr, n_bands):

    sigma_srf = fwhm_srf_nm / 2.3548
    return float(np.sqrt(2.0 * np.e) * sigma_srf / (sigma_target_nm * snr * np.sqrt(n_bands)))
