# -*- coding: utf-8 -*-

import numpy as np


_ABSORPTION_LINES = [688.0, 720.0, 760.0, 820.0, 940.0]
_LINE_HALFWIDTH_NM = 25.0


def _band_masks(wl, fwhm=None):

    wl = np.asarray(wl, float)
    if fwhm is not None:
        fwhm = np.asarray(fwhm, float)
    line_mask = np.zeros(len(wl), bool)
    for lc in _ABSORPTION_LINES:
        for i in range(len(wl)):
            hw = max(_LINE_HALFWIDTH_NM, 2.5 * (fwhm[i] if fwhm is not None else 10.0))
            if abs(wl[i] - lc) <= hw:
                line_mask[i] = True
    cont_mask = ~line_mask
    return line_mask, cont_mask


def decompose_ramp_gain(cm, wl, fwhm=None):

    cm = np.asarray(cm, float)
    wl = np.asarray(wl, float)
    X, B = cm.shape
    line_mask, cont_mask = _band_masks(wl, fwhm)


    cm_norm = cm / (np.nanmean(cm, axis=1, keepdims=True) + 1e-12)

    r = cm_norm - np.nanmedian(cm_norm, axis=0, keepdims=True)


    U, S, Vt = np.linalg.svd(r, full_matrices=False)
    s1 = S[0] if S.size > 0 else 1e-12
    sv_ratio = float(S[1] / s1) if S.size > 1 and s1 > 1e-12 else 0.0
    sv_ratio_3 = float(S[2] / s1) if S.size > 2 and s1 > 1e-12 else 0.0

    total_var = float((S ** 2).sum()) if S.size else 1e-12
    rank1_fraction = float(S[0] ** 2 / total_var) if total_var > 1e-12 else 0.0


    gain_profile = U[:, 0] * S[0] if U.shape[0] == X else np.zeros(X)
    gain_profile = gain_profile - np.nanmean(gain_profile)


    if S.size >= 2:
        ramp_part = (U[:, 1:] * S[1:]) @ Vt[1:]
        ramp_profile = np.nanmedian(ramp_part, axis=1)
        ramp_profile = ramp_profile - np.nanmean(ramp_profile)
    else:
        ramp_profile = np.zeros(X)

    gain_energy = float(np.nanmax(gain_profile) - np.nanmin(gain_profile))
    ramp_energy = float(np.nanmax(ramp_profile) - np.nanmin(ramp_profile)) if np.std(ramp_profile) > 1e-12 else 0.0


    if line_mask.sum() >= 3 and cont_mask.sum() >= 3:
        e_line = float(np.nanmean(r[:, line_mask] ** 2))
        e_cont = float(np.nanmean(r[:, cont_mask] ** 2))
        line_contour_ratio = e_line / max(e_cont, 1e-12)
        cl = np.nanmedian(r[:, line_mask], axis=1)
        cc = np.nanmedian(r[:, cont_mask], axis=1)
        cl, cc = cl - cl.mean(), cc - cc.mean()
        if np.std(cl) > 1e-12 and np.std(cc) > 1e-12:
            line_contour_corr = float(np.corrcoef(cl, cc)[0, 1])
        else:
            line_contour_corr = 0.0
    else:
        line_contour_corr = 0.0
        line_contour_ratio = 1.0

    return {
        "gain_profile": gain_profile,
        "ramp_profile": ramp_profile,
        "gain_energy": gain_energy,
        "ramp_energy": ramp_energy,
        "sv_ratio_2": sv_ratio,
        "sv_ratio_3": sv_ratio_3,
        "rank1_fraction": rank1_fraction,
        "line_contour_corr": line_contour_corr,
        "line_contour_ratio": line_contour_ratio,
        "n_line_bands": int(line_mask.sum()),
        "n_cont_bands": int(cont_mask.sum()),
        "singular_values": S[:6].tolist(),
    }


def gain_striping_score(decomp):

    rank1 = decomp["rank1_fraction"]
    sv2 = decomp["sv_ratio_2"]

    rank1_term = min(max(0.0, rank1 - 0.5) / 0.4, 1.0)
    sv2_term = max(0.0, 1.0 - sv2 / 0.15)
    score = 0.6 * rank1_term + 0.4 * sv2_term
    return float(np.clip(score, 0.0, 1.0))


def ramp_score(decomp):

    sv2 = decomp["sv_ratio_2"]
    sv3 = decomp.get("sv_ratio_3", 0.0)
    rank1 = decomp["rank1_fraction"]

    sv2_term = min(max(0.0, sv2 - 0.04) / 0.12, 1.0)
    rank1_term = max(0.0, 1.0 - max(0.0, rank1 - 0.8) / 0.15)
    score = 0.6 * sv2_term + 0.4 * rank1_term
    return float(np.clip(score, 0.0, 1.0))


def classify_confound(decomp, gain_thresh=0.55, ramp_thresh=0.55):

    gs = gain_striping_score(decomp)
    rs = ramp_score(decomp)
    if gs >= gain_thresh and gs > rs:
        return "gain_dominant", gs, rs
    if rs >= ramp_thresh and rs > gs:
        return "ramp_dominant", gs, rs
    return "mixed", gs, rs


def ramp_identifiability_bound(decomp, fwhm_nm, snr):

    o2a_fwhm = 2.3548 * 1.8
    eff_fwhm = float(np.sqrt(o2a_fwhm ** 2 + fwhm_nm ** 2))
    base = 3.5 * eff_fwhm / max(np.sqrt(max(snr, 30.0)), 1.0)

    sv2 = decomp.get("sv_ratio_2", 0.0)
    rank1 = decomp.get("rank1_fraction", 1.0)

    ramp_factor = 1.0 + max(0.0, sv2 - 0.04) / 0.04
    ramp_factor = min(ramp_factor, 3.0)
    return float(base * ramp_factor)
