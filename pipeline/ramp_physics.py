# -*- coding: utf-8 -*-

import numpy as np


_LINE_CENTERS = np.array([688.0, 720.0, 760.0, 820.0, 940.0])
_LINE_SEARCH_HALFWIDTH_NM = 25.0


def _line_localization(cm, wl, line_center, search_hw=None):

    cm = np.asarray(cm, float)
    wl = np.asarray(wl, float)
    X, B = cm.shape
    hw = search_hw or _LINE_SEARCH_HALFWIDTH_NM
    near = np.abs(wl - line_center) <= hw
    if near.sum() < 3:
        return np.full(X, np.nan)
    wl_near = wl[near]

    lam_min = np.full(X, np.nan)
    for x in range(X):
        spec = cm[x, near]
        if not np.all(np.isfinite(spec)) or spec.max() <= 0:
            continue
        spec_n = spec / (np.nanmean(spec) + 1e-12)

        if len(spec_n) >= 4:
            coeffs = np.polyfit(np.arange(len(spec_n)), spec_n, 2)
            cont = np.polyval(coeffs, np.arange(len(spec_n)))
            depth = spec_n - cont
        else:
            depth = spec_n - 1.0
        i_min = int(np.nanargmin(depth))

        if 0 < i_min < len(depth) - 1:
            d0, d1, d2 = depth[i_min - 1], depth[i_min], depth[i_min + 1]
            denom = (d0 - 2 * d1 + d2)
            if abs(denom) > 1e-12:
                delta = 0.5 * (d0 - d2) / denom
                delta = np.clip(delta, -1, 1)
            else:
                delta = 0.0
        else:
            delta = 0.0
        idx_f = i_min + delta

        lam_min[x] = np.interp(idx_f, np.arange(len(wl_near)), wl_near)
    return lam_min


def line_localization_profile(cm, wl, fwhm=None):

    cm = np.asarray(cm, float)
    wl = np.asarray(wl, float)
    X = cm.shape[0]
    hw = _LINE_SEARCH_HALFWIDTH_NM
    if fwhm is not None:
        fwhm = np.asarray(fwhm, float)

    profiles = []
    for lc in _LINE_CENTERS:

        local_hw = hw
        if fwhm is not None:
            near_f = np.abs(wl - lc) <= hw
            if near_f.any():
                local_hw = max(hw, 2.0 * float(np.nanmedian(fwhm[near_f])))
        lam_min = _line_localization(cm, wl, lc, search_hw=local_hw)
        profiles.append(lam_min)

    lam_min_profiles = np.vstack(profiles)

    lam_shift = lam_min_profiles - np.nanmean(lam_min_profiles, axis=1, keepdims=True)


    n_lines = len(_LINE_CENTERS)
    sync_corrs = []
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            a, b = lam_shift[i], lam_shift[j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 5 and np.std(a[ok]) > 1e-9 and np.std(b[ok]) > 1e-9:
                sync_corrs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    sync_corr = float(np.nanmean(np.abs(sync_corrs))) if sync_corrs else float("nan")


    p2p_per_line = np.array([
        float(np.nanmax(lam_shift[i]) - np.nanmin(lam_shift[i]))
        for i in range(n_lines)
    ])

    p2p_spread = float(np.nanstd(p2p_per_line) / max(np.nanmean(p2p_per_line), 1e-9))

    return {
        "lines": _LINE_CENTERS,
        "lam_min_profiles": lam_min_profiles,
        "lam_shift_profiles": lam_shift,
        "sync_corr": sync_corr,
        "p2p_per_line": p2p_per_line,
        "p2p_mean": float(np.nanmean(p2p_per_line)),
        "p2p_spread_cv": p2p_spread,
        "n_lines_finite": int(np.sum(np.isfinite(p2p_per_line))),
    }


def diagnose_confound_physical(loc):

    sync = loc["sync_corr"]
    p2p_mean = loc["p2p_mean"]
    spread = loc["p2p_spread_cv"]
    reasons = []
    if p2p_mean < 0.5:
        cls = "gain_like"
        reasons.append(f"line_p2p_mean={p2p_mean:.2f}nm<0.5 (gain不移线)")
    elif sync > 0.6 and spread < 0.4:
        cls = "smile_like"
        reasons.append(f"sync_corr={sync:.2f}>0.6 spread={spread:.2f}<0.4 (各线同步)")
    else:
        cls = "ramp_like"
        reasons.append(f"sync_corr={sync:.2f} or spread={spread:.2f} (线间波长依赖)")
    return cls, reasons
