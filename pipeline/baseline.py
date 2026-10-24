# -*- coding: utf-8 -*-

import numpy as np


def _poly_detrend(y, deg=4):
    x = np.linspace(-1, 1, len(y)); V = np.vander(x, deg + 1)
    c, *_ = np.linalg.lstsq(V, y, rcond=None); return y - V @ c


def phasecorr_shift(colmean, wl, good_mask=None, max_lag=3):

    X, B = colmean.shape
    good = np.ones(B, bool) if good_mask is None else np.asarray(good_mask, bool)
    band_step = np.median(np.diff(wl))

    feats = np.zeros((X, B))
    for x in range(X):
        s = colmean[x].copy(); s = s / (s[good].mean() + 1e-8)
        feats[x] = _poly_detrend(s)
    feats[:, ~good] = 0.0
    ref = feats.mean(0)
    shifts = np.zeros(X)
    lags = np.arange(-max_lag, max_lag + 1)

    def overlap_corr(a, b, lag):
        if lag < 0:
            aa, bb = a[-lag:], b[:lag]
        elif lag > 0:
            aa, bb = a[:-lag], b[lag:]
        else:
            aa, bb = a, b
        if len(aa) < 4:
            return -np.inf
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        den = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)) + 1e-12
        return float(np.sum(aa * bb) / den)

    for x in range(X):
        cc = np.array([overlap_corr(feats[x], ref, int(L)) for L in lags])
        k = int(np.argmax(cc))
        if 0 < k < len(lags) - 1:
            d = 0.5 * (cc[k - 1] - cc[k + 1]) / (cc[k - 1] - 2 * cc[k] + cc[k + 1] + 1e-12)
        else:
            d = 0.0
        shifts[x] = (lags[k] + d) * band_step
    return shifts - shifts.mean()
