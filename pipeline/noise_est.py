# -*- coding: utf-8 -*-

import numpy as np

_EPS = 1e-12
_MIN_SIGMA = 1e-3


def build_homogeneous_mask(cube_brc, frac_keep=0.12, k=9):

    cube = np.asarray(cube_brc, float)
    if cube.ndim != 3:
        raise ValueError("期望 [bands,rows,cols]")
    img = np.nanmean(cube, axis=0)
    R, C = img.shape
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    box = np.ones((k, k)) / (k * k)

    from numpy.fft import fft2, ifft2
    lm = np.real(ifft2(fft2(padded) * fft2(box, s=padded.shape)))[pad:pad + R, pad:pad + C]
    lv = np.real(ifft2(fft2((img - lm) ** 2) * fft2(box, s=img.shape)))[:R, :C]
    if lv.shape != img.shape:
        from numpy.pad import pad as npad
        lv = npad(lv, ((0, R - lv.shape[0]), (0, C - lv.shape[1])), mode="edge")[:R, :C]

    gy, gx = np.gradient(img.astype(float))
    edge = np.hypot(gx, gy)

    def _n01(a):
        p1, p99 = np.nanpercentile(a, (1, 99))
        return np.clip((a - p1) / (p99 - p1 + _EPS), 0, 1)

    score = 0.5 * _n01(lv) + 0.5 * _n01(edge)
    thr = np.quantile(score, frac_keep)
    return score <= thr


def _mad(x):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    return float(np.median(np.abs(a - np.median(a))) / 0.6745)


def snr_columnwise_diff(cube_brc, diff_axis=1, mask=None, min_valid=16):

    cube = np.asarray(cube_brc, float)
    if cube.ndim != 3:
        raise ValueError("期望 [bands,rows,cols]")
    B, R, C = cube.shape
    if R < 4:
        return float("nan")
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != (R, C):
            raise ValueError(f"mask 形状 {mask.shape} != ({R},{C})")

        snrs = []
        for b in range(B):
            for c in range(C):
                col_mask = mask[:, c]
                if col_mask.sum() < min_valid:
                    continue
                col = cube[b, col_mask, c]
                mu = np.nanmean(col)
                d = np.diff(col)
                sigma = np.nanstd(d) / np.sqrt(2.0)
                if sigma < _MIN_SIGMA:
                    continue
                snrs.append(mu / sigma)
        snrs = np.array(snrs)
        snrs = snrs[np.isfinite(snrs) & (snrs > 0)]
        return float(np.nanmedian(snrs)) if snrs.size else float("nan")

    diff = np.diff(cube, axis=diff_axis)
    sigma_bc = np.nanstd(diff, axis=1) / np.sqrt(2.0)
    mu_bc = np.nanmean(cube, axis=1)
    sigma_bc = np.where(sigma_bc < _MIN_SIGMA, _MIN_SIGMA, sigma_bc)
    snr_bc = mu_bc / sigma_bc
    snr_bc = snr_bc[np.isfinite(snr_bc) & (snr_bc > 0)]
    return float(np.nanmedian(snr_bc)) if snr_bc.size else float("nan")


def pca_decompose(cube_brc, k=4):

    cube = np.asarray(cube_brc, float)
    if cube.ndim != 3:
        raise ValueError(f"期望 [bands,rows,cols], got {cube.shape}")
    B, R, C = cube.shape
    flat = cube.reshape(B, -1).T
    finite = np.all(np.isfinite(flat), axis=1)
    X = flat[finite]
    if X.shape[0] < max(3, k):
        X = flat[np.isfinite(flat).any(1)]
    if X.shape[0] < 2:
        mean_spec = np.nanmean(flat, 0)
        recon = np.broadcast_to(mean_spec, cube.shape).copy()
        return recon, cube - recon, {"k": 0, "n_pixels": int(X.shape[0])}
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    kk = int(min(k, Vt.shape[0]))
    recon_X = (Xc @ Vt[:kk].T) @ Vt[:kk] + mu
    recon_flat = np.full_like(flat, mu[None, :])
    recon_flat[finite] = recon_X
    recon_flat[~finite] = mu[None, :]
    recon = recon_flat.T.reshape(B, R, C)
    residual = cube - recon
    evr = (S ** 2 / (S ** 2).sum())
    return recon, residual, {
        "k": kk, "n_pixels": int(X.shape[0]),
        "explained_variance_ratio": evr[:kk].tolist(),
        "cumulative_variance": float(evr[:kk].sum()),
    }


def snr_pca_residual(cube_brc, k=4, diff_axis=1, mask=None, min_valid=16):

    cube = np.asarray(cube_brc, float)
    if cube.ndim != 3:
        raise ValueError("期望 [bands,rows,cols]")
    B, R, C = cube.shape
    recon, residual, model = pca_decompose(cube, k=k)

    if mask is not None:
        mask = np.asarray(mask, bool)
        snrs = []
        for b in range(B):
            for c in range(C):
                cm_col = mask[:, c]
                if cm_col.sum() < min_valid:
                    continue
                mu = float(np.nanmean(cube[b, cm_col, c]))
                rc = residual[b, cm_col, c]
                sigma = float(np.nanstd(np.diff(rc)) / np.sqrt(2.0))
                if sigma < _MIN_SIGMA or not np.isfinite(mu):
                    continue
                snrs.append(mu / sigma)
        snrs = np.array(snrs)
        snrs = snrs[np.isfinite(snrs) & (snrs > 0)]
        return float(np.nanmedian(snrs)) if snrs.size else float("nan")


    diff = np.diff(residual, axis=diff_axis)
    sigma_bc = np.nanstd(diff, axis=1) / np.sqrt(2.0)
    mu_bc = np.nanmean(cube, axis=1)
    sigma_bc = np.where(sigma_bc < _MIN_SIGMA, _MIN_SIGMA, sigma_bc)
    snr_bc = mu_bc / sigma_bc
    snr_bc = snr_bc[np.isfinite(snr_bc) & (snr_bc > 0)]
    return float(np.nanmedian(snr_bc)) if snr_bc.size else float("nan")


def snr_first_diff_global(cube_brc, axis=1):

    cube = np.asarray(cube_brc, float)
    diff = np.diff(cube, axis=axis)
    noise = float(np.nanstd(diff) / np.sqrt(2.0))
    if noise < _MIN_SIGMA:
        noise = _MIN_SIGMA
    signal = float(np.nanmedian(np.abs(cube)))
    return max(signal, _EPS) / noise


def snr_col_mad(cm):

    cm = np.asarray(cm, float)
    if cm.ndim != 2:
        return float("nan")
    mean_spec = np.nanmedian(cm, axis=0)
    signal = float(np.median(np.abs(mean_spec)))
    dev = cm - mean_spec[None, :]
    noise = float(np.nanmedian(np.abs(dev)) / 0.6745)
    if noise < _MIN_SIGMA:
        noise = _MIN_SIGMA
    return max(signal, _EPS) / noise


def snr_combined(cm=None, cube=None, k=4, diff_axis=1, use_mask=True):

    if cube is not None:
        try:
            cube_arr = np.asarray(cube, float)
            if cube_arr.ndim == 3 and min(cube_arr.shape) >= 4:
                mask = build_homogeneous_mask(cube_arr) if use_mask else None

                snr = snr_pca_residual(cube_arr, k=k, diff_axis=diff_axis, mask=mask)
                if np.isfinite(snr) and snr > 0:
                    return float(snr), "pca_residual_masked" if mask is not None else "pca_residual"

                snr = snr_columnwise_diff(cube_arr, diff_axis=diff_axis, mask=mask)
                if np.isfinite(snr) and snr > 0:
                    return float(snr), "columnwise_diff_masked" if mask is not None else "columnwise_diff"
        except Exception:
            pass
    if cm is not None:
        return float(snr_col_mad(cm)), "col_mad"
    return float("nan"), "unavailable"
