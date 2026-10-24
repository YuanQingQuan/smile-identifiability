# -*- coding: utf-8 -*-

import os
import re
import numpy as np


def hyperion_vnir(hdr_path=None):

    idx = np.arange(8, 58)
    if hdr_path:
        wl_full, fwhm_full = load_hyperion_hdr(hdr_path)

        wl = wl_full[7:57]
        fwhm = fwhm_full[7:57] if fwhm_full is not None else np.full_like(wl, 10.0)
        name = "Hyperion-VNIR(hdr)"
    else:
        wl = 356.0 + (idx - 1) * 10.2106
        fwhm = np.full_like(wl, 10.0)
        name = "Hyperion-VNIR"
    return dict(name=name, band_index=idx, wavelength=wl, fwhm=fwhm,
                radiance_scale=1.0/40.0,
                across_track=256)


def load_hyperion_hdr(hdr_path):

    if not hdr_path or not os.path.exists(hdr_path):
        raise FileNotFoundError(f"hdr 不存在: {hdr_path}")
    txt = open(hdr_path, "r", errors="ignore").read()
    num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    def block(key):
        m = re.search(key + r"\s*=\s*\{([^}]*)\}", txt, re.S)
        if not m:
            return None
        vals = re.findall(num, m.group(1))
        return np.array([float(v) for v in vals]) if vals else None

    wl = block("wavelength")
    fwhm = block("fwhm")
    if wl is None:
        raise ValueError(f"{hdr_path}: 缺 wavelength 表(可能是 KX 类未定标景)")
    return wl, fwhm

def hyperion_full_badmask(nbands=242):

    good = np.zeros(nbands, bool)
    good[7:57] = True
    good[76:224] = True
    return good


def enmap_vnir():
    wl = np.arange(420.0, 1000.0+1e-9, 6.5)
    fwhm = np.full_like(wl, 6.5)
    return dict(name="EnMAP-VNIR", band_index=np.arange(len(wl)), wavelength=wl,
                fwhm=fwhm, radiance_scale=1.0, across_track=1000)

def gaussian_srf(fine_grid, centers, fwhm):

    sig = np.asarray(fwhm) / 2.3548
    G = np.exp(-0.5 * ((fine_grid[None, :] - np.asarray(centers)[:, None]) / sig[:, None]) ** 2)
    return G / G.sum(1, keepdims=True)

def load_measured_srf(path):

    if path is None:
        return None
    d = np.load(path)
    return d["grid"], d["response"]


def load_enmap_l1b_vnir(hdr_path, max_rows=None):

    import os, re
    import numpy as np
    txt = open(hdr_path, "r", errors="ignore").read()
    def sca(k):
        m = re.search(k + r"\s*=\s*([0-9]+)", txt); return int(m.group(1)) if m else None
    def strv(k):
        m = re.search(k + r"\s*=\s*([A-Za-z0-9_]+)", txt); return m.group(1).lower() if m else None
    def block(k):
        m = re.search(k + r"\s*=\s*\{([^}]*)\}", txt, re.S)
        return np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+", m.group(1))]) if m else None
    S, L, B = sca("samples"), sca("lines"), sca("bands")
    dt_code = sca("data type"); bo = sca("byte order") or 0
    interleave = strv("interleave") or "bsq"
    wl = block("wavelength"); fwhm = block("fwhm")
    if wl is None or len(wl) != B:
        raise ValueError(f"{hdr_path}: 波长表缺失或长度不符")
    ENVI_DT = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 12: "u2", 13: "u4"}
    dt = ("<" if bo == 0 else ">") + ENVI_DT[dt_code]
    l1r = hdr_path.replace(".HDR", ".BSQ").replace(".hdr", ".BSQ")
    usable = min(L, os.path.getsize(l1r) // (B * S * np.dtype(dt).itemsize))
    if interleave == "bsq":


        mm = np.memmap(l1r, dtype=dt, mode="r", shape=(B, usable, S))
        n = min(usable, max_rows) if max_rows else usable
        sub = np.asarray(mm[:, :n, :], np.float32)
        cube = np.transpose(sub, (1, 2, 0))
    elif interleave == "bil":
        if max_rows:
            usable = min(usable, max_rows)
        mm = np.memmap(l1r, dtype=dt, mode="r", shape=(usable, B, S))
        cube = np.asarray(mm[:usable, :, :], np.float32)
        cube = np.transpose(cube, (0, 2, 1))
    else:
        if max_rows:
            usable = min(usable, max_rows)
        mm = np.memmap(l1r, dtype=dt, mode="r", shape=(usable, S, B))
        cube = np.asarray(mm[:usable, :, :], np.float32)
    return cube, wl, (fwhm if fwhm is not None else np.full(B, 8.0))

