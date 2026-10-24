# -*- coding: utf-8 -*-

import os
import numpy as np


def load_npz(path):
    d = np.load(path)
    cube = d["cube"] if "cube" in d else d["data"]
    wl = d["wavelength"]; fwhm = d["fwhm"] if "fwhm" in d else None
    return np.asarray(cube, float), np.asarray(wl, float), (None if fwhm is None else np.asarray(fwhm, float))


def load_envi(path):
    try:
        from spectral.io import envi
    except ImportError:
        raise ImportError("ENVI 读取需要: pip install spectral")
    base = os.path.splitext(path)[0]

    candidates = [base + ext for ext in (".L1R", ".dat", ".bil", ".bsq", ".img")]
    image = next((p for p in candidates if os.path.exists(p)), None)
    img = envi.open(path, image=image)
    cube = np.asarray(img.load(), float)
    md = img.metadata
    wl = np.array([float(v) for v in md.get("wavelength", [])]) if "wavelength" in md else None
    fwhm = np.array([float(v) for v in md.get("fwhm", [])]) if "fwhm" in md else None
    return cube, wl, fwhm


def load_geotiff(path):
    import tifffile
    arr = tifffile.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"GeoTIFF 维度异常 {arr.shape}, 期望 3D 立方体")

    b = int(np.argmin(arr.shape))
    cube = np.moveaxis(arr, b, -1).astype(float)
    return cube, None, None


def load_cube(path, sensor=None):

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        cube, wl, fwhm = load_npz(path)
    elif ext in (".hdr", ".dat", ".bil", ".bsq", ".bip", ".img"):
        cube, wl, fwhm = load_envi(path)
    elif ext in (".tif", ".tiff"):
        cube, wl, fwhm = load_geotiff(path)
    else:
        raise ValueError(f"未知格式 {ext}; 请转成 .npz (见 README)")
    if (wl is None or fwhm is None) and sensor is not None:
        wl = sensor["wavelength"] if wl is None else wl
        fwhm = sensor["fwhm"] if fwhm is None else fwhm
    if cube.shape[-1] != len(wl):
        raise ValueError(f"波段数不一致: cube {cube.shape[-1]} vs wl {len(wl)} — 检查维度顺序/坏波段")
    if len(wl) < 60:
        print(f"[warn] 波段数仅 {len(wl)} — 这像多光谱不是高光谱, 确认数据类型!")
    return cube, np.asarray(wl, float), np.asarray(fwhm, float)


def column_mean(cube, good_rows=None, method="trimmed", trim=0.1, radiance_scale=1.0):

    c = cube.astype(float) * radiance_scale
    if good_rows is not None:
        c = c[good_rows]
    if method == "mean":
        cm = c.mean(0)
    elif method == "median":
        cm = np.median(c, 0)
    else:
        lo, hi = np.quantile(c, [trim, 1 - trim], axis=0)
        m = (c >= lo) & (c <= hi)
        cm = np.where(m, c, np.nan)
        cm = np.nanmean(cm, 0)
    return cm
