# -*- coding: utf-8 -*-

import os, sys, csv, zipfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sensors import load_enmap_l1b_vnir            # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")


def find_hdr(scene_dir):
    for r, _d, files in os.walk(scene_dir):
        for f in files:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(scene_dir)


def rho_of(cm):
    cm = np.asarray(cm, float)
    good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)
    cm = cm[:, good]
    cm = cm / (np.nanmean(cm, axis=1, keepdims=True) + 1e-12)
    r = cm - np.nanmedian(cm, axis=0, keepdims=True)
    s = np.linalg.svd(r, compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


def process(cube, grid, name, out):

    rows, cols = cube.shape[0], cube.shape[1]
    nr, nc = grid
    mean_band = np.nanmean(cube, axis=2)
    for i in range(nr):
        for j in range(nc):
            r0, r1 = i * rows // nr, (i + 1) * rows // nr
            c0, c1 = j * cols // nc, (j + 1) * cols // nc
            sub = cube[r0:r1, c0:c1, :]
            cm = np.nanmean(sub, axis=0)
            rho = rho_of(cm)
            het = float(np.nanstd(mean_band[r0:r1, c0:c1]))
            out.append(dict(dataset=name, region="r%dc%d" % (i, j),
                            heterogeneity=round(het, 1), rho=round(rho, 3)))
            print("%s r%dc%d rho=%.3f het=%.0f" % (name, i, j, rho, het),
                  flush=True)


def main():
    out = []
    hdr = find_hdr(os.path.join(ROOT, "Nevada_USA_L1B"))
    cube, wl, fwhm = load_enmap_l1b_vnir(hdr, max_rows=None)
    process(cube, (5, 6), "Nevada", out)


    mat_zip = os.path.join(ROOT, "PaviaU.mat.zip")
    mat = os.path.join(ROOT, "PaviaU.mat")
    if not os.path.exists(mat) and os.path.exists(mat_zip):
        with zipfile.ZipFile(mat_zip) as z:
            z.extractall(ROOT)
    import scipy.io as sio
    pv = sio.loadmat(mat)
    key = [k for k in pv if not k.startswith("_")][0]
    pcube = np.asarray(pv[key], float)
    process(pcube, (6, 3), "PaviaU", out)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "terrain_subregions.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
