# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sensors import load_enmap_l1b_vnir            # noqa: E402
from io_data import column_mean                    # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = ["Nevada_USA_L1B", "Groningen_Netherlands_L1B", "Tyrol_Austria_L1B"]


def find_hdr(scene_dir):
    for r, _d, files in os.walk(os.path.join(ROOT, scene_dir)):
        for f in files:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(scene_dir)


def fit_line(wl, spec):

    from scipy.optimize import curve_fit
    m_fit = (wl > 754) & (wl < 772)
    m_sh = ((wl > 746) & (wl < 754)) | ((wl > 772) & (wl < 782))

    p_cont = np.polyfit(wl[m_sh], spec[m_sh], 1)
    cont = np.polyval(p_cont, wl)
    y = spec / cont
    x = wl

    def gauss(xx, depth, c, s):
        return 1.0 - depth * np.exp(-0.5 * ((xx - c) / s) ** 2)

    p0 = [1.0 - y[m_fit].min(), 763.5, 2.0]
    bounds = ([0.0, 758.0, 0.5], [1.0, 768.0, 12.0])
    popt, _ = curve_fit(gauss, x[m_fit], y[m_fit], p0=p0, bounds=bounds, maxfev=20000)
    depth, c, s = popt
    return dict(center_nm=float(c), d_eff=float(depth),
                sigma_eff_nm=float(s), fwhm_eff_nm=float(2.3548 * s),
                cont=float(np.polyval(p_cont, c)), amp=float(depth))


def main():
    rows = []
    for scene in SCENES:
        cube, wl, fwhm = load_enmap_l1b_vnir(find_hdr(scene), max_rows=None)
        cm = column_mean(cube, method="trimmed")
        spec = np.nanmean(cm, axis=0)
        r = fit_line(wl, spec)
        r["scene"] = scene.split("_")[0]
        rows.append(r)
        print("%-10s center=%.2f nm  d_eff=%.3f  sigma_eff=%.2f nm  FWHM_eff=%.2f nm"
              % (r["scene"], r["center_nm"], r["d_eff"], r["sigma_eff_nm"],
                 r["fwhm_eff_nm"]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "o2a_line_fit.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
