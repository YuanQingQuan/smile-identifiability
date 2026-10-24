# -*- coding: utf-8 -*-

import os, sys, csv, time
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sensors import load_enmap_l1b_vnir                       # noqa: E402
from io_data import column_mean                               # noqa: E402
from enmap_ground_truth import load_enmap_truth_curve         # noqa: E402
from atmosphere import VNIR_LINES                             # noqa: E402
from model import BLSPIC                                      # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = {"Nevada": "Nevada_USA_L1B",
          "Groningen": "Groningen_Netherlands_L1B",
          "Tyrol": "Tyrol_Austria_L1B"}
SEED, ITERS = 3, 800


def find_hdr(d):
    for r, _x, fs in os.walk(d):
        for f in fs:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(d)


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def parabola_shape(X):
    u = np.linspace(-1, 1, X); s = u ** 2; s = s - s.mean()
    return s / (s.max() - s.min())


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for name, d in SCENES.items():
        t0 = time.time()
        sdir = os.path.join(ROOT, d)
        cube, wl, fwhm = load_enmap_l1b_vnir(find_hdr(sdir), max_rows=None)
        cm = column_mean(cube, method="trimmed")
        diff = np.diff(cube.astype(np.float64), axis=0)
        sig_cm = (np.array([mad(diff[:, :, b]) for b in range(len(wl))])
                  / np.sqrt(cube.shape[0]))
        X = cm.shape[0]
        good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)
        meta = np.asarray(load_enmap_truth_curve(sdir, target_nm=763.7, n_est=X)["dlam"], float)
        spl = [CubicSpline(wl, cm[x], extrapolate=True) for x in range(X)]
        base = np.empty_like(cm)
        for x in range(X):
            base[x] = spl[x](wl + meta[x])
        bspl = [CubicSpline(wl, base[x], extrapolate=True) for x in range(X)]
        shape = parabola_shape(X)

        def observe(ns, rng_):
            obs = np.empty_like(base)
            for x in range(X):
                obs[x] = bspl[x](wl)
            return obs + rng_.standard_normal(obs.shape) * (sig_cm[None, :] * ns)

        for ns in (1.0, 0.1):
            obs = observe(ns, rng)
            m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES)
            res = m.fit(obs, iters=ITERS, learn_atm=True, low_rank=2, o2_only_anchor=True)
            dl = res["dlam"]; p2p = float(dl.max() - dl.min())
            print("%-10s A=0 noise x%4.2f  p2p=%6.2f nm  meta_p2p=%.3f  (%.0fs)"
                  % (name, ns, p2p, meta.max() - meta.min(), time.time() - t0), flush=True)
            rows.append(dict(scene=name, noise=ns, p2p_nm=p2p,
                             meta_p2p_nm=float(meta.max() - meta.min())))
            with open(os.path.join(OUT, "injection_blspic_refzero_zerostate.csv"), "w",
                      newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
