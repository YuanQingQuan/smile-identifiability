# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sensors import load_enmap_l1b_vnir                       # noqa: E402
from io_data import column_mean                               # noqa: E402
from enmap_ground_truth import load_enmap_truth_curve         # noqa: E402
from ramp_physics import _line_localization                   # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = {"Nevada": "Nevada_USA_L1B",
          "Groningen": "Groningen_Netherlands_L1B",
          "Tyrol": "Tyrol_Austria_L1B"}
AMPS = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
NOISE = [1.0, 0.5, 0.25, 0.1, 0.03]
NREP = 3
SEED = 3


def find_hdr(d):
    for r, _x, fs in os.walk(d):
        for f in fs:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(d)


def mad(x):
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def smooth(c, deg=4):
    x = np.linspace(-1, 1, len(c)); ok = np.isfinite(c)
    return np.polyval(np.polyfit(x[ok], c[ok], deg), x)


def smooth_p2p(c, deg=4):
    f = smooth(c, deg); return float(f.max() - f.min())


def local_xcorr(cm, wl, center=760.0, half=15.0, max_lag_nm=10.0):
    near = np.abs(wl - center) <= half
    wln = wl[near]; step = float(np.median(np.diff(wl)))
    ref = cm[:, near].mean(0); ref = ref / ref.mean(); refc = ref - ref.mean()
    lags = np.arange(-max_lag_nm, max_lag_nm + 1e-9, step)
    out = np.full(cm.shape[0], np.nan)
    for x in range(cm.shape[0]):
        s = cm[x, near]; s = s / s.mean()
        ccs = np.empty(len(lags))
        for j, L in enumerate(lags):
            si = np.interp(wln + L, wln, s); a = si - si.mean()
            ccs[j] = np.dot(a, refc) / (np.linalg.norm(a) * np.linalg.norm(refc) + 1e-30)
        k = int(np.argmax(ccs)); d = 0.0
        if 0 < k < len(lags) - 1:
            d = 0.5 * (ccs[k-1] - ccs[k+1]) / (ccs[k-1] - 2*ccs[k] + ccs[k+1] + 1e-12)
            d = float(np.clip(d, -1, 1))
        out[x] = lags[k] + d * step
    return out - np.nanmean(out)


def parabola_shape(X):
    u = np.linspace(-1, 1, X); s = u ** 2; s = s - s.mean()
    return s / (s.max() - s.min())


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for name, d in SCENES.items():
        cube, wl, _f = load_enmap_l1b_vnir(find_hdr(os.path.join(ROOT, d)), max_rows=None)
        rows_n = cube.shape[0]
        cm = column_mean(cube, method="trimmed")
        diff = np.diff(cube.astype(np.float64), axis=0)
        sig_cm = np.array([mad(diff[:, :, b]) for b in range(len(wl))]) / np.sqrt(rows_n)
        X = cm.shape[0]
        tr = load_enmap_truth_curve(os.path.join(ROOT, d), target_nm=763.7, n_est=X)
        meta = np.asarray(tr["dlam"], float)
        spl = [CubicSpline(wl, cm[x], extrapolate=True) for x in range(X)]
        base = np.empty_like(cm)
        for x in range(X):
            base[x] = spl[x](wl + meta[x])
        bspl = [CubicSpline(wl, base[x], extrapolate=True) for x in range(X)]
        shape = parabola_shape(X)
        print("\n===== %s  meta_p2p=%.3f nm" % (name, meta.max() - meta.min()), flush=True)

        def observe(A, ns, rng_):
            obs = np.empty_like(base)
            for x in range(X):
                obs[x] = bspl[x](wl + A * shape[x])
            return obs + rng_.standard_normal(obs.shape) * (sig_cm[None, :] * ns)

        for A in AMPS:
            rc, rb = [], []
            for _ in range(NREP):
                obs = observe(A, 1.0, rng)
                pc = _line_localization(obs, wl, 760.0, search_hw=25.0)
                rc.append(smooth_p2p(pc - np.nanmean(pc)))
                rb.append(smooth_p2p(local_xcorr(obs, wl)))
            print("  A=%+5.2f  probeC=%6.2f  probeB=%6.2f" % (A, np.median(rc), np.median(rb)),
                  flush=True)
            rows.append(dict(scene=name, kind="amplitude", param=A,
                             probeC_p2p=np.median(rc), probeB_p2p=np.median(rb)))
        for ns in NOISE:
            rc, rb = [], []
            for _ in range(NREP):
                obs = observe(0.0, ns, rng)
                pc = _line_localization(obs, wl, 760.0, search_hw=25.0)
                rc.append(smooth_p2p(pc - np.nanmean(pc)))
                rb.append(smooth_p2p(local_xcorr(obs, wl)))
            print("  noise x%4.2f  probeC=%6.2f  probeB=%6.2f" % (ns, np.median(rc), np.median(rb)),
                  flush=True)
            rows.append(dict(scene=name, kind="noise", param=ns,
                             probeC_p2p=np.median(rc), probeB_p2p=np.median(rb)))
    with open(os.path.join(OUT, "injection_reference_zero_probes.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
