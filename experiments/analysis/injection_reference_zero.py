# -*- coding: utf-8 -*-

import os, sys, csv, time
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sensors import load_enmap_l1b_vnir                       # noqa: E402
from io_data import column_mean                               # noqa: E402
from enmap_ground_truth import load_enmap_truth_curve         # noqa: E402
from ramp_physics import _line_localization                   # noqa: E402
from atmosphere import VNIR_LINES                             # noqa: E402
from model import BLSPIC                                      # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = {
    "Nevada": os.path.join(ROOT, "Nevada_USA_L1B"),
    "Groningen": os.path.join(ROOT, "Groningen_Netherlands_L1B"),
    "Tyrol": os.path.join(ROOT, "Tyrol_Austria_L1B"),
}
AMPS = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
NOISE = [1.0, 0.5, 0.25, 0.1, 0.03]
BLSPIC_AMPS = [0.0, 1.0, 4.0]
BLSPIC_NOISE = [1.0, 0.1]
NREP = 3
SEED = 3
ITERS = 800


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


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    rng = np.random.default_rng(SEED)
    rows, brows = [], []
    for name, sdir in SCENES.items():
        t0 = time.time()
        cube, wl, fwhm = load_enmap_l1b_vnir(find_hdr(sdir), max_rows=None)
        rows_n = cube.shape[0]
        cm = column_mean(cube, method="trimmed")
        d = np.diff(cube.astype(np.float64), axis=0)
        sig_cm = np.array([mad(d[:, :, b]) for b in range(len(wl))]) / np.sqrt(rows_n)
        X = cm.shape[0]
        good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)

        tr = load_enmap_truth_curve(sdir, target_nm=763.7, n_est=X)
        meta = np.asarray(tr["dlam"], float)
        spl = [CubicSpline(wl, cm[x], extrapolate=True) for x in range(X)]
        base = np.empty_like(cm)
        for x in range(X):
            base[x] = spl[x](wl + meta[x])
        bspl = [CubicSpline(wl, base[x], extrapolate=True) for x in range(X)]
        shape = parabola_shape(X)
        print(f"\n===== {name}: X={X} rows={rows_n} meta_p2p={meta.max()-meta.min():.3f} nm",
              flush=True)

        def observe(A, noise_scale, rng_):
            obs = np.empty_like(base)
            for x in range(X):
                obs[x] = bspl[x](wl + A * shape[x])
            return obs + rng_.standard_normal(obs.shape) * (sig_cm[None, :] * noise_scale)

        for A in AMPS:
            recC, recB, slC, slB = [], [], [], []
            for _ in range(NREP):
                obs = observe(A, 1.0, rng)
                pc = _line_localization(obs, wl, 760.0, search_hw=25.0); pc -= np.nanmean(pc)
                pb = local_xcorr(obs, wl)
                recC.append(smooth_p2p(pc)); recB.append(smooth_p2p(pb))
                if A != 0.0:
                    slC.append(float(np.polyfit(A * shape, pc, 1)[0]))
                    slB.append(float(np.polyfit(A * shape, pb, 1)[0]))
            med = lambda v: float(np.median(v)) if v else float("nan")
            print(f"  A={A:+5.2f}  probeC p2p={np.median(recC):6.2f} slope={med(slC):+.2f}"
                  f"  probeB p2p={np.median(recB):6.2f} slope={med(slB):+.2f}", flush=True)
            rows.append(dict(scene=name, kind="amplitude", param=A,
                             probeC_p2p=np.median(recC), probeC_slope=med(slC),
                             probeB_p2p=np.median(recB), probeB_slope=med(slB)))
        write_csv(os.path.join(OUT, "injection_reference_zero.csv"), rows)

        for sc in NOISE:
            recC, recB = [], []
            for _ in range(NREP):
                obs = observe(0.0, sc, rng)
                pc = _line_localization(obs, wl, 760.0, search_hw=25.0)
                recC.append(smooth_p2p(pc - np.nanmean(pc)))
                recB.append(smooth_p2p(local_xcorr(obs, wl)))
            print(f"  noise x{sc:4.2f}  probeC p2p={np.median(recC):6.2f}"
                  f"  probeB p2p={np.median(recB):6.2f}", flush=True)
            rows.append(dict(scene=name, kind="noise", param=sc,
                             probeC_p2p=np.median(recC), probeC_slope=np.nan,
                             probeB_p2p=np.median(recB), probeB_slope=np.nan))
        write_csv(os.path.join(OUT, "injection_reference_zero.csv"), rows)

        for A in BLSPIC_AMPS:
            obs = observe(A, 1.0, rng)
            m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES)
            res = m.fit(obs, iters=ITERS, learn_atm=True, low_rank=2, o2_only_anchor=True)
            dl = res["dlam"]; p2p = float(dl.max() - dl.min())
            R = float(np.corrcoef(dl - dl.mean(), A * shape)[0, 1]) if A > 0 else np.nan
            print(f"  BLSPIC A={A:+5.2f}  p2p={p2p:6.2f} nm  R={R:+.2f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            brows.append(dict(scene=name, kind="amplitude", param=A, p2p=p2p, R=R,
                              noise=1.0, meta_p2p=float(meta.max() - meta.min())))
            write_csv(os.path.join(OUT, "injection_blspic_refzero.csv"), brows)
        for sc in BLSPIC_NOISE:
            obs = observe(0.0, sc, rng)
            m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES)
            res = m.fit(obs, iters=ITERS, learn_atm=True, low_rank=2, o2_only_anchor=True)
            dl = res["dlam"]; p2p = float(dl.max() - dl.min())
            print(f"  BLSPIC A=0 noise x{sc:4.2f}  p2p={p2p:6.2f} nm", flush=True)
            brows.append(dict(scene=name, kind="noise", param=0.0, p2p=p2p, R=np.nan,
                              noise=sc, meta_p2p=float(meta.max() - meta.min())))
            write_csv(os.path.join(OUT, "injection_blspic_refzero.csv"), brows)

    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
