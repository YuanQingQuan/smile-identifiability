# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sv2_independence_proof import (sensor_observation, gaussian_srf,
                                    build_reflectance,
                                    build_atmospheric_transmittance)  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
SENSORS = [("EnMAP-like", 7.8, 6.4), ("Hyperion-like", 10.0, 10.0),
           ("PRISMA-like", 11.0, 9.0)]
N_PER_CLASS = 18
RAMP_AMPS = (1.0, 2.0, 3.0, 5.0)


def gain_background(x_cols, rng):
    a = rng.uniform(0.05, 0.15)
    b = rng.uniform(0.01, 0.03)
    return 1.0 + a * np.linspace(-1, 1, x_cols) + b * rng.standard_normal(x_cols)


def smile_shape(x_cols, p2p_nm, rng):
    u = np.linspace(-1, 1, x_cols)
    c1, c2 = rng.uniform(0.5, 1.0, 2)
    d = c1 * u ** 2 + c2 * u ** 4
    d -= d.min()
    return d / d.max() * p2p_nm


def observe_with_smile(wl, srf_fwhm, delta_x):

    fine = np.arange(420, 1000, 0.5)
    L = build_reflectance(fine) * build_atmospheric_transmittance(fine)
    X = len(delta_x)
    out = np.zeros((X, len(wl)))
    for xi in range(X):
        for b, w in enumerate(wl):
            srf = gaussian_srf(fine, w + delta_x[xi], srf_fwhm)
            srf /= srf.sum()
            out[xi, b] = np.dot(L, srf)
    return out


def spectrum(mat):

    cm = np.asarray(mat, float)
    cm = cm / (np.nanmean(cm, axis=1, keepdims=True) + 1e-12)
    r = cm - np.nanmedian(cm, axis=0, keepdims=True)
    s = np.linalg.svd(r, compute_uv=False)
    return s[:6] / s[0]


def main():
    rng = np.random.default_rng(7)
    rows = []
    spectra = {}
    for sname, fwhm, samp in SENSORS:
        obs0, wl, _, _, _ = sensor_observation(fwhm, samp)
        X = obs0.shape[0]

        for k in range(N_PER_CLASS):
            g = gain_background(X, rng)
            sv = spectrum(obs0 * g[:, None])
            rows.append(dict(sensor=sname, cls="gain", amp_nm=0.0,
                             rho=float(sv[1]), rep=k))

        for k in range(N_PER_CLASS):
            g = gain_background(X, rng)
            A = float(rng.choice(RAMP_AMPS))
            d = smile_shape(X, A, rng)
            obs = observe_with_smile(wl, fwhm, d) * g[:, None]
            sv = spectrum(obs)
            rows.append(dict(sensor=sname, cls="ramp", amp_nm=A,
                             rho=float(sv[1]), rep=k))

        if sname == "EnMAP-like":
            g0 = gain_background(X, np.random.default_rng(11))
            spectra["gain"] = spectrum(obs0 * g0[:, None])
            for A, tag in [(0.38, "smile038"), (3.0, "smile3")]:
                d = smile_shape(X, A, np.random.default_rng(13))
                sv = spectrum(observe_with_smile(wl, fwhm, d) * g0[:, None])
                spectra[tag] = sv
                print("spectrum %-8s s2/s1=%.4f" % (tag, sv[1]), flush=True)


    def auc_youden(gain_vals, ramp_vals):
        g = np.asarray(gain_vals); r = np.asarray(ramp_vals)
        c = sum(1 for p in r for n in g if p > n) + 0.5 * sum(
            1 for p in r for n in g if p == n)
        auc = c / (len(g) * len(r))
        allv = np.sort(np.concatenate([g, r]))
        best, thr = -1, None
        for t in allv:
            tpr = (r >= t).mean(); fpr = (g >= t).mean()
            j = tpr - fpr
            if j > best:
                best, thr = j, t
        return auc, thr

    summary = []
    for sname, _f, _s in SENSORS:
        g = [r["rho"] for r in rows if r["sensor"] == sname and r["cls"] == "gain"]
        rp = [r["rho"] for r in rows if r["sensor"] == sname and r["cls"] == "ramp"]
        auc, thr = auc_youden(g, rp)
        summary.append((sname, np.mean(g), np.mean(rp), auc, thr))
        print("%-14s gain rho=%.4f  ramp rho=%.3f  AUC=%.3f  Youden=%.4f"
              % (sname, np.mean(g), np.mean(rp), auc, thr), flush=True)
    gall = [r["rho"] for r in rows if r["cls"] == "gain"]
    rall = [r["rho"] for r in rows if r["cls"] == "ramp"]
    auc, thr = auc_youden(gall, rall)
    print("POOLED        gain rho=%.4f  ramp rho=%.3f  AUC=%.3f  Youden=%.4f"
          % (np.mean(gall), np.mean(rall), auc, thr), flush=True)

    for A in RAMP_AMPS:
        rr = [r["rho"] for r in rows if r["cls"] == "ramp" and abs(r["amp_nm"] - A) < 1e-9]
        print("  ramp A=%.0f nm: rho mean=%.3f  min=%.3f" % (A, np.mean(rr), np.min(rr)),
              flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rank_criterion_validation.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez(os.path.join(OUT, "rank_spectra.npz"), **spectra)
    with open(os.path.join(OUT, "rank_criterion_summary.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "gain_rho_mean", "ramp_rho_mean", "AUC", "youden"])
        for row in summary:
            w.writerow(row)
        w.writerow(["pooled", np.mean(gall), np.mean(rall), auc, thr])
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
