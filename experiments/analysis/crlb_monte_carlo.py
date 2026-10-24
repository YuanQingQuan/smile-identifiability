# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

from crlb_theory import (closed_form_crlb, numerical_fisher_info,   # noqa: E402
                         phase_averaged_fisher_info)

D_LINE, SIG_LINE = 0.6, 1.8
SENSORS = [("Hyperion-like", 10.0, 10.0), ("EnMAP-like", 7.8, 6.4),
           ("PRISMA-like", 11.0, 9.0)]


def _geom(fwhm_srf, sampling):
    sigma_s = fwhm_srf / 2.3548
    se = np.sqrt(SIG_LINE ** 2 + sigma_s ** 2)
    de = D_LINE * SIG_LINE / se
    lam = np.arange(-3 * se, 3 * se + 1e-9, sampling)
    return lam, se, de


def _mu(lam, d, se, de):
    return 1.0 - de * np.exp(-0.5 * ((lam - d) / se) ** 2)


def gauss_newton_mc(fwhm_srf, sampling, snr, ntrial=20000, seed=1):

    lam, se, de = _geom(fwhm_srf, sampling)
    rng = np.random.default_rng(seed)
    sigma = 1.0 / snr
    err = np.empty(ntrial)
    for t in range(ntrial):
        d0 = rng.uniform(-0.05, 0.05)
        obs = _mu(lam, d0, se, de) + rng.standard_normal(lam.size) * sigma
        d = float(d0)
        for _ in range(15):
            jm = -de * (lam - d) / se ** 2 * np.exp(-0.5 * ((lam - d) / se) ** 2)
            step = np.sum(jm * (obs - _mu(lam, d, se, de))) / np.sum(jm * jm)
            d += step
            if abs(step) < 1e-12:
                break
        err[t] = d - d0
    return err.std(ddof=1)


def main():
    print("== 1/2. efficient-estimator vs the bounds ==")
    print(f"{'geometry':16s} {'SNR':>6s} {'closed':>9s} {'discrete':>9s} "
          f"{'GN std':>9s} {'GN/closed':>9s} {'GN/discrete':>11s}")
    for name, fwhm, samp in SENSORS + [("EnMAP dense 0.5", 7.8, 0.5)]:
        a = numerical_fisher_info(fwhm, samp)
        c = closed_form_crlb(fwhm, samp)
        for snr in (300, 3000):
            sd = gauss_newton_mc(fwhm, samp, snr)
            cl = c['sigma_crlb_times_snr'] / snr
            di = a['sigma_crlb_times_snr'] / snr
            print(f"{name:16s} {snr:6d} {cl:9.5f} {di:9.5f} {sd:9.5f} "
                  f"{sd/cl:9.3f} {sd/di:11.3f}")

    print("\n== 3. phase average equals the closed form ==")
    for name, fwhm, samp in SENSORS:
        c = closed_form_crlb(fwhm, samp)['sigma_crlb_times_snr']
        p = phase_averaged_fisher_info(fwhm, samp)['sigma_crlb_times_snr']
        print(f"  {name:14s} closed/phase-average = {c/p:.6f}  (must be 1.000000)")


if __name__ == "__main__":
    main()
