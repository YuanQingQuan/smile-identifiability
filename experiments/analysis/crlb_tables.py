# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from crlb_theory import (phase_averaged_fisher_info, closed_form_crlb,   # noqa: E402
                         _effective_width_depth)

OUT = os.path.join(ROOT, "experiments", "out")
SENSORS = [("Hyperion-like", 10.0, 10.0), ("EnMAP-like", 7.8, 6.4),
           ("PRISMA-like", 11.0, 9.0)]
K_P2P = (3.3, 3.8)
N_PHASE = 360


def phase_spread(fwhm_srf, sampling, line_depth=0.6, line_sigma=1.8):

    se, de, _ = _effective_width_depth(fwhm_srf, line_depth, line_sigma)
    n_half = int(np.ceil(4.0 * se / sampling))
    ks = np.arange(-n_half, n_half + 1)
    vals = []
    for phi in np.linspace(0.0, sampling, N_PHASE, endpoint=False):
        lam = ks * sampling + phi
        d = de * lam / se ** 2 * np.exp(-0.5 * (lam / se) ** 2)
        vals.append(1.0 / np.sqrt(np.sum(d ** 2)))
    vals = np.array(vals)
    cf = closed_form_crlb(fwhm_srf, sampling)["sigma_crlb_times_snr"]
    return float(vals.min() / cf), float(vals.max() / cf)


def main():
    rows = []
    print("=== CRLB verification (d=0.6, sigma_l=1.8 nm) ===")
    for name, fwhm, samp in SENSORS:
        p = phase_averaged_fisher_info(fwhm, samp)
        cl = closed_form_crlb(fwhm, samp)
        lo, hi = phase_spread(fwhm, samp)
        sigma = cl["sigma_crlb_times_snr"] / 300.0
        rows.append(dict(sensor=name, delta_nm=samp, fwhm_nm=fwhm,
                         sigma_eff_nm=round(cl["sigma_eff_nm"], 2),
                         d_eff=round(cl["d_eff"], 3),
                         sigma_crlb_nm=round(sigma, 3),
                         phase_lo=round(lo, 2), phase_hi=round(hi, 2),
                         ratio_closed_phaseavg=round(
                             cl["sigma_crlb_times_snr"] / p["sigma_crlb_times_snr"], 3)))
        print("%-14s sigma_eff=%.2f d_eff=%.3f  sigma(SNR300)=%.3f nm  "
              "phase %.2f-%.2f  closed/phaseavg=%.3f"
              % (name, cl["sigma_eff_nm"], cl["d_eff"], sigma, lo, hi,
                 cl["sigma_crlb_times_snr"] / p["sigma_crlb_times_snr"]))


    scene_rows = []
    metrics = {r["scene"]: r for r in
               csv.DictReader(open(os.path.join(OUT, "enmap_scene_metrics.csv"),
                                   encoding="utf-8"))
               if int(r["rows"]) == 1024}
    refs = {"Nevada": 0.380, "Groningen": 0.357, "Tyrol": 0.355}
    cf = closed_form_crlb(7.8, 6.4)["sigma_crlb_times_snr"]
    for sc in ("Nevada", "Groningen", "Tyrol"):
        snr_cm = float(metrics[sc]["snr_cm"])
        sigma = cf / snr_cm
        p2p_lo, p2p_hi = sigma * K_P2P[0], sigma * K_P2P[1]
        ref_lo = refs[sc] / p2p_hi
        ref_hi = refs[sc] / p2p_lo
        scene_rows.append(dict(scene=sc, snr_cm=snr_cm,
                               sigma_crlb_nm=round(sigma, 4),
                               p2p_floor_lo_nm=round(p2p_lo, 3),
                               p2p_floor_hi_nm=round(p2p_hi, 3),
                               ref_over_floor_lo=round(ref_lo, 2),
                               ref_over_floor_hi=round(ref_hi, 2)))
        print("%-10s SNR_cm=%.0f  sigma_CRLB=%.4f nm  p2p floor=%.3f-%.3f nm  "
              "ref/floor=%.2f-%.2f"
              % (sc, snr_cm, sigma, p2p_lo, p2p_hi, ref_lo, ref_hi))

    with open(os.path.join(OUT, "crlb_table.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "crlb_scene_floor.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scene_rows[0].keys()))
        w.writeheader(); w.writerows(scene_rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
