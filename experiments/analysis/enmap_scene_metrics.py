# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sensors import load_enmap_l1b_vnir            # noqa: E402
from io_data import column_mean                    # noqa: E402
from ramp_gain import decompose_ramp_gain          # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = ["Nevada_USA_L1B", "Groningen_Netherlands_L1B", "Tyrol_Austria_L1B"]
ROWS = (200, 400, 1024)
N_MC = 40


def find_hdr(scene_dir):
    for root, _d, files in os.walk(os.path.join(ROOT, scene_dir)):
        for f in files:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(root, f)
    raise RuntimeError("no VNIR HDR in " + scene_dir)


def _mad(x):
    a = np.asarray(x, float)
    return float(np.median(np.abs(a - np.median(a))) / 0.6745)


def residual_sv(mat):
    cm = np.asarray(mat, float)
    cm = cm / (np.nanmean(cm, axis=1, keepdims=True) + 1e-12)
    r = cm - np.nanmedian(cm, axis=0, keepdims=True)
    return np.linalg.svd(r, compute_uv=False), r


def null_sv(n_cols, mean_spec, sigma_colmean, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_cols, len(mean_spec))) * sigma_colmean
    return residual_sv(mean_spec[None, :] + noise)[0]


def main():
    rows_out = []
    spectra = {}
    for scene in SCENES:
        hdr = find_hdr(scene)
        cube, wl, fwhm = load_enmap_l1b_vnir(hdr, max_rows=None)
        name = scene.split("_")[0]
        diff = np.diff(cube.astype(np.float64), axis=0)
        sigma_row = np.array([_mad(diff[:, :, b]) for b in range(cube.shape[2])])
        for nr in ROWS:
            cm = column_mean(cube[:nr], method="trimmed")
            mean_spec = np.nanmean(cm, axis=0)
            sigma_cm = sigma_row / np.sqrt(nr)
            s_obs, _r = residual_sv(cm)
            d = decompose_ramp_gain(cm, wl, fwhm)
            s_null = np.array([null_sv(cm.shape[0], mean_spec, sigma_cm, seed=sd)
                               for sd in range(N_MC)])
            s1n, s2n = s_null[:, 0], s_null[:, 1]
            s1, s2 = float(s_obs[0]), float(s_obs[1])
            s1c = np.sqrt(max(0.0, s1 ** 2 - s1n.mean() ** 2))
            s2c = np.sqrt(max(0.0, s2 ** 2 - s2n.mean() ** 2))
            row = dict(scene=name, rows=nr,
                       rho=round(float(d["sv_ratio_2"]), 4),
                       rank1_fraction=round(float(d["rank1_fraction"]), 4),
                       s1=round(s1, 3), s2=round(s2, 4),
                       s1_null=round(float(s1n.mean()), 4),
                       s2_null=round(float(s2n.mean()), 5),
                       s2_over_null=round(s2 / max(s2n.mean(), 1e-12), 1),
                       rho_null=round(float(s2n.mean() / s1n.mean()), 5),
                       rho_c=round(float(s2c / s1c) if s1c > 0 else 0.0, 4),
                       snr_pixel=round(float(np.median(mean_spec / sigma_row)), 1),
                       snr_cm=round(float(np.median(mean_spec / sigma_cm)), 1))
            rows_out.append(row)
            print("%-10s rows=%4d  rho=%.3f  s2/null=%.0f  rho_null=%.4f  "
                  "SNRpix=%.0f SNRcm=%.0f  s1=%.1f"
                  % (name, nr, row["rho"], row["s2_over_null"], row["rho_null"],
                     row["snr_pixel"], row["snr_cm"], row["s1"]), flush=True)
            if name == "Nevada" and nr == 1024:
                spectra["nevada"] = s_obs[:6] / s_obs[0]
                spectra["nevada_null"] = (s_null[:, :6].mean(axis=0)
                                          / s_null[:, 0].mean())
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "enmap_scene_metrics.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    if spectra:
        np.savez(os.path.join(OUT, "nevada_spectrum.npz"), **spectra)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
