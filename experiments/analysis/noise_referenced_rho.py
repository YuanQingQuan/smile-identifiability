# -*- coding: utf-8 -*-

import os
import sys
import csv
import glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EXP)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))

from sensors import load_enmap_l1b_vnir  # noqa: E402
from io_data import column_mean          # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "legend.framealpha": 1.0,
    "savefig.bbox": "tight",
})

SCENES = {
    "Nevada": "Nevada_USA_L1B",
    "Groningen": "Groningen_Netherlands_L1B",
    "Tyrol": "Tyrol_Austria_L1B",
}
OUT = os.path.join(_EXP, "out")
FIGS = os.path.join(_EXP, "figs")
MAX_ROWS = None
N_MC = 40


def find_hdr(scene_dir):
    for root, _, files in os.walk(os.path.join(_ROOT, scene_dir)):
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
    return np.linalg.svd(r, compute_uv=False)


def null_sv(n_cols, mean_spec, sigma_colmean, seed=0):

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_cols, len(mean_spec))) * sigma_colmean
    mat = mean_spec[None, :] + noise
    return residual_sv(mat)


def analyse_scene(name):
    hdr = find_hdr(SCENES[name])
    cube, wl, fwhm = load_enmap_l1b_vnir(hdr, max_rows=MAX_ROWS)
    rows, cols, bands = cube.shape


    diff = np.diff(cube.astype(float), axis=0)
    sigma_row = np.array([_mad(diff[:, :, b]) for b in range(bands)])
    sigma_cm = sigma_row / np.sqrt(rows)

    cm = column_mean(cube, method="trimmed")
    mean_spec = np.nanmean(cm, axis=0)
    s_obs = residual_sv(cm)

    s_null = np.array([null_sv(cols, mean_spec, sigma_cm, seed=s)
                       for s in range(N_MC)])
    s1n, s2n = s_null[:, 0], s_null[:, 1]

    s1, s2 = s_obs[0], s_obs[1]
    s1c = np.sqrt(max(0.0, s1 ** 2 - s1n.mean() ** 2))
    s2c = np.sqrt(max(0.0, s2 ** 2 - s2n.mean() ** 2))
    rho_obs = float(s2 / s1)
    rho_c = float(s2c / s1c) if s1c > 0 else 0.0
    return dict(
        scene=name, rows=rows, cols=cols, bands=bands,
        snr_med=float(np.median(mean_spec / sigma_row)),
        snr_cm_med=float(np.median(mean_spec / sigma_cm)),
        s1=float(s1), s2=float(s2), rho_obs=rho_obs,
        s1_null=float(s1n.mean()), s1_null_std=float(s1n.std()),
        s2_null=float(s2n.mean()), s2_null_std=float(s2n.std()),
        rho_null=float(s2n.mean() / s1n.mean()),
        s2_over_null=float(s2 / s2n.mean()), rho_c=rho_c)


def main():
    rows = []
    for name in SCENES:
        r = analyse_scene(name)
        rows.append(r)
        print(f"{name:10s} SNR_cm={r['snr_cm_med']:6.1f}  "
              f"s2={r['s2']:8.3f}  s2_null={r['s2_null']:.3f}  "
              f"s2/null={r['s2_over_null']:7.1f}  "
              f"rho_obs={r['rho_obs']:.3f}  rho_c={r['rho_c']:.3f}")

    csv_path = os.path.join(OUT, "noise_referenced_rho.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = list(rows[0].keys())
        w.writerow(keys)
        for r in rows:
            w.writerow([f"{r[k]:.5f}" if isinstance(r[k], float) else r[k]
                        for k in keys])
    print("[csv]", csv_path)


    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    names = [r["scene"] for r in rows]
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x - 0.19, [r["rho_obs"] for r in rows], 0.36, color="#0072B2",
           edgecolor="k", lw=0.5, label=r"observed $\rho$")
    ax.bar(x + 0.19, [r["rho_null"] for r in rows], 0.36, color="#56B4E9",
           edgecolor="k", lw=0.5, label=r"noise null $\rho$")
    ax.plot(x + 0.19, [r["rho_c"] for r in rows], "o", ms=4.5,
            color="#D55E00", label=r"noise-corrected $\rho_c$")
    ax.axhline(0.1, color="0.35", ls="--", lw=0.9)
    ax.axhline(0.02, color="darkred", ls="-", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"rank criterion $\rho$")
    ax.set_ylim(0, 1.05)

    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.17),
              ncol=2, borderaxespad=0.0)

    ax = axes[1]
    ax.bar(x - 0.19, [r["s2"] for r in rows], 0.36, color="#D55E00",
           edgecolor="k", lw=0.5, label=r"$s_2$ (data)")
    ax.bar(x + 0.19, [r["s2_null"] for r in rows], 0.36, color="#E69F00",
           edgecolor="k", lw=0.5, label=r"$s_2$ (noise null)")
    for i, r in enumerate(rows):
        ax.text(x[i], r["s2"] * 1.25, f"$\\times${r['s2_over_null']:.0f}",
                ha="center", fontsize=8.5, color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 400)
    ax.set_ylabel(r"second singular value $s_2$ (log)")
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.17),
              ncol=2, borderaxespad=0.0)

    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.tick_params(labelsize=8.5)
        ax.set_title(tag, fontsize=9, loc="left")
    fig.subplots_adjust(wspace=0.34, left=0.10, right=0.99, top=0.72, bottom=0.16)
    png = os.path.join(FIGS, "fig12_noise_ref_rho.png")
    fig.savefig(png, dpi=600)
    fig.savefig(png.replace(".png", ".pdf"))
    print("[fig]", png)


if __name__ == "__main__":
    main()
