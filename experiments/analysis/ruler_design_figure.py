# -*- coding: utf-8 -*-

import os
import sys
import csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
_OWR = os.path.join(os.path.dirname(_EXP), "owr")
FIGS = os.path.join(_EXP, "figs")

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


def main():
    rows = list(csv.DictReader(open(os.path.join(_OWR, "out", "design_space.csv"),
                                    encoding="utf-8")))
    c = np.array([float(r["contrast"]) * 100 for r in rows])
    s = np.array([float(r["snr"]) for r in rows])
    sig = np.array([float(r["sigma_ruler_nm"]) for r in rows])

    imm = list(csv.DictReader(open(os.path.join(_OWR, "out", "immunity.csv"),
                                   encoding="utf-8")))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))

    ax = axes[0]

    cg = np.linspace(1.0, 10.0, 120)
    sg = np.logspace(np.log10(30), np.log10(1000), 120)
    CG, SG = np.meshgrid(cg, sg)


    K = 0.100 * 3.0 * 300.0
    SIG = K / (CG * SG)
    cs = ax.contour(CG, SG, SIG, levels=[0.02, 0.05, 0.1, 0.2, 0.5],
                    colors="0.45", linewidths=0.7)
    ax.clabel(cs, fmt="%.2f", fontsize=8.0, inline=True)
    ax.plot(3.0, 300.0, "*", ms=9, color="#D55E00", mec="w", mew=0.6)
    ax.annotate("EnMAP VNIR\n$c$=3%, SNR=300\n$\\sigma_\\lambda$=0.10 nm",
                xy=(3.0, 300.0), xytext=(4.4, 120.0), fontsize=8.5,
                color="0.15",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))
    ax.set_xlabel("fringe contrast $c$ (%)")
    ax.set_ylabel("SNR per band")
    ax.set_yscale("log")
    ax.set_title("(a) design map: readout noise (nm)", fontsize=9)

    ax = axes[1]
    styles = {"gain_gradient": ("#0072B2", "o", "gain gradient"),
              "spectral_ramp": ("#D55E00", "s", "spectral ramp"),
              "water_vapor_var": ("#009E73", "^", "water vapour")}
    for key, (col, mk, lab) in styles.items():
        sel = [r for r in imm if r["confound"] == key]
        amp = np.array([float(r["amplitude"]) * 100 for r in sel])
        rmse = np.array([float(r["ruler_rmse_nm"]) for r in sel])
        ax.plot(amp, rmse, mk + "-", ms=3.0, lw=1.0, color=col, label=lab)
    base = np.mean([float(r["baseline_rmse_nm"]) for r in imm])
    ax.axhline(base, color="0.45", ls="--", lw=1.0)
    ax.text(19.5, base * 1.25, "scene-based (same data)", ha="right",
            fontsize=8.5, color="0.35")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 30)
    ax.set_xlabel("confound amplitude (%)")
    ax.set_ylabel("pooled smile RMSE (nm)")
    ax.legend(fontsize=8.5, loc="center left")
    ax.set_title("(b) confound immunity", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(wspace=0.36, left=0.10, right=0.99, top=0.86, bottom=0.20)
    png = os.path.join(FIGS, "fig13_ruler_design.png")
    fig.savefig(png, dpi=600)
    fig.savefig(png.replace(".png", ".pdf"))
    print("[fig]", png)


if __name__ == "__main__":
    main()
