# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
FIGS = os.path.join(_EXP, "figs")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "legend.framealpha": 1.0,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    "savefig.bbox": "tight",
})

FSR = 23.3779
CONTRAST = 0.03
M_COEF = 2 * CONTRAST / (1 - CONTRAST)


def airy(wl, phase0=0.0):
    delta = 2 * np.pi * (wl - 760.0) / FSR + phase0
    return 1.0 / (1.0 + M_COEF * np.sin(delta / 2.0) ** 2)


def o2a_spectrum(wl, depth=0.60, sigma=1.8):
    t = 1.0 - depth * np.exp(-0.5 * ((wl - 760.0) / sigma) ** 2)
    r = 0.20 + 0.25 / (1.0 + np.exp(-(wl - 720.0) / 140.0))
    return r * t


def main():
    wl = np.linspace(735.0, 790.0, 2000)
    smile_nm = 0.38
    phase_shift = 2 * np.pi * smile_nm / FSR / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axes[0]
    base = o2a_spectrum(wl)
    ax.plot(wl, base, "-", color="0.45", lw=1.2, label="scene spectrum (O$_2$-A)")
    ax.plot(wl, base * airy(wl, 0.0), "-", color="#0072B2", lw=1.0,
            label="+ ruler comb, column $x$")
    ax.plot(wl, base * airy(wl, phase_shift), "--", color="#D55E00", lw=1.0,
            label="+ ruler comb, column $x'$")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("normalised radiance")
    ax.set_ylim(0.0, 0.92)
    ax.legend(fontsize=8.0, loc="upper left")
    ax.text(760.0, 0.10, r"comb phase drift $=\Delta\lambda(x)$",
            fontsize=8.0, color="#D55E00", ha="center")
    ax.set_title("(a) weak etalon = per-column wavelength ruler", fontsize=8.5)

    ax = axes[1]
    labels = ["scene-based\n(confounded)", "metadata\nreference",
              "ruler design\nlimit", "ruler pooled\n(real L1B)"]
    vals = [10.6, 0.38, 0.10, 0.11]
    txt = ["10.6 nm", "0.38 nm", "0.10 nm", "0.11 nm"]
    colors = ["#D55E00", "#009E73", "#0072B2", "#56B4E9"]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.5, color=colors, alpha=0.85)
    for yy, v, s in zip(y, vals, txt):
        ax.text(v * 1.25, yy, s, va="center", fontsize=8.0,
                color="0.15")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.0)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 90)
    ax.set_xlabel(r"smile uncertainty / amplitude (nm)")
    ax.set_title("(b) beyond the confound limit", fontsize=8.5)

    for ax in axes:
        ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(wspace=0.42, left=0.09, right=0.985, top=0.86, bottom=0.20)
    png = os.path.join(FIGS, "fig11_ruler_concept.png")
    fig.savefig(png, dpi=600)
    fig.savefig(png.replace(".png", ".pdf"))
    print("[fig]", png)


if __name__ == "__main__":
    main()
