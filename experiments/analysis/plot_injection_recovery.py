# -*- coding: utf-8 -*-

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "experiments", "out")
FIGS = os.path.join(ROOT, "experiments", "figs")

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "figure.dpi": 300, "savefig.dpi": 400,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

SCENE_COLOR = {"Nevada": "#D55E00", "Groningen": "#0072B2", "Tyrol": "#009E73"}
SCENE_MARK = {"Nevada": "o", "Groningen": "s", "Tyrol": "^"}
REF = {"Nevada": 0.380, "Groningen": 0.357, "Tyrol": 0.355}
SCENES = ("Nevada", "Groningen", "Tyrol")


def read(fn):
    with open(os.path.join(OUT, fn), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    rows = read("injection_reference_zero_probes.csv")
    try:
        brows = read("injection_blspic_refzero.csv")
    except FileNotFoundError:
        brows = []

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    ax, ax2 = axes


    for scene in SCENES:
        amp = [r for r in rows if r["scene"] == scene and r["kind"] == "amplitude"]
        amp.sort(key=lambda r: float(r["param"]))
        A = np.array([float(r["param"]) for r in amp])
        pc = np.array([float(r["probeC_p2p"]) for r in amp])
        ax.plot(A, pc, "-", color=SCENE_COLOR[scene], marker=SCENE_MARK[scene],
                ms=3.6, lw=1.2, label=f"{scene} (probe)")
        i0 = int(np.argmin(np.abs(A)))
        ax.plot(A[i0], pc[i0], marker="x", ms=6.0, mew=1.6,
                color=SCENE_COLOR[scene])
    for scene in SCENES:
        amp = [r for r in brows if r["scene"] == scene and r["kind"] == "amplitude"]
        if len(amp) < 4:
            continue
        amp.sort(key=lambda r: float(r["param"]))
        A = np.array([float(r["param"]) for r in amp])
        p = np.array([float(r["p2p"]) for r in amp])
        ax.plot(A, p, "--", color=SCENE_COLOR[scene], marker="d", ms=3.2,
                lw=1.2, label=f"{scene} (BL-SPIC)")

    ax.axhline(0.37, color="0.6", ls=(0, (4, 3)), lw=0.9,
               label="metadata reference (0.36-0.38 nm)")
    ax.axvline(0.0, color="0.8", lw=0.7)
    ax.set_yscale("log")
    ax.set_xlim(-4.6, 4.6); ax.set_ylim(0.25, 15)
    ax.set_xlabel("signed injection amplitude $A$ [nm]")
    ax.set_ylabel("recovered p2p [nm]")
    ax.set_title("(a) Reference-zero injection-recovery")
    h, lab = ax.get_legend_handles_labels()


    for scene in SCENES:
        nz = [r for r in rows if r["scene"] == scene and r["kind"] == "noise"]
        nz.sort(key=lambda r: float(r["param"]))
        sc = np.array([float(r["param"]) for r in nz])
        pc = np.array([float(r["probeC_p2p"]) for r in nz])
        ax2.plot(1.0 / sc, pc, "-", color=SCENE_COLOR[scene],
                 marker=SCENE_MARK[scene], ms=3.6, lw=1.2,
                 label=f"{scene} (probe)")
    ax2.axhline(0.37, color="0.6", ls=(0, (4, 3)), lw=0.9)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_ylim(0.25, 15)
    ax2.set_xlabel("noise reduction factor (33$\\times$ at right)")
    ax2.set_ylabel("recovered p2p, zero injected smile [nm]")
    ax2.set_title("(b) Bias, not noise")
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.22), handlelength=1.8,
               columnspacing=1.4, labelspacing=0.3)

    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig14_injection_recovery.{ext}"),
                    bbox_inches="tight")
    print("wrote", os.path.join(FIGS, "fig14_injection_recovery.png"))


if __name__ == "__main__":
    main()
