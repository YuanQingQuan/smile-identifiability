# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theory.ruler_fisher import (
    closed_form_sigma_lambda_broadband, optimal_fsr, sigma_lambda_min,
    confound_immunity, fringe_period, index_fused_silica,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)


LAM = np.arange(420.0, 1000.0 + 1e-9, 6.5)
FWHM = 6.5
NB = len(LAM)

CONTRASTS = [0.01, 0.02, 0.03, 0.05, 0.10]
SNRS = [30.0, 100.0, 300.0]
L_GRID = np.linspace(2.0e3, 6.0e4, 200)


def optimize_spacer(contrast, snr, fwhm=FWHM, lam=LAM, l_grid=L_GRID):

    sig = np.array([closed_form_sigma_lambda_broadband(
        lam, contrast, snr, L, fwhm_srf_nm=fwhm) for L in l_grid])
    i = int(np.argmin(sig))
    return float(l_grid[i]), float(sig[i]), sig


def main():
    rows = []
    curves = {}
    for c in CONTRASTS:
        for snr in SNRS:
            L_opt, sig_opt, curve = optimize_spacer(c, snr)
            P_opt_at_760 = fringe_period(760.0, L_opt, group=True)

            sigma_scene = 4.6 * 300.0 / snr
            rows.append(dict(contrast=c, snr=snr, L_opt_nm=L_opt,
                             fringe_period_at_760_nm=P_opt_at_760,
                             sigma_ruler_nm=sig_opt,
                             sigma_scene_based_nm=sigma_scene,
                             gain_x=sigma_scene / sig_opt))
            curves[(c, snr)] = (L_opt, sig_opt, curve)


    keys = ["contrast", "snr", "L_opt_nm", "fringe_period_at_760_nm",
            "sigma_ruler_nm", "sigma_scene_based_nm", "gain_x"]
    with open(os.path.join(OUT, "design_space.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" for k in keys) + "\n")


    snr_band = np.full(NB, 300.0)
    imm = {}
    for kind in ["gain", "ramp", "lines"]:
        imm[kind] = confound_immunity(LAM, snr_band, curves[(0.03, 300.0)][0],
                                      contrast=0.03, fwhm_srf_nm=FWHM,
                                      nuisance=kind)

    L_star_constP = optimal_fsr(FWHM) * 2 * index_fused_silica(760.0) * 1e0
    rules = {
        "sensor_model": "EnMAP VNIR-like: 420-1000nm, SSD=FWHM=6.5nm, 90 bands",
        "constant_P_rule_fsr_nm": optimal_fsr(FWHM),

        "immunity_c03_snr300": {k: {"projection_coeffs": v["projection_coeffs"],
                                    "bias_nm_per_unit_rel_amplitude": v["bias_nm_per_unit"]}
                                for k, v in imm.items()},
        "headline": rows[[ (r["contrast"], r["snr"]) for r in rows].index((0.03, 300.0))],
    }
    with open(os.path.join(OUT, "design_rules.json"), "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2, default=float)


    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(OUT), "experiments"))
    import figstyle
    figstyle.apply()
    from figstyle import (label_panels, C_BLUE, C_VERM, C_GREEN, C_ORANGE,
                          C_PURPLE, C_SKY, C_GRAY, COL2)

    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.6))

    ax = axes[0, 0]
    for c, col in zip(CONTRASTS, [C_SKY, C_ORANGE, C_BLUE, C_VERM, C_PURPLE]):
        _, _, curve = curves[(c, 300.0)]
        ax.plot(L_GRID / 1000.0, curve, lw=1.0, color=col, label=f"c={c*100:.0f}%")
    ax.set_xlabel("etalon spacer $L$ (µm)")
    ax.set_ylabel(r"readout precision $\sigma_\lambda$ (nm)")
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 30)
    ax.legend(loc="upper left", handlelength=1.0)
    L_star = curves[(0.03, 300.0)][0] / 1000.0
    ax.axvline(L_star, color=C_GRAY, lw=0.6, ls=":")
    ax.text(L_star + 1, 5e-3, f"$L^*$={L_star:.1f} µm", color=C_GRAY, fontsize=6)

    ax = axes[0, 1]
    cgrid = np.linspace(0.005, 0.12, 90)
    sgrid = np.linspace(20, 500, 90)
    Cg, Sg = np.meshgrid(cgrid, sgrid)
    L_opt, sig_opt, _ = curves[(0.03, 300.0)]
    Sig = np.array([[closed_form_sigma_lambda_broadband(LAM, cc, ss, L_opt, fwhm_srf_nm=FWHM)
                     for cc in cgrid] for ss in sgrid])
    pc = ax.contourf(Cg * 100, Sg, np.log10(Sig), levels=25, cmap="viridis_r")
    cb = fig.colorbar(pc, ax=ax, pad=0.02, ticks=[-2, -1.5, -1, -0.5, 0])
    cb.set_label(r"$\log_{10}\sigma_\lambda$ (nm)")
    cb.ax.tick_params(labelsize=6)
    cs = ax.contour(Cg * 100, Sg, Sig, levels=[0.02, 0.05, 0.1, 0.2, 0.5],
                    colors="w", linewidths=0.7)

    cs_lab = ax.contour(Cg * 100, Sg, Sig, levels=[0.02, 0.05, 0.2, 0.5],
                        colors="w", linewidths=0.0)
    ax.clabel(cs_lab, fmt="%.2f nm", fontsize=5.5, inline=True)
    ax.set_xlabel("fringe contrast $c$ (%)")
    ax.set_ylabel("SNR per band")
    ax.plot(3, 300, "*", ms=7, color=C_VERM, mec="w", mew=0.6)
    ax.annotate("EnMAP VNIR\n$c$=3%, SNR=300\n$\\sigma_\\lambda$=0.093 nm",
                xy=(3, 300), xytext=(4.4, 168), fontsize=6, color="w",
                bbox=dict(facecolor="#3B4A5A", edgecolor="none",
                          alpha=0.80, pad=2.0))

    ax = axes[1, 0]
    from theory.ruler_fisher import numerical_fisher_phase
    nf = numerical_fisher_phase(LAM, np.ones(NB), np.full(NB, 300.0), L_opt, 0.03,
                                fwhm_srf_nm=FWHM)
    ax.plot(LAM, 1.0 / nf["sigma_lambda_band_nm"] ** 2 / np.max(1.0 / nf["sigma_lambda_band_nm"] ** 2),
            lw=1.0, color=C_BLUE)
    ax.axvline(760, color=C_GREEN, ls=":", lw=0.7)
    ax.text(766, 0.08, "O2-A 760 nm", color=C_GREEN, fontsize=6)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("relative Fisher contribution")
    ax.set_ylim(0, 1.08)

    ax = axes[1, 1]
    kinds = ["gain", "ramp", "lines"]
    biases = [0.01 * imm[k]["bias_nm_per_unit"] for k in kinds]
    labels = ["gain\nstriping", "spectral\nramp", "line-depth\nerror"]
    sigma_ruler = rules["headline"]["sigma_ruler_nm"]
    floor = min(min(abs(b) for b in biases if abs(b) > 0), sigma_ruler) / 30.0
    bars = ax.bar(labels, [max(abs(b), floor) for b in biases],
                  color=[C_BLUE, C_VERM, C_GREEN], width=0.55)
    ax.axhline(sigma_ruler, color="k", ls="--", lw=0.8)
    ax.text(0.02, sigma_ruler * 0.6, f"ruler noise floor {sigma_ruler:.3f} nm",
            transform=ax.get_yaxis_transform(), fontsize=6, va="top")
    ax.set_ylabel("wavelength bias (nm)")
    ax.set_yscale("log")
    ax.set_ylim(floor / 3, 0.4)
    for b, v in zip(bars, biases):
        av = abs(v)
        txt = f"{av*1000:.2g} pm" if av >= floor else f"<{floor*1000:.2g} pm"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.6, txt,
                ha="center", va="bottom", fontsize=6)

    label_panels(axes)
    fig.subplots_adjust(wspace=0.42, hspace=0.34, left=0.07, right=0.98,
                        top=0.97, bottom=0.1)
    out = os.path.join(OUT, "fig_design_space.png")
    plt.savefig(out)
    plt.savefig(out.replace(".png", ".pdf"))
    print("[fig]", out)
    for r in rows:
        print(f"c={r['contrast']*100:4.1f}%  SNR={r['snr']:5.0f}  L*={r['L_opt_nm']/1000:5.2f}µm  "
              f"P760={r['fringe_period_at_760_nm']:5.1f}nm  σ_ruler={r['sigma_ruler_nm']:.4f}nm  "
              f"σ_scene={r['sigma_scene_based_nm']:.2f}nm  gain={r['gain_x']:.0f}x")


if __name__ == "__main__":
    main()
