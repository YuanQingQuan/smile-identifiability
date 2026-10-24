# -*- coding: utf-8 -*-

import os
import sys
import csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
OUT = os.path.join(_EXP, "out")
FIGS = os.path.join(_EXP, "figs")
os.makedirs(OUT, exist_ok=True)

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

FINE = np.arange(400.0, 2500.0, 0.5)
LINES = {
    "O$_2$-B 688":  (688.0, 2.0, 0.20, "vnir"),
    "O$_2$-A 760":  (760.0, 1.8, 0.60, "vnir"),
    "H$_2$O 940":   (940.0, 15.0, 0.70, "swir"),
    "H$_2$O 1130":  (1130.0, 20.0, 0.45, "swir"),
    "H$_2$O 1650":  (1650.0, 25.0, 0.35, "swir"),
}
OTHER = [(720.0, 6.0, 0.15), (820.0, 8.0, 0.30), (1400.0, 30.0, 0.85),
         (1900.0, 40.0, 0.95), (2000.0, 15.0, 0.25), (2200.0, 60.0, 0.20)]
SENSOR = {"vnir": dict(sampling=6.5, fwhm=6.5, lo=400.0, hi=1000.0),
          "swir": dict(sampling=8.8, fwhm=10.0, lo=900.0, hi=2500.0)}
DEPTHS = (0.05, 0.10, 0.20, 0.40, 0.60)
X_COLS = 1000
DLAM_P2P = 0.38
G1 = 0.015

ENMAP = [("Nevada", 18.0, 0.186), ("Groningen", 5.0, 0.811), ("Tyrol", 3.0, 0.728)]


def reflectance(fine, albedo=0.30):
    r = 0.10 + 0.55 / (1.0 + np.exp(-(fine - 720.0) / 140.0))
    r -= 0.10 * np.exp(-0.5 * ((fine - 2200.0) / 90.0) ** 2)
    r = np.clip(r, 0.03, 0.75)
    return r * (albedo / 0.30)


def transmittance(fine, target=None, depth=None):
    t = np.ones_like(fine)
    for c, s, d in OTHER:
        t -= d * np.exp(-0.5 * ((fine - c) / s) ** 2)
    for name, (c, s, d, _) in LINES.items():
        dd = depth if (target is not None and name == target) else d
        t -= dd * np.exp(-0.5 * ((fine - c) / s) ** 2)
    return np.clip(t, 0.02, 1.0)


def gaussian_srf(fine, center, fwhm):
    return np.exp(-0.5 * ((fine - center) / (fwhm / 2.3548)) ** 2)


def band_matrix(wl, fwhm, fine):
    M = np.zeros((len(wl), len(fine)))
    for b, w in enumerate(wl):
        s = gaussian_srf(fine, w, fwhm)
        M[b] = s / s.sum()
    return M


def h_band(wl, fwhm, fine, RT):
    hf = np.abs(np.gradient(RT, fine)) / (RT + 1e-12)
    return band_matrix(wl, fwhm, fine) @ hf


def build_line(name, depth, albedo=0.30):
    c, s, d0, sensor = LINES[name]
    cfg = SENSOR[sensor]
    fine = FINE[(FINE >= cfg["lo"] - 60) & (FINE <= cfg["hi"] + 60)]
    wl = np.arange(cfg["lo"], cfg["hi"], cfg["sampling"])
    RT = reflectance(fine, albedo=albedo) * transmittance(fine, target=name, depth=depth)
    L0 = band_matrix(wl, cfg["fwhm"], fine) @ RT
    h = h_band(wl, cfg["fwhm"], fine, RT)
    return wl, L0, h


def smooth_pattern(x_cols, seed, rms=1.0, deg=5):
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, x_cols)
    p = np.zeros_like(u)
    for n in range(deg + 1):
        p += rng.standard_normal() * np.polynomial.legendre.Legendre.basis(n)(u)
    p -= p.mean()
    p /= (np.std(p) + 1e-12)
    return rms * p


def rho_sv2(obs):
    cm = np.asarray(obs, float)
    cm = cm / (cm.mean(1, keepdims=True) + 1e-12)
    r = cm - np.median(cm, 0, keepdims=True)
    _, S, _ = np.linalg.svd(r, full_matrices=False)
    if S.size < 2 or S[0] < 1e-12:
        return 0.0
    return float(S[1] / S[0])


def sigma_psi(L0, h, wl, center, halfwin=45.0):
    phi = L0 / L0.mean()
    psi = phi * (h - np.median(h))
    m = np.abs(wl - center) <= halfwin
    return float(np.std(psi[m]))


def measured_rho(L0, h, wl, smile_p2p=DLAM_P2P, g1=G1, seed=5):

    x = X_COLS
    lam = (wl - wl.mean()) / (wl.max() - wl.min())
    B1 = (lam - lam.mean()) / (np.abs(lam - lam.mean()).max() + 1e-12)
    c = smooth_pattern(x, seed=41, rms=g1 / np.sqrt(2), deg=3)
    g = smooth_pattern(x, seed=11, rms=g1 / np.sqrt(2), deg=4)
    dlam = smooth_pattern(x, seed=seed, rms=1.0, deg=4)
    dlam *= smile_p2p / (dlam.max() - dlam.min())
    obs = L0[None, :] * (1.0 + np.outer(c, B1) + g[:, None])
    obs += dlam[:, None] * (L0 * h)[None, :]
    return rho_sv2(obs)


def main():
    rows = []
    psi_ref = None
    for name in LINES:
        c = LINES[name][0]
        for depth in DEPTHS:
            wl, L0, h = build_line(name, depth)
            sp = sigma_psi(L0, h, wl, c)
            rows.append(dict(line=name, depth=depth, center=c, sigma_psi=sp,
                             dlam_thresh=0.1 * G1 / sp if sp > 0 else np.nan))
            if name == "O$_2$-A 760" and depth == 0.05:
                psi_ref = sp


    alb_rows = []
    for albedo in (0.03, 0.10, 0.30, 0.60):
        wl, L0, h = build_line("O$_2$-A 760", 0.60, albedo=albedo)
        alb_rows.append(dict(albedo=albedo,
                             sigma_psi=sigma_psi(L0, h, wl, 760.0)))


    csv_path = os.path.join(OUT, "threshold_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "line", "depth", "center_nm", "sigma_psi",
                    "dlam_thresh_nm", "albedo", "contrast", "snr",
                    "rho_med", "rho_p95"])
        for r in rows:
            w.writerow(["line_depth", r["line"], r["depth"], r["center"],
                        f"{r['sigma_psi']:.5f}", f"{r['dlam_thresh']:.3f}",
                        "", "", "", "", ""])
        for r in alb_rows:
            w.writerow(["albedo", "O$_2$-A 760", 0.60, 760.0,
                        f"{r['sigma_psi']:.5f}", "", r["albedo"], "", "", "", ""])
    print("[csv]", csv_path)


    tr = list(csv.DictReader(open(os.path.join(OUT, "threshold_transferability.csv"),
                                  encoding="utf-8")))
    z = np.load(os.path.join(OUT, "threshold_curves.npz"))
    amps = z["amps"]


    styles = {"O$_2$-B 688": dict(color="#0072B2", marker="o", ms=3.4),
              "O$_2$-A 760": dict(color="#D55E00", marker="s", ms=3.4),
              "H$_2$O 940": dict(color="#009E73", marker="^", ms=3.6),
              "H$_2$O 1650": dict(color="#E69F00", marker="v", ms=3.6),
              "H$_2$O 1130": dict(color="#CC79A7", marker="D", ms=5.4,
                                  mfc="none", mew=1.1)}
    order = ["O$_2$-B 688", "O$_2$-A 760", "H$_2$O 940",
             "H$_2$O 1650", "H$_2$O 1130"]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))

    ax = axes[0]
    for name in order:
        sel = sorted([r for r in tr if r["line"] == name], key=lambda r: float(r["depth"]))
        ax.plot([float(r["depth"]) for r in sel],
                [float(r["A_star_nm"]) for r in sel], ls="-", lw=1.0,
                label=name, **styles[name])
    ax.set_xlabel("line depth $d$")
    ax.set_ylabel(r"$A^*$ at $\rho=0.1$ (nm, p2p)")
    ax.set_yscale("log")
    ax = axes[1]
    missing = []
    for name in order:


        key = ("%s|0.30" % name).replace("|", "_").replace("$", "").replace("_2", "2")
        if key in z:
            ax.plot(amps, z[key], ls="-", lw=1.0, label=name, **styles[name])
        else:
            missing.append(key)
    if missing:
        raise KeyError("threshold_curves.npz missing keys: %s" % missing)
    ax.axhline(0.1, color="red", ls="--", lw=1.0)
    ax.axhline(0.02, color="darkred", ls="-", lw=1.1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("injected smile p2p (nm)")
    ax.set_ylabel(r"$\rho = s_2/s_1$")


    handles, labels = axes[0].get_legend_handles_labels()
    idx = [labels.index(n) for n in LINES]
    fig.legend([handles[i] for i in idx], [labels[i] for i in idx], fontsize=7.6,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               borderaxespad=0.0, frameon=True)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.tick_params(labelsize=8.5)
        ax.set_title(tag, fontsize=9, loc="left")
    fig.subplots_adjust(wspace=0.36, left=0.11, right=0.99, top=0.80, bottom=0.20)
    png = os.path.join(FIGS, "fig9_threshold_sensitivity.png")
    fig.savefig(png, dpi=600)
    fig.savefig(png.replace(".png", ".pdf"))
    print("[fig]", png)

    print("\n--- line/depth ---")
    for r in rows:
        print(f"  {r['line']:12s} d={r['depth']:.2f}  sigma_psi={r['sigma_psi']:.5f}  "
              f"dlam*={r['dlam_thresh']:.3f} nm")
    print("psi_ref:", f"{psi_ref:.5f}")
    print("\n--- albedo invariance ---")
    for r in alb_rows:
        print(f"  albedo={r['albedo']:.2f}  sigma_psi={r['sigma_psi']:.5f}")


if __name__ == "__main__":
    main()
