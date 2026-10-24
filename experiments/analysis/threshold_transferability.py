# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from threshold_sensitivity import (LINES, SENSOR, FINE, build_line,  # noqa: E402
                                   smooth_pattern, rho_sv2)

OUT = os.path.join(ROOT, "experiments", "out")
X_COLS = 1000
G1 = 0.015
DEPTHS = (0.05, 0.15, 0.30, 0.60)
AMPS = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])


def rho_at(L0, h, wl, smile_p2p, seed=5):
    lam = (wl - wl.mean()) / (wl.max() - wl.min())
    B1 = (lam - lam.mean()) / (np.abs(lam - lam.mean()).max() + 1e-12)
    c = smooth_pattern(X_COLS, seed=41, rms=G1 / np.sqrt(2), deg=3)
    g = smooth_pattern(X_COLS, seed=11, rms=G1 / np.sqrt(2), deg=4)
    dlam = smooth_pattern(X_COLS, seed=seed, rms=1.0, deg=4)
    dlam *= smile_p2p / (dlam.max() - dlam.min())
    obs = L0[None, :] * (1.0 + np.outer(c, B1) + g[:, None])
    obs += dlam[:, None] * (L0 * h)[None, :]
    return rho_sv2(obs)


def main():
    rows, curves = [], {}
    for name in LINES:
        for depth in DEPTHS:
            wl, L0, h = build_line(name, depth)
            rhos = np.array([rho_at(L0, h, wl, A) for A in AMPS])
            if rhos.max() < 0.1:
                A_star = float("nan")
            else:
                idx = int(np.argmax(rhos >= 0.1))
                if idx == 0:
                    A_star = float(AMPS[0])
                else:
                    x0, x1 = np.log(AMPS[idx - 1]), np.log(AMPS[idx])
                    y0, y1 = rhos[idx - 1], rhos[idx]
                    A_star = float(np.exp(x0 + (0.1 - y0) * (x1 - x0) / (y1 - y0)))
            rows.append(dict(line=name, depth=depth,
                             A_star_nm=round(A_star, 3) if np.isfinite(A_star) else "",
                             **{("rho_A%.2f" % A): round(r, 4)
                                for A, r in zip(AMPS, rhos)}))
            curves["%s|%.2f" % (name, depth)] = rhos
            print("%-12s depth=%.2f  A*(rho=0.1)=%.3f nm  rho(0.38)=%.4f  rho(3nm)=%.3f"
                  % (name, depth, A_star,
                     rhos[list(AMPS).index(0.3)], rhos[list(AMPS).index(3.0)]),
                  flush=True)

    inv = []
    for alb in (0.03, 0.10, 0.30, 0.60):
        wl, L0, h = build_line("O$_2$-A 760", 0.60, albedo=alb)
        inv.append((alb, rho_at(L0, h, wl, 3.0)))
    print("albedo invariance rho(3 nm):", [(a, round(r, 3)) for a, r in inv], flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "threshold_transferability.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez(os.path.join(OUT, "threshold_curves.npz"), amps=AMPS,
             **{k.replace("|", "_").replace("$", "").replace("_2", "2"): v
                for k, v in curves.items()})
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
