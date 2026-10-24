# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from sensors import load_enmap_l1b_vnir            # noqa: E402
from io_data import column_mean                    # noqa: E402
from atmosphere import VNIR_LINES                  # noqa: E402
from model import BLSPIC                           # noqa: E402
from enmap_ground_truth import load_enmap_truth_curve, O2A_TRUTH_TARGET_NM  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = {
    "Nevada": os.path.join(ROOT, "Nevada_USA_L1B"),
    "Groningen": os.path.join(ROOT, "Groningen_Netherlands_L1B"),
    "Tyrol": os.path.join(ROOT, "Tyrol_Austria_L1B"),
}
ROWS, ITERS, SEED = 200, 800, 0


def find_hdr(scene_dir):
    for r, _d, files in os.walk(scene_dir):
        for f in files:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(scene_dir)


def main():
    arrays, rows_out = {}, []
    for name, sdir in SCENES.items():
        hdr = find_hdr(sdir)
        cube, wl, fwhm = load_enmap_l1b_vnir(hdr, max_rows=None)
        cm = column_mean(cube[:ROWS], method="trimmed")
        good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)
        torch.manual_seed(SEED); np.random.seed(SEED)
        torch.cuda.empty_cache()
        m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES, device=DEVICE)
        res = m.fit(cm, iters=ITERS, learn_atm=True, low_rank=2, o2_only_anchor=True)
        d = np.asarray(res["dlam"], float)

        ref = load_enmap_truth_curve(sdir, target_nm=O2A_TRUTH_TARGET_NM,
                                     n_est=len(d))
        t = np.asarray(ref["dlam"], float)
        L = min(len(d), len(t))
        d, t = d[:L], t[:L]
        dc, tc = d - d.mean(), t - t.mean()
        r_raw = float(np.corrcoef(tc, dc)[0, 1])
        sign = 1.0 if r_raw >= 0 else -1.0
        r_abs = abs(r_raw)
        p2p_d = float(d.max() - d.min())
        p2p_t = float(t.max() - t.min())
        print(f"{name:10s} est p2p={p2p_d:6.2f} nm  ref p2p={p2p_t:.3f} nm  "
              f"ratio={p2p_d / p2p_t:5.1f}x  R={r_raw:+.3f}  band={ref['wavelength_nm']:.2f} nm",
              flush=True)
        arrays[f"{name}_dlam"] = d
        arrays[f"{name}_ref"] = t
        rows_out.append(dict(scene=name, rows=ROWS, iters=ITERS, seed=SEED,
                             est_p2p_nm=p2p_d, ref_p2p_nm=p2p_t,
                             est_to_ref=p2p_d / p2p_t,
                             shape_R_raw=r_raw, shape_R_abs=r_abs, sign=sign,
                             ref_band_nm=ref["wavelength_nm"], n_cols=L))

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "xtrack_curves.npz"), **arrays)
    with open(os.path.join(OUT, "xtrack_curves.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
