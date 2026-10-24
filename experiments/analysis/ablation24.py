# -*- coding: utf-8 -*-

import os, sys, csv, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from sensors import load_enmap_l1b_vnir            # noqa: E402
from io_data import column_mean                    # noqa: E402
from atmosphere import VNIR_LINES                  # noqa: E402
from model import BLSPIC                           # noqa: E402
from enmap_ground_truth import (load_enmap_truth_curve,  # noqa: E402
                                O2A_TRUTH_TARGET_NM)

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = ["Nevada_USA_L1B", "Groningen_Netherlands_L1B", "Tyrol_Austria_L1B"]
ROWS, ITERS, SEED = 200, 150, 0
CONFIGS = [
    ("default_K2",        dict(low_rank=2, lam_r_x=3e-3, shared_reflectance=False, o2_only_anchor=True)),
    ("K4",                dict(low_rank=4, lam_r_x=3e-3, shared_reflectance=False, o2_only_anchor=True)),
    ("K8",                dict(low_rank=8, lam_r_x=3e-3, shared_reflectance=False, o2_only_anchor=True)),
    ("full_rank",         dict(low_rank=0, lam_r_x=3e-3, shared_reflectance=False, o2_only_anchor=True)),
    ("strong_reg",        dict(low_rank=2, lam_r_x=3e-2, shared_reflectance=False, o2_only_anchor=True)),
    ("weak_reg",          dict(low_rank=2, lam_r_x=3e-4, shared_reflectance=False, o2_only_anchor=True)),
    ("shared_refl",       dict(low_rank=2, lam_r_x=3e-3, shared_reflectance=True,  o2_only_anchor=True)),
    ("five_line_anchor",  dict(low_rank=2, lam_r_x=3e-3, shared_reflectance=False, o2_only_anchor=False)),
]


def find_hdr(scene_dir):
    for r, _d, files in os.walk(scene_dir):
        for f in files:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(scene_dir)


def main():
    rows_out = []
    for scene in SCENES:
        sdir = os.path.join(ROOT, scene)
        name = scene.split("_")[0]
        hdr = find_hdr(sdir)
        cube, wl, fwhm = load_enmap_l1b_vnir(hdr, max_rows=None)
        cm = column_mean(cube[:ROWS], method="trimmed")
        good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)
        ref = load_enmap_truth_curve(sdir, target_nm=O2A_TRUTH_TARGET_NM,
                                     n_est=cm.shape[0])
        t = np.asarray(ref["dlam"], float)
        for label, kw in CONFIGS:
            torch.manual_seed(SEED); np.random.seed(SEED)
            torch.cuda.empty_cache()
            m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES, device=DEVICE)
            t0 = time.time()
            res = m.fit(cm, iters=ITERS, learn_atm=True, verbose=False, **kw)
            d = np.asarray(res["dlam"], float)
            L = min(len(d), len(t))
            dc, tc = d[:L] - d[:L].mean(), t[:L] - t[:L].mean()
            R = float(np.corrcoef(tc, dc)[0, 1]) if dc.std() > 0 else 0.0
            p2p = float(d.max() - d.min())
            ref_p2p = float(t[:L].max() - t[:L].min())
            rows_out.append(dict(scene=name, config=label, rows=ROWS,
                                 iters=ITERS, seed=SEED,
                                 est_p2p_nm=round(p2p, 2),
                                 ref_p2p_nm=round(ref_p2p, 3),
                                 est_to_ref=round(p2p / ref_p2p, 1),
                                 shape_R=round(R, 3),
                                 loss=float(res["loss"])))
            print("%-10s %-16s p2p=%6.2f nm  ratio=%5.1fx  R=%+.3f  (%4.0fs)"
                  % (name, label, p2p, p2p / ref_p2p, R, time.time() - t0),
                  flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ablation24.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
