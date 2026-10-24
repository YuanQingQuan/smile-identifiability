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
ROWS, ITERS, SEED = 200, 150, 0
CONFIGS = [
    ("full default",        dict(low_rank=0, lam_r_lam=3e-3, lam_r_x=3e-3)),
    ("full strong smooth",  dict(low_rank=0, lam_r_lam=3e-2, lam_r_x=3e-2)),
    ("full very strong",    dict(low_rank=0, lam_r_lam=3e-1, lam_r_x=3e-1)),
    ("full extreme",        dict(low_rank=0, lam_r_lam=3.0,  lam_r_x=3.0)),
    ("full strong spectral", dict(low_rank=0, lam_r_lam=3e-1, lam_r_x=3e-3)),
    ("full strong x-track", dict(low_rank=0, lam_r_lam=3e-3, lam_r_x=3e-1)),
    ("K=2 strong x-track (ref)", dict(low_rank=2, lam_r_lam=3e-3, lam_r_x=3e-1)),
]


def find_hdr(scene_dir):
    for r, _d, files in os.walk(scene_dir):
        for f in files:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(scene_dir)


def main():
    sdir = os.path.join(ROOT, "Nevada_USA_L1B")
    cube, wl, fwhm = load_enmap_l1b_vnir(find_hdr(sdir), max_rows=None)
    cm = column_mean(cube[:ROWS], method="trimmed")
    good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)
    ref = load_enmap_truth_curve(sdir, target_nm=O2A_TRUTH_TARGET_NM,
                                 n_est=cm.shape[0])
    t = np.asarray(ref["dlam"], float)
    rows = []
    for label, kw in CONFIGS:
        torch.manual_seed(SEED); np.random.seed(SEED)
        torch.cuda.empty_cache()
        m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES, device=DEVICE)
        t0 = time.time()
        res = m.fit(cm, iters=ITERS, learn_atm=True, verbose=False,
                    o2_only_anchor=True, **kw)
        d = np.asarray(res["dlam"], float)
        L = min(len(d), len(t))
        p2p = float(d.max() - d.min())
        rows.append(dict(config=label, est_p2p_nm=round(p2p, 2),
                         ref_p2p_nm=round(float(t[:L].max() - t[:L].min()), 3),
                         est_to_ref=round(p2p / (t[:L].max() - t[:L].min()), 1),
                         lam_r_lam=kw["lam_r_lam"], lam_r_x=kw["lam_r_x"],
                         low_rank=kw["low_rank"]))
        print("%-26s p2p=%6.2f nm  ratio=%5.1fx  (%.0fs)"
              % (label, p2p, rows[-1]["est_to_ref"], time.time() - t0), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fullrank_sweep.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
