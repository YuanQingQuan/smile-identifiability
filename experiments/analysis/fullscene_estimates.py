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

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = {
    "Nevada": os.path.join(ROOT, "Nevada_USA_L1B"),
    "Groningen": os.path.join(ROOT, "Groningen_Netherlands_L1B"),
    "Tyrol": os.path.join(ROOT, "Tyrol_Austria_L1B"),
}
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
ROWS_SEEDS = [(200, (0,)), (400, (0,)), (1024, (0, 1, 2))]


def find_hdr(scene_dir):
    for r, _d, files in os.walk(scene_dir):
        for f in files:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(scene_dir)


def main():
    rows_out = []
    for name, sdir in SCENES.items():
        hdr = find_hdr(sdir)
        cube, wl, fwhm = load_enmap_l1b_vnir(hdr, max_rows=None)
        for nr, seeds in ROWS_SEEDS:
            cm = column_mean(cube[:nr], method="trimmed")
            good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                torch.cuda.empty_cache()
                m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES, device=DEVICE)
                t0 = time.time()
                res = m.fit(cm, iters=ITERS, learn_atm=True, low_rank=2,
                            o2_only_anchor=True)
                d = res["dlam"]
                p2p = float(d.max() - d.min())
                print(f"{name:10s} rows={nr:5d} seed={seed}  p2p={p2p:6.2f} nm  "
                      f"mean={d.mean():+6.2f}  ({time.time()-t0:.0f}s)", flush=True)
                rows_out.append(dict(scene=name, rows=nr, seed=seed, iters=ITERS,
                                     p2p_nm=p2p, mean_nm=float(d.mean()),
                                     n_cols=cm.shape[0], n_bands=cm.shape[1]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fullscene_estimates.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
