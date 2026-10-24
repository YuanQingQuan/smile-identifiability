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
SCENE = os.path.join(
    ROOT, "Nevada_USA_L1B",
    "ENMAP01-____L1B-DT0000025905_20230707T192008Z_001_V010506_20260528T200634Z",
    "ENMAP01-____L1B-DT0000025905_20230707T192008Z_001_V010506_20260528T200634Z-SPECTRAL_IMAGE_VNIR.HDR")
ITERS = (200, 800, 2400)


def main():
    cube, wl, fwhm = load_enmap_l1b_vnir(SCENE, max_rows=None)
    cm = column_mean(cube, method="trimmed")
    good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)
    rows = []
    for iters in ITERS:
        torch.manual_seed(0); np.random.seed(0)
        torch.cuda.empty_cache()
        m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES, device=DEVICE)
        t0 = time.time()
        res = m.fit(cm, iters=iters, learn_atm=True, low_rank=2, o2_only_anchor=True)
        d = res["dlam"]
        p2p = float(d.max() - d.min())
        print(f"iters={iters:5d}  p2p={p2p:6.3f} nm  loss={res['loss']:.3e}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(scene="Nevada", rows=1024, seed=0, iters=iters,
                         p2p_nm=p2p, loss=res["loss"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "iteration_drift.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", os.path.join(OUT, "iteration_drift.csv"))


if __name__ == "__main__":
    main()
