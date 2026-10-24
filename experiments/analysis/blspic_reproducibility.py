# -*- coding: utf-8 -*-

import os, sys, csv, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sensors import load_enmap_l1b_vnir            # noqa: E402
from io_data import column_mean                    # noqa: E402
from atmosphere import VNIR_LINES                  # noqa: E402
from model import BLSPIC                           # noqa: E402

HDR = os.path.join(
    ROOT, "Nevada_USA_L1B",
    "ENMAP01-____L1B-DT0000025905_20230707T192008Z_001_V010506_20260528T200634Z",
    "ENMAP01-____L1B-DT0000025905_20230707T192008Z_001_V010506_20260528T200634Z-SPECTRAL_IMAGE_VNIR.HDR")
OUT = os.path.join(ROOT, "experiments", "out")
ITERS = 600

cube, wl, fwhm = load_enmap_l1b_vnir(HDR, max_rows=None)
rows = []
for nr in (200, 400, 1024):
    cm = column_mean(cube[:nr], method="trimmed")
    good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)
    for seed in (0, 1, 2):
        torch.manual_seed(seed); np.random.seed(seed)
        m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES)
        t0 = time.time()
        res = m.fit(cm, iters=ITERS, learn_atm=True, low_rank=2, o2_only_anchor=True)
        d = res["dlam"]; p2p = float(d.max() - d.min())
        print(f"rows={nr:5d} seed={seed}  p2p={p2p:6.2f} nm  mean={d.mean():+6.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(rows=nr, seed=seed, p2p=p2p, mean=float(d.mean()), iters=ITERS))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "blspic_reproducibility.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
