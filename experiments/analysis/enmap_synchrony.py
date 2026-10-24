# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from sensors import load_enmap_l1b_vnir          # noqa: E402
from io_data import column_mean                  # noqa: E402
from ramp_physics import _line_localization      # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
SCENES = {"Nevada": "Nevada_USA_L1B",
          "Groningen": "Groningen_Netherlands_L1B",
          "Tyrol": "Tyrol_Austria_L1B"}
ANCHOR = 760.0
LINES = [688.0, 720.0, 760.0, 820.0, 940.0]


def find_hdr(d):
    for r, _x, fs in os.walk(d):
        for f in fs:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(r, f)
    raise FileNotFoundError(d)


def main():
    rows = []
    for name, d in SCENES.items():
        cube, wl, _f = load_enmap_l1b_vnir(find_hdr(os.path.join(ROOT, d)), max_rows=None)
        cm = column_mean(cube, method="trimmed")
        curves = {}
        for c in LINES:
            if c < wl.min() or c > wl.max():
                continue
            pc = _line_localization(cm, wl, c, search_hw=25.0)
            curves[c] = pc - np.nanmean(pc)
        a = curves[ANCHOR]
        cors = {}
        for c, v in curves.items():
            if c == ANCHOR:
                continue
            cors[c] = float(np.corrcoef(v, a)[0, 1])
        vals = list(cors.values())
        s_sync = float(np.median(vals))
        rows.append(dict(scene=name,
                         **{("corr_%.0f" % c): round(v, 3) for c, v in cors.items()},
                         S_sync=round(s_sync, 3)))
        print("%-10s " % name + "  ".join("%.0f: %+.2f" % (c, v)
                                          for c, v in cors.items())
              + "   S_sync=%.3f" % s_sync, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "enmap_synchrony.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
