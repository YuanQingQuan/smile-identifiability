# -*- coding: utf-8 -*-

import csv
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'analysis'))

CSV = os.path.join(HERE, 'out', 'noise_referenced_rho.csv')
before = hashlib.md5(io.open(CSV, 'rb').read()).hexdigest()

import noise_referenced_rho as m                      # noqa: E402

cached = {r['scene']: r for r in csv.DictReader(io.open(CSV, encoding='utf-8'))}


def fake(name):
    out = {}
    for k, v in cached[name].items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


m.analyse_scene = fake
m.main()

after = hashlib.md5(io.open(CSV, 'rb').read()).hexdigest()
print('csv unchanged by the replay:', before == after, '(%s)' % after[:8])
