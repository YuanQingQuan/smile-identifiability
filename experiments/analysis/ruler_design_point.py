import glob
import os
import re
import sys

import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(EXP)
sys.path.insert(0, os.path.join(ROOT, 'owr'))
from theory.ruler_fisher import fringe_period, optimal_fsr  # noqa: E402

CONTRAST, SNR, L_ENG = 0.03, 300.0, 8400.0


def hdr_arrays(path):
    txt = open(path, errors='ignore').read()
    get = lambda k: np.array([float(x) for x in re.findall(  # noqa: E731
        r'[-+]?\d*\.?\d+', re.search(k + r'\s*=\s*\{([^}]*)\}', txt, re.S).group(1))])
    return get('wavelength'), get('fwhm')


hdr = [x for x in glob.glob(os.path.join(ROOT, 'Nevada_USA_L1B', '*', '*VNIR*.HDR'))][0]
wl, fw = hdr_arrays(hdr)
N = len(wl)
print('EnMAP VNIR: %d bands, %.1f-%.1f nm, FWHM median %.2f nm (mean %.2f)'
      % (N, wl.min(), wl.max(), np.median(fw), fw.mean()))


def broadband(L, use_fwhm=True):
    P = fringe_period(wl, L)
    A = (np.exp(-2.0 * np.pi ** 2 * (fw / 2.3548) ** 2 / P ** 2) if use_fwhm
         else np.ones_like(P))
    terms = (CONTRAST * A * SNR * 2.0 * np.pi / P) ** 2
    return float(np.sqrt(2.0) / np.sqrt(np.sum(terms))), float(np.median(A))


P760 = fringe_period(760.0, L_ENG)
print('engineering spacer: L = %.0f nm -> fringe period at 760 nm = %.2f nm'
      % (L_ENG, P760))
for use in (True, False):
    s, A = broadband(L_ENG, use)
    print('   broadband sigma (SRF attenuation %s) = %.4f nm'
          % ('on' if use else 'off', s))
A760 = np.exp(-2.0 * np.pi ** 2 * (np.median(fw) / 2.3548) ** 2 / P760 ** 2)
s_single = np.sqrt(2.0) * P760 / (2.0 * np.pi * CONTRAST * A760 * SNR * np.sqrt(N))
print('   Eq.(8) single-band form (P=P(760), median FWHM, N=%d) = %.4f nm' % (N, s_single))
Pstar = optimal_fsr(float(np.median(fw)))
print('   analytic optimum for median FWHM: P* = %.2f nm' % Pstar)
print('   sigma at the analytic optimum = %.4f nm'
      % (np.sqrt(2.0) * Pstar / (2.0 * np.pi * CONTRAST * SNR * np.sqrt(N))))
print('\nN = %d bands, FWHM median %.2f nm' % (N, np.median(fw)))
