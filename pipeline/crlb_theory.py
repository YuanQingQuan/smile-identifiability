# -*- coding: utf-8 -*-

import numpy as np

_O2A_DEPTH = 0.6
_O2A_SIGMA_NM = 1.8


def _effective_width_depth(fwhm_srf_nm, line_depth, line_sigma):
    sigma_s = fwhm_srf_nm / 2.3548
    sigma_eff = np.sqrt(line_sigma ** 2 + sigma_s ** 2)
    d_eff = line_depth * line_sigma / sigma_eff
    return sigma_eff, d_eff, 2.3548 * sigma_eff


def numerical_fisher_info(fwhm_srf_nm, sampling_nm, line_depth=_O2A_DEPTH,
                          line_sigma=_O2A_SIGMA_NM, reflectance=0.3):

    sigma_eff, d_eff, fwhm_eff = _effective_width_depth(
        fwhm_srf_nm, line_depth, line_sigma)
    lam = np.arange(-3 * sigma_eff, 3 * sigma_eff + 1e-9, max(sampling_nm, 1e-6))
    n_eff = len(lam)

    deriv = d_eff * lam / sigma_eff ** 2 * np.exp(-0.5 * (lam / sigma_eff) ** 2)
    fisher_per_snr2 = float(np.sum(deriv ** 2))
    sigma_crlb_times_snr = 1.0 / np.sqrt(max(fisher_per_snr2, 1e-300))
    p2p_floor_times_snr = 2.0 * sigma_crlb_times_snr
    return {
        "sigma_eff_nm": sigma_eff,
        "fwhm_eff_nm": fwhm_eff,
        "d_eff": d_eff,
        "n_eff": n_eff,
        "fisher_per_snr2": fisher_per_snr2,
        "sigma_crlb_times_snr": float(sigma_crlb_times_snr),
        "p2p_floor_times_snr": float(p2p_floor_times_snr),
    }


def closed_form_crlb(fwhm_srf_nm, sampling_nm, line_depth=_O2A_DEPTH,
                     line_sigma=_O2A_SIGMA_NM):

    sigma_eff, d_eff, fwhm_eff = _effective_width_depth(
        fwhm_srf_nm, line_depth, line_sigma)
    sigma_crlb_times_snr = np.sqrt(2 * sigma_eff * sampling_nm / np.sqrt(np.pi)) / d_eff
    floor_p2p_times_snr = 2.0 * sigma_crlb_times_snr
    C = floor_p2p_times_snr / fwhm_eff
    return {
        "sigma_crlb_times_snr": float(sigma_crlb_times_snr),
        "floor_p2p_times_snr": float(floor_p2p_times_snr),
        "C_closed": float(C),
        "sigma_eff_nm": sigma_eff,
        "fwhm_eff_nm": fwhm_eff,
        "d_eff": d_eff,
    }


def phase_averaged_fisher_info(fwhm_srf_nm, sampling_nm, line_depth=_O2A_DEPTH,
                               line_sigma=_O2A_SIGMA_NM, n_phase=128):

    sigma_eff, d_eff, fwhm_eff = _effective_width_depth(
        fwhm_srf_nm, line_depth, line_sigma)
    n_half = int(np.ceil(4.0 * sigma_eff / max(sampling_nm, 1e-6)))
    ks = np.arange(-n_half, n_half + 1)
    fisher = 0.0
    for phi in np.linspace(0.0, sampling_nm, n_phase, endpoint=False):
        lam = ks * sampling_nm + phi
        deriv = d_eff * lam / sigma_eff ** 2 * np.exp(-0.5 * (lam / sigma_eff) ** 2)
        fisher += float(np.sum(deriv ** 2))
    fisher /= n_phase
    sigma_crlb_times_snr = 1.0 / np.sqrt(max(fisher, 1e-300))
    return {
        "sigma_eff_nm": sigma_eff,
        "fwhm_eff_nm": fwhm_eff,
        "d_eff": d_eff,
        "fisher_per_snr2": fisher,
        "sigma_crlb_times_snr": float(sigma_crlb_times_snr),
        "p2p_floor_times_snr": float(2.0 * sigma_crlb_times_snr),
    }


def calibrate_floor_constant(snr=300):

    print(f"[CRLB] floor constant C (p2p = C*FWHM_eff/SNR), SNR={snr}")
    print(f"{'fwhm_srf':>9s} {'sampling':>8s} {'fwhm_eff':>8s} "
          f"{'sigma_crlb':>10s} {'floor_p2p':>9s} {'C':>8s}")
    Cs = []
    for fwhm_s in [1.0, 2.0, 3.0, 6.5, 10.0]:
        for samp in [1.0, 5.0, 10.0]:
            if samp > fwhm_s * 1.5:
                continue
            info = numerical_fisher_info(fwhm_s, samp)
            floor = info["p2p_floor_times_snr"] / snr
            C = info["p2p_floor_times_snr"] / info["fwhm_eff_nm"]
            Cs.append(C)
            print(f"{fwhm_s:9.1f} {samp:8.1f} {info['fwhm_eff_nm']:8.2f} "
                  f"{info['sigma_crlb_times_snr']/snr:10.4f} {floor:9.4f} {C:8.3f}")
    C_med = float(np.median(Cs))
    print(f"\n=> median C = {C_med:.2f} (dimensionless)")
    return C_med


if __name__ == "__main__":
    calibrate_floor_constant(snr=300)
    print("\n[C stability vs SNR] (fwhm=6.5, samp=5.0) -- must be constant")
    for snr in [50, 100, 300, 500]:
        info = numerical_fisher_info(6.5, 5.0)
        floor = info["p2p_floor_times_snr"] / snr
        C = info["p2p_floor_times_snr"] / info["fwhm_eff_nm"]
        print(f"  SNR={snr:4d}: floor={floor:.4f} nm  C={C:.3f}")
