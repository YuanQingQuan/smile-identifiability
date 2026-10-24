# -*- coding: utf-8 -*-

import numpy as np


def _finite(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def _p2p(a):
    v = _finite(a)
    return float(v.max() - v.min()) if v.size else float("nan")


def _median_sampling_nm(wl):
    wl = _finite(wl)
    if wl.size < 2:
        return float("nan")
    return float(np.nanmedian(np.diff(np.sort(wl))))


def estimate_snr(cm, cube=None, window=5):

    if cube is not None:
        try:
            from noise_est import snr_combined
            snr, _ = snr_combined(cm=cm, cube=cube)
            if np.isfinite(snr) and snr > 0:
                return float(snr)
        except Exception:
            pass
    cm = np.asarray(cm, float)
    if cm.ndim != 2 or min(cm.shape) < 5:
        return float("nan")
    mean_spec = np.nanmedian(cm, axis=0)
    signal = max(float(np.nanmedian(np.abs(mean_spec))), 1e-9)
    dev = cm - mean_spec[None, :]
    noise = max(float(np.nanmedian(np.abs(dev)) / 0.6745), 1e-6)
    return signal / noise


_O2A_LINE_FWHM_NM = 2.3548 * 1.8


def expected_min_p2p_nm(sampling_nm, snr, fwhm_nm=None):

    if not np.isfinite(sampling_nm) or sampling_nm <= 0:
        return float("nan")
    snr_eff = np.clip(snr if np.isfinite(snr) else 100.0, 30.0, 1000.0)
    if fwhm_nm is not None and np.isfinite(fwhm_nm) and fwhm_nm > 0:
        eff_fwhm = float(np.sqrt(_O2A_LINE_FWHM_NM ** 2 + fwhm_nm ** 2))
        ratio = sampling_nm / max(eff_fwhm, 0.1)
        C = 1.2 + 2.8 * min(ratio, 1.0)
        noise_floor = C * eff_fwhm / np.sqrt(snr_eff)
    else:
        noise_floor = 15.0 / np.sqrt(snr_eff)
    samp_floor = sampling_nm / 3.5 if sampling_nm >= 2.0 else 0.0
    return float(max(noise_floor, samp_floor))


def audit_identifiability(cm, wl, fwhm, estimated_dlam=None, baseline_dlam=None,
                          max_physical_p2p_nm=8.0, truth_dlam=None, cube=None):

    wl = np.asarray(wl, float)
    fwhm = np.asarray(fwhm, float)
    sampling_nm = _median_sampling_nm(wl)
    snr = estimate_snr(cm, cube=cube)
    sensor_fwhm_nm = float(np.nanmedian(fwhm)) if fwhm.size else None
    min_p2p = expected_min_p2p_nm(sampling_nm, snr, fwhm_nm=sensor_fwhm_nm)
    dlam_p2p = _p2p(estimated_dlam) if estimated_dlam is not None else float("nan")
    baseline_p2p = _p2p(baseline_dlam) if baseline_dlam is not None else float("nan")


    ramp_gain_info = {}
    ramp_class = "unknown"
    ramp_bound = min_p2p
    try:
        from ramp_gain import decompose_ramp_gain, gain_striping_score, ramp_score, classify_confound
        rg = decompose_ramp_gain(cm, wl, fwhm)
        gs = gain_striping_score(rg)
        rs = ramp_score(rg)
        ramp_class, _, _ = classify_confound(rg)
        from ramp_gain import ramp_identifiability_bound
        ramp_bound = ramp_identifiability_bound(rg, sensor_fwhm_nm or 10.0, snr)
        ramp_gain_info = {
            "ramp_gain_class": ramp_class,
            "gain_striping_score": float(gs),
            "ramp_score": float(rs),
            "sv_ratio_2": float(rg["sv_ratio_2"]),
            "rank1_fraction": float(rg["rank1_fraction"]),
            "ramp_identifiability_bound_nm": float(ramp_bound),
        }
    except Exception:
        pass

    effective_floor = max(min_p2p, ramp_bound) if np.isfinite(ramp_bound) else min_p2p

    o2a_sigma_nm = 1.8
    o2a_fwhm_nm = 2.3548 * o2a_sigma_nm
    o2a_samples_per_line_fwhm = o2a_fwhm_nm / sampling_nm if sampling_nm > 0 else float("nan")
    if wl.size and fwhm.size == wl.size:
        near_o2 = np.abs(wl - 760.0) <= max(15.0, 2.0 * sampling_nm)
        sensor_fwhm_near_o2 = float(np.nanmedian(fwhm[near_o2])) if np.any(near_o2) else float(np.nanmedian(fwhm))
    else:
        sensor_fwhm_near_o2 = float("nan")
    o2a_sensor_fwhm_over_line = sensor_fwhm_near_o2 / o2a_fwhm_nm if np.isfinite(sensor_fwhm_near_o2) else float("nan")

    reasons = []
    warnings = []
    if np.isfinite(snr) and snr < 50.0:
        reasons.append("low_snr")
    if np.isfinite(o2a_samples_per_line_fwhm) and o2a_samples_per_line_fwhm < 0.75:
        warnings.append("o2a_line_under_sampled")
    if np.isfinite(o2a_sensor_fwhm_over_line) and o2a_sensor_fwhm_over_line > 1.5:
        warnings.append("o2a_line_smeared_by_srf")

    status = "not_evaluated"
    if np.isfinite(dlam_p2p) and np.isfinite(effective_floor):
        if dlam_p2p < 0.95 * effective_floor:

            if ramp_class == "ramp_dominant":
                status = "unidentifiable_ramp"
                reasons.append("spectral_ramp_dominant_unidentifiable")
            else:
                status = "unidentifiable_snr"
                reasons.append("estimated_amplitude_below_detectability_floor")
        elif np.isfinite(max_physical_p2p_nm) and dlam_p2p > max_physical_p2p_nm:
            status = "artifact_dominated"
            reasons.append("estimated_amplitude_above_physical_prior")
            if ramp_class == "ramp_dominant":
                reasons.append("artifact_likely_spectral_ramp")
        elif warnings:
            status = "borderline"
        else:
            status = "recoverable"
    elif reasons:
        status = "unidentifiable_snr" if ramp_class != "ramp_dominant" else "unidentifiable_ramp"

    if np.isfinite(baseline_p2p) and np.isfinite(dlam_p2p) and baseline_p2p > max(dlam_p2p, 1e-9) * 2.0:
        warnings.append("baseline_relative_shift_much_larger_than_blspic")


    truth_metrics = {}
    if truth_dlam is not None and estimated_dlam is not None:
        t = np.asarray(truth_dlam, float); e = np.asarray(estimated_dlam, float)
        L = min(len(t), len(e)); t, e = t[:L], e[:L]
        tc, ec = t - t.mean(), e - e.mean()
        truth_p2p = float(t.max() - t.min())
        if truth_p2p > 1e-9:
            truth_metrics = {
                "truth_p2p_nm": truth_p2p,
                "est_to_truth_amplitude_ratio": float((e.max() - e.min()) / truth_p2p),
                "truth_shape_R": float(np.corrcoef(tc, ec)[0, 1])
                                 if np.std(tc) > 0 and np.std(ec) > 0 else 0.0,
                "truth_rmse_nm": float(np.sqrt(np.mean((tc - ec) ** 2))),
            }
            if truth_metrics["est_to_truth_amplitude_ratio"] > 5.0:
                reasons.append("estimate_confirmed_artifact_by_truth")
                status = "artifact_dominated"

    out = {
        "identifiability_status": status,
        "identifiability_reasons": reasons,
        "identifiability_warnings": warnings,
        "identifiability_sampling_nm": sampling_nm,
        "identifiability_snr_est": float(snr) if np.isfinite(snr) else float("nan"),
        "identifiability_min_detectable_p2p_nm": effective_floor,
        "identifiability_snr_floor_nm": min_p2p,
        "identifiability_ramp_floor_nm": float(ramp_bound) if np.isfinite(ramp_bound) else float("nan"),
        "identifiability_estimated_p2p_nm": dlam_p2p,
        "identifiability_baseline_p2p_nm": baseline_p2p,
        "identifiability_max_physical_p2p_nm": float(max_physical_p2p_nm),
        "identifiability_o2a_samples_per_line_fwhm": float(o2a_samples_per_line_fwhm),
        "identifiability_o2a_sensor_fwhm_over_line": float(o2a_sensor_fwhm_over_line),
    }
    out.update(ramp_gain_info)
    out.update(truth_metrics)
    return out
