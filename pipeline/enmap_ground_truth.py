# -*- coding: utf-8 -*-

import os, sys, csv
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ENMAP_SCENES = {
    "Nevada":   (os.path.join(ROOT, "Nevada_USA_L1B"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "enmap_nevada_result.npz"),
                 (300, 700)),
    "Groningen":(os.path.join(ROOT, "Groningen_Netherlands_L1B"), None, None),
    "Tyrol":    (os.path.join(ROOT, "Tyrol_Austria_L1B"), None, None),
}
O2A_TRUTH_TARGET_NM = 763.7


def _strip_ns(root):
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _find_metadata(scene_dir):
    for root, _dirs, files in os.walk(scene_dir):
        for f in files:
            if f.endswith("METADATA.XML"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"{scene_dir}: METADATA.XML not found")


def parse_enmap_smile_truth(scene_dir, nc=1000):

    m = _find_metadata(scene_dir)
    root = _strip_ns(ET.parse(m).getroot())
    smile = root.find(".//smileCorrection")
    if smile is None:
        raise RuntimeError(f"{scene_dir}: 无 smileCorrection 段")
    vnir = smile.find("VNIR")
    x = np.arange(1, nc + 1, dtype=float)
    out = []
    for b in vnir.findall(".//bandID"):
        li = int(b.attrib["number"])
        coeffs = [float(b.findtext(f"coeff{k}")) for k in range(5)]
        wl = float(b.findtext("wavelength"))
        dl = np.polyval(coeffs[::-1], x)
        out.append((li, wl, dl))
    return out


def smile_truth_spectrum(truth_list):

    wl = np.array([t[1] for t in truth_list])
    p2p = np.array([t[2].max() - t[2].min() for t in truth_list])
    return wl, p2p


def select_truth_band(truth_list, target_nm=760.0):

    if not truth_list:
        raise ValueError("empty EnMAP truth list")
    return min(truth_list, key=lambda t: abs(float(t[1]) - float(target_nm)))


def crop_truth_to_estimate(truth_dlam, n_est, crop_start=None):

    truth = np.asarray(truth_dlam, float)
    n_est = int(n_est)
    if n_est <= 0:
        return truth[:0]
    if len(truth) <= n_est:
        return truth[:n_est]
    if crop_start is None:
        crop_start = max((len(truth) - n_est) // 2, 0)
    crop_start = int(crop_start)
    crop_stop = min(crop_start + n_est, len(truth))
    out = truth[crop_start:crop_stop]
    if len(out) < n_est:
        out = truth[-n_est:]
    return out


def load_enmap_truth_curve(scene_dir, target_nm=760.0, n_est=None, crop_start=None, nc=1000):

    truth = parse_enmap_smile_truth(scene_dir, nc=nc)
    local_idx, wl_nm, dlam = select_truth_band(truth, target_nm=target_nm)
    if n_est is not None:
        dlam = crop_truth_to_estimate(dlam, n_est=n_est, crop_start=crop_start)
    return {
        "local_idx": int(local_idx),
        "wavelength_nm": float(wl_nm),
        "dlam": np.asarray(dlam, float),
        "target_nm": float(target_nm),
        "source": _find_metadata(scene_dir),
    }


def validate_against_truth(dlam_est, truth_dlam, crop=None):

    t = np.asarray(truth_dlam, float)
    e = np.asarray(dlam_est, float)
    if crop:
        t = t[crop[0]:crop[1]]; e = e[:len(t)] if len(e) >= len(t) else e
    L = min(len(t), len(e))
    t, e = t[:L], e[:L]
    tc, ec = t - t.mean(), e - e.mean()
    r = float(np.corrcoef(tc, ec)[0, 1]) if np.std(tc) > 0 and np.std(ec) > 0 else 0.0
    rmse = float(np.sqrt(np.mean((tc - ec) ** 2)))
    return {
        "truth_p2p": float(t.max() - t.min()),
        "est_p2p": float(e.max() - e.min()),
        "amplitude_ratio": float((e.max() - e.min()) / max(t.max() - t.min(), 1e-9)),
        "shape_R": r,
        "rmse_nm": rmse,
    }


def run_enmap_truth_validation():

    print("=" * 64)
    print("EnMAP metadata 真值验证")
    print("=" * 64)


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_truth = {}
    rows = []
    for name, (sdir, _, _) in ENMAP_SCENES.items():
        if not os.path.exists(sdir):
            continue
        truth = parse_enmap_smile_truth(sdir)
        all_truth[name] = truth
        wl, p2p = smile_truth_spectrum(truth)
        o2_band = select_truth_band(truth, target_nm=O2A_TRUTH_TARGET_NM)
        rows.append({
            "scene": name,
            "truth_type": "enmap_metadata_smileCorrection",
            "n_vnir_bands": len(truth),
            "mean_p2p_nm": float(np.mean(p2p)),
            "max_p2p_nm": float(np.max(p2p)),
            "max_p2p_wavelength_nm": float(wl[np.argmax(p2p)]),
            "o2a_band_local_idx": int(o2_band[0]),
            "o2a_band_wavelength_nm": float(o2_band[1]),
            "o2a_full_p2p_nm": float(o2_band[2].max() - o2_band[2].min()),
        })
        axes[0].plot(wl, p2p, lw=1.5, label=f"{name} (mean p2p={np.mean(p2p):.3f}nm)")
        print(f"{name}: 真 smile 全波段 p2p mean={np.mean(p2p):.3f} "
              f"max={np.max(p2p):.3f}nm @{wl[np.argmax(p2p)]:.0f}nm", flush=True)

    axes[0].set_xlabel("wavelength (nm)"); axes[0].set_ylabel("true smile p2p (nm)")
    axes[0].set_title("EnMAP true smile (from metadata 4th-order coeffs)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[0].axhline(2.91, color="r", ls="--", lw=1, label="detectability floor")

    axes[0].axvline(760, color="g", ls=":", lw=0.8, alpha=0.5)


    nev_npz = ENMAP_SCENES["Nevada"][1]
    nev_crop = ENMAP_SCENES["Nevada"][2]
    if nev_npz and os.path.exists(nev_npz):
        d = np.load(nev_npz)
        dlam = d["dlam"]
        truth = all_truth.get("Nevada", [])
        o2a = select_truth_band(truth, target_nm=O2A_TRUTH_TARGET_NM)
        cmp = validate_against_truth(dlam, o2a[2], crop=nev_crop)
        rows.append({
            "scene": "Nevada",
            "truth_type": "estimate_vs_metadata_truth_o2a",
            "n_vnir_bands": "",
            "mean_p2p_nm": "",
            "max_p2p_nm": "",
            "max_p2p_wavelength_nm": "",
            "o2a_band_local_idx": int(o2a[0]),
            "o2a_band_wavelength_nm": float(o2a[1]),
            "o2a_full_p2p_nm": float(o2a[2].max() - o2a[2].min()),
            "truth_crop_p2p_nm": cmp["truth_p2p"],
            "est_p2p_nm": cmp["est_p2p"],
            "est_to_truth_amplitude_ratio": cmp["amplitude_ratio"],
            "truth_shape_R": cmp["shape_R"],
            "truth_rmse_nm": cmp["rmse_nm"],
        })
        print(f"\nNevada O2-A 估计 vs 真值:")
        print(f"  真smile p2p={cmp['truth_p2p']:.3f}nm, BL-SPIC p2p={cmp['est_p2p']:.3f}nm")
        print(f"  幅度比={cmp['amplitude_ratio']:.1f}x, 形状R={cmp['shape_R']:.3f}")

        x = np.arange(len(dlam))
        ts = o2a[2][nev_crop[0]:nev_crop[0] + len(dlam)]
        axes[1].plot(x, ts - ts.mean(), "g-", lw=2, label=f"true smile (p2p={cmp['truth_p2p']:.3f}nm)")
        axes[1].plot(x, dlam - dlam.mean(), "r-", lw=1.5,
                     label=f"BL-SPIC estimate (p2p={cmp['est_p2p']:.1f}nm, {cmp['amplitude_ratio']:.0f}x)")
        axes[1].set_xlabel("across-track column"); axes[1].set_ylabel("Δλ (nm, de-meaned)")
        axes[1].set_title(f"Nevada O2-A: estimate vs truth ({cmp['amplitude_ratio']:.0f}x overestimate)")
        axes[1].legend(fontsize=9); axes[1].grid(alpha=.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "enmap_truth_validation.png")
    plt.savefig(out, dpi=140)
    print(f"\n[fig] {out}", flush=True)
    csv_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "enmap_truth_validation.csv")
    if rows:
        fields = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {csv_out}", flush=True)


def _find_enmap_vnir_hdr(scene_dir):

    for root, _dirs, files in os.walk(scene_dir):
        for f in files:
            if f.upper().endswith("SPECTRAL_IMAGE_VNIR.HDR"):
                return os.path.join(root, f)
    return None


def run_enmap_ramp_gain_validation(max_rows=400, blspic_iters=2000):

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sensors import load_enmap_l1b_vnir
    from io_data import column_mean
    from model import BLSPIC
    from baseline import phasecorr_shift
    from identifiability import audit_identifiability, expected_min_p2p_nm
    from ramp_gain import decompose_ramp_gain, gain_striping_score, ramp_score, classify_confound
    from atmosphere import VNIR_LINES

    print("=" * 64)
    print("EnMAP 三景 ramp/gain 参考对比")
    print("=" * 64)

    scenes = {}
    for name in ["Nevada", "Groningen", "Tyrol"]:
        sdir = os.path.join(ROOT, f"{name.split('_')[0] if '_' in name else name}_USA_L1B"
                          if name == "Nevada" else
                          os.path.join(ROOT, {"Nevada": "Nevada_USA_L1B",
                                              "Groningen": "Groningen_Netherlands_L1B",
                                              "Tyrol": "Tyrol_Austria_L1B"}[name]))
        if not os.path.exists(sdir):
            continue
        scenes[name] = sdir

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (name, sdir) in enumerate(scenes.items()):
        hdr = _find_enmap_vnir_hdr(sdir)
        if hdr is None:
            print(f"[skip] {name}: 无 VNIR HDR")
            continue
        print(f"\n[{name}] {os.path.basename(sdir)}", flush=True)
        cube, wl, fwhm = load_enmap_l1b_vnir(hdr, max_rows=max_rows)
        cm = column_mean(cube, method="trimmed")
        good = np.isfinite(cm).all(0) & (np.nanmedian(cm, 0) > 1e-6)


        m = BLSPIC(wl, fwhm, good_mask=good, lines=VNIR_LINES)
        res = m.fit(cm, iters=blspic_iters, learn_atm=True, low_rank=2,
                    o2_only_anchor=True, verbose=False)
        dlam = res["dlam"]
        dlam_p2p = float(dlam.max() - dlam.min())


        truth = parse_enmap_smile_truth(sdir)
        o2a = select_truth_band(truth, target_nm=O2A_TRUTH_TARGET_NM)
        truth_dlam = o2a[2]

        n_est = len(dlam)
        crop_start = max((len(truth_dlam) - n_est) // 2, 0)
        truth_crop = truth_dlam[crop_start:crop_start + n_est]
        truth_p2p = float(truth_crop.max() - truth_crop.min()) if len(truth_crop) else float("nan")


        audit = audit_identifiability(cm, wl, fwhm, estimated_dlam=dlam, cube=cube)
        snr = audit["identifiability_snr_est"]
        floor = audit["identifiability_min_detectable_p2p_nm"]


        rg = decompose_ramp_gain(cm, wl, fwhm)
        gs = gain_striping_score(rg)
        rs = ramp_score(rg)
        cls, _, _ = classify_confound(rg)
        sv2 = rg["sv_ratio_2"]

        ratio = dlam_p2p / truth_p2p if truth_p2p > 1e-9 else float("nan")
        print(f"  SNR={snr:.0f}  真smile={truth_p2p:.3f}nm  BL-SPIC={dlam_p2p:.1f}nm  比={ratio:.0f}x")
        print(f"  ramp_gain={cls}  sv2={sv2:.3f}  floor={floor:.2f}nm  status={audit['identifiability_status']}")

        rows.append({
            "scene": name,
            "snr_pca_residual": round(snr, 0),
            "truth_smile_p2p_nm": round(truth_p2p, 3),
            "blspic_p2p_nm": round(dlam_p2p, 1),
            "est_to_truth_ratio": round(ratio, 0),
            "ramp_gain_class": cls,
            "sv_ratio_2": round(sv2, 3),
            "rank1_fraction": round(rg["rank1_fraction"], 3),
            "gain_score": round(gs, 2),
            "ramp_score": round(rs, 2),
            "floor_eff_nm": round(floor, 2),
            "status": audit["identifiability_status"],
            "reasons": ",".join(audit["identifiability_reasons"]),
        })


        wl_spec, p2p_spec = smile_truth_spectrum(truth)
        axes[0, 0].plot(wl_spec, p2p_spec, lw=1.5, label=f"{name} (mean {np.mean(p2p_spec):.3f}nm)")

        x = np.arange(n_est)
        axes[0, 1].plot(x, dlam - dlam.mean(), "r-", lw=1.5,
                        label=f"BL-SPIC ({dlam_p2p:.1f}nm, {ratio:.0f}x)" if idx == 0 else None)
        if idx == 0:
            tc = truth_crop - truth_crop.mean() if len(truth_crop) == n_est else None
            if tc is not None:
                axes[0, 1].plot(x, tc, "g-", lw=2, label=f"truth ({truth_p2p:.3f}nm)")
            axes[0, 1].axhline(floor, color="b", ls="--", lw=1, label=f"floor ({floor:.1f}nm)")
            axes[0, 1].axhline(-floor, color="b", ls="--", lw=1)
            axes[0, 1].legend(fontsize=8)


    axes[0, 0].set_xlabel("wavelength (nm)")
    axes[0, 0].set_ylabel("true smile p2p (nm)")
    axes[0, 0].set_title("EnMAP metadata true smile spectrum")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].axvline(760, color="g", ls=":", lw=0.8, alpha=0.5)

    axes[0, 1].set_xlabel("across-track column")
    axes[0, 1].set_ylabel("Δλ (nm, de-meaned)")
    axes[0, 1].set_title("BL-SPIC estimate vs metadata truth (Nevada)")
    axes[0, 1].grid(alpha=0.3)


    names = [r["scene"] for r in rows]
    sv2s = [r["sv_ratio_2"] for r in rows]
    colors = ["C0" if r["ramp_gain_class"] == "gain_dominant" else "C3" for r in rows]
    axes[1, 0].bar(names, sv2s, color=colors)
    axes[1, 0].axhline(0.1, color="k", ls="--", lw=1, label="ramp threshold (sv2>0.1)")
    axes[1, 0].set_ylabel("SVD sv2/sv1 ratio")
    axes[1, 0].set_title("ramp/gain rank structure (sv2>0.1 = ramp dominant)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3, axis="y")


    if rows:
        ratios = [r["est_to_truth_ratio"] for r in rows]
        floors = [r["floor_eff_nm"] for r in rows]
        truths = [r["truth_smile_p2p_nm"] for r in rows]
        x = np.arange(len(names))
        w = 0.35
        axes[1, 1].bar(x - w, ratios, w, color="C3", label="BL-SPIC / truth (×)")
        axes[1, 1].bar(x + w, floors, w, color="C0", label="floor (nm)")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(names)
        axes[1, 1].set_ylabel("amplitude ratio / floor (nm)")
        axes[1, 1].set_title("BL-SPIC overestimate ratio vs detectability floor")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.3, axis="y")
        for i, (r_, t_) in enumerate(zip(ratios, truths)):
            axes[1, 1].annotate(f"truth={t_:.3f}nm", (i, 5), ha="center", fontsize=7)

    fig.suptitle("EnMAP spectral smile identifiability: ramp/gain decomposition + metadata truth",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    fig_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "enmap_ramp_gain_validation.png")
    plt.savefig(fig_out, dpi=140, bbox_inches="tight")
    print(f"\n[fig] {fig_out}", flush=True)

    csv_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "enmap_ramp_gain_validation.csv")
    if rows:
        fields = list(rows[0].keys())
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {csv_out}", flush=True)

    print("\n=== 核心结论 ===")
    for r in rows:
        print(f"  {r['scene']}: 真{r['truth_smile_p2p_nm']:.3f}nm vs BL-SPIC {r['blspic_p2p_nm']:.0f}nm "
              f"({r['est_to_truth_ratio']:.0f}x) | {r['ramp_gain_class']} sv2={r['sv_ratio_2']} | {r['status']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ramp_gain":
        run_enmap_ramp_gain_validation()
    else:
        run_enmap_truth_validation()
