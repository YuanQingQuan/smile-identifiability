# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'legend.framealpha': 1.0,
    'legend.borderpad': 0.35,
    'legend.labelspacing': 0.35,
    'savefig.bbox': 'tight',

    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figs')
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(HERE, '..', 'pipeline', 'out')


sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))
from crlb_theory import numerical_fisher_info, closed_form_crlb  # noqa: E402


def read_csv(name):
    with open(os.path.join(CSV, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_local(name):
    with open(os.path.join(HERE, 'out', name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _tint(color, t=0.82):

    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (t + (1 - t) * r, t + (1 - t) * g, t + (1 - t) * b)


def save(fig, stem):
    png = os.path.join(OUT, stem + '.png')
    eps = os.path.join(OUT, stem + '.eps')
    pdf = os.path.join(OUT, stem + '.pdf')
    fig.tight_layout(pad=0.5)
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, format='pdf')
    try:
        fig.savefig(eps, format='eps')
    except Exception as e:
        print('  [warn] EPS failed for %s: %s' % (stem, e))
    plt.close(fig)
    print('[fig]', png)


def fig1_crlb_closed_vs_numerical():

    from crlb_theory import _effective_width_depth
    fig, ax = plt.subplots(figsize=(5.3, 3.51))


    sensors = [(3.0, 'PRISMA-like', 'C0', 'o', '-'),
               (6.5, 'EnMAP-like', 'C1', 's', '--'),
               (10.0, 'Hyperion-like', 'C2', '^', '-.')]
    ratios = np.linspace(0.1, 2.4, 48)
    for fwhm_s, label, color, mk, lsty in sensors:
        sigma_eff = _effective_width_depth(fwhm_s, 0.6, 1.8)[0]
        cf, lo, hi = [], [], []
        for r in ratios:
            s = max(r * sigma_eff, 1e-3)
            c = closed_form_crlb(fwhm_s, s)['sigma_crlb_times_snr']
            n_half = int(np.ceil(4.0 * sigma_eff / s))
            ks = np.arange(-n_half, n_half + 1)
            vals = []
            for phi in np.linspace(0.0, s, 180, endpoint=False):
                lam = ks * s + phi
                d = (0.6 * 1.8 / sigma_eff) * lam / sigma_eff ** 2 * \
                    np.exp(-0.5 * (lam / sigma_eff) ** 2)
                vals.append(1.0 / np.sqrt(np.sum(d ** 2)))
            cf.append(c); lo.append(min(vals)); hi.append(max(vals))
        ax.fill_between(ratios, lo, hi, facecolor=_tint(color, 0.82), lw=0)
        ax.plot(ratios, cf, color=color, lw=2.0, marker=mk, ms=4.2,
                markevery=6, ls=lsty,
                label=label + ' (closed form = phase average)')
    for x0, color in [(1.70, 'C1'), (1.80, 'C0'), (2.17, 'C2')]:
        ax.axvline(x0, color=_tint(color, 0.45), ls=':', lw=1.0)
    ax.set_xlabel(r'sampling ratio  $\Delta/\sigma_{\mathrm{eff}}$')
    ax.set_ylabel(r'$\sigma_{\mathrm{CRLB}}\times\mathrm{SNR}$  (nm)')
    ax.set_title('Closed form and phase-averaged numerical Fisher information agree')
    ax.legend(fontsize=7.5, ncol=1, loc='upper left')
    save(fig, 'fig1_crlb_validation')


def fig2_sv2_independence():

    import csv as _csv

    def _read(path):
        with open(path, encoding='utf-8') as f:
            return list(_csv.DictReader(f))

    rc = _read(os.path.join(HERE, 'out', 'rank_criterion_validation.csv'))
    gain = [float(r['rho']) for r in rc if r['cls'] == 'gain']
    smile = {}
    for A in ('1.0', '3.0', '5.0'):
        vals = [float(r['rho']) for r in rc
                if r['cls'] == 'ramp' and abs(float(r['amp_nm']) - float(A)) < 1e-9]
        smile[A] = float(np.mean(vals)) if vals else float('nan')
    met = _read(os.path.join(HERE, 'out', 'enmap_scene_metrics.csv'))
    full = {r['scene']: float(r['rho']) for r in met if int(r['rows']) == 1024}
    terr = _read(os.path.join(HERE, 'out', 'terrain_subregions.csv'))
    pavia = [float(r['rho']) for r in terr if r['dataset'] == 'PaviaU']

    cats = ['gain\nonly', 'smile\n1 nm', 'smile\n3 nm', 'smile\n5 nm',
            'Nevada\n(EnMAP)', 'Groningen\n(EnMAP)', 'Tyrol\n(EnMAP)', 'PaviaU\n(urban)']
    vals = [float(np.mean(gain)), smile['1.0'], smile['3.0'], smile['5.0'],
            full['Nevada'], full['Groningen'], full['Tyrol'], float(np.mean(pavia))]
    errs = [float(np.std(gain)), 0, 0, 0, 0, 0, 0, float(np.std(pavia))]
    colors = ['#888888', '#ffbb78', '#ff7f0e', '#d62728',
              '#1f77b4', '#1f77b4', '#1f77b4', '#9467bd']

    fig, ax = plt.subplots(figsize=(5.4, 3.05))
    ax.bar(cats, vals, yerr=errs, color=colors, capsize=3, edgecolor='k', lw=0.5)
    ax.axhline(0.1, color='red', ls='--', lw=1.2, label=r'reference level $\rho=0.1$')
    ax.axhline(0.02, color='darkred', ls='-', lw=1.3, label=r'decision level $\tau=0.02$')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 2.0)
    ax.set_ylabel(r'$\rho = s_2/s_1$  (log scale)')
    ax.legend(fontsize=8.5, loc='upper left')
    save(fig, 'fig2_sv2_independence')


def fig3_enmap_closed_loop():

    rows = read_csv('enmap_ramp_gain_validation.csv')
    scenes = [r['scene'] for r in rows]
    truth = [float(r['truth_p2p_nm']) for r in rows]
    est_map = {'Nevada': 21.9, 'Groningen': 3.0, 'Tyrol': 4.5}
    ratio_map = {'Nevada': '58', 'Groningen': '8.3', 'Tyrol': '12.8'}
    est = [est_map[r['scene']] for r in rows]
    inj_floor = {'Nevada': 1.10, 'Groningen': 0.33, 'Tyrol': 7.19}
    phys = [inj_floor[r['scene']] for r in rows]
    fl = read_local('crlb_scene_floor.csv')
    floor_map = {r['scene']: 0.5 * (float(r['p2p_floor_lo_nm'])
                                    + float(r['p2p_floor_hi_nm'])) for r in fl}
    floor = [floor_map[r['scene']] for r in rows]
    x = np.arange(len(scenes))
    w = 0.2
    fig, ax = plt.subplots(figsize=(5.5, 3.48))
    ax.bar(x - 1.5*w, truth, w, label='metadata reference', color='#2ca02c', edgecolor='k', lw=0.5)
    ax.bar(x - 0.5*w, est, w, label='BL-SPIC estimate', color='#1f77b4', edgecolor='k', lw=0.5)
    ax.bar(x + 0.5*w, phys, w, label='reference-conditioned confound floor', color='#ff7f0e', edgecolor='k', lw=0.5)
    ax.bar(x + 1.5*w, floor, w, label='CRLB floor (p2p)', color='#d62728', edgecolor='k', lw=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('peak-to-peak $\\Delta\\lambda$ at O$_2$-A (nm, log)')
    ax.set_xticks(x); ax.set_xticklabels(scenes)
    ax.legend(fontsize=8.5, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=4, borderaxespad=0.0)
    ax.set_ylim(0.05, max(max(phys), max(est), max(floor)) * 2.2)

    for i, r in enumerate(rows):
        ax.text(x[i], est[i]*1.15, '%s$\\times$' % ratio_map[r['scene']],
                ha='center', fontsize=8.5, fontweight='bold', color='#1f77b4')
    save(fig, 'fig3_enmap_closed_loop')


def fig4_ablation():

    rows = read_local('ablation24.csv')
    order = ['default_K2', 'K4', 'K8', 'full_rank', 'strong_reg', 'weak_reg',
             'shared_refl', 'five_line_anchor']
    config_labels = ['K=2', 'K=4', 'K=8', 'full-rank', 'strong reg', 'weak reg',
                     'shared refl', '5-line anchor']
    scenes = ['Nevada', 'Groningen', 'Tyrol']
    fig, ax = plt.subplots(figsize=(6.0, 3.27))
    width = 0.25
    x = np.arange(len(order))
    colors = {'Nevada': '#1f77b4', 'Groningen': '#ff7f0e', 'Tyrol': '#2ca02c'}
    for i, sc in enumerate(scenes):
        vals = []
        for cfg in order:
            match = [r for r in rows if r['scene'] == sc and r['config'] == cfg]
            vals.append(float(match[0]['est_to_ref']) if match else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=sc, color=colors[sc],
               edgecolor='k', lw=0.5)
    ax.axhline(1.0, color='green', ls='--', lw=1.5, label='correct (=1)')
    ax.set_xticks(x); ax.set_xticklabels(config_labels, rotation=20, ha='right')
    ax.set_ylabel('estimated / reference  (O$_2$-A p2p)')
    ax.legend(fontsize=8.5, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              borderaxespad=0.0)
    ax.set_ylim(0, 180)
    save(fig, 'fig4_ablation_24config')


def fig5_smooth_full_rank():

    rows = read_local('fullrank_sweep.csv')
    labels = [r['config'] for r in rows]
    ratios = [float(r['est_to_ref']) for r in rows]
    colors = ['#1f77b4'] * (len(rows) - 1) + ['#888888']
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.barh(labels, ratios, color=colors, edgecolor='k', lw=0.5)
    ax.axvline(1.0, color='green', ls='--', lw=1.5, label='correct (=1)')
    ax.set_xlabel('estimated / reference  (O$_2$-A p2p)')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_xlim(0, max(ratios) * 1.25)
    save(fig, 'fig5_smooth_full_rank')


def fig6_hyperion_cross_sensor():

    rows = read_csv('hyperion_ramp_gain.csv')
    scenes = [r['scene'] for r in rows]
    sv2 = [float(r['sv2']) for r in rows]
    sync = [float(r['phys_sync']) for r in rows]
    x = np.arange(len(scenes))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 3.61), sharex=True,
                                  gridspec_kw={'height_ratios': [1, 1]})
    a1.bar(x, sv2, color='#9467bd', edgecolor='k', lw=0.5)
    a1.axhline(0.1, color='red', ls='--', lw=1.2, label=r'reference level $\rho=0.1$')
    a1.axhline(0.02, color='darkred', ls='-', lw=1.3, label=r'decision level $\tau=0.02$')
    a1.set_ylabel(r'$\rho = s_2/s_1$')
    a1.set_title('Hyperion (WATER dataset): null rejected and synchronous')
    a1.legend(fontsize=8.5, loc='upper right'); a1.set_ylim(0, 1.02)
    a2.bar(x, sync, color='#17becf', edgecolor='k', lw=0.5)
    a2.axhline(0.65, color='green', ls='--', lw=1.2, label='synchronous threshold')
    a2.set_ylabel('cross-line sync. corr.')
    a2.set_xticks(x); a2.set_xticklabels(scenes, rotation=40, ha='right', fontsize=8.5)
    a2.set_ylim(0, 1.02); a2.legend(fontsize=8.5, loc='upper right')
    save(fig, 'fig6_hyperion_cross_sensor')


def fig7_terrain_subregions():

    rows = read_local('terrain_subregions.csv')
    nev = [r for r in rows if r['dataset'] == 'Nevada']
    pav = [r for r in rows if r['dataset'] == 'PaviaU']
    nh = [float(r['heterogeneity']) for r in nev]
    ns = [float(r['rho']) for r in nev]
    nr = sum(1 for r in nev if float(r['rho']) >= 0.02)
    ph = [float(r['heterogeneity']) for r in pav]
    ps = [float(r['rho']) for r in pav]
    pr = sum(1 for r in pav if float(r['rho']) >= 0.02)
    fig, ax = plt.subplots(figsize=(5.3, 3.48))
    ax.scatter(nh, ns, c='#d62728', s=40, edgecolor='k', lw=0.3,
               label='Nevada desert sub-regions (%d/%d $\\rho>\\tau$)' % (nr, len(nev)))
    ax.scatter(ph, ps, c='#9467bd', s=40, marker='s', edgecolor='k', lw=0.3,
               label='PaviaU urban sub-regions (%d/%d $\\rho>\\tau$)' % (pr, len(pav)))
    ax.axhline(0.1, color='red', ls='--', lw=1.2, label=r'reference level $\rho=0.1$')
    ax.axhline(0.02, color='darkred', ls='-', lw=1.3, label=r'decision level $\tau=0.02$')
    ax.set_xscale('log')
    ax.set_xlabel('sub-region spatial heterogeneity (local std, log)')
    ax.set_ylabel(r'$\rho = s_2/s_1$')
    ax.legend(fontsize=8.5, loc='lower right')
    save(fig, 'fig7_terrain_subregions')


def fig8_floor_vs_truth_snr():

    from crlb_theory import numerical_fisher_info
    fig, ax = plt.subplots(figsize=(5.4, 3.41))
    snrs = np.array([100, 200, 300, 500, 1000, 2000, 5000])
    k_p2p = 3.35
    for fwhm, samp, color, label, mk, lsty in [
            (7.8, 6.4, '#1f77b4', 'EnMAP-like O$_2$-A', 'o', '-'),
            (10.0, 10.0, '#ff7f0e', 'Hyperion-like O$_2$-A', 's', '--')]:
        info = numerical_fisher_info(fwhm, samp)
        floors = k_p2p * info['sigma_crlb_times_snr'] / snrs
        ax.plot(snrs, floors, color=color, lw=2.2, marker=mk, markersize=5,
                ls=lsty, label=label)
    ax.axhspan(0.33, 7.19, color='#ffe2c4', lw=0, zorder=0)
    ax.axhline(0.38, color='#2ca02c', ls='--', lw=2.0,
               label='EnMAP reference $\\approx$0.38 nm')
    ax.axhline(4.25, color='#9467bd', ls=':', lw=2.0,
               label='Hyperion reference up to 4.25 nm')
    for s0 in (276, 763):
        ax.axvline(s0, color='0.35', ls='-.', lw=1.1, zorder=1)
    ax.text(460, 0.023, 'measured operating SNR (276--763)', fontsize=7.5,
            ha='center', color='0.25')
    ax.text(105, 2.6, 'measured scene-confound floor\n(0.33--7.2 nm)',
            fontsize=7.5, color='#b35900', va='center')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('column-mean SNR (log)')
    ax.set_ylabel('peak-to-peak $\\Delta\\lambda$ (nm, log)')
    ax.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=2, borderaxespad=0.0)
    ax.set_ylim(0.02, 30)
    save(fig, 'fig8_floor_vs_truth')


if __name__ == '__main__':
    print('Generating figures ->', OUT)
    fig1_crlb_closed_vs_numerical()
    fig2_sv2_independence()
    fig3_enmap_closed_loop()
    fig4_ablation()
    fig5_smooth_full_rank()
    fig6_hyperion_cross_sensor()
    fig7_terrain_subregions()
    fig8_floor_vs_truth_snr()
    print('Done.')
