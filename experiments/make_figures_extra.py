# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon

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
CSV = os.path.join(HERE, '..', 'pipeline', 'out')
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))


def read_csv(name):
    with open(os.path.join(CSV, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save(fig, stem, tight=True):
    if tight:
        fig.tight_layout(pad=0.5)
    old_bbox = plt.rcParams['savefig.bbox']
    if not tight:
        plt.rcParams['savefig.bbox'] = None
    fig.savefig(os.path.join(OUT, stem + '.png'), dpi=300)
    fig.savefig(os.path.join(OUT, stem + '.pdf'), format='pdf')
    try:
        fig.savefig(os.path.join(OUT, stem + '.eps'), format='eps')
    except Exception as e:
        print('  [warn] EPS:', e)
    plt.rcParams['savefig.bbox'] = old_bbox
    plt.close(fig)
    print('[fig]', stem)


def fig_schematic():

    fig = plt.figure(figsize=(6.5, 3.0))


    ax = fig.add_axes([0.004, 0.229, 0.523, 0.671])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 59)
    ax.axis('off')
    ax.set_title('(a) Pushbroom optical geometry', fontsize=9)

    edge, fill = '#333333', '#dbe1e8'
    ray, smile = '#808080', '#d62728'
    grid = '#9aa2ab'
    bbox = dict(facecolor='white', edgecolor='none', pad=0.6)


    ax.add_patch(Rectangle((5.0, 15.0), 2.4, 29.0, facecolor=fill,
                           edgecolor=edge, lw=0.8, zorder=2))
    ax.plot([6.2, 6.2], [19.0, 40.0], color='white', lw=3.0, zorder=3,
            solid_capstyle='butt')
    ax.plot([6.2, 6.2], [19.0, 40.0], color=edge, lw=1.5, zorder=4,
            solid_capstyle='butt')
    ax.text(6.2, 13.5, 'entrance\nslit', ha='center', va='top', fontsize=7.5,
            linespacing=1.15)


    for y0, x1, y1 in ((24.4, 37.1, 25.4), (28.0, 38.5, 29.0),
                       (31.6, 39.9, 32.6)):
        ax.plot([7.6, x1], [y0, y1], color=ray, lw=0.8, zorder=1)
    ax.annotate('', xy=(21.0, 25.6), xytext=(13.5, 24.5),
                arrowprops=dict(arrowstyle='-|>', color=ray, lw=0.8,
                                mutation_scale=7))


    ax.add_patch(Polygon([[35.0, 20.0], [49.0, 20.0], [42.0, 38.0]], closed=True,
                         facecolor=fill, edgecolor=edge, lw=0.9, zorder=2))
    ax.text(42.0, 41.0, 'grating / prism', ha='center', va='bottom', fontsize=7.5)


    for x0, y0, y1 in ((46.9, 25.4, 23.0), (45.5, 29.0, 29.0), (44.1, 32.6, 35.0)):
        ax.plot([x0, 66.0], [y0, y1], color=ray, lw=0.8, zorder=1)


    dx0, dy0, dx1, dy1 = 66.0, 13.0, 95.0, 47.0
    ax.add_patch(Rectangle((dx0, dy0), dx1 - dx0, dy1 - dy0, facecolor='white',
                           edgecolor=edge, lw=1.0, zorder=2))
    for i in range(1, 8):
        xg = dx0 + (dx1 - dx0) * i / 8.0
        ax.plot([xg, xg], [dy0, dy1], color=grid, lw=0.3, zorder=3)
    for j in range(1, 5):
        yg = dy0 + (dy1 - dy0) * j / 5.0
        ax.plot([dx0, dx1], [yg, yg], color=grid, lw=0.3, zorder=3)


    ax.plot([69.0, 92.0], [32.0, 32.0], color=edge, lw=0.9, ls='--', zorder=5)
    xc = np.linspace(69.0, 92.0, 160)
    ax.plot(xc, 32.0 - 7.0 * (1.0 - ((xc - 80.5) / 11.5) ** 2),
            color=smile, lw=1.6, zorder=6)
    ax.text(73.5, 33.5, 'nominal', ha='left', va='bottom', fontsize=7,
            color=edge, bbox=bbox, zorder=7)
    ax.text(80.5, 22.0, 'slit image (smile)', ha='center', va='top', fontsize=7,
            color=smile, bbox=bbox, zorder=7)


    ax.text(80.5, 48.5, 'detector array', ha='center', va='bottom', fontsize=7.5)
    ax.annotate('', xy=(94.0, 9.5), xytext=(67.0, 9.5),
                arrowprops=dict(arrowstyle='->', color=edge, lw=0.8))
    ax.text(80.5, 7.0, 'cross-track $x$', ha='center', va='top', fontsize=7.5)
    ax.annotate('', xy=(98.0, 46.0), xytext=(98.0, 14.0),
                arrowprops=dict(arrowstyle='->', color=edge, lw=0.8))
    ax.text(98.0, 48.5, '$\\lambda$', ha='center', va='bottom', fontsize=8.5)


    ax = fig.add_axes([0.615, 0.20, 0.380, 0.70])
    x = np.linspace(0, 1000, 200)
    u = (x - 500.0) / 500.0
    shape = 1.0 - u ** 2 + 0.3 * u ** 4


    dl = 0.38 * (shape - 0.3) / 0.7
    ax.plot(x, dl, 'r-', lw=2.2)
    ax.axhline(0, color='k', lw=0.5)
    y0, y1 = dl.min(), dl.max()
    xb = 500
    ax.annotate('', xy=(xb, y0), xytext=(xb, y1),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.0))
    ax.text(xb, y0 - 0.015, 'p2p $\\approx$ 0.38 nm',
            color='red', fontsize=8.5, ha='center', va='top')
    ax.legend(fontsize=8.5, loc='upper left',
              labels=['EnMAP O$_2$-A smile (metadata reference)'])
    ax.set_ylim(-0.055, 0.46)
    ax.set_xlabel('cross-track detector pixel $x$')
    ax.set_ylabel(r'$\Delta\lambda$ (nm)')
    ax.set_title('(b) Smile as cross-track wavelength shift', fontsize=9)
    save(fig, 'fig0_schematic', tight=False)


def fig_o2a_line():

    fig, ax = plt.subplots(figsize=(4.7, 3.14))
    lam = np.linspace(755, 770, 400)

    sigma = 1.0
    T = 1 - 0.05 * np.exp(-((lam - 760) ** 2) / (2 * sigma ** 2))


    R = 0.3 * (1 + 0.0034 * (lam - 755))
    RT = R * T

    dlam = lam[1] - lam[0]
    h = np.abs(np.gradient(RT, dlam)) / RT
    ax.plot(lam, T, 'b-', lw=2, label=r'transmittance $T(\lambda)$ (O$_2$-A)')
    ax2 = ax.twinx()
    ax2.plot(lam, h, 'r-', lw=1.8, label=r'spectral gradient $h(\lambda)$')
    ax2.set_ylabel(r'$h = |\partial(RT)/\partial\lambda|/(RT)$', color='r', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_xlabel(r'wavelength $\lambda$ (nm)')
    ax.set_ylabel(r'transmittance', color='b', fontsize=9)
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('O$_2$-A line flanks set the rank threshold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), ncol=2, borderaxespad=0.0)
    save(fig, 'fig_o2a_line_shape')


def fig_svd_scree():

    z = np.load(os.path.join(HERE, 'out', 'rank_spectra.npz'))
    nz = np.load(os.path.join(HERE, 'out', 'nevada_spectrum.npz'))
    gain, s038, s3 = z['gain'], z['smile038'], z['smile3']
    real = nz['nevada']

    fig, ax = plt.subplots(figsize=(4.9, 3.05))
    idx = np.arange(1, 7)
    ax.semilogy(idx, gain, 'o-', lw=1.8, ms=7, color='#888888',
                label='gain only ($s_2/s_1=4\\times10^{-4}$)')
    ax.semilogy(idx, s038, 'D-', lw=1.8, ms=6, color='#ff7f0e',
                label='smile 0.38 nm p2p ($1.3\\times10^{-2}$)')
    ax.semilogy(idx, s3, 's-', lw=1.8, ms=7, color='#d62728',
                label='smile 3 nm p2p ($9\\times10^{-2}$)')
    ax.semilogy(idx, real, '^-', lw=1.8, ms=7, color='#1f77b4',
                label='real EnMAP Nevada ($0.51$)')
    ax.axhline(0.1, color='red', ls='--', lw=1.2, label='reference level $\\rho=0.1$')
    ax.axhline(0.02, color='darkred', ls='-', lw=1.3, label='decision level $\\tau=0.02$')
    ax.set_xlabel('singular value index $k$')
    ax.set_ylabel(r'$s_k / s_1$ (log)')
    ax.set_ylim(1e-5, 3.0)
    ax.legend(fontsize=8.0, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=2, borderaxespad=0.0)
    ax.set_xticks(idx)
    save(fig, 'fig_svd_scree')


def fig_xtrack_curves():

    z = np.load(os.path.join(HERE, 'out', 'xtrack_curves.npz'))
    meta = {}
    with open(os.path.join(HERE, 'out', 'xtrack_curves.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            meta[row['scene']] = row

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, sc, tag in [(axes[0], 'Nevada', 'a'), (axes[1], 'Tyrol', 'b')]:
        d = z[sc + '_dlam']
        t = z[sc + '_ref']
        s = float(meta[sc]['sign'])
        x = np.linspace(0, 1, len(d))
        axr = ax.twinx()
        axr.plot(x, t - t.mean(), '-', color='#2ca02c', lw=1.8)
        ax.plot(x, s * (d - d.mean()), '-', color='#1f77b4', lw=1.6)
        ax.set_xlabel('normalized cross-track coordinate')
        ax.set_ylabel('estimate $\\Delta\\lambda$ (nm)', color='#1f77b4')
        axr.set_ylabel('reference $\\Delta\\lambda$ (nm)', color='#2ca02c')
        ax.tick_params(axis='y', colors='#1f77b4')
        axr.tick_params(axis='y', colors='#2ca02c')
        ax.set_title('(%s) %s: %.1f nm est. vs %.3f nm ref., $R$ = %.2f'
                     % (tag, sc, float(meta[sc]['est_p2p_nm']),
                        float(meta[sc]['ref_p2p_nm']),
                        float(meta[sc]['shape_R_raw'])), fontsize=9)
        ax.set_xlim(0, 1)
    h = [plt.Line2D([], [], color='#1f77b4', lw=1.6),
         plt.Line2D([], [], color='#2ca02c', lw=1.8)]
    fig.legend(h, ['BL-SPIC estimate (left axis)', 'metadata reference (right axis)'],
               fontsize=8.5, loc='lower center', ncol=2, borderaxespad=0.0,
               bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.55)
    save(fig, 'fig_xtrack_curves')


if __name__ == '__main__':
    print('Generating extra key figures ->', OUT)
    fig_schematic()
    fig_o2a_line()
    fig_svd_scree()
    fig_xtrack_curves()
    print('Done.')
