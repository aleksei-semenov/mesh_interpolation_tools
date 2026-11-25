"""§4 Figure 4: error components versus subdivision depth, one panel per reconstruction.

    python -m article.experiments.s04_smooth_field.plot_fig4 [--fit-k 1 4]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from article.common import plt
from matplotlib.lines import Line2D

from ..s01_benchmark_setup.diagnostics import read_csv
from .run_smooth import DEFAULT_OUT

COLORS = {'constant': '#0072B2', 'linear': '#D55E00', 'quadratic': '#009E73'}
STYLES = {'geometric_E2': '-', 'representation_E2': ':', 'E2': '--'}


def num(row, key):
    return float(row[key])


def curve(ax, x, y, color, style, label=None, mark='o', status=0):
    """Positive values on log axis; exact zeros (▼) and unavailable (×) in a strip below."""
    x = np.array(x, float)
    y = np.array(y, float)
    ok = np.isfinite(y) & (y > 0)
    # NaNs preserve gaps. Exact zeros/NA shown in an axes-relative strip: not fake log values.
    ax.plot(x, np.where(ok, y, np.nan), color=color, linestyle=style, marker=mark, markersize=3, label=label)
    trans = ax.get_xaxis_transform()
    z = np.isfinite(y) & (y == 0)
    na = ~np.isfinite(y)
    if z.any():
        ax.scatter(x[z], np.full(z.sum(), -.10 - .018 * (status % 3)), marker='v', color=color, s=13,
                   transform=trans, clip_on=False)
    if na.any():
        ax.scatter(x[na], np.full(na.sum(), -.10 - .018 * (status % 3)), marker='x', color=color, s=13,
                   transform=trans, clip_on=False)


def plot_field(rows, field, out, fit_k=None):
    rows = [r for r in rows if r['field'] == field]
    recs = [r for r in COLORS if any(row['reconstruction'] == r for row in rows)]
    fig, axes = plt.subplots(1, len(recs), figsize=(5.5 * len(recs), 5), sharey=True, squeeze=False)
    axes = axes[0]
    fits = []
    for ax, rec in zip(axes, recs):
        sub = [r for r in rows if r['reconstruction'] == rec]
        for method, mark in (('W', 'o'), ('A', 's')):
            rr = sorted([r for r in sub if r['normalization'] == method], key=lambda r: int(r['k']))
            for key in ('geometric_E2', 'E2'):
                curve(ax, [num(r, 'k') for r in rr], [num(r, key) for r in rr], COLORS[rec], STYLES[key], mark=mark)
            if fit_k is not None:
                left, right = fit_k
                used = [r for r in rr if left <= int(r['k']) <= right and float(r['missing_fraction']) == 0
                        and np.isfinite(num(r, 'geometric_E2')) and num(r, 'geometric_E2') > 0]
                # Require 3 observed valid depths and retain the actual used interval.
                slope = float(np.polyfit([-int(r['k']) * np.log(2) for r in used],
                                         [np.log(num(r, 'geometric_E2')) for r in used], 1)[0]) \
                    if len(used) >= 3 else np.nan
                fits.append(dict(field=field, reconstruction=rec, normalization=method,
                                 requested_k_start=left, requested_k_end=right,
                                 used_depths=';'.join(r['k'] for r in used), n=len(used),
                                 fitted_geometric_E2_order=slope))
        rr = sorted([r for r in sub if r['normalization'] == 'A'], key=lambda r: int(r['k']))
        curve(ax, [num(r, 'k') for r in rr], [num(r, 'representation_E2') for r in rr], COLORS[rec], ':', mark='')
        if fit_k is not None:
            ax.axvspan(*fit_k, color='gray', alpha=.10)
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=.22)
        ax.set(xlabel='Subdivision depth $k$', title=f'{rec} reconstruction')
    axes[0].set_ylabel('Area-weighted $E_2$')
    if fits:
        with open(out / f'fitted_rates_{field}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(fits[0]))
            w.writeheader()
            w.writerows(fits)
    handles = ([Line2D([0], [0], color='black', ls=STYLES[k], label=lab)
                for k, lab in [('geometric_E2', 'geometric'), ('representation_E2', 'source representation'),
                               ('E2', 'total')]]
               + [Line2D([0], [0], color='black', marker='o', ls='None', label='W sampled'),
                  Line2D([0], [0], color='black', marker='s', ls='None', label='A physical')])
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(.5, .0))
    fig.suptitle(f'Smooth field "{field}": package subdivision centroids; ▼ exact zero, × unavailable')
    fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.24, wspace=.08)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig4_error_components_{field}.{ext}', dpi=220)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(DEFAULT_OUT), help='directory with smooth_diagnostics.csv')
    p.add_argument('--field', choices=['sine', 'quadratic', 'all'], default='all')
    p.add_argument('--fit-k', nargs=2, type=int, metavar=('START', 'END'))
    a = p.parse_args(argv)
    if a.fit_k and a.fit_k[0] >= a.fit_k[1]:
        p.error('--fit-k requires START < END')
    out = Path(a.out)
    rows = read_csv(out / 'smooth_diagnostics.csv')
    fields = sorted({r['field'] for r in rows}) if a.field == 'all' else [a.field]
    for f in fields:
        plot_field(rows, f, out, a.fit_k)
    print(f'Saved Figure 4 to {out}')


if __name__ == '__main__':
    main()
