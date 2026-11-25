"""§6 Figure 6: A geometric E2 vs 2^-k (from §4), B total E2 vs h_S, C geometric E2 vs h_T.

    python -m article.experiments.s06_independent_refinement.plot_fig6
"""
import argparse
from pathlib import Path

import numpy as np

from article.common import plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

from ..s01_benchmark_setup.diagnostics import read_csv
from ..s04_smooth_field.plot_fig4 import COLORS, curve, num
from ..s04_smooth_field.run_smooth import DEFAULT_OUT as S04_RESULTS
from .run_refinement import DEFAULT_OUT


def plot_field(sub_rows, ref_rows, field, out):
    sub_rows = [r for r in sub_rows if r['field'] == field]
    ref_rows = [r for r in ref_rows if r['field'] == field]
    recs = [r for r in COLORS if any(row['reconstruction'] == r for row in sub_rows + ref_rows)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, rec in enumerate(recs):
        for method, style, mark in (('W', '-', 'o'), ('A', '--', 's')):
            rr = sorted([r for r in sub_rows if r['reconstruction'] == rec and r['normalization'] == method],
                        key=lambda r: int(r['k']))
            curve(ax[0], [2. ** -int(r['k']) for r in rr], [num(r, 'geometric_E2') for r in rr],
                  COLORS[rec], style, mark=mark, status=i)
            for panel, sweep, xkey, ykey in [(1, 'source', 'hS', 'E2'), (2, 'target', 'hT', 'geometric_E2')]:
                rr = sorted([r for r in ref_rows if r['sweep'] == sweep and r['reconstruction'] == rec
                             and r['normalization'] == method], key=lambda r: num(r, xkey))
                curve(ax[panel], [num(r, xkey) for r in rr], [num(r, ykey) for r in rr], COLORS[rec], style,
                      mark=mark, status=i)
                # Flag source points where geometry is not small relative to total error.
                if panel == 1:
                    flagged = [r for r in rr if np.isfinite(num(r, 'geometric_E2')) and np.isfinite(num(r, 'E2'))
                               and num(r, 'E2') > 0 and num(r, 'geometric_E2') >= .1 * num(r, 'E2')]
                    if flagged:
                        ax[1].scatter([num(r, xkey) for r in flagged], [num(r, ykey) for r in flagged],
                                      facecolors='none', edgecolors='black', s=70, linewidths=.8, zorder=5)
    for a in ax:
        a.set_yscale('log')
        a.set_xscale('log')
        a.grid(True, which='both', alpha=.22)
    for panel, sweep, xkey in [(1, 'source', 'hS'), (2, 'target', 'hT')]:
        ticks = sorted({num(r, xkey) for r in ref_rows if r['sweep'] == sweep})
        if ticks:
            ax[panel].xaxis.set_major_locator(FixedLocator(ticks))
            ax[panel].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax[panel].xaxis.set_minor_formatter(NullFormatter())
    ax[0].set(xlabel=r'Subdivision scale $2^{-k}$', ylabel=r'Geometric $E_2$', title='A: subdivision (fixed meshes)')
    ax[1].set(xlabel=r'Source size $h_S=\sqrt{1/N_S}$', ylabel=r'Total $E_2$', title='B: source refinement')
    ax[2].set(xlabel=r'Target size $h_T=\sqrt{1/N_T}$', ylabel=r'Geometric $E_2$', title='C: target refinement')
    handles = ([Line2D([0], [0], color=COLORS[r], label=r) for r in recs]
               + [Line2D([0], [0], color='black', marker='o', ls='-', label='W sampled'),
                  Line2D([0], [0], color='black', marker='s', ls='--', label='A physical')])
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8, bbox_to_anchor=(.5, .02))
    fig.suptitle(f'Smooth field "{field}": independent refinement with package subdivision centroids')
    fig.text(.5, .085, '▼ exact zero, × unavailable (status strips); circled source point: '
             'geometric error ≥ 10% of total', ha='center', fontsize=8)
    fig.subplots_adjust(left=.06, right=.99, top=.88, bottom=.28, wspace=.32)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig6_refinement_{field}.{ext}', dpi=220)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(DEFAULT_OUT), help='directory with refinement_diagnostics.csv')
    p.add_argument('--subdivision-data', default=str(S04_RESULTS), help='§4 results (panel A)')
    p.add_argument('--field', choices=['sine', 'quadratic', 'all'], default='all')
    a = p.parse_args(argv)
    out = Path(a.out)
    ref_rows = read_csv(out / 'refinement_diagnostics.csv')
    sub_rows = read_csv(Path(a.subdivision_data) / 'smooth_diagnostics.csv')
    fields = sorted({r['field'] for r in ref_rows}) if a.field == 'all' else [a.field]
    for f in fields:
        plot_field(sub_rows, ref_rows, f, out)
    print(f'Saved Figure 6 to {out}')


if __name__ == '__main__':
    main()
