"""§7 Figure 7: heatmaps of geometric E2 and missing valid-area fraction (case x k).

    python -m article.experiments.s07_robustness.plot_fig7
"""
import argparse
from pathlib import Path

import numpy as np

from article.common import plt
from matplotlib.colors import LogNorm

from ..s01_benchmark_setup.diagnostics import read_csv
from .run_robustness import DEFAULT_OUT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(DEFAULT_OUT), help='directory with robustness_diagnostics.csv')
    a = p.parse_args(argv)
    out = Path(a.out)
    rows = read_csv(out / 'robustness_diagnostics.csv')
    cases = list(dict.fromkeys(r['case'] for r in rows))
    ks = sorted(set(int(r['k']) for r in rows))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in [(axs[0], 'geometric_E2', 'A: fixed-field geometric $E_2$ (A-normalization)'),
                           (axs[1], 'missing_valid_area_fraction', 'B: missing valid-area fraction')]:
        values = np.full((len(cases), len(ks)), np.nan)
        for r in rows:
            values[cases.index(r['case']), ks.index(int(r['k']))] = float(r[key])
        positive = values[np.isfinite(values) & (values > 0)]
        if len(positive):
            vmin, vmax = positive.min(), positive.max()
            vmax = vmax if vmax > vmin else vmin * 10
            im = ax.imshow(np.ma.masked_where(~np.isfinite(values) | (values <= 0), values), aspect='auto',
                           cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
            fig.colorbar(im, ax=ax, shrink=.8, label='positive values (log scale)')
        for i in range(len(cases)):
            for j in range(len(ks)):
                if values[i, j] == 0:
                    ax.text(j, i, '0', ha='center', va='center', fontsize=8)
                elif not np.isfinite(values[i, j]):
                    ax.text(j, i, '×', ha='center', va='center', fontsize=8)
        ax.set_xticks(range(len(ks)), ks)
        ax.set_yticks(range(len(cases)), [s.replace('_to_', ' → ').replace('_', ' ') for s in cases], fontsize=8)
        ax.set(xlabel='Subdivision depth k', title=title)
        ax.set_facecolor('#eeeeee')
    fig.suptitle('Mesh/coverage robustness with package subdivision centroids')
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig7_robustness.{ext}', dpi=220)
    plt.close(fig)
    print(f'Saved Figure 7 to {out}')


if __name__ == '__main__':
    main()
