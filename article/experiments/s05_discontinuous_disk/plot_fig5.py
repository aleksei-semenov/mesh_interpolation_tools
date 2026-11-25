"""§5 Figure 5: disk reference, output, signed error and a cell-average profile.

    python -m article.experiments.s05_discontinuous_disk.plot_fig5
"""
import argparse
from pathlib import Path

import numpy as np

from article.common import plt

from .run_disk import DEFAULT_OUT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(DEFAULT_OUT), help='directory with disk_fields.npz')
    a = p.parse_args(argv)
    out = Path(a.out)
    z = np.load(out / 'disk_fields.npz')
    x, y, ref, q, e = z['x'], z['y'], z['reference'], z['output'], z['error']
    fig, axs = plt.subplots(2, 2, figsize=(11, 9.5))
    limits = max(float(np.max(abs(e))), 1e-16)
    vmax = max(1., float(np.max(q)))
    vmin = min(0., float(np.min(q)))
    for ax, v, title in [(axs[0, 0], ref, 'A: target cell-average reference'),
                         (axs[0, 1], q, 'B: output (package, physical normalization)'),
                         (axs[1, 0], e, 'C: signed error')]:
        im = ax.pcolormesh(x, y, v, shading='flat', cmap='RdBu_r' if v is e else 'viridis',
                           vmin=-limits if v is e else vmin, vmax=limits if v is e else vmax)
        fig.colorbar(im, ax=ax, shrink=.82)
        ax.set(xlabel='x', ylabel='y', title=title, aspect='equal')
    mid = int(np.argmin(abs((y[:-1] + y[1:]) / 2 - .5)))
    center = (y[mid] + y[mid + 1]) / 2
    axs[1, 1].stairs(ref[mid], x, label='reference', linewidth=1.7)
    axs[1, 1].stairs(q[mid], x, label='output', linewidth=1.2)
    axs[1, 1].set(xlabel='x (cell edges)', ylabel='Cell-average value',
                  title=f'D: profile, target row centered at y={center:.3f}')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=.2)
    fig.suptitle('Disk indicator: discontinuity behaviour')
    fig.tight_layout(rect=(0, 0, 1, .96))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig5_disk.{ext}', dpi=220)
    plt.close(fig)
    print(f'Saved Figure 5 to {out}')


if __name__ == '__main__':
    main()
