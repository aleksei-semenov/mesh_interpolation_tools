"""§2 Figures 1 and 2 from the §1 benchmark data, without rerunning remaps.

    python -m article.experiments.s01_benchmark_setup.run_benchmark      # first
    python -m article.experiments.s02_constant_field.plot_fig1_fig2 [--map-k 0 5]
"""
import argparse
from pathlib import Path

import numpy as np

from article.common import plt
from matplotlib.colors import TwoSlopeNorm

from ..s01_benchmark_setup.diagnostics import read_csv
from ..s01_benchmark_setup.run_benchmark import DEFAULT_OUT as S01_RESULTS

DEFAULT_OUT = Path(__file__).resolve().parent / 'results'


def curve(ax, x, y, label, style='o-', color=None):
    """Positive values on log axis; exact zeros/NA on a labelled status strip."""
    x, y = np.asarray(x), np.asarray(y)
    positive = np.isfinite(y) & (y > 0)
    line, = ax.plot(x, np.where(positive, y, np.nan), style, label=label, color=color, ms=4)
    color = line.get_color()
    for mask, marker, height in [(np.isfinite(y) & (y == 0), 'v', .025), (~np.isfinite(y), 'x', .075)]:
        if mask.any():
            ax.plot(x[mask], np.full(mask.sum(), height), linestyle='none', marker=marker,
                    color=color, transform=ax.get_xaxis_transform(), clip_on=False, ms=5)
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=.22)
    return line


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=str(S01_RESULTS), help='§1 results directory')
    p.add_argument('--out', default=str(DEFAULT_OUT))
    p.add_argument('--map-k', type=int, nargs=2, help='coarse and fine saved depths')
    args = p.parse_args(argv)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(data / 'diagnostics.csv')
    levels = sorted({int(r['k']) for r in rows})
    maps = args.map_k or [levels[0], levels[-1]]
    if not set(maps) <= set(levels):
        p.error('map depths must have saved data')
    with np.load(data / 'mesh_reference.npz') as d:
        x, y = d['x_edges'], d['y_edges']
    shape = (len(y) - 1, len(x) - 1)
    fields = []
    for k in maps:
        with np.load(data / f'fields_k{k}.npz') as d:
            fields.append([d['defect'], d['constant_W'] - 1, d['constant_A'] - 1])

    # ---- Figure 1: spatial diagnostic
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained', squeeze=False)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#b5b5b5')
    for col, title in enumerate([r'$W_j/A_j^T-1$', r'$q_j^{T,W}-1$', r'$q_j^{T,A}-1$']):
        finite = np.concatenate([f[col][np.isfinite(f[col])] for f in fields])
        vmax = max(abs(finite)) if len(finite) else 0
        # A display-only scale for an identically zero column, not an error floor.
        limit = vmax if vmax > 0 else 1.
        for row, k in enumerate(maps):
            ax = axes[row, col]
            im = ax.pcolormesh(x, y, np.ma.masked_invalid(fields[row][col].reshape(shape)),
                               cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), shading='flat')
            ax.set(title=f'{title}; k={k}', xlabel='x', ylabel='y', aspect='equal')
        cb = fig.colorbar(im, ax=axes[:, col], shrink=.8)
        if vmax == 0:
            cb.set_ticks([0])
            cb.set_label('identically zero on defined cells')
    fig.suptitle('Constant field: gray = undefined; unsampled area defect = −1')
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'area_diagnostics.{ext}', dpi=220)
    plt.close(fig)

    # ---- Figure 2: conservation and coverage versus subdivision
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')

    def values(field, norm, key):
        d = {int(r['k']): float(r[key]) for r in rows if r['field'] == field and r['normalization'] == norm}
        return [d[k] for k in levels]

    for norm, color in [('W', 'tab:blue'), ('A', 'tab:orange')]:
        for key, style, measure in [('physical_residual', 'o-', 'physical'), ('sampled_residual', 's--', 'sampled')]:
            curve(axes[0, 0], levels, np.abs(values('constant', norm, key)), f'{norm}: {measure}', style, color)
        curve(axes[0, 1], levels, values('constant', norm, 'constant_error'), f'{norm}: whole domain', 'o-', color)
        if norm == 'W':
            curve(axes[0, 1], levels, values('constant', norm, 'constant_error_sampled'),
                  'W: sampled subset only', 's--', color)
        for key, style in [('L1', 'o-'), ('L2', 's--'), ('Linf', '^:')]:
            curve(axes[1, 1], 2. ** (-np.array(levels)), values('fixed', norm, key), f'{norm}: {key}', style, color)
    curve(axes[1, 0], levels, values('constant', 'A', 'missing_area_fraction'), 'unsampled area / target area')
    axes[1, 1].set_xscale('log')
    axes[1, 1].invert_xaxis()
    for ax, title, xlabel, ylabel in [
            (axes[0, 0], 'Constant-field mass residuals', 'Subdivision depth k', 'Absolute normalized residual'),
            (axes[0, 1], 'Constant preservation', 'Subdivision depth k', 'Maximum absolute constant error'),
            (axes[1, 0], 'Coverage', 'Subdivision depth k', 'Missing physical-area fraction'),
            (axes[1, 1], 'Fixed-source geometric convergence (§3)', r'Subdivision scale $2^{-k}$',
             'Area-weighted error / maximum error')]:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if not any(np.any(np.asarray(line.get_ydata(), dtype=float) > 0)
                   for line in ax.lines if line.get_transform() == ax.transData):
            ax.set_ylim(1e-16, 1)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('W = sampled-area normalization; A = physical-area normalization')
    fig.supxlabel('Status strip in each panel: ▼ exact zero (not a log value); × unavailable', fontsize=10)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'constant_convergence.{ext}', dpi=220)
    plt.close(fig)
    print(f'Saved area_diagnostics and constant_convergence (PDF, PNG) to {out}')


if __name__ == '__main__':
    main()
