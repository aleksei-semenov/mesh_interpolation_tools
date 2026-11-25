"""§4 smooth analytic field: subdivision sweep on fixed meshes.

    python -m article.experiments.s04_smooth_field.run_smooth
    python -m article.experiments.s04_smooth_field.run_smooth --source-n 32 --target-n 32 --k 0 1 2 3 4 5
"""
import argparse
from pathlib import Path

import numpy as np

from ..s01_benchmark_setup.diagnostics import write_csv, write_metadata
from .smooth import smooth_group

DEFAULT_OUT = Path(__file__).resolve().parent / 'results'


def latex_table(rows, out):
    def fmt(x):
        return '--' if not np.isfinite(x) else f'{x:.3e}'
    for fieldname in sorted({r['field'] for r in rows}):
        data = [r for r in rows if r['field'] == fieldname and r['missing_fraction'] == 0]
        lines = [r'\begin{tabular}{llllllllll}', r'\toprule',
                 r'Field & $n_S/n_T$ & Recon. & Norm. & $k$ & $E_1$ & $E_2$ & $E_\infty$ & $p_k$ & $f_0$ \\',
                 r'\midrule']
        for r in data:
            lines.append(f"{r['field']} & {r['source_n']}/{r['target_n']} & {r['reconstruction']} & "
                         f"{r['normalization']} & {r['k']} & "
                         + ' & '.join(fmt(r[q]) for q in ('E1', 'E2', 'Einf', 'pk', 'missing_fraction')) + r' \\')
        if not data:
            lines.append(r'\multicolumn{10}{c}{No fully covered runs} \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (out / f'convergence_{fieldname}.tex').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-n', type=int, default=8)
    p.add_argument('--target-n', type=int, default=11)
    p.add_argument('--k', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--fields', nargs='+', choices=['sine', 'quadratic'], default=['sine', 'quadratic'])
    p.add_argument('--reconstructions', nargs='+', choices=['constant', 'linear', 'quadratic'],
                   default=['constant', 'linear'])
    p.add_argument('--quad-order', type=int, default=10)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--out', default=str(DEFAULT_OUT))
    a = p.parse_args(argv)
    levels = sorted(set(a.k))
    if a.source_n < 3 or a.target_n < 1 or min(levels) < 0 or a.quad_order < 4:
        p.error('Invalid mesh sizes, subdivision depths or quadrature order')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.fields:
        print('smooth', name, 'subdivision sweep', flush=True)
        rows += smooth_group(a.source_n, a.target_n, levels, name, a.seed, a.quad_order,
                             'subdivision', a.reconstructions)
    write_csv(out / 'smooth_diagnostics.csv', rows)
    latex_table(rows, out)
    write_metadata(out / 'metadata.json', vars(a))
    print(f'Saved to {out}; NaN means unavailable.')


if __name__ == '__main__':
    main()
