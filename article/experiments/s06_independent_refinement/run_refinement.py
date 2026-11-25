"""§6 independent refinement: source-mesh sweep (B) and target-grid sweep (C).

Sweep A (subdivision) is the §4 run; Figure 6 reads it from there.

    python -m article.experiments.s06_independent_refinement.run_refinement
    python -m article.experiments.s06_independent_refinement.run_refinement \\
        --source-sweep 16 32 64 --target-sweep 16 32 64 --source-n 32 --target-n 32 --refine-k 5
"""
import argparse
from pathlib import Path

from ..s01_benchmark_setup.diagnostics import write_csv, write_metadata
from ..s04_smooth_field.smooth import smooth_group

DEFAULT_OUT = Path(__file__).resolve().parent / 'results'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-n', type=int, default=8, help='fixed source size for the target sweep')
    p.add_argument('--target-n', type=int, default=11, help='fixed target size for the source sweep')
    p.add_argument('--source-sweep', nargs='+', type=int, default=[5, 8, 12])
    p.add_argument('--target-sweep', nargs='+', type=int, default=[6, 11, 16])
    p.add_argument('--refine-k', type=int, default=4, help='subdivision depth used in both sweeps')
    p.add_argument('--fields', nargs='+', choices=['sine', 'quadratic'], default=['sine', 'quadratic'])
    p.add_argument('--reconstructions', nargs='+', choices=['constant', 'linear', 'quadratic'],
                   default=['constant', 'linear'])
    p.add_argument('--quad-order', type=int, default=10)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--out', default=str(DEFAULT_OUT))
    a = p.parse_args(argv)
    if min([a.source_n, *a.source_sweep]) < 3 or min([a.target_n, *a.target_sweep]) < 1 or a.refine_k < 0:
        p.error('Invalid mesh sizes or subdivision depth')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.fields:
        for n in sorted(set(a.source_sweep)):
            print('smooth', name, 'source', n, flush=True)
            rows += smooth_group(n, a.target_n, [a.refine_k], name, a.seed, a.quad_order, 'source',
                                 a.reconstructions)
        for n in sorted(set(a.target_sweep)):
            print('smooth', name, 'target', n, flush=True)
            rows += smooth_group(a.source_n, n, [a.refine_k], name, a.seed, a.quad_order, 'target',
                                 a.reconstructions)
    write_csv(out / 'refinement_diagnostics.csv', rows)
    write_metadata(out / 'metadata.json', vars(a))
    print(f'Saved to {out}; NaN means unavailable.')


if __name__ == '__main__':
    main()
