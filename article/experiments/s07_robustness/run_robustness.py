"""§7 robustness suite: fixed piecewise-constant field, package weights, k-sweep per case.

    python -m article.experiments.s07_robustness.run_robustness
"""
import argparse
from pathlib import Path

import numpy as np

from ..s01_benchmark_setup.diagnostics import metrics, write_csv, write_metadata
from ..s01_benchmark_setup.package_sampling import dropped_area, package_weight_matrix
from ..s01_benchmark_setup.reference_geometry import build_case, polygons_to_matrix

DEFAULT_OUT = Path(__file__).resolve().parent / 'results'
CASES = ('baseline', 'coarse_to_fine', 'fine_to_coarse', 'shifted', 'distorted', 'mask')


def case_settings(case, ns, nt):
    """(source_n, target_n, jitter, shift, mask) for each named case."""
    return {'baseline': (ns, nt, .17, False, False),
            'coarse_to_fine': (max(3, ns // 2), 2 * nt, .17, False, False),
            'fine_to_coarse': (2 * ns, max(2, nt // 2), .17, False, False),
            'shifted': (ns, nt, .17, True, False),
            'distorted': (ns, nt, .24, False, False),
            'mask': (ns, nt, .17, False, True)}[case]


def robustness_case(case, ns, nt, levels, seed):
    sn, tn, jit, shift, mask = case_settings(case, ns, nt)
    nodes, tri, x, y, src, A, polys, cover = build_case(sn, tn, seed, jit, shift, mask)
    c = nodes[tri].mean(axis=1)
    v = 1 + 0.5 * np.sin(2 * np.pi * c[:, 0]) * np.cos(2 * np.pi * c[:, 1])
    ref = polygons_to_matrix(polys, len(A), len(tri)) @ v / A
    active = cover > 1e-14
    rows = []
    for k in levels:
        B, info = package_weight_matrix(nodes, tri, x, y, k, return_info=True)
        W = np.asarray(B.sum(axis=1)).ravel()
        q = (B @ v) / A
        err = metrics(q, ref, A, active)
        # Valid physical source-covered area, not full rectangular cell area.
        f0 = cover[(W == 0) & active].sum() / cover[active].sum()
        rows.append(dict(case=case, k=k, source_n=sn, target_n=tn, mask=mask, normalization='A',
                         geometric_E2=err[1], missing_valid_area_fraction=f0,
                         unassigned_area=dropped_area(info, src, k),
                         physical_mass_residual=(np.dot(q, A) - np.dot(v, src)) / np.dot(abs(v), src),
                         active_area_fraction=cover.sum() / A.sum(), package_weight_seconds=info['seconds']))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-n', type=int, default=8)
    p.add_argument('--target-n', type=int, default=11)
    p.add_argument('--k', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--out', default=str(DEFAULT_OUT))
    a = p.parse_args(argv)
    levels = sorted(set(a.k))
    if a.source_n < 3 or a.target_n < 2 or min(levels) < 0:
        p.error('Invalid mesh sizes or subdivision depths')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in a.cases:
        print('robustness', case, flush=True)
        rows += robustness_case(case, a.source_n, a.target_n, levels, a.seed)
    write_csv(out / 'robustness_diagnostics.csv', rows)
    write_metadata(out / 'metadata.json', vars(a))
    print(f'Saved to {out}; NaN means unavailable.')


if __name__ == '__main__':
    main()
