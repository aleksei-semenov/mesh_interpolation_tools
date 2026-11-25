"""§5 disk indicator: bounds, interface distortion and errors with the package weights.

Piecewise-constant source (exact disk/triangle averages), physical-area
normalization, package subdivision at depth k.

    python -m article.experiments.s05_discontinuous_disk.run_disk [--k 3]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..s01_benchmark_setup.diagnostics import metrics, write_metadata
from ..s01_benchmark_setup.package_sampling import dropped_area, package_weight_matrix
from ..s01_benchmark_setup.reference_geometry import build_case, polygons_to_matrix
from .disk import RADIUS, disk_integral_triangle, disk_rectangle

DEFAULT_OUT = Path(__file__).resolve().parent / 'results'


def disk_case(ns, nt, k, seed, out, tol):
    nodes, tri, x, y, src, A, polys, cover = build_case(ns, nt, seed)
    v = np.array([disk_integral_triangle(t, tol=tol) / a for t, a in zip(nodes[tri], src)])
    ref = np.array([disk_rectangle(x[i], x[i + 1], y[j], y[j + 1], tol=tol) / A[j * nt + i]
                    for j in range(nt) for i in range(nt)])
    over = polygons_to_matrix(polys, len(A), len(tri)) @ v / A
    B, info = package_weight_matrix(nodes, tri, x, y, k, return_info=True)
    W = np.asarray(B.sum(axis=1)).ravel()
    q = (B @ v) / A
    geo = metrics(q, over, A)
    total = metrics(q, ref, A)
    negative = np.dot(A, np.maximum(-q, 0))
    above = np.dot(A, np.maximum(q - 1, 0))
    np.savez_compressed(out / 'disk_fields.npz', x=x, y=y, reference=ref.reshape(nt, nt),
                        output=q.reshape(nt, nt), error=(q - ref).reshape(nt, nt),
                        overlap=over.reshape(nt, nt), W=W.reshape(nt, nt))
    return dict(source_n=ns, target_n=nt, k=k, method='package constant', normalization='A',
                samples=len(tri) * 4 ** k,
                disk_source_integral=float(np.dot(v, src)), reference_disk_integral=float(np.dot(ref, A)),
                analytic_disk_integral=float(np.pi * RADIUS ** 2), disk_integral_tolerance=tol,
                minimum=float(q.min()), maximum=float(q.max()),
                undershoot=float(max(0, -q.min())), overshoot=float(max(0, q.max() - 1)),
                integrated_negative_part=negative, integrated_above_one=above, bound_count_tolerance=1e-12,
                area_fraction_below_zero=float(A[q < -1e-12].sum() / A.sum()),
                area_fraction_above_one=float(A[q > 1 + 1e-12].sum() / A.sum()),
                physical_mass_residual=float((np.dot(q, A) - np.dot(v, src)) / np.dot(abs(v), src)),
                L1=total[0], L2=total[1], Linf=total[2], geometric_E2=geo[1],
                missing_fraction=float(A[W == 0].sum() / A.sum()), dropped_area=dropped_area(info, src, k),
                package_weight_seconds=info['seconds'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-n', type=int, default=8)
    p.add_argument('--target-n', type=int, default=11)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--disk-tol', type=float, default=1e-10)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--out', default=str(DEFAULT_OUT))
    a = p.parse_args(argv)
    if a.source_n < 1 or a.target_n < 1 or a.k < 0 or a.disk_tol <= 0:
        p.error('Invalid mesh sizes, subdivision depth or tolerance')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    disk = disk_case(a.source_n, a.target_n, a.k, a.seed, out, a.disk_tol)
    (out / 'disk_diagnostics.json').write_text(json.dumps(disk, indent=2))
    write_metadata(out / 'metadata.json', vars(a))
    print('disk', json.dumps(disk))


if __name__ == '__main__':
    main()
