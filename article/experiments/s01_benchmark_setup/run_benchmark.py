"""§1 benchmark: constant and fixed piecewise-constant fields over a k-sweep.

Weights come from the package (``calculate_interpolation_weights`` at fixed k);
the reference is independent triangle/rectangle clipping. Prints the diagnostic
record for every run and saves the raw data used by §2 and §3.

    python -m article.experiments.s01_benchmark_setup.run_benchmark
    python -m article.experiments.s01_benchmark_setup.run_benchmark --source-n 32 --target-n 41 --k 0 1 2 3 4 5 6
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import save_npz

from .diagnostics import errors, write_csv, write_metadata
from .package_sampling import package_weight_matrix
from .reference_geometry import mesh, overlap_matrix, target_areas, validate

DEFAULT_OUT = Path(__file__).resolve().parent / 'results'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-n', type=int, default=12, help='source lattice intervals per side')
    p.add_argument('--target-n', type=int, default=17, help='target cells per side')
    p.add_argument('--k', type=int, nargs='+', default=list(range(6)))
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--mesh', help='Optional NPZ with nodes(N,2), triangles(M,3), x_edges, y_edges')
    p.add_argument('--out', default=str(DEFAULT_OUT))
    args = p.parse_args(argv)
    levels = sorted(set(args.k))
    if min(levels) < 0 or args.target_n < 1:
        p.error('k must be nonnegative and target-n positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mesh:
        with np.load(args.mesh, allow_pickle=False) as d:
            nodes, tri, x, y = [d[v] for v in ['nodes', 'triangles', 'x_edges', 'y_edges']]
    else:
        nodes, tri = mesh(args.source_n, seed=args.seed)
        x = y = np.linspace(0, 1, args.target_n + 1)
    a = validate(nodes, tri, x, y)
    target_a = target_areas(x, y)
    start = time.perf_counter()
    overlap = overlap_matrix(nodes, tri, x, y)
    reference_seconds = time.perf_counter() - start
    # This benchmark intentionally requires full source and target coverage, no masks.
    # Imported meshes must also be conforming and nonoverlapping (caller responsibility).
    np.testing.assert_allclose(np.asarray(overlap.sum(axis=0)).ravel(), a, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(np.asarray(overlap.sum(axis=1)).ravel(), target_a, rtol=1e-10, atol=1e-13)
    save_npz(out / 'exact_overlaps.npz', overlap)
    c = nodes[tri].mean(axis=1)
    q_source = 1 + 0.5 * np.sin(2 * np.pi * c[:, 0]) * np.cos(2 * np.pi * c[:, 1])
    ref = overlap @ q_source / target_a
    np.savez_compressed(out / 'mesh_reference.npz', nodes=nodes, triangles=tri, x_edges=x, y_edges=y,
                        source_areas=a, target_areas=target_a, q_source=q_source, q_reference=ref)
    # Select the largest overlap-to-cell ratios below unity: genuinely cut cells.
    # Area defect then prioritizes difficult cells at the coarsest tested depth.
    boundary = np.diff(overlap.indptr) > 1
    rows, comparisons = [], []
    chosen = None
    for k in levels:
        print(f'k={k}: package weights ...', flush=True)
        b, info = package_weight_matrix(nodes, tri, x, y, k, return_info=True)
        seconds = info['seconds']
        save_npz(out / f'assigned_areas_k{k}.npz', b)
        w = np.asarray(b.sum(axis=1)).ravel()
        hit = w > 0
        defect = w / target_a - 1
        deposited = b @ q_source
        qw = np.divide(deposited, w, out=np.full_like(w, np.nan), where=hit)
        qa = deposited / target_a
        cw = np.where(hit, 1., np.nan)
        ca = w / target_a
        np.savez_compressed(out / f'fields_k{k}.npz', W=w, defect=defect, constant_W=cw,
                            constant_A=ca, field_W=qw, field_A=qa)
        if chosen is None:
            candidates = np.flatnonzero(boundary)
            chosen = candidates[np.argsort(abs(defect[candidates]))[-min(6, len(candidates)):]]
        for field, source, s, outputs in [('constant', np.ones_like(a), w, {'W': cw, 'A': ca}),
                                          ('fixed', q_source, deposited, {'W': qw, 'A': qa})]:
            ms = np.dot(source, a)
            scale = np.dot(abs(source), a)
            for norm, q in outputs.items():
                complete = np.isfinite(q).all()
                mw = np.dot(q[hit], w[hit])  # zero W contributes nothing, even if q is undefined
                ma = np.dot(q, target_a) if complete else np.nan
                reference = np.ones_like(w) if field == 'constant' else ref
                e = errors(q, reference, target_a)
                cond = errors(q[hit], reference[hit], target_a[hit]) if hit.any() else [np.nan] * 3
                row = dict(field=field, normalization=norm, k=k, Ns=len(tri), Nt=len(w),
                           number_of_samples=len(tri) * 4 ** k, source_integral=ms,
                           deposited_integral=s.sum(), sampled_area_integral=mw,
                           physical_area_integral=ma, physical_residual=(ma - ms) / scale,
                           sampled_residual=(mw - ms) / scale, deposition_residual=(s.sum() - ms) / scale,
                           unassigned_source_area=a.sum() - w.sum(),
                           missing_area=target_a[~hit].sum(),
                           missing_area_fraction=target_a[~hit].sum() / target_a.sum(),
                           min_W_over_A=min(w / target_a), max_W_over_A=max(w / target_a),
                           mean_absolute_area_defect=np.dot(target_a, abs(defect)) / target_a.sum(),
                           max_absolute_area_defect=max(abs(defect)),
                           constant_error=max(abs(q - 1)) if field == 'constant' and complete else np.nan,
                           constant_error_sampled=max(abs(q[hit] - 1)) if field == 'constant' and hit.any() else np.nan,
                           L1=e[0], L2=e[1], Linf=e[2],
                           conditional_L1=cond[0], conditional_L2=cond[1], conditional_Linf=cond[2],
                           weight_build_seconds=seconds)
                rows.append(row)
                print(json.dumps(row, allow_nan=True))
        for j in chosen:
            ids = np.union1d(overlap.getrow(j).indices, b.getrow(j).indices)
            for i in ids:
                comparisons.append(dict(k=k, target_cell=int(j), target_ix=int(j % (len(x) - 1)),
                                        target_iy=int(j // (len(x) - 1)), source_triangle=int(i),
                                        exact_overlap_area=overlap[j, i], assigned_sample_area=b[j, i],
                                        area_error=b[j, i] - overlap[j, i], target_is_unsampled=not hit[j]))
    write_csv(out / 'diagnostics.csv', rows)
    rates = []
    for norm in ('W', 'A'):
        series = [r for r in rows if r['field'] == 'fixed' and r['normalization'] == norm]
        for left, right in zip(series[:-1], series[1:]):
            for metric in ('L1', 'L2', 'Linf'):
                e0, e1 = left[metric], right[metric]
                valid = np.isfinite(e0) and np.isfinite(e1) and e0 > 0 and e1 > 0
                rate = np.log(e0 / e1) / ((right['k'] - left['k']) * np.log(2)) if valid else np.nan
                rates.append(dict(normalization=norm, metric=metric, k_start=left['k'],
                                  k_end=right['k'], empirical_rate=rate))
    if rates:
        write_csv(out / 'adjacent_rates.csv', rates)
    if comparisons:
        write_csv(out / 'boundary_cell_areas.csv', comparisons)
    write_metadata(out / 'metadata.json', vars(args), levels=levels,
                   reference_seconds=reference_seconds, boundary_cells=chosen.tolist())
    # Compact LaTeX table, preserving NA instead of hiding missing-cell failures.
    lines = [r'\begin{tabular}{rrlrrrr}', r'\hline',
             r'$k$ & Field & Norm. & $|\epsilon_A|$ & $|\epsilon_W|$ & $E_2$ & Missing fraction \\', r'\hline']
    fmt = lambda v: '--' if not np.isfinite(v) else f'{abs(v):.3e}'  # noqa: E731
    for r in rows:
        lines.append(f"{r['k']} & {r['field']} & {r['normalization']} & " +
                     ' & '.join(fmt(r[t]) for t in ['physical_residual', 'sampled_residual', 'L2',
                                                    'missing_area_fraction']) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    (out / 'diagnostics_table.tex').write_text('\n'.join(lines) + '\n')
    print(f'Saved diagnostics and overlap reference to {out}. NaN means unavailable, not zero.')


if __name__ == '__main__':
    main()
