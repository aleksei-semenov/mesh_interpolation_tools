"""Smooth-field remapping with the package's subdivision-centroid method.

Shared by §4 (subdivision sweep) and §6 (source/target refinement sweeps).

- Source values are true cell averages (independent Duffy-product Gauss rule,
  verified with a higher order).
- Reconstructions are mean-preserving least-squares fits to cell averages:
  ``constant`` (the package's own setting) and ``linear`` (benchmark code, the
  package has no reconstruction yet). ``quadratic`` is available but optional.
- Deposition: for ``constant`` the package's weights are used directly
  (S = B q). For ``linear``/``quadratic`` the reconstruction is evaluated at the
  package's child centroids, which are checked to give exactly the package's
  weights. So every reconstruction uses the centroid (degree-one) rule.
- References: exact polygon-clipped integrals of the reconstruction
  (``overlap``) and analytic target averages (``analytic``).
"""
import time

import numpy as np
from scipy.special import roots_legendre

from ..s01_benchmark_setup.diagnostics import metrics, normalize
from ..s01_benchmark_setup.package_sampling import (
    check_samples_match_package, dropped_area, package_samples, package_weight_matrix,
    samples_to_matrix)
from ..s01_benchmark_setup.reference_geometry import area, build_case

RECONSTRUCTIONS = ('constant', 'linear')


def field(name, x, y):
    if name == 'sine':
        return 1 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    if name == 'quadratic':
        return 1 + 0.3 * x + 0.2 * y + 0.4 * x * x - 0.2 * x * y + 0.1 * y * y
    raise ValueError(name)


def quad_nodes(triangles, degree):
    """Centroid (degree 1) or symmetric three-node degree-2 rule; equal weights.
    Used only for the *reference* integrals, never for the method."""
    if degree not in (1, 2):
        raise ValueError('quadrature degree must be 1 or 2')
    b = np.array([[1 / 3, 1 / 3, 1 / 3]]) if degree == 1 else \
        np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    return np.einsum('qv,nvd->nqd', b, triangles)


def gauss_triangle(t, fun, n=10):
    """Independent Duffy-product Gaussian integration; returns the cell mean."""
    z, w = roots_legendre(n)
    z = (z + 1) / 2
    w = w / 2
    u, v = np.meshgrid(z, z, indexing='ij')
    wu, wv = np.meshgrid(w, w, indexing='ij')
    p = t[0] + u[..., None] * (t[1] - t[0]) + (1 - u[..., None]) * v[..., None] * (t[2] - t[0])
    return float(np.sum(2 * wu * wv * (1 - u) * fun(p[..., 0], p[..., 1])))


def target_average(name, x, y, n=10):
    """Independent tensor Gauss rule on each rectangle; flattened j * nx + i."""
    z, w = roots_legendre(n)
    z = (z + 1) / 2
    w = w / 2
    xx = x[:-1, None] + np.diff(x)[:, None] * z
    yy = y[:-1, None] + np.diff(y)[:, None] * z
    return np.array([[np.sum(w[:, None] * w[None, :] * field(name, X[:, None], Y[None, :]))
                      for X in xx] for Y in yy]).ravel()


def moments(t):
    c = t.mean(axis=0)
    d = t - c
    # Uniform triangle covariance = sum(vertex offsets outer products)/12.
    return c, np.einsum('ni,nj->ij', d, d) / 12


def reconstruct(nodes, triangles, values, method):
    """Unweighted least-squares fit to cell averages; mean-preserving basis.
    Full two-ring vertex neighborhood, expanding if underdetermined. Fail on
    rank deficiency instead of silently falling back to a lower-order fit.
    """
    ts = nodes[triangles]
    cc = np.array([moments(t)[0] for t in ts])
    cv = np.array([moments(t)[1] for t in ts])
    n = len(ts)
    if method == 'constant':
        return cc, cv, np.zeros((n, 5))
    if method not in ('linear', 'quadratic'):
        raise ValueError(method)
    vertex = [set() for _ in nodes]
    for i, tri in enumerate(triangles):
        for v in tri:
            vertex[v].add(i)
    neigh = [set().union(*(vertex[v] for v in tri)) - {i} for i, tri in enumerate(triangles)]
    coef = np.zeros((n, 5))
    dim = 2 if method == 'linear' else 5
    for i in range(n):
        pool = set(neigh[i])
        frontier = set(pool)
        for _ in range(3):
            if len(pool) >= max(2 * dim, dim + 3):
                break
            frontier = set().union(*(neigh[j] for j in frontier)) - pool - {i} if frontier else set()
            pool |= frontier
        ids = np.array(sorted(pool), dtype=int)
        d = cc[ids] - cc[i]
        h = np.sqrt(area(ts[i]))
        dx, dy = d[:, 0], d[:, 1]
        # Average of parent's centered basis over neighboring triangles.
        X = np.column_stack([dx / h, dy / h, (dx * dx + cv[ids, 0, 0] - cv[i, 0, 0]) / h ** 2,
                             (dx * dy + cv[ids, 0, 1] - cv[i, 0, 1]) / h ** 2,
                             (dy * dy + cv[ids, 1, 1] - cv[i, 1, 1]) / h ** 2])[:, :dim]
        fit, _, rank, s = np.linalg.lstsq(X, values[ids] - values[i], rcond=None)
        if rank < dim or s[-1] / s[0] < 1e-12:
            raise ValueError(f'Rank-deficient {method} fit for triangle {i}; use a larger/well-shaped mesh')
        coef[i, :dim] = fit / np.array([h, h, h * h, h * h, h * h])[:dim]
    return cc, cv, coef


def evaluate(p, i, values, cc, cv, coef):
    """Reconstruction of parent(s) ``i`` at points ``p``; ``i`` scalar or per point."""
    d = p - cc[i]
    x, y = d[..., 0], d[..., 1]
    c = coef[i]
    return (values[i] + c[..., 0] * x + c[..., 1] * y + c[..., 2] * (x * x - cv[i, 0, 0])
            + c[..., 3] * (x * y - cv[i, 0, 1]) + c[..., 4] * (y * y - cv[i, 1, 1]))


def overlap_integral(polys, n_target, values, rec):
    """Exact degree-2 polynomial integral over clipped polygons (floating point)."""
    total = np.zeros(n_target)
    cc, cv, co = rec
    for i, j, p in polys:
        for v in range(1, len(p) - 1):
            tri = np.array([p[0], p[v], p[v + 1]])
            a = area(tri)
            if a:
                total[j] += a * np.mean(evaluate(quad_nodes(tri[None, :, :], 2)[0], i, values, cc, cv, co))
    return total


def package_deposits(case, k, values, recs):
    """Deposited quantities S per reconstruction, sampled areas W, dropped area.

    One package weight computation per k, shared by all reconstructions.
    """
    nodes, tri, x, y, src, A = case[:6]
    B, info = package_weight_matrix(nodes, tri, x, y, k, return_info=True)
    W = np.asarray(B.sum(axis=1)).ravel()
    out = {}
    samples = None
    for name, rec in recs.items():
        if name == 'constant':
            out[name] = B @ values
            continue
        if samples is None:
            samples = package_samples(nodes, tri, src, x, y, k)
            parent, pts, cell, sub = samples
            check_samples_match_package(samples_to_matrix(parent, cell, sub, len(A), len(tri)), B)
        parent, pts, cell, sub = samples
        ok = cell >= 0
        q = evaluate(pts[ok], parent[ok], values, *rec)
        out[name] = np.bincount(cell[ok], weights=sub[ok] * q, minlength=len(A))
    return out, W, dropped_area(info, src, k), info['seconds']


def smooth_group(source_n, target_n, levels, fieldname, seed, quad_order, sweep,
                 reconstructions=RECONSTRUCTIONS):
    case = build_case(source_n, target_n, seed)
    nodes, tri, x, y, src, A, polys, cover = case
    fun = lambda X, Y: field(fieldname, X, Y)  # noqa: E731
    values = np.array([gauss_triangle(t, fun, quad_order) for t in nodes[tri]])
    check = np.array([gauss_triangle(t, fun, quad_order + 4) for t in nodes[tri]])
    delta = float(np.max(abs(values - check)))
    if delta > 1e-10:
        raise ValueError(f'Source quadrature verification failed: {delta}; increase --quad-order')
    target = target_average(fieldname, x, y, quad_order)
    if np.max(abs(target - target_average(fieldname, x, y, quad_order + 4))) > 1e-10:
        raise ValueError('Target quadrature verification failed; increase --quad-order')
    recs = {r: reconstruct(nodes, tri, values, r) for r in reconstructions}
    exact = {r: overlap_integral(polys, len(A), values, rec) / A for r, rec in recs.items()}
    mass, B = np.dot(values, src), np.dot(abs(values), src)
    rows = []
    for k in levels:
        begin = time.perf_counter()
        deposits, W, dropped, weight_seconds = package_deposits(case, k, values, recs)
        elapsed = time.perf_counter() - begin
        missing = A[W == 0].sum() / A.sum()
        for recname in reconstructions:
            S = deposits[recname]
            for method in ('W', 'A'):
                q = normalize(S, W, A, method)
                e = metrics(q, target, A)
                ge = metrics(q, exact[recname], A)
                rep = metrics(exact[recname], target, A)
                rows.append(dict(sweep=sweep, field=fieldname, source_n=source_n, target_n=target_n,
                                 Ns=len(tri), Nt=len(A), hS=np.sqrt(1 / len(tri)), hT=np.sqrt(1 / len(A)),
                                 reconstruction=recname, normalization=method, quadrature='package centroids',
                                 k=k, samples=len(tri) * 4 ** k,
                                 E1=e[0], E2=e[1], Einf=e[2], geometric_E2=ge[1], representation_E2=rep[1],
                                 missing_fraction=missing,
                                 physical_residual=(np.dot(q, A) - mass) / B if np.isfinite(q).all() else np.nan,
                                 sampled_residual=(np.dot(q[W > 0], W[W > 0]) - mass) / B,
                                 dropped_area=dropped, package_weight_seconds=weight_seconds,
                                 deposit_seconds=elapsed, source_quadrature_check=delta, pk=np.nan))
    # Adjacent total-error rates only within a fixed mesh pair/method/field.
    for recname in reconstructions:
        for method in ('W', 'A'):
            rr = sorted([r for r in rows if r['reconstruction'] == recname and r['normalization'] == method],
                        key=lambda r: r['k'])
            for a, b in zip(rr[:-1], rr[1:]):
                if (a['missing_fraction'] == 0 and b['missing_fraction'] == 0 and a['E2'] > 0 and b['E2'] > 0
                        and np.isfinite(a['E2']) and np.isfinite(b['E2'])):
                    b['pk'] = np.log(a['E2'] / b['E2']) / ((b['k'] - a['k']) * np.log(2))
    return rows
