"""Benchmark meshes and the *independent* exact-overlap reference.

Nothing here is the method under test: these functions build the planar
unit-square benchmark and compute triangle/rectangle intersections by polygon
clipping, which serve as the reference that the package's subdivision-centroid
weights are compared against. The method itself comes from the package, see
``package_sampling.py``.

Array convention in the benchmark: ``nodes`` is (N, 2) and ``triangles`` is
(M, 3). The package uses the transposed arrays; ``package_sampling`` converts.
"""
from collections import namedtuple

import numpy as np
from scipy.sparse import coo_matrix


def area(poly):
    """Shoelace area of a polygon given as (n, 2) vertices."""
    p = np.asarray(poly)
    if len(p) < 3:
        return 0.0
    # Translation avoids unnecessary cancellation for small polygons.
    p = p - p[0]
    return abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
               np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2


def mesh(n, jitter=0.17, seed=1729):
    """Nonmatching conforming unit-square mesh: an n x n lattice, every square
    split into two triangles (alternating diagonals), with interior vertices
    jittered reproducibly by up to ``jitter / n``."""
    if n < 1 or not 0 <= jitter < 0.25:
        raise ValueError('n >= 1 and 0 <= jitter < 0.25 required')
    xx, yy = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    p = np.column_stack([xx.ravel(), yy.ravel()])
    interior = ((p > 0) & (p < 1)).all(axis=1)
    p[interior] += np.random.default_rng(seed).uniform(-jitter / n, jitter / n,
                                                       (interior.sum(), 2))
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + n + 2, a + n + 1
            cells.extend([(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else
                         [(a, b, d), (b, c, d)])
    return p, np.asarray(cells)


def target_edges(n, shift=False):
    """Cell boundaries of an n-cell axis on [0, 1]. With ``shift``, interior
    boundaries move by half a nominal cell while the exterior stays fixed."""
    if not shift:
        return np.linspace(0, 1, n + 1)
    return np.r_[0., (np.arange(1, n) + .5) / n, 1.]


def validate(nodes, triangles, x, y):
    """Check inputs and return the source triangle areas."""
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise ValueError('nodes must be finite (N,2)')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu':
        raise ValueError('triangles must be integer (M,3)')
    if triangles.min() < 0 or triangles.max() >= len(nodes):
        raise ValueError('invalid triangle index')
    for e in (x, y):
        if e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all() or not (np.diff(e) > 0).all():
            raise ValueError('edges must be strictly increasing finite 1D arrays')
    a = np.array([area(t) for t in nodes[triangles]])
    if (a <= 0).any():
        raise ValueError('degenerate triangle')
    return a


def target_areas(x, y):
    """Physical target cell areas A^T, flattened as j * nx + i."""
    return np.outer(np.diff(y), np.diff(x)).ravel()


def clip(poly, axis, edge, keep_greater):
    """Sutherland-Hodgman half-plane clipping; no sampling used."""
    if len(poly) == 0:
        return np.empty((0, 2))
    out = []
    prev = poly[-1]
    prev_in = prev[axis] >= edge if keep_greater else prev[axis] <= edge
    for cur in poly:
        cur_in = cur[axis] >= edge if keep_greater else cur[axis] <= edge
        if cur_in != prev_in:
            t = (edge - prev[axis]) / (cur[axis] - prev[axis])
            out.append(prev + t * (cur - prev))
        if cur_in:
            out.append(cur)
        prev, prev_in = cur, cur_in
    return np.asarray(out).reshape(-1, 2)


def intersection_polygons(nodes, triangles, x, y, min_area=1e-18):
    """All triangle/rectangle intersection polygons.

    Returns ``polys`` as a list of (source i, target j, polygon), plus the
    covered area per target cell and per source triangle.
    """
    nx, ny = len(x) - 1, len(y) - 1
    polys = []
    cover = np.zeros(nx * ny)
    source_cover = np.zeros(len(triangles))
    for i, t in enumerate(nodes[triangles]):
        lo, hi = t.min(axis=0), t.max(axis=0)
        x0 = max(0, np.searchsorted(x, lo[0], side='right') - 1)
        x1 = min(nx - 1, np.searchsorted(x, hi[0], side='left'))
        y0 = max(0, np.searchsorted(y, lo[1], side='right') - 1)
        y1 = min(ny - 1, np.searchsorted(y, hi[1], side='left'))
        for iy in range(y0, y1 + 1):
            for ix in range(x0, x1 + 1):
                poly = t.copy()
                for axis, edge, greater in ((0, x[ix], True), (0, x[ix + 1], False),
                                            (1, y[iy], True), (1, y[iy + 1], False)):
                    poly = clip(poly, axis, edge, greater)
                a = area(poly)
                if a > min_area:
                    j = iy * nx + ix
                    polys.append((i, j, poly))
                    cover[j] += a
                    source_cover[i] += a
    return polys, cover, source_cover


def polygons_to_matrix(polys, n_target, n_source):
    """Sparse (n_target, n_source) matrix of exact overlap areas a_ji."""
    if not polys:
        return coo_matrix((n_target, n_source)).tocsr()
    rows = [j for _, j, _ in polys]
    cols = [i for i, _, _ in polys]
    vals = [area(p) for _, _, p in polys]
    return coo_matrix((vals, (rows, cols)), shape=(n_target, n_source)).tocsr()


def overlap_matrix(nodes, triangles, x, y):
    """Floating-point polygon overlap areas; independent of subdivision depth."""
    polys, _, _ = intersection_polygons(nodes, triangles, x, y)
    return polygons_to_matrix(polys, (len(x) - 1) * (len(y) - 1), len(triangles))


Case = namedtuple('Case', 'nodes triangles x y source_areas target_areas polys cover')


def build_case(source_n, target_n, seed=1729, jitter=.17, shift=False, mask=False):
    """Source mesh + target grid + exact intersections, with coverage checks.

    ``mask`` removes a central patch of source triangles (|x-.5|,|y-.5| < .13).
    """
    nodes, tri = mesh(source_n, jitter=jitter, seed=seed)
    if mask:
        c = nodes[tri].mean(axis=1)
        tri = tri[~((abs(c[:, 0] - .5) < .13) & (abs(c[:, 1] - .5) < .13))]
    x = y = target_edges(target_n, shift)
    src = validate(nodes, tri, x, y)
    A = target_areas(x, y)
    polys, cover, parent = intersection_polygons(nodes, tri, x, y)
    if not np.allclose(parent, src, rtol=1e-9, atol=1e-12):
        raise ValueError('Source is not fully within target domain')
    if not mask and not np.allclose(cover, A, rtol=1e-9, atol=1e-12):
        raise ValueError('Source and target coverage disagree')
    if np.any(cover > A + 1e-10):
        raise ValueError('Source mesh overlaps itself')
    return Case(nodes, tri, x, y, src, A, polys, cover)
