"""The method under test, taken from the package: subdivision-centroid weights.

Everything that decides *where* source area is deposited comes from
``mesh_interpolation_tools.tri2reg`` (imported from this repository's ``src/``
via ``article.common``):

``package_weight_matrix``
    Runs the package's own ``calculate_interpolation_weights``: midpoint
    subdivision (``subdivide_triangle_iterative``), centroid-to-cell assignment
    (``searchsorted``), equal child areas, and accumulation. The only change is
    that the automatic depth choice (``compute_required_subdivisions``) is
    replaced by a fixed ``k`` while the call runs, so that the benchmark can
    sweep k. Runs serially, because the replacement only exists in this process.

``package_samples``
    For reconstructions that need child positions (linear), the centroids
    come from the package's ``subdivide_triangle_iterative``, and are assigned
    with the package's rule. ``check_samples_match_package`` verifies that this
    gives the same weight matrix as ``package_weight_matrix``.
"""
import importlib
import time
from contextlib import contextmanager

import numpy as np
from scipy.sparse import coo_matrix

from article.common import mit  # noqa: F401  (guarantees import from src/)

# The package re-exports the *function* ``tri2reg`` under the submodule's name,
# so ``from mesh_interpolation_tools import tri2reg`` would give the function.
_tri2reg = importlib.import_module('mesh_interpolation_tools.tri2reg')

PACKAGE_FILE = _tri2reg.__file__


@contextmanager
def fixed_depth(k):
    """Temporarily make the package use subdivision depth ``k`` everywhere."""
    original = _tri2reg.compute_required_subdivisions
    _tri2reg.compute_required_subdivisions = lambda *args, **kwargs: k
    try:
        yield
    finally:
        _tri2reg.compute_required_subdivisions = original


def package_weight_matrix(nodes, triangles, x, y, k, return_info=False):
    """Assigned-area matrix B (n_target x n_source), rows ``j * nx + i``,
    computed by the package's ``calculate_interpolation_weights`` at depth k.

    W = B.sum(axis=1) are the sampled areas, S = B @ q the deposited quantities.
    With ``return_info``, also returns ``dict(seconds, assigned_samples)``:
    the package's wall time and the number of assigned samples per source
    triangle (out of 4**k).
    """
    nx, ny = len(x) - 1, len(y) - 1
    start = time.perf_counter()
    with fixed_depth(k):
        weights, gx, gy, area_map = _tri2reg.calculate_interpolation_weights(
            np.asarray(nodes, float).T, np.asarray(triangles).T,
            grid_x=np.asarray(x, float), grid_y=np.asarray(y, float), n_processes=1)
    seconds = time.perf_counter() - start
    if not (np.array_equal(gx, x) and np.array_equal(gy, y)):
        raise ValueError('package changed the target axes')
    if weights:
        e, j, i, w = (np.asarray(v) for v in zip(*weights))
    else:
        e = j = i = np.empty(0, int)
        w = np.empty(0)
    B = coo_matrix((w, (j * nx + i, e)), shape=(nx * ny, len(triangles))).tocsr()
    np.testing.assert_allclose(np.asarray(B.sum(axis=1)).ravel(), area_map.ravel(),
                               rtol=1e-12, atol=1e-15)
    if not return_info:
        return B
    return B, dict(seconds=seconds,
                   assigned_samples=np.bincount(e.astype(int), minlength=len(triangles)))


def dropped_area(info, source_areas, k):
    """Source area whose child centroids fell outside the target grid."""
    return float(np.dot(4 ** k - info['assigned_samples'], source_areas) / 4 ** k)


def package_samples(nodes, triangles, source_areas, x, y, k):
    """Child centroids from the package, assigned with the package's rule.

    Returns ``parent`` (n,), ``points`` (n, 2), ``cell`` (n,; -1 = outside the
    grid) and ``sub_area`` (n,), in the package's centroid order.
    """
    nx, ny = len(x) - 1, len(y) - 1
    points, parent, sub_area = [], [], []
    for i, t in enumerate(nodes[triangles]):
        c = np.asarray(_tri2reg.subdivide_triangle_iterative(np.asarray(t, float), k))
        points.append(c)
        parent.append(np.full(len(c), i))
        sub_area.append(np.full(len(c), source_areas[i] / len(c)))  # as in the package
    points = np.concatenate(points)
    parent = np.concatenate(parent)
    sub_area = np.concatenate(sub_area)
    # Package rule (tri2reg._process_element_weights): searchsorted(side='right') - 1,
    # samples outside [0, nx) x [0, ny) are dropped.
    ix = np.searchsorted(x, points[:, 0], side='right') - 1
    iy = np.searchsorted(y, points[:, 1], side='right') - 1
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cell = np.where(inside, iy * nx + ix, -1)
    return parent, points, cell, sub_area


def samples_to_matrix(parent, cell, sub_area, n_target, n_source):
    ok = cell >= 0
    return coo_matrix((sub_area[ok], (cell[ok], parent[ok])),
                      shape=(n_target, n_source)).tocsr()


def check_samples_match_package(samples_matrix, package_matrix, rtol=1e-12):
    """Raise if the sample-based matrix differs from the package's own weights."""
    scale = max(abs(package_matrix).max(), 1e-300)
    diff = abs(samples_matrix - package_matrix)
    worst = diff.max() if diff.nnz else 0.0
    if worst > rtol * scale:
        raise AssertionError(f'package_samples differs from package weights by {worst:.3e}')
    return worst
