import numpy as np

from ..s01_benchmark_setup.reference_geometry import area, build_case, intersection_polygons, mesh
from .smooth import (evaluate, field, gauss_triangle, overlap_integral, package_deposits, quad_nodes,
                     reconstruct, smooth_group, target_average)


def test_affine_and_quadratic_rules():
    t = np.array([[0., 0.], [1., 0.], [0., 1.]])
    f = lambda x, y: x * x + y * y + x * y + 2 * x  # noqa: E731
    exact = gauss_triangle(t, f, 12)
    pts = quad_nodes(t[None, :, :], 2)[0]
    assert abs(np.mean(f(pts[:, 0], pts[:, 1])) - exact) < 1e-12
    p = quad_nodes(t[None, :, :], 1)[0]
    assert abs(np.mean(f(p[:, 0], p[:, 1])) - exact) > .01
    lin = lambda x, y: 1 + 2 * x - 3 * y  # noqa: E731
    assert abs(float(lin(*p[0])) - gauss_triangle(t, lin)) < 1e-12


def test_reconstruction_and_clipped_overlap():
    nodes, tri = mesh(5)
    vals = np.array([gauss_triangle(t, lambda x, y: field('quadratic', x, y)) for t in nodes[tri]])
    rec = reconstruct(nodes, tri, vals, 'quadratic')
    for i, t in enumerate(nodes[tri]):
        pts = quad_nodes(t[None, :, :], 2)[0]
        np.testing.assert_allclose(evaluate(pts, i, vals, *rec), field('quadratic', pts[:, 0], pts[:, 1]),
                                   atol=2e-12)
    x, y = np.linspace(0, 1, 5), np.linspace(0, 1, 6)
    polys, cover, parent = intersection_polygons(nodes, tri, x, y)
    A = np.outer(np.diff(y), np.diff(x)).ravel()
    np.testing.assert_allclose(cover, A, atol=1e-12)
    exact = overlap_integral(polys, len(A), vals, rec)
    np.testing.assert_allclose(exact / A, target_average('quadratic', x, y), atol=2e-12)
    assert abs(np.sum(exact) - np.dot(vals, [area(t) for t in nodes[tri]])) < 1e-12


def test_linear_reconstruction_exact_for_linear_field():
    nodes, tri = mesh(5)
    lin = lambda x, y: 1 + 2 * x - 3 * y  # noqa: E731
    vals = np.array([gauss_triangle(t, lin) for t in nodes[tri]])
    rec = reconstruct(nodes, tri, vals, 'linear')
    pts = nodes[tri].mean(axis=1) + 0.01
    np.testing.assert_allclose(evaluate(pts, np.arange(len(tri)), vals, *rec), lin(pts[:, 0], pts[:, 1]),
                               atol=1e-12)


def test_package_deposits_conserve_and_match():
    case = build_case(4, 5)
    nodes, tri, x, y, src, A = case[:6]
    vals = np.linspace(1, 2, len(tri))
    recs = {r: reconstruct(nodes, tri, vals, r) for r in ('constant', 'linear')}
    dep, W, dropped, _ = package_deposits(case, 2, vals, recs)
    assert dropped == 0
    np.testing.assert_allclose(W.sum(), src.sum(), rtol=1e-13)
    for S in dep.values():  # both reconstructions are mean preserving per parent
        np.testing.assert_allclose(S.sum(), vals @ src, rtol=1e-12)


def test_smooth_group_runs():
    rows = smooth_group(4, 5, [0, 2], 'sine', 1729, 10, 'subdivision')
    assert {r['reconstruction'] for r in rows} == {'constant', 'linear'}
    assert all(np.isfinite(r['representation_E2']) for r in rows)
