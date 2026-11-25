"""Run from the repository root: python -m pytest article/experiments"""
import numpy as np
import pytest

from .package_sampling import _tri2reg as tri2reg
from .diagnostics import errors
from .package_sampling import (check_samples_match_package, package_samples,
                               package_weight_matrix, samples_to_matrix)
from .reference_geometry import mesh, overlap_matrix, target_areas, validate


@pytest.fixture
def case():
    p, t = mesh(3)
    x, y = np.linspace(0, 1, 6), np.linspace(0, 1, 8)
    return p, t, x, y, validate(p, t, x, y), target_areas(x, y)


def test_package_is_repository_code():
    from article.common import SRC_DIR
    from pathlib import Path
    assert SRC_DIR in Path(tri2reg.__file__).resolve().parents


def test_known_clip():
    p = np.array([[0., 0.], [1., 0.], [0., 1.]])
    o = overlap_matrix(p, np.array([[0, 1, 2]]), np.array([0., .5, 1.]), np.array([0., 1.]))
    np.testing.assert_allclose(o.toarray()[:, 0], [.375, .125])


def test_overlap_partition(case):
    p, t, x, y, a, at = case
    o = overlap_matrix(p, t, x, y)
    np.testing.assert_allclose(np.asarray(o.sum(axis=0)).ravel(), a, atol=1e-14)
    np.testing.assert_allclose(np.asarray(o.sum(axis=1)).ravel(), at, atol=1e-14)


def test_fixed_depth_is_restored():
    original = tri2reg.compute_required_subdivisions
    p, t = mesh(2)
    package_weight_matrix(p, t, np.linspace(0, 1, 3), np.linspace(0, 1, 3), 2)
    assert tri2reg.compute_required_subdivisions is original


def test_package_single_triangle():
    """k=1 children centroids of the unit triangle: three left of x=0.4, one right."""
    p = np.array([[0., 0.], [1., 0.], [0., 1.]])
    b = package_weight_matrix(p, np.array([[0, 1, 2]]), np.array([0, .4, 1]), np.array([0, 1.]), 1)
    np.testing.assert_allclose(b.toarray()[:, 0], [.375, .125])


@pytest.mark.parametrize('k', [0, 1, 3])
def test_samples_match_package(case, k):
    p, t, x, y, a, at = case
    b = package_weight_matrix(p, t, x, y, k)
    parent, pts, cell, sub = package_samples(p, t, a, x, y, k)
    assert len(pts) == len(t) * 4 ** k
    check_samples_match_package(samples_to_matrix(parent, cell, sub, len(at), len(t)), b)


@pytest.mark.parametrize('k', [0, 1, 3])
def test_deposition_and_normalizations(case, k):
    p, t, x, y, a, at = case
    b = package_weight_matrix(p, t, x, y, k)
    np.testing.assert_allclose(np.asarray(b.sum(axis=0)).ravel(), a, atol=1e-14)
    q = np.arange(len(a), dtype=float)
    w = np.asarray(b.sum(axis=1)).ravel()
    s = b @ q
    hit = w > 0
    np.testing.assert_allclose(s.sum(), a @ q, atol=1e-12)
    np.testing.assert_allclose(at @ (s / at), a @ q, atol=1e-12)
    np.testing.assert_allclose(w[hit] @ (s[hit] / w[hit]), a @ q, atol=1e-12)


def test_package_centroid_mean_and_count(case):
    p, t = case[0], case[1]
    tr = p[t[0]]
    for k in range(4):
        c = np.asarray(tri2reg.subdivide_triangle_iterative(tr, k))
        assert c.shape == (4 ** k, 2)
        np.testing.assert_allclose(c.mean(axis=0), tr.mean(axis=0), atol=1e-14)


def test_undefined_error():
    assert np.isnan(errors(np.array([np.nan]), np.array([1.]), np.array([1.]))).all()


def test_invalid_connectivity(case):
    p, t, x, y = case[:4]
    t = t.copy()
    t[0, 0] = -1
    with pytest.raises(ValueError):
        validate(p, t, x, y)
