import numpy as np
import pytest
from scipy.spatial import Delaunay

import mesh_interpolation_tools as mit


def _mesh(n=40, seed=0):
    rng = np.random.default_rng(seed)
    nodes = rng.random((2, n))
    elem = Delaunay(nodes.T).simplices.T
    x, y = nodes
    field = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    return nodes, elem, field


def test_version():
    assert isinstance(mit.__version__, str)


def test_tri2reg():
    nodes, elem, field = _mesh()
    gx, gy = np.linspace(0, 1, 20), np.linspace(0, 1, 20)
    _, _, heat = mit.tri2reg(nodes, elem, field, gx, gy, n_processes=1)
    assert np.isfinite(np.nan_to_num(heat)).all()


def test_tri2arb():
    nodes, elem, field = _mesh()
    gx = np.array([0, 0.1, 0.4, 1.0])
    gy = np.array([0, 0.2, 0.5, 1.0])
    _, _, heat = mit.tri2arb(nodes, elem, field, gx, gy, n_processes=1)
    assert np.isfinite(np.nan_to_num(heat)).all()


def test_tri2tri():
    nodes, elem, field = _mesh()
    out = mit.conservative_interp_tri2tri(nodes, elem, field, nodes, elem, n_processes=1)
    assert np.asarray(out).size > 0


def test_healpix_import():
    pytest.importorskip("healpy")
    assert callable(mit.conservative_interp_spherical_tri2healpix)
