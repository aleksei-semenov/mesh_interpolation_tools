import numpy as np

from ..s01_benchmark_setup.reference_geometry import mesh
from .disk import RADIUS, disk_integral_triangle, disk_rectangle


def test_disk_reference():
    nodes, tri = mesh(5)
    total = sum(disk_integral_triangle(t) for t in nodes[tri])
    assert abs(total - np.pi * RADIUS ** 2) < 1e-8
    edges = np.linspace(0, 1, 8)
    summed = sum(disk_rectangle(edges[i], edges[i + 1], edges[j], edges[j + 1])
                 for i in range(7) for j in range(7))
    assert abs(summed - np.pi * RADIUS ** 2) < 1e-8
