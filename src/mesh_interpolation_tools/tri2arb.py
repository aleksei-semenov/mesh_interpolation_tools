import numpy as np
import matplotlib.pyplot as plt
import math


# ----------------------------
# Geometry helpers
# ----------------------------

def triangle_area(vertices):
    """Calculate area of a triangle given its vertices (3x2 array)."""
    a, b, c = vertices
    return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))


def subdivide_triangle_iterative(triangle, k):
    """Subdivide a triangle into 4^k smaller ones, return centroids."""
    triangles = [triangle.copy()]
    for _ in range(k):
        new_tris = []
        for tri in triangles:
            A, B, C = tri
            AB = (A + B) / 2
            BC = (B + C) / 2
            CA = (C + A) / 2
            new_tris.extend([
                np.array([A, AB, CA]),
                np.array([AB, B, BC]),
                np.array([CA, BC, C]),
                np.array([AB, BC, CA]),
            ])
        triangles = new_tris
    return [np.mean(t, axis=0) for t in triangles]


def _prepare_axis(arr, name="axis"):
    arr = np.asarray(arr, dtype=float).ravel()
    arr = np.sort(np.unique(arr))
    diffs = np.diff(arr)
    if diffs.size == 0 or np.min(diffs) <= 0:
        raise ValueError(f"{name} must have increasing unique values.")
    return arr, float(np.min(diffs))


def compute_required_subdivisions(triangle_vertices, target_cell_size, *, min_k=0, max_k=6):
    edges = [
        np.linalg.norm(triangle_vertices[0] - triangle_vertices[1]),
        np.linalg.norm(triangle_vertices[1] - triangle_vertices[2]),
        np.linalg.norm(triangle_vertices[2] - triangle_vertices[0]),
    ]
    tri_diam = max(edges)
    EPS = 1e-12
    if tri_diam < EPS:
        return min_k
    cell = max(target_cell_size, EPS)
    ratio = tri_diam / (0.5 * cell)
    if ratio <= 1.0:
        return min_k
    ratio = min(ratio, 2.0 ** max_k)
    return int(max(min_k, min(max_k, math.ceil(math.log2(ratio)))))


# ----------------------------
# Interpolator
# ----------------------------

def tri2arb(nodes, elem, field_at_nodes,
                                        grid_x=None, grid_y=None, grid_size=100,
                                        elem_area=None):
    """
    Conservative interpolation from triangular mesh to (possibly non-uniform) structured grid.

    Parameters
    ----------
    nodes : (2, nod_n) array
        Node coordinates (x,y).
    elem : (3, elem_n) array
        Node indices for each triangular element.
    field_at_nodes : (nod_n,) array
        Scalar values per node.
    grid_x, grid_y : 1D arrays
        Non-uniform grid axes (cell boundaries).
    elem_area : (elem_n,) array, optional
        Precomputed element areas.

    Returns
    -------
    grid_x, grid_y, heatmap
    """
    node_coords = nodes.T       # shape (nod_n, 2)
    elem_conn = elem.T          # shape (elem_n, 3)

    elem_values = np.array([field_at_nodes[tri].mean() for tri in elem_conn], dtype=float)

    if grid_x is None:
        grid_x = np.linspace(node_coords[:,0].min(), node_coords[:,0].max(), grid_size)
    if grid_y is None:
        grid_y = np.linspace(node_coords[:,1].min(), node_coords[:,1].max(), grid_size)

    grid_x, dx_min = _prepare_axis(grid_x, "grid_x")
    grid_y, dy_min = _prepare_axis(grid_y, "grid_y")
    target_cell_size = min(dx_min, dy_min)

    nx = len(grid_x) - 1
    ny = len(grid_y) - 1
    heatmap = np.zeros((ny, nx), dtype=float)
    area_map = np.zeros((ny, nx), dtype=float)

    for e_idx, tri_idx in enumerate(elem_conn):
        tri_vertices = node_coords[tri_idx]
        value = elem_values[e_idx]
        A = elem_area[e_idx] if elem_area is not None else triangle_area(tri_vertices)

        k = compute_required_subdivisions(tri_vertices, target_cell_size)
        centroids = subdivide_triangle_iterative(tri_vertices, k)
        sub_area = A / max(len(centroids), 1)

        for (x, y) in centroids:
            i = np.searchsorted(grid_x, x, side='right') - 1
            j = np.searchsorted(grid_y, y, side='right') - 1
            if 0 <= i < nx and 0 <= j < ny:
                heatmap[j, i] += value * sub_area
                area_map[j, i] += sub_area

    mask = area_map > 0
    heatmap[mask] /= area_map[mask]

    return grid_x, grid_y, heatmap


# ----------------------------
# Example with Random Non-Uniform Grid
# ----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- Generate synthetic triangular mesh ---
    n_nodes = 60
    nodes = rng.random((2, n_nodes))   # in [0,1]^2

    n_elem = 100
    elem = np.zeros((3, n_elem), dtype=int)
    for i in range(n_elem):
        elem[:, i] = rng.choice(n_nodes, 3, replace=False)

    # Element areas
    elem_area = np.zeros(n_elem)
    for i in range(n_elem):
        tri = nodes[:, elem[:, i]].T
        elem_area[i] = triangle_area(tri)

    # Scalar field at nodes
    x, y = nodes
    field_at_nodes = np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    # --- Random non-uniform grid ---
    grid_x = np.sort(rng.random(15))   # 15 irregular boundaries in x
    grid_y = np.sort(rng.random(20))   # 20 irregular boundaries in y

    gx, gy, H = tri2arb(
        nodes, elem, field_at_nodes,
        grid_x=grid_x, grid_y=grid_y,
        elem_area=elem_area
    )

    # --- Plot ---
    plt.figure(figsize=(10,8))
    plt.pcolormesh(gx, gy, H, shading="auto", cmap="viridis")
    plt.colorbar(label="Interpolated Value")
    plt.title("Interpolation to Non-Uniform Rectangular Grid")
    plt.show()
