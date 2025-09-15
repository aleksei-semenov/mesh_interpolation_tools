import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay

# --------------------------------
# Geometry helpers
# --------------------------------
def triangle_area(vertices):
    a, b, c = vertices
    return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))

def subdivide_triangle_iterative(triangle, k):
    """Subdivide triangle into 4^k smaller ones, return centroids."""
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

def compute_required_subdivisions(triangle_vertices, target_size, min_k=0, max_k=6):
    """Choose subdivision depth based on triangle size relative to target size."""
    edges = [
        np.linalg.norm(triangle_vertices[0] - triangle_vertices[1]),
        np.linalg.norm(triangle_vertices[1] - triangle_vertices[2]),
        np.linalg.norm(triangle_vertices[2] - triangle_vertices[0]),
    ]
    tri_diam = max(edges)
    EPS = 1e-12
    if tri_diam < EPS:
        return min_k
    ratio = tri_diam / (0.5 * target_size)
    if ratio <= 1.0:
        return min_k
    ratio = min(ratio, 2.0 ** max_k)
    return int(max(min_k, min(max_k, math.ceil(math.log2(ratio)))))

# --------------------------------
# Point location in source mesh
# --------------------------------
def point_in_triangle(pt, tri):
    """Check if point lies in triangle using barycentric coordinates."""
    A, B, C = tri
    v0 = C - A
    v1 = B - A
    v2 = pt - A
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-14)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def interpolate_from_source(pt, nodes_src, elem_src, elem_values):
    """Find which source triangle contains pt and return its value."""
    for e_idx, tri_idx in enumerate(elem_src.T):
        tri = nodes_src[:, tri_idx].T
        if point_in_triangle(pt, tri):
            return elem_values[e_idx]
    return np.nan  # outside domain

# --------------------------------
# Interpolator: triangle-to-triangle
# --------------------------------
def conservative_interp_tri2tri(nodes_src, elem_src, field_src,
                                nodes_tgt, elem_tgt,
                                k_subdivide=3):
    """
    Conservative interpolation from source triangular mesh to target triangular mesh.
    
    Parameters
    ----------
    nodes_src : (2, Nsrc) array
    elem_src  : (3, Msrc) array
    field_src : (Nsrc,) array
    nodes_tgt : (2, Ntgt) array
    elem_tgt  : (3, Mtgt) array
    
    Returns
    -------
    field_tgt : (Mtgt,) array of interpolated values per target triangle
    """
    nodes_src = nodes_src
    nodes_tgt = nodes_tgt
    elem_src = elem_src
    elem_tgt = elem_tgt

    # Assign mean value to each source element
    elem_values_src = np.array([field_src[tri].mean() for tri in elem_src.T])

    field_tgt = np.zeros(elem_tgt.shape[1])

    for e_idx, tri_idx in enumerate(elem_tgt.T):
        tri_vertices = nodes_tgt[:, tri_idx].T
        A = triangle_area(tri_vertices)

        # Subdivide target triangle into smaller centroids
        centroids = subdivide_triangle_iterative(tri_vertices, k_subdivide)
        sub_area = A / len(centroids)

        val_sum = 0.0
        area_sum = 0.0

        for c in centroids:
            val = interpolate_from_source(c, nodes_src, elem_src, elem_values_src)
            if not np.isnan(val):
                val_sum += val * sub_area
                area_sum += sub_area

        field_tgt[e_idx] = val_sum / area_sum if area_sum > 0 else np.nan

    return field_tgt


# --------------------------------
# Example usage
# --------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Source mesh (random points → Delaunay triangulation)
    n_src_nodes = 50
    nodes_src = rng.random((2, n_src_nodes))
    tri_src = Delaunay(nodes_src.T)
    elem_src = tri_src.simplices.T

    # Target mesh (fewer points, coarser triangulation)
    n_tgt_nodes = 20
    nodes_tgt = rng.random((2, n_tgt_nodes))
    tri_tgt = Delaunay(nodes_tgt.T)
    elem_tgt = tri_tgt.simplices.T

    # Scalar field on source nodes
    x, y = nodes_src
    field_src = np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    # Interpolate
    field_tgt = conservative_interp_tri2tri(nodes_src, elem_src, field_src,
                                            nodes_tgt, elem_tgt, k_subdivide=3)

    # Plot
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.tripcolor(nodes_src[0], nodes_src[1], elem_src.T, field_src,
                  shading='flat', cmap='viridis')
    plt.title("Source triangular mesh")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.tripcolor(nodes_tgt[0], nodes_tgt[1], elem_tgt.T, field_tgt,
                  shading='flat', cmap='viridis')
    plt.title("Interpolated to target triangular mesh")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
