import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay
from multiprocessing import Pool, cpu_count
from functools import partial

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

def interpolate_from_source(pt, source_triangles, source_values):
    """Find which source triangle contains pt and return its value."""
    for i, tri in enumerate(source_triangles):
        if point_in_triangle(pt, tri):
            return source_values[i]
    return np.nan  # outside domain

# --------------------------------
# Weight calculation for triangle-to-triangle interpolation
# --------------------------------

def calculate_tri2tri_weights(nodes_src, elem_src, nodes_tgt, elem_tgt, 
                             k_subdivide=3, n_processes=None):
    """
    Calculate interpolation weights from source triangular mesh to target triangular mesh.
    
    Parameters
    ----------
    nodes_src : (2, Nsrc) array
    elem_src  : (3, Msrc) array
    nodes_tgt : (2, Ntgt) array
    elem_tgt  : (3, Mtgt) array
    k_subdivide : int
        Number of subdivisions for each target triangle
    n_processes : int, optional
        Number of processes for parallelization
        
    Returns
    -------
    weights : list of tuples
        Each tuple contains (target_element_index, source_element_index, weight)
    target_areas : array
        Area of each target element
    """
    # Precompute source triangles and their values
    source_triangles = [nodes_src[:, tri].T for tri in elem_src.T]
    
    # Precompute target triangles and their areas
    target_triangles = [nodes_tgt[:, tri].T for tri in elem_tgt.T]
    target_areas = np.array([triangle_area(tri) for tri in target_triangles])
    
    # Prepare arguments for parallel processing
    target_args = [(e_idx, tri_vertices, source_triangles, k_subdivide, target_areas[e_idx]) 
                  for e_idx, tri_vertices in enumerate(target_triangles)]
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    n_processes = min(n_processes, len(target_args))
    
    # Process target elements in parallel
    weights = []
    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            results = pool.map(_process_target_element_weights, target_args)
    else:
        # Serial processing
        results = map(_process_target_element_weights, target_args)
    
    # Combine results
    for result in results:
        weights.extend(result)
    
    return weights, target_areas

def _process_target_element_weights(args):
    """Process a single target element to calculate interpolation weights."""
    e_idx, tri_vertices, source_triangles, k_subdivide, target_area = args
    element_weights = []
    
    # Subdivide target triangle into smaller centroids
    centroids = subdivide_triangle_iterative(tri_vertices, k_subdivide)
    sub_area = target_area / len(centroids)
    
    for c in centroids:
        # Find which source triangle contains this centroid
        for src_idx, src_tri in enumerate(source_triangles):
            if point_in_triangle(c, src_tri):
                element_weights.append((e_idx, src_idx, sub_area))
                break
    
    return element_weights

# --------------------------------
# Interpolation function for triangle-to-triangle
# --------------------------------

def apply_tri2tri_weights(weights, source_values, target_areas, n_processes=None):
    """
    Apply precomputed interpolation weights to interpolate field values.
    
    Parameters
    ----------
    weights : list of tuples
        Precomputed weights from calculate_tri2tri_weights
    source_values : array
        Values for each source element
    target_areas : array
        Area of each target element
    n_processes : int, optional
        Number of processes for parallelization
        
    Returns
    -------
    field_tgt : array
        Interpolated values for target elements
    """
    n_target = len(target_areas)
    field_tgt = np.zeros(n_target)
    area_covered = np.zeros(n_target)
    
    # Process weights in parallel
    if n_processes is None:
        n_processes = cpu_count()
    n_processes = min(n_processes, len(weights))
    
    if n_processes > 1:
        # Split weights into chunks for parallel processing
        chunk_size = len(weights) // n_processes + 1
        weight_chunks = [weights[i:i+chunk_size] for i in range(0, len(weights), chunk_size)]
        
        with Pool(processes=n_processes) as pool:
            results = pool.map(partial(_process_tri2tri_weights_chunk, 
                                     source_values=source_values, 
                                     n_target=n_target), 
                             weight_chunks)
    else:
        # Serial processing
        results = [_process_tri2tri_weights_chunk(weights, source_values, n_target)]
    
    # Combine results
    for result in results:
        field_tgt_chunk, area_chunk = result
        field_tgt += field_tgt_chunk
        area_covered += area_chunk
    
    # Normalize by area covered
    mask = area_covered > 0
    field_tgt[mask] /= area_covered[mask]
    field_tgt[~mask] = np.nan
    
    return field_tgt

def _process_tri2tri_weights_chunk(weight_chunk, source_values, n_target):
    """Process a chunk of weights to calculate target field contributions."""
    field_tgt_chunk = np.zeros(n_target)
    area_chunk = np.zeros(n_target)
    
    for tgt_idx, src_idx, weight in weight_chunk:
        field_tgt_chunk[tgt_idx] += source_values[src_idx] * weight
        area_chunk[tgt_idx] += weight
    
    return field_tgt_chunk, area_chunk

# --------------------------------
# Combined function for triangle-to-triangle interpolation
# --------------------------------

def conservative_interp_tri2tri(nodes_src, elem_src, field_src,
                                nodes_tgt, elem_tgt,
                                k_subdivide=3, n_processes=None):
    """
    Conservative interpolation from source triangular mesh to target triangular mesh.
    
    Parameters
    ----------
    nodes_src : (2, Nsrc) array
    elem_src  : (3, Msrc) array
    field_src : (Nsrc,) array
    nodes_tgt : (2, Ntgt) array
    elem_tgt  : (3, Mtgt) array
    k_subdivide : int
        Number of subdivisions for each target triangle
    n_processes : int, optional
        Number of processes for parallelization
        
    Returns
    -------
    field_tgt : (Mtgt,) array of interpolated values per target triangle
    """
    # Precompute source element values
    source_values = np.array([field_src[tri].mean() for tri in elem_src.T])
    
    # Calculate weights
    weights, target_areas = calculate_tri2tri_weights(
        nodes_src, elem_src, nodes_tgt, elem_tgt, k_subdivide, n_processes)
    
    # Apply weights
    field_tgt = apply_tri2tri_weights(
        weights, source_values, target_areas, n_processes)
    
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

    # Interpolate with parallel processing
    field_tgt = conservative_interp_tri2tri(nodes_src, elem_src, field_src,
                                            nodes_tgt, elem_tgt, 
                                            k_subdivide=3, n_processes=4)

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