import numpy as np
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import time

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
# Weight calculation function
# ----------------------------

def calculate_interpolation_weights(nodes, elem, grid_x=None, grid_y=None, grid_size=100, 
                                   elem_area=None, n_processes=None):
    """
    Calculate interpolation weights from triangular mesh to structured grid.
    
    Parameters
    ----------
    nodes : (2, nod_n) array
        Node coordinates (x,y).
    elem : (3, elem_n) array
        Node indices for each triangular element.
    grid_x, grid_y : 1D arrays, optional
        Grid axes (cell boundaries).
    grid_size : int, optional
        Number of grid cells if grid_x or grid_y not provided.
    elem_area : (elem_n,) array, optional
        Precomputed element areas. If None, computed from nodes.
    n_processes : int, optional
        Number of processes for parallelization. Defaults to CPU count.
        
    Returns
    -------
    weights : list of tuples
        Each tuple contains (element_index, grid_j, grid_i, weight)
    grid_x, grid_y : 1D arrays
        Grid axes
    area_map : 2D array
        Total area in each grid cell
    """
    node_coords = nodes.T
    elem_conn = elem.T
    
    if elem_area is None:
        elem_area = np.array([triangle_area(node_coords[tri]) for tri in elem_conn])

    if grid_x is None:
        grid_x = np.linspace(node_coords[:,0].min(), node_coords[:,0].max(), grid_size)
    if grid_y is None:
        grid_y = np.linspace(node_coords[:,1].min(), node_coords[:,1].max(), grid_size)

    grid_x, dx_min = _prepare_axis(grid_x, "grid_x")
    grid_y, dy_min = _prepare_axis(grid_y, "grid_y")
    target_cell_size = min(dx_min, dy_min)

    nx = len(grid_x) - 1
    ny = len(grid_y) - 1
    area_map = np.zeros((ny, nx), dtype=float)
    
    # Prepare arguments for parallel processing
    element_args = [(e_idx, tri_idx, node_coords, elem_area, grid_x, grid_y, nx, ny, target_cell_size) 
                    for e_idx, tri_idx in enumerate(elem_conn)]
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    n_processes = min(n_processes, len(element_args))
    
    # Process elements in parallel
    weights = []
    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            results = pool.map(_process_element_weights, element_args)
    else:
        # Serial processing
        results = map(_process_element_weights, element_args)
    
    # Combine results
    for result in results:
        for e_idx, j, i, weight in result:
            weights.append((e_idx, j, i, weight))
            area_map[j, i] += weight
    
    return weights, grid_x, grid_y, area_map

def _process_element_weights(args):
    """Process a single element to calculate interpolation weights."""
    e_idx, tri_idx, node_coords, elem_area, grid_x, grid_y, nx, ny, target_cell_size = args
    tri_vertices = node_coords[tri_idx]
    A = elem_area[e_idx]
    
    k = compute_required_subdivisions(tri_vertices, target_cell_size)
    centroids = subdivide_triangle_iterative(tri_vertices, k)
    sub_area = A / len(centroids) if centroids else 0
    
    element_weights = []
    for (x, y) in centroids:
        i = np.searchsorted(grid_x, x, side='right') - 1
        j = np.searchsorted(grid_y, y, side='right') - 1
        if 0 <= i < nx and 0 <= j < ny:
            element_weights.append((e_idx, j, i, sub_area))
            
    return element_weights

# ----------------------------
# Interpolation function
# ----------------------------

def apply_interpolation_weights(weights, area_map, elem_conn, field_at_nodes, n_processes=None):
    """
    Apply precomputed interpolation weights to interpolate field values.
    
    Parameters
    ----------
    weights : list of tuples
        Precomputed weights from calculate_interpolation_weights
    area_map : 2D array
        Total area in each grid cell
    elem_conn : array
        Element connectivity
    field_at_nodes : (nod_n,) array
        Scalar values per node.
    n_processes : int, optional
        Number of processes for parallelization. Defaults to CPU count.
        
    Returns
    -------
    heatmap : 2D array
        Interpolated field values on the grid
    """
    ny, nx = area_map.shape
    heatmap = np.zeros((ny, nx), dtype=float)
    
    # Precompute element values
    elem_values = np.array([field_at_nodes[tri].mean() for tri in elem_conn], dtype=float)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    n_processes = min(n_processes, len(weights))
    
    
    grid_shape = area_map.shape
    if n_processes > 1:
        chunk_size = len(weights) // n_processes + 1
        weight_chunks = [weights[i:i+chunk_size] for i in range(0, len(weights), chunk_size)]
        
        with Pool(processes=n_processes) as pool:
            results = pool.map(partial(_process_weights_chunk, 
                                    elem_values=elem_values, 
                                    grid_shape=grid_shape), 
                            weight_chunks)
    else:
        results = [_process_weights_chunk(weights, elem_values, grid_shape)]
    
    # Combine results
    for result in results:
        heatmap += result
    
    # Normalize by area
    mask = area_map > 0
    heatmap[mask] /= area_map[mask]
    
    return heatmap

def _process_weights_chunk(weight_chunk, elem_values, grid_shape):
    """Process a chunk of weights to calculate heatmap contributions."""
    chunk_heatmap = np.zeros(grid_shape)
    for e_idx, j, i, weight in weight_chunk:
        if j < grid_shape[0] and i < grid_shape[1]:
            chunk_heatmap[j, i] += elem_values[e_idx] * weight
    return chunk_heatmap

# ----------------------------
# Combined function for backward compatibility
# ----------------------------

def tri2reg(nodes, elem, field_at_nodes, grid_x=None, grid_y=None, 
            grid_size=100, elem_area=None, n_processes=None):
    """
    Conservative interpolation from triangular mesh to structured grid.
    
    Parameters
    ----------
    nodes : (2, nod_n) array
        Node coordinates (x,y).
    elem : (3, elem_n) array
        Node indices for each triangular element.
    field_at_nodes : (nod_n,) array
        Scalar values per node.
    grid_x, grid_y : 1D arrays, optional
        Grid axes (cell boundaries).
    grid_size : int, optional
        Number of grid cells if grid_x or grid_y not provided.
    elem_area : (elem_n,) array, optional
        Precomputed element areas. If None, computed from nodes.
    n_processes : int, optional
        Number of processes for parallelization. Defaults to CPU count.
        
    Returns
    -------
    grid_x, grid_y, heatmap
    """
    # Calculate weights
    weights, grid_x, grid_y, area_map = calculate_interpolation_weights(
        nodes, elem, grid_x, grid_y, grid_size, elem_area, n_processes)
    
    # Apply weights
    heatmap = apply_interpolation_weights(
        weights, area_map, elem.T, field_at_nodes, n_processes)
    
    return grid_x, grid_y, heatmap

# ----------------------------
# Synthetic data generation
# ----------------------------

def generate_synthetic_mesh(n_nodes=50, n_elem=80, seed=0):
    """
    Generate synthetic FEM-style mesh arrays.
    """
    rng = np.random.default_rng(seed)

    # --- Nodes (2, nod_n)
    nodes = rng.random((2, n_nodes))

    # --- Elements (3, elem_n): choose 3 distinct node indices per element
    elem = np.zeros((3, n_elem), dtype=int)
    for i in range(n_elem):
        elem[:, i] = rng.choice(n_nodes, 3, replace=False)

    # --- Element areas (elem_n)
    elem_area = np.zeros(n_elem)
    node_coords = nodes.T
    for i in range(n_elem):
        tri = node_coords[elem[:, i]]
        elem_area[i] = triangle_area(tri)

    # --- Scalar field at nodes (nod_n,)
    x, y = nodes
    zbar_n_bottom = np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    return nodes, elem, elem_area, zbar_n_bottom

# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    # Generate synthetic data
    nodes, elem, elem_area, zbar_n_bottom = generate_synthetic_mesh()
    print("nodes shape:", nodes.shape)
    print("elem shape:", elem.shape)
    print("elem_area shape:", elem_area.shape)
    print("zbar_n_bottom shape:", zbar_n_bottom.shape)

    # Time the weight calculation
    start_time = time.time()
    weights, gx, gy, area_map = calculate_interpolation_weights(
        nodes, elem, 
        grid_x=np.linspace(0, 1, 100),
        grid_y=np.linspace(0, 1, 100),
        elem_area=elem_area,
        n_processes=4
    )
    weight_time = time.time() - start_time
    print(f"Weight calculation time: {weight_time:.3f} seconds")

    # Time the interpolation
    start_time = time.time()
    H = apply_interpolation_weights(
        weights, area_map, elem.T, zbar_n_bottom, n_processes=4
    )
    interp_time = time.time() - start_time
    print(f"Interpolation time: {interp_time:.3f} seconds")

    # Compare with direct method
    start_time = time.time()
    gx_direct, gy_direct, H_direct = tri2reg(
        nodes, elem, zbar_n_bottom,
        grid_x=np.linspace(0, 1, 100),
        grid_y=np.linspace(0, 1, 100),
        elem_area=elem_area,
        n_processes=4
    )
    direct_time = time.time() - start_time
    print(f"Direct method time: {direct_time:.3f} seconds")

    # Verify results are the same
    print(f"Results match: {np.allclose(H, H_direct)}")

    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.pcolormesh(gx, gy, H, shading="auto", cmap="viridis")
    plt.colorbar(label="Interpolated Value")
    plt.title("With weights")
    
    plt.subplot(132)
    plt.pcolormesh(gx_direct, gy_direct, H_direct, shading="auto", cmap="viridis")
    plt.colorbar(label="Interpolated Value")
    plt.title("Direct method")
    
    plt.subplot(133)
    plt.pcolormesh(gx, gy, np.abs(H - H_direct), shading="auto", cmap="hot")
    plt.colorbar(label="Absolute Difference")
    plt.title("Difference")
    
    plt.tight_layout()
    plt.show()