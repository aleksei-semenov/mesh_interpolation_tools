import numpy as np
import matplotlib.pyplot as plt
import math
import healpy as hp
from scipy.spatial import SphericalVoronoi, geometric_slerp
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# --------------------------------
# Spherical geometry helpers
# --------------------------------
def spherical_triangle_area(vertices):
    """Calculate area of a spherical triangle using Girard's theorem."""
    # Normalize vertices to ensure they're on the unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    
    # Calculate the angles at each vertex
    angles = []
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        
        # Vectors from vertex i to j and i to k
        v_ij = vertices[j] - vertices[i]
        v_ik = vertices[k] - vertices[i]
        
        # Angle at vertex i
        angle = np.arccos(np.dot(v_ij, v_ik) / (np.linalg.norm(v_ij) * np.linalg.norm(v_ik)))
        angles.append(angle)
    
    # Use Girard's theorem to calculate area
    area = sum(angles) - np.pi
    return abs(area)  # Ensure positive area

def subdivide_spherical_triangle(triangle, k):
    """Subdivide a spherical triangle into 4^k smaller ones, return centroids."""
    # Ensure the triangle vertices are on the unit sphere
    triangle = triangle / np.linalg.norm(triangle, axis=1)[:, np.newaxis]
    
    triangles = [triangle.copy()]
    for _ in range(k):
        new_tris = []
        for tri in triangles:
            A, B, C = tri
            
            # Find midpoints on the sphere using slerp
            AB = geometric_slerp(A, B, 0.5)
            BC = geometric_slerp(B, C, 0.5)
            CA = geometric_slerp(C, A, 0.5)
            
            # Ensure points are on unit sphere
            AB = AB / np.linalg.norm(AB)
            BC = BC / np.linalg.norm(BC)
            CA = CA / np.linalg.norm(CA)
            
            # Create four new triangles
            new_tris.extend([
                np.array([A, AB, CA]),
                np.array([AB, B, BC]),
                np.array([CA, BC, C]),
                np.array([AB, BC, CA]),
            ])
        triangles = new_tris
    
    # Return centroids (normalized to ensure they're on the unit sphere)
    centroids = [np.mean(t, axis=0) for t in triangles]
    centroids = [c / np.linalg.norm(c) for c in centroids]
    return centroids

def point_in_spherical_triangle(pt, tri):
    """Check if a point lies in a spherical triangle using barycentric coordinates."""
    # Normalize all points
    pt = pt / np.linalg.norm(pt)
    tri = tri / np.linalg.norm(tri, axis=1)[:, np.newaxis]
    
    # Use the same approach as planar triangles but with dot products
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

# --------------------------------
# Spherical mesh generation
# --------------------------------
def generate_spherical_triangular_mesh(n_points):
    """Generate a spherical triangular mesh using Fibonacci sphere points."""
    # Generate points on sphere using Fibonacci method
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    points = np.vstack((x, y, z)).T
    
    # Create spherical Voronoi diagram
    sv = SphericalVoronoi(points)
    sv.sort_vertices_of_regions()
    
    # Convert to triangular mesh
    nodes = points.T  # (3, n_points)
    
    # Create elements from the Voronoi regions
    elements = []
    for region in sv.regions:
        if len(region) >= 3:  # Need at least 3 points to form a triangle
            # Triangulate the polygon
            for i in range(1, len(region) - 1):
                # Make sure indices are within bounds
                if all(idx < n_points for idx in [region[0], region[i], region[i+1]]):
                    elements.append([region[0], region[i], region[i+1]])
    
    elem_src = np.array(elements).T
    
    return nodes, elem_src

# --------------------------------
# Weight calculation and interpolation functions
# --------------------------------

def _process_element_weights(args):
    """Process a single element to calculate interpolation weights."""
    e_idx, tri_idx, nodes_src, elem_area, nside, k_subdivide = args
    tri_vertices = nodes_src[:, tri_idx].T
    A = elem_area[e_idx]
    
    # Subdivide the triangle
    centroids = subdivide_spherical_triangle(tri_vertices, k_subdivide)
    sub_area = A / len(centroids) if centroids else 0
    
    element_weights = []
    for c in centroids:
        # Convert centroid to spherical coordinates
        x, y, z = c
        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2 * np.pi
            
        # Find the HEALPix pixel index
        pix_idx = hp.ang2pix(nside, theta, phi)
        element_weights.append((pix_idx, sub_area))
            
    return e_idx, element_weights

def calculate_spherical_weights(nodes_src, elem_src, nside, k_subdivide=3, n_processes=None):
    """
    Calculate interpolation weights from spherical triangular mesh to HEALPix grid.
    
    Parameters
    ----------
    nodes_src : (3, Nsrc) array
        Cartesian coordinates of source mesh nodes (on unit sphere)
    elem_src  : (3, Msrc) array
        Indices of triangles in source mesh
    nside : int
        HEALPix NSIDE parameter
    k_subdivide : int
        Number of subdivisions for each source triangle
    n_processes : int, optional
        Number of processes for parallelization. Defaults to CPU count.
        
    Returns
    -------
    weights : list of tuples
        Each tuple contains (element_index, list of (pixel_index, weight))
    area_map : array
        Total area in each HEALPix pixel
    """
    npix = hp.nside2npix(nside)
    
    # Calculate element areas
    elem_area = np.zeros(elem_src.shape[1])
    for e_idx, tri_idx in enumerate(elem_src.T):
        tri_vertices = nodes_src[:, tri_idx].T
        elem_area[e_idx] = spherical_triangle_area(tri_vertices)
    
    # Prepare arguments for parallel processing
    element_args = [(e_idx, tri_idx, nodes_src, elem_area, nside, k_subdivide) 
                    for e_idx, tri_idx in enumerate(elem_src.T)]
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    n_processes = min(n_processes, len(element_args))
    
    # Initialize area map
    area_map = np.zeros(npix)
    
    # Process elements in parallel
    weights = []
    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            results = pool.map(_process_element_weights, element_args)
    else:
        # Serial processing
        results = map(_process_element_weights, element_args)
    
    # Process results
    for e_idx, element_weights in results:
        weights.append((e_idx, element_weights))
        for pix_idx, weight in element_weights:
            area_map[pix_idx] += weight
    
    return weights, area_map

def _process_weights_chunk(weight_chunk, elem_values):
    """Process a chunk of weights to calculate HEALPix map contributions."""
    npix = len(elem_values) if hasattr(elem_values, 'shape') else hp.nside2npix(256)  # Default fallback
    chunk_healpix = np.zeros(npix)
    
    for e_idx, element_weights in weight_chunk:
        for pix_idx, weight in element_weights:
            if pix_idx < npix:
                chunk_healpix[pix_idx] += elem_values[e_idx] * weight
            
    return chunk_healpix

def apply_spherical_weights(weights, area_map, elem_src, field_src, n_processes=None):
    """
    Apply precomputed interpolation weights to interpolate field values.
    
    Parameters
    ----------
    weights : list of tuples
        Precomputed weights from calculate_spherical_weights
    area_map : array
        Total area in each HEALPix pixel
    elem_src : array
        Element connectivity
    field_src : array
        Field values at source nodes
    n_processes : int, optional
        Number of processes for parallelization. Defaults to CPU count.
        
    Returns
    -------
    healpix_map : array
        Interpolated field values on the HEALPix grid
    """
    npix = len(area_map)
    
    # Precompute element values
    elem_values = np.array([field_src[tri].mean() for tri in elem_src.T], dtype=float)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = cpu_count()
    n_processes = min(n_processes, len(weights))
    
    # Process weights in parallel
    healpix_map = np.zeros(npix)
    if n_processes > 1:
        # Split weights into chunks for parallel processing
        chunk_size = len(weights) // n_processes + 1
        weight_chunks = [weights[i:i+chunk_size] for i in range(0, len(weights), chunk_size)]
        
        with Pool(processes=n_processes) as pool:
            results = pool.map(partial(_process_weights_chunk, elem_values=elem_values), weight_chunks)
    else:
        # Serial processing
        results = [_process_weights_chunk(weights, elem_values)]
    
    # Combine results
    for result in results:
        healpix_map += result
    
    # Normalize by area
    mask = area_map > 0
    healpix_map[mask] /= area_map[mask]
    healpix_map[~mask] = np.nan  # Mark pixels with no data
    
    return healpix_map

# --------------------------------
# Combined function for backward compatibility
# --------------------------------

def conservative_interp_spherical_tri2healpix(nodes_src, elem_src, field_src, nside, k_subdivide=3, n_processes=None):
    """
    Conservative interpolation from source spherical triangular mesh to HEALPix grid.
    
    Parameters
    ----------
    nodes_src : (3, Nsrc) array
        Cartesian coordinates of source mesh nodes (on unit sphere)
    elem_src  : (3, Msrc) array
        Indices of triangles in source mesh
    field_src : (Nsrc,) array
        Field values at source nodes
    nside : int
        HEALPix NSIDE parameter
    k_subdivide : int
        Number of subdivisions for each source triangle
    n_processes : int, optional
        Number of processes for parallelization. Defaults to CPU count.
        
    Returns
    -------
    healpix_map : (Npix,) array of interpolated values per HEALPix pixel
    """
    # Calculate weights
    weights, area_map = calculate_spherical_weights(
        nodes_src, elem_src, nside, k_subdivide, n_processes)
    
    # Apply weights
    healpix_map = apply_spherical_weights(
        weights, area_map, elem_src, field_src, n_processes)
    
    return healpix_map

# --------------------------------
# Example usage
# --------------------------------
if __name__ == "__main__":
    # Generate a spherical triangular mesh
    n_src_nodes = 500  # Reduced number for faster computation
    nodes_src, elem_src = generate_spherical_triangular_mesh(n_src_nodes)
    
    # Create a sample field on the source mesh
    x, y, z = nodes_src
    # Simple field: warmer near equator, cooler at poles
    field_src = 1.0 - np.abs(z)  # z ranges from -1 to 1
    
    # Set up HEALPix grid
    nside = 16
    npix = hp.nside2npix(nside)
    
    # Time the weight calculation
    start_time = time.time()
    weights, area_map = calculate_spherical_weights(
        nodes_src, elem_src, nside, k_subdivide=2, n_processes=4
    )
    weight_time = time.time() - start_time
    print(f"Weight calculation time: {weight_time:.3f} seconds")
    
    # Time the interpolation
    start_time = time.time()
    healpix_map = apply_spherical_weights(
        weights, area_map, elem_src, field_src, n_processes=4
    )
    interp_time = time.time() - start_time
    print(f"Interpolation time: {interp_time:.3f} seconds")
    
    # Compare with direct method
    start_time = time.time()
    healpix_map_direct = conservative_interp_spherical_tri2healpix(
        nodes_src, elem_src, field_src, nside, k_subdivide=2, n_processes=4
    )
    direct_time = time.time() - start_time
    print(f"Direct method time: {direct_time:.3f} seconds")
    
    # Verify results are the same
    print(f"Results match: {np.allclose(healpix_map, healpix_map_direct, equal_nan=True)}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot HEALPix map
    hp.mollview(healpix_map, title="Interpolated HEALPix map (with weights)", sub=(1, 3, 1))
    
    # For the source mesh, we need to convert to spherical coordinates and plot
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    
    # Create a scatter plot of the source mesh
    ax = plt.subplot(1, 3, 2, projection='mollweide')
    sc = ax.scatter(phi, np.pi/2 - theta, c=field_src, s=5, cmap='viridis')
    plt.colorbar(sc, ax=ax)
    ax.set_title("Source spherical mesh")
    
    # Plot the difference (for validation)
    pix_idx_src = hp.ang2pix(nside, theta, phi)
    field_interp = healpix_map[pix_idx_src]
    
    ax = plt.subplot(1, 3, 3, projection='mollweide')
    diff = field_src - field_interp
    sc = ax.scatter(phi, np.pi/2 - theta, c=diff, s=5, cmap='RdBu_r')
    plt.colorbar(sc, ax=ax)
    ax.set_title("Difference (source - interpolated)")
    
    plt.tight_layout()
    plt.show()