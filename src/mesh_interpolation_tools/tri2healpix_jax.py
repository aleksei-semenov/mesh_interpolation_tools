import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
import numpy as np
import matplotlib.pyplot as plt
import math
import healpy as hp
from scipy.spatial import SphericalVoronoi

# --------------------------------
# JAX-compatible Spherical geometry helpers
# --------------------------------

@jit
def spherical_triangle_area(vertices):
    """Calculate area of a spherical triangle using Girard's theorem (JAX version)."""
    # Normalize vertices to ensure they're on the unit sphere
    vertices = vertices / jnp.linalg.norm(vertices, axis=1)[:, jnp.newaxis]
    
    # Calculate the angles at each vertex
    def compute_angle(i):
        j = (i + 1) % 3
        k = (i + 2) % 3
        
        # Vectors from vertex i to j and i to k
        v_ij = vertices[j] - vertices[i]
        v_ik = vertices[k] - vertices[i]
        
        # Angle at vertex i
        angle = jnp.arccos(jnp.dot(v_ij, v_ik) / 
                          (jnp.linalg.norm(v_ij) * jnp.linalg.norm(v_ik) + 1e-10))
        return angle
    
    angles = jnp.array([compute_angle(i) for i in range(3)])
    
    # Use Girard's theorem to calculate area
    area = jnp.sum(angles) - jnp.pi
    return jnp.abs(area)

# Vectorized version for multiple triangles
batch_spherical_triangle_area = vmap(spherical_triangle_area)

@jit
def geometric_slerp_jax(p0, p1, t):
    """JAX implementation of spherical linear interpolation."""
    # Normalize inputs
    p0 = p0 / jnp.linalg.norm(p0)
    p1 = p1 / jnp.linalg.norm(p1)
    
    dot = jnp.dot(p0, p1)
    dot = jnp.clip(dot, -1.0, 1.0)
    
    omega = jnp.arccos(dot)
    sin_omega = jnp.sin(omega)
    
    # Handle small angles
    return jnp.where(
        jnp.abs(sin_omega) < 1e-10,
        (1 - t) * p0 + t * p1,
        (jnp.sin((1 - t) * omega) * p0 + jnp.sin(t * omega) * p1) / sin_omega
    )

@jit
def subdivide_spherical_triangle_jax(triangle, k):
    """Subdivide a spherical triangle into 4^k smaller ones (JAX version)."""
    # Ensure the triangle vertices are on the unit sphere
    triangle = triangle / jnp.linalg.norm(triangle, axis=1)[:, jnp.newaxis]
    
    A, B, C = triangle
    
    def subdivide_once(vertices):
        """Subdivide a single triangle once."""
        A, B, C = vertices
        
        # Find midpoints on the sphere using slerp
        AB = geometric_slerp_jax(A, B, 0.5)
        BC = geometric_slerp_jax(B, C, 0.5)
        CA = geometric_slerp_jax(C, A, 0.5)
        
        # Ensure points are on unit sphere
        AB = AB / jnp.linalg.norm(AB)
        BC = BC / jnp.linalg.norm(BC)
        CA = CA / jnp.linalg.norm(CA)
        
        # Create four new triangles
        return jnp.array([
            [A, AB, CA],
            [AB, B, BC],
            [CA, BC, C],
            [AB, BC, CA]
        ])
    
    # Start with initial triangle
    triangles = triangle[jnp.newaxis, :, :]
    
    # Apply subdivision k times
    for _ in range(k):
        # Reshape to flatten and then subdivide
        triangles = triangles.reshape(-1, 3, 3)
        triangles = vmap(subdivide_once)(triangles).reshape(-1, 3, 3)
    
    # Compute centroids
    centroids = jnp.mean(triangles, axis=1)
    centroids = centroids / jnp.linalg.norm(centroids, axis=1, keepdims=True)
    
    return centroids

# Vectorized version for multiple triangles
batch_subdivide_triangle = vmap(subdivide_spherical_triangle_jax, in_axes=(0, None))

@jit
def compute_healpix_weights_and_indices(centroid, nside):
    """Compute HEALPix pixel index for a centroid."""
    x, y, z = centroid
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)
    
    # Convert to pixel index (using numpy for healpy compatibility)
    return hp.ang2pix(nside, theta, phi)

# Vectorized version
batch_compute_healpix_indices = vmap(compute_healpix_weights_and_indices, in_axes=(0, None))

# --------------------------------
# JAX-compatible conservative interpolation with parallelization
# --------------------------------

@jit
def conservative_interp_spherical_tri2healpix_jax(nodes_src, elem_src, field_src, nside, k_subdivide=3):
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
    
    Returns
    -------
    healpix_map : (Npix,) array of interpolated values per HEALPix pixel
    healpix_weights : (Npix,) array of weights per pixel
    """
    # Calculate HEALPix pixel centers
    npix = hp.nside2npix(nside)
    
    # Get triangle vertices
    elem_indices = elem_src.T  # (Msrc, 3)
    triangle_vertices = nodes_src[:, elem_indices].transpose(1, 2, 0)  # (Msrc, 3, 3)
    
    # Calculate areas of all triangles in parallel
    triangle_areas = batch_spherical_triangle_area(triangle_vertices)  # (Msrc,)
    
    # Calculate mean values for each triangle
    tri_values = jnp.mean(field_src[elem_indices], axis=1)  # (Msrc,)
    
    # Subdivide all triangles in parallel
    all_centroids = batch_subdivide_triangle(triangle_vertices, k_subdivide)  # (Msrc, 4^k, 3)
    
    # Calculate area per centroid
    Msrc = triangle_vertices.shape[0]
    num_subdivisions = 4 ** k_subdivide
    sub_area_per_triangle = triangle_areas[:, jnp.newaxis] / num_subdivisions  # (Msrc, 1)
    
    # Flatten centroids and their corresponding values and areas
    flat_centroids = all_centroids.reshape(-1, 3)  # (Msrc * 4^k, 3)
    flat_values = jnp.repeat(tri_values, num_subdivisions)  # (Msrc * 4^k,)
    flat_areas = jnp.repeat(sub_area_per_triangle.flatten(), num_subdivisions)  # (Msrc * 4^k,)
    
    # Compute HEALPix indices for all centroids in parallel
    # Note: healpy functions are not JAX-compatible, so we convert to numpy first
    flat_centroids_np = np.array(flat_centroids)
    pix_indices = batch_compute_healpix_indices(flat_centroids_np, nside)
    
    # Convert back to JAX arrays
    pix_indices = jnp.array(pix_indices)
    flat_values = jnp.array(flat_values)
    flat_areas = jnp.array(flat_areas)
    
    # Initialize output arrays
    healpix_map = jnp.zeros(npix)
    healpix_weights = jnp.zeros(npix)
    
    # Use scatter-add to accumulate values and weights
    def accumulate_pixel(i, accum):
        map_array, weight_array = accum
        idx = pix_indices[i]
        val = flat_values[i]
        area = flat_areas[i]
        
        # Scatter add
        map_array = map_array.at[idx].add(val * area)
        weight_array = weight_array.at[idx].add(area)
        return (map_array, weight_array)
    
    # Parallel accumulation
    healpix_map, healpix_weights = lax.fori_loop(
        0, len(flat_centroids), 
        accumulate_pixel, 
        (healpix_map, healpix_weights)
    )
    
    # Normalize by weights where weights > 0
    nonzero_mask = healpix_weights > 0
    normalized_map = jnp.where(
        nonzero_mask,
        healpix_map / (healpix_weights + 1e-10),
        jnp.nan
    )
    
    return normalized_map, healpix_weights

# --------------------------------
# Routine to obtain weights for conservative interpolation
# --------------------------------

def get_conservative_interpolation_weights(nodes_src, elem_src, nside, k_subdivide=3):
    """
    Compute weights for conservative interpolation from spherical triangles to HEALPix.
    
    Returns a sparse weight matrix W where:
        healpix_map = W @ field_src
    
    Parameters
    ----------
    nodes_src : (3, Nsrc) array
        Cartesian coordinates of source mesh nodes
    elem_src : (3, Msrc) array
        Indices of triangles in source mesh
    nside : int
        HEALPix NSIDE parameter
    k_subdivide : int
        Number of subdivisions for each triangle
    
    Returns
    -------
    weight_matrix : sparse matrix in COO format (indices, values, shape)
        where shape = (Npix, Nsrc)
    """
    # Get triangle vertices
    elem_indices = elem_src.T  # (Msrc, 3)
    triangle_vertices = nodes_src[:, elem_indices].transpose(1, 2, 0)  # (Msrc, 3, 3)
    
    # Calculate areas of all triangles
    triangle_areas = batch_spherical_triangle_area(triangle_vertices)  # (Msrc,)
    
    # Subdivide all triangles
    all_centroids = batch_subdivide_triangle(triangle_vertices, k_subdivide)  # (Msrc, 4^k, 3)
    
    # Calculate area per centroid
    Msrc = triangle_vertices.shape[0]
    num_subdivisions = 4 ** k_subdivide
    sub_area = triangle_areas[:, jnp.newaxis] / num_subdivisions  # (Msrc, 1)
    
    # Flatten centroids
    flat_centroids = all_centroids.reshape(-1, 3)  # (Msrc * 4^k, 3)
    
    # Get corresponding triangle indices for each centroid
    tri_indices = jnp.repeat(jnp.arange(Msrc), num_subdivisions)  # (Msrc * 4^k,)
    
    # Get vertex indices for each triangle
    vertex_indices = elem_indices[tri_indices]  # (Msrc * 4^k, 3)
    
    # Compute HEALPix indices for all centroids
    flat_centroids_np = np.array(flat_centroids)
    pix_indices = batch_compute_healpix_indices(flat_centroids_np, nside)
    pix_indices = jnp.array(pix_indices)
    
    # Flat areas for each centroid
    flat_areas = jnp.repeat(sub_area.flatten(), num_subdivisions)  # (Msrc * 4^k,)
    
    # Weight per vertex in each triangle (assuming linear interpolation within triangle)
    # Each vertex gets equal weight (1/3) for the triangle's value
    vertex_weights = jnp.ones_like(vertex_indices, dtype=jnp.float32) / 3.0  # (Msrc * 4^k, 3)
    
    # Distribute area weight to vertices
    area_weights = flat_areas[:, jnp.newaxis] * vertex_weights  # (Msrc * 4^k, 3)
    
    # Prepare COO matrix data
    npix = hp.nside2npix(nside)
    Nsrc = nodes_src.shape[1]
    
    # Repeat pixel indices for each vertex (3 per centroid)
    row_indices = jnp.repeat(pix_indices, 3)  # (3 * Msrc * 4^k,)
    col_indices = vertex_indices.flatten()  # (3 * Msrc * 4^k,)
    values = area_weights.flatten()  # (3 * Msrc * 4^k,)
    
    # Sum duplicate entries (same pixel, same source vertex)
    # For efficiency, we can use a dictionary approach
    from collections import defaultdict
    
    weight_dict = defaultdict(float)
    for r, c, v in zip(row_indices, col_indices, values):
        weight_dict[(int(r), int(c))] += float(v)
    
    # Convert to COO format
    rows, cols, vals = [], [], []
    for (r, c), v in weight_dict.items():
        rows.append(r)
        cols.append(c)
        vals.append(v)
    
    return {
        'rows': jnp.array(rows),
        'cols': jnp.array(cols),
        'values': jnp.array(vals),
        'shape': (npix, Nsrc)
    }

# --------------------------------
# Example usage with parallelization
# --------------------------------

def generate_spherical_triangular_mesh(n_points):
    """Generate a spherical triangular mesh using Fibonacci sphere points."""
    # Same as original, but we'll keep it for mesh generation
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
        if len(region) >= 3:
            for i in range(1, len(region) - 1):
                if all(idx < n_points for idx in [region[0], region[i], region[i+1]]):
                    elements.append([region[0], region[i], region[i+1]])
    
    elem_src = np.array(elements).T
    
    return nodes, elem_src

if __name__ == "__main__":
    # Generate a spherical triangular mesh
    n_src_nodes = 500
    nodes_src, elem_src = generate_spherical_triangular_mesh(n_src_nodes)
    
    # Convert to JAX arrays
    nodes_src_jax = jnp.array(nodes_src)
    elem_src_jax = jnp.array(elem_src)
    
    # Create a sample field on the source mesh
    x, y, z = nodes_src
    field_src = 1.0 - np.abs(z)
    field_src_jax = jnp.array(field_src)
    
    # Set up HEALPix grid
    nside = 16
    
    # Get interpolation weights
    print("Computing interpolation weights...")
    weights = get_conservative_interpolation_weights(
        nodes_src_jax, elem_src_jax, nside, k_subdivide=2
    )
    
    print(f"Weight matrix shape: {weights['shape']}")
    print(f"Number of non-zero entries: {len(weights['values'])}")
    
    # Perform interpolation using JAX
    print("Performing interpolation with JAX...")
    healpix_map, healpix_weights = conservative_interp_spherical_tri2healpix_jax(
        nodes_src_jax, elem_src_jax, field_src_jax, nside, k_subdivide=2
    )
    
    # Convert to numpy for plotting
    healpix_map_np = np.array(healpix_map)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot HEALPix map
    hp.mollview(healpix_map_np, title="JAX Interpolated HEALPix map", sub=(1, 3, 1))
    
    # Plot source mesh
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    
    ax = plt.subplot(1, 3, 2, projection='mollweide')
    sc = ax.scatter(phi, np.pi/2 - theta, c=field_src, s=5, cmap='viridis')
    plt.colorbar(sc, ax=ax)
    ax.set_title("Source spherical mesh")
    
    # Plot weights distribution
    ax = plt.subplot(1, 3, 3)
    ax.hist(healpix_weights[healpix_weights > 0], bins=50, alpha=0.7)
    ax.set_xlabel("Pixel weight")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of interpolation weights")
    
    plt.tight_layout()
    plt.show()