import numpy as np
import matplotlib.pyplot as plt
import math
import healpy as hp
from scipy.spatial import SphericalVoronoi, geometric_slerp

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

def interpolate_from_spherical_source(pt, nodes_src, elem_src, elem_values):
    """Find which source spherical triangle contains pt and return its value."""
    for e_idx, tri_idx in enumerate(elem_src.T):
        tri = nodes_src[:, tri_idx].T
        if point_in_spherical_triangle(pt, tri):
            return elem_values[e_idx]
    return np.nan  # outside domain

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
# Interpolator: spherical triangle to HEALPix
# --------------------------------
def conservative_interp_spherical_tri2healpix(nodes_src, elem_src, field_src, nside, k_subdivide=3):
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
    """
    # Calculate HEALPix pixel centers
    npix = hp.nside2npix(nside)
    
    # Assign mean value to each source element
    elem_values_src = np.array([field_src[tri].mean() for tri in elem_src.T])
    
    # Initialize output map
    healpix_map = np.zeros(npix)
    healpix_weights = np.zeros(npix)  # For area weighting
    
    # For each source triangle
    for e_idx, tri_idx in enumerate(elem_src.T):
        tri_vertices = nodes_src[:, tri_idx].T
        
        # Calculate area of source triangle
        A = spherical_triangle_area(tri_vertices)
        
        # Subdivide source triangle into smaller centroids
        centroids = subdivide_spherical_triangle(tri_vertices, k_subdivide)
        sub_area = A / len(centroids)
        
        # For each centroid, find which HEALPix pixel it falls into
        for c in centroids:
            # Convert centroid to spherical coordinates
            x, y, z = c
            theta = np.arccos(z)
            phi = np.arctan2(y, x)
            if phi < 0:
                phi += 2 * np.pi
                
            # Find the HEALPix pixel index
            pix_idx = hp.ang2pix(nside, theta, phi)
            
            # Add weighted value to this pixel
            healpix_map[pix_idx] += elem_values_src[e_idx] * sub_area
            healpix_weights[pix_idx] += sub_area
    
    # Normalize by the total area in each pixel
    nonzero_weights = healpix_weights > 0
    healpix_map[nonzero_weights] /= healpix_weights[nonzero_weights]
    healpix_map[~nonzero_weights] = np.nan  # Mark pixels with no data
    
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
    
    # Interpolate from spherical triangles to HEALPix
    healpix_map = conservative_interp_spherical_tri2healpix(
        nodes_src, elem_src, field_src, nside, k_subdivide=2  # Reduced subdivisions for speed
    )
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot HEALPix map
    hp.mollview(healpix_map, title="Interpolated HEALPix map", sub=(1, 3, 1))
    
    # For the source mesh, we need to convert to spherical coordinates and plot
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    
    # Create a scatter plot of the source mesh
    ax = plt.subplot(1, 3, 2, projection='mollweide')
    sc = ax.scatter(phi, np.pi/2 - theta, c=field_src, s=5, cmap='viridis')
    plt.colorbar(sc, ax=ax)
    ax.set_title("Source spherical mesh")
    
    # Plot the difference (for validation)
    # For this, we need to interpolate back to the source points
    theta_src = np.arccos(z)
    phi_src = np.arctan2(y, x)
    pix_idx_src = hp.ang2pix(nside, theta_src, phi_src)
    field_interp = healpix_map[pix_idx_src]
    
    ax = plt.subplot(1, 3, 3, projection='mollweide')
    diff = field_src - field_interp
    sc = ax.scatter(phi_src, np.pi/2 - theta_src, c=diff, s=5, cmap='RdBu_r')
    plt.colorbar(sc, ax=ax)
    ax.set_title("Difference (source - interpolated)")
    
    plt.tight_layout()
    plt.show()