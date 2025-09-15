# mesh-interpolation-tools

A Python package for **conservative interpolation** of scalar fields between triangular meshes and structured grids.  
The package supports:
- Interpolation from **triangular mesh → rectangular grid** (uniform or non-uniform cell sizes).  
- Interpolation from **triangular mesh → triangular mesh**.  
- Extension to spherical meshes and HEALPix.  

This package is useful for geophysics, climate/ocean modeling, and numerical methods where conservation of scalar fields is required.

---

## Features
- **Conservative interpolation**: preserves integrals of the field.
- **Adaptive subdivision**: triangles subdivide until smaller than grid cells.
- **Non-uniform grids**: handles irregular spacing in structured grids.
- **Mesh-to-mesh**: interpolate from one triangular mesh to another.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aleksei-semenov/mesh-interpolation-tools.git
cd mesh-interpolation-tools
pip install -r requirements.txt
```

## Usage example

```python
import numpy as np
from scipy.spatial import Delaunay
from meshinterp import adaptive_conservative_interpolation

# Generate a triangular mesh
points = np.random.rand(50, 2)
tri = Delaunay(points)
triangles = tri.simplices
values = np.sin(2*np.pi*points[:,0]) * np.cos(2*np.pi*points[:,1])

# Interpolate to structured grid
gx = np.linspace(0, 1, 30)
gy = np.linspace(0, 1, 20)
gx, gy, heatmap = adaptive_conservative_interpolation(points, triangles, values,
                                                      grid_x=gx, grid_y=gy)

print("Interpolated heatmap shape:", heatmap.shape)
```

## Requirements

numpy

scipy

matplotlib