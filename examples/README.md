# Examples for Mesh Interpolation Tools

This folder contains Jupyter Notebooks showing how to use the different interpolation methods implemented in this package.

Each notebook generates a synthetic test field, runs the interpolation, and visualizes the results.

---

## Available Examples

- **`tri2reg_example.ipynb`**  
  Interpolation from a triangular mesh to a regular structured grid.

- **`tri2arb_example.ipynb`**  
  Interpolation from a triangular mesh to a non-uniform rectangular grid (cells may have different sizes).

- **`tri2tri_example.ipynb`**  
  Interpolation from one triangular mesh to another triangular mesh.

- **`tri2healpix_example.ipynb`**  
  Interpolation from a spherical triangular mesh to a HEALPix spherical grid.

---

## How to run

1. Install the package locally:
   ```bash
   pip install -e 
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook examples/ 
   ```
3. Open any of the notebooks and run all cells.

## Import style

When running examples, import functions like this:
```python
from mesh_interpolation_tools.tri2arb import tri2arb
from mesh_interpolation_tools.tri2reg import tri2reg
from mesh_interpolation_tools.tri2tri import conservative_interp_tri2tri
from mesh_interpolation_tools.tri2healpix import conservative_interp_spherical_tri2healpix
```