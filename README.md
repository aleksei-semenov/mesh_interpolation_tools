# Mesh Interpolation Tools

Conservative interpolation routines for mapping fields between triangular meshes, structured grids, and HEALPix spherical grids.
This package is designed for geophysical and climate applications where flux conservation is essential.

---

## Features

* **Triangular → Regular grid** (`tri2reg`)
* **Triangular → Arbitrary rectangular grid** (`tri2arb`)
* **Triangular → Triangular mesh** (`tri2tri`)
* **Spherical triangular mesh → HEALPix** (`tri2healpix`)

All methods conserve integrals of the interpolated field.

---

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/aleksei-semenov/mesh_interpolation_tools.git"
```

Optional extras:

```bash
pip install "mesh-interpolation-tools[healpix] @ git+https://github.com/aleksei-semenov/mesh_interpolation_tools.git"  # HEALPix (healpy)
pip install "mesh-interpolation-tools[jax] @ git+https://github.com/aleksei-semenov/mesh_interpolation_tools.git"      # + JAX version
```

> `healpy` provides no Windows wheels — use Linux, macOS or WSL for HEALPix interpolation.
> The 2D tools (`tri2reg`, `tri2arb`, `tri2tri`) work everywhere.

For development (editable install):

```bash
git clone https://github.com/aleksei-semenov/mesh_interpolation_tools.git
cd mesh_interpolation_tools
pip install -e ".[dev]"
```

### Building and publishing

```bash
python -m build          # creates dist/*.whl and dist/*.tar.gz
twine check dist/*
twine upload --repository testpypi dist/*   # try TestPyPI first
twine upload dist/*                          # then real PyPI
```

After uploading to PyPI, users can simply `pip install mesh-interpolation-tools`.

---

## Usage

### 1. Triangular → Regular grid

```python
import numpy as np
from mesh_interpolation_tools.tri2reg import tri2reg

# Example triangular mesh (nodes, elements, field)
nodes = np.random.rand(2, 10)
elem = np.array([[0, 1, 2], [2, 3, 4]]).T
field = np.random.rand(10)

# Structured grid axes
grid_x = np.linspace(0, 1, 50)
grid_y = np.linspace(0, 1, 50)

grid_x, grid_y, heatmap = tri2reg(nodes, elem, field, grid_x, grid_y)
```

---

### 2. Triangular → Arbitrary grid

```python
from mesh_interpolation_tools.tri2arb import tri2arb

grid_x = np.array([0, 0.1, 0.4, 1.0])  # non-uniform
grid_y = np.array([0, 0.2, 0.5, 1.0])

grid_x, grid_y, heatmap = tri2arb(nodes, elem, field, grid_x, grid_y)
```

---

### 3. Triangular → Triangular mesh

```python
from mesh_interpolation_tools.tri2tri import conservative_interp_tri2tri

field_tgt = conservative_interp_tri2tri(nodes, elem, field, nodes, elem)
```

---

### 4. Spherical triangular mesh → HEALPix

```python
from mesh_interpolation_tools.tri2healpix import conservative_interp_spherical_tri2healpix

healpix_map = conservative_interp_spherical_tri2healpix(nodes, elem, field, nside=32)
```

---

## Examples

Interactive Jupyter notebooks are available in the [`examples/`](./examples) folder:

* `tri2reg_example.ipynb` – triangular → regular grid
* `tri2arb_example.ipynb` – triangular → arbitrary rectangular grid
* `tri2tri_example.ipynb` – triangular → triangular mesh
* `tri2healpix_example.ipynb` – spherical triangular mesh → HEALPix

To run:

```bash
cd examples
jupyter notebook
```

---

## Tests

Smoke tests are in the `tests/` folder. To run:

```bash
pytest
```