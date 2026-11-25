# §8 Later: beyond the planar benchmark

Per "What to do first" in [Experiment List.md](../Experiment%20List.md), start these
only after §1–§3 are established.

## Planned topics (one subfolder each, when started)

- `spherical/`: spherical triangles to HEALPix via `tri2healpix` / `tri2healpix_jax`.
  Repeat the area accounting with spherical areas. Needs `healpy` (Linux/WSL).
- `higher_order/`: gradient (linear) reconstruction in the package itself. Today the
  linear reconstruction exists only as benchmark code (`s04_smooth_field/smooth.py`),
  evaluated at the package's centroids.
- `spectral/`: spectral analysis of remapped fields.
- `realistic/`: large realistic meshes (e.g. FESOM) for performance and memory.
- `other_targets/`: `tri2arb` (non-uniform grids) and `tri2tri`.

## Status

Not started.
