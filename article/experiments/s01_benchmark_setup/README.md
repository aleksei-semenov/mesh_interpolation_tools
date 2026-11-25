# §1 Reproducible benchmark and diagnostic record

Spec: [Experiment List.md](../Experiment%20List.md), section 1.

## Goal

A single planar unit-square benchmark: a jittered lattice source split into triangles,
a Cartesian target, and a k-sweep. Two normalizations: `q = S/W` (sampled area) and
`q = S/A` (physical area). Every run prints the same diagnostic record.

## Code

| File | Role |
|------|------|
| `package_sampling.py` | **method under test, from the package**: `package_weight_matrix` (the package's `calculate_interpolation_weights` at fixed k), `package_samples` (the package's `subdivide_triangle_iterative` plus its assignment rule), `check_samples_match_package`, `dropped_area` |
| `reference_geometry.py` | independent reference: lattice mesh, validation, exact triangle/rectangle clipping (`overlap_matrix`, `intersection_polygons`), `build_case` with coverage checks |
| `diagnostics.py` | E1/E2/E∞, normalizations, CSV and `metadata.json` with provenance |
| `run_benchmark.py` | constant and fixed piecewise-constant fields over k; data for §2 and §3 |
| `test_s01.py` | clipping and partition checks, package path vs. sample path, conservation, restoring the fixed-depth patch |

```bash
python -m article.experiments.s01_benchmark_setup.run_benchmark                       # demo: 12/17, k 0–5, ~13 s
python -m article.experiments.s01_benchmark_setup.run_benchmark --source-n 32 --target-n 32 --k 0 1 2 3 4 5 6
python -m article.experiments.s01_benchmark_setup.run_benchmark --mesh my_mesh.npz    # nodes(N,2), triangles(M,3), x_edges, y_edges
```

Outputs in `results/`:
- `diagnostics.csv`: the record, one row per field, normalization and k; also printed as JSON lines
- `diagnostics_table.tex`
- `adjacent_rates.csv`
- `boundary_cell_areas.csv`: assigned vs. exact area in selected cut cells
- `exact_overlaps.npz` and `assigned_areas_k*.npz`: sparse matrices
- `fields_k*.npz`, `mesh_reference.npz`, `metadata.json`

## Status

| Item | Status |
|------|--------|
| exact-overlap reference | done |
| weights from the package | **done** (`weight_build_seconds` = package wall time) |
| diagnostic record per run | done, except `peak_memory` |
| protocol sizes (32/32, k ≤ 6) | run with the flags above; watch memory (see ../README.md) |

## Open work

- [ ] Add `peak_memory` (e.g. `tracemalloc` around `package_weight_matrix`) to the record.
- [ ] Use one diagnostic record format in all sections (§4–§7 currently write their own column sets).
