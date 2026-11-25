# §1 Reproducible benchmark and diagnostic record

Spec: [Experiment List.md](../Experiment%20List.md), section 1.

## Goal

A single planar unit-square benchmark (32×32 lattice source split into triangles,
32×32 Cartesian target, k = 0…6), run with two normalizations: `q = S/W` (sampled
area) and `q = S/A` (physical area). Every run prints the same diagnostic record:
integrals M_S, M_deposit, M_W, M_A; residuals normalized by B = Σ|q|A; unassigned
and unsampled area; E1, E2, E∞; output extrema; timings; **peak memory**.

## Status

| Item | Where | Status |
|------|-------|--------|
| triangle–rectangle exact-overlap reference | `suites/*/geometry.py` (`overlap_matrix`) | done |
| diagnostic record per run | `suites/constant_geometry_benchmarks` → `diagnostics.csv` | done, except peak memory |
| same record in smooth suite | `suites/smooth_robustness_benchmarks` → `smooth_diagnostics.csv` | different column set |
| protocol sizes (32/32, k ≤ 6) | suite defaults are 12/17, k ≤ 5 | run with `--source-n 32 --target-n 32 --k 0 1 2 3 4 5 6` |
| driven by the package's own weights | – | **open** (see ../README.md) |

## Open work, belongs here

- [ ] One shared `diagnostics.py`, used by every section, that produces the full
      record (including `peak_memory`, via `tracemalloc`) from (source mesh, target grid,
      q_source, k, normalization).
- [ ] An adapter that obtains W and S from `mesh_interpolation_tools.tri2reg`
      (`calculate_interpolation_weights` with fixed k) instead of the suites' re-implementation.
- [ ] A consistency test: package weights vs. suite `sample_matrix` on the same mesh.
