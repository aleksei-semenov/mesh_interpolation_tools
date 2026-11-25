# §3 Fixed piecewise-constant field: geometric assignment error

Spec: [Experiment List.md](../Experiment%20List.md), section 3.

## Goal

Give each source triangle a fixed value q_i = 1 + 0.5 sin(2πx_c) cos(2πy_c), held
constant across the k-sweep. The reference is the exact-overlap remap
q_ref = Σ a_ji q_i / A_j. The question this answers: does subdivision converge to
the correct remap of the *same* source?

## Figure 3

- **A**: E1, E2, E∞ vs k (log y), separated by normalization.
- **B**: E2 vs 2^-k (log–log), with the slope fitted only over a coverage-complete range.
- **C**: E2 vs total runtime (log–log), with the exact-overlap method as a single point.
- An error map q − q_ref at an intermediate k.

## Status

| Item | Where | Status |
|------|-------|--------|
| fixed field, overlap reference, errors per k | `suites/constant_geometry_benchmarks` (`diagnostics.csv`, `fields_k*.npz`) | done |
| panel A | partly in `constant_convergence.pdf` | check whether it suffices |
| panel B, slope | only `adjacent_rates.csv` | **open** |
| panel C, runtime vs error | timings are saved; no plot | **open** |
| error map | `fields_k*.npz` holds the data; no plot | **open** |
| assigned vs exact areas in cut cells | `boundary_cell_areas.csv` | done |

## Open work, belongs here

- [ ] `fig3_geometric_convergence.py`: reads the suite's results directory
      (`diagnostics.csv`, `fields_k*.npz`, `metadata.json`) and draws panels A–C plus the error map.
- [ ] Measure runtime consistently for the package path and the overlap reference.
