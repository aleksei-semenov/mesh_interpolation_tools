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

All data is produced by `../s01_benchmark_setup/run_benchmark.py` with the **package's
weights**, in `../s01_benchmark_setup/results/`.

| Item | Where | Status |
|------|-------|--------|
| fixed field, overlap reference, errors per k | `diagnostics.csv` (`field == fixed`), `fields_k*.npz` (`field_W`, `field_A`), `mesh_reference.npz` (`q_reference`) | **done, package weights** |
| panel A | Fig. 2 (d) in §2 | done |
| panel B, slope | only `adjacent_rates.csv` | **open** |
| panel C, runtime vs error | package time in `weight_build_seconds`, overlap time in `metadata.json` (`reference_seconds`); not plotted | **open** |
| error map | `fields_k*.npz` holds the data; not plotted | **open** |
| assigned vs exact areas in cut cells | `boundary_cell_areas.csv` | done |

## Open work, belongs here

- [ ] `plot_fig3.py`: reads `../s01_benchmark_setup/results/` and draws panels A–C plus the error map.
- [ ] Decide whether panel A stays in Fig. 2 (d) or moves into Fig. 3.
