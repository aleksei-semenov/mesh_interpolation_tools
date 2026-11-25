# §6 Independent refinement

Spec: [Experiment List.md](../Experiment%20List.md), section 6.

## Goal

Using the smooth field, refine one thing at a time:

- **A**: subdivision k, with both meshes fixed. This is the §4 run.
- **B**: source lattice size, on a fixed target at depth `--refine-k`.
- **C**: target grid size, on a fixed source at depth `--refine-k`.

## Figure 6

`fig6_refinement_<field>.pdf`:
- A: geometric E2 vs 2^-k, read from §4 results
- B: total E2 vs h_S
- C: geometric E2 vs h_T

Colours mark the reconstruction; ○/solid = W and □/dashed = A. Circled points in B
have geometric error ≥ 10 % of total error.

## Code

| File | Role |
|------|------|
| `run_refinement.py` | sweeps B and C with `s04_smooth_field.smooth.smooth_group` (package-based) → `results/refinement_diagnostics.csv` |
| `plot_fig6.py` | Figure 6; panel A uses `--subdivision-data` (default: §4 results) |

```bash
python -m article.experiments.s04_smooth_field.run_smooth          # needed for panel A
python -m article.experiments.s06_independent_refinement.run_refinement
python -m article.experiments.s06_independent_refinement.plot_fig6
```

## Status

| Item | Status |
|------|--------|
| sweeps A–C and Figure 6 | **done, package-based** (demo: source 5/8/12, target 6/11/16, k = 4) |
| contamination flag on the source sweep | done |
| protocol sizes 16–128 | **open** |

Larger example (memory grows ~ N·4^k; see ../README.md):

```bash
python -m article.experiments.s06_independent_refinement.run_refinement \
  --source-n 32 --target-n 32 --source-sweep 16 32 64 --target-sweep 16 32 64 --refine-k 5
```

## Open work

- [ ] Production-size runs, possibly one sweep and field at a time (`--fields sine`).
