# §2 Constant field: normalization trade-off

Spec: [Experiment List.md](../Experiment%20List.md), section 2.

## Goal

Use q = 1. Show the area defect d = W/A − 1 and how the two normalizations differ:
W-normalization returns 1 wherever a cell is sampled, while A-normalization returns
W/A. Extra printouts: maximum constant error, area-weighted mean and maximum |d|,
and min/max W/A.

## Figures

- **Fig. 1** `area_diagnostics.pdf`. Rows are a coarse and a fine k; columns are
  the map of d, q−1 (W) and q−1 (A). Colours are diverging and centred at 0, undefined
  cells are gray, and colour limits match across k.
- **Fig. 2** `constant_convergence.pdf`, panels vs k:
  - (a) |ε_M| for each normalization
  - (b) maximum constant error
  - (c) unsampled area fraction
  - (d) fixed-field E1/E2/E∞, which is §3 Fig. 3A

  Log y-axes; exact zeros and unavailable values appear in a status strip.

## Code and status

| Item | Where | Status |
|------|-------|--------|
| data | `../s01_benchmark_setup/run_benchmark.py` → `../s01_benchmark_setup/results/` | **done, package weights** |
| Fig. 1, Fig. 2 | `plot_fig1_fig2.py` → `results/` | **done** |
| diagnostic table | `../s01_benchmark_setup/results/diagnostics_table.tex` | done |
| protocol sizes | – | open: rerun §1 with `--source-n 32 --target-n 32 --k 0 … 6` |

```bash
python -m article.experiments.s01_benchmark_setup.run_benchmark
python -m article.experiments.s02_constant_field.plot_fig1_fig2 [--map-k 0 5]
```
