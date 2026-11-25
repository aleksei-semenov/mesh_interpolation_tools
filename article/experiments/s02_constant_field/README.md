# §2 Constant field: normalization trade-off

Spec: [Experiment List.md](../Experiment%20List.md), section 2.

## Goal

Use q = 1. Show the area defect d = W/A − 1 and how the two normalizations differ:
W-normalization returns 1 wherever a cell is sampled, while A-normalization returns
W/A. Extra printouts: maximum constant error, area-weighted mean and maximum |d|,
and min/max W/A.

## Figures

- **Fig. 1** (coarse and fine k): A = map of d, B = q−1 (W), C = q−1 (A).
  Use diverging colours centred at 0, mark undefined cells, and match colour limits across k.
- **Fig. 2** vs k: A = |ε_M| per normalization, B = max constant error,
  C = unsampled area fraction. Use log y-axes and show exact zeros separately.

## Status

| Item | Where | Status |
|------|-------|--------|
| Fig. 1 | `suites/constant_geometry_benchmarks` → `area_diagnostics.pdf` | done |
| Fig. 2 | `suites/constant_geometry_benchmarks` → `constant_convergence.pdf` | done |
| diagnostic table | `diagnostics_table.tex` | done |
| run at protocol sizes | – | open |

Run (from `../suites/constant_geometry_benchmarks`):

```bash
python run_benchmarks.py --source-n 32 --target-n 41 --k 0 1 2 3 4 5 6 --out ../../s02_constant_field/results
python plot_benchmarks.py --out ../../s02_constant_field/results
```

## Open work, belongs here

- [ ] Rerun Figs. 1–2 with W taken from the package (see §1).
