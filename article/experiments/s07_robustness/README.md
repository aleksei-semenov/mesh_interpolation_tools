# §7 Robustness case suite

Spec: [Experiment List.md](../Experiment%20List.md), section 7.

## Goal

Repeat the constant and fixed piecewise-constant tests for these cases:
coarse→fine, fine→coarse, shifted target, rotated source, distorted source, mask,
and partial coverage. Define the reference quantity for masks and partial coverage
**before** running. For partial coverage, report retained, excluded and unexplained mass.

## Figure 7

`fig7_robustness.pdf` has two heatmaps with k on x and the case on y:
- A: geometric E2 of the fixed field (A-normalization, log colour scale)
- B: missing fraction of the valid area

Exact zeros are printed as `0`; unavailable values as `×`.

## Code

| File | Role |
|------|------|
| `run_robustness.py` | package weights per case and k → `results/robustness_diagnostics.csv` (also physical mass residual and package time) |
| `plot_fig7.py` | Figure 7 |

```bash
python -m article.experiments.s07_robustness.run_robustness [--cases baseline mask]
python -m article.experiments.s07_robustness.plot_fig7
```

For the mask, output is the full-target-cell average, with zero outside the valid source.
E2 counts only cells with any valid overlap.

## Status

| Case | Status |
|------|--------|
| baseline, coarse→fine, fine→coarse | **done, package weights** |
| shifted target | **done, package weights** |
| distorted source (jitter 0.24) | **done, package weights** |
| mask (central patch removed) | **done, package weights** |
| rotated source (tapered rotation, reject inverted) | **open** |
| partial coverage [0.25, 0.75]×[0, 1] + mass bookkeeping | **open** |
| constant-field version of the heatmaps | **open**: only the fixed field is run |

## Open work

- [ ] Add the rotated-source and partial-coverage cases in `case_settings` / `build_case`.
- [ ] Show the conservation residuals (already in the CSV) next to the heatmaps.
