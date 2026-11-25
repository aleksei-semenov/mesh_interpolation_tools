# §7 Robustness case suite

Spec: [Experiment List.md](../Experiment%20List.md), section 7.

## Goal

Repeat the constant and fixed piecewise-constant tests for these cases:
coarse→fine, fine→coarse, shifted target, rotated source, distorted source, mask,
and partial coverage. Define the reference quantity for masks and partial coverage
**before** running. For partial coverage, report retained, excluded and unexplained mass.

## Figure 7

Two heatmaps with k on x and the case on y. A = geometric E2 (log colour scale);
B = unsampled fraction of the intended valid area. Mark unavailable values.

## Status

| Case | In `suites/smooth_robustness_benchmarks` |
|------|------------------------------------------|
| baseline, coarse→fine, fine→coarse | done |
| shifted target | done |
| distorted source (jitter) | done |
| mask (central patch removed) | done (full-cell-average definition) |
| rotated source (tapered rotation, reject inverted) | **open** |
| partial coverage [0.25, 0.75]×[0, 1] + mass bookkeeping | **open** |
| constant-field version of the heatmaps | **open**: the suite uses only the fixed field |

## Open work, belongs here

- [ ] Add the rotated-source and partial-coverage cases, and extend the heatmaps.
- [ ] Print conservation residuals next to the heatmaps, without combining them into a score.
