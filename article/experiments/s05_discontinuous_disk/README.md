# §5 Discontinuous field: bounds and interface distortion

Spec: [Experiment List.md](../Experiment%20List.md), section 5.

## Goal

Use a disk indicator (r = 0.2, centred at (0.5, 0.5)), with source averages taken from
exact disk–triangle areas. Report undershoot/overshoot, area fractions below 0 and
above 1, the integrated negative part, and the L1 error against analytic cell averages,
all with a declared tolerance.

## Figure 5

- A: reference map
- B: computed map on the same colour scale
- C: signed error map
- D: step profile through the row nearest y = 0.5
- optional: displacement of the 0.5 contour (mean and maximum)

## Status

| Item | Where | Status |
|------|-------|--------|
| disk source/target averages, bound diagnostics | `suites/smooth_robustness_benchmarks` → `disk_diagnostics.json`, `disk_fields.npz` | done |
| panels A–D | `robustness.pdf` (top row) | done |
| contour displacement | – | **open** (optional) |

## Open work, belongs here

- [ ] `contour_displacement.py`: run one contour procedure on both output and reference,
      report mean and maximum distance, and flag missing or spurious contours.
- [ ] Consider giving Fig. 5 its own file, separate from the robustness heatmaps (§7).
