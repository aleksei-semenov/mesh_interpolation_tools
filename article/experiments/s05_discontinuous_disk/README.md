# §5 Discontinuous field: bounds and interface distortion

Spec: [Experiment List.md](../Experiment%20List.md), section 5.

## Goal

Use a disk indicator (r = 0.2, centred at (0.5, 0.5)), with source averages taken from
exact disk–triangle areas. Report undershoot/overshoot, area fractions below 0 and
above 1, the integrated negative part, and the L1 error against analytic cell averages,
all with a declared tolerance (1e-12).

## Figure 5

`fig5_disk.pdf`:
- A: reference
- B: package output (A-normalization), same colour scale as A
- C: signed error
- D: cell-average step profile through the row nearest y = 0.5

## Code

| File | Role |
|------|------|
| `disk.py` | independent disk/triangle and disk/rectangle areas (adaptive quadrature) |
| `run_disk.py` | package weights at depth k → `results/disk_diagnostics.json`, `disk_fields.npz` |
| `plot_fig5.py` | Figure 5 |
| `test_disk.py` | disk integrals sum to πr² |

```bash
python -m article.experiments.s05_discontinuous_disk.run_disk [--k 3]
python -m article.experiments.s05_discontinuous_disk.plot_fig5
```

## Status

| Item | Status |
|------|--------|
| bound diagnostics, errors vs analytic and vs overlap reference | **done, package weights** |
| Figure 5, panels A–D | **done** (now a separate figure, previously combined with §7) |
| contour displacement | **open** (optional) |

## Open work

- [ ] `contour_displacement.py`: run one contour procedure on both output and reference,
      report mean and maximum distance, and flag missing or spurious contours.
