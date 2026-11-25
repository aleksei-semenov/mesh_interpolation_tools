# §4 Smooth analytic field: total approximation error

Spec: [Experiment List.md](../Experiment%20List.md), section 4.

## Goal

Use q = 1 + 0.5 sin(2πx) cos(2πy) (`sine`), and also a `quadratic` field, with source
cells initialised from true cell averages. Separate the error into three norms; do not
add them: geometric = ‖q_sub − q_overlap‖, representation = ‖q_overlap − q_analytic‖,
total = ‖q_sub − q_analytic‖.

## Figure 4

`fig4_error_components_<field>.pdf` has one panel per reconstruction (constant, linear).
x = k; y = E2 (log). Lines: geometric (solid), representation (dotted, flat) and total
(dashed). Markers: ○ = W, □ = A. With `--fit-k`, the fit range is shaded and the slopes
are written to `fitted_rates_<field>.csv`.

## Code

| File | Role |
|------|------|
| `smooth.py` | fields, independent quadrature references, least-squares reconstruction, `package_deposits` (package weights for constant; package centroids, verified against package weights, for linear), `smooth_group`. Also used by §6 |
| `run_smooth.py` | subdivision sweep → `results/smooth_diagnostics.csv`, `convergence_<field>.tex` |
| `plot_fig4.py` | Figure 4 |
| `test_smooth.py` | quadrature rules, reconstruction exactness, clipped integrals, conservation of package deposits |

```bash
python -m article.experiments.s04_smooth_field.run_smooth            # demo: 8/11, k 0–4
python -m article.experiments.s04_smooth_field.plot_fig4 --fit-k 1 4
```

## Status

| Item | Status |
|------|--------|
| three-way error split, constant and linear | **done, package-based** |
| quadrature accuracy checks of the references | done (stops if > 1e-10) |
| convergence tables | done |

## Changes from the old suite

- Every reconstruction is now deposited with the package's **centroid** rule.
  The old quadratic reconstruction used three degree-2 quadrature nodes per child,
  which is not the package's method. It was removed from the defaults.
  `--reconstructions constant linear quadratic` still runs it, evaluated at the package centroids.
- The linear reconstruction is benchmark code; the package has no reconstruction yet (see §8).

## Open work

- [ ] Decide which reconstruction the article attributes to the package (currently only constant is the package's own).
