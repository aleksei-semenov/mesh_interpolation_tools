# §4 Smooth analytic field: total approximation error

Spec: [Experiment List.md](../Experiment%20List.md), section 4.

## Goal

Use q = 1 + 0.5 sin(2πx) cos(2πy), with source cells initialised from true cell
averages. Separate the error into three norms; do not add them:
geometric = ‖q_sub − q_overlap‖, representation = ‖q_overlap − q_analytic‖,
total = ‖q_sub − q_analytic‖.

## Figure 4

E2 vs k (log). Draw one curve each for geometric, representation (flat) and total error.
Use separate panels for piecewise-constant and linear reconstruction.

## Status

| Item | Where | Status |
|------|-------|--------|
| three-way error split, constant/linear/quadratic reconstruction | `suites/smooth_robustness_benchmarks` → `smooth_refinement_*.pdf` (panel 1) | done |
| quadrature accuracy check of source/target references | same suite (stops if > 1e-10) | done |
| convergence tables | `convergence_sine.tex`, `convergence_quadratic.tex` | done |
| reconstruction = the package's reconstruction? | suite's own least-squares fit | **check** |

Run (from `../suites/smooth_robustness_benchmarks`):

```bash
python run_benchmarks.py --out ../../s04_smooth_field/results
python plot_benchmarks.py --out ../../s04_smooth_field/results --fit-k 1 4
```

The same run also produces the outputs for §5–§7.

## Open work, belongs here

- [ ] Decide which reconstruction the article attributes to the package, and make
      sure the figure uses exactly that one.
