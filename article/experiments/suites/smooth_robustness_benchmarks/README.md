# Smooth-field and robustness benchmark bundle

This standalone planar benchmark generates the smooth-refinement figure,
convergence tables, and disk/coverage figure in your manuscript. It reuses the
midpoint-subdivision and rectangle-clipping geometry from the previous benchmark.
It does **not** call or validate your repository's full public API. Linear and
quadratic least-squares reconstructions here are explicit benchmark
implementations, not necessarily the same reconstruction as in your package.

The `defaults/` folder contains actual outputs of a synthetic demonstration
run, not mock-ups and not evidence from your final application. Review and rerun
with your intended settings before using them as manuscript results.

## Install and run

Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python run_benchmarks.py --out results
python plot_benchmarks.py --out results --fit-k 1 4
```

The default calculation is deliberately small. `--help` lists all settings. For
example:

```bash
python run_benchmarks.py --source-n 12 --target-n 17 \
  --source-sweep 6 12 24 --target-sweep 9 17 25 \
  --k 0 1 2 3 4 5 --refine-k 5 --disk-k 5 --out results_large
python plot_benchmarks.py --out results_large --fit-k 2 5
```

Work and memory increase rapidly with subdivision depth. Test a small run
first. `source-n` is the number of square lattice intervals along a side, with
two source triangles per square. `target-n` is the number of target cells
along a side. The source mesh is jittered reproducibly using `--seed`.

## Outputs

- `smooth_refinement_sine.pdf/.png` and `smooth_refinement_quadratic.pdf/.png`:
  three panels in the requested arrangement. Colors distinguish reconstruction;
  circles/squares distinguish sampled-area (W) and physical-area (A)
  normalization. Lines in the first panel distinguish geometric,
  representation, and total errors. Source-refinement points circled in black
  have geometric error at least 10% of total error: their source-refinement
  rates may be contaminated by geometric assignment error. The threshold is
  a diagnostic choice, not a guarantee that other points are asymptotic.
- `convergence_sine.tex`, `convergence_quadratic.tex`: complete-coverage rows
  of the subdivision sweep, with field, mesh sizes, reconstruction,
  normalization, $k$, $E_1$, $E_2$, $E_\infty$, adjacent-depth $p_k$, and
  missing fraction $f_0$. `--` means unavailable. Tables can be lengthy; split
  or filter rows for publication. All other rows, including incomplete-coverage
  runs, remain in `smooth_diagnostics.csv`.
- `robustness.pdf/.png`: disk reference, output and signed-error maps; target
  cell-average step profile; and fixed-field robustness heatmaps.
- `smooth_diagnostics.csv`: all three sweeps, every field, method, norm,
  quadrature degree, sample count, errors, residuals, missing fraction and
  deposition time. Timings are not a complete performance comparison.
- `disk_fields.npz`, `disk_diagnostics.json`: disk fields, exact-reference
  checks, extrema, overshoot, undershoot, integral of the negative/above-one
  parts, area fractions outside bounds (using a documented tolerance), mass
  residual and errors.
- `robustness_diagnostics.csv`: geometric errors and missing valid-area
  fractions by configuration and depth. `metadata.json` stores command settings.
- If `--fit-k START END` is supplied when plotting, the selected interval is
  shaded and `fitted_rates_FIELD.csv` records least-squares slopes of geometric
  $E_2$ versus $2^{-k}$, the actual included depths, and sample count. Each fit
  uses only fully sampled, finite, positive points and requires at least three.
  Without `--fit-k`, no fit or fitted-rate file is generated. A fitted slope is
  descriptive, not proof of an asymptotic order.

Exact zeros on logarithmic plots appear as downward markers in a separate
status strip, not at artificial positive values. Crosses mark unavailable
values. Heatmap zeros are printed as `0`; unavailable values are marked `×`.

## Definitions and safeguards

The domain is the unit square and source input consists of **true source-cell
averages**, evaluated independently with a Duffy-product Gauss rule. The code
recomputes them with a higher-order rule and stops if the maximum discrepancy
exceeds `1e-10`. Target references use an independent tensor Gauss rule and
are similarly checked. The smooth fields are
`1+0.5*sin(2*pi*x)*cos(2*pi*y)` and
`1+0.3*x+0.2*y+0.4*x*x-0.2*x*y+0.1*y*y`.

- Constant, linear, and quadratic source representations are mean-preserving.
  Linear and quadratic coefficients come from least-squares fits to neighboring
  **cell averages**, including triangle second moments. Rank-deficient or
  severely ill-conditioned fits raise an error; no silent fallback is made.
- Constant and linear reconstructions use centroid quadrature. For the
  quadratic reconstruction, each child triangle uses the three symmetric
  degree-two nodes, each with one-third of its area; **each node is located
  independently in a target cell**. Thus a quadratic reconstruction has
  three times as many quadrature nodes as the other reconstructions at the same
  mesh and subdivision depth. The rule is selected by the reconstruction,
  not by the name of the underlying analytic field.
- The independent geometric reference clips each source triangle against
  each target rectangle and integrates the chosen reconstructed polynomial
  over the resulting polygons using a degree-two triangle rule. 'Exact overlap'
  means exact for the polynomial up to floating-point geometry and rounding,
  not symbolic exact arithmetic.
- Geometric error compares the subdivision output against that overlap
  reconstruction; representation error compares overlap reconstruction with
  analytic target-cell averages; total error compares subdivision output
  with the analytic averages. Norms are not added to each other. The source
  representation is fixed during the subdivision sweep, so its representation
  curve is constant across $k$.
- $E_1$ and $E_2$ are physical-area-weighted; $E_\infty$ is the maximum
  absolute difference. Whole-domain norms and physical integrals for W are
  unavailable if any required target cell is unsampled. A defines output as
  zero when deposited quantity is zero, but the missing-area fraction remains
  visible. `p_k` uses adjacent available depths of the fixed-mesh sweep and
  total $E_2$ versus $2^{-k}$; both endpoints must be fully sampled, finite,
  and positive. The first depth has no previous rate.
- Source and target meshes are varied **separately**. The source sweep holds
  target resolution fixed and uses `--refine-k`; inspect geometric errors and
  increase that depth if needed before drawing a source-order conclusion.
  The target sweep fixes the source mesh and reconstruction and measures
  geometric error. It is not, by itself, a convergence study of the original
  continuous field.

## Disk and robustness suite

The disk is an indicator of radius 0.2 centered at (0.5, 0.5). Source and
independent target cell averages are computed by adaptive one-dimensional
circle/triangle and circle/rectangle integration, respectively. The code also
prints the source and target disk integrals alongside $\pi r^2$. Tighten
`--disk-tol` and compare results when reference precision matters. The disk
output uses piecewise-constant reconstruction and A normalization; its error
map compares to analytic disk target averages. The saved geometric error uses
instead the clipped overlap reference of the source reconstruction.

The heatmaps use a **different fixed piecewise-constant sinusoid-based source
field**, not the disk. Cases are baseline, coarse-to-fine, fine-to-coarse,
shifted target boundaries, distorted source vertices, and a mask made by
removing a central patch of source triangles. Heatmaps use A normalization and
the exact-overlap reference of that fixed field. For the mask, values outside
the valid source region are defined as zero and output is a **full-target-cell
average**. Geometric $E_2$ uses physical whole-cell areas of cells with any
valid intersection; entirely invalid target cells are excluded. Missing valid
area is the exact valid overlap area in cells with no sample divided by the
total valid overlap area, so it does not mistake intentionally masked area for
a sampling hole. This is not a covered-area-normalized field product.

No rotation, external target-domain partial coverage, nonrectangular target,
spherical, or HEALPix experiment is implemented here. Only the deterministic
synthetic geometry is supported by the driver; adapting it to your own mesh
requires explicit input handling and geometry-validation checks.

## Insert into the manuscript

Add `\usepackage{graphicx}` and `\usepackage{booktabs}` in the preamble. Replace
figure placeholders with, for example:

```latex
\includegraphics[width=\linewidth]{results/smooth_refinement_sine.pdf}
% Alternatively: results/smooth_refinement_quadratic.pdf
```

```latex
\includegraphics[width=\linewidth]{results/robustness.pdf}
```

Replace the original table's `tabular` block with the generated one; do not
nest it inside another `tabular`:

```latex
\begin{table}[htbp]
\centering
\caption{Smooth-field convergence for fully sampled target cells.}
\label{tab:convergence}
\resizebox{\linewidth}{!}{\input{results/convergence_sine.tex}}
\end{table}
```

Update the captions to identify the field, normalizations, quadrature, and the
fact that the heatmaps use a fixed field distinct from the disk.

## Validation performed

The test suite verifies degree-one/degree-two integration, node-by-node target
assignment, exact quadratic least-squares reconstruction and clipped-polygon
integrals, disk/triangle and disk/rectangle area totals, and unavailable-error
handling. Both the default run and a smaller run were executed and the PDF/PNG
plots rendered. These checks do not establish correctness on every mesh, for
spherical geometry, or for the original package's public implementation.
