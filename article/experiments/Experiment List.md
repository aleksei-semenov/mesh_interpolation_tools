Use a **planar unit-square benchmark first**, with a fixed set of diagnostics printed for every run. Establish the area accounting there before adding spherical geometry. Below is a proposed numerical protocol—not reported results or a prescribed literature standard. All parameter values are suggested starting choices.

## 1. Set up one reproducible benchmark

Use domain \(\Omega=[0,1]^2\).

- **Source:** triangulate a regular square lattice, initially with 32 squares along each direction. Split each square into two triangles.
- **Target:** initially use a Cartesian grid with 32 cells along each direction.
- **Subdivision:** run \(k=0,1,\ldots,6\).
- **Methods:** run your current \(q_j=S_j/W_j\) and, as a diagnostic alternative, \(q_j=S_j/A_j^T\). Label them separately; do not silently switch normalization between tests.
- **Reference:** implement or use an independent triangle–rectangle intersection calculation on these small meshes.

Start with full domain coverage, no masks, and piecewise-constant source values. Add the gradient reconstruction only after these tests work.

Your current formulation normalizes with sampled area and proves conservation under that measure; your stated conservation requirement uses physical target area. Therefore, print both integrals in every test. [^1]

### Print this diagnostic record for every run

```text
field, source_mesh_id, target_mesh_id, normalization, reconstruction
Ns, Nt, k, number_of_samples
source_integral
sum_of_deposited_quantities
sampled_area_integral
physical_area_integral
physical_mass_residual_normalized
sampled_mass_residual_normalized
unassigned_source_area
unsampled_target_area_fraction
L1_error, L2_error, Linfinity_error
minimum_output, maximum_output
geometry_setup_seconds, field_application_seconds, peak_memory
```

Use these definitions:

\[
M_S=\sum_i q_i^S A_i^S,\quad
M_{\rm deposit}=\sum_j S_j,\quad
M_W=\sum_j q_jW_j,\quad
M_A=\sum_j q_jA_j^T.
\]

For signed or zero-mean fields, normalize mass residuals by
\[
B=\sum_i |q_i^S|A_i^S,
\qquad \epsilon_M=(M_A-M_S)/B,
\]
not by \(M_S\). Treat an identically zero input separately.

For full coverage, define \(e_j=q_j-q_j^{\rm ref}\) and print
\[
E_1=\frac{\sum_j A_j^T|e_j|}{\sum_j A_j^T},\qquad
E_2=\sqrt{\frac{\sum_j A_j^T e_j^2}{\sum_j A_j^T}},\qquad
E_\infty=\max_j|e_j|.
\]

**If any required target cell is undefined, mark whole-domain field errors as unavailable.** You may additionally print errors on the sampled subset, explicitly labelled “conditional on sampled cells,” alongside the missing-area fraction. Do not present those as whole-domain errors. Your draft currently excludes unsampled cells from local errors. [^1]

## 2. Constant field: expose the normalization trade-off

**Input:** \(q_i^S=1\). The exact answer is one in every target cell, and the exact domain integral is one.

**Print additionally:**

```text
maximum_absolute_constant_error
area_weighted_mean_absolute_area_defect
maximum_absolute_area_defect
minimum_W_over_A, maximum_W_over_A
```

Define the area defect
\[
d_j=W_j/A_j^T-1.
\]
For this diagnostic, include unsampled cells: they have \(d_j=-1\).

### Figure 1 — spatial diagnostic

Make three panels at a coarse subdivision depth and repeat at a finer depth:

- **Panel A:** map of \(d_j\).
- **Panel B:** map of \(q_j-1\) for sampled-area normalization.
- **Panel C:** map of \(q_j-1\) for physical-area normalization.

**Axes:** horizontal = physical \(x\); vertical = physical \(y\). Use equal aspect ratio, diverging colors centered at zero, and explicitly mark undefined cells. Use matched color limits when comparing the same quantity across subdivision depths.

### Figure 2 — conservation and coverage versus subdivision

Use three panels:

- **A:** x = \(k\), y = absolute physical mass residual \(|\epsilon_M|\); separate curves for the two normalizations.
- **B:** x = \(k\), y = maximum constant error; separate curves for the two normalizations.
- **C:** x = \(k\), y = unsampled physical-area fraction.

Use logarithmic y-axes for positive errors; display exact zeros separately rather than replacing them with an arbitrary small value.

**Decisive interpretation:** for your piecewise-constant deposition, \(S_j=W_j\) in this test. Thus sampled-area normalization returns one wherever sampled, whereas physical-area normalization returns \(W_j/A_j^T\). This is a direct deduction from your accumulation and normalization equations—not an empirical performance claim. [^1]

## 3. Fixed piecewise-constant field: isolate geometric assignment error

**Input:** assign each source triangle its own value, for example
\[
q_i^S=1+0.5\sin(2\pi x_i^c)\cos(2\pi y_i^c).
\]
Once assigned, treat these as **piecewise-constant triangle values**. Keep the source mesh and values unchanged throughout the \(k\)-sweep.

Construct the reference using exact overlap areas:
\[
a_{ji}=|C_i^S\cap C_j^T|,\qquad
q_j^{\rm ref}=\frac{\sum_i a_{ji}q_i^S}{A_j^T}.
\]
The reference is the remapped piecewise-constant source field—not the smooth formula used to generate the triangle values.

### Figure 3 — geometric convergence

- **Panel A:** x = \(k\); y = \(E_1,E_2,E_\infty\), logarithmic y-axis. Use separate panels or line styles for normalization variants.
- **Panel B:** x = sample-size proxy \(2^{-k}\); y = \(E_2\), both axes logarithmic. Add a fitted slope only over a clearly identified, coverage-complete range.
- **Panel C:** x = total runtime; y = \(E_2\), both logarithmic. Include the exact-overlap method as a point, with its reference-construction cost measured consistently.

Also show one error map at an intermediate \(k\): x and y are physical coordinates; color is \(q_j-q_j^{\rm ref}\).

**Question answered:** does increasing subdivision approach the correct remap of the *same* source reconstruction?

## 4. Smooth analytic field: measure total approximation error

Use
\[
q(x,y)=1+0.5\sin(2\pi x)\cos(2\pi y).
\]

Initialize source triangles with true cell averages, using an accurate independent integration routine. Compute target reference averages by analytic integration over rectangles. Verify source quadrature accuracy by tightening its tolerance until the benchmark errors no longer change materially.

Compute two references:

- \(q^{\rm overlap}\): exact-overlap remapping of the source reconstruction.
- \(q^{\rm analytic}\): actual analytic target-cell averages.

Print three errors:

```text
geometric_error: norm(q_subdivision - q_overlap)
source_representation_error: norm(q_overlap - q_analytic)
total_error: norm(q_subdivision - q_analytic)
```

Do not add their norms as though they were scalar error components; cancellation is possible.

### Figure 4 — explain the error floor

- x = \(k\).
- y = area-weighted \(E_2\), logarithmic scale.
- Curves = geometric error, source-representation error, total error.
- Separate panels = piecewise-constant and linear reconstruction, once the latter is fully defined.

The source-representation curve is horizontal because the source mesh and reconstruction are fixed during this sweep. This makes it clear whether subdivision has become more accurate than the source representation warrants.

## 5. Discontinuous field: test bounds and interface distortion

Use the disk indicator
\[
q(x,y)=\begin{cases}
1,&(x-0.5)^2+(y-0.5)^2<0.2^2,\\
0,&\text{otherwise}.
\end{cases}
\]

Initialize source-cell averages from disk–triangle areas, not centroid classifications. Use independently accurate disk–target-cell areas for the analytic reference. Also retain the overlap reference of the source reconstruction so geometric error can be distinguished from source-resolution error.

**Print additionally:**

```text
undershoot = max(0, -minimum_output)
overshoot = max(0, maximum_output - 1)
area_fraction_below_0
area_fraction_above_1
integrated_negative_part
L1_error_against_analytic_cell_averages
```

Use a declared floating-point tolerance when counting bound violations; do not count numerical noise as a meaningful violation.

### Figure 5 — discontinuity behavior

- **Panel A:** target reference map.
- **Panel B:** computed target map, using the same color scale.
- **Panel C:** signed error map with a separate diverging color scale.
- **Panel D:** horizontal profile through the row nearest \(y=0.5\): x = target-cell-center \(x\); y = **cell-average value**. Plot the computed and reference cell averages as steps.

For interface displacement, optionally extract the 0.5 contour using the **same contour procedure** for output and target reference. Print mean and maximum contour-to-contour distance. Flag missing or spurious contours rather than ignoring them. This is a secondary visualization metric; it is not a replacement for cell-average error.

## 6. Independent refinement: do not change everything at once

Use the smooth field and conduct separate sweeps:

**Sweep A — subdivision:** fix both meshes; vary \(k\).

**Sweep B — source mesh:** fix a fine target grid; use source lattice sizes 16, 32, 64, and 128. Increase \(k\) as needed so geometric error is demonstrably below the error being studied. Compare against exact-overlap remapping too.

**Sweep C — target mesh:** fix the source field and source mesh; use target grid sizes 16, 32, 64, and 128. Compare with exact-overlap references for each target grid. Do not interpret finer targets alone as a convergence study of the underlying source field.

### Figure 6 — refinement panels

- **A:** x = \(2^{-k}\); y = geometric \(E_2\), log–log.
- **B:** x = source size \(h_S=\sqrt{|\Omega|/N_S}\); y = total \(E_2\), log–log. Separate curves for constant and linear reconstructions.
- **C:** x = target size \(h_T=\sqrt{|\Omega|/N_T}\); y = geometric \(E_2\), log–log. Separate curves for fixed subdivision depths.

Add slope guides only as guides—not as evidence of an order the data do not support.

## 7. Robustness: use a small, defined case suite

Repeat the constant and fixed piecewise-constant tests for:

- **Coarse → fine:** source lattice 16, target grid 64.
- **Fine → coarse:** source lattice 64, target grid 16.
- **Shifted target:** move internal target boundaries by half a nominal cell width while retaining the outer domain boundaries.
- **Rotated source:** rotate interior source vertices around the domain center by a tapered angle that goes to zero at the boundary. Reject inverted triangles.
- **Distorted source:** jitter interior vertices by up to 20% of lattice spacing using fixed random seeds; retain valid triangles and report minimum angle.
- **Mask:** remove a defined central patch from the valid source domain.
- **Partial coverage:** restrict the target domain to \([0.25,0.75]\times[0,1]\).

For masks and partial coverage, define the reference quantity **before running**. If the intended output is a full-cell average with zero outside the valid source domain, divide by full target area. If it is a valid-covered-area average, divide by valid overlap area and deliver that area fraction with the field. These are different products.

For partial coverage, compare deposited mass with the source reconstruction integrated over the source–target intersection—not with the entire source mass. Print retained mass, intentionally excluded mass, and unexplained residual separately.

### Figure 7 — robustness summary

Use two heatmaps:

- x = subdivision depth \(k\).
- y = mesh/coverage case.
- **Heatmap A color:** geometric \(E_2\), logarithmic color normalization.
- **Heatmap B color:** unsampled fraction of the intended valid target area.

Mark unavailable errors explicitly. Print conservation residuals alongside these results; do not combine conservation, accuracy, and coverage into one score.

## What to do first

Implement **the diagnostic record and constant-field figures first**. Then implement the exact-overlap reference and the fixed-source convergence test. Those establish what the method conserves and whether its geometric approximation converges. Only after that should you spend time on smooth-field order, higher-order reconstruction, spectral analysis, and large realistic cases.

[^1]: pasted-text-1.txt.