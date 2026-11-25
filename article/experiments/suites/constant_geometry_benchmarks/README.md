# Constant-field and geometric-assignment manuscript benchmarks

This is an executable companion to the two figure placeholders in your subsection.
It does not modify your package. The default engine implements the same planar
midpoint subdivision, centroid assignment, and equal child-area accumulation as
your `tri2reg` helper, but allows explicit subdivision depths and cell-valued input.
An optional mode calls the original helper directly. This does **not** exercise the
repository's complete public `tri2reg` API, its automatic depth selection, nodal
averaging, or multiprocessing. It is a controlled benchmark of the core algorithm.

## Run

Requires Python 3.10+ with NumPy, SciPy and Matplotlib:

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python run_benchmarks.py --source-n 12 --target-n 17 --k 0 1 2 3 4 5
python plot_benchmarks.py
```

For a larger study, use the same commands with `--source-n 32 --target-n 41`
and `--k 0 1 2 3 4 5 6`. Start small: work grows as `Ns * 4**k`.
The nonmatching target resolution and reproducible source jitter help avoid
accidental source/target alignment. These are proposed settings, not standards.

To use your original subdivision helper:

```bash
python run_benchmarks.py --repository-root /path/to/mesh_interpolation_tools-main --out repo_results
python plot_benchmarks.py --out repo_results
```

This loads `tri2reg.py` directly, bypassing unrelated package-level optional
HEALPix imports. The module must be the trusted code from your own checkout.

## Your own mesh

Save an NPZ containing:

```python
np.savez('benchmark_mesh.npz',
         nodes=nodes,             # (N,2), Cartesian x,y
         triangles=triangles,     # (M,3), zero-based integer indices
         x_edges=x_edges,         # strictly increasing target boundaries
         y_edges=y_edges)
```

Your repository uses transposed arrays: convert with `nodes=nodes_repo.T` and
`triangles=elem_repo.T` when needed. Then:

```bash
python run_benchmarks.py --mesh benchmark_mesh.npz --k 0 1 2 3 4 --out my_results
python plot_benchmarks.py --out my_results --map-k 0 4
```

This suite intentionally requires a conforming, nonoverlapping planar mesh and
full common-domain coverage. It validates overlap row and column area sums, but
these checks are not a general proof of mesh topology validity. Masks, partial
coverage and spherical coordinates require a different benchmark definition and
are not silently supported here. Targets can have nonuniform rectangular edges.

## Outputs and manuscript insertion

- `area_diagnostics.pdf` / `.png`: coarse/fine rows; columns for area defect,
  sampled-normalized constant error, and physical-normalized constant error.
  Gray means undefined; an empty cell has area defect -1, not undefined defect.
  Color limits match across depths for each quantity.
- `constant_convergence.pdf` / `.png`: four panels matching your second placeholder.
  Positive ordinates use log axes. Downward triangles in a status strip mark
  exact zeros; crosses mark unavailable diagnostics. Neither is assigned a fake
  logarithmic value. Roundoff-sized nonzero residuals remain nonzero.
- `diagnostics.csv`: every requested scalar, both fields and both normalizations.
  The same diagnostic record is printed as one JSON-style line per run (with
  Python's NaN extension for unavailable values).
- `diagnostics_table.tex`: compact table ready for `\\input`; `--` means unavailable.
- `boundary_cell_areas.csv`: exact overlap and assigned sample area for each source
  contributor to selected cut target cells. Selection is fixed across depths,
  prioritizing large coarse-depth area defects; cell indices are zero-based.
- `exact_overlaps.npz`, `assigned_areas_k*.npz`: sparse target-by-source area matrices.
  Load with `scipy.sparse.load_npz` for further boundary diagnostics.
- `mesh_reference.npz`, `fields_k*.npz`, `metadata.json`: saved inputs, references,
  maps, seed, settings and timing for reproducing plots without rerunning remaps.
- `adjacent_rates.csv`: observed pairwise rates in the scale `2**(-k)`, only when
  both whole-domain errors are finite and positive. Negative/nonmonotone rates are
  retained; these are not a fitted or proven asymptotic convergence order.

Replace the first placeholder with:

```latex
\includegraphics[width=\linewidth]{results/area_diagnostics.pdf}
```

Replace the second with:

```latex
\includegraphics[width=\linewidth]{results/constant_convergence.pdf}
```

Add `\usepackage{graphicx}` to the preamble. Optionally include the diagnostic table:

```latex
\begin{table}[htbp]
\centering
\small
\input{results/diagnostics_table.tex}
\caption{Constant-field and fixed-source diagnostics. W and A denote sampled-
area and physical-area normalization. Dashes indicate unavailable whole-domain
errors or integrals, not zeros.}
\end{table}
```

## Meaning of the diagnostics

The constant and fixed fields use exactly the same geometry at each depth.
The fixed field is assigned once at parent centroids and remains piecewise
constant: it is **not** re-evaluated at subdivision centroids or compared with
the underlying sine/cosine expression on targets.

The overlap reference uses independent triangle/rectangle polygon clipping and
shoelace areas, not a high-depth subdivision approximation. “Exact overlap” here
means geometric intersection computed in floating-point arithmetic, not symbolic
exact arithmetic. Its target-area and parent-area sums are checked before use.

`source_integral = sum(q_source * source_area)`.
Residuals use `B = sum(abs(q_source) * source_area)`:

- physical residual: `(sum(q_target * physical_target_area) - source_integral)/B`;
- sampled residual: `(sum(q_target * W) - source_integral)/B`;
- deposition residual: `(sum(S) - source_integral)/B`.

For W normalization, empty cells remain NaN. Their zero W contributes nothing
to the sampled-area integral; the physical integral and whole-domain field errors
remain unavailable. Conditional sampled-subset errors are printed separately.
For A normalization, empty cells have an explicitly defined zero value, so the
whole-domain physical integral and errors can be computed, while missing coverage
is still reported. This is not an assumption that the true field is zero there.
The A-normalized sampled-area integral is not generally conserved; it is printed
to distinguish the measures, not as a requirement for that normalization.

Area-weighted L1 and L2 errors use physical target areas; Linf is the maximum
absolute error. All cells, including empty ones, enter the area-defect statistics.
No integral is repaired by global rescaling. No unavailable error is zero-filled.

The finite subdivision method can assign a whole child area across a target
boundary. Comparing the saved assigned-area and overlap matrices diagnoses this
misassignment. This bundle does not render individual crossing-child polygons.

## Completing the result paragraph

Use your own run's diagnostics, not the included demonstration as a publication
result. Report:

1. The domain, mesh sizes, seed and tested depth range.
2. For W: constant error on defined cells, first tested complete-coverage depth,
   and the physical residual where that integral is available.
3. For A: physical residual and constant error, keeping missing area visible.
4. Whole-domain fixed-field errors and the coverage-complete depth range used to
   discuss observed convergence. Inspect adjacent rates before quoting an order;
   a short or nonmonotone sequence does not establish an asymptotic rate.
5. A selected boundary-cell comparison of assigned versus exact source overlaps.

Constant preservation of W can coexist with a nonzero physical-area residual;
physical-area conservation of A does not by itself demonstrate constant preservation.
For the unit constant field on full common coverage, A's constant error is exactly
its area defect, apart from floating-point arithmetic. Do not infer these local
properties solely from a globally small residual.

## Validation performed for this delivery

The unit suite passed checks for a known triangle/rectangle overlap, overlap row
and column sums on nonmatching meshes, deposited mass, both appropriate weighted
integrals, centroid count/mean, unavailable errors and invalid connectivity.
A small sweep and the default sweep completed, and both figure layouts were
rendered. The small sweep's saved fields agreed with the original uploaded
repository subdivision helper to the declared comparison tolerances
(relative 1e-12, absolute 1e-13). This validates the helper equivalence in those
cases, not the correctness of the repository's entire public API.

The `results` folder contains the default demonstration output. It is supplied
for checking the workflow, not as evidence for your final manuscript claims.
