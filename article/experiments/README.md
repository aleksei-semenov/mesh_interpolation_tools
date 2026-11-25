# Experiments

Organised along the protocol in [Experiment List.md](Experiment%20List.md).
Each `sNN_*` folder matches one section of that list. It holds the section's
README (goal, required outputs, status, open work) and any new code you write
for it.

```
experiments/
├── Experiment List.md            protocol (sections 1–7 + "what to do first")
├── sNN_*/                        one folder per section, for your experiments
├── suites/                       existing self-contained benchmark bundles
│   ├── constant_geometry_benchmarks/   covers §2, §3 (and §1's diagnostic record)
│   └── smooth_robustness_benchmarks/   covers §4, §5, §6, §7 (partially)
├── e01_tri2reg_conservation.py   quick demo run by `python -m article run e01`
└── _template.py                  template for runner-discoverable scripts
```

## Status overview

| § | Folder | Figure | Implemented in | Status |
|---|--------|--------|----------------|--------|
| 1 | [s01_benchmark_setup](s01_benchmark_setup/) | diagnostic record | both suites, separately | partial: no peak memory, no shared record format |
| 2 | [s02_constant_field](s02_constant_field/) | Fig. 1, Fig. 2 | `suites/constant_geometry_benchmarks` | done (demo settings) |
| 3 | [s03_fixed_piecewise_constant](s03_fixed_piecewise_constant/) | Fig. 3 | `suites/constant_geometry_benchmarks` | partial: panels B and C and the error map are missing |
| 4 | [s04_smooth_field](s04_smooth_field/) | Fig. 4 | `suites/smooth_robustness_benchmarks` | done (demo settings) |
| 5 | [s05_discontinuous_disk](s05_discontinuous_disk/) | Fig. 5 | `suites/smooth_robustness_benchmarks` | partial: no contour displacement |
| 6 | [s06_independent_refinement](s06_independent_refinement/) | Fig. 6 | `suites/smooth_robustness_benchmarks` | done at small sizes; list asks 16–128 |
| 7 | [s07_robustness](s07_robustness/) | Fig. 7 | `suites/smooth_robustness_benchmarks` | partial: no rotated source, no partial coverage |
| – | [s08_beyond_planar](s08_beyond_planar/) | – | – | not started (spherical/HEALPix, higher order, spectral, realistic meshes) |

## Important: which code is being tested?

The suites **re-implement** the subdivision/centroid algorithm in their own
`geometry.py` (identical copy in both). They do not call the package:

- `constant_geometry_benchmarks`: re-implementation by default. With
  `--repository-root ../../../..` it loads only `subdivide_triangle_iterative`
  from `src/mesh_interpolation_tools/tri2reg.py`.
- `smooth_robustness_benchmarks`: never calls repository code.

Neither exercises `calculate_interpolation_weights` / `apply_interpolation_weights`,
the automatic choice of `k`, nodal averaging or `tri2arb` / `tri2tri`. Before quoting
numbers in the article as properties of *the package*, either route the suites
through `article.common` (which imports `src/` and refuses any other copy), or add
a check that their output matches the package on the same inputs.

## Conventions for new work in a section folder

- Put code in the section folder. Name a figure script `figN_*.py` after the
  figure in the protocol, and write outputs to `<section>/results/`.
- Import the package via `article.common` (`from article.common import mit, ...`).
  Run from the repository root with `python -m article.experiments.s02_constant_field.fig1_x`,
  (every section folder already has an empty `__init__.py` for this).
- Record the diagnostic record from §1 for every run, and save raw numbers
  (`.npz`/`.csv`) separately from plots so figures can be redrawn without rerunning.
- Treat the `results/` and `defaults/` folders inside `suites/` as demonstration
  output, not manuscript results.
