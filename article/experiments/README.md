# Experiments

Organised along the protocol in [Experiment List.md](Experiment%20List.md).
Each `sNN_*` folder matches one section of that list. It holds the section's
README (goal, required outputs, status, open work), its code, and its outputs
in `results/` (git-ignored; regenerate them with the commands below).

## Which code is tested

The method comes **from the package** in `src/` (imported via `article.common`,
which refuses any other copy). The interface is
[s01_benchmark_setup/package_sampling.py](s01_benchmark_setup/package_sampling.py):

- **Constant source values** (§2, §3, §5, §7 and constant reconstruction in §4/§6):
  the weights are those of `mesh_interpolation_tools.tri2reg.calculate_interpolation_weights`.
  That covers the package's subdivision, centroid-to-cell assignment, equal child areas
  and accumulation. The only change is that `compute_required_subdivisions` is
  temporarily replaced by a fixed `k`, so that `k` can be swept. These runs are serial.
- **Linear reconstruction** (§4/§6) needs the child positions. It uses the package's
  `subdivide_triangle_iterative` plus the package's assignment rule, and every run
  checks that these samples give exactly the package's weight matrix.

The *references* are independent benchmark code: exact triangle/rectangle clipping,
Gauss quadrature and adaptive disk integrals.

The old self-contained `suites/` re-implemented the algorithm. They were removed after
the package-based code reproduced all of their saved results to rounding: max relative
difference 3e-11 in E2, and identical coverage fractions.

## Status overview

| § | Folder | Figure | Status |
|---|--------|--------|--------|
| 1 | [s01_benchmark_setup](s01_benchmark_setup/) | diagnostic record | **done**, package-based (peak memory not yet recorded) |
| 2 | [s02_constant_field](s02_constant_field/) | Fig. 1, Fig. 2 | **done**, package-based (demo sizes) |
| 3 | [s03_fixed_piecewise_constant](s03_fixed_piecewise_constant/) | Fig. 3 | partial: data and panel A (= Fig. 2d) package-based; panels B, C and the error map open |
| 4 | [s04_smooth_field](s04_smooth_field/) | Fig. 4 | **done**, package-based (constant and linear reconstruction) |
| 5 | [s05_discontinuous_disk](s05_discontinuous_disk/) | Fig. 5 | **done**, package-based; optional contour displacement open |
| 6 | [s06_independent_refinement](s06_independent_refinement/) | Fig. 6 | **done**, package-based at small sizes; protocol sizes 16–128 open |
| 7 | [s07_robustness](s07_robustness/) | Fig. 7 | partial: 6 of 8 cases package-based; rotated source, partial coverage and the constant-field heatmap open |
| – | [s08_beyond_planar](s08_beyond_planar/) | – | not started |

"Demo sizes" means the defaults below, which take about 95 s in total. They are not results for the manuscript.

## Running everything (from the repository root)

```bash
E=article.experiments
python -m pytest article/experiments                            # 20 tests, ~1 s
python -m $E.s01_benchmark_setup.run_benchmark                  # §1 data (also used by §2, §3)
python -m $E.s02_constant_field.plot_fig1_fig2                  # Fig. 1, 2
python -m $E.s04_smooth_field.run_smooth                        # §4 data
python -m $E.s04_smooth_field.plot_fig4 --fit-k 1 4             # Fig. 4
python -m $E.s06_independent_refinement.run_refinement          # §6 sweeps B, C
python -m $E.s06_independent_refinement.plot_fig6               # Fig. 6 (panel A from §4)
python -m $E.s05_discontinuous_disk.run_disk
python -m $E.s05_discontinuous_disk.plot_fig5                   # Fig. 5
python -m $E.s07_robustness.run_robustness
python -m $E.s07_robustness.plot_fig7                           # Fig. 7
```

Every script has `--help`. Every run writes `metadata.json`, which records the
settings, git commit, `src_dirty` flag and the package file actually used.

Order matters in two places: §2 reads §1's results, and Fig. 6 panel A reads §4's results.

## Performance note

The package builds weights as a Python list with one tuple per child centroid
(4^k per source triangle). Time and memory therefore grow roughly as N_s·4^k. At
protocol size (32×32 lattice = 2048 triangles, k = 6) that is about 8.4 M tuples,
probably above 1 GB. That is a property of the package worth reporting, but start with smaller runs.

## Conventions for new work

- Put code in the section folder: `run_*.py` computes and saves raw data,
  `plot_fig*.py` draws figures from the saved data only. Both write to `<section>/results/`.
- Import shared code relatively, e.g. `from ..s01_benchmark_setup.package_sampling import package_weight_matrix`.
- Tests go in `test_*.py` next to the code; run them with `python -m pytest article/experiments`.
