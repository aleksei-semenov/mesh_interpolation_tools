# Article experiments

Experiments and figures for the article about `mesh_interpolation_tools`.
This folder is **not** shipped with the pip package. It lives in the repository
so that every figure is produced by the code currently in `src/`.

## Running

Run these from the repository root:

```bash
python -m article list                     # what is available
python -m article run e01                  # one experiment (prefix match)
python -m article run all                  # regenerate every figure
python -m article.experiments.e01_tri2reg_conservation   # run a module directly
```

Outputs go to `article/figures/<experiment>/`:

| file            | content                                                           |
|-----------------|-------------------------------------------------------------------|
| `<name>.pdf/png`| the figure (PDF for the manuscript, PNG for a quick look)        |
| `<name>.json`   | parameters, git commit, whether `src/` had uncommitted changes, package path |
| `<name>.npz`    | raw numbers (from `save_data`), for tables or re-plotting       |

`article/figures/` is git-ignored because it can be regenerated. Remove that line
from `.gitignore` if you want to version the final figures.

## Guarantee that the repository code is used

`article/common.py` puts `src/` first on `sys.path` and **fails loudly** if
`mesh_interpolation_tools` was imported from anywhere else, for example an older
pip-installed copy. Each figure's `.json` records the commit and a `src_dirty`
flag, so you can always tell which version of the algorithms produced it.

## Layout of `experiments/`

`experiments/` follows the protocol in `experiments/Experiment List.md`. There is one
folder `sNN_*` per section, plus `suites/` with the existing benchmark bundles.
[experiments/README.md](experiments/README.md) has the status table: what is done,
what is open, and which code each result actually uses.

## Adding a quick experiment (runner-discoverable)

1. Copy `experiments/_template.py` to `experiments/eNN_short_name.py`.
2. Write the first docstring line (shown by `list`) and a `main()` function.
3. Always import from `article.common` first. It sets up paths, the figure
   style, `SINGLE_COL`/`DOUBLE_COL` widths and the helpers below.

Helpers in `common.py`:

- meshes: `random_delaunay_mesh`, `structured_tri_mesh`
- fields: `field_on_nodes(nodes, "wave" | "gaussian" | "step" | "linear")`
- diagnostics: `triangle_areas`, `mesh_integral` (the conserved quantity)
- output: `save_figure`, `save_data`
- `N_PROCESSES` controls the library's multiprocessing (`None` = all CPUs, `1` = serial)

Experiments may import any internal function of the package, e.g.
`from mesh_interpolation_tools.tri2reg import calculate_interpolation_weights`,
to look at the weights themselves.

## Notebooks

For interactive exploration, put notebooks in `article/notebooks/` and start them with:

```python
import sys; sys.path.insert(0, "../..")   # repository root
from article.common import *               # same guarantees as the scripts
%matplotlib inline
```

## Notes

- On Windows, the library uses `multiprocessing` (spawn), so experiments must
  run via `main()`, never as top-level module code. The runner handles this.
- HEALPix experiments need `healpy`, which is unavailable on Windows; run those under WSL/Linux.
