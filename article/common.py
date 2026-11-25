"""Shared helpers for article experiments.

Importing this module guarantees that ``mesh_interpolation_tools`` is loaded
from *this repository's* ``src/`` folder (not from some other installed copy),
so every figure is produced by the code you are currently working on.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths + import of the package under study
# ----------------------------------------------------------------------------

ARTICLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARTICLE_DIR.parent
SRC_DIR = REPO_ROOT / "src"
FIGURES_DIR = ARTICLE_DIR / "figures"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # scripts write files; no GUI windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

import mesh_interpolation_tools as mit  # noqa: E402

_pkg_path = Path(mit.__file__).resolve()
if SRC_DIR not in _pkg_path.parents:
    raise RuntimeError(
        f"mesh_interpolation_tools was imported from {_pkg_path}, "
        f"not from this repository ({SRC_DIR}). Remove the other copy or "
        f"run `pip install -e .` from {REPO_ROOT}."
    )

# Number of worker processes passed to the library (None = all CPUs, 1 = serial).
N_PROCESSES = None

# ----------------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------------

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)

# Common article widths in inches (adjust to the journal template).
SINGLE_COL = 3.5
DOUBLE_COL = 7.2

# ----------------------------------------------------------------------------
# Provenance + saving
# ----------------------------------------------------------------------------


def git_info() -> dict:
    """Commit hash and dirty flag of the repository, for reproducibility."""

    def _git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _git("status", "--porcelain", "--", "src")
    match = re.search(r'^version\s*=\s*"([^"]+)"',
                      (REPO_ROOT / "pyproject.toml").read_text(), re.MULTILINE)
    return {
        "package_version": match.group(1) if match else None,
        "commit": _git("rev-parse", "HEAD"),
        "src_dirty": bool(status) if status is not None else None,
    }


def save_figure(fig, experiment: str, name: str, params: dict | None = None,
                formats=("pdf", "png")) -> Path:
    """Save ``fig`` to ``article/figures/<experiment>/<name>.<fmt>`` plus a
    ``<name>.json`` file recording parameters and the code version used."""
    out_dir = FIGURES_DIR / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}")
    meta = {
        "experiment": experiment,
        "figure": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_info(),
        "package_path": str(_pkg_path.parent),
        "params": params or {},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    plt.close(fig)
    print(f"  saved {out_dir / name}.{{{','.join(formats)}}}")
    return out_dir


def save_data(experiment: str, name: str, **arrays) -> Path:
    """Save raw experiment results (``.npz``) next to the figures."""
    out_dir = FIGURES_DIR / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez(path, **arrays)
    return path


# ----------------------------------------------------------------------------
# Test meshes and fields
# ----------------------------------------------------------------------------


def random_delaunay_mesh(n_nodes: int, seed: int = 0, bbox=(0.0, 1.0, 0.0, 1.0)):
    """Random points in ``bbox`` + Delaunay triangulation.

    Returns ``nodes (2, N)`` and ``elem (3, M)`` in the package's convention.
    Corner points are added so the mesh covers the whole box.
    """
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = bbox
    pts = np.column_stack([rng.uniform(x0, x1, n_nodes), rng.uniform(y0, y1, n_nodes)])
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    pts = np.vstack([corners, pts])
    elem = Delaunay(pts).simplices.T
    return pts.T, elem


def structured_tri_mesh(nx: int, ny: int, bbox=(0.0, 1.0, 0.0, 1.0)):
    """Regular grid where every quad is split into two triangles."""
    x0, x1, y0, y1 = bbox
    X, Y = np.meshgrid(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))
    nodes = np.vstack([X.ravel(), Y.ravel()])
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    elem = np.hstack([np.vstack([a, b, d]), np.vstack([a, d, c])])
    return nodes, elem


FIELDS = {
    "wave": lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
    "gaussian": lambda x, y: np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02),
    "step": lambda x, y: (x > 0.5).astype(float),
    "linear": lambda x, y: x + 2 * y,
}


def field_on_nodes(nodes, kind: str = "wave"):
    return FIELDS[kind](nodes[0], nodes[1])


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


def triangle_areas(nodes, elem):
    p = nodes.T[elem.T]  # (M, 3, 2)
    return 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )


def mesh_integral(nodes, elem, field_at_nodes):
    """Integral of the element-mean field over the triangular mesh
    (the quantity the package's interpolation conserves)."""
    elem_vals = field_at_nodes[elem].mean(axis=0)
    return float(np.sum(elem_vals * triangle_areas(nodes, elem)))
