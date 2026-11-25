"""Error norms and output helpers shared by all sections."""
import csv
import json

import numpy as np

from article.common import git_info
from .package_sampling import PACKAGE_FILE


def errors(q, ref, a):
    """Area-weighted E1, E2 and E_inf; all NaN if any value is undefined."""
    if not np.isfinite(q).all():
        return [np.nan] * 3
    e = q - ref
    return [np.dot(a, abs(e)) / a.sum(), np.sqrt(np.dot(a, e * e) / a.sum()), max(abs(e))]


def metrics(q, ref, a, active=None):
    """``errors`` restricted to ``active`` cells (default: all)."""
    if active is None:
        active = np.ones(len(a), bool)
    if not np.any(active) or not np.isfinite(q[active]).all():
        return (np.nan, np.nan, np.nan)
    return errors(q[active], ref[active], a[active])


def normalize(S, W, A, method):
    """q = S/A (physical, 'A') or q = S/W (sampled, 'W'; NaN where W = 0)."""
    if method == 'A':
        return S / A
    return np.divide(S, W, out=np.full_like(S, np.nan), where=W > 0)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_metadata(path, arguments, **extra):
    """Settings plus provenance: commit, dirty flag of src/, package file used."""
    meta = dict(arguments=arguments, method='mesh_interpolation_tools.tri2reg '
                '(calculate_interpolation_weights at fixed k)',
                package_file=PACKAGE_FILE, numpy_version=np.__version__,
                **git_info(), **extra)
    path.write_text(json.dumps(meta, indent=2, default=str))
