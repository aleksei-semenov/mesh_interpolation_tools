"""Planar source subdivision and independent triangle/rectangle clipping."""
import numpy as np
from scipy.sparse import coo_matrix


def area(poly):
    p = np.asarray(poly)
    if len(p) < 3:
        return 0.0
    # Translation avoids unnecessary cancellation for small polygons.
    p = p - p[0]
    return abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
               np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2


def centroids(triangle, k):
    """Same midpoint/centroid construction as tri2reg, vectorized per parent."""
    t = np.asarray(triangle, dtype=float)[None, :, :]
    for _ in range(k):
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
        t = np.stack([np.stack(v, axis=1) for v in
                      [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]],
                     axis=1).reshape(-1, 3, 2)
    return t.mean(axis=1)


def mesh(n, jitter=0.17, seed=1729):
    """Nonmatching conforming unit-square mesh; fixed connectivity, valid cells."""
    if n < 1 or not 0 <= jitter < 0.25:
        raise ValueError('n >= 1 and 0 <= jitter < 0.25 required')
    xx, yy = np.meshgrid(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    p = np.column_stack([xx.ravel(), yy.ravel()])
    interior = ((p > 0) & (p < 1)).all(axis=1)
    p[interior] += np.random.default_rng(seed).uniform(-jitter/n, jitter/n,
                                                     (interior.sum(), 2))
    cells = []
    for j in range(n):
        for i in range(n):
            a = j*(n+1)+i
            b, c, d = a+1, a+n+2, a+n+1
            cells.extend([(a,b,c), (a,c,d)] if (i+j)%2 == 0 else
                         [(a,b,d), (b,c,d)])
    return p, np.asarray(cells)


def validate(nodes, triangles, x, y):
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise ValueError('nodes must be finite (N,2)')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu':
        raise ValueError('triangles must be integer (M,3)')
    if triangles.min() < 0 or triangles.max() >= len(nodes):
        raise ValueError('invalid triangle index')
    for e in (x, y):
        if e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all() or not (np.diff(e)>0).all():
            raise ValueError('edges must be strictly increasing finite 1D arrays')
    a = np.array([area(t) for t in nodes[triangles]])
    if (a <= 0).any():
        raise ValueError('degenerate triangle')
    return a


def clip(poly, axis, edge, keep_greater):
    """Sutherland--Hodgman half-plane clipping; no sampling used."""
    if len(poly) == 0:
        return np.empty((0, 2))
    out = []
    prev = poly[-1]
    prev_in = prev[axis] >= edge if keep_greater else prev[axis] <= edge
    for cur in poly:
        cur_in = cur[axis] >= edge if keep_greater else cur[axis] <= edge
        if cur_in != prev_in:
            t = (edge-prev[axis])/(cur[axis]-prev[axis])
            out.append(prev+t*(cur-prev))
        if cur_in:
            out.append(cur)
        prev, prev_in = cur, cur_in
    return np.asarray(out).reshape(-1, 2)


def overlap_matrix(nodes, triangles, x, y):
    """Floating-point polygon overlap areas; independent of subdivision depth."""
    nx, ny = len(x)-1, len(y)-1
    rows, cols, vals = [], [], []
    for i, t in enumerate(nodes[triangles]):
        lo, hi = t.min(axis=0), t.max(axis=0)
        ix0 = max(0, np.searchsorted(x, lo[0], side='right')-1)
        ix1 = min(nx-1, np.searchsorted(x, hi[0], side='left'))
        iy0 = max(0, np.searchsorted(y, lo[1], side='right')-1)
        iy1 = min(ny-1, np.searchsorted(y, hi[1], side='left'))
        for j in range(iy0, iy1+1):
            for h in range(ix0, ix1+1):
                p = t.copy()
                for axis, edge, greater in [(0,x[h],True),(0,x[h+1],False),
                                             (1,y[j],True),(1,y[j+1],False)]:
                    p = clip(p, axis, edge, greater)
                a = area(p)
                if a > 0:
                    rows.append(j*nx+h); cols.append(i); vals.append(a)
    return coo_matrix((vals,(rows,cols)), shape=(nx*ny,len(triangles))).tocsr()


def sample_matrix(nodes, triangles, source_areas, x, y, k, sampler=centroids):
    nx, ny = len(x)-1, len(y)-1
    rows, cols, vals = [], [], []
    for i, t in enumerate(nodes[triangles]):
        p = np.asarray(sampler(t, k))
        if p.shape != (4**k, 2) or not np.isfinite(p).all():
            raise ValueError('sampler must return finite (4**k,2) centroids')
        ix = np.searchsorted(x, p[:,0], side='right')-1
        iy = np.searchsorted(y, p[:,1], side='right')-1
        # Last boundary belongs to the final cell; internal boundaries go right/up.
        ix[p[:,0] == x[-1]] = nx-1
        iy[p[:,1] == y[-1]] = ny-1
        valid = (ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)
        ids, count = np.unique(iy[valid]*nx+ix[valid], return_counts=True)
        rows.extend(ids); cols.extend([i]*len(ids))
        vals.extend(count*source_areas[i]/4**k)
    return coo_matrix((vals,(rows,cols)), shape=(nx*ny,len(triangles))).tocsr()


def errors(q, ref, a):
    if not np.isfinite(q).all():
        return [np.nan]*3
    e = q-ref
    return [np.dot(a,abs(e))/a.sum(), np.sqrt(np.dot(a,e*e)/a.sum()), max(abs(e))]
