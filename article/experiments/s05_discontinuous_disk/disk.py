"""Independent disk references: exact-to-tolerance disk/triangle and disk/rectangle areas."""
import numpy as np
from scipy.integrate import quad

CENTER = (0.5, 0.5)
RADIUS = 0.2


def disk_integral_triangle(t, c=CENTER, r=RADIUS, tol=1e-10):
    """Adaptive x integration of the vertical intersection of triangle and disk."""
    lo = max(t[:, 0].min(), c[0] - r)
    hi = min(t[:, 0].max(), c[0] + r)
    if hi <= lo:
        return 0.

    def section(x):
        ys = []
        for a, b in zip(t, np.roll(t, -1, axis=0)):
            if abs(b[0] - a[0]) < 1e-15:
                continue
            if min(a[0], b[0]) - 1e-14 <= x <= max(a[0], b[0]) + 1e-14:
                ys.append(a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))
        if len(ys) < 2:
            return 0.
        dy = np.sqrt(max(0., r * r - (x - c[0]) ** 2))
        return max(0., min(max(ys), c[1] + dy) - max(min(ys), c[1] - dy))

    breaks = sorted(set([lo, hi] + [float(z) for z in t[:, 0] if lo < z < hi]))
    # Explicit circle/edge crossings provide breakpoints for nonsmooth changes.
    for a, b in zip(t, np.roll(t, -1, axis=0)):
        d = b - a
        v = a - np.array(c)
        A = np.dot(d, d)
        B = 2 * np.dot(v, d)
        C = np.dot(v, v) - r * r
        disc = B * B - 4 * A * C
        if disc >= 0:
            for s in ((-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)):
                z = a[0] + s * d[0]
                if 0 < s < 1 and lo < z < hi:
                    breaks.append(float(z))
    breaks = sorted(set(breaks))
    return sum(quad(section, a, b, epsabs=tol, epsrel=tol, limit=100)[0] for a, b in zip(breaks[:-1], breaks[1:]))


def disk_rectangle(x0, x1, y0, y1, c=CENTER, r=RADIUS, tol=1e-10):
    lo = max(x0, c[0] - r)
    hi = min(x1, c[0] + r)
    if hi <= lo:
        return 0.

    def f(x):
        d = np.sqrt(max(0., r * r - (x - c[0]) ** 2))
        return max(0., min(y1, c[1] + d) - max(y0, c[1] - d))

    breaks = [lo, hi]
    for yy in (y0, y1):
        d2 = r * r - (yy - c[1]) ** 2
        if d2 >= 0:
            for xx in (c[0] - np.sqrt(d2), c[0] + np.sqrt(d2)):
                if lo < xx < hi:
                    breaks.append(xx)
    b = sorted(set(breaks))
    return sum(quad(f, a, e, epsabs=tol, epsrel=tol, limit=100)[0] for a, e in zip(b[:-1], b[1:]))
