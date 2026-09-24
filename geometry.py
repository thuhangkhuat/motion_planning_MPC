"""
geometry.py — Small geometry helpers shared by config.py, map_gen.py and
target_manual.py (no dependency on config, so config can import it).
"""

import numpy as np


def catmull_rom(points, ds, alpha=0.5):
    """
    Centripetal Catmull-Rom spline (alpha=0.5) passing THROUGH every point.
    Unlike the uniform variant in target_rrt.py, centripetal does not overshoot
    or self-intersect when points are unevenly spaced.
    Returns the sampled curve (M x 2) and the index in M of each input waypoint.
    """
    P = np.asarray(points, float)
    if len(P) < 3:
        return P.copy(), np.arange(len(P))
    ctrl = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])   # reflective padding
    out, knot_idx = [P[0]], [0]
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ctrl[i:i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** alpha
        n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) / ds)))
        for t in np.linspace(t1, t2, n + 1)[1:]:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
        knot_idx.append(len(out) - 1)
    return np.asarray(out), np.asarray(knot_idx)


def densify(points, ds):
    """Resample a polyline so consecutive points are <= ds apart."""
    P = np.asarray(points, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / ds)))
        out.extend(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.asarray(out)
