"""
map_gen.py — Seeded random obstacle maps.

Used by config.py when a scenario file has a `generate:` block:

    generate:
      seed: 7
      density: 0.08            # target fraction of the map covered by obstacles
      rect_size: [20, 60]      # side length range of rectangles (m)
      circle_radius: [10, 25]  # radius range of circles (m)
      circle_fraction: 0.4     # share of obstacles that are circles
      min_gap: 15              # minimum clearance between any two obstacles (m)
      border_margin: 10        # keep obstacles this far from the map border (m)
      keep_clear_radius: 40    # no obstacle within this distance of UAV starts
                               # and target waypoints (m)
      path_clearance: 0        # no obstacle within this distance of the straight
                               # segments between target waypoints (m); 0 = off
      max_tries: 50000

Determinism: obstacles are drawn and accepted using only the seed and the
map parameters. Keep-clear zones (UAV starts, target waypoints) are applied
AFTERWARDS by dropping obstacles that intersect them, so moving a start or a
waypoint only removes / restores the obstacles next to it — the rest of the
map never changes.

    python map_gen.py big1000      # print statistics for scenarios/big1000.yaml
"""

import numpy as np

DEFAULTS = dict(
    seed=0,
    density=0.08,
    rect_size=[20.0, 60.0],
    circle_radius=[10.0, 25.0],
    circle_fraction=0.4,
    min_gap=15.0,
    border_margin=10.0,
    keep_clear_radius=40.0,
    path_clearance=0.0,
    max_tries=50000,
)


# ============================================================
# Signed distances between primitives
#   rect   = (x, y, w, h)   axis-aligned, (x, y) = lower-left corner
#   circle = (cx, cy, r)
# ============================================================
def _rect_rect_gap(a, R):
    """Gap between rect a and every rect in R (N x 4); <= 0 means overlap."""
    if len(R) == 0:
        return np.empty(0)
    dx = np.maximum(R[:, 0] - (a[0] + a[2]), a[0] - (R[:, 0] + R[:, 2]))
    dy = np.maximum(R[:, 1] - (a[1] + a[3]), a[1] - (R[:, 1] + R[:, 3]))
    return np.where((dx > 0) | (dy > 0), np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)),
                    np.maximum(dx, dy))


def _point_rect_dist(p, R):
    """Distance from points p (M x 2) to rects R (N x 4) -> (M x N); negative inside."""
    p = np.atleast_2d(p)
    cx = R[None, :, 0] + R[None, :, 2] / 2
    cy = R[None, :, 1] + R[None, :, 3] / 2
    qx = np.abs(p[:, None, 0] - cx) - R[None, :, 2] / 2
    qy = np.abs(p[:, None, 1] - cy) - R[None, :, 3] / 2
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside


def _circle_rect_gap(c, R):
    if len(R) == 0:
        return np.empty(0)
    return _point_rect_dist(c[:2], R)[0] - c[2]


def _circle_circle_gap(c, C):
    if len(C) == 0:
        return np.empty(0)
    return np.hypot(C[:, 0] - c[0], C[:, 1] - c[1]) - C[:, 2] - c[2]


# ============================================================
# Generator
# ============================================================
def _drop_near(pts, radius, gen_R, gen_C):
    """Remove generated rects / circles closer than `radius` to any point in pts."""
    if len(pts) == 0 or radius <= 0:
        return gen_R, gen_C
    if len(gen_R):
        gen_R = gen_R[np.all(_point_rect_dist(pts, gen_R) > radius, axis=0)]
    if len(gen_C):
        d = np.hypot(pts[:, None, 0] - gen_C[None, :, 0],
                     pts[:, None, 1] - gen_C[None, :, 1]) - gen_C[None, :, 2]
        gen_C = gen_C[np.all(d > radius, axis=0)]
    return gen_R, gen_C


def generate_obstacles(spec, xlim, ylim, keep_clear_points=(), fixed_rects=(),
                       fixed_circles=(), target_path=()):
    """
    Return (rects, circles) as lists of lists.

    spec              : dict, the scenario's `generate:` block (missing keys -> DEFAULTS)
    keep_clear_points : (K x 2) points that must stay free (UAV starts, target waypoints)
    target_path       : (W x 2) target waypoints; with `path_clearance` > 0 the
                        straight segments between them are kept free as well
    fixed_rects/circles: obstacles written explicitly in the scenario; generated
                        obstacles keep `min_gap` from them as well
    """
    p = {**DEFAULTS, **(spec or {})}
    rng = np.random.default_rng(int(p["seed"]))
    x0, x1 = map(float, xlim)
    y0, y1 = map(float, ylim)
    m = float(p["border_margin"])
    gap = float(p["min_gap"])
    area_goal = float(p["density"]) * (x1 - x0) * (y1 - y0)

    R = np.asarray(fixed_rects, float).reshape(-1, 4)
    C = np.asarray(fixed_circles, float).reshape(-1, 3)
    n_fixed_r, n_fixed_c = len(R), len(C)
    area = 0.0
    tries = 0
    while area < area_goal and tries < int(p["max_tries"]):
        tries += 1
        if rng.random() < float(p["circle_fraction"]):
            r = rng.uniform(*p["circle_radius"])
            if x1 - x0 - 2 * (m + r) <= 0 or y1 - y0 - 2 * (m + r) <= 0:
                continue
            c = np.array([rng.uniform(x0 + m + r, x1 - m - r),
                          rng.uniform(y0 + m + r, y1 - m - r), r])
            if np.any(_circle_circle_gap(c, C) < gap) or np.any(_circle_rect_gap(c, R) < gap):
                continue
            C = np.vstack([C, c])
            area += np.pi * r * r
        else:
            w, h = rng.uniform(*p["rect_size"], size=2)
            if x1 - x0 - 2 * m - w <= 0 or y1 - y0 - 2 * m - h <= 0:
                continue
            a = np.array([rng.uniform(x0 + m, x1 - m - w), rng.uniform(y0 + m, y1 - m - h), w, h])
            if np.any(_rect_rect_gap(a, R) < gap):
                continue
            if len(C) and np.any(_point_rect_dist(C[:, :2], a[None, :])[:, 0] - C[:, 2] < gap):
                continue
            R = np.vstack([R, a])
            area += w * h

    # Drop generated (never fixed) obstacles inside keep-clear zones
    gen_R, gen_C = R[n_fixed_r:], C[n_fixed_c:]
    pts = np.asarray(keep_clear_points, float).reshape(-1, 2)
    gen_R, gen_C = _drop_near(pts, float(p["keep_clear_radius"]), gen_R, gen_C)
    W = np.asarray(target_path, float).reshape(-1, 2)
    pc = float(p["path_clearance"])
    if pc > 0 and len(W) >= 2:
        step = max(pc / 2, 0.5)
        samples = [W[0]]
        for a, b in zip(W[:-1], W[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            samples.extend(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
        gen_R, gen_C = _drop_near(np.asarray(samples), pc, gen_R, gen_C)

    stats = dict(tries=tries, n_rects=len(gen_R), n_circles=len(gen_C),
                 density=(np.sum(gen_R[:, 2] * gen_R[:, 3]) + np.sum(np.pi * gen_C[:, 2] ** 2))
                 / ((x1 - x0) * (y1 - y0)) if (len(gen_R) + len(gen_C)) else 0.0,
                 reached_density=area >= area_goal)
    rects = R[:n_fixed_r].tolist() + gen_R.tolist()
    circles = C[:n_fixed_c].tolist() + gen_C.tolist()
    return rects, circles, stats


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) > 1:
        os.environ["SCENARIO"] = sys.argv[1]
    import config as c
    st = c.GENERATION_STATS
    if st is None:
        print(f"Scenario {c.SCENARIO} has no `generate:` block.")
    else:
        print(f"Scenario {c.SCENARIO}: {st['n_rects']} rects + {st['n_circles']} circles "
              f"generated in {st['tries']} tries, density {100 * st['density']:.1f}%"
              + ("" if st["reached_density"] else "  (target density NOT reached: "
                 "raise max_tries or lower min_gap / density)"))
