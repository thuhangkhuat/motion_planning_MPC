"""
target_manual.py — Target trajectory that passes EXACTLY through user-chosen points.

Replaces RRT (target_rrt.py): deterministic, no cache needed, fast on large maps.

Waypoint source (first match wins):
    1. File  scenarios/target_scen{N}.json   (created with pick_waypoints.py)
    2. Key   'waypoints' in SCENARIOS[N] of config.py

JSON format:
    {
      "waypoints": [[x, y], [x, y], ...],      # or [x, y, z]
      "speeds":    [v0, v1, ...]                # (optional) speed per segment,
                                                # len = len(waypoints) - 1
    }

Usage in main.py:
    from target_manual import create_target
    target = create_target()          # .state, .trajectory, .update(), .final_destination
"""

import json
import os

import numpy as np

from config import (SCENARIO, TIMESTEP, TAR_MAX_SPEED, TAR_RADIUS, SAFETY_MARGIN,
                    TAR_SMOOTH_ENABLE, TAR_SPLINE_DS, TAR_WAYPOINTS,
                    RECTANGLE_OBSTACLES, OBSTACLES)

TARGET_MODE = os.environ.get("TARGET_MODE", "manual")   # "manual" | "rrt"
WAYPOINT_DIR = "scenarios"


def waypoint_file(scenario=SCENARIO):
    return os.path.join(WAYPOINT_DIR, f"target_scen{scenario}.json")


# ============================================================
# Waypoint I/O
# ============================================================
def load_waypoints(scenario=SCENARIO):
    """Return (waypoints Nx2, speeds (N-1,) or None, source)."""
    path = waypoint_file(scenario)
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        wps = np.asarray(d["waypoints"], dtype=float)[:, :2]
        speeds = d.get("speeds")
        return wps, (np.asarray(speeds, float) if speeds else None), path
    wps = np.asarray([np.asarray(w, float)[:2] for w in TAR_WAYPOINTS])
    return wps, None, "config.py"


def save_waypoints(waypoints, speeds=None, scenario=SCENARIO):
    os.makedirs(WAYPOINT_DIR, exist_ok=True)
    d = {"waypoints": [[round(float(x), 3), round(float(y), 3)]
                       for x, y in np.asarray(waypoints)[:, :2]]}
    if speeds is not None:
        d["speeds"] = [float(v) for v in speeds]
    path = waypoint_file(scenario)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return path


# ============================================================
# Geometry
# ============================================================
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


def clearance(points, rects=None, circles=None):
    """Signed distance (negative = inside) from each point to the nearest obstacle."""
    rects = RECTANGLE_OBSTACLES if rects is None else rects
    circles = OBSTACLES if circles is None else circles
    P = np.asarray(points, float)
    d = np.full(len(P), np.inf)
    for r in rects:
        x, y, w, h = r[:4]
        c = np.array([x + w / 2, y + h / 2])
        q = np.abs(P - c) - np.array([w / 2, h / 2])
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0)
        d = np.minimum(d, outside + inside)
    for c in (circles if len(circles) else []):
        d = np.minimum(d, np.linalg.norm(P - c[:2], axis=1) - c[2])
    return d


def check_path(points, margin, ds=None):
    """
    Return a list of (piece index, worst point, clearance) for every piece whose
    clearance < margin; empty if the path is safe. Vectorised: the whole path is
    densified once.
    """
    ds = ds or max(margin / 2, 0.05)
    P = np.asarray(points, float)
    if len(P) < 2:
        return []
    seg = np.diff(P, axis=0)
    n = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / ds).astype(int))
    piece = np.repeat(np.arange(len(seg)), n)
    frac = np.concatenate([np.arange(1, k + 1) / k for k in n])
    pts = np.vstack([P[:1], P[piece] + seg[piece] * frac[:, None]])
    piece = np.concatenate([[0], piece])
    c = clearance(pts)
    bad = []
    for i in np.unique(piece[c < margin]):
        m = piece == i
        j = np.argmin(np.where(m, c, np.inf))
        bad.append((int(i), pts[j], float(c[j])))
    return bad


# ============================================================
# Target
# ============================================================
class ManualTarget:
    """
    Same interface as SequentialRRTPlanner:
        .state (3,)  .trajectory list[(3,)]  .update()  .final_destination
    """

    def __init__(self, waypoints, speeds=None, z=0.0, smooth=TAR_SMOOTH_ENABLE,
                 ds=TAR_SPLINE_DS, dt=TIMESTEP, margin=TAR_RADIUS + SAFETY_MARGIN,
                 strict=False):
        W = np.asarray(waypoints, float)[:, :2]
        if len(W) < 2:
            raise ValueError("At least 2 waypoints are required.")
        n_seg = len(W) - 1
        speeds = np.full(n_seg, TAR_MAX_SPEED, float) if speeds is None \
            else np.asarray(speeds, float)
        if len(speeds) != n_seg:
            raise ValueError(f"speeds needs {n_seg} values, got {len(speeds)}.")
        if np.any(speeds <= 0):
            raise ValueError("speeds must be > 0.")

        self.waypoints_2d = [tuple(w) for w in W]
        self.waypoints_3d = [np.array([w[0], w[1], z]) for w in W]
        self.speeds = speeds

        # 1) Geometry: polyline, or a spline through the exact points
        if smooth and len(W) >= 3:
            path, knots = catmull_rom(W, ds)
        else:
            path, knots = W.copy(), np.arange(len(W))

        # 2) Collision check (the spline may bulge outside the polyline)
        worst = {}                                   # waypoint segment -> (point, clearance)
        for i, p, c in check_path(path, margin):
            k = self._seg_of(i, knots)
            if k not in worst or c < worst[k][1]:
                worst[k] = (p, c)
        self.violations = sorted(worst.items())
        if self.violations:
            msg = "\n".join(f"  W{k}→W{k+1}: closest at ({p[0]:.1f}, {p[1]:.1f}), "
                            f"clearance {c:.2f} m < {margin:.2f} m"
                            for k, (p, c) in self.violations)
            if strict:
                raise ValueError("Target trajectory collides with obstacles:\n" + msg)
            print("[TARGET][WARNING] Target trajectory too close to / through obstacles:\n" + msg)

        # 3) Time parameterisation: constant speed on each waypoint segment
        seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
        seg_of_piece = np.searchsorted(knots, np.arange(1, len(path)), side="left") - 1
        seg_of_piece = np.clip(seg_of_piece, 0, n_seg - 1)
        t_nodes = np.concatenate([[0.0], np.cumsum(seg_len / speeds[seg_of_piece])])
        t_query = np.arange(0.0, t_nodes[-1], dt)
        xs = np.interp(t_query, t_nodes, path[:, 0])
        ys = np.interp(t_query, t_nodes, path[:, 1])
        traj = np.column_stack([xs, ys, np.full_like(xs, z)])
        traj = np.vstack([traj, [path[-1, 0], path[-1, 1], z]])   # end exactly on the last point

        self.path_2d = path
        self.trajectory = [p for p in traj]
        self.final_destination = self.trajectory[-1].copy()
        self.traj_index = 0
        self.state = self.trajectory[0].copy()
        self.duration = float(t_nodes[-1])
        self.length = float(seg_len.sum())

    @staticmethod
    def _seg_of(piece_idx, knots):
        return int(np.clip(np.searchsorted(knots, piece_idx + 1) - 1, 0, len(knots) - 2))

    def update(self):
        self.traj_index = min(self.traj_index + 1, len(self.trajectory) - 1)
        self.state = self.trajectory[self.traj_index]


# ============================================================
# Factory used by main.py / plot_scenario.py
# ============================================================
def create_target(force_regen=False, verbose=True, strict=False):
    if TARGET_MODE == "rrt":
        from target_cache import load_or_generate_target
        return load_or_generate_target(TAR_WAYPOINTS, force_regen=force_regen,
                                       verbose=verbose)
    wps, speeds, src = load_waypoints()
    z = float(np.asarray(TAR_WAYPOINTS[0])[2]) if len(TAR_WAYPOINTS[0]) > 2 else 0.0
    tgt = ManualTarget(wps, speeds=speeds, z=z, strict=strict)
    if verbose:
        print(f"[TARGET] manual: {len(wps)} waypoints from {src}, "
              f"length {tgt.length:.1f} m, {tgt.duration:.1f} s, "
              f"{len(tgt.trajectory)} steps")
    return tgt