"""
target_manual.py — Target trajectory that passes EXACTLY through user-chosen points.

Replaces RRT (target_rrt.py): deterministic, no cache needed, fast on large maps.

Waypoint source (resolved by config.py, first match wins):
    1. scenarios/<name>.picks.json   (written by pick_waypoints.py)
    2. target.waypoints / target.speeds in scenarios/<name>.yaml

picks.json format:
    {
      "target": {
        "waypoints": [[x, y], [x, y], ...],     # or [x, y, z]
        "speeds":    [v0, v1, ...]               # optional, one per segment
      },
      "starts": [[x, y, z], ...]                 # optional, UAV start positions
    }

Usage in main.py:
    from target_manual import create_target
    target = create_target()          # .state, .trajectory, .update(), .final_destination
"""

import json
import os
import re

import numpy as np

from geometry import catmull_rom, densify  # noqa: F401  (re-exported)
from config import (SCENARIO, TIMESTEP, TAR_MAX_SPEED, TAR_RADIUS, SAFETY_MARGIN,
                    TAR_SMOOTH_ENABLE, TAR_SPLINE_DS, TAR_WAYPOINTS, TAR_SPEEDS,
                    RECTANGLE_OBSTACLES, OBSTACLES, SCENARIO_DEF, picks_path)

TARGET_MODE = os.environ.get("TARGET_MODE", "manual")   # "manual" | "rrt"


# ============================================================
# Waypoint I/O
# ============================================================
def load_waypoints():
    """Return (waypoints Nx2, speeds (N-1,) or None, source file)."""
    wps = np.asarray([np.asarray(w, float)[:2] for w in TAR_WAYPOINTS]).reshape(-1, 2)
    src = SCENARIO_DEF["picks_file"] or SCENARIO_DEF["file"]
    return wps, TAR_SPEEDS, os.path.relpath(src)


def save_picks(waypoints, speeds=None, starts=None, scenario=SCENARIO):
    """Write <scenario>.picks.json (target waypoints / speeds and UAV starts)."""
    d = {"target": {"waypoints": [[round(float(x), 3), round(float(y), 3)]
                                  for x, y in np.asarray(waypoints)[:, :2]]}}
    if speeds is not None:
        d["target"]["speeds"] = [float(v) for v in speeds]
    if starts is not None:
        d["starts"] = [[round(float(v), 3) for v in p] for p in np.asarray(starts)]
    path = picks_path(scenario)
    # one point per line: indent the structure, keep each [x, y(, z)] on one line
    text = json.dumps(d, indent=2)
    text = re.sub(r"\[\s+(-?[\d.e+-]+),\s+(-?[\d.e+-]+)(?:,\s+(-?[\d.e+-]+))?\s+\]",
                  lambda m: "[" + ", ".join(g for g in m.groups() if g is not None) + "]", text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return path


# ============================================================
# Geometry
# ============================================================
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


def check_path(points, margin, ds=None, rects=None, circles=None):
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
    c = clearance(pts, rects, circles)
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
