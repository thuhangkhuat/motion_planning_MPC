"""
target_manual.py — Target trajectory that passes EXACTLY through user-chosen points.

Replaces RRT (target_rrt.py): deterministic, no cache needed, fast on large maps.

Waypoints: target.waypoints / target.speeds in scenarios/<name>.yaml.
pick_waypoints.py writes them (and the UAV starts) back into that file with
save_to_scenario(), which rewrites only those entries and keeps the comments
and layout of the rest of the file.

Usage in main.py:
    from target_manual import create_target
    target = create_target()          # .state, .trajectory, .update(), .final_destination
"""

import os

import numpy as np

from geometry import catmull_rom, densify  # noqa: F401  (re-exported)
from config import (SCENARIO, TIMESTEP, TAR_MAX_SPEED, TAR_RADIUS, SAFETY_MARGIN,
                    TAR_SMOOTH_ENABLE, TAR_SPLINE_DS, TAR_WAYPOINTS, TAR_SPEEDS,
                    RECTANGLE_OBSTACLES, OBSTACLES, SCENARIO_DEF, scenario_path)

TARGET_MODE = os.environ.get("TARGET_MODE", "manual")   # "manual" | "rrt"


# ============================================================
# Waypoint I/O
# ============================================================
def load_waypoints():
    """Return (waypoints Nx2, speeds (N-1,) or None, source file)."""
    wps = np.asarray([np.asarray(w, float)[:2] for w in TAR_WAYPOINTS]).reshape(-1, 2)
    return wps, TAR_SPEEDS, os.path.relpath(SCENARIO_DEF["file"])


def _num(v):
    v = round(float(v), 3)
    return str(int(v)) if v.is_integer() else repr(v)


def _flow(p):
    return "[" + ", ".join(_num(v) for v in p) + "]"


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _is_value(line):
    """A line that holds data (not blank, not a comment)."""
    st = line.strip()
    return bool(st) and not st.startswith("#")


def _find_key(lines, key, lo, hi, indent):
    """Index of the line `<indent spaces>key:` in lines[lo:hi], or None."""
    for i in range(lo, hi):
        ln = lines[i]
        if _indent(ln) == indent and ln.lstrip(" ").startswith(key + ":"):
            return i
    return None


def _value_end(lines, i):
    """End (exclusive) of the value of the key on line i: deeper-indented lines
    and '- ' items at the key's indent. Stops at the first blank/comment line."""
    ind, j = _indent(lines[i]), i + 1
    while j < len(lines) and _is_value(lines[j]) and (
            _indent(lines[j]) > ind or lines[j].lstrip(" ").startswith("- ")):
        j += 1
    return j


def _block_end(lines, i):
    """End (exclusive) of the top-level block that starts on line i: up to the
    next top-level key, with trailing blank / comment lines left outside."""
    j = next((j for j in range(i + 1, len(lines))
              if _is_value(lines[j]) and _indent(lines[j]) == 0
              and not lines[j].startswith("- ")), len(lines))
    while j > i + 1 and not _is_value(lines[j - 1]):
        j -= 1
    return j


def _key_line(line, key):
    """`key:` keeping the line's indent and trailing comment, dropping an inline value."""
    rest = line.split(":", 1)[1]
    k = rest.find("#")
    comment = rest[k:] if k >= 0 else ""
    pad = " " * max(1, len(rest) - len(rest.lstrip(" "))) if comment and not rest[:k].strip() else " "
    return " " * _indent(line) + key + ":" + (pad + comment if comment else "")


def _set_list(lines, i, key, items):
    """Replace the value of the key on line i by a block list of flow items."""
    end = _value_end(lines, i)
    old = [ln for ln in lines[i + 1:end] if ln.lstrip(" ").startswith("- ")]
    ind = _indent(old[0]) if old else _indent(lines[i])
    lines[i:end] = [_key_line(lines[i], key)] + [" " * ind + "- " + _flow(p) for p in items]


def save_to_scenario(waypoints, speeds=None, starts=None, scenario=SCENARIO,
                     obstacles=None):
    """
    Write target waypoints / speeds and UAV starts into scenarios/<name>.yaml.
    Only `starts:`, `target: waypoints:` and `target: speeds:` are rewritten;
    comments and every other line stay as they are. speeds=None removes an
    existing `speeds:` entry (e.g. after the number of waypoints changed).

    obstacles=(rects, circles) freezes the map: both lists are written into
    `obstacles:` and the `generate:` block is commented out, so the map no
    longer depends on the seed or on the keep-clear zones around the points.

    The result is parsed back and compared before the file is replaced.
    """
    import yaml
    path = scenario_path(scenario)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    before = yaml.safe_load(text) or {}
    lines = text.splitlines()
    wps = [[float(x), float(y)] for x, y in np.asarray(waypoints, float)[:, :2]]

    # ── starts (top level) ──
    if starts is not None:
        st = [[float(v) for v in p] for p in starts]
        i = _find_key(lines, "starts", 0, len(lines), 0)
        if i is None:
            lines += ["", "starts:"]
            i = len(lines) - 1
        _set_list(lines, i, "starts", st)

    # ── target: waypoints / speeds ──
    t = _find_key(lines, "target", 0, len(lines), 0)
    if t is None:
        lines += ["", "target:"]
        t = len(lines) - 1
    t_end = next((j for j in range(t + 1, len(lines))
                  if _is_value(lines[j]) and _indent(lines[j]) == 0
                  and not lines[j].startswith("- ")), len(lines))
    sub = next((_indent(lines[j]) for j in range(t + 1, t_end) if _is_value(lines[j])), 2)
    w = _find_key(lines, "waypoints", t + 1, t_end, sub)
    if w is None:
        lines.insert(t + 1, " " * sub + "waypoints:")
        w, t_end = t + 1, t_end + 1
    n_before = len(lines)
    _set_list(lines, w, "waypoints", wps)
    t_end += len(lines) - n_before
    v = _find_key(lines, "speeds", t + 1, t_end, sub)
    if speeds is None:
        if v is not None:
            del lines[v:_value_end(lines, v)]
    else:
        sp = "speeds: [" + ", ".join(_num(x) for x in speeds) + "]"
        if v is not None:
            comment = _key_line(lines[v], "speeds")[_indent(lines[v]) + len("speeds:"):]
            lines[v:_value_end(lines, v)] = [" " * sub + sp + comment]
        else:
            lines.insert(_value_end(lines, w), " " * sub + sp)

    # ── obstacles (freeze the generated map) ──
    if obstacles is not None:
        rects = [[float(v) for v in r] for r in obstacles[0]]
        circles = [[float(v) for v in c] for c in np.asarray(obstacles[1], float).reshape(-1, 3)]
        o = _find_key(lines, "obstacles", 0, len(lines), 0)
        if o is None:
            lines += ["", "obstacles:"]
            o = len(lines) - 1
        for key, items in (("rects", rects), ("circles", circles)):
            o_end = _block_end(lines, o)
            k = _find_key(lines, key, o + 1, o_end, 2)
            if k is None:
                lines.insert(o_end, "  " + key + ":")
                k = o_end
            if items:
                _set_list(lines, k, key, items)
            else:
                lines[k:_value_end(lines, k)] = [_key_line(lines[k], key).replace(
                    key + ":", key + ": []", 1)]
        g = _find_key(lines, "generate", 0, len(lines), 0)
        if g is not None:
            g_end = _block_end(lines, g)
            lines[g:g_end] = (["# generate: frozen into `obstacles:` by pick_waypoints.py "
                               "(key f). To use the seed again, uncomment this block",
                               "# and empty `obstacles:` (or keep only the fixed ones)."]
                              + ["# " + ln if ln.strip() else ln for ln in lines[g:g_end]])

    new_text = "\n".join(lines) + "\n"
    after = yaml.safe_load(new_text) or {}

    # check: the edited entries hold the new values, everything else is unchanged
    def same(a, b):
        return (a is None) == (b is None) and (a is None or np.allclose(
            np.asarray(a, float), np.asarray(b, float), atol=1e-3))
    tb, ta = dict(before.get("target") or {}), dict(after.get("target") or {})
    ok = (same(ta.get("waypoints"), wps)
          and same(ta.get("speeds"), None if speeds is None else list(speeds))
          and (starts is None or same(after.get("starts"), st)))
    if obstacles is not None:
        ob = after.get("obstacles") or {}
        ok &= (same(ob.get("rects") or None, rects or None)
               and same(ob.get("circles") or None, circles or None)
               and "generate" not in after)
    edited = ("starts", "target") + (("obstacles", "generate") if obstacles is not None else ())
    for k in set(before) | set(after):
        if k not in edited:
            ok &= before.get(k) == after.get(k)
    for k in set(tb) | set(ta):
        if k not in ("waypoints", "speeds"):
            ok &= tb.get(k) == ta.get(k)
    if not ok:
        raise RuntimeError(f"Could not update {path} safely (unusual layout around "
                           f"'starts' / 'target' / 'obstacles'); the file was not changed.")

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new_text)
    os.replace(tmp, path)
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
