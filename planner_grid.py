"""
planner_grid.py — Local-window grid planner (replaces pathfind/JPS for large maps).

Why: `pathfind` builds a Python object graph over the WHOLE grid on every
replan (~40 us per cell -> minutes on a 1000 m map). This planner

  * keeps the persistent log-odds grid from planner_jps.py, but rebuilds the
    inflated "blocked" layer only inside the region touched since the last
    rebuild (dirty bounding box) instead of over the whole map;
  * plans only inside a square window of half size LOCAL_PLAN_RADIUS around
    the UAV, with an 8-connected Dijkstra compiled by numba (pure-Python
    fallback if numba is missing);
  * picks the cell to plan to as follows:
      - goal reachable inside the window          -> the goal;
      - goal outside the window                   -> the reachable window-edge
        cell minimising g(c) + ||c - goal|| (cost so far + straight-line rest);
      - otherwise (goal in an obstacle / behind unknown space, or no edge cell
        reachable)                                -> the reachable cell closest
        to the goal (ties: lowest g).

Unknown space (never observed) follows UNKNOWN_POLICY:
    "blocked"    : cannot be entered (original behaviour; fine on small maps
                   where the target is always close)
    "optimistic" : can be entered at UNKNOWN_COST times the normal step cost;
                   required on large maps, where the goal is usually far
                   beyond what the UAV has seen

Same interface as planner_jps.JPSPlanner:
    initialize(start, goal); observe(pos, points);
    plan(...) -> (success, path, set(), set(), n); .gmap (grid)
"""

import heapq

import numpy as np

from planner_jps import PersistentLogOddsGrid, _dilate_disk, _string_pull

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:                                     # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# ============================================================
# Grid with incremental rebuild + vectorised queries
# ============================================================
class IncrementalLogOddsGrid(PersistentLogOddsGrid):
    """PersistentLogOddsGrid whose rebuild() only touches the dirty region."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._full_dirty = True
        self._dirty_box = None          # [iy0, iy1, ix0, ix1) cells touched by observe()

    def set_inflate(self, radius):
        old = getattr(self, "inflate_cells", None)
        super().set_inflate(radius)
        if old is not None and old != self.inflate_cells:
            self._full_dirty = True
            self._dirty = True

    def observe(self, robot_pos, hit_points):
        self._observe_vectorised(robot_pos, hit_points)

    def _observe_vectorised(self, robot_pos, hit_points):
        """Same update as PersistentLogOddsGrid.observe, without the per-ray loop."""
        rx, ry = float(robot_pos[0]), float(robot_pos[1])
        res = self.resolution
        step = res * 0.5
        nb = self.n_bins
        ncell = self.size_x * self.size_y

        bin_range = np.full(nb, np.inf)
        hp = None
        if hit_points is not None and len(hit_points) > 0:
            hp = np.asarray(hit_points, dtype=float)[:, :2]
            d = hp - np.array([rx, ry])
            rng = np.hypot(d[:, 0], d[:, 1])
            ang = np.arctan2(d[:, 1], d[:, 0])
            b = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
            np.minimum.at(bin_range, b, rng)

        ang = -np.pi + (np.arange(nb) + 0.5) * (2 * np.pi / nb)
        ca, sa = np.cos(ang), np.sin(ang)
        finite = np.isfinite(bin_range)
        seen = finite & (bin_range <= self.sensing_radius)
        free_len = np.where(finite, np.minimum(bin_range, self.sensing_radius),
                            self.sensing_radius)
        n = np.maximum((free_len - res) / step, 0).astype(int)

        Lflat = self.L.reshape(-1)
        nmax = int(n.max()) if nb else 0
        if nmax > 0:
            rr = np.arange(nmax) * step
            mask = np.arange(nmax)[None, :] < n[:, None]            # first n[k] samples
            xs = rx + rr[None, :] * ca[:, None]
            ys = ry + rr[None, :] * sa[:, None]
            ix = ((xs - self.ox) / res).astype(np.int64)
            iy = ((ys - self.oy) / res).astype(np.int64)
            m = mask & (ix >= 0) & (ix < self.size_x) & (iy >= 0) & (iy < self.size_y)
            ray = np.broadcast_to(np.arange(nb)[:, None], m.shape)[m]
            lin = iy[m] * self.size_x + ix[m]
            key = np.unique(ray.astype(np.int64) * ncell + lin)      # unique within a ray
            np.add.at(Lflat, key % ncell, self.l_free)

        occ = []
        if seen.any():
            hx = rx + bin_range[seen] * ca[seen]
            hy = ry + bin_range[seen] * sa[seen]
            occ.append(self._lin(hx, hy))
        if hp is not None:
            occ.append(self._lin(hp[:, 0], hp[:, 1]))
        if occ:
            np.add.at(Lflat, np.concatenate(occ), self.l_occ)
        np.clip(self.L, self.l_min, self.l_max, out=self.L)
        self._dirty = True

        r = int(np.ceil(self.sensing_radius / self.resolution)) + 2
        ix, iy = self.world_to_grid(rx, ry)
        box = [max(0, iy - r), min(self.size_y, iy + r + 1),
               max(0, ix - r), min(self.size_x, ix + r + 1)]
        if self._dirty_box is None:
            self._dirty_box = box
        else:
            bb = self._dirty_box
            self._dirty_box = [min(bb[0], box[0]), max(bb[1], box[1]),
                               min(bb[2], box[2]), max(bb[3], box[3])]

    def _classify(self, Ls):
        occ = Ls >= self.occ_logit
        free = Ls <= self.free_logit
        unknown = (~free) & (~occ)
        return occ, unknown

    def rebuild(self):
        r = self.inflate_cells
        if self._full_dirty or self._dirty_box is None:
            occ, unknown = self._classify(self.L)
            blocked = _dilate_disk(occ, r)
            self.unknown = unknown
            if self.treat_unknown_as_blocked:
                blocked |= unknown
            self.blocked = blocked
            self._full_dirty = False
        else:
            y0, y1, x0, x1 = self._dirty_box
            # output region = dirty box grown by r; input region grown by 2r
            oy0, oy1 = max(0, y0 - r), min(self.size_y, y1 + r)
            ox0, ox1 = max(0, x0 - r), min(self.size_x, x1 + r)
            iy0, iy1 = max(0, oy0 - r), min(self.size_y, oy1 + r)
            ix0, ix1 = max(0, ox0 - r), min(self.size_x, ox1 + r)
            occ, unknown = self._classify(self.L[iy0:iy1, ix0:ix1])
            dil = _dilate_disk(occ, r)
            sy, sx = slice(oy0 - iy0, oy1 - iy0), slice(ox0 - ix0, ox1 - ix0)
            blk = dil[sy, sx]
            if self.treat_unknown_as_blocked:
                blk = blk | unknown[sy, sx]
            self.blocked[oy0:oy1, ox0:ox1] = blk
            self.unknown[oy0:oy1, ox0:ox1] = unknown[sy, sx]
        self._dirty_box = None
        self._dirty = False

    def segment_clear(self, wp0, wp1):
        """Vectorised line-of-sight check on the blocked layer."""
        self._ensure()
        p0 = np.asarray(wp0[:2], float)
        p1 = np.asarray(wp1[:2], float)
        dist = float(np.hypot(*(p1 - p0)))
        n = int(dist / (self.resolution * 0.5)) + 1
        ts = np.linspace(0.0, 1.0, max(n, 1))
        xs = p0[0] + ts * (p1[0] - p0[0])
        ys = p0[1] + ts * (p1[1] - p0[1])
        ix = np.floor((xs - self.ox) / self.resolution).astype(np.int64)
        iy = np.floor((ys - self.oy) / self.resolution).astype(np.int64)
        inb = (ix >= 0) & (ix < self.size_x) & (iy >= 0) & (iy < self.size_y)
        if not inb.all():
            return False
        return not self.blocked[iy, ix].any()


# ============================================================
# 8-connected Dijkstra on a window (numba)
# ============================================================
@njit(cache=True)
def _dijkstra8(blocked, step_cost, sy, sx):
    """
    blocked   : (H, W) bool — cells that cannot be entered
    step_cost : (H, W) float — multiplier for entering a cell (1 = normal)
    Returns g (H, W) in cell units (inf = unreachable) and parent (H, W) flat idx.
    Diagonal moves are not allowed to cut a blocked corner.
    """
    H, W = blocked.shape
    g = np.full((H, W), np.inf)
    parent = np.full((H, W), -1, np.int64)
    done = np.zeros((H, W), np.bool_)
    g[sy, sx] = 0.0
    heap = [(0.0, sy * W + sx)]
    dys = (-1, -1, -1, 0, 0, 1, 1, 1)
    dxs = (-1, 0, 1, -1, 1, -1, 0, 1)
    sq2 = np.sqrt(2.0)
    while len(heap) > 0:
        d, idx = heapq.heappop(heap)
        y = idx // W
        x = idx - y * W
        if done[y, x]:
            continue
        done[y, x] = True
        for k in range(8):
            ny = y + dys[k]
            nx = x + dxs[k]
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if blocked[ny, nx] or done[ny, nx]:
                continue
            diag = dys[k] != 0 and dxs[k] != 0
            if diag and (blocked[y, nx] or blocked[ny, x]):
                continue
            nd = d + (sq2 if diag else 1.0) * step_cost[ny, nx]
            if nd < g[ny, nx]:
                g[ny, nx] = nd
                parent[ny, nx] = idx
                heapq.heappush(heap, (nd, ny * W + nx))
    return g, parent


# ============================================================
# Planner
# ============================================================
class LocalGridPlanner:

    _MAP_REGISTRY = {}

    def __init__(self, world_bounds, agent_id=None, grid_resolution=0.5,
                 inflate_radius=0.4, sensing_radius=3.0, local_radius=100.0,
                 unknown_policy="blocked", unknown_cost=1.0,
                 start_snap_radius=10.0, goal_clamp_margin=2.0, **grid_kwargs):
        grid_kwargs.pop("goal_snap_radius", None)          # not needed (argmin rule)
        self.agent_id = agent_id
        self.local_radius = float(local_radius)
        self.unknown_policy = unknown_policy
        self.unknown_cost = float(unknown_cost)
        self.start_snap_radius = float(start_snap_radius)
        self.goal_clamp_margin = float(goal_clamp_margin)
        self.start = None
        self.goal = None
        self.last_window = None                            # for debugging / plots

        key = (agent_id, tuple(np.round(world_bounds, 3)), round(grid_resolution, 4),
               unknown_policy)
        if agent_id is not None and key in LocalGridPlanner._MAP_REGISTRY:
            self.gmap = LocalGridPlanner._MAP_REGISTRY[key]
            self.gmap.set_inflate(inflate_radius)
        else:
            self.gmap = IncrementalLogOddsGrid(
                world_bounds, resolution=grid_resolution, inflate_radius=inflate_radius,
                sensing_radius=sensing_radius,
                treat_unknown_as_blocked=(unknown_policy == "blocked"), **grid_kwargs)
            self.gmap.unknown = np.ones_like(self.gmap.blocked)
            if agent_id is not None:
                LocalGridPlanner._MAP_REGISTRY[key] = self.gmap

    @classmethod
    def reset_map(cls, agent_id=None):
        if agent_id is None:
            cls._MAP_REGISTRY.clear()
        else:
            for k in [k for k in cls._MAP_REGISTRY if k[0] == agent_id]:
                del cls._MAP_REGISTRY[k]

    # ---- API ----
    def initialize(self, start, goal):
        self.start = np.asarray(start[:2], dtype=float)
        self.goal = np.asarray(goal[:2], dtype=float)

    def observe(self, robot_pos, obstacle_points):
        self.gmap.observe(robot_pos, obstacle_points)

    def _dist_to(self, p, ix, iy):
        gm = self.gmap
        return np.hypot(gm.ox + (ix + 0.5) * gm.resolution - p[0],
                        gm.oy + (iy + 0.5) * gm.resolution - p[1])

    def _cells(self, metres):
        return max(1, int(np.ceil(metres / self.gmap.resolution - 1e-9)))

    def plan(self, obstacle_points, robot_radius, max_iter=None, time_budget_ms=None,
             observe=True):
        gm = self.gmap
        if self.start is None or self.goal is None:
            return False, [], set(), set(), 0
        if observe:
            gm.observe(self.start, obstacle_points)
        gm.set_inflate(max(gm.inflate_radius, robot_radius))
        gm._ensure()

        # 1) start cell (snap out of inflated obstacles)
        sx, sy = gm.world_to_grid(self.start[0], self.start[1])
        if not gm.in_bounds(sx, sy):
            return False, [], set(), set(), 0
        if gm.blocked[sy, sx]:
            r = gm.nearest_free(sx, sy, max_radius=self._cells(self.start_snap_radius))
            if r[0] is None:
                return False, [], set(), set(), 0
            sx, sy = r

        # 2) window around the start
        rc = self._cells(self.local_radius)
        y0, y1 = max(0, sy - rc), min(gm.size_y, sy + rc + 1)
        x0, x1 = max(0, sx - rc), min(gm.size_x, sx + rc + 1)
        self.last_window = (x0, y0, x1, y1)
        blocked = gm.blocked[y0:y1, x0:x1]
        step = np.ones(blocked.shape)
        if self.unknown_policy != "blocked" and self.unknown_cost != 1.0:
            step[gm.unknown[y0:y1, x0:x1]] = self.unknown_cost
        g, parent = _dijkstra8(blocked, step, sy - y0, sx - x0)

        # 3) choose the target cell: the goal if reachable, else argmin g + dist
        goal = self.goal
        m = self.goal_clamp_margin
        goal = np.array([np.clip(goal[0], gm.ox + m, gm.ox + gm.size_x * gm.resolution - m),
                         np.clip(goal[1], gm.oy + m, gm.oy + gm.size_y * gm.resolution - m)])
        gx, gy = gm.world_to_grid(goal[0], goal[1])
        wy, wx = gy - y0, gx - x0
        H, W = g.shape
        reach = np.isfinite(g)
        goal_in_window = 0 <= wy < H and 0 <= wx < W
        if goal_in_window and reach[wy, wx]:
            ty, tx = wy, wx
        else:
            ty = tx = None
            if not goal_in_window:
                # exit the window where (cost so far + straight line to goal) is lowest
                edge = np.zeros_like(reach)
                edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
                ys, xs = np.nonzero(reach & edge)
                if len(ys):
                    d = self._dist_to(goal, xs + x0, ys + y0)
                    k = int(np.argmin(g[ys, xs] * gm.resolution + d))
                    ty, tx = ys[k], xs[k]
            if ty is None:
                # goal unreachable (inside an obstacle, behind unknown space, or the
                # window edge cannot be reached): closest reachable approach
                ys, xs = np.nonzero(reach)
                d = self._dist_to(goal, xs + x0, ys + y0)
                k = int(np.lexsort((g[ys, xs], np.round(d / gm.resolution)))[0])
                ty, tx = ys[k], xs[k]

        if (ty, tx) == (sy - y0, sx - x0):
            wx_, wy_ = gm.grid_to_world(sx, sy)
            return True, [[wx_, wy_], [wx_, wy_]], set(), set(), 2

        # 4) backtrack
        cells = []
        idx = ty * W + tx
        while idx >= 0:
            yy, xx = divmod(int(idx), W)
            cells.append((yy, xx))
            idx = parent[yy, xx]
        cells.reverse()
        path = [list(gm.grid_to_world(xx + x0, yy + y0)) for yy, xx in cells]

        # 5) string pulling on the same grid
        smoothed = _string_pull(path, gm)
        if len(smoothed) < 2:
            smoothed = path if len(path) >= 2 else [path[0], path[0]]
        return True, smoothed, set(), set(), len(smoothed)


def warmup():
    """Compile the numba kernel once (otherwise the first replan takes ~1 s)."""
    b = np.zeros((3, 3), np.bool_)
    _dijkstra8(b, np.ones((3, 3)), 1, 1)
