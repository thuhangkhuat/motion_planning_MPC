"""
A* planner module — drop-in replacement cho RRT-Connect trong robot pipeline.

Tính năng:
- Local occupancy grid quanh UAV (size cố định, không phụ thuộc goal)
- Goal projection: nếu goal nằm ngoài grid -> chiếu về biên
- A* 8-connected với octile heuristic
- Smoothing: basic và clearance-aware
- Determinism: cùng input -> cùng output

API tương thích với RRTConnect để dễ swap:
    planner = AStarPlanner()
    planner.initialize(start, goal)
    success, path, _, _, iters = planner.plan(obstacle_points, robot_radius,
                                              time_budget_ms=30)
"""

import math
import time
import heapq
import numpy as np

from config import *


# ============================================================
# Parameters
# ============================================================
ASTAR_GRID_RESOLUTION = 0.5
ASTAR_INFLATE_RADIUS = ROBOT_RADIUS + 0.2
ASTAR_LOCAL_HALF_SIZE = SENSING_RADIUS   # nửa cạnh local grid: 120x120m quanh UAV
ASTAR_GOAL_PROJECTION_MARGIN = 0.5   # cách biên grid bao nhiêu khi project


# ============================================================
# Occupancy grid
# ============================================================
class OccupancyGrid:
    def __init__(self, center, half_size, obstacle_points,
                 resolution=ASTAR_GRID_RESOLUTION,
                 inflate_radius=ASTAR_INFLATE_RADIUS):
        """
        Local grid centered tại `center`, mỗi chiều dài 2*half_size.
        center: tuple/array (x, y) world coords
        """
        self.resolution = resolution
        self.inflate_radius = inflate_radius
        self.center = np.asarray(center[:2], dtype=float)
        self.half_size = half_size

        # Grid bounds
        self.origin = self.center - half_size
        self.size_x = int(np.ceil(2 * half_size / resolution)) + 1
        self.size_y = self.size_x   # vuông

        # Build raw occupancy
        self.grid = np.zeros((self.size_y, self.size_x), dtype=np.uint8)
        self._mark_obstacles(obstacle_points)
        self._inflate(inflate_radius)

    def _mark_obstacles(self, obstacle_points):
        if obstacle_points is None or len(obstacle_points) == 0:
            return
        # Chỉ giữ points trong bounds
        in_bounds = (
            (obstacle_points[:, 0] >= self.origin[0]) &
            (obstacle_points[:, 0] < self.origin[0] + self.size_x * self.resolution) &
            (obstacle_points[:, 1] >= self.origin[1]) &
            (obstacle_points[:, 1] < self.origin[1] + self.size_y * self.resolution)
        )
        pts = obstacle_points[in_bounds]
        for pt in pts:
            ix, iy = self.world_to_grid(pt[0], pt[1])
            if 0 <= ix < self.size_x and 0 <= iy < self.size_y:
                self.grid[iy, ix] = 1

    def _inflate(self, radius):
        r_cells = int(np.ceil(radius / self.resolution))
        if r_cells <= 0:
            return

        # Kernel hình tròn
        kernel_size = 2 * r_cells + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i - r_cells)**2 + (j - r_cells)**2 <= r_cells**2:
                    kernel[i, j] = 1

        occupied_idx = np.argwhere(self.grid > 0)
        inflated = self.grid.copy()
        for (iy, ix) in occupied_idx:
            y0 = max(0, iy - r_cells)
            y1 = min(self.size_y, iy + r_cells + 1)
            x0 = max(0, ix - r_cells)
            x1 = min(self.size_x, ix + r_cells + 1)
            ky0 = y0 - (iy - r_cells)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (ix - r_cells)
            kx1 = kx0 + (x1 - x0)
            inflated[y0:y1, x0:x1] |= kernel[ky0:ky1, kx0:kx1]
        self.grid = inflated

    def world_to_grid(self, wx, wy):
        ix = int((wx - self.origin[0]) / self.resolution)
        iy = int((wy - self.origin[1]) / self.resolution)
        return ix, iy

    def grid_to_world(self, ix, iy):
        wx = self.origin[0] + (ix + 0.5) * self.resolution
        wy = self.origin[1] + (iy + 0.5) * self.resolution
        return wx, wy

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.size_x and 0 <= iy < self.size_y

    def is_free(self, ix, iy):
        if self.in_bounds(ix, iy):
            return self.grid[iy, ix] == 0
        return False

    def is_within_world(self, wx, wy):
        xmax = self.origin[0] + self.size_x * self.resolution
        ymax = self.origin[1] + self.size_y * self.resolution
        return (self.origin[0] <= wx < xmax and
                self.origin[1] <= wy < ymax)


def project_goal_to_grid(start, goal, grid: OccupancyGrid, margin=ASTAR_GOAL_PROJECTION_MARGIN):
    """
    Nếu goal nằm trong bounds của grid -> return goal nguyên.
    Nếu nằm ngoài -> chiếu về điểm trên line(start->goal) cắt biên grid
    (cách biên `margin` để A* có chỗ search).

    Trả về (projected_goal, was_projected).
    """
    start = np.asarray(start[:2], dtype=float)
    goal = np.asarray(goal[:2], dtype=float)

    if grid.is_within_world(goal[0], goal[1]):
        return goal, False

    # Goal ngoài bounds -> chiếu về biên grid (trừ margin)
    xmin = grid.origin[0] + margin
    ymin = grid.origin[1] + margin
    xmax = grid.origin[0] + grid.size_x * grid.resolution - margin
    ymax = grid.origin[1] + grid.size_y * grid.resolution - margin

    direction = goal - start
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-6:
        return start.copy(), True
    direction = direction / d_norm

    # Tìm t lớn nhất sao cho start + t*direction nằm trong [xmin,xmax]x[ymin,ymax]
    # Liang-Barsky kiểu đơn giản
    t_candidates = [d_norm]   # max là chính goal nếu trong bounds (nhưng đã check rồi)
    if abs(direction[0]) > 1e-9:
        t_candidates.append((xmin - start[0]) / direction[0])
        t_candidates.append((xmax - start[0]) / direction[0])
    if abs(direction[1]) > 1e-9:
        t_candidates.append((ymin - start[1]) / direction[1])
        t_candidates.append((ymax - start[1]) / direction[1])

    # Chọn t > 0 nhỏ nhất mà điểm thu được nằm trong bounds (gần biên nhất)
    best_t = None
    for t in t_candidates:
        if t <= 0:
            continue
        pt = start + t * direction
        if xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax:
            if best_t is None or t > best_t:
                best_t = t

    if best_t is None:
        # Fallback: clamp goal về bounds
        clamped = np.array([
            np.clip(goal[0], xmin, xmax),
            np.clip(goal[1], ymin, ymax)
        ])
        return clamped, True

    projected = start + best_t * direction
    return projected, True


# ============================================================
# A* search
# ============================================================
NEIGHBORS_8 = [
    (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
    (-1,  0, 1.0),                          (1,  0, 1.0),
    (-1,  1, math.sqrt(2)), (0,  1, 1.0), (1,  1, math.sqrt(2)),
]


def _octile(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def _astar(grid: OccupancyGrid, start_world, goal_world, time_budget_ms=None):
    """
    A* search.
    Returns: (success, path_world_list, expanded_set, iters)
    """
    t_start = time.time()

    sx, sy = grid.world_to_grid(start_world[0], start_world[1])
    gx, gy = grid.world_to_grid(goal_world[0], goal_world[1])

    if not grid.in_bounds(sx, sy) or not grid.in_bounds(gx, gy):
        return False, [], set(), 0

    if not grid.is_free(sx, sy):
        sx, sy = _nearest_free(grid, sx, sy)
        if sx is None:
            return False, [], set(), 0
    if not grid.is_free(gx, gy):
        gx, gy = _nearest_free(grid, gx, gy)
        if gx is None:
            return False, [], set(), 0

    open_heap = []
    counter = 0
    g_score = {(sx, sy): 0.0}
    heapq.heappush(open_heap, (_octile(sx, sy, gx, gy), counter, (sx, sy)))
    parent = {}
    closed = set()
    iters = 0

    while open_heap:
        iters += 1
        if time_budget_ms is not None and iters % 100 == 0:
            if (time.time() - t_start) * 1000 > time_budget_ms:
                return False, [], closed, iters

        f_cur, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == (gx, gy):
            path_grid = [current]
            while current in parent:
                current = parent[current]
                path_grid.append(current)
            path_grid.reverse()
            path_world = [list(grid.grid_to_world(ix, iy))
                          for (ix, iy) in path_grid]
            return True, path_world, closed, iters

        cx, cy = current
        cur_g = g_score[current]

        for dx, dy, move_cost in NEIGHBORS_8:
            nx, ny = cx + dx, cy + dy
            if not grid.is_free(nx, ny):
                continue
            # No corner cutting
            if dx != 0 and dy != 0:
                if not grid.is_free(cx + dx, cy) or not grid.is_free(cx, cy + dy):
                    continue
            neighbor = (nx, ny)
            tentative_g = cur_g + move_cost
            if neighbor in closed:
                continue
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + _octile(nx, ny, gx, gy)
                parent[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))

    return False, [], closed, iters


def _nearest_free(grid, ix, iy, max_radius=20):
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                nx, ny = ix + dx, iy + dy
                if grid.is_free(nx, ny):
                    return nx, ny
    return None, None


# ============================================================
# AStarPlanner — API tương thích RRTConnect
# ============================================================
class AStarPlanner:
    """
    Drop-in replacement cho RRTConnect.

    Khác RRT:
      - Plan trên grid không phải sample
      - Deterministic
      - Tự project goal nếu nằm ngoài local grid
      - "tree_a/tree_b" trong plan() trả về là expanded set (cho viz)
    """

    def __init__(self,
                 grid_resolution=ASTAR_GRID_RESOLUTION,
                 inflate_radius=ASTAR_INFLATE_RADIUS,
                 local_half_size=ASTAR_LOCAL_HALF_SIZE):
        self.grid_resolution = grid_resolution
        self.inflate_radius = inflate_radius
        self.local_half_size = local_half_size

        self.start = None
        self.goal = None
        self.grid = None
        self.expanded = set()
        self.was_goal_projected = False
        self.projected_goal = None

    def initialize(self, start, goal):
        self.start = np.asarray(start[:2], dtype=float)
        self.goal = np.asarray(goal[:2], dtype=float)

    def plan(self, obstacle_points, robot_radius,
             max_iter=None, time_budget_ms=None):
        """
        Trả về (success, path, _placeholder1, _placeholder2, iters)
        path: list các [x, y] từ start -> goal (hoặc projected goal)
        """
        if self.start is None or self.goal is None:
            return False, [], set(), set(), 0

        # Build local grid centered tại UAV
        self.grid = OccupancyGrid(
            center=self.start,
            half_size=self.local_half_size,
            obstacle_points=obstacle_points,
            resolution=self.grid_resolution,
            inflate_radius=max(self.inflate_radius, robot_radius))

        # Goal projection nếu ngoài grid
        local_goal, was_projected = project_goal_to_grid(
            self.start, self.goal, self.grid)
        self.was_goal_projected = was_projected
        self.projected_goal = local_goal

        # A* search
        success, path, expanded, iters = _astar(
            self.grid, self.start, local_goal,
            time_budget_ms=time_budget_ms)
        self.expanded = expanded

        # Trả về với cấu trúc tương thích RRTConnect
        return success, path, expanded, set(), iters


# ============================================================
# Smoothing — y hệt rrt_connect.py
# ============================================================
class _Node:
    """Lightweight Node để compat với utils.is_collision."""
    def __init__(self, p):
        self.x = p[0]
        self.y = p[1]
        self.parent = None


def _is_segment_free(p1, p2, obstacle_points, robot_radius, ignore_start=False):
    from utils import is_collision
    return not is_collision(_Node(p1), _Node(p2), obstacle_points,
                            robot_radius, ignore_start=ignore_start)


def _min_clearance_along_segment(p1, p2, obstacle_points):
    if obstacle_points is None or len(obstacle_points) == 0:
        return np.inf
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    seg_vec = p2 - p1
    seg_len_sq = float(np.dot(seg_vec, seg_vec))
    if seg_len_sq < 1e-12:
        d = obstacle_points - p1
        return float(np.min(np.hypot(d[:, 0], d[:, 1])))
    rel = obstacle_points - p1
    t = (rel @ seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = p1 + t[:, None] * seg_vec
    diffs = obstacle_points - closest
    return float(np.min(np.hypot(diffs[:, 0], diffs[:, 1])))


def remove_residual_node(path, obstacle_points, robot_radius):
    """Basic shortcut."""
    if path is None or len(path) < 2:
        return path
    smoothed = [path[0]]
    i, n = 0, len(path)
    while i < n - 1:
        j_best = i + 1
        for j in range(n - 1, i, -1):
            if _is_segment_free(smoothed[-1], path[j], obstacle_points,
                                robot_radius,
                                ignore_start=(len(smoothed) == 1)):
                j_best = j
                break
        smoothed.append(path[j_best])
        i = j_best
    return smoothed


def remove_residual_node_clearance(path, obstacle_points, robot_radius,
                                    required_clearance=3.0,
                                    fallback_to_basic=True):
    """Clearance-aware shortcut."""
    if path is None or len(path) < 2:
        return path
    smoothed = [path[0]]
    i, n = 0, len(path)
    while i < n - 1:
        j_best_clearance = None
        j_best_collision = None
        for j in range(n - 1, i, -1):
            if not _is_segment_free(smoothed[-1], path[j], obstacle_points,
                                    robot_radius,
                                    ignore_start=(len(smoothed) == 1)):
                continue
            clearance = _min_clearance_along_segment(
                smoothed[-1], path[j], obstacle_points)
            if clearance >= required_clearance:
                j_best_clearance = j
                break
            if j_best_collision is None:
                j_best_collision = j
        if j_best_clearance is not None:
            j_best = j_best_clearance
        elif fallback_to_basic and j_best_collision is not None:
            j_best = j_best_collision
        else:
            j_best = i + 1
        smoothed.append(path[j_best])
        i = j_best
    return smoothed