"""
A* planner standalone — cùng style visualization với planner.py (RRT-Connect)
để dễ so sánh head-to-head.

Chạy:
    python planner_astar.py

Pipeline:
1. Build LOCAL occupancy grid quanh UAV (kích thước tự co theo pose-goal)
2. Inflate obstacle bằng dilation kernel theo robot_radius
3. A* 8-connected với octile heuristic
4. Path smoothing: basic và clearance-aware (giống RRT version)
5. Visualize: grid, expanded nodes, path comparison
"""

import math
import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import *
from utils import Node, is_collision
from motion_planning_MPC.lidar import LidarScanner


# ============================================================
# Tham số A*
# ============================================================
ASTAR_GRID_RESOLUTION = 0.5      # mỗi cell = 0.5m (cân bằng chi tiết/tốc độ)
ASTAR_INFLATE_RADIUS = ROBOT_RADIUS + 0.5   # buffer thêm an toàn
ASTAR_LOCAL_PADDING = 20.0       # mở rộng grid thêm padding m quanh start-goal


# ============================================================
# Occupancy grid construction
# ============================================================
class OccupancyGrid:
    """
    Local 2D occupancy grid quanh start-goal.
    - origin: world coordinate của cell (0, 0)
    - resolution: kích thước 1 cell (m)
    - grid: np.array (rows, cols), 0=free, 1=occupied (sau inflate)
    """

    def __init__(self, start, goal, obstacle_points,
                 resolution=ASTAR_GRID_RESOLUTION,
                 inflate_radius=ASTAR_INFLATE_RADIUS,
                 padding=ASTAR_LOCAL_PADDING):
        self.resolution = resolution
        self.inflate_radius = inflate_radius

        # Bounding box cover start, goal, và obstacle_points
        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        if len(obstacle_points) > 0:
            xs += [float(obstacle_points[:, 0].min()),
                   float(obstacle_points[:, 0].max())]
            ys += [float(obstacle_points[:, 1].min()),
                   float(obstacle_points[:, 1].max())]
        xmin, xmax = min(xs) - padding, max(xs) + padding
        ymin, ymax = min(ys) - padding, max(ys) + padding

        # Snap origin về multiple của resolution để dễ debug
        self.origin = np.array([xmin, ymin])
        self.size_x = int(np.ceil((xmax - xmin) / resolution)) + 1
        self.size_y = int(np.ceil((ymax - ymin) / resolution)) + 1

        # Build raw occupancy
        self.grid = np.zeros((self.size_y, self.size_x), dtype=np.uint8)
        self._mark_obstacles(obstacle_points)

        # Inflate
        self._inflate(inflate_radius)

    def _mark_obstacles(self, obstacle_points):
        """Đánh dấu cell có lidar point là occupied."""
        if len(obstacle_points) == 0:
            return
        for pt in obstacle_points:
            ix, iy = self.world_to_grid(pt[0], pt[1])
            if 0 <= ix < self.size_x and 0 <= iy < self.size_y:
                self.grid[iy, ix] = 1

    def _inflate(self, radius):
        """
        Inflate obstacle bằng dilation: với mỗi cell occupied, đánh dấu
        các cell trong bán kính radius là occupied.

        Implementation: circular kernel, scipy-free để khỏi import thêm.
        """
        r_cells = int(np.ceil(radius / self.resolution))
        if r_cells <= 0:
            return

        # Tạo kernel hình tròn
        kernel_size = 2 * r_cells + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        cx, cy = r_cells, r_cells
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i - cy)**2 + (j - cx)**2 <= r_cells**2:
                    kernel[i, j] = 1

        # Convolve: với mỗi cell occupied của grid gốc, OR kernel vào output
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

    # ─── Conversions ───
    def world_to_grid(self, wx, wy):
        ix = int((wx - self.origin[0]) / self.resolution)
        iy = int((wy - self.origin[1]) / self.resolution)
        return ix, iy

    def grid_to_world(self, ix, iy):
        wx = self.origin[0] + (ix + 0.5) * self.resolution
        wy = self.origin[1] + (iy + 0.5) * self.resolution
        return wx, wy

    def is_free(self, ix, iy):
        if 0 <= ix < self.size_x and 0 <= iy < self.size_y:
            return self.grid[iy, ix] == 0
        return False

    def world_is_free(self, wx, wy):
        return self.is_free(*self.world_to_grid(wx, wy))


# ============================================================
# A* algorithm
# ============================================================
# 8-connected neighbors: (dx, dy, cost)
NEIGHBORS_8 = [
    (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
    (-1,  0, 1.0),                          (1,  0, 1.0),
    (-1,  1, math.sqrt(2)), (0,  1, 1.0), (1,  1, math.sqrt(2)),
]


def octile_distance(x1, y1, x2, y2):
    """Heuristic admissible cho 8-connected grid."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def astar_search(grid: OccupancyGrid, start_world, goal_world,
                 time_budget_ms=None):
    """
    A* search trên occupancy grid.

    Trả về:
        success (bool)
        path_world: list các [x, y] (world coords) từ start -> goal
        expanded_set: set các (ix, iy) đã expand (cho visualization)
        iters: số node expand
    """
    t_start = time.time()

    sx, sy = grid.world_to_grid(start_world[0], start_world[1])
    gx, gy = grid.world_to_grid(goal_world[0], goal_world[1])

    # Validate
    if not (0 <= sx < grid.size_x and 0 <= sy < grid.size_y):
        return False, [], set(), 0
    if not (0 <= gx < grid.size_x and 0 <= gy < grid.size_y):
        return False, [], set(), 0

    # Nếu start hoặc goal bị inflate vào occupied -> tìm cell free gần nhất
    if not grid.is_free(sx, sy):
        sx, sy = _nearest_free(grid, sx, sy)
        if sx is None:
            return False, [], set(), 0
    if not grid.is_free(gx, gy):
        gx, gy = _nearest_free(grid, gx, gy)
        if gx is None:
            return False, [], set(), 0

    # Priority queue: (f_score, counter, (ix, iy))
    # counter để break tie ổn định
    open_heap = []
    counter = 0
    g_score = {(sx, sy): 0.0}
    f_start = octile_distance(sx, sy, gx, gy)
    heapq.heappush(open_heap, (f_start, counter, (sx, sy)))
    parent = {}
    closed = set()
    iters = 0

    while open_heap:
        iters += 1

        # Time budget check (mỗi 100 iter để giảm overhead)
        if time_budget_ms is not None and iters % 100 == 0:
            if (time.time() - t_start) * 1000 > time_budget_ms:
                return False, [], closed, iters

        f_cur, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        # Goal check
        if current == (gx, gy):
            # Reconstruct path (grid -> world)
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

        # Expand 8 neighbors
        for dx, dy, move_cost in NEIGHBORS_8:
            nx, ny = cx + dx, cy + dy
            if not grid.is_free(nx, ny):
                continue
            # Cấm "góc cắt" qua obstacle (diagonal move ngang qua occupied corner)
            if dx != 0 and dy != 0:
                if not grid.is_free(cx + dx, cy) or not grid.is_free(cx, cy + dy):
                    continue
            neighbor = (nx, ny)
            tentative_g = cur_g + move_cost
            if neighbor in closed:
                continue
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + octile_distance(nx, ny, gx, gy)
                parent[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))

    return False, [], closed, iters


def _nearest_free(grid, ix, iy, max_radius=20):
    """BFS tìm cell free gần nhất khi start/goal bị inflate vào occupied."""
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
# Smoothing (giống RRT version - import lại để self-contained)
# ============================================================
def path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    arr = np.asarray(path, dtype=float)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


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


def _is_segment_collision_free(p1, p2, obstacle_points, robot_radius,
                                ignore_start=False):
    """Dùng is_collision của utils nhưng wrap để truyền Node."""
    return not is_collision(Node(p1), Node(p2), obstacle_points,
                            robot_radius, ignore_start=ignore_start)


def remove_residual_node_astar(path, obstacle_points, robot_radius):
    """Basic shortcut (no clearance) — dùng is_collision của utils.py."""
    if path is None or len(path) < 2:
        return path
    smoothed = [path[0]]
    i, n = 0, len(path)
    while i < n - 1:
        j_best = i + 1
        for j in range(n - 1, i, -1):
            if _is_segment_collision_free(smoothed[-1], path[j],
                                          obstacle_points, robot_radius,
                                          ignore_start=(len(smoothed) == 1)):
                j_best = j
                break
        smoothed.append(path[j_best])
        i = j_best
    return smoothed


def remove_residual_node_clearance_astar(path, obstacle_points, robot_radius,
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
            cf = _is_segment_collision_free(
                smoothed[-1], path[j], obstacle_points, robot_radius,
                ignore_start=(len(smoothed) == 1))
            if not cf:
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


def min_clearance_full_path(path, obstacle_points):
    if path is None or len(path) < 2 or len(obstacle_points) == 0:
        return float('inf')
    min_c = float('inf')
    for i in range(len(path) - 1):
        c = _min_clearance_along_segment(path[i], path[i + 1], obstacle_points)
        if c < min_c:
            min_c = c
    return min_c


def avg_clearance_full_path(path, obstacle_points, n_samples=10):
    if path is None or len(path) < 2 or len(obstacle_points) == 0:
        return float('inf')
    arr = np.asarray(path, dtype=float)
    all_d = []
    for i in range(len(arr) - 1):
        for k in range(n_samples):
            t = k / max(n_samples - 1, 1)
            pt = arr[i] + t * (arr[i + 1] - arr[i])
            d = obstacle_points - pt
            all_d.append(np.min(np.hypot(d[:, 0], d[:, 1])))
    return float(np.mean(all_d))


# ============================================================
# Plot helpers
# ============================================================
def draw_obstacles(ax):
    for rect in RECTANGLE_OBSTACLES:
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor='black',
            facecolor='gray', alpha=0.6, zorder=0))


def draw_grid_overlay(ax, grid: OccupancyGrid, alpha=0.25):
    """Vẽ inflated obstacles từ grid (xanh nhạt)."""
    occupied = np.argwhere(grid.grid > 0)
    if len(occupied) == 0:
        return
    res = grid.resolution
    for (iy, ix) in occupied:
        wx, wy = grid.grid_to_world(ix, iy)
        ax.add_patch(patches.Rectangle(
            (wx - res / 2, wy - res / 2), res, res,
            facecolor='cornflowerblue', edgecolor='none',
            alpha=alpha, zorder=1))


def draw_expanded_nodes(ax, grid: OccupancyGrid, expanded_set, alpha=0.2):
    """Vẽ các cell A* đã expand (cho biết A* khám phá những đâu)."""
    if not expanded_set:
        return
    pts = [grid.grid_to_world(ix, iy) for (ix, iy) in expanded_set]
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c='yellow', alpha=alpha,
               edgecolors='none', zorder=2,
               label=f'A* expanded ({len(expanded_set)})')


def draw_path_with_clearance_band(ax, path, obstacle_points, threshold,
                                   color, label, lw=2.0):
    if not path or len(path) < 2:
        return
    arr = np.asarray(path, dtype=float)
    for i in range(len(arr) - 1):
        c = _min_clearance_along_segment(arr[i], arr[i + 1], obstacle_points)
        style = '--' if c < threshold else '-'
        ax.plot(arr[i:i+2, 0], arr[i:i+2, 1], style, color=color,
                linewidth=lw, zorder=5,
                label=label if i == 0 else None)
    ax.scatter(arr[:, 0], arr[:, 1], s=20, c=color, edgecolors='black',
               linewidths=0.5, zorder=6)


# ============================================================
# Main: chạy A*, so sánh smoothing variants
# ============================================================
if __name__ == "__main__":
    from robot import Robot

    # ─── Test scenario (giống planner.py để so sánh dễ) ───
    pose = np.array([197.0, 273.0])
    goal = np.array([300.0, 400.0])

    robots = [
        Robot(0, np.concatenate([[-2.5, 0., 5., 0, 0, 0]]), np.zeros(3)),
        Robot(0, np.concatenate([[11., 6., 5., 0, 0, 0]]), np.zeros(3)),
    ]

    lidar = LidarScanner(range_min=0.2, range_max=SENSING_RADIUS,
                         angle_min=-math.pi, angle_max=math.pi,
                         resolution=math.pi / 360)

    print("Scanning lidar...")
    data = lidar.senseObstacle(np.concatenate([pose, [0]]), robots)
    obstacle_points = lidar.getObstaclePoints(data, pose)
    print(f"  → {len(obstacle_points)} obstacle points")

    # ─── Build occupancy grid ───
    print("\nBuilding occupancy grid...")
    t0 = time.time()
    grid = OccupancyGrid(pose, goal, obstacle_points,
                         resolution=ASTAR_GRID_RESOLUTION,
                         inflate_radius=ASTAR_INFLATE_RADIUS,
                         padding=ASTAR_LOCAL_PADDING)
    t_grid = (time.time() - t0) * 1000
    n_occ = int(np.sum(grid.grid > 0))
    n_free = int(grid.size_x * grid.size_y - n_occ)
    print(f"  → Grid {grid.size_x}x{grid.size_y} cells "
          f"(res={grid.resolution}m, inflate={grid.inflate_radius}m)")
    print(f"  → Build time: {t_grid:.1f} ms")
    print(f"  → Occupied : {n_occ} cells")
    print(f"  → Free     : {n_free} cells")

    # ─── A* search ───
    print("\nRunning A*...")
    t0 = time.time()
    success, raw_path, expanded, iters = astar_search(
        grid, pose, goal, time_budget_ms=100)
    t_search = (time.time() - t0) * 1000

    if not success:
        print(f"  ❌ A* FAILED ({iters} nodes expanded)")
        # Vẫn vẽ kết quả debug
    else:
        print(f"  ✅ Success in {iters} expansions, {t_search:.1f} ms")
        print(f"  → Raw path: {len(raw_path)} grid cells, "
              f"length={path_length(raw_path):.2f}m")

    # ─── Smoothing variants (giống RRT version) ───
    if success:
        print("\nSmoothing comparison...")
        variants = {
            'basic': {
                'path': remove_residual_node_astar(
                    raw_path, obstacle_points, ROBOT_RADIUS),
                'label': 'Basic (no clearance)',
                'color': 'red',
                'threshold': 1.0,
            },
            'clearance_1m': {
                'path': remove_residual_node_clearance_astar(
                    raw_path, obstacle_points, ROBOT_RADIUS, 1.0),
                'label': 'Clearance 1m',
                'color': 'orange',
                'threshold': 1.0,
            },
            'clearance_3m': {
                'path': remove_residual_node_clearance_astar(
                    raw_path, obstacle_points, ROBOT_RADIUS, 3.0),
                'label': 'Clearance 3m',
                'color': 'green',
                'threshold': 3.0,
            },
            'clearance_5m': {
                'path': remove_residual_node_clearance_astar(
                    raw_path, obstacle_points, ROBOT_RADIUS, 5.0),
                'label': 'Clearance 5m',
                'color': 'blue',
                'threshold': 5.0,
            },
            'clearance_10m': {
                'path': remove_residual_node_clearance_astar(
                    raw_path, obstacle_points, ROBOT_RADIUS, 10.0),
                'label': 'Clearance 10m',
                'color': 'purple',
                'threshold': 10.0,
            },
        }

        # Bảng metrics
        print("\n" + "=" * 80)
        print(f"{'Variant':<22} {'Waypoints':>10} {'Length (m)':>12} "
              f"{'Min clear':>12} {'Avg clear':>12}")
        print("=" * 80)
        # Raw row
        print(f"{'Raw A* (grid cells)':<22} {len(raw_path):>10} "
              f"{path_length(raw_path):>12.2f} "
              f"{min_clearance_full_path(raw_path, obstacle_points):>12.2f} "
              f"{avg_clearance_full_path(raw_path, obstacle_points):>12.2f}")
        for name, v in variants.items():
            p = v['path']
            length = path_length(p)
            min_c = min_clearance_full_path(p, obstacle_points)
            avg_c = avg_clearance_full_path(p, obstacle_points)
            print(f"{v['label']:<22} {len(p):>10} {length:>12.2f} "
                  f"{min_c:>12.2f} {avg_c:>12.2f}")
        print("=" * 80)
        print(f"\nTotal time: grid {t_grid:.1f}ms + A* {t_search:.1f}ms "
              f"= {t_grid + t_search:.1f}ms")

    # ─── Visualization ───
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    pad = 50
    xmin = min(pose[0], goal[0]) - pad
    xmax = max(pose[0], goal[0]) + pad
    ymin = min(pose[1], goal[1]) - pad
    ymax = max(pose[1], goal[1]) + pad

    # Subplot 0: raw A* + grid + expanded nodes
    ax = axes[0]
    draw_obstacles(ax)
    draw_grid_overlay(ax, grid, alpha=0.25)
    draw_expanded_nodes(ax, grid, expanded, alpha=0.15)
    if len(obstacle_points) > 0:
        ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1],
                   s=4, c='red', alpha=0.5, zorder=3, label='Lidar points')
    if success:
        arr_raw = np.asarray(raw_path)
        ax.plot(arr_raw[:, 0], arr_raw[:, 1], '-', color='black',
                linewidth=1.0, alpha=0.7, zorder=4,
                label=f'Raw A* ({len(raw_path)} cells)')
    ax.plot(pose[0], pose[1], 'o', color='lime', markersize=14,
            markeredgecolor='black', label='Start', zorder=7)
    ax.plot(goal[0], goal[1], '*', color='magenta', markersize=18,
            markeredgecolor='black', label='Goal', zorder=7)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    grid_info = f"Grid {grid.size_x}x{grid.size_y} (res={grid.resolution}m)"
    if success:
        ax.set_title(f"A* Search Process\n{grid_info}\n"
                     f"{iters} expansions, {t_search:.1f}ms")
    else:
        ax.set_title(f"A* FAILED\n{grid_info}\n"
                     f"{iters} expansions before timeout")
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Subplots 1-5: variants
    if success:
        for ax, (name, v) in zip(axes[1:], variants.items()):
            draw_obstacles(ax)
            if len(obstacle_points) > 0:
                ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1],
                           s=4, c='red', alpha=0.4, zorder=2)
            draw_path_with_clearance_band(
                ax, v['path'], obstacle_points,
                threshold=v['threshold'], color=v['color'], label=v['label'])
            ax.plot(pose[0], pose[1], 'o', color='lime', markersize=12,
                    markeredgecolor='black')
            ax.plot(goal[0], goal[1], '*', color='magenta', markersize=16,
                    markeredgecolor='black')
            p = v['path']
            min_c = min_clearance_full_path(p, obstacle_points)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.set_title(f"{v['label']}\n{len(p)} wpts, "
                         f"L={path_length(p):.1f}m, min_c={min_c:.1f}m")
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
    else:
        # Nếu A* fail, các subplot còn lại để trống
        for ax in axes[1:]:
            ax.set_visible(False)

    plt.suptitle("A* Path Planning: Search + Smoothing Comparison",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()