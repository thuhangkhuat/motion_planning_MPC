import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import *
from utils import *
from lidar_fixed import *


# ============================================================
# RRT-Connect parameters
# ============================================================
RRT_STEP_LENGTH = 0.3
RRT_GOAL_SAMPLE_RATE = 0.05   # thấp hơn — bi-directional đã tự kéo về goal
RRT_MAX_ITER = 1500

# Bridge test sampling
BRIDGE_SAMPLE_RATE = 0.25     # 25% sample dùng bridge test
BRIDGE_LENGTH = 1.0           # khoảng cách giữa 2 endpoint của bridge
BRIDGE_POINT_RADIUS = None    # None -> dùng ROBOT_RADIUS + safety
BRIDGE_MAX_TRIES = 8          # max thử bridge trước khi fallback uniform


# ============================================================
# Geometric collision (giữ làm debug, OFF mặc định)
# ============================================================
def _segment_intersects_rect(p1, p2, rect, inflate):
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    xmin, ymin = x - inflate, y - inflate
    xmax, ymax = x + w + inflate, y + h + inflate
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    p = [-dx, dx, -dy, dy]
    q = [p1[0] - xmin, xmax - p1[0], p1[1] - ymin, ymax - p1[1]]
    u1, u2 = 0.0, 1.0
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return False
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > u2: return False
                if t > u1: u1 = t
            else:
                if t < u1: return False
                if t < u2: u2 = t
    return u1 <= u2


def is_collision_geom(node_a, node_b, robot_radius, rectangles=None):
    if rectangles is None:
        rectangles = RECTANGLE_OBSTACLES
    p1, p2 = (node_a.x, node_a.y), (node_b.x, node_b.y)
    for rect in rectangles:
        if _segment_intersects_rect(p1, p2, rect, inflate=robot_radius):
            return True
    return False


# ============================================================
# RRT-Connect
# ============================================================
# Trạng thái extend
TRAPPED = 0       # gặp collision ngay
ADVANCED = 1      # extend được 1 bước, chưa tới target
REACHED = 2       # tới hoặc gần như tới target


class RRTConnect:
    """
    RRT-Connect: 2 cây (start_tree, goal_tree) mọc về phía nhau.
    Mỗi vòng:
      1. Sample random point
      2. Cây active 'extend' về phía sample
      3. Cây còn lại 'connect' (greedy extend) về phía node mới vừa thêm
      4. Nếu connect REACHED -> merge 2 cây thành path
      5. Đổi vai active <-> passive
    """

    def __init__(self,
                 step_length: float = RRT_STEP_LENGTH,
                 goal_sample_rate: float = RRT_GOAL_SAMPLE_RATE):
        self.step_length = step_length
        self.goal_sample_rate = goal_sample_rate

        self.s_start = None
        self.s_goal = None
        self.tree_a = None    # cây mọc từ start
        self.tree_b = None    # cây mọc từ goal

        # debug: lưu lại các bridge midpoint thành công để visualize
        self.bridge_samples = []
        self._last_bridge = None

    # ----- Setup -----
    def initialize(self, x_start: tuple, x_goal: tuple):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.tree_a = [self.s_start]
        self.tree_b = [self.s_goal]

    # ----- Collision -----
    @staticmethod
    def _collides(node_a, node_b, obstacle_points, robot_radius,
                  ignore_start=False, use_geom=False):
        if is_collision(node_a, node_b, obstacle_points, robot_radius,
                        ignore_start=ignore_start):
            return True
        if use_geom and is_collision_geom(node_a, node_b, robot_radius):
            return True
        return False

    # ----- Sampling -----
    @staticmethod
    def _point_in_obstacle(pt, obstacle_points, radius):
        """
        Heuristic: 1 điểm coi là 'collision' nếu có ít nhất 1 lidar hit
        nằm trong bán kính `radius` xung quanh nó.
        Trả về True nếu kẹt trong/sát obstacle.
        """
        if obstacle_points is None or len(obstacle_points) == 0:
            return False
        # vectorized cho nhanh
        d2 = (obstacle_points[:, 0] - pt[0]) ** 2 + \
             (obstacle_points[:, 1] - pt[1]) ** 2
        return bool(np.any(d2 <= radius * radius))

    def _uniform_sample(self):
        return Node((np.random.uniform(XLIM[0] + ROBOT_RADIUS, XLIM[1] - ROBOT_RADIUS),
                     np.random.uniform(YLIM[0] + ROBOT_RADIUS, YLIM[1] - ROBOT_RADIUS)))

    def _bridge_sample(self, obstacle_points, radius):
        """
        Bridge test:
          1. sample p1, kiểm tra p1 có 'in obstacle' không
          2. nếu có -> sample p2 cách p1 một đoạn BRIDGE_LENGTH (hướng ngẫu nhiên)
          3. nếu p2 cũng in obstacle -> midpoint pm
          4. nếu pm FREE -> trả về Node(pm)  (narrow passage candidate)
          5. fail -> trả về None
        """
        for _ in range(BRIDGE_MAX_TRIES):
            p1 = (np.random.uniform(XLIM[0] + ROBOT_RADIUS, XLIM[1] - ROBOT_RADIUS),
                  np.random.uniform(YLIM[0] + ROBOT_RADIUS, YLIM[1] - ROBOT_RADIUS))
            if not self._point_in_obstacle(p1, obstacle_points, radius):
                continue
            # p2: xa p1 BRIDGE_LENGTH theo hướng random
            theta = np.random.uniform(0, 2 * math.pi)
            p2 = (p1[0] + BRIDGE_LENGTH * math.cos(theta),
                  p1[1] + BRIDGE_LENGTH * math.sin(theta))
            # check trong bounds
            if not (XLIM[0] <= p2[0] <= XLIM[1] and YLIM[0] <= p2[1] <= YLIM[1]):
                continue
            if not self._point_in_obstacle(p2, obstacle_points, radius):
                continue
            # midpoint
            pm = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
            if not self._point_in_obstacle(pm, obstacle_points, radius):
                return Node(pm), p1, p2  # trả thêm endpoints để debug/visualize
        return None, None, None

    def _random_node(self, obstacle_points=None, robot_radius=None):
        r = np.random.random()
        # 5% goal bias
        if r < self.goal_sample_rate:
            return self.s_goal
        # 25% bridge test (nếu có obstacle_points)
        if (obstacle_points is not None and len(obstacle_points) > 0
                and r < self.goal_sample_rate + BRIDGE_SAMPLE_RATE):
            radius = BRIDGE_POINT_RADIUS if BRIDGE_POINT_RADIUS is not None \
                else (robot_radius if robot_radius is not None else ROBOT_RADIUS) + 0.3
            n_mid, p1, p2 = self._bridge_sample(obstacle_points, radius)
            if n_mid is not None:
                # lưu endpoints để vẽ debug (optional)
                self._last_bridge = (p1, p2, (n_mid.x, n_mid.y))
                return n_mid
            # bridge fail -> fallback uniform
        return self._uniform_sample()

    @staticmethod
    def _nearest(tree, node):
        d = [math.hypot(n.x - node.x, n.y - node.y) for n in tree]
        return tree[int(np.argmin(d))]

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _step_toward(self, src, dst):
        d = self._dist(src, dst)
        if d < 1e-9:
            return None
        step = min(self.step_length, d)
        theta = math.atan2(dst.y - src.y, dst.x - src.x)
        new_n = Node((src.x + step * math.cos(theta),
                      src.y + step * math.sin(theta)))
        new_n.parent = src
        return new_n

    # ----- Extend / Connect primitives -----
    def _extend(self, tree, target, obstacle_points, robot_radius):
        """Extend tree 1 bước về phía target. Trả về (status, new_node)."""
        near = self._nearest(tree, target)
        new_n = self._step_toward(near, target)
        if new_n is None:
            return REACHED, near

        ignore_start = (near is self.s_start)  # giữ logic ignore_start cũ
        if self._collides(near, new_n, obstacle_points, robot_radius,
                          ignore_start=ignore_start):
            return TRAPPED, None

        tree.append(new_n)
        # Đã tới target nếu khoảng cách rất nhỏ
        if self._dist(new_n, target) < self.step_length * 0.5:
            return REACHED, new_n
        return ADVANCED, new_n

    def _connect(self, tree, target, obstacle_points, robot_radius):
        """Greedy extend tree về target cho tới khi REACHED hoặc TRAPPED."""
        status = ADVANCED
        last_new = None
        while status == ADVANCED:
            status, new_n = self._extend(tree, target, obstacle_points, robot_radius)
            if new_n is not None:
                last_new = new_n
        return status, last_new

    # ----- Build path từ 2 cây gặp nhau -----
    @staticmethod
    def _trace_to_root(node):
        path = []
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return path

    def _merge_path(self, meet_a, meet_b, swapped):
        """
        meet_a là node mới nhất trong tree_a sau khi extend
        meet_b là node trong tree_b connect tới meet_a
        swapped cho biết tree_a/tree_b có bị đổi vai trong vòng đó không
        """
        path_a = self._trace_to_root(meet_a)   # node -> ... -> start
        path_b = self._trace_to_root(meet_b)   # node -> ... -> goal

        # Quy chuẩn: path đi từ START -> GOAL
        if swapped:
            # meet_a thuộc tree_goal, meet_b thuộc tree_start
            path_start_to_meet = list(reversed(path_b))    # start -> meet
            path_meet_to_goal = path_a                     # meet  -> goal
        else:
            path_start_to_meet = list(reversed(path_a))    # start -> meet
            path_meet_to_goal = path_b                     # meet  -> goal

        # path_meet_to_goal[0] có thể trùng path_start_to_meet[-1] -> bỏ
        if (path_start_to_meet and path_meet_to_goal
                and path_start_to_meet[-1] == path_meet_to_goal[0]):
            path_meet_to_goal = path_meet_to_goal[1:]
        return path_start_to_meet + path_meet_to_goal

    # ----- Main plan -----
    def plan(self, obstacle_points, robot_radius,
             max_iter=RRT_MAX_ITER):
        """
        Trả về (success, path, tree_a, tree_b, iters_used).
        Path theo thứ tự start -> goal.
        """
        if self.tree_a is None or self.tree_b is None:
            return False, [], [], [], 0

        # Thử connect thẳng start - goal trước (lazy shortcut)
        if not self._collides(self.s_start, self.s_goal,
                              obstacle_points, robot_radius,
                              ignore_start=True):
            self.s_goal.parent = self.s_start
            return True, [[self.s_start.x, self.s_start.y],
                          [self.s_goal.x, self.s_goal.y]], \
                   self.tree_a, self.tree_b, 0

        ta, tb = self.tree_a, self.tree_b
        swapped = False

        for it in range(max_iter):
            self._last_bridge = None
            rand_n = self._random_node(obstacle_points, robot_radius)
            if self._last_bridge is not None:
                self.bridge_samples.append(self._last_bridge)

            # Extend cây active 1 bước về phía sample
            status, new_in_a = self._extend(ta, rand_n, obstacle_points, robot_radius)
            if status != TRAPPED:
                # Connect cây kia tới node mới
                conn_status, new_in_b = self._connect(
                    tb, new_in_a, obstacle_points, robot_radius)
                if conn_status == REACHED and new_in_b is not None:
                    path = self._merge_path(new_in_a, new_in_b, swapped)
                    # đảm bảo self.tree_a/_b ánh xạ đúng start/goal trees
                    if swapped:
                        return True, path, tb, ta, it + 1
                    return True, path, ta, tb, it + 1

            # Đổi vai
            ta, tb = tb, ta
            swapped = not swapped

        return False, [], self.tree_a, self.tree_b, max_iter


# ============================================================
# Path smoothing (shortcut) - giữ y như bản RRT trước
# ============================================================
def remove_residual_node(path, obstacle_points, robot_radius):
    if path is None or len(path) < 2:
        return path
    smoothed = [path[0]]
    i, n = 0, len(path)
    while i < n - 1:
        j_best = i + 1
        for j in range(n - 1, i, -1):
            if not RRTConnect._collides(Node(smoothed[-1]), Node(path[j]),
                                        obstacle_points, robot_radius,
                                        ignore_start=(len(smoothed) == 1)):
                j_best = j
                break
        smoothed.append(path[j_best])
        i = j_best
    return smoothed


# ============================================================
# Visualization
# ============================================================
def draw_tree(ax, vertex, color='steelblue', label=None):
    if not vertex:
        return
    for node in vertex:
        if node.parent is not None:
            ax.plot([node.parent.x, node.x],
                    [node.parent.y, node.y],
                    color=color, linewidth=0.6, alpha=0.6, zorder=1)
    xs = [n.x for n in vertex]
    ys = [n.y for n in vertex]
    ax.scatter(xs, ys, s=4, c=color, zorder=2,
               label=f'{label} ({len(vertex)})' if label else None)


def _rect_in_range(rect, center, radius):
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    cx = max(x, min(center[0], x + w))
    cy = max(y, min(center[1], y + h))
    return math.hypot(cx - center[0], cy - center[1]) <= radius


def draw_obstacles(ax, sensor_center=None, sensing_radius=None):
    in_drawn, out_drawn = False, False
    for rect in RECTANGLE_OBSTACLES:
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        if sensor_center is not None and sensing_radius is not None \
                and _rect_in_range(rect, sensor_center, sensing_radius):
            patch = patches.Rectangle(
                (x, y), w, h, linewidth=1.5, edgecolor='darkred',
                facecolor='salmon', alpha=0.7, zorder=0,
                label='Obstacle (in lidar range)' if not in_drawn else None)
            in_drawn = True
        else:
            patch = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor='dimgray',
                facecolor='lightgray', alpha=0.5, zorder=0,
                label='Obstacle (out of range)' if not out_drawn else None)
            out_drawn = True
        ax.add_patch(patch)


def draw_sensing_radius(ax, center, radius):
    circ = patches.Circle(center, radius, fill=False, linestyle='--',
                          edgecolor='royalblue', linewidth=1.2, alpha=0.6,
                          zorder=1, label=f'Sensing radius ({radius:.1f})')
    ax.add_patch(circ)


def draw_lidar_rays(ax, center, obstacle_points, max_rays=200):
    if obstacle_points is None or len(obstacle_points) == 0:
        return
    n = len(obstacle_points)
    step = max(1, n // max_rays)
    first = True
    for i in range(0, n, step):
        pt = obstacle_points[i]
        ax.plot([center[0], pt[0]], [center[1], pt[1]],
                color='orange', linewidth=0.7, alpha=0.45, zorder=2,
                label='Lidar ray (hit)' if first else None)
        first = False


# ============================================================
# Main test
# ============================================================
if __name__ == "__main__":
    from motion_planning_MPC.robot_rrt import Robot
    import time

    # Test pose/goal — bạn có thể đổi để thử nhiều case
    pose = np.array([197.0, 273.0])
    goal = np.array([300.0, 400.0])

    robots = [
        Robot(0, np.concatenate([[-2.5, 0., 5., 0, 0, 0]]), np.zeros(3)),
        Robot(0, np.concatenate([[11., 6., 5., 0, 0, 0]]), np.zeros(3)),
    ]

    # Tăng độ phân giải lidar lên 720 tia (resolution = π/360 ≈ 0.5°)
    lidar = LidarScanner(range_min=0.2, range_max=SENSING_RADIUS,
                         angle_min=-math.pi, angle_max=math.pi,
                         resolution=math.pi / 360)

    st = time.time()
    data = lidar.senseObstacle(np.concatenate([pose, [0]]), robots)
    obstacle_points = lidar.getObstaclePoints(data, pose)
    t_lidar = time.time() - st

    # Planning
    st = time.time()
    planner = RRTConnect()
    planner.initialize(tuple(pose), tuple(goal))
    success, raw_path, tree_a, tree_b, iters = planner.plan(
        obstacle_points, ROBOT_RADIUS, max_iter=RRT_MAX_ITER)
    smoothed_path = (remove_residual_node(raw_path, obstacle_points, ROBOT_RADIUS)
                     if success else [])
    t_plan = time.time() - st

    print(f"Lidar scan : {t_lidar*1000:.2f} ms ({len(obstacle_points)} hits)")
    print(f"Planning   : {t_plan*1000:.2f} ms, iters={iters}")
    print(f"Tree start : {len(tree_a)} nodes")
    print(f"Tree goal  : {len(tree_b)} nodes")
    print(f"Bridges    : {len(planner.bridge_samples)} narrow-passage samples")
    print(f"Success    : {success}")
    if success:
        print(f"Raw path   : {len(raw_path)} pts")
        print(f"Smoothed   : {len(smoothed_path)} pts")

    # ---- Benchmark success rate ----
    print("\n--- Benchmark (20 lần) ---")
    n_trials = 1
    ok_count = 0
    times = []
    iters_list = []
    for _ in range(n_trials):
        p = RRTConnect()
        p.initialize(tuple(pose), tuple(goal))
        t0 = time.time()
        s, _, _, _, it = p.plan(obstacle_points, ROBOT_RADIUS,
                                max_iter=RRT_MAX_ITER)
        times.append((time.time() - t0) * 1000)
        iters_list.append(it)
        if s:
            ok_count += 1
    print(f"Success rate : {ok_count}/{n_trials} ({100*ok_count/n_trials:.0f}%)")
    print(f"Time         : avg={np.mean(times):.1f}ms, "
          f"max={np.max(times):.1f}ms")
    print(f"Iterations   : avg={np.mean(iters_list):.0f}, "
          f"max={np.max(iters_list)}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"RRT-Connect (SCENARIO {SCENARIO}) — "
                 f"{'SUCCESS' if success else 'FAILED'} — "
                 f"start_tree={len(tree_a)}, goal_tree={len(tree_b)}, "
                 f"iters={iters}, {t_plan*1000:.1f} ms")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    # Auto-zoom
    visible_r = min(float(SENSING_RADIUS), 100.0)
    xs_ref = [pose[0] - visible_r, pose[0] + visible_r, goal[0]]
    ys_ref = [pose[1] - visible_r, pose[1] + visible_r, goal[1]]
    mx = (max(xs_ref) - min(xs_ref)) * 0.1
    my = (max(ys_ref) - min(ys_ref)) * 0.1
    ax.set_xlim(min(xs_ref) - 200, max(xs_ref) + 200)
    ax.set_ylim(min(ys_ref) - 200, max(ys_ref) + 200)

    draw_obstacles(ax, sensor_center=pose, sensing_radius=SENSING_RADIUS)
    draw_sensing_radius(ax, pose, SENSING_RADIUS)
    draw_lidar_rays(ax, pose, obstacle_points, max_rays=200)

    if obstacle_points.size > 0:
        ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1],
                   s=10, c='red', edgecolors='darkred', linewidths=0.3,
                   alpha=0.8, label=f'Lidar hits ({len(obstacle_points)})',
                   zorder=4)

    # 2 cây — màu khác nhau
    draw_tree(ax, tree_a, color='steelblue', label='Start tree')
    draw_tree(ax, tree_b, color='purple',    label='Goal tree')

    # Bridge midpoints (narrow passage candidates)
    if planner.bridge_samples:
        mids = np.array([s[2] for s in planner.bridge_samples])
        ax.scatter(mids[:, 0], mids[:, 1], s=35, marker='D',
                   facecolor='yellow', edgecolor='black', linewidths=0.8,
                   alpha=0.9, zorder=5,
                   label=f'Bridge midpoints ({len(mids)})')

    if raw_path:
        px = [p[0] for p in raw_path]
        py = [p[1] for p in raw_path]
        ax.plot(px, py, 'o--', color='orange', markersize=4, linewidth=1.2,
                label='Raw path', zorder=5)

    if smoothed_path:
        px = [p[0] for p in smoothed_path]
        py = [p[1] for p in smoothed_path]
        ax.plot(px, py, '-', color='green', linewidth=2.5,
                label='Smoothed path', zorder=6)

    ax.plot(pose[0], pose[1], 'o', color='lime', markersize=14,
            markeredgecolor='black', label='Start', zorder=7)
    ax.plot(goal[0], goal[1], '*', color='magenta', markersize=18,
            markeredgecolor='black', label='Goal', zorder=7)

    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()