import math
import numpy as np
import matplotlib.pyplot as plt

from config import *


def _ray_rect_intersection(ox, oy, dx, dy, rect):
    """
    Trả về khoảng cách nhỏ nhất t >= 0 mà tia (ox,oy)+t*(dx,dy) chạm rectangle
    hoặc None nếu không chạm. dx,dy là unit vector.
    rect = (x, y, w, h)  (axis-aligned)
    Dùng slab method (Liang-Barsky cho ray).
    """
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    xmin, ymin = x, y
    xmax, ymax = x + w, y + h

    tmin = -math.inf
    tmax = math.inf

    # X slab
    if abs(dx) < 1e-12:
        if ox < xmin or ox > xmax:
            return None
    else:
        t1 = (xmin - ox) / dx
        t2 = (xmax - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None

    # Y slab
    if abs(dy) < 1e-12:
        if oy < ymin or oy > ymax:
            return None
    else:
        t1 = (ymin - oy) / dy
        t2 = (ymax - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None

    # Lấy entry point >= 0
    if tmax < 0:
        return None
    return max(tmin, 0.0)


class LidarScanner:
    def __init__(self, range_min=0.1, range_max=100.0,
                 angle_min=-math.pi/2, angle_max=math.pi/2,
                 resolution=math.pi/90, noise=0.01):
        self.range_min = range_min
        self.range_max = range_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.resolution = resolution
        self.angle_num = int((self.angle_max - self.angle_min) / self.resolution) + 1
        self.range_num = int(10 * range_max)
        self.noise = noise

    def distance(self, pose, obs_pose):
        ex = obs_pose[0] - pose[0]
        ey = obs_pose[1] - pose[1]
        return math.hypot(ex, ey)

    def isCollision(self, pose, obstacles):
        for (cx, cy, cr) in obstacles:
            ex = cx - pose[0]
            ey = cy - pose[1]
            if ex**2 + ey**2 - cr**2 <= 0:
                return True
        return False

    # ------------------------------------------------------------------
    # Pick rectangles trong tầm sensing (1 lần cho mỗi scan, không phải
    # mỗi tia) để tăng tốc.
    # ------------------------------------------------------------------
    def _rects_in_range(self, pose):
        rects = []
        for rect in RECTANGLE_OBSTACLES:
            x, y, w, h = rect[0], rect[1], rect[2], rect[3]
            cx = max(x, min(pose[0], x + w))
            cy = max(y, min(pose[1], y + h))
            if math.hypot(cx - pose[0], cy - pose[1]) <= self.range_max:
                rects.append(rect)
        return rects

    def senseObstacle(self, pose, robots):
        # ---- Circular obstacles (giữ logic cũ) ----
        obstacles = []
        if OBSTACLES.size > 0:
            for i in range(OBSTACLES.shape[0]):
                if np.hypot(pose[0] - OBSTACLES[i, 0],
                            pose[1] - OBSTACLES[i, 1]) < self.range_max + OBSTACLES[i, 2]:
                    obstacles.append(OBSTACLES[i, :])

        # ---- Rectangle obstacles trong tầm ----
        rects = self._rects_in_range(pose)

        # Sense robots (giữ comment như bản gốc)
        # for i in range(len(robots)):
        #     ...

        data = []
        x, y, _ = pose
        angles = np.linspace(self.angle_min + pose[2],
                             self.angle_max + pose[2],
                             self.angle_num, True)

        for angle in angles:
            dx = math.cos(angle)
            dy = math.sin(angle)

            best_dist = self.range_max  # mặc định: không chạm gì

            # 1) Check circular obstacles bằng raymarching (như cũ)
            if obstacles:
                x1 = x + self.range_min * dx
                y1 = y + self.range_min * dy
                x2 = x + self.range_max * dx
                y2 = y + self.range_max * dy
                for i in range(self.range_num + 1):
                    u = i / self.range_num
                    x3 = x2 * u + x1 * (1 - u)
                    y3 = y2 * u + y1 * (1 - u)
                    if self.isCollision([x3, y3], obstacles):
                        d = self.distance(pose, [x3, y3])
                        if d < best_dist:
                            best_dist = d
                        break

            # 2) Check rectangle obstacles bằng ray-AABB analytic (nhanh & chính xác)
            for rect in rects:
                t_hit = _ray_rect_intersection(x, y, dx, dy, rect)
                if t_hit is not None and self.range_min <= t_hit < best_dist:
                    best_dist = t_hit

            if best_dist < self.range_max:
                data.append(best_dist + np.random.rand() * self.noise)
            else:
                data.append(self.range_max)

        angle_arr = np.linspace(self.angle_min, self.angle_max, self.angle_num)
        data = np.array(data)

        idx = np.where(data < self.range_max)
        angle_arr = angle_arr[idx]
        data = data[idx]
        return angle_arr, data

    def getObstaclePoints(self, data, pose):
        scan_angles, scan_ranges = data
        valid_indices = scan_ranges < self.range_max

        detected_angles = scan_angles[valid_indices]
        detected_ranges = scan_ranges[valid_indices]

        obstacle_points = np.vstack([
            pose[0] + detected_ranges * np.cos(detected_angles),
            pose[1] + detected_ranges * np.sin(detected_angles)]).T

        return obstacle_points


# =====================================================================
# Helpers giữ nguyên từ bản gốc (createGridMap, openingMap, getCircle)
# =====================================================================
def getCircle(x, y, r):
    theta = np.linspace(0, 2 * np.pi, 50)
    a = x + r * np.cos(theta)
    b = y + r * np.sin(theta)
    return a, b


def createGridMap(data, pose, goal):
    size_x = int(2 * max(SENSING_RADIUS, abs(goal[0] - pose[0])) / GRID_SIZE) + 1
    size_y = int(2 * max(SENSING_RADIUS, abs(goal[1] - pose[1])) / GRID_SIZE) + 1

    grid_map = np.zeros((size_x, size_y))
    origin_x = size_x // 2
    origin_y = size_y // 2

    ang, dist = data
    for i in range(dist.shape[0]):
        angle = ang[i]
        distance = dist[i]
        if distance > 0:
            x = (distance - ROBOT_RADIUS) * np.cos(angle)
            y = (distance - ROBOT_RADIUS) * np.sin(angle)
            grid_x = int(origin_x + x / GRID_SIZE)
            grid_y = int(origin_y + y / GRID_SIZE)
            if 0 <= grid_x < size_x and 0 <= grid_y < size_y:
                grid_map[grid_x, grid_y] = 1

    start_idx = (origin_x, origin_y)
    goal_idx = (int(origin_x + (goal[0] - pose[0]) / GRID_SIZE),
                int(origin_y + (goal[1] - pose[1]) / GRID_SIZE))
    return grid_map, start_idx, goal_idx


def openingMap(grid_map):
    rows, cols = grid_map.shape
    mask = np.zeros((rows + 2 * EXPAND_SIZE, cols + 2 * EXPAND_SIZE))
    mask[EXPAND_SIZE:EXPAND_SIZE + rows, EXPAND_SIZE:EXPAND_SIZE + cols] = grid_map
    idxs, idys = np.where(grid_map > 0)
    for i in range(idxs.shape[0]):
        mask[idxs[i]:idxs[i] + 2 * EXPAND_SIZE + 1,
             idys[i]:idys[i] + 2 * EXPAND_SIZE + 1] = np.ones(
            (2 * EXPAND_SIZE + 1, 2 * EXPAND_SIZE + 1))
    grid_map = mask[EXPAND_SIZE:EXPAND_SIZE + rows, EXPAND_SIZE:EXPAND_SIZE + cols]
    return grid_map