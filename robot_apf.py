"""
BASELINE 1: Artificial Potential Field (APF)
============================================
Cùng interface với Robot của JPS+MPC+CBF (robot_jps1.py):
    - __init__(index, state, goal)
    - computeControlSignal(robots)
    - updateState(control, dt)
    - Lưu path / traj_refs / corridors_plot cùng format
=> main.py và plot_scenario.py dùng lại NGUYÊN VẸN.

Nguyên lý:
    F_att : lực hút về target (có standoff distance để không đè lên target)
    F_rep : lực đẩy từ obstacle points (LiDAR) + từ UAV khác
    v_des = clip(F_total, VMAX)
    u     = K_VEL * (v_des - v),  clip ||u|| <= UMAX

Đặc trưng để phân tích trong thesis:
    - Không có dự đoán (myopic, reactive)
    - Dễ kẹt local minima (obstacle lõm / hẹp)
    - Có thể dao động khi F_att và F_rep cân bằng
"""

import numpy as np

from lidar import LidarScanner
from config import *
from formation import get_formation_goal


class Robot:
    def __init__(self, index, state: np.array, goal: np.array,
                 control=np.zeros(3)):
        self.time_stamp = 0.0
        self.index = index
        self.state = state
        self.control = control
        self.goal = goal

        self.n_state = 6
        self.n_control = 3

        self.lidar = LidarScanner(range_min=0.1, range_max=SENSING_RADIUS,
                                  angle_min=-np.pi, angle_max=np.pi,
                                  resolution=np.pi / 90)

        # ─── APF parameters (tune nếu cần) ───
        self.K_ATT      = 1.2                     # gain lực hút
        self.STANDOFF   = 0.5 * VIEWING_RADIUS    # khoảng cách mong muốn tới target
        self.K_REP_OBS  = 1.5                     # gain đẩy obstacle
        self.D0_OBS     = 1.5                     # tầm ảnh hưởng obstacle (m)
        self.K_REP_UAV  = 1.0                     # gain đẩy UAV khác
        self.D0_UAV     = max(COLLISION_AVOID_DISTANCE, 4 * ROBOT_RADIUS)
        self.K_VEL      = 3.0                     # P-gain velocity -> accel
        self.F_REP_MAX  = 5.0                     # bão hoà tổng lực đẩy

        # ─── Storage (cùng format với robot_jps1) ───
        self.path = []
        self.traj_refs = []
        self.corridors = []
        self.corridors_plot = []
        self.traj_ref = None
        self.list_A, self.list_b = [], []

        self.states_prediction = np.ones(
            (HORIZON_LENGTH + 1, self.n_state)) * self.state
        self.controls_prediction = np.zeros((HORIZON_LENGTH, self.n_control))

    # ------------------------------------------------------------
    def updateState(self, control: np.array, dt: float):
        position = self.state[:3]
        velocity = self.state[3:6]

        next_position = position + velocity * dt
        next_velocity = velocity + (control - D_FRAC * velocity) * dt

        self.state = np.concatenate([next_position, next_velocity])
        self.control = control
        self.time_stamp += dt

        self.path.append(np.concatenate(
            [[self.time_stamp], self.state, self.control]))
        self.traj_refs.append(self.traj_ref)
        self.corridors_plot.append({'A': self.list_A, 'b': self.list_b})

    # ------------------------------------------------------------
    def computeControlSignal(self, robots):
        scan_data = self.lidar.senseObstacle(
            np.concatenate([self.state[:2], [0]]), robots)
        obstacle_points = self.lidar.getObstaclePoints(
            scan_data, np.concatenate([self.state[:2], [0]]))

        pos = self.state[:2]
        vel = self.state[3:5]
        tgt = self.goal[:2]

        # ─── Formation goal DÙNG CHUNG (fair comparison) ───
        # leader/search -> target ; satellite -> slot được gán
        role, goal_xy = get_formation_goal(self, robots)
        self._role = role

        # ─── Attractive force: kéo thẳng về formation goal ───
        to_goal = goal_xy - pos
        d_goal = float(np.linalg.norm(to_goal))
        if role == "search" and d_goal > 1e-6:
            # search: giữ standoff quanh target (không đè lên target)
            F_att = self.K_ATT * (d_goal - self.STANDOFF) * (to_goal / d_goal)
        else:
            # leader/satellite: nhắm đúng điểm goal (target-center / slot)
            F_att = self.K_ATT * to_goal

        # ─── Repulsive: static obstacles (LiDAR points) ───
        F_rep = np.zeros(2)
        pts = np.asarray(obstacle_points, dtype=float)
        if pts.size > 0:
            pts = pts.reshape(-1, pts.shape[-1])[:, :2]
            diff = pos[None, :] - pts                     # (M,2)
            dist = np.linalg.norm(diff, axis=1)
            mask = (dist > 1e-6) & (dist < self.D0_OBS)
            if np.any(mask):
                dm = dist[mask]
                dirs = diff[mask] / dm[:, None]
                mag = self.K_REP_OBS * (1.0 / dm - 1.0 / self.D0_OBS) / (dm**2)
                F_rep += (mag[:, None] * dirs).sum(axis=0)

        # ─── Repulsive: other UAVs ───
        for r in robots:
            if r.index == self.index:
                continue
            diff = pos - r.state[:2]
            d = float(np.linalg.norm(diff))
            if 1e-6 < d < self.D0_UAV:
                mag = self.K_REP_UAV * (1.0 / d - 1.0 / self.D0_UAV) / (d**2)
                F_rep += mag * (diff / d)

        # Bão hoà lực đẩy để tránh control giật
        f_rep_norm = float(np.linalg.norm(F_rep))
        if f_rep_norm > self.F_REP_MAX:
            F_rep *= self.F_REP_MAX / f_rep_norm

        # ─── Force -> desired velocity -> acceleration control ───
        F_total = F_att + F_rep
        v_des = F_total
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > VMAX:
            v_des *= VMAX / v_norm

        u_xy = self.K_VEL * (v_des - vel)
        u_norm = float(np.linalg.norm(u_xy))
        if u_norm > UMAX:
            u_xy *= UMAX / u_norm

        control = np.array([u_xy[0], u_xy[1], 0.0])

        # ─── Logging (giữ format để plot_scenario dùng được) ───
        self.traj_ref = np.array([list(pos), list(goal_xy)])
        self.list_A, self.list_b = [], []

        self._update_prediction(control)
        self.updateState(control, TIMESTEP)

    # ------------------------------------------------------------
    def _update_prediction(self, control):
        """Propagate horizon với control hiện tại (chỉ để tương thích format)."""
        s = self.state.copy()
        self.states_prediction[0] = s
        for k in range(HORIZON_LENGTH):
            u = control if k == 0 else np.zeros(3)
            p = s[:3] + s[3:6] * TIMESTEP
            v = s[3:6] + (u - D_FRAC * s[3:6]) * TIMESTEP
            s = np.concatenate([p, v])
            self.states_prediction[k + 1] = s
            self.controls_prediction[k] = u