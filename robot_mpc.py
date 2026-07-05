"""
BASELINE 2: MPC thuần (pure MPC)
================================
Cùng interface với robot_jps1.Robot => main.py / plot_scenario.py dùng lại.

Khác biệt so với phương pháp đề xuất (JPS + Corridor + MPC + CBF):
    - KHÔNG có global planner (JPS)      -> tracking THẲNG về target
    - KHÔNG có safe corridor (pydecomp)  -> tránh vật cản bằng:
          + hard constraint trên N_OBS điểm LiDAR gần nhất
          + soft margin cost (one-sided quadratic)
    - KHÔNG có CBF visibility            -> không đảm bảo giữ target trong FOV
    - Giữ hard constraint inter-UAV (giống đề xuất, để so sánh công bằng)

Đặc trưng để phân tích trong thesis:
    - Chỉ "thấy" vật cản trong SENSING_RADIUS, không có đường đi toàn cục
      -> dễ kẹt sau obstacle lớn (MPC horizon ngắn = local optimizer)
    - Không có corridor -> có thể infeasible / vi phạm margin khi hẹp
"""

import numpy as np
import casadi as ca

from lidar import LidarScanner
from config import *
from formation import get_formation_goal

# Tham số riêng của baseline
N_OBS_POINTS   = 10                       # số điểm LiDAR gần nhất đưa vào MPC
OBS_HARD_MARGIN = ROBOT_RADIUS + 0.05    # hard: d >= margin
OBS_SOFT_MARGIN = ROBOT_RADIUS + 0.4     # soft: phạt khi d < margin
W_obs_soft     = 10.0
W_track_pure   = 1.0


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

        self.STANDOFF = 0.5 * VIEWING_RADIUS   # đứng cách target 1 khoảng

        # ─── Storage (cùng format) ───
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

        # Shift prediction (warm start cho cycle sau)
        self.states_prediction[:-1, :] = self.states_prediction[1:, :]
        self.controls_prediction[:-1, :] = self.controls_prediction[1:, :]

    # ------------------------------------------------------------
    def computeControlSignal(self, robots):
        scan_data = self.lidar.senseObstacle(
            np.concatenate([self.state[:2], [0]]), robots)
        obstacle_points = self.lidar.getObstaclePoints(
            scan_data, np.concatenate([self.state[:2], [0]]))

        pos = self.state[:2]
        tgt = self.goal[:2]

        # ─── Formation goal DÙNG CHUNG (fair comparison) ───
        role, goal_xy = get_formation_goal(self, robots)
        self._role = role

        if role == "search":
            # search: standoff quanh target để không đè lên target
            to_uav = pos - tgt
            d = float(np.linalg.norm(to_uav))
            track_pt = tgt + self.STANDOFF * (to_uav / d) if d > 1e-6 else tgt.copy()
        else:
            # leader -> target (target ở tâm FOV) ; satellite -> slot
            track_pt = goal_xy

        # traj_ref chỉ để plot (đường thẳng pos -> track_pt)
        self.traj_ref = np.array([list(pos), list(track_pt)])
        self.list_A, self.list_b = [], []

        # ─── Chọn N điểm obstacle gần nhất ───
        sel_pts = np.zeros((0, 2))
        pts = np.asarray(obstacle_points, dtype=float)
        if pts.size > 0:
            pts = pts.reshape(-1, pts.shape[-1])[:, :2]
            dist = np.linalg.norm(pts - pos[None, :], axis=1)
            idx = np.argsort(dist)[:N_OBS_POINTS]
            sel_pts = pts[idx]

        neighbor_robots = self.getNeighbors(robots)

        # ─── MPC setup ───
        opti = ca.Opti()
        opt_states = opti.variable(HORIZON_LENGTH + 1, self.n_state)
        opt_controls = opti.variable(HORIZON_LENGTH, self.n_control)

        f = lambda x_, u_: ca.horzcat(*[x_[3:], u_])

        opti.subject_to(opt_states[0, :] == np.array([self.state]))
        for i in range(HORIZON_LENGTH):
            x_next = opt_states[i, :] + \
                f(opt_states[i, :], opt_controls[i, :]) * TIMESTEP
            opti.subject_to(opt_states[i + 1, :] == x_next)

        # Hard: tránh UAV khác (giống phương pháp đề xuất)
        for i in range(HORIZON_LENGTH):
            for other in neighbor_robots:
                if other.index == self.index:
                    continue
                other_pos = ca.reshape(
                    ca.DM(other.states_prediction[i, :2]), 1, 2)
                dist_sq = ca.sumsqr(opt_states[i, :2] - other_pos)
                opti.subject_to(dist_sq >= (2 * ROBOT_RADIUS)**2)

        # Hard: tránh N điểm obstacle gần nhất
        for p in sel_pts:
            p_dm = ca.DM(p).T
            for i in range(HORIZON_LENGTH + 1):
                dist_sq = ca.sumsqr(opt_states[i, :2] - p_dm)
                opti.subject_to(dist_sq >= OBS_HARD_MARGIN**2)

        # Bounds
        for i in range(HORIZON_LENGTH):
            opti.subject_to(ca.sumsqr(opt_states[i + 1, 3:]) <= VMAX**2)
            opti.subject_to(ca.sumsqr(opt_controls[i, :]) <= UMAX**2)

        # ─── Cost ───
        obj = 0
        track_dm = ca.DM(track_pt).T
        for k in range(HORIZON_LENGTH + 1):
            obj += W_track_pure * ca.sumsqr(opt_states[k, :2] - track_dm)
        for i in range(HORIZON_LENGTH):
            obj += W_u * ca.sumsqr(opt_controls[i, :])
        # Soft obstacle margin
        for p in sel_pts:
            p_dm = ca.DM(p).T
            for k in range(HORIZON_LENGTH + 1):
                dsq = ca.sumsqr(opt_states[k, :2] - p_dm)
                dd = ca.sqrt(dsq + 1e-6)
                viol = ca.fmax(0, OBS_SOFT_MARGIN - dd)
                obj += W_obs_soft * viol * viol
        # Soft inter-UAV margin (giống costCollisionAvoid của đề xuất)
        for k in range(HORIZON_LENGTH + 1):
            for other in robots:
                if other.index == self.index:
                    continue
                other_pos = ca.reshape(
                    ca.DM(other.states_prediction[k, :2]), 1, 2)
                dd = ca.sqrt(ca.sumsqr(opt_states[k, :2] - other_pos) + 1e-6)
                viol = ca.fmax(0, COLLISION_AVOID_DISTANCE - dd)
                obj += W_collision_avoid * viol * viol

        opti.minimize(obj)

        opts_setting = {'ipopt.max_iter': 10000,
                        'ipopt.print_level': 0,
                        'ipopt.tol': 1e-4,
                        'ipopt.acceptable_tol': 1e-2,
                        'print_time': 0,
                        'ipopt.acceptable_iter': 15}
        opti.solver('ipopt', opts_setting)

        opti.set_initial(opt_states, self.states_prediction)
        opti.set_initial(opt_controls, self.controls_prediction)

        try:
            sol = opti.solve()
            self.controls_prediction = sol.value(opt_controls)
            self.states_prediction = sol.value(opt_states)
            control = self.controls_prediction[0, :]
        except RuntimeError as e:
            print(f"[PureMPC Robot {self.index}] solve failed at "
                  f"t={self.time_stamp:.2f}: {type(e).__name__}")
            control = self.controls_prediction[0, :]

        self.updateState(control, TIMESTEP)

    # ------------------------------------------------------------
    def getNeighbors(self, robots):
        neighbors = []
        current_pos = self.state[:3]
        for other in robots:
            if self.index == other.index:
                continue
            if np.linalg.norm(current_pos - other.state[:3]) < SENSING_NEIGHBOR:
                neighbors.append(other)
        return neighbors