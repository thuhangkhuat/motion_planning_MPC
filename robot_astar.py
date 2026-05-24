"""
Robot class dùng A* planner thay RRT-Connect.

Sync với robot.py phiên bản mới nhất:
- Leader-follower architecture với dynamic leader (hysteresis)
- Corridor hard constraint (safety)
- CBF FOV vuông cho leader only
- Centroid tracking cost
- Formation cost (giữ nguyên softmax design)
- Kalman lead pursuit
- Path commit cho A*
- Try-except MPC fallback

Để swap planner trong main.py:
    from robot import Robot           # RRT-Connect
    -> from robot_astar import Robot  # A*
"""

import numpy as np
import casadi as ca
import pydecomp as pdc

from lidar_fixed import LidarScanner
from utils import Node, is_collision

from planner_astar import AStarPlanner, remove_residual_node

from kalman import KalmanTargetTracker
from config import *


# Parameters (đồng bộ với robot.py)
W_centroid = 0.3
LEADER_HYSTERESIS = 5.0
W_leader_slack = 1e3


def _point_to_segment_distance(p, a, b):
    ab = b - a
    seg_len_sq = float(np.dot(ab, ab))
    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / seg_len_sq)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


class Robot:
    def __init__(self, index, state, goal, control=np.zeros(3)):
        self.time_stamp = 0.0
        self.index = index
        self.state = state
        self.control = control
        self.goal = goal

        self.planner_update_counter = 0
        self.PLANNER_UPDATE_RATE = 1
        self.n_state = 6
        self.n_control = 3

        self.lidar = LidarScanner(range_min=0.1, range_max=SENSING_RADIUS,
                                  angle_min=-np.pi, angle_max=np.pi,
                                  resolution=np.pi / 90)

        self.planner = None

        # Fallback
        self.last_valid_path = None
        self.planner_fail_count = 0
        self.MAX_FAIL_BEFORE_STOP = 3
        self.PLANNER_TIME_BUDGET_MS = 30

        # Storage
        self.corridors = []
        self.corridors_plot = []
        self.path = []
        self.traj_refs = []
        self.full_path = None
        self.path_update_counter = 0
        self.cached_path = None

        # Path commit (tune cho map nhỏ)
        self._committed_path = None
        self._committed_goal = None
        self._commit_age = 0
        self.MAX_COMMIT_AGE = 10
        self.TARGET_REPLAN_THRESHOLD = 3.0
        self.PATH_DEVIATION_THRESHOLD = 2.0

        # Leader-follower
        self._is_current_leader = False

        # Lead pursuit (Kalman)
        self.USE_LEAD_PURSUIT = True
        self.target_tracker = KalmanTargetTracker(
            dt=TIMESTEP, process_noise_std=2.0, obs_noise_std=0.3)
        self.LEAD_GAIN_MIN = 0.4
        self.LEAD_GAIN_MAX = 1.0
        self.VELOCITY_TRUSTED_THRESHOLD = 1.5

        self.states_prediction = np.ones((HORIZON_LENGTH + 1, self.n_state)) * self.state
        self.controls_prediction = np.zeros((HORIZON_LENGTH, self.n_control))

    def updateState(self, control, dt):
        position = self.state[:3]
        velocity = self.state[3:6]
        next_position = position + velocity * dt
        next_velocity = velocity + (control - D_FRAC * velocity) * dt
        self.state = np.concatenate([next_position, next_velocity])
        self.control = control
        self.time_stamp = self.time_stamp + dt

        self.path.append(np.concatenate([[self.time_stamp], self.state, self.control]))
        self.traj_refs.append(self.traj_ref)
        self.corridors_plot.append({'A': self.list_A, 'b': self.list_b})

        self.states_prediction[:-1, :] = self.states_prediction[1:, :]
        self.controls_prediction[:-1, :] = self.controls_prediction[1:, :]

    def computeControlSignal(self, robots):
        scan_data = self.lidar.senseObstacle(
            np.concatenate([self.state[:2], [0]]), robots)
        obstacle_points = self.lidar.getObstaclePoints(
            scan_data, np.concatenate([self.state[:2], [0]]))

        self.target_tracker.update(self.goal[:2])
        planning_goal = self._compute_predicted_goal()

        self.traj_ref = self.getOrientedGoalTrajectory(
            obstacle_points, planning_goal)

        if self.traj_ref is None:
            print(f"[Robot {self.index}] Emergency stop "
                  f"(fail count: {self.planner_fail_count})")
            self.list_A, self.list_b = [], []
            self.updateState(np.zeros(self.n_control), TIMESTEP)
            return

        target_pos = self.goal[:3].reshape(1, 3)
        self.list_A, self.list_b = self.generateSafeCorridor(
            self.traj_ref, obstacle_points)
        neighbor_robots = self.getNeighbors(robots)

        opti = ca.Opti()
        opt_states = opti.variable(HORIZON_LENGTH + 1, self.n_state)
        opt_controls = opti.variable(HORIZON_LENGTH, self.n_control)

        f = lambda x_, u_: ca.horzcat(x_[3:], u_)

        opti.subject_to(opt_states[0, :] == np.array([self.state]))
        for i in range(HORIZON_LENGTH):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]) * TIMESTEP
            opti.subject_to(opt_states[i + 1, :] == x_next)

        # Corridor hard
        if self.list_A:
            active_A, active_b = None, None
            for A, b in zip(self.list_A, self.list_b):
                if np.all(A @ self.state[:2] - b.flatten() <= 1e-5):
                    active_A = A
                    active_b = b
                    break
            self.corridors.append({'A': active_A, 'b': active_b})
            if active_A is not None:
                for i in range(HORIZON_LENGTH + 1):
                    opti.subject_to(
                        ca.mtimes(active_A, opt_states[i, :2].T)
                        <= active_b - ROBOT_RADIUS)

        # Neighbor collision (full, không filter)
        for i in range(HORIZON_LENGTH):
            for other_robot in neighbor_robots:
                if self.index == other_robot.index:
                    continue
                other_pos = ca.reshape(ca.DM(other_robot.states_prediction[i, :2]), 1, 2)
                dist_sq = ca.sumsqr(opt_states[i, :2] - other_pos)
                opti.subject_to(dist_sq >= (2 * ROBOT_RADIUS)**2)

        # Leader-follower CBF
        is_leader = self._update_leader_status(robots)
        if is_leader:
            slack_leader = opti.variable(HORIZON_LENGTH, 4)
            L = VIEWING_RADIUS
            for i in range(HORIZON_LENGTH):
                cur_pos = opt_states[i, :2]
                nxt_pos = opt_states[i + 1, :2]
                tgt = target_pos[0, :2]
                h_cur = ca.vertcat(
                    L - (cur_pos[0] - tgt[0]),
                    L - (tgt[0] - cur_pos[0]),
                    L - (cur_pos[1] - tgt[1]),
                    L - (tgt[1] - cur_pos[1]),
                )
                h_nxt = ca.vertcat(
                    L - (nxt_pos[0] - tgt[0]),
                    L - (tgt[0] - nxt_pos[0]),
                    L - (nxt_pos[1] - tgt[1]),
                    L - (tgt[1] - nxt_pos[1]),
                )
                for d in range(4):
                    opti.subject_to(
                        h_nxt[d] - (1 - DT_CBF_GAMMA) * h_cur[d]
                        >= -slack_leader[i, d])
            self._slack_leader = slack_leader
        else:
            self._slack_leader = None

        # Velocity & control bounds
        for i in range(HORIZON_LENGTH):
            vel = opt_states[i + 1, 3:]
            opti.subject_to(ca.sumsqr(vel) <= VMAX**2)
            con = opt_controls[i, :]
            opti.subject_to(ca.sumsqr(con) <= UMAX**2)

        opts_setting = {'ipopt.max_iter': 10000,
                        'ipopt.print_level': 0,
                        'ipopt.tol': 1e-4,
                        'ipopt.acceptable_tol': 1e-2,
                        'print_time': 0,
                        'ipopt.acceptable_iter': 15}
        opti.solver('ipopt', opts_setting)

        obj = self.costFunction(opt_states, opt_controls, self.traj_ref,
                                scan_data, self._slack_leader,
                                neighbor_robots, target_pos)
        opti.minimize(obj)
        opti.set_initial(opt_states, self.states_prediction)
        opti.set_initial(opt_controls, self.controls_prediction)

        try:
            sol = opti.solve()
            self.controls_prediction = sol.value(opt_controls)
            self.states_prediction = sol.value(opt_states)
            control = self.controls_prediction[0, :]
        except RuntimeError:
            print(f"[Robot {self.index}] MPC solve failed at t={self.time_stamp:.2f}")
            control = self.controls_prediction[0, :]
            if not hasattr(self, 'infeasible_log'):
                self.infeasible_log = []
            self.infeasible_log.append({
                't': self.time_stamp,
                'state': self.state.copy(),
                'goal': self.goal.copy(),
            })

        self.updateState(control, TIMESTEP)

    def _compute_predicted_goal(self):
        current_target = self.goal.copy()
        if not self.USE_LEAD_PURSUIT:
            return current_target
        vel_uncertainty = self.target_tracker.get_velocity_uncertainty()
        if vel_uncertainty > self.VELOCITY_TRUSTED_THRESHOLD:
            return current_target
        target_pos = self.target_tracker.get_position()
        uav_pos = self.state[:2]
        dist = float(np.linalg.norm(target_pos - uav_pos))
        travel_time = dist / max(VMAX, 1e-3)
        target_speed = self.target_tracker.get_speed()
        speed_ratio = target_speed / VMAX
        gain = 1.0 - speed_ratio
        gain = max(self.LEAD_GAIN_MIN, min(self.LEAD_GAIN_MAX, gain))
        target_vel = self.target_tracker.get_velocity()
        predicted_xy = target_pos + target_vel * travel_time * gain
        return np.array([predicted_xy[0], predicted_xy[1], current_target[2]])

    def _update_leader_status(self, robots):
        target_xy = self.goal[:2]
        my_dist = float(np.linalg.norm(self.state[:2] - target_xy))
        min_dist = my_dist
        nearest_index = self.index
        for other in robots:
            if other.index == self.index:
                continue
            d = float(np.linalg.norm(other.state[:2] - target_xy))
            if d < min_dist:
                min_dist = d
                nearest_index = other.index
        if self._is_current_leader:
            if nearest_index == self.index:
                self._is_current_leader = True
            elif (my_dist - min_dist) > LEADER_HYSTERESIS:
                self._is_current_leader = False
        else:
            self._is_current_leader = (nearest_index == self.index)
        return self._is_current_leader

    def getOrientedGoalTrajectory(self, obstacle_points, goal):
        current_robot_pos = np.array(self.state[:2])
        current_goal_pos = np.array(goal[:2])

        need_replan = True
        reason = ""

        # if self._committed_path is None:
        #     need_replan = True
        #     reason = "no committed path"
        # else:
        #     if self._commit_age >= self.MAX_COMMIT_AGE:
        #         need_replan = True
        #         reason = f"age limit ({self._commit_age})"
        #     elif self._committed_goal is not None:
        #         drift = np.linalg.norm(current_goal_pos - self._committed_goal)
        #         if drift > self.TARGET_REPLAN_THRESHOLD:
        #             need_replan = True
        #             reason = f"target moved {drift:.1f}m"
        #     if not need_replan:
        #         trimmed = self._trim_path_to_pose(
        #             self._committed_path, current_robot_pos)
        #         if not self._is_path_valid(trimmed, obstacle_points):
        #             need_replan = True
        #             reason = "path collision"
        #         else:
        #             dist = self._distance_to_path(
        #                 current_robot_pos, self._committed_path)
        #             if dist > self.PATH_DEVIATION_THRESHOLD:
        #                 need_replan = True
        #                 reason = f"deviation {dist:.1f}m"

        if need_replan:
            self.planner = AStarPlanner()
            self.planner.initialize(tuple(current_robot_pos),
                                    tuple(current_goal_pos))
            success, raw_path, _, _, _ = self.planner.plan(
                obstacle_points, ROBOT_RADIUS,
                time_budget_ms=self.PLANNER_TIME_BUDGET_MS)

            if success and len(raw_path) >= 2:
                smoothed = remove_residual_node(raw_path, obstacle_points, ROBOT_RADIUS)
                self._committed_path = np.array(smoothed)
                self._committed_goal = current_goal_pos.copy()
                self._commit_age = 0
                self.last_valid_path = self._committed_path.copy()
                self.planner_fail_count = 0
                return self._committed_path

            self.planner_fail_count += 1
            print(f"[Robot {self.index}] A* failed "
                  f"(consecutive: {self.planner_fail_count}, reason: {reason})")

            if self._committed_path is not None:
                trimmed = self._trim_path_to_pose(
                    self._committed_path, current_robot_pos)
                if self._is_path_valid(trimmed, obstacle_points):
                    self._commit_age += 1
                    return self._committed_path

            if self.planner_fail_count >= self.MAX_FAIL_BEFORE_STOP:
                return None

            return np.array([list(current_robot_pos), list(current_robot_pos)])

        self._commit_age += 1
        return self._committed_path

    @staticmethod
    def _trim_path_to_pose(path, pose):
        if path is None or len(path) < 2:
            return path
        arr = np.asarray(path)
        dists = np.linalg.norm(arr - pose, axis=1)
        idx_nearest = int(np.argmin(dists))
        remaining = list(path[idx_nearest + 1:]) if idx_nearest + 1 < len(path) else []
        if not remaining:
            return [list(pose), list(path[-1])]
        return [list(pose)] + [list(p) for p in remaining]

    @staticmethod
    def _is_path_valid(path, obstacle_points):
        if path is None or len(path) < 2:
            return False
        for i in range(len(path) - 1):
            a = Node(tuple(path[i]))
            b = Node(tuple(path[i + 1]))
            if is_collision(a, b, obstacle_points, ROBOT_RADIUS,
                            ignore_start=(i == 0)):
                return False
        return True

    @staticmethod
    def _distance_to_path(pose, path):
        if path is None or len(path) < 2:
            return float('inf')
        arr = np.asarray(path)
        min_d = float('inf')
        for i in range(len(arr) - 1):
            d = _point_to_segment_distance(pose, arr[i], arr[i + 1])
            if d < min_d:
                min_d = d
        return min_d

    # ===== Cost functions =====
    def costFunction(self, opt_states, opt_controls, traj_ref, scan_data,
                     slack_leader, neighbors, target_pos):
        c_u = self.costControl(opt_controls)
        c_tra = self.costTracking(opt_states, traj_ref)
        c_form = self.costFormation(opt_states, neighbors)
        c_corr_barrier = self.costCorridor(opt_states, self.list_A, self.list_b)
        c_centroid = self.costCentroidTracking(opt_states, neighbors, target_pos)
        c_leader_slack = self.costLeaderSlack(slack_leader)
        return c_tra + c_u + c_form + c_corr_barrier + c_centroid + c_leader_slack

    def costLeaderSlack(self, slack_leader):
        if slack_leader is None:
            return 0.0
        pos_slack = ca.fmax(slack_leader, 0)
        return W_leader_slack * ca.sum1(ca.sum2(pos_slack**3))

    def costCentroidTracking(self, opt_states, neighbors, target_pos):
        if neighbors is None or len(neighbors) == 0:
            cost = 0
            tgt_xy = target_pos[0, :2]
            for k in range(HORIZON_LENGTH + 1):
                cost += ca.sumsqr(opt_states[k, :2] - tgt_xy.reshape(1, 2))
            return W_centroid * cost

        n = 1 + len(neighbors)
        tgt_xy = target_pos[0, :2]
        cost = 0
        for k in range(HORIZON_LENGTH + 1):
            my_pos = opt_states[k, :2]
            neighbor_sum = ca.DM([0.0, 0.0])
            for other in neighbors:
                nb_pos = ca.DM(other.states_prediction[k, :2])
                neighbor_sum = neighbor_sum + nb_pos
            neighbor_sum_row = ca.reshape(neighbor_sum, 1, 2)
            centroid = (my_pos + neighbor_sum_row) / n
            cost += ca.sumsqr(centroid - tgt_xy.reshape(1, 2))
        return W_centroid * cost

    def costCorridor(self, traj, A, b):
        cost = 0
        if A is None or b is None or len(A) == 0:
            return 0.
        if isinstance(A, list) and len(A) == 1:
            A = A[0]
        if isinstance(b, list) and len(b) == 1:
            b = b[0]
        A = np.asarray(A)
        b = np.asarray(b)
        eps = 1e-1
        for i in range(HORIZON_LENGTH):
            pos = traj[i, :2]
            d = b - ca.mtimes(A, pos.T) - ROBOT_RADIUS
            cost += ca.sum1(1.0 / (d + eps))
        return W_corridor * cost

    def costControl(self, u):
        cost_u = 0
        for i in range(HORIZON_LENGTH):
            cost_u += ca.sumsqr(u[i, :])
        return W_u * cost_u

    def costTracking(self, traj, traj_ref):
        cost_tra = 0
        cost_gui = 0
        if traj_ref is not None and len(traj_ref) > 2:
            dist_guide = ca.sumsqr(traj[-1, :2] - traj_ref[1, :2].reshape(1, 2))
            dist_goal = 0
        else:
            dist_guide = 0
            dist_goal = ca.sumsqr(traj[-1, :2] - self.goal[:2].reshape(1, 2))
            cost_tra += (dist_goal - (VIEWING_RADIUS - TAR_MAX_SPEED)**2)**2
        cost_gui += dist_guide**2
        return W_tra * cost_tra + W_gui * cost_gui

    def costFormation(self, traj, neighbors):
        if not neighbors:
            return 0
        current_pos_start = self.state[:2]
        scores = []
        neighbor_predictions = []
        beta = 0.4
        for other_robot in neighbors:
            other_pos_start = other_robot.state[:2]
            dist_sq_start = np.sum((current_pos_start - other_pos_start)**2)
            scores.append(-beta * dist_sq_start)
            neighbor_predictions.append(other_robot.states_prediction)

        scores_ca = ca.DM(scores)
        exp_scores = ca.exp(scores_ca)
        sum_exp_scores = ca.sum1(exp_scores)
        alpha_weights = exp_scores / sum_exp_scores

        total = 0
        for i in range(HORIZON_LENGTH):
            current_pos_k = traj[i, :2]
            for j, other_prediction in enumerate(neighbor_predictions):
                other_pos_k = ca.reshape(ca.DM(other_prediction[i, :2]), 1, 2)
                alpha_j = alpha_weights[j]
                dist_sq = ca.sumsqr(current_pos_k - other_pos_k)
                cost_dist_j = W_form_dist * (dist_sq - DESIRED_SEPARATION**2)**2
                cost_spread_j = -W_form_spread * dist_sq
                combined = alpha_j * cost_dist_j + (1 - alpha_j) * cost_spread_j
                total += combined
        return total

    def getNeighbors(self, robots):
        neighbors = []
        current_pos = self.state[:3]
        for other_robot in robots:
            if self.index == other_robot.index:
                continue
            other_current_pos = other_robot.state[:3]
            distance = np.linalg.norm(current_pos - other_current_pos)
            if distance < SENSING_NEIGHBOR:
                neighbors.append(other_robot)
        return neighbors

    def generateSafeCorridor(self, path_ref, obstacle_points):
        if obstacle_points.shape[0] < 1:
            return [], []
        box = np.array([[VIEWING_RADIUS, VIEWING_RADIUS]])
        try:
            list_A, list_b = pdc.convex_decomposition_2D(
                obstacle_points, path_ref[0:2], box)
            return list_A, list_b
        except Exception as e:
            print(f"Error in generating safe corridor: {e}")
            return [], []