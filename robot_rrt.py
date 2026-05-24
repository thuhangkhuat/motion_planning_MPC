import numpy as np
import casadi as ca

import pydecomp as pdc

from lidar_fixed import LidarScanner
from utils import *
from planner_connect1 import RRTConnect, remove_residual_node   # <-- thay vì from planner import RRT
from kalman import KalmanTargetTracker

from config import *

import matplotlib.pyplot as plt


# ============================================================
# DESIGN MỚI - tham số (override hoặc bổ sung config.py)
# ------------------------------------------------------------
# Để dễ tune, đặt ở đây thay vì sửa config.py.
# ============================================================
W_centroid = 0.3              # weight cho cost centroid-target
                              # 0.3 = vừa, cân bằng với formation cost
                              # tăng -> centroid bám sát target hơn
                              # giảm -> UAV tự do hơn (có thể drift)

# Leader-follower
LEADER_HYSTERESIS = 3.0       # m, leader switch chỉ khi candidate
                              # gần target hơn current_leader > X m
                              # tránh flip-flop khi 2 UAV gần ngang nhau
W_leader_slack = 1e3          # weight phạt slack visibility của leader
                              # vi phạm tạm thời OK (vd né obstacle)
                              # nhưng cost cao -> ít vi phạm


# ============================================================
# Helper: distance từ 1 điểm đến 1 segment (cho path commit logic)
# ============================================================
def _point_to_segment_distance(p, a, b):
    """Khoảng cách Euclidean từ p tới segment ab. Tất cả là np.array (2,)."""
    ab = b - a
    seg_len_sq = float(np.dot(ab, ab))
    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / seg_len_sq)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


class Robot:
    def __init__(self, index, state: np.array, goal: np.array, control=np.zeros(3)):
        # Robot state and control
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

        # ----- Planner -----
        # CHANGE #3: dùng RRT-Connect thay vì RRT (file planner.py cũ).
        # RRT-Connect không cần warm-start (init mới mỗi cycle), nhưng nhanh
        # nhờ bi-directional + greedy connect.
        self.planner = None   # sẽ tạo mới mỗi cycle trong getOrientedGoalTrajectory

        # CHANGE #4: fallback path khi RRT fail
        # Lưu path gần nhất tìm được để dùng lại khi planner fail
        self.last_valid_path = None        # path 2D thành công gần nhất
        self.planner_fail_count = 0        # đếm số cycle liên tiếp planner fail
        self.MAX_FAIL_BEFORE_STOP = 3      # quá ngưỡng -> dừng UAV
        self.RRT_TIME_BUDGET_MS = 30       # time budget cho mỗi plan call

        # ─── Path commit parameters (chống flip-flop homotopy class) ───
        # Path đã commit chỉ replan khi cần thiết, không random mỗi cycle.
        self._committed_path = None              # path đang dùng (np.array Nx2)
        self._committed_goal = None              # goal lúc commit (cho check drift)
        self._commit_age = 0                     # số cycle đã dùng path này
        self.MAX_COMMIT_AGE = 50                 # force replan sau N cycles (5s)
        self.TARGET_REPLAN_THRESHOLD = 2.0       # target dịch > X m -> replan
        self.PATH_DEVIATION_THRESHOLD = 5.0      # UAV lệch path > X m -> replan

        # ─── Lead pursuit: dự đoán vị trí target tại thời điểm UAV đến ───
        # UAV KHÔNG biết trajectory target. Chỉ đo position mỗi frame
        # qua sensor (giả định: vision/radar). Kalman filter estimate
        # velocity từ history -> predict future position.
        self.USE_LEAD_PURSUIT = True             # bật/tắt lead pursuit
        self.target_tracker = KalmanTargetTracker(
            dt=TIMESTEP,
            process_noise_std=2.0,    # target có thể tăng tốc tới 2 m/s²
            obs_noise_std=0.3)        # sensor noise 0.3m

        # Adaptive gain parameters
        # gain = clamp(1.0 - target_speed/VMAX, MIN_GAIN, MAX_GAIN)
        # target nhanh ~ UAV -> gain thấp (aim gần để không over-shoot)
        # target chậm     -> gain cao (aim xa để intercept hiệu quả)
        self.LEAD_GAIN_MIN = 0.4
        self.LEAD_GAIN_MAX = 1.0
        # Velocity uncertainty threshold - dưới đây thì TIN tracker
        self.VELOCITY_TRUSTED_THRESHOLD = 1.5     # std velocity < 1.5 m/s thì dùng

        # ─── Lead pursuit: dự đoán vị trí target tại thời điểm UAV đến ───
        # Giải quyết: target di chuyển -> path "đuổi" target hiện tại sẽ
        # outdated khi UAV đến nơi. Cần aim trước.
        self.USE_LEAD_PURSUIT = True             # bật/tắt lead pursuit
        self.LEAD_PURSUIT_GAIN = 1.0             # 1.0 = aim đúng predict
                                                  # 0.5 = aim giữa current và predict
                                                  # >1.0 = aim quá xa, aggressive
        self.target_ref = None                   # tham chiếu tới target object (set bên ngoài)

        # Store the corridor
        self.corridors = []
        self.corridors_plot = []
        # Store robot path
        self.path = []
        self.traj_refs = []
        self.full_path = None
        self.path_update_counter = 0
        self.cached_path = None

        self.states_prediction = np.ones((HORIZON_LENGTH + 1, self.n_state)) * self.state
        self.controls_prediction = np.zeros((HORIZON_LENGTH, self.n_control))

    def updateState(self, control: np.array, dt: float):
        """Computes the states of robot after applying control signals"""
        position = self.state[:3]
        velocity = self.state[3:6]

        next_position = position + velocity * dt
        next_velocity = velocity + (control - D_FRAC * velocity) * dt

        self.state = np.concatenate([next_position, next_velocity])
        self.control = control
        self.time_stamp = self.time_stamp + dt

        # Store
        self.path.append(np.concatenate([[self.time_stamp], self.state, self.control]))
        self.traj_refs.append(self.traj_ref)
        self.corridors_plot.append({'A': self.list_A, 'b': self.list_b})

        # Shift predictive values
        self.states_prediction[:-1, :] = self.states_prediction[1:, :]
        self.controls_prediction[:-1, :] = self.controls_prediction[1:, :]

    def computeControlSignal(self, robots):
        """Computes control velocity of the copter"""
        scan_data = self.lidar.senseObstacle(
            np.concatenate([self.state[:2], [0]]), robots)
        obstacle_points = self.lidar.getObstaclePoints(
            scan_data, np.concatenate([self.state[:2], [0]]))

        # ─── Update Kalman tracker với target position hiện tại ───
        # UAV "đo" target qua sensor (ở đây giả định bằng self.goal)
        # Trong thực tế: replace bằng vision/radar measurement
        self.target_tracker.update(self.goal[:2])

        # ─── Lead pursuit: predict future target position ───
        planning_goal = self._compute_predicted_goal()

        # CHANGE #3 + #4: RRT-Connect + path commit + lead pursuit
        self.traj_ref = self.getOrientedGoalTrajectory(
            obstacle_points, planning_goal)

        # Fallback: nếu fail quá nhiều cycle -> emergency stop
        if self.traj_ref is None:
            # UAV dừng: control = 0, không update state qua MPC
            print(f"[Robot {self.index}] Emergency stop "
                  f"(fail count: {self.planner_fail_count})")
            self.list_A, self.list_b = [], []
            self.updateState(np.zeros(self.n_control), TIMESTEP)
            return

        target_pos = self.goal[:3].reshape(1, 3)
        self.list_A, self.list_b = self.generateSafeCorridor(self.traj_ref, obstacle_points)
        neighbor_robots = self.getNeighbors(robots)

        opti = ca.Opti()
        opt_states = opti.variable(HORIZON_LENGTH + 1, self.n_state)
        opt_controls = opti.variable(HORIZON_LENGTH, self.n_control)
        # DESIGN MỚI: bỏ slack_cbf cho FOV.
        # slack_corr cho corridor sẽ được tạo trong block corridor bên dưới.

        f = lambda x_, u_: ca.horzcat(*[
            x_[3:],
            u_
        ])

        opti.subject_to(opt_states[0, :] == np.array([self.state]))
        for i in range(HORIZON_LENGTH):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]) * TIMESTEP
            opti.subject_to(opt_states[i + 1, :] == x_next)

        # Convex corridor constraint (giữ nguyên — không spline)
        if self.list_A:
            active_A, active_b = None, None
            for A, b in zip(self.list_A, self.list_b):
                if np.all(A @ self.state[:2] - b.flatten() <= 1e-5):
                    active_A = A
                    active_b = b
                    break
            self.corridors.append({'A': active_A, 'b': active_b})

            # ============================================================
            # CORRIDOR: HARD CONSTRAINT (safety - không slack)
            # ------------------------------------------------------------
            # Đã thử CBF với slack -> UAV xuyên obstacle vì slack cho phép
            # ra ngoài corridor. Safety constraint NÊN hard.
            # MPC có thể infeasible khi hẹp -> dùng try-except fallback.
            # ============================================================
            if active_A is not None:
                for i in range(HORIZON_LENGTH + 1):
                    opti.subject_to(
                        ca.mtimes(active_A, opt_states[i, :2].T)
                        <= active_b - ROBOT_RADIUS)

        # ============================================================
        # CHANGE #1: BỎ filter `index <` trong neighbor collision
        # ------------------------------------------------------------
        # Lý do: code là PURE DECENTRALIZED (mỗi UAV solve riêng).
        # Filter `index <` chỉ đúng cho centralized MPC.
        # Trong decentralized, MỖI UAV phải tự đảm bảo tránh va chạm
        # với MỌI neighbor, vì UAV index lớn không biết constraint của
        # UAV index nhỏ.
        # ============================================================
        for i in range(HORIZON_LENGTH):
            for other_robot in neighbor_robots:
                # chỉ skip chính mình (không skip neighbor index nhỏ hơn)
                if self.index == other_robot.index:
                    continue
                other_pos = ca.reshape(ca.DM(other_robot.states_prediction[i, :2]), 1, 2)
                dist_sq = ca.sumsqr(opt_states[i, :2] - other_pos)
                opti.subject_to(dist_sq >= (2 * ROBOT_RADIUS)**2)

        # ============================================================
        # LEADER-FOLLOWER: chỉ leader có constraint visibility
        # ------------------------------------------------------------
        # Leader = UAV gần target nhất (với hysteresis chống flip-flop)
        # Leader: CBF FOV vuông với slack (visibility constraint)
        # Followers: không có visibility constraint, chỉ centroid cost
        # ============================================================
        is_leader = self._update_leader_status(robots)

        if is_leader:
            # Slack cho CBF leader (vi phạm tạm thời OK)
            slack_leader = opti.variable(HORIZON_LENGTH, 4)  # 4 hướng FOV
            L = VIEWING_RADIUS  # nửa cạnh FOV vuông

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

        # Velocity and control bounds (giữ nguyên — code gốc có lặp 2 lần,
        # mình gộp lại 1 lần cho gọn)
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

        # Cost function (leader-follower design)
        obj = self.costFunction(opt_states, opt_controls, self.traj_ref,
                                scan_data, self._slack_leader,
                                neighbor_robots, target_pos)
        opti.minimize(obj)

        # Initial guess
        opti.set_initial(opt_states, self.states_prediction)
        opti.set_initial(opt_controls, self.controls_prediction)

        # ============================================================
        # Try solve with fallback for infeasibility
        # ------------------------------------------------------------
        # Khi MPC infeasible (vd 3 UAV bị nén vào polytope hẹp + phải
        # nhìn target + không được chạm nhau):
        #   - Collision constraint vẫn HARD (không được vi phạm SAFETY)
        #   - Slack CBF cho FOV (preference, có thể tạm xa target)
        #   - Khi vẫn infeasible: KHÔNG apply control mới
        #     → dùng controls_prediction shifted từ frame trước
        #     → UAV decelerate tự nhiên (vì shifted có entry cuối = 0)
        #     → tạo "buffer time" để target di chuyển ra khỏi vùng khó
        # ============================================================
        try:
            sol = opti.solve()
            self.controls_prediction = sol.value(opt_controls)
            self.states_prediction = sol.value(opt_states)
            control = self.controls_prediction[0, :]
        except RuntimeError as e:
            # Solver fail (thường là infeasibility)
            print(f"[Robot {self.index}] MPC solve failed at t={self.time_stamp:.2f}: "
                  f"{type(e).__name__}")
            # Fallback: dùng prediction shifted từ frame trước.
            # controls_prediction[0] là control gốc cho frame TIẾP của lần trước,
            # tức control bây giờ "lẽ ra" sẽ dùng. UAV vẫn theo plan cũ.
            control = self.controls_prediction[0, :]
            # Lưu lại để debug audit
            if not hasattr(self, 'infeasible_log'):
                self.infeasible_log = []
            self.infeasible_log.append({
                't': self.time_stamp,
                'state': self.state.copy(),
                'goal': self.goal.copy(),
            })

        self.updateState(control, TIMESTEP)

    # ============================================================
    # LEAD PURSUIT: dự đoán vị trí target tại thời điểm UAV đến
    # ============================================================
    def _compute_predicted_goal(self):
        """
        Tính predicted target position dùng cho planning, dựa trên:
        - Vị trí target hiện tại (self.goal)
        - Velocity estimate từ Kalman tracker
        - Travel time = ||target - uav|| / VMAX
        - Adaptive gain theo tốc độ target

        Trả về np.array(3,) — predicted position [x, y, z].
        z giữ nguyên = self.goal[2].

        Nếu LEAD_PURSUIT tắt hoặc Kalman chưa converge -> trả về self.goal.
        """
        current_target = self.goal.copy()

        if not self.USE_LEAD_PURSUIT:
            return current_target

        # Kalman chưa tin được velocity -> aim current target
        vel_uncertainty = self.target_tracker.get_velocity_uncertainty()
        if vel_uncertainty > self.VELOCITY_TRUSTED_THRESHOLD:
            return current_target

        # ─── Tính travel time ───
        target_pos = self.target_tracker.get_position()
        uav_pos = self.state[:2]
        dist = float(np.linalg.norm(target_pos - uav_pos))
        # Travel time conservative: dùng VMAX (giả định UAV bay max)
        travel_time = dist / max(VMAX, 1e-3)

        # ─── Adaptive gain ───
        # target nhanh -> gain thấp (gần như aim current)
        # target chậm  -> gain cao (aim xa hơn)
        target_speed = self.target_tracker.get_speed()
        speed_ratio = target_speed / VMAX
        gain = 1.0 - speed_ratio
        gain = max(self.LEAD_GAIN_MIN, min(self.LEAD_GAIN_MAX, gain))

        # ─── Predicted position ───
        target_vel = self.target_tracker.get_velocity()
        predicted_xy = target_pos + target_vel * travel_time * gain

        # Giữ z component từ goal gốc
        predicted_goal = np.array([predicted_xy[0], predicted_xy[1],
                                    current_target[2]])
        return predicted_goal

    # ============================================================
    # ------------------------------------------------------------
    # Vấn đề: RRT random -> mỗi cycle ra path khác homotopy class
    #         -> UAV flip-flop giữa "đi trái" và "đi phải" obstacle
    #         -> bám tường, không tiến được
    #
    # Giải pháp: COMMIT path. Một khi tìm được path, LOCK lại.
    # Chỉ replan khi:
    #   (A) Path cũ đụng obstacle mới
    #   (B) Target di chuyển > threshold
    #   (C) UAV lệch khỏi path > threshold
    #   (D) Đã dùng path cũ quá N cycles (force refresh)
    # ============================================================
    # ============================================================
    # RRT-Connect with PATH COMMIT
    # ------------------------------------------------------------
    # Vấn đề: RRT random -> mỗi cycle ra path khác homotopy class
    #         -> UAV flip-flop giữa "đi trái" và "đi phải" obstacle
    #         -> bám tường, không tiến được
    #
    # Giải pháp: COMMIT path. Một khi tìm được path, LOCK lại.
    # Chỉ replan khi:
    #   (A) Path cũ đụng obstacle mới
    #   (B) Target di chuyển > threshold (sau khi áp lead pursuit)
    #   (C) UAV lệch khỏi path > threshold
    #   (D) Đã dùng path cũ quá N cycles (force refresh)
    # ============================================================
    def getOrientedGoalTrajectory(self, obstacle_points, goal):
        current_robot_pos = np.array(self.state[:2])
        current_goal_pos = np.array(goal[:2])

        # Khởi tạo các attribute nếu chưa có (lần đầu chạy)
        if not hasattr(self, '_committed_path'):
            self._committed_path = None
            self._committed_goal = None
            self._commit_age = 0

        # ─── Trigger checks ───
        need_replan = False
        reason = ""

        if self._committed_path is None:
            need_replan = True
            reason = "no committed path"
        else:
            # (D) Periodic refresh
            if self._commit_age >= self.MAX_COMMIT_AGE:
                need_replan = True
                reason = f"age limit ({self._commit_age})"

            # (B) Target moved significantly
            elif self._committed_goal is not None:
                target_drift = np.linalg.norm(current_goal_pos - self._committed_goal)
                if target_drift > self.TARGET_REPLAN_THRESHOLD:
                    need_replan = True
                    reason = f"target moved {target_drift:.1f}m"

            # (A) Path collision check
            if not need_replan:
                trimmed = self._trim_path_to_pose(self._committed_path,
                                                  current_robot_pos)
                if not self._is_path_valid(trimmed, obstacle_points):
                    need_replan = True
                    reason = "path collision"
                else:
                    # (C) UAV deviation check
                    dist_to_path = self._distance_to_path(
                        current_robot_pos, self._committed_path)
                    if dist_to_path > self.PATH_DEVIATION_THRESHOLD:
                        need_replan = True
                        reason = f"deviation {dist_to_path:.1f}m"

        # ─── Replan nếu cần ───
        if need_replan:
            # print(f"[Robot {self.index}] Replan: {reason}")
            self.planner = RRTConnect()
            self.planner.initialize(tuple(current_robot_pos),
                                    tuple(current_goal_pos))
            success, raw_path, _, _, _ = self.planner.plan(
                obstacle_points, ROBOT_RADIUS)

            if success and len(raw_path) >= 2:
                smoothed = remove_residual_node(raw_path, obstacle_points, ROBOT_RADIUS)
                self._committed_path = np.array(smoothed)
                self._committed_goal = current_goal_pos.copy()
                self._commit_age = 0
                self.last_valid_path = self._committed_path.copy()
                self.planner_fail_count = 0
                return self._committed_path

            # Plan fail
            self.planner_fail_count += 1
            print(f"[Robot {self.index}] RRT failed "
                  f"(consecutive: {self.planner_fail_count}, reason was: {reason})")
            # Vẫn thử dùng committed_path nếu nó còn valid
            if self._committed_path is not None:
                trimmed = self._trim_path_to_pose(self._committed_path,
                                                  current_robot_pos)
                if self._is_path_valid(trimmed, obstacle_points):
                    self._commit_age += 1
                    return self._committed_path

            # Emergency stop sau quá nhiều fail
            if self.planner_fail_count >= self.MAX_FAIL_BEFORE_STOP:
                return None

            # Tạm dừng UAV (path 1 điểm = pose hiện tại)
            return np.array([list(current_robot_pos), list(current_robot_pos)])

        # ─── Dùng path đã commit ───
        self._commit_age += 1
        return self._committed_path

    # ─── Helpers cho path commit ───
    @staticmethod
    def _trim_path_to_pose(path, pose):
        """Cắt bỏ phần đầu path UAV đã đi qua, prepend pose hiện tại."""
        if path is None or len(path) < 2:
            return path
        arr = np.asarray(path)
        dists = np.linalg.norm(arr - pose, axis=1)
        idx_nearest = int(np.argmin(dists))
        # Giữ từ idx_nearest+1 (điểm phía trước UAV)
        remaining = list(path[idx_nearest + 1:]) if idx_nearest + 1 < len(path) else []
        if not remaining:
            return [list(pose), list(path[-1])]
        return [list(pose)] + [list(p) for p in remaining]

    @staticmethod
    def _is_path_valid(path, obstacle_points):
        """Check toàn bộ path còn collision-free."""
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
        """Tính min distance từ pose tới các segment của path."""
        if path is None or len(path) < 2:
            return float('inf')
        arr = np.asarray(path)
        min_d = float('inf')
        for i in range(len(arr) - 1):
            d = _point_to_segment_distance(pose, arr[i], arr[i + 1])
            if d < min_d:
                min_d = d
        return min_d

    def _is_path_still_valid(self, path, obstacle_points):
        """Backward compat with previous fallback logic."""
        return self._is_path_valid(path, obstacle_points)

    # ============================================================
    # Cost function (LEADER-FOLLOWER DESIGN)
    # ------------------------------------------------------------
    # - Centroid tracking (mọi UAV): kéo centroid về target
    # - Leader slack penalty: chỉ áp khi self là leader
    # - Các cost khác như cũ
    # ============================================================
    def costFunction(self, opt_states, opt_controls, traj_ref, scan_data,
                     slack_leader, neighbors, target_pos):
        c_u = self.costControl(opt_controls)
        c_tra = self.costTracking(opt_states, traj_ref)
        c_form = self.costFormation(opt_states, neighbors)
        c_corr_barrier = self.costCorridor(opt_states, self.list_A, self.list_b)
        c_centroid = self.costCentroidTracking(opt_states, neighbors, target_pos)
        c_leader_slack = self.costLeaderSlack(slack_leader)

        total = (c_tra + c_u + c_form + c_corr_barrier
                 + c_centroid + c_leader_slack)
        return total

    # ============================================================
    # Cost slack cho leader visibility
    # ============================================================
    def costLeaderSlack(self, slack_leader):
        if slack_leader is None:
            return 0.0
        pos_slack = ca.fmax(slack_leader, 0)
        return W_leader_slack * ca.sum1(ca.sum2(pos_slack**3))

    # ============================================================
    # Leader selection: dynamic + hysteresis
    # ------------------------------------------------------------
    # Trả về True nếu self là leader trong cycle này.
    # Logic:
    #   - Leader = UAV gần target nhất
    #   - Nhưng nếu self đang là leader, switch chỉ khi
    #     candidate gần hơn self một khoảng LEADER_HYSTERESIS
    # ============================================================
    def _update_leader_status(self, robots):
        # Khởi tạo state nếu cần
        if not hasattr(self, '_is_current_leader'):
            self._is_current_leader = False

        # Tính khoảng cách mỗi UAV tới target (= self.goal)
        target_xy = self.goal[:2]
        my_dist = float(np.linalg.norm(self.state[:2] - target_xy))

        # Tìm UAV gần target nhất hiện tại
        min_dist = my_dist
        nearest_index = self.index
        for other in robots:
            if other.index == self.index:
                continue
            d = float(np.linalg.norm(other.state[:2] - target_xy))
            if d < min_dist:
                min_dist = d
                nearest_index = other.index

        # Hysteresis logic
        if self._is_current_leader:
            # Đang là leader - giữ leader trừ khi có UAV khác gần hơn rõ rệt
            if nearest_index == self.index:
                self._is_current_leader = True
            elif (my_dist - min_dist) > LEADER_HYSTERESIS:
                # Có UAV khác gần hơn self LEADER_HYSTERESIS m -> nhường
                self._is_current_leader = False
            # else: giữ leader (chưa đủ chênh lệch)
        else:
            # Không phải leader - chỉ thành leader nếu mình rõ ràng gần nhất
            self._is_current_leader = (nearest_index == self.index)

        return self._is_current_leader

    # ============================================================
    # NEW: Cost centroid tracking
    # ------------------------------------------------------------
    # Kéo centroid của swarm (UAV + neighbors) về target position.
    # Mọi UAV (cả leader lẫn followers) đều có cost này.
    # ============================================================
    def costCentroidTracking(self, opt_states, neighbors, target_pos):
        if neighbors is None or len(neighbors) == 0:
            # Trường hợp 1 UAV: rơi về tracking trực tiếp target
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
            # Tổng vị trí neighbors tại step k (constant từ states_prediction)
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

    def costSlack(self, slack_vars):
        positive_slack = ca.fmax(slack_vars, 0)
        return W_slack * ca.sum1(positive_slack**3)

    def costControl(self, u):
        cost_u = 0
        for i in range(HORIZON_LENGTH):
            control = u[i, :]
            cost_u += ca.sumsqr(control)
        return W_u * cost_u

    def costTracking(self, traj, traj_ref):
        """
        Giữ nguyên logic gốc:
        - Nếu có traj_ref với >= 3 điểm -> track traj_ref[1] (waypoint kế tiếp)
        - Ngược lại -> track về goal cuối horizon
        """
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

    def costCollision(self, traj, scan_data):
        """
        CHANGE #5: giữ method để reference nhưng KHÔNG GỌI nữa.
        Nếu sau này muốn bật làm safety layer cho obstacle thì uncomment
        trong costFunction. Hiện corridor đã handle obstacle.
        """
        cost_col = 0
        ang, dist = scan_data
        if dist.shape[0] != 0:
            min_idx = np.argmin(dist)
            obs_x = dist[min_idx] * np.cos(ang[min_idx]) + self.state[0]
            obs_y = dist[min_idx] * np.sin(ang[min_idx]) + self.state[1]
            for i in range(HORIZON_LENGTH):
                dist_sq = ca.sumsqr(traj[i, :2] - ca.DM([obs_x, obs_y]).T)
                margin = dist_sq - ROBOT_RADIUS**2
                cost_col += 1 / (margin + 1e-4)
        return W_col * cost_col

    # ============================================================
    # CHANGE #2: FIX BUG INDENT trong costFormation
    # ------------------------------------------------------------
    # Code gốc:
    #     for i in range(HORIZON_LENGTH):
    #         current_pos_k = traj[i, :2]
    #
    #     for j, other_prediction in ...:    # <-- nằm NGOÀI for i!
    #         ...
    #
    # => Loop j chỉ chạy 1 lần với i = HORIZON_LENGTH-1
    # => Formation cost chỉ áp tại state cuối horizon
    #
    # Fix: đưa loop j vào TRONG loop i, để mọi state của horizon
    # đều có formation cost.
    #
    # LƯU Ý: Sau fix, cost mạnh hơn ~HORIZON_LENGTH lần. Có thể
    # cần giảm W_form_dist, W_form_spread trong config tương ứng.
    # ============================================================
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

        total_formation_cost = 0
        for i in range(HORIZON_LENGTH):
            current_pos_k = traj[i, :2]
            # FIX: loop j ở TRONG loop i
            for j, other_prediction in enumerate(neighbor_predictions):
                other_pos_k = ca.reshape(ca.DM(other_prediction[i, :2]), 1, 2)
                alpha_j = alpha_weights[j]
                dist_sq = ca.sumsqr(current_pos_k - other_pos_k)

                cost_dist_j = W_form_dist * (dist_sq - DESIRED_SEPARATION**2)**2
                # cost_spread_j = -W_form_spread * dist_sq
                cost_spread_j = 0

                combined_cost_j = alpha_j * cost_dist_j + (1 - alpha_j) * cost_spread_j

                total_formation_cost += combined_cost_j

        return total_formation_cost

    def predictTrajectory(self, state, controls):
        """Computes states after applying a control sequence on initial state"""
        trajectory = []
        for i in range(HORIZON_LENGTH):
            position = state[:3]
            velocity = state[3:]
            control = controls[self.n_control * i:self.n_control * (i + 1)]
            next_position = position + velocity * TIMESTEP
            next_velocity = velocity + (control - D_FRAC * velocity) * TIMESTEP
            state = np.concatenate([next_position, next_velocity])
            trajectory.append(state)
        return np.array(trajectory)

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
        """Create convex polygon using pydecomp"""
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