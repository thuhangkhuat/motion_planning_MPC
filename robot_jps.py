import numpy as np
import casadi as ca

import pydecomp as pdc

from lidar import LidarScanner
from utils import *
from planner_jps import JPSPlanner
from planner_grid import LocalGridPlanner
from mpc_problem import MPCProblem
import dynamics
from kalman_target import KalmanTargetTracker

from config import *

import logging

log = logging.getLogger("robot")

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


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


def _MPC_CFG():
    """Config constants used by MPCProblem (read at build time)."""
    import config as _c
    keys = ["HORIZON_LENGTH", "TIMESTEP", "ROBOT_RADIUS", "VMAX", "UMAX", "W_u", "W_gui",
            "W_tra", "STANDOFF_DISTANCE", "CORRIDOR_BARRIER_EPS", "W_corridor",
            "COLLISION_AVOID_DISTANCE", "W_collision_avoid", "UAV_SAFE_DISTANCE", "W_uav_slack",
            "WORLD_BOUNDS", "W_bounds_slack", "CORRIDOR_RECOVER_ACCEL_RATIO", "CORRIDOR_RECOVER_HOLD", "W_leader_slack", "D_FRAC", "ACCEL_TAU",
            "CBF_BOX_RATIO", "VIEWING_RADIUS", "DT_CBF_GAMMA", "IPOPT_OPTIONS",
            "MPC_SOLVER", "SQP_OPTIONS", "SQP_FALLBACK_IPOPT", "COST_LENGTH_SCALE", "COST_ACCEL_SCALE"]
    return {k: getattr(_c, k) for k in keys}


class Robot:
    # Slot assignment shared by the whole team (what the leader would broadcast).
    # Each UAV used to run Hungarian + hysteresis on its own: UAVs that entered
    # TRACK at different times kept different, equally cheap assignments, and
    # two satellites could hold the same cell forever. Now the first UAV that
    # needs it in a step computes it, the others in that step reuse it, and
    # the hysteresis compares against the team's previous assignment.
    _team_assign = {"step": None, "key": None, "assign": None}
    # Leader choice shared the same way: with a sticky leader kept per UAV,
    # UAVs could disagree after a hand-off (A follows B while B follows C),
    # leaving steps with no leader or with two.
    _team_leader = {"step": None, "leader": None}

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
        # actual acceleration (lags the command by ACCEL_TAU, see dynamics.py)
        self.accel = np.zeros(3)
        self._A, self._B = dynamics.model(TIMESTEP, D_FRAC, ACCEL_TAU, 3)

        self.lidar = LidarScanner(range_min=LIDAR_RANGE_MIN, range_max=SENSING_RADIUS,
                                  angle_min=-np.pi, angle_max=np.pi,
                                  resolution=LIDAR_ANGULAR_RES, noise=LIDAR_NOISE,
                                  march_step=LIDAR_MARCH_STEP, mode=LIDAR_MODE)
        self.mpc = None                 # MPCProblem, built on first use (parametric backend)

        # ----- Planner -----
        # CHANGE #3: dùng RRT-Connect thay vì RRT (file planner.py cũ).
        # RRT-Connect không cần warm-start (init mới mỗi cycle), nhưng nhanh
        # nhờ bi-directional + greedy connect.
        self.planner = None   # sẽ tạo mới mỗi cycle trong getOrientedGoalTrajectory

        # CHANGE #4: fallback path khi RRT fail
        # Lưu path gần nhất tìm được để dùng lại khi planner fail
        self.last_valid_path = None        # path 2D thành công gần nhất
        self.planner_fail_count = 0        # đếm số cycle liên tiếp planner fail
        self.MAX_FAIL_BEFORE_STOP = MAX_FAIL_BEFORE_STOP    # (config)
        self.RRT_TIME_BUDGET_MS = PLANNER_TIME_BUDGET_MS    # (config)

        # ─── Path commit parameters (chống flip-flop homotopy class) ───
        # Path đã commit chỉ replan khi cần thiết, không random mỗi cycle.
        self._committed_path = None              # path đang dùng (np.array Nx2)
        self._committed_goal = None              # goal lúc commit (cho check drift)
        self._commit_age = 0                     # số cycle đã dùng path này
        self.MAX_COMMIT_AGE = MAX_COMMIT_AGE                        # (config)
        self.TARGET_REPLAN_THRESHOLD = TARGET_REPLAN_THRESHOLD      # (config)
        self.PATH_DEVIATION_THRESHOLD = PATH_DEVIATION_THRESHOLD    # (config)

        # ─── Lead pursuit: dự đoán vị trí target tại thời điểm UAV đến ───
        # UAV KHÔNG biết trajectory target. Chỉ đo position mỗi frame
        # qua sensor (giả định: vision/radar). Kalman filter estimate
        # velocity từ history -> predict future position.
        self.USE_LEAD_PURSUIT = USE_LEAD_PURSUIT                    # (config)
        self.target_tracker = KalmanTargetTracker(
            dt=TIMESTEP,
            process_noise_std=KF_PROCESS_NOISE_STD,
            obs_noise_std=KF_OBS_NOISE_STD)

        # Adaptive gain parameters
        # gain = clamp(1.0 - target_speed/VMAX, MIN_GAIN, MAX_GAIN)
        # target nhanh ~ UAV -> gain thấp (aim gần để không over-shoot)
        # target chậm     -> gain cao (aim xa để intercept hiệu quả)
        self.LEAD_GAIN_MIN = LEAD_GAIN_MIN                          # (config)
        self.LEAD_GAIN_MAX = LEAD_GAIN_MAX                          # (config)
        # Velocity uncertainty threshold - dưới đây thì TIN tracker
        self.VELOCITY_TRUSTED_THRESHOLD = VELOCITY_TRUSTED_THRESHOLD  # (config)

        # ─── 2-PHASE FORMATION: Mode state ───
        # SEARCH: no UAV sees target. All UAVs chase homogeneously.
        # TRACK: ≥1 UAV sees target. Leader (the one seeing) + satellites.
        # Hysteresis: counter-based để tránh oscillation.
        self.mode = MODE_SEARCH
        self.c_in = 0                # counter: cycles liên tiếp có UAV thấy
        self.c_out = 0               # counter: cycles liên tiếp không UAV nào thấy
        self.c_handoff = 0           # counter: cycles UAV khác gần target hơn

        self.leader_index = None     # index của leader hiện tại (chỉ khi TRACK)
        self.is_leader_role = False  # self có phải leader không (cached mỗi cycle)

        # Satellite info (computed mỗi cycle khi TRACK)
        self.satellite_indices = []   # list các index satellites
        self.leader_state_pred = None # states_prediction của leader (cho cost)

        # Anchor state cho slot dynamic (hysteresis)
        # Anchor = satellite tham chiếu, slots tính từ anchor angle.
        # Hysteresis tránh anchor flip-flop khi 2 UAV gần ngang nhau.
        self.anchor_index = None      # index của anchor hiện tại
        self.my_slot_angle = None     # slot angle của self (computed mỗi cycle)

        # ─── Lead pursuit: dự đoán vị trí target tại thời điểm UAV đến ───
        # Giải quyết: target di chuyển -> path "đuổi" target hiện tại sẽ
        # outdated khi UAV đến nơi. Cần aim trước.
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
        # per-step debug record (debug_viewer.py): MPC plan used at this step and
        # [is_track, is_leader, mpc_fail_streak] and the formation contraction
        self.pred_hist = []
        self.status_hist = []
        self.scale_hist = []
        self.full_path = None
        self.path_update_counter = 0
        self.cached_path = None

        self.states_prediction = np.ones((HORIZON_LENGTH + 1, self.n_state)) * self.state
        self.controls_prediction = np.zeros((HORIZON_LENGTH, self.n_control))
        # last parametric-MPC solution in its own planar layout (warm start)
        self._mpc_X = None              # (H+1, 6) [px, py, vx, vy, ax, ay]
        self._mpc_U = None              # (H, 2)

    def updateState(self, control: np.array, dt: float):
        """Applies the commanded acceleration for one step (dynamics.py, exact)."""
        assert abs(dt - TIMESTEP) < 1e-12, "the discrete model is built for TIMESTEP"
        s = np.concatenate([self.state[:6], self.accel])
        s = self._A @ s + self._B @ np.asarray(control, float)
        self.state, self.accel = s[:6], s[6:]
        self.control = control
        self.time_stamp = self.time_stamp + dt

        # Store: [t, x, y, z, vx, vy, vz, u(3), a(3)]
        self.path.append(np.concatenate([[self.time_stamp], self.state, self.control, self.accel]))
        self.traj_refs.append(self.traj_ref)
        self.corridors_plot.append({'A': self.list_A, 'b': self.list_b})
        self.pred_hist.append(self.states_prediction[:, :2].astype(np.float32))
        self.status_hist.append((self.mode == MODE_TRACK, self.is_leader_role,
                                 getattr(self, "mpc_fail_streak", 0)))
        self.scale_hist.append(getattr(self, "slot_scale", 1.0)
                               if getattr(self, "_slot", None) is not None else np.nan)

        # Shift predictive values
        self.states_prediction[:-1, :] = self.states_prediction[1:, :]
        self.controls_prediction[:-1, :] = self.controls_prediction[1:, :]
        if self._mpc_X is not None:
            self._mpc_X[:-1, :] = self._mpc_X[1:, :]
            self._mpc_U[:-1, :] = self._mpc_U[1:, :]

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
        self._track_goal = self._compute_predicted_goal()

        # ─── Mode / role / slot, BEFORE planning: a satellite plans to its slot ───
        # (once per step: the mode hysteresis counters advance on every call)
        self._update_mode_and_role(robots)
        self._slot = None
        if self.mode == MODE_TRACK and not self.is_leader_role:
            self._slot = self._slot_target(robots)
        planning_goal = self._track_goal
        if self._slot is not None:
            # a slot inside an obstacle is fine: the planner goes to the reachable
            # cell closest to it, and the MPC pulls to the end of that path
            planning_goal = np.array([self._slot[0], self._slot[1], self._track_goal[2]])

        # CHANGE #3 + #4: RRT-Connect + path commit + lead pursuit
        self.traj_ref = self.getOrientedGoalTrajectory(
            obstacle_points, planning_goal)

        # Fallback: nếu fail quá nhiều cycle -> emergency stop
        if self.traj_ref is None:
            # UAV dừng: control = 0, không update state qua MPC
            log.warning("[Robot %d] Emergency stop (fail count: %d)",
                        self.index, self.planner_fail_count)
            self.list_A, self.list_b = [], []
            # zero acceleration keeps the current velocity (the UAV coasts on);
            # with FAILSAFE_BRAKE the UAV brakes at up to UMAX instead
            stop = self._brake_prediction() if FAILSAFE_BRAKE else np.zeros(self.n_control)
            self.updateState(stop, TIMESTEP)
            return

        target_pos = self.goal[:3].reshape(1, 3)
        self.list_A, self.list_b = self.generateSafeCorridor(self.traj_ref, obstacle_points)
        neighbor_robots = self.getNeighbors(robots)

        if MPC_BACKEND == "parametric":
            self._control_parametric(robots, neighbor_robots, target_pos)
            return

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
        # 2-PHASE MODE: SEARCH vs TRACK
        # ------------------------------------------------------------
        # SEARCH: all UAV chase target. No CBF.
        # TRACK: leader has CBF FOV. Satellites do formation.
        # ============================================================
        is_leader = self.is_leader_role          # updated in computeControlSignal

        # CBF chỉ áp khi self là LEADER trong TRACK mode
        if is_leader:
            slack_leader = opti.variable(HORIZON_LENGTH, 4)  # 4 hướng FOV
            L = CBF_BOX_RATIO * VIEWING_RADIUS   # half side of the leader's CBF box

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

        opti.solver('ipopt', dict(IPOPT_OPTIONS))

        # Cost function (2-phase formation)
        # Pass `robots` để truy cập satellites' states_prediction
        obj = self.costFunction(opt_states, opt_controls, self.traj_ref,
                                scan_data, self._slack_leader,
                                neighbor_robots, target_pos, robots)
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
            self.mpc_fail_streak = 0
        except RuntimeError as e:
            # Solver fail (thường là infeasibility)
            log.warning("[Robot %d] MPC solve failed at t=%.2f: %s",
                        self.index, self.time_stamp, type(e).__name__)
            # Fallback: dùng prediction shifted từ frame trước.
            # controls_prediction[0] là control gốc cho frame TIẾP của lần trước,
            # tức control bây giờ "lẽ ra" sẽ dùng. UAV vẫn theo plan cũ.
            control = self._fallback_control()
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
    # Parametric MPC backend (mpc_problem.py): same problem as the code above,
    # built once and re-solved with new parameter values every step.
    # ============================================================
    def _path_carrot(self, dist):
        """Point `dist` metres along traj_ref from the UAV, or None if the path is shorter."""
        P = np.vstack([self.state[:2], np.asarray(self.traj_ref)[1:, :2]])
        for a, b in zip(P[:-1], P[1:]):
            seg = float(np.linalg.norm(b - a))
            if seg >= dist:
                return a + (b - a) * (dist / seg)
            dist -= seg
        return None

    def _slot_trajectory(self):
        """(H+1, 2) slot position over the horizon: the leader's predicted path plus
        this satellite's cell offset. A fixed slot point makes the MPC plan to stop
        where the slot is NOW, so the satellite trails the moving formation."""
        lp = self.leader_state_pred
        cell = getattr(self, "my_slot_cell", None)
        if lp is None or cell is None:
            return np.asarray(self._slot[:2], float)
        return np.asarray(lp)[:, :2] + np.asarray(cell, float) * (
            SLOT_SPACING_RATIO * VIEWING_RADIUS * getattr(self, "slot_scale", 1.0))

    def _brake_control(self, s=None):
        """Command that cancels the velocity in one step, scaled down to UMAX.
        s = [state(6), accel(3)]; default: the current state."""
        if s is None:
            s = np.concatenate([self.state[:6], self.accel])
        u = -(self._A[3:6] @ s) / self._B[3, 0]       # v[k+1] = A[3:6] s + B[3,0] u
        n = np.linalg.norm(u)
        return u if n <= UMAX else u * (UMAX / n)

    def _brake_prediction(self):
        """Publish the braking trajectory as this UAV's prediction. The other
        UAVs plan around states_prediction; while braking it used to hold the
        last MPC plan (shifted), which this UAV no longer follows, and a
        neighbour planning 1 m from that stale plan flew into it."""
        s = np.concatenate([self.state[:6], self.accel])
        X = np.zeros((HORIZON_LENGTH + 1, self.n_state))
        Uc = np.zeros((HORIZON_LENGTH, self.n_control))
        X[0] = s[:6]
        for k in range(HORIZON_LENGTH):
            Uc[k] = self._brake_control(s)
            s = self._A @ s + self._B @ Uc[k]
            X[k + 1] = s[:6]
        self.states_prediction, self.controls_prediction = X, Uc
        self._mpc_X = self._mpc_U = None              # the stale plan is no warm start either
        return Uc[0]

    def _fallback_control(self):
        """
        Control when the MPC fails. The first MPC_FAILS_BEFORE_BRAKE consecutive
        failures reuse the previous plan (shifted by one step); after that the
        stale plan is no longer trusted and the UAV brakes. Without the brake,
        repeated failures replay the last planned control indefinitely, which
        drove UAVs into obstacles in the original code.
        """
        self.mpc_fail_streak = getattr(self, "mpc_fail_streak", 0) + 1
        if FAILSAFE_BRAKE and self.mpc_fail_streak > MPC_FAILS_BEFORE_BRAKE:
            return self._brake_prediction()
        # the shifted plan is what this UAV executes, so it stays the prediction
        return self.controls_prediction[0, :]

    def _control_parametric(self, robots, neighbor_robots, target_pos):
        active_A = active_b = None
        if self.list_A:
            for A, b in zip(self.list_A, self.list_b):
                if np.all(A @ self.state[:2] - b.flatten() <= 1e-5):
                    active_A, active_b = A, b
                    break
            else:
                # outside every polytope: the hard corridor used to be dropped
                # altogether, and the MPC planned straight through the obstacle.
                # Use the polytope the UAV is least outside of; the MPC's recovery
                # rule (mpc_problem.py) then lets it go no further out, and brings
                # it back inside (CORRIDOR_RECOVER_ACCEL_RATIO / _HOLD).
                k = int(np.argmin([np.max(A @ self.state[:2] - b.flatten())
                                   for A, b in zip(self.list_A, self.list_b)]))
                active_A, active_b = self.list_A[k], self.list_b[k]
            self.corridors.append({'A': active_A, 'b': active_b})

        is_leader = self.is_leader_role          # updated in computeControlSignal
        others = [r for r in robots if r.index != self.index]

        A_cost = self.list_A[0] if self.list_A else None
        b_cost = self.list_b[0] if self.list_b else None
        faces = max(0 if active_b is None else len(active_b),
                    0 if b_cost is None else len(b_cost))
        if self.mpc is None or faces > self.mpc.n_faces:
            n_faces = max(CORRIDOR_MAX_FACES, 16 * int(np.ceil(faces / 16)))
            if self.mpc is not None:
                log.info("[Robot %d] rebuilding MPC for %d corridor faces", self.index, n_faces)
            self.mpc = MPCProblem(len(others), n_faces, _MPC_CFG())

        w_search = w_slot = 0.0
        w_tra = W_tra
        slot = np.zeros(2)
        if self.mode == MODE_SEARCH:
            w_search = W_search_track
        elif self._slot is not None:
            # far from the slot: pull along the planned path (a straight pull through
            # an obstacle pins the UAV against the corridor; the path ends at the slot
            # or at the reachable cell closest to it). Within one horizon of it: track
            # the slot as it moves with the leader.
            carrot = (self._path_carrot(VMAX * HORIZON_LENGTH * TIMESTEP)
                      if self.traj_ref is not None else None)
            slot = carrot if carrot is not None else self._slot_trajectory()
            w_slot = W_sat_slot
            w_tra = 0.0       # the target standoff ring would fight the slot pull

        guide = self.traj_ref is not None and len(self.traj_ref) > 2
        ref = self.traj_ref[1, :2] if guide else np.zeros(2)
        trk_goal = getattr(self, '_track_goal', self.goal)
        nb_idx = {r.index for r in neighbor_robots}

        x0 = np.concatenate([self.state[[0, 1, 3, 4]], self.accel[:2]])
        if self._mpc_X is None:
            self._mpc_X = np.tile(x0, (HORIZON_LENGTH + 1, 1))
            self._mpc_U = np.zeros((HORIZON_LENGTH, 2))
        ok, Xs, Us = self.mpc.solve(
            x0=x0,
            A_hard=active_A, b_hard=active_b,
            A_cost=A_cost, b_cost=b_cost, use_corr_cost=A_cost is not None,
            others=[r.states_prediction for r in others],
            nb_flags=[1.0 if r.index in nb_idx else 0.0 for r in others],
            tgt=self._target_trajectory(target_pos[0, :2]), w_search=w_search, slot=slot, w_slot=w_slot,
            leader=is_leader, ref=ref, trk_goal=trk_goal, s_ref=1.0 if guide else 0.0,
            w_tra=w_tra,
            X_init=self._mpc_X, U_init=self._mpc_U)
        if ok:
            self._mpc_X, self._mpc_U = Xs, Us
            # back to the 3D layout the rest of the code uses (altitude held)
            H = HORIZON_LENGTH
            self.states_prediction = np.column_stack(
                [Xs[:, :2], np.full(H + 1, self.state[2]), Xs[:, 2:4], np.zeros(H + 1)])
            self.controls_prediction = np.column_stack([Us, np.zeros(H)])
            control = self.controls_prediction[0, :]
            self.mpc_fail_streak = 0
        else:
            log.warning("[Robot %d] MPC solve failed at t=%.2f", self.index, self.time_stamp)
            control = self._fallback_control()
            if not hasattr(self, 'infeasible_log'):
                self.infeasible_log = []
            self.infeasible_log.append({'t': self.time_stamp, 'state': self.state.copy(),
                                        'goal': self.goal.copy()})
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
    # JPS deterministic - cùng input -> cùng path. Path commit để tiết kiệm compute.
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
    # JPS with PATH COMMIT
    # ------------------------------------------------------------
    # JPS deterministic - cùng input -> cùng path. Path commit để tiết kiệm compute.
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
        current_goal_pos  = np.array(goal[:2])

        # ── Khởi tạo state + planner MỘT LẦN ──
        if not hasattr(self, '_committed_path'):
            self._committed_path  = None
            self._committed_goal  = None
            self._commit_age      = 0
            self.planner_fail_count = 0
            self.last_valid_path  = None
        if not hasattr(self, 'planner') or self.planner is None:
            # map bền vững sống suốt vòng đời robot (KHÔNG tạo lại mỗi replan)
            if PLANNER == "local":
                self.planner = LocalGridPlanner(
                    world_bounds=WORLD_BOUNDS, agent_id=self.index,
                    grid_resolution=GRID_RESOLUTION, inflate_radius=INFLATE_RADIUS,
                    sensing_radius=SENSING_RADIUS, local_radius=LOCAL_PLAN_RADIUS,
                    unknown_policy=UNKNOWN_POLICY, unknown_cost=UNKNOWN_COST,
                    start_snap_radius=START_SNAP_RADIUS,
                    goal_clamp_margin=GOAL_CLAMP_MARGIN, n_ray_bins=GRID_RAY_BINS)
            else:
                self.planner = JPSPlanner(world_bounds=WORLD_BOUNDS,
                                      agent_id=self.index,
                                      grid_resolution=GRID_RESOLUTION,
                                      inflate_radius=INFLATE_RADIUS,
                                      sensing_radius=SENSING_RADIUS,
                                      start_snap_radius=START_SNAP_RADIUS,
                                      goal_snap_radius=GOAL_SNAP_RADIUS,
                                      goal_clamp_margin=GOAL_CLAMP_MARGIN,
                                      n_ray_bins=GRID_RAY_BINS)

        # ── Nạp scan LiDAR vào map MỖI control-step (kể cả khi không replan) ──
        self.planner.observe(current_robot_pos, obstacle_points)

        # ✅ check va chạm DÙNG CHUNG grid của planner -> hết bất đối xứng
        #    (KHÔNG dùng is_collision trên obstacle_points 1-frame nữa)
        def _path_clear(path):
            pts = np.asarray(path, dtype=float)
            if len(pts) < 2:
                ix, iy = self.planner.gmap.world_to_grid(pts[0][0], pts[0][1])
                return self.planner.gmap.is_free(ix, iy)
            return all(self.planner.gmap.segment_clear(a, b)
                       for a, b in zip(pts[:-1], pts[1:]))

        # ── Trigger checks ──
        need_replan = False
        reason = ""

        if self._committed_path is None:
            need_replan = True; reason = "no committed path"
        else:
            # (D) Periodic refresh
            if self._commit_age >= self.MAX_COMMIT_AGE:
                need_replan = True; reason = f"age limit ({self._commit_age})"

            # (B) Target moved significantly
            elif self._committed_goal is not None:
                target_drift = np.linalg.norm(current_goal_pos - self._committed_goal)
                if target_drift > self.TARGET_REPLAN_THRESHOLD:
                    need_replan = True; reason = f"target moved {target_drift:.1f}m"

            # (A) Path collision check  +  (C) UAV deviation check
            if not need_replan:
                trimmed = self._trim_path_to_pose(self._committed_path,
                                                  current_robot_pos)
                if not _path_clear(trimmed):
                    need_replan = True; reason = "path collision"
                else:
                    dist_to_path = self._distance_to_path(
                        current_robot_pos, self._committed_path)
                    if dist_to_path > self.PATH_DEVIATION_THRESHOLD:
                        need_replan = True; reason = f"deviation {dist_to_path:.1f}m"

        # ── Replan nếu cần ──
        if need_replan:
            log.debug("[Robot %d] Replan: %s", self.index, reason)
            self.planner.initialize(tuple(current_robot_pos),
                                    tuple(current_goal_pos))
            # observe=False vì đã observe frame này ở trên rồi
            success, raw_path, _, _, _ = self.planner.plan(
                obstacle_points, ROBOT_RADIUS,
                time_budget_ms=self.RRT_TIME_BUDGET_MS,
                observe=False)

            if success and len(raw_path) >= 2:
                self._committed_path  = np.array(raw_path)
                self._committed_goal  = current_goal_pos.copy()
                self._commit_age      = 0
                self.last_valid_path  = self._committed_path.copy()
                self.planner_fail_count = 0
                return self._committed_path

            # Plan fail
            self.planner_fail_count += 1
            log.warning("[Robot %d] JPS failed (consecutive: %d, reason was: %s)",
                        self.index, self.planner_fail_count, reason)

            # Vẫn thử dùng committed_path nếu nó còn clear (check trên CÙNG grid)
            if self._committed_path is not None:
                trimmed = self._trim_path_to_pose(self._committed_path,
                                                  current_robot_pos)
                if _path_clear(trimmed):
                    self._commit_age += 1
                    return self._committed_path

            # Emergency stop sau quá nhiều fail
            if self.planner_fail_count >= self.MAX_FAIL_BEFORE_STOP:
                return None

            # Tạm dừng UAV (path 2 điểm = pose hiện tại)
            return np.array([list(current_robot_pos), list(current_robot_pos)])

        # ── Dùng path đã commit ──
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
    # Cost function (2-PHASE FORMATION DESIGN, TARGET-CENTERED)
    # ------------------------------------------------------------
    # Dispatch theo mode + role:
    #   SEARCH:   J_search = w_t * tracking target + common
    #   TRACK Leader:   J_leader = w_cbf * slack + common
    #                   (leader tự do, KHÔNG có slot, chỉ CBF)
    #   TRACK Satellite: J_sat = distance + angle + spread + common
    #                   (distance/angle relative to TARGET, không phải leader)
    # Common = control + corridor + path + collision_avoid
    # ============================================================
    def costFunction(self, opt_states, opt_controls, traj_ref, scan_data,
                     slack_leader, neighbors, target_pos, robots):
        # Common (mọi mode)
        c_u = self.costControl(opt_controls)
        c_tra = self.costTracking(opt_states, traj_ref)
        c_corr_barrier = self.costCorridor(opt_states, self.list_A, self.list_b)
        c_ca = self.costCollisionAvoid(opt_states, robots)

        common = c_u + c_tra + c_corr_barrier + c_ca

        # Mode-specific
        if self.mode == MODE_SEARCH:
            # SEARCH: kéo về target (predicted)
            c_search = self.costSearchTracking(opt_states, target_pos)
            return c_search + common

        # TRACK mode
        if self.is_leader_role:
            # Leader: penalize CBF slack, KHÔNG có slot
            c_leader_slack = self.costLeaderSlack(slack_leader)
            return c_leader_slack + common

        # Satellite (target-centered)
        c_slot = self.costSatelliteSlotDynamic(opt_states, robots)
        return c_slot + common

    # ============================================================
    # SEARCH cost: kéo UAV về (predicted) target
    # ------------------------------------------------------------
    # Áp lên cả horizon, không chỉ terminal.
    # ============================================================
    def costSearchTracking(self, opt_states, target_pos):
        tgt_xy = target_pos[0, :2]
        cost = 0
        for k in range(HORIZON_LENGTH + 1):
            cost += ca.sumsqr(opt_states[k, :2] - tgt_xy.reshape(1, 2))
        return W_search_track * cost

    # ============================================================
    # COMMON cost: Collision avoidance (soft, every mode)
    # ------------------------------------------------------------
    # Hard constraint d >= 2R đã bảo đảm KHÔNG đụng (feasibility).
    # Soft cost này maintain MARGIN an toàn d >= d_safe (> 2R).
    #
    # One-sided quadratic:
    #   d >= d_safe : cost = 0 (no penalty)
    #   d < d_safe  : cost = (d_safe - d)^2 (penalize violation)
    #
    # Lý do cần soft + hard:
    # 1. Hard có thể infeasible khi quá nhiều constraint -> fallback control cũ
    #    -> Soft kéo UAV ra xa từ trước, giảm risk infeasibility
    # 2. states_prediction (dùng cho hard) có error -> cần buffer
    # 3. d=2R là sát biên, không có margin cho disturbance
    # ============================================================
    def costCollisionAvoid(self, opt_states, robots):
        if not robots:
            return 0.0

        d_safe = COLLISION_AVOID_DISTANCE
        cost = 0
        for k in range(HORIZON_LENGTH + 1):
            my_pos = opt_states[k, :2]
            for other in robots:
                if other.index == self.index:
                    continue
                other_pos = ca.reshape(
                    ca.DM(other.states_prediction[k, :2]), 1, 2)
                diff = my_pos - other_pos
                d_sq = ca.sumsqr(diff)
                # ca.sqrt with small eps để smooth gradient
                d = ca.sqrt(d_sq + 1e-6)
                violation = ca.fmax(0, d_safe - d)
                cost += violation * violation
        return W_collision_avoid * cost

    # ============================================================
    # SATELLITE costs (chỉ áp khi self là satellite trong TRACK)
    # ============================================================
    def costSatelliteSlotDynamic(self, opt_states, robots):
        slot = getattr(self, "_slot", None)      # computed once per step in computeControlSignal
        if slot is None:
            return 0.0
        target_xy = ca.DM([slot[0], slot[1]])
        cost = 0
        for k in range(HORIZON_LENGTH + 1):
            cost += ca.sumsqr(opt_states[k, :2].T - target_xy)
        return W_sat_slot * cost

    def _slot_target(self, robots):
        """
        Gán slot ĐỘNG mỗi cycle bằng Hungarian (follower × slot), có hysteresis
        chống lật. Square-FOV: slot = ô lưới cách leader bội số 2L, axis-locked.

        OPEN_ALL_SLOTS=False: dùng n ô đầu (E,W,N,S...) -> đội hình đối xứng, ổn định.
        OPEN_ALL_SLOTS=True : mở 4 ô axis, Hungarian chọn n ô GẦN nhất -> thích nghi.

        Nhất quán decentralized: dùng r.state[:2] (global, mọi UAV thấy như nhau)
        + GRID_CELLS chung -> assignment giống hệt trên mọi UAV cùng cycle.
        """
        n = len(self.satellite_indices)
        if n == 0 or self.leader_current_pos is None:
            return None

        side = SLOT_SPACING_RATIO * VIEWING_RADIUS
        lead = np.asarray(self.leader_current_pos)

        # ── Tập slot ứng viên ──
        if OPEN_ALL_SLOTS:
            m = min(4, len(GRID_CELLS))        # 4 ô axis (vòng trong)
        else:
            m = min(n, len(GRID_CELLS))        # n ô đầu, danh tính cố định
        m = max(m, n)                          # every satellite needs a slot
        cells = (list(GRID_CELLS) + list(EXTRA_CELLS))[:m]
        slot_pos = [lead + np.array([dx, dy]) * side for dx, dy in cells]

        sats = sorted(self.satellite_indices)
        pos_of = {r.index: r.state[:2] for r in robots if r.index in self.satellite_indices}
        if self.index not in pos_of:
            return None

        shared = Robot._team_assign
        step = int(round(self.time_stamp / TIMESTEP))
        key = (self.leader_index, tuple(sats), m)
        if shared["step"] is not None and step < shared["step"]:
            shared.update(step=None, key=None, assign=None)    # a new run started
        if shared["step"] == step and shared["key"] == key:
            new_assign = shared["assign"]                      # already computed this step
        else:
            new_assign = self._assign_slots(sats, pos_of, slot_pos, m, shared)
            shared.update(step=step, key=key, assign=new_assign)
        self._assign = new_assign

        # ── Cost kéo self về slot của nó ──
        dx, dy = cells[new_assign[self.index]]
        self.my_slot_cell = (dx, dy)
        self._update_slot_scale(robots)
        return lead + np.array([dx, dy], float) * side * self.slot_scale

    def _target_trajectory(self, pos):
        """Target over the horizon, constant velocity from the Kalman estimate,
        for the leader CBF. The velocity is capped at 2*VMAX against filter
        spikes; before the filter has started the target is held fixed."""
        tr = self.target_tracker
        if not tr.initialized:
            return pos
        v = np.asarray(tr.get_velocity(), float)[:2]
        sp = np.linalg.norm(v)
        if not np.isfinite(sp):
            return pos
        if sp > 2 * VMAX:
            v *= 2 * VMAX / sp
        k = np.arange(HORIZON_LENGTH + 1)[:, None] * TIMESTEP
        return np.asarray(pos, float)[None, :2] + k * v[None]

    def _assign_slots(self, sats, pos_of, slot_pos, m, shared):
        """Hungarian satellite -> slot, keeping the team's previous assignment
        unless the new one is cheaper by SWITCH_MARGIN."""
        n = len(sats)

        # ── Ma trận chi phí n×m (m >= n) ──
        C = np.array([[float(np.linalg.norm(pos_of[i] - slot_pos[j]))
                    for j in range(m)] for i in sats])

        # ── Giải gán ──
        if _HAS_SCIPY:
            row, col = linear_sum_assignment(C)          # xử lý được ma trận chữ nhật
            new_assign = {sats[r]: int(c) for r, c in zip(row, col)}
        else:
            # Fallback không scipy: greedy theo chi phí tăng dần (đủ tốt cho n<=4)
            new_assign, used = {}, set()
            order = sorted(((C[a][b], a, b) for a in range(n) for b in range(m)))
            for _, a, b in order:
                if sats[a] in new_assign or b in used:
                    continue
                new_assign[sats[a]] = b
                used.add(b)
        new_cost = sum(C[sats.index(i)][new_assign[i]] for i in sats)

        # ── Hysteresis: giữ assignment cũ trừ khi cái mới rẻ hơn SWITCH_MARGIN ──
        # (the team's previous assignment, valid while the leader and the
        # satellite set are unchanged; it is a permutation, so no shared slot)
        prev = shared["assign"]
        if (prev is not None and shared["key"] is not None
                and shared["key"][1] == tuple(sats)
                and all(i in prev and prev[i] < m for i in sats)):
            prev_cost = sum(C[sats.index(i)][prev[i]] for i in sats)
            if new_cost > prev_cost - SWITCH_MARGIN:
                new_assign = prev
        return new_assign

    # ============================================================
    # Formation contraction near obstacles
    # ------------------------------------------------------------
    # A slot inside an obstacle, or behind one as seen from the leader, is moved
    # in along the leader->slot ray to just before the first obstacle cell. The
    # satellite then stays on the target's side instead of flying around the
    # obstacle; coverage shrinks (FOVs overlap) until the obstacle is passed.
    # ============================================================
    def _free_ray_length(self, robots, origin, direction, length):
        """How far along `direction` from `origin` the slot can sit, using every
        UAV's map (shared knowledge, so every UAV gets the same answer):
          1. line of sight: stop before the first OCCUPIED cell (the obstacle
             itself, not its inflation). A ray that only grazes an obstacle's
             inflated border is still a clear view; testing the inflated layer
             here made slots jump inwards whenever the ray passed ~3 m from a
             corner.
          2. the slot point itself must be outside the inflated layer, so the
             satellite can actually be there: step back until it is.
        Returns `length` if the slot is fine where it is."""
        step = 0.5 * GRID_RESOLUTION
        s = np.arange(step, length + 1e-9, step)
        P = np.asarray(origin, float)[None, :2] + s[:, None] * direction[None]
        # the map border counts as an obstacle: a slot beyond it is pulled in
        # (the planner clamps its goals there, but the MPC slot pull would not)
        (x0, y0, x1, y1), m = WORLD_BOUNDS, GOAL_CLAMP_MARGIN
        occ = ~((P[:, 0] >= x0 + m) & (P[:, 0] <= x1 - m) & (P[:, 1] >= y0 + m) & (P[:, 1] <= y1 - m))
        blk = np.zeros(len(s), bool)
        for r in robots:
            gm = getattr(getattr(r, "planner", None), "gmap", None)
            if gm is None:
                continue
            gm.is_blocked(0, 0)                       # rebuilds the inflated layer if stale
            ny, nx = gm.blocked.shape
            ix = np.floor((P[:, 0] - gm.ox) / gm.resolution).astype(int)
            iy = np.floor((P[:, 1] - gm.oy) / gm.resolution).astype(int)
            inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            occ[inb] |= gm.L[iy[inb], ix[inb]] >= gm.occ_logit
            b = gm.blocked[iy[inb], ix[inb]].astype(bool)
            unk = getattr(gm, "unknown", None)
            if unk is not None:                       # unseen space is not an obstacle
                b &= ~unk[iy[inb], ix[inb]].astype(bool)
            blk[inb] |= b
        n = len(s) if not occ.any() else int(np.argmax(occ))   # samples before the obstacle
        free = np.flatnonzero(~blk[:n])
        if n == len(s) and len(free) and free[-1] == len(s) - 1:
            return length
        return float(s[free[-1]]) if len(free) else 0.0

    def _update_slot_scale(self, robots):
        """self.slot_scale in [SLOT_MIN_RATIO, 1]: fraction of the slot distance that
        is free along the leader->slot ray, now and along the leader's predicted
        path (so the formation contracts before reaching the obstacle). Contracts
        at once, expands at most SLOT_EXPAND_SPEED."""
        if not SLOT_CONTRACT:
            self.slot_scale = 1.0
            return
        cell = np.asarray(self.my_slot_cell, float)
        full = float(np.linalg.norm(cell)) * SLOT_SPACING_RATIO * VIEWING_RADIUS
        u = cell / np.linalg.norm(cell)
        lp = self.leader_state_pred
        if lp is None:
            pts = [self.leader_current_pos]
        else:
            # the leader's MPC plan (one horizon), then its final velocity extrapolated
            # up to SLOT_LOOKAHEAD_TIME: moving a slot in by 20-35 m takes 3-4 s, more
            # than one horizon, so a later contraction makes the satellite trail it
            lp = np.asarray(lp)
            pts = list(lp[::max(1, HORIZON_LENGTH // 4), :2])
            t_extra = SLOT_LOOKAHEAD_TIME - HORIZON_LENGTH * TIMESTEP
            for tau in np.arange(0.5, t_extra + 1e-9, 0.5):
                pts.append(lp[-1, :2] + lp[-1, 3:5] * tau)
        free = min(self._free_ray_length(robots, p, u, full) for p in pts)
        target = float(np.clip(free / full, SLOT_MIN_RATIO, 1.0))
        prev = getattr(self, "slot_scale", 1.0)
        self.slot_scale = min(target, prev + SLOT_EXPAND_SPEED * TIMESTEP / full)
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
    # ============================================================
    # 2-PHASE MODE STATE MACHINE (replace _update_leader_status)
    # ------------------------------------------------------------
    # Mode = SEARCH (no UAV sees target) hoặc TRACK (≥1 UAV sees)
    # Hysteresis counter-based để tránh oscillation
    #
    # Visibility check: FOV vuông cạnh 2L quanh UAV
    #   sees(i) ⟺ ||p_i - p_target||∞ ≤ L
    #
    # Strict threshold L - ε: dùng để trigger TRACK (cần "rõ ràng thấy")
    # Relaxed threshold L: dùng để giữ TRACK (chấp nhận ở biên)
    # ============================================================
    def _update_mode_and_role(self, robots):
        target_xy = self.goal[:2]
        L = VIEWING_RADIUS
        epsilon = VISIBILITY_MARGIN_RATIO * L
        L_strict, L_relaxed = L - epsilon, L

        # ─── (1) Visibility set + chọn leader (sticky) ───
        dist_to_target, seers_strict, seers_relaxed = {}, [], []
        for r in robots:
            d_inf = max(abs(r.state[0] - target_xy[0]),
                        abs(r.state[1] - target_xy[1]))
            dist_to_target[r.index] = d_inf
            if d_inf <= L_strict:  seers_strict.append(r.index)
            if d_inf <= L_relaxed: seers_relaxed.append(r.index)

        shared = Robot._team_leader
        step = int(round(self.time_stamp / TIMESTEP))
        if shared["step"] is not None and step < shared["step"]:
            shared.update(step=None, leader=None)              # a new run started
        if shared["step"] != step:
            prev = shared["leader"]
            if prev is not None and prev in seers_relaxed:
                leader = prev  # giữ leader cũ -> tránh churn (BỎ proximity-handoff)
            elif seers_strict:
                leader = min(seers_strict, key=lambda i: dist_to_target[i])
            elif seers_relaxed:
                leader = min(seers_relaxed, key=lambda i: dist_to_target[i])
            else:
                leader = None  # mất target hoàn toàn
            shared.update(step=step, leader=leader)
        self.leader_index = shared["leader"]

        # ─── (2) Không leader -> mọi UAV SEARCH (reacquire) ───
        if self.leader_index is None:
            self.mode = MODE_SEARCH
            self.is_leader_role = False
            self.satellite_indices = []
            self.leader_state_pred = self.leader_current_pos = None
            self.c_in = 0
            return False

        # ─── (3) self là leader -> luôn TRACK ───
        if self.index == self.leader_index:
            self.mode = MODE_TRACK
            self.is_leader_role = True
            self.satellite_indices = [r.index for r in robots if r.index != self.leader_index]
            self.leader_state_pred = self.states_prediction
            self.leader_current_pos = self.state[:2].copy()
            return True

        # ─── (4) self là follower -> mode theo distance-to-leader ───
        leader_pos = None
        for r in robots:
            if r.index == self.leader_index:
                leader_pos = r.state[:2]
                self.leader_state_pred = r.states_prediction
                self.leader_current_pos = r.state[:2].copy()
                break
        if leader_pos is None:           # an toàn
            self.mode = MODE_SEARCH
            self.is_leader_role = False
            return False

        d_enter = VIEWING_RADIUS + FORMATION_GAP      # >= 2L theo khuyến nghị
        d_exit  = d_enter + TRACK_EXIT_HYSTERESIS
        cell = getattr(self, "my_slot_cell", None)
        if cell is not None:
            # a corner / outer slot is farther than d_enter: never drop out on the way to it
            d_exit = max(d_exit, float(np.hypot(*cell)) * SLOT_SPACING_RATIO * VIEWING_RADIUS
                         + TRACK_EXIT_HYSTERESIS)
        d2leader = float(np.linalg.norm(self.state[:2] - leader_pos))

        if self.mode == MODE_SEARCH:
            if d2leader <= d_enter:
                self.c_in += 1
                if self.c_in >= K_IN_THRESHOLD:
                    self.mode = MODE_TRACK; self.c_out = 0
            else:
                self.c_in = 0
        else:  # TRACK (satellite)
            if d2leader >= d_exit:
                self.c_out += 1
                if self.c_out >= K_OUT_THRESHOLD:
                    self.mode = MODE_SEARCH; self.c_in = 0
            else:
                self.c_out = 0

        self.is_leader_role = False
        self.satellite_indices = [r.index for r in robots if r.index != self.leader_index]

        log.debug("[t=%.1f] fol%d %s d2L=%.2f enter=%.2f",
                  self.time_stamp, self.index, self.mode, d2leader, d_enter)
        return False
    

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
        eps = CORRIDOR_BARRIER_EPS
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
            g = getattr(self, '_track_goal', self.goal)
            dist_goal = ca.sumsqr(traj[-1, :2] - g[:2].reshape(1, 2))
            # desired standoff = VIEWING_RADIUS - TAR_MAX_SPEED * STANDOFF_TIME (m);
            # the old expression subtracted a speed from a length
            cost_tra += (dist_goal - STANDOFF_DISTANCE**2)**2
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
        box = np.array([[CORRIDOR_BOX, CORRIDOR_BOX]])
        seg = np.array(path_ref[0:2], dtype=float)
        if len(seg) < 2 or np.linalg.norm(seg[1] - seg[0]) < 1e-6:
            # UAV already at its goal (e.g. a satellite on its slot): a zero-length
            # segment makes pydecomp divide by zero -> NaN faces. Use a short segment.
            seg = np.vstack([seg[0], seg[0] + [GRID_RESOLUTION, 0.0]])
        try:
            list_A, list_b = pdc.convex_decomposition_2D(obstacle_points, seg, box)
            keep = [i for i in range(len(list_A))
                    if np.all(np.isfinite(list_A[i])) and np.all(np.isfinite(list_b[i]))]
            if len(keep) < len(list_A):
                log.warning("[Robot %d] Dropped %d corridor polytope(s) with non-finite faces",
                            self.index, len(list_A) - len(keep))
            return [list_A[i] for i in keep], [list_b[i] for i in keep]
        except Exception as e:
            log.warning("[Robot %d] Error in generating safe corridor: %s", self.index, e)
            return [], []