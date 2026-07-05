import numpy as np
import casadi as ca

import pydecomp as pdc

from lidar import LidarScanner
from utils import *
from motion_planning_MPC.planner_jps import JPSPlanner
from kalman_target import KalmanTargetTracker

from config import *

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _point_to_segment_distance(p, a, b):
    """Euclidean distance from point p to segment ab. All are np.array (2,)."""
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

        self.n_state = 6
        self.n_control = 3

        self.lidar = LidarScanner(range_min=0.1, range_max=SENSING_RADIUS,
                                  angle_min=-np.pi, angle_max=np.pi,
                                  resolution=np.pi / 90)

        # ----- Planner -----
        # Created once on first use and kept for the robot's whole lifetime
        # (the JPS map is persistent, not rebuilt every replan).
        self.planner = None

        # Fallback path bookkeeping when the planner fails.
        self.last_valid_path = None        # most recent successful 2D path
        self.planner_fail_count = 0        # consecutive failed plan cycles
        self.MAX_FAIL_BEFORE_STOP = 3      # above this -> stop the UAV
        self.RRT_TIME_BUDGET_MS = 30       # time budget per plan call

        # ─── Path commit parameters (avoid homotopy-class flip-flop) ───
        # A committed path is only replanned when necessary, not every cycle.
        self._committed_path = None              # active path (np.array Nx2)
        self._committed_goal = None              # goal at commit time (drift check)
        self._commit_age = 0                     # cycles this path has been used
        self.MAX_COMMIT_AGE = 20                 # force replan after N cycles
        self.TARGET_REPLAN_THRESHOLD = 3.0       # target moved > X m -> replan
        self.PATH_DEVIATION_THRESHOLD = 5.0      # UAV off path > X m -> replan

        # ─── Lead pursuit: predict target position at UAV arrival time ───
        # The UAV does NOT know the target trajectory. It only measures
        # position each frame via a sensor (assumed vision/radar). A Kalman
        # filter estimates velocity from history to predict future position.
        self.USE_LEAD_PURSUIT = True
        self.target_tracker = KalmanTargetTracker(
            dt=TIMESTEP,
            process_noise_std=2.0,    # target may accelerate up to 2 m/s^2
            obs_noise_std=0.3)        # sensor noise 0.3 m

        # Adaptive gain: gain = clamp(1.0 - target_speed/VMAX, MIN, MAX)
        # fast target (~UAV speed) -> low gain (aim close, avoid overshoot)
        # slow target             -> high gain (aim far, intercept earlier)
        self.LEAD_GAIN_MIN = 0.4
        self.LEAD_GAIN_MAX = 1.0
        # Velocity-uncertainty threshold: below this we trust the tracker.
        self.VELOCITY_TRUSTED_THRESHOLD = 1.5     # std velocity < 1.5 m/s -> use

        # ─── 2-PHASE FORMATION: mode state ───
        # SEARCH: no UAV sees the target. All UAVs chase homogeneously.
        # TRACK:  >=1 UAV sees the target. Leader (the seer) + satellites.
        # Hysteresis is counter-based to avoid oscillation.
        self.mode = MODE_SEARCH
        self.c_in = 0                # consecutive cycles a UAV sees the target
        self.c_out = 0               # consecutive cycles no UAV sees the target
        self.c_handoff = 0           # consecutive cycles another UAV is closer

        self.leader_index = None     # current leader index (only in TRACK)
        self.is_leader_role = False  # whether self is leader (cached per cycle)

        # Satellite info (computed each cycle in TRACK).
        self.satellite_indices = []   # list of satellite indices
        self.leader_state_pred = None # leader's states_prediction (for cost)
        self.leader_current_pos = None  # leader's current position

        # Anchor state for dynamic slots (hysteresis).
        self.anchor_index = None      # current anchor index
        self.my_slot_angle = None     # self's slot angle (computed per cycle)
        self.my_slot_cell = None      # self's assigned grid cell (dx, dy)

        # Slot cells that SKIP the soft collision-avoidance margin (c_ca).
        # The front/lead slot [1,0] sits close to the leader's heading; the soft
        # margin keeps pushing it away so it cannot hold the front position.
        # The HARD constraint dist >= 2R still prevents a real collision, so
        # this only drops the extra buffer for that robot.
        self.NO_CA_SLOT_CELLS = {(1, 0)}

        # Store corridors / path for logging and plotting.
        self.corridors = []
        self.corridors_plot = []
        self.path = []
        self.traj_refs = []

        self.states_prediction = np.ones((HORIZON_LENGTH + 1, self.n_state)) * self.state
        self.controls_prediction = np.zeros((HORIZON_LENGTH, self.n_control))

    def updateState(self, control: np.array, dt: float):
        """Compute the robot state after applying control signals."""
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
        """Compute the control velocity of the copter."""
        scan_data = self.lidar.senseObstacle(
            np.concatenate([self.state[:2], [0]]), robots)
        obstacle_points = self.lidar.getObstaclePoints(
            scan_data, np.concatenate([self.state[:2], [0]]))

        # ─── Update Kalman tracker with the current target measurement ───
        # The UAV "measures" the target through a sensor (here assumed to be
        # self.goal). In practice, replace with a vision/radar measurement.
        self.target_tracker.update(self.goal[:2])

        # ─── Lead pursuit: predict the future target position ───
        planning_goal = self._compute_predicted_goal()
        self._track_goal = planning_goal

        # JPS + path commit + lead pursuit
        self.traj_ref = self.getOrientedGoalTrajectory(
            obstacle_points, planning_goal)

        # Fallback: too many failed cycles -> emergency stop
        if self.traj_ref is None:
            # UAV stops: control = 0, no MPC state update.
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

        f = lambda x_, u_: ca.horzcat(*[
            x_[3:],
            u_
        ])

        opti.subject_to(opt_states[0, :] == np.array([self.state]))
        for i in range(HORIZON_LENGTH):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]) * TIMESTEP
            opti.subject_to(opt_states[i + 1, :] == x_next)

        # Convex corridor constraint
        if self.list_A:
            active_A, active_b = None, None
            for A, b in zip(self.list_A, self.list_b):
                if np.all(A @ self.state[:2] - b.flatten() <= 1e-5):
                    active_A = A
                    active_b = b
                    break
            self.corridors.append({'A': active_A, 'b': active_b})

            # ============================================================
            # CORRIDOR: HARD CONSTRAINT (safety - no slack)
            # ------------------------------------------------------------
            # A CBF with slack let the UAV penetrate obstacles because slack
            # allowed leaving the corridor. Safety constraints should be hard.
            # The MPC may become infeasible in tight spaces -> handled by the
            # try/except fallback below.
            # ============================================================
            if active_A is not None:
                for i in range(HORIZON_LENGTH + 1):
                    opti.subject_to(
                        ca.mtimes(active_A, opt_states[i, :2].T)
                        <= active_b - ROBOT_RADIUS)

        # ============================================================
        # Neighbor collision avoidance (PURE DECENTRALIZED)
        # ------------------------------------------------------------
        # Each UAV solves on its own, so every UAV must avoid EVERY neighbor.
        # There is no `index <` filter (that only holds for centralized MPC):
        # a higher-index UAV cannot see the constraints of a lower-index one.
        # ============================================================
        for i in range(HORIZON_LENGTH):
            for other_robot in neighbor_robots:
                # only skip self (do not skip lower-index neighbors)
                if self.index == other_robot.index:
                    continue
                other_pos = ca.reshape(ca.DM(other_robot.states_prediction[i, :2]), 1, 2)
                dist_sq = ca.sumsqr(opt_states[i, :2] - other_pos)
                opti.subject_to(dist_sq >= (2 * ROBOT_RADIUS)**2)

        # ============================================================
        # 2-PHASE MODE: SEARCH vs TRACK
        # ------------------------------------------------------------
        # SEARCH: all UAVs chase the target. No CBF.
        # TRACK:  leader has the CBF FOV constraint; satellites do formation.
        # ============================================================
        is_leader = self._update_mode_and_role(robots)

        # CBF only applies when self is the LEADER in TRACK mode.
        if is_leader:
            slack_leader = opti.variable(HORIZON_LENGTH, 4)  # 4 FOV directions
            L = VIEWING_RADIUS / 10

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

        # Velocity and control bounds
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

        # Cost function (2-phase formation). `robots` is passed so the cost can
        # access the satellites' states_prediction.
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
        # When the MPC is infeasible (e.g. several UAVs squeezed into a narrow
        # polytope while also needing to see the target and stay apart):
        #   - Collision constraints stay HARD (safety is never violated).
        #   - The FOV uses slack (a preference; may briefly move off-target).
        #   - If still infeasible: do NOT apply a new control
        #       -> reuse the shifted controls_prediction from the previous frame
        #       -> the UAV decelerates naturally (last shifted entry = 0)
        #       -> this buys time for the target to leave the hard region.
        # ============================================================
        try:
            sol = opti.solve()
            self.controls_prediction = sol.value(opt_controls)
            self.states_prediction = sol.value(opt_states)
            control = self.controls_prediction[0, :]
        except RuntimeError as e:
            # Solver failed (usually infeasibility).
            print(f"[Robot {self.index}] MPC solve failed at t={self.time_stamp:.2f}: "
                  f"{type(e).__name__}")
            # Fallback: reuse the shifted prediction from the previous frame.
            control = self.controls_prediction[0, :]
            # Keep a debug audit log.
            if not hasattr(self, 'infeasible_log'):
                self.infeasible_log = []
            self.infeasible_log.append({
                't': self.time_stamp,
                'state': self.state.copy(),
                'goal': self.goal.copy(),
            })

        self.updateState(control, TIMESTEP)

    # ============================================================
    # LEAD PURSUIT: predict the target position at UAV arrival time
    # ============================================================
    def _compute_predicted_goal(self):
        """
        Compute the predicted target position used for planning, based on:
        - current target position (self.goal)
        - velocity estimate from the Kalman tracker
        - travel time = ||target - uav|| / VMAX
        - adaptive gain by target speed

        Returns np.array(3,) -- predicted position [x, y, z]; z stays = self.goal[2].
        If lead pursuit is off or the Kalman filter has not converged, returns self.goal.
        """
        current_target = self.goal.copy()

        if not self.USE_LEAD_PURSUIT:
            return current_target

        # Kalman velocity not yet trustworthy -> aim at the current target.
        vel_uncertainty = self.target_tracker.get_velocity_uncertainty()
        if vel_uncertainty > self.VELOCITY_TRUSTED_THRESHOLD:
            return current_target

        # ─── Travel time ───
        target_pos = self.target_tracker.get_position()
        uav_pos = self.state[:2]
        dist = float(np.linalg.norm(target_pos - uav_pos))
        # Conservative travel time: assume the UAV flies at VMAX.
        travel_time = dist / max(VMAX, 1e-3)

        # ─── Adaptive gain ───
        # fast target -> low gain (close to current); slow target -> high gain.
        target_speed = self.target_tracker.get_speed()
        speed_ratio = target_speed / VMAX
        gain = 1.0 - speed_ratio
        gain = max(self.LEAD_GAIN_MIN, min(self.LEAD_GAIN_MAX, gain))

        # ─── Predicted position ───
        target_vel = self.target_tracker.get_velocity()
        predicted_xy = target_pos + target_vel * travel_time * gain

        # Keep the z component from the original goal.
        predicted_goal = np.array([predicted_xy[0], predicted_xy[1],
                                   current_target[2]])
        return predicted_goal

    # ============================================================
    # JPS with PATH COMMIT
    # ------------------------------------------------------------
    # JPS is deterministic: same input -> same path, so without commit the UAV
    # flip-flops between going left/right of an obstacle, hugs walls and stalls.
    #
    # Solution: COMMIT the path. Once found, LOCK it. Replan only when:
    #   (A) the old path hits a new obstacle
    #   (B) the target moved > threshold (after lead pursuit)
    #   (C) the UAV deviated from the path > threshold
    #   (D) the path has been used for more than N cycles (force refresh)
    # ============================================================
    def getOrientedGoalTrajectory(self, obstacle_points, goal):
        current_robot_pos = np.array(self.state[:2])
        current_goal_pos = np.array(goal[:2])

        # ── Initialize state + planner once ──
        if not hasattr(self, '_committed_path'):
            self._committed_path = None
            self._committed_goal = None
            self._commit_age = 0
            self.planner_fail_count = 0
            self.last_valid_path = None
        if not hasattr(self, 'planner') or self.planner is None:
            # Persistent map living for the whole robot lifetime (not rebuilt per replan).
            self.planner = JPSPlanner(world_bounds=(0, 0, 50, 15),
                                      agent_id=self.index)

        # ── Feed the LiDAR scan into the map every control step (even without replan) ──
        self.planner.observe(current_robot_pos, obstacle_points)

        # Collision check uses the planner's SHARED grid -> no asymmetry
        # (no longer using is_collision on single-frame obstacle_points).
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

        # ── Replan if needed ──
        if need_replan:
            self.planner.initialize(tuple(current_robot_pos),
                                    tuple(current_goal_pos))
            # observe=False since we already observed this frame above.
            success, raw_path, _, _, _ = self.planner.plan(
                obstacle_points, ROBOT_RADIUS,
                time_budget_ms=self.RRT_TIME_BUDGET_MS,
                observe=False)

            if success and len(raw_path) >= 2:
                # raw_path is already smoothed inside plan().
                self._committed_path = np.array(raw_path)
                self._committed_goal = current_goal_pos.copy()
                self._commit_age = 0
                self.last_valid_path = self._committed_path.copy()
                self.planner_fail_count = 0
                return self._committed_path

            # Plan failed
            self.planner_fail_count += 1
            print(f"[Robot {self.index}] JPS failed "
                  f"(consecutive: {self.planner_fail_count}, reason was: {reason})")

            # Still try the committed path if it remains clear (same grid).
            if self._committed_path is not None:
                trimmed = self._trim_path_to_pose(self._committed_path,
                                                  current_robot_pos)
                if _path_clear(trimmed):
                    self._commit_age += 1
                    return self._committed_path

            # Emergency stop after too many failures.
            if self.planner_fail_count >= self.MAX_FAIL_BEFORE_STOP:
                return None

            # Hold the UAV in place (2-point path = current pose).
            return np.array([list(current_robot_pos), list(current_robot_pos)])

        # ── Use the committed path ──
        self._commit_age += 1
        return self._committed_path

    # ─── Path-commit helpers ───
    @staticmethod
    def _trim_path_to_pose(path, pose):
        """Drop the part of the path already passed and prepend the current pose."""
        if path is None or len(path) < 2:
            return path
        arr = np.asarray(path)
        dists = np.linalg.norm(arr - pose, axis=1)
        idx_nearest = int(np.argmin(dists))
        # Keep from idx_nearest+1 (the point ahead of the UAV).
        remaining = list(path[idx_nearest + 1:]) if idx_nearest + 1 < len(path) else []
        if not remaining:
            return [list(pose), list(path[-1])]
        return [list(pose)] + [list(p) for p in remaining]

    @staticmethod
    def _distance_to_path(pose, path):
        """Minimum distance from pose to the path segments."""
        if path is None or len(path) < 2:
            return float('inf')
        arr = np.asarray(path)
        min_d = float('inf')
        for i in range(len(arr) - 1):
            d = _point_to_segment_distance(pose, arr[i], arr[i + 1])
            if d < min_d:
                min_d = d
        return min_d

    # ============================================================
    # Cost function (2-PHASE FORMATION, TARGET-CENTERED)
    # ------------------------------------------------------------
    # Dispatch by mode + role:
    #   SEARCH:          J = w_t * track_target + common
    #   TRACK leader:    J = w_cbf * slack + common (free, no slot, CBF only)
    #   TRACK satellite: J = distance + angle + spread + common
    #                    (relative to TARGET, not to the leader)
    # common = control + corridor + path + collision_avoid
    # ============================================================
    def costFunction(self, opt_states, opt_controls, traj_ref, scan_data,
                     slack_leader, neighbors, target_pos, robots):
        # Common (every mode)
        c_u = self.costControl(opt_controls)
        c_tra = self.costTracking(opt_states, traj_ref)
        c_corr_barrier = self.costCorridor(opt_states, self.list_A, self.list_b)
        c_ca = self.costCollisionAvoid(opt_states, robots)

        common = c_u + c_tra + c_corr_barrier + c_ca

        # Mode-specific
        if self.mode == MODE_SEARCH:
            # SEARCH: pull toward the (predicted) target.
            c_search = self.costSearchTracking(opt_states, target_pos)
            return c_search + common

        # TRACK mode
        if self.is_leader_role:
            # Leader: penalize CBF slack, no slot.
            c_leader_slack = self.costLeaderSlack(slack_leader)
            return c_leader_slack + common

        # Satellite (target-centered)
        c_slot = self.costSatelliteSlotDynamic(opt_states, robots)
        return c_slot + common

    # ============================================================
    # SEARCH cost: pull the UAV toward the (predicted) target
    # ------------------------------------------------------------
    # Applied over the whole horizon, not just the terminal state.
    # ============================================================
    def costSearchTracking(self, opt_states, target_pos):
        tgt_xy = target_pos[0, :2]
        cost = 0
        for k in range(HORIZON_LENGTH + 1):
            cost += ca.sumsqr(opt_states[k, :2] - tgt_xy.reshape(1, 2))
        return W_search_track * cost

    # ============================================================
    # COMMON cost: collision avoidance (soft, every mode)
    # ------------------------------------------------------------
    # The hard constraint d >= 2R guarantees no collision (feasibility).
    # This soft cost maintains a safety margin d >= d_safe (> 2R).
    #
    # One-sided quadratic:
    #   d >= d_safe : cost = 0
    #   d <  d_safe : cost = (d_safe - d)^2
    #
    # Why both soft + hard:
    # 1. The hard constraint can be infeasible with too many constraints
    #    (-> fallback to old control). The soft cost pushes the UAV away early,
    #    reducing the risk of infeasibility.
    # 2. states_prediction (used for the hard constraint) has error -> needs buffer.
    # 3. d = 2R is right at the boundary, leaving no margin for disturbance.
    # ============================================================
    def costCollisionAvoid(self, opt_states, robots):
        if not robots:
            return 0.0

        # Skip the soft margin for the satellite holding an exempt (front) slot,
        # so it can stay in front of the leader without being pushed away.
        # Hard safety (dist >= 2R in computeControlSignal) is unaffected.
        # my_slot_cell is set in costSatelliteSlotDynamic; it lags by one cycle
        # here, which is fine given the slot hysteresis.

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
                # ca.sqrt with a small eps for a smooth gradient.
                d = ca.sqrt(d_sq + 1e-6)
                violation = ca.fmax(0, d_safe - d)
                cost += violation * violation
        return W_collision_avoid * cost

    # ============================================================
    # SATELLITE cost (only when self is a satellite in TRACK)
    # ============================================================
    def costSatelliteSlotDynamic(self, opt_states, robots):
        """
        Assign a DYNAMIC slot each cycle via Hungarian (follower x slot) with
        hysteresis to prevent flipping. Square-FOV: a slot is a grid cell offset
        from the leader by a multiple of 2L, axis-locked.

        OPEN_ALL_SLOTS=False: use the first n cells (E,W,N,S...) -> symmetric, stable.
        OPEN_ALL_SLOTS=True : open 4 axis cells; Hungarian picks the n NEAREST -> adaptive.

        Decentralized consistency: uses r.state[:2] (global, identical for every UAV)
        + the shared GRID_CELLS, so the assignment is identical on all UAVs in the same cycle.
        """
        n = len(self.satellite_indices)
        if n == 0 or self.leader_current_pos is None:
            return 0.0

        side = 4.0 * VIEWING_RADIUS
        lead = np.asarray(self.leader_current_pos)

        # ── Candidate slot set ──
        if OPEN_ALL_SLOTS:
            m = min(4, len(GRID_CELLS))        # 4 axis cells (inner ring)
        else:
            m = min(n, len(GRID_CELLS))        # first n cells, fixed identity
        cells = GRID_CELLS[:m]
        slot_pos = [lead + np.array([dx, dy]) * side for dx, dy in cells]

        sats = sorted(self.satellite_indices)
        pos_of = {r.index: r.state[:2] for r in robots if r.index in self.satellite_indices}
        if self.index not in pos_of:
            return 0.0

        # ── Cost matrix n x m (m >= n) ──
        C = np.array([[float(np.linalg.norm(pos_of[i] - slot_pos[j]))
                       for j in range(m)] for i in sats])

        # ── Solve assignment ──
        if _HAS_SCIPY:
            row, col = linear_sum_assignment(C)          # handles rectangular matrices
            new_assign = {sats[r]: int(c) for r, c in zip(row, col)}
        else:
            # No-scipy fallback: greedy by increasing cost (good enough for n <= 4).
            new_assign, used = {}, set()
            order = sorted(((C[a][b], a, b) for a in range(n) for b in range(m)))
            for _, a, b in order:
                if sats[a] in new_assign or b in used:
                    continue
                new_assign[sats[a]] = b
                used.add(b)
        new_cost = sum(C[sats.index(i)][new_assign[i]] for i in sats)

        # ── Hysteresis: keep the old assignment unless the new one is cheaper by SWITCH_MARGIN ──
        prev = getattr(self, '_assign', None)
        if (prev is not None and all(i in prev for i in sats)
                and all(prev[i] < m for i in sats)):
            prev_cost = sum(C[sats.index(i)][prev[i]] for i in sats)
            if new_cost > prev_cost - SWITCH_MARGIN:
                new_assign = prev
        self._assign = new_assign

        # ── Cost pulling self toward its slot ──
        # Anchor the slot to the leader's PREDICTED position at each horizon
        # step, not the static current position. Otherwise the front slot [1,0]
        # sits at a fixed point while the leader keeps advancing into it -> the
        # leader rear-ends the front satellite and it can never lead ahead.
        # Using the leader's plan, the slot moves forward together with it.
        dx, dy = cells[new_assign[self.index]]
        self.my_slot_cell = (dx, dy)

        lead_pred = self.leader_state_pred  # (HORIZON_LENGTH+1, n_state)
        cost = 0
        for k in range(HORIZON_LENGTH + 1):
            if lead_pred is not None:
                lx, ly = float(lead_pred[k, 0]), float(lead_pred[k, 1])
            else:
                lx, ly = lead[0], lead[1]
            slot_k = ca.DM([lx + dx * side, ly + dy * side])
            cost += ca.sumsqr(opt_states[k, :2].T - slot_k)
        return W_sat_slot * cost

    # ============================================================
    # Leader visibility slack cost
    # ============================================================
    def costLeaderSlack(self, slack_leader):
        if slack_leader is None:
            return 0.0
        pos_slack = ca.fmax(slack_leader, 0)
        return W_leader_slack * ca.sum1(ca.sum2(pos_slack**3))

    # ============================================================
    # 2-PHASE MODE STATE MACHINE
    # ------------------------------------------------------------
    # Mode = SEARCH (no UAV sees target) or TRACK (>=1 UAV sees it).
    # Counter-based hysteresis to avoid oscillation.
    #
    # Visibility check: a square FOV of side 2L around the UAV
    #   sees(i)  <=>  ||p_i - p_target||_inf <= L
    #
    # Strict threshold L - eps : triggers TRACK (must clearly see the target).
    # Relaxed threshold L      : holds TRACK (accepts being at the boundary).
    #
    # Returns True if self is the leader this cycle.
    # ============================================================
    def _update_mode_and_role(self, robots):
        target_xy = self.goal[:2]
        L = VIEWING_RADIUS
        epsilon = VISIBILITY_MARGIN_RATIO * L
        L_strict, L_relaxed = L - epsilon, L

        # ─── (1) Visibility set + leader selection (sticky) ───
        dist_to_target, seers_strict, seers_relaxed = {}, [], []
        for r in robots:
            d_inf = max(abs(r.state[0] - target_xy[0]),
                        abs(r.state[1] - target_xy[1]))
            dist_to_target[r.index] = d_inf
            if d_inf <= L_strict:  seers_strict.append(r.index)
            if d_inf <= L_relaxed: seers_relaxed.append(r.index)

        if self.leader_index is not None and self.leader_index in seers_relaxed:
            pass  # keep the old leader -> avoid churn (no proximity handoff)
        elif seers_strict:
            self.leader_index = min(seers_strict, key=lambda i: dist_to_target[i])
        elif seers_relaxed:
            self.leader_index = min(seers_relaxed, key=lambda i: dist_to_target[i])
        else:
            self.leader_index = None  # target fully lost

        # ─── (2) No leader -> every UAV goes SEARCH (reacquire) ───
        if self.leader_index is None:
            self.mode = MODE_SEARCH
            self.is_leader_role = False
            self.satellite_indices = []
            self.leader_state_pred = self.leader_current_pos = None
            self.c_in = 0
            return False

        # ─── (3) self is leader -> always TRACK ───
        if self.index == self.leader_index:
            self.mode = MODE_TRACK
            self.is_leader_role = True
            self.satellite_indices = [r.index for r in robots if r.index != self.leader_index]
            self.leader_state_pred = self.states_prediction
            self.leader_current_pos = self.state[:2].copy()
            return True

        # ─── (4) self is a follower -> mode by distance-to-leader ───
        leader_pos = None
        for r in robots:
            if r.index == self.leader_index:
                leader_pos = r.state[:2]
                self.leader_state_pred = r.states_prediction
                self.leader_current_pos = r.state[:2].copy()
                break
        if leader_pos is None:           # safety
            self.mode = MODE_SEARCH
            self.is_leader_role = False
            return False

        d_enter = VIEWING_RADIUS + FORMATION_GAP      # >= 2L as recommended
        d_exit = d_enter + TRACK_EXIT_HYSTERESIS
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

        if self.index == 1:   # debug one follower
            print(f"[t={self.time_stamp:.1f}] fol{self.index} {self.mode} "
                  f"d2L={d2leader:.2f} enter={d_enter:.2f}")
        return False

    def costCorridor(self, traj, A, b):
        cell = self.my_slot_cell
        # if (self.mode == MODE_TRACK and not self.is_leader_role
        #         and cell is not None and tuple(cell) in self.NO_CA_SLOT_CELLS):
        #     return 0.0
        if (self.mode == MODE_TRACK and self.is_leader_role):
            return 0.0
        
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
            control = u[i, :]
            cost_u += ca.sumsqr(control)
        return W_u * cost_u

    def costTracking(self, traj, traj_ref):
        """
        - If traj_ref has >= 3 points -> track traj_ref[1] (next waypoint).
        - Otherwise -> track the final goal at the horizon end.
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
            cost_tra += (dist_goal - (VIEWING_RADIUS - TAR_MAX_SPEED)**2)**2
        cost_gui += dist_guide**2
        return W_tra * cost_tra + W_gui * cost_gui

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
        """Create a convex polygon using pydecomp."""
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