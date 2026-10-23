"""
apfc.py — Baseline: active-perception formation control (Tallamraju, Price,
Ludwig, Karlapalem, Bülthoff, Black, Ahmad, "Active Perception based Formation
Control for Multiple Aerial Vehicles", IEEE RA-L 4(4), 2019,
arXiv:1901.07813; reference ROS code github.com/AIRCAP/AIRCAP, package
flight/nmpc_planner, mpc_act.cpp — read for the formulas, not copied).

Per UAV and step, a convex MPC (paper eq. 4-9) over N steps:
    min  sum_n |u_n + f_n|^2_R + kappa |p_N - p_des|^2 + |v_N - v_target|^2_W
    s.t. x_{n+1} = A x_n + B (u_n + f_n), state and input bounds
p_des: the point at distance d_des from the estimated target, on the line
from the target through the UAV; v_target: the target velocity estimate.
Every non-convex requirement is an external control input f_n computed
before the solve along the previous plan (paper Sec. III-E), each term a
cotangential potential F(d) (getPotential below) clamped to F_max:
    * away from the target below d_des (keep the safety distance);
    * away from each teammate (its published plan at step n);
    * tangential, scaled by (|r - d_des| + c), to spread the UAVs to angular
      gaps of 2*pi/K around the target, on the side away from the teammate;
    * away from each obstacle point, and tangential when an obstacle point
      within the approach range lies at a similar angle about the target.
There is no leader, slot or path planner: the formation is the circle.

Adaptations to this simulator (all in the comparison write-up):
  * planar, at the fixed altitude (the paper also keeps a desired height);
  * obstacles = the current LiDAR points (the reference code's
    POINT_OBSTACLES mode); target = the shared Kalman estimate (the paper
    fuses detections of all MAVs; here every UAV gets the same estimate);
  * the simulator's UAV model (exact discretisation, acceleration lag) with
    u + f as the commanded acceleration, |u + f| <= UMAX (physical limit);
    F_max = UMAX and |u| <= APFC_U_RATIO * UMAX with the reference ratio
    3/8 (input limit 3, FORCE_LIMIT 8). The cost penalises u + f, so with
    u_max >= F_max the MPC simply cancels the forces (u = -f): the paper's
    F_max > u_max is what makes the repulsion act;
  * N = HORIZON_LENGTH (paper 15 at 0.1 s; 20 here, as for the other methods);
  * thresholds of the reference launch/cfg files scaled to this simulator:
    d_des = APFC_RADIUS, target guarantee 7/8 of it (cfg 7.0 / 8.0);
    teammates: influence COLLISION_AVOID_DISTANCE, guarantee UAV_SAFE_DISTANCE
    (the proposed method's values); obstacles: guarantee d_min = R + VMAX*dt
    (paper eq. 15 with e_max = R), influence 2 d_min (cfg 3.0 / 1.5),
    approach range 2 x 3 m (VMAX 10 vs 5 m/s), approach angle 0.5 rad,
    active guarantee 0.01 rad (planner_realTarget.launch), gain 50, c = 0.1;
  * weights (not given in the paper): kappa = 100 (activeTrackingWeight),
    W = 100, R = 10 (energyWeight);
  * state bounds (VMAX, map border) are soft with a heavy penalty: with
    F_max > u_max hard bounds are not always feasible; the applied command is
    then saturated so that the speed stays <= VMAX (the low-level waypoint
    controller of the real MAVs has that limit);
  * solved with IPOPT through CasADi (the reference uses CVXGEN).
"""

import time

import numpy as np
import casadi as ca

import dynamics


def cotan_potential(d, threshold, inf_d, gain=50.0):
    """Cotangential repulsive potential gradient (reference getPotential):
    0 beyond `threshold`, growing to a large value as d approaches `inf_d`
    (held at its d = inf_d + 0.1 (threshold - inf_d) value below inf_d)."""
    if d >= threshold or threshold <= inf_d:
        return 0.0
    span = threshold - inf_d
    z = (np.pi / 2) * (0.1 if d < inf_d else (d - inf_d) / span)
    z = max(z, 1e-6)
    return gain * (np.pi / 2) / span * (np.cos(z) / np.sin(z) + z - np.pi / 2)


def _clamp(f, fmax):
    n = np.linalg.norm(f)
    return f if n <= fmax or n == 0 else f * (fmax / n)


def _angle_gap2(a, b):
    d = abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return d * d



def speed_limit(cfg, x0, a):
    """Low-level velocity saturation (also used by cbfqp.py): the real MAVs
    track the planner's output with a controller that has the speed limit,
    so an external input cannot drive them past VMAX. Scales the commanded
    acceleration a (2,) so that the next velocity of state x0 = [p, v, a]
    stays within VMAX, and brakes along the velocity if it is already above."""
    Ad, Bd = dynamics.model(cfg["TIMESTEP"], cfg["D_FRAC"], cfg["ACCEL_TAU"], 2)
    v0 = (Ad @ x0)[2:4]
    Bv = Bd[2:4, :]
    vmax = cfg["VMAX"]
    if np.linalg.norm(v0) > vmax:
        return -cfg["UMAX"] * v0 / np.linalg.norm(v0)
    dv = Bv @ a
    if np.linalg.norm(v0 + dv) <= vmax:
        return a
    A2, B2, C2 = dv @ dv, 2 * v0 @ dv, v0 @ v0 - vmax ** 2
    s = (-B2 + np.sqrt(max(B2 * B2 - 4 * A2 * C2, 0.0))) / (2 * A2)
    return np.clip(s, 0.0, 1.0) * a


class APFCProblem:
    """The convex MPC, built once per UAV and re-solved with parameters."""

    def __init__(self, cfg):
        self.cfg = cfg
        N = cfg["HORIZON_LENGTH"]
        dt = cfg["TIMESTEP"]
        opti = ca.Opti()
        X = opti.variable(N + 1, 6)                       # [px, py, vx, vy, ax, ay]
        U = opti.variable(N, 2)
        Sv = opti.variable(N, 1)                          # slack of the state bounds (see below)
        p = {"x0": opti.parameter(1, 6), "F": opti.parameter(N, 2),
             "pdes": opti.parameter(1, 2), "vdes": opti.parameter(1, 2)}
        Ad, Bd = dynamics.model(dt, cfg["D_FRAC"], cfg["ACCEL_TAU"], 2)
        opti.subject_to(X[0, :] == p["x0"])
        x0, y0, x1, y1 = cfg["WORLD_BOUNDS"]
        R = cfg["ROBOT_RADIUS"]
        for n in range(N):
            a = U[n, :] + p["F"][n, :]
            opti.subject_to(X[n + 1, :] == ca.mtimes(X[n, :], Ad.T) + ca.mtimes(a, Bd.T))
            opti.subject_to(ca.sumsqr(U[n, :]) <= (cfg["APFC_U_RATIO"] * cfg["UMAX"]) ** 2)
            opti.subject_to(ca.sumsqr(a) <= cfg["UMAX"] ** 2)
            # state bounds are soft: with F_max > u_max the forces can push the
            # UAV past VMAX or the map border and u cannot always undo it, so
            # hard bounds made ~40% of the solves infeasible
            opti.subject_to(ca.sumsqr(X[n + 1, 2:4]) <= cfg["VMAX"] ** 2 + Sv[n])
            opti.subject_to(opti.bounded(x0 + R - Sv[n], X[n + 1, 0], x1 - R + Sv[n]))
            opti.subject_to(opti.bounded(y0 + R - Sv[n], X[n + 1, 1], y1 - R + Sv[n]))
        opti.subject_to(Sv >= 0)
        cost = 0
        for n in range(N):
            cost += cfg["APFC_W_INPUT"] * ca.sumsqr(U[n, :] + p["F"][n, :])
        cost += cfg["APFC_W_POS"] * ca.sumsqr(X[N, :2] - p["pdes"])
        cost += cfg["APFC_W_VEL"] * ca.sumsqr(X[N, 2:4] - p["vdes"])
        cost += 1e4 * (ca.sum1(Sv) + ca.sumsqr(Sv))
        opti.minimize(cost)
        opts = dict(cfg["IPOPT_OPTIONS"])
        opts.setdefault("expand", True)
        opti.solver("ipopt", opts)
        self.opti, self.X, self.U, self.p, self.N = opti, X, U, p, N
        self.Sv = Sv
        self.solve_times, self.n_fail = [], 0

    def solve(self, x0, F, pdes, vdes, X_init, U_init):
        t0 = time.perf_counter()
        o, p = self.opti, self.p
        o.set_value(p["x0"], np.asarray(x0, float).reshape(1, 6))
        o.set_value(p["F"], np.asarray(F, float).reshape(self.N, 2))
        o.set_value(p["pdes"], np.asarray(pdes, float).reshape(1, 2))
        o.set_value(p["vdes"], np.asarray(vdes, float).reshape(1, 2))
        o.set_initial(self.X, X_init)
        o.set_initial(self.U, U_init)
        o.set_initial(self.Sv, 0)
        try:
            sol = o.solve()
            out = True, sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            self.n_fail += 1
            out = False, None, None
        self.solve_times.append(time.perf_counter() - t0)
        return out


class APFC:
    """One per UAV. plan() returns the commanded acceleration (u_0 + f_0)."""

    def __init__(self, robot, n_robots, cfg):
        self.robot, self.cfg, self.K = robot, cfg, n_robots
        self.N = cfg["HORIZON_LENGTH"]
        self.prob = APFCProblem(cfg)
        self.fmax = cfg["UMAX"]
        r = cfg["APFC_RADIUS"]
        self.r_des, self.r_guard = r, 7.0 / 8.0 * r
        self.nb_thr, self.nb_inf = cfg["COLLISION_AVOID_DISTANCE"], cfg["UAV_SAFE_DISTANCE"]
        dmin = cfg["ROBOT_RADIUS"] + cfg["VMAX"] * cfg["TIMESTEP"]
        self.ob_inf, self.ob_thr = dmin, 2.0 * dmin
        self.approach_range = 2.0 * 3.0
        self.approach_angle, self.active_guard = 0.5, 0.01
        self.gain, self.c = 50.0, 0.1
        self.X = self.U = None

    def _forces(self, plan_xy, target_xy, mates, obs):
        """External control input f_n (2,) for n = 0..N-1 along plan_xy."""
        F = np.zeros((self.N, 2))
        K = self.K
        for n in range(self.N):
            p = plan_xy[n]
            dT = p - target_xy
            rT = np.linalg.norm(dT)
            uT = dT / rT if rT > 1e-9 else np.array([1.0, 0.0])
            f = _clamp(cotan_potential(rT, self.r_des, self.r_guard, self.gain) * uT, self.fmax)
            tang = np.array([-uT[1], uT[0]])
            theta = np.arctan2(dT[1], dT[0])
            for m in mates:
                q = m[min(n, len(m) - 1)]
                d = p - q
                dist = np.linalg.norm(d)
                ud = d / dist if dist > 1e-9 else np.zeros(2)
                f = f + _clamp(cotan_potential(dist, self.nb_thr, self.nb_inf, self.gain) * ud, self.fmax)
                th_m = np.arctan2(q[1] - target_xy[1], q[0] - target_xy[0])
                fa = cotan_potential(_angle_gap2(theta, th_m), (2 * np.pi / K) ** 2,
                                     self.active_guard ** 2, self.gain)
                side = tang if ud @ tang > ud @ (-tang) else -tang
                f = f + _clamp(fa * side * (abs(rT - self.r_des) + self.c), self.fmax)
            if len(obs):
                d = p - obs
                dist = np.linalg.norm(d, axis=1)
                near = dist < max(self.ob_thr, self.approach_range)
                for q, dd, dv in zip(obs[near], dist[near], d[near]):
                    ud = dv / dd if dd > 1e-9 else np.zeros(2)
                    f = f + _clamp(cotan_potential(dd, self.ob_thr, self.ob_inf, self.gain) * ud, self.fmax)
                    if dd <= self.approach_range:
                        th_o = np.arctan2(q[1] - target_xy[1], q[0] - target_xy[0])
                        fa = cotan_potential(_angle_gap2(theta, th_o), self.approach_angle ** 2,
                                             self.active_guard ** 2, self.gain)
                        f = f + _clamp(fa * tang, self.fmax)
            F[n] = _clamp(f, self.fmax)                  # paper eq. 14
        return F

    def plan(self, target_xy, target_vel, obs, others):
        r, N = self.robot, self.N
        x0 = np.concatenate([r.state[[0, 1, 3, 4]], r.accel[:2]])
        if self.X is None:
            plan_xy = np.repeat(x0[None, :2], N + 1, axis=0)
        else:
            plan_xy = np.vstack([self.X[1:, :2], self.X[-1:, :2]])
        plan_xy[0] = x0[:2]
        mates = [np.asarray(o.states_prediction, float)[:, :2] for o in others]
        F = self._forces(plan_xy, target_xy, mates, np.asarray(obs, float).reshape(-1, 2))
        dT = x0[:2] - target_xy
        uT = dT / np.linalg.norm(dT) if np.linalg.norm(dT) > 1e-9 else np.array([1.0, 0.0])
        pdes = target_xy + self.r_des * uT
        Xi = np.tile(x0, (N + 1, 1)) if self.X is None else np.vstack([self.X[1:], self.X[-1:]])
        Ui = np.zeros((N, 2)) if self.U is None else np.vstack([self.U[1:], self.U[-1:]])
        ok, X, U = self.prob.solve(x0, F, pdes, target_vel, Xi, Ui)
        if ok:
            self.X, self.U = X, U
            r.states_prediction = np.column_stack(
                [X[:, :2], np.full(N + 1, r.state[2]), X[:, 2:4], np.zeros(N + 1)])
            r.controls_prediction = np.column_stack([U + F, np.zeros(N)])
            r.mpc_fail_streak = 0
            a = U[0] + F[0]
        else:
            r.mpc_fail_streak = getattr(r, "mpc_fail_streak", 0) + 1
            u = _clamp(self.U[1], self.cfg["APFC_U_RATIO"] * self.cfg["UMAX"]) if self.U is not None else np.zeros(2)
            a = _clamp(u + F[0], self.cfg["UMAX"])
        return np.array([*speed_limit(self.cfg, x0, a), 0.0])
