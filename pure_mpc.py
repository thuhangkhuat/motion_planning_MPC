"""
pure_mpc.py — Baseline "pure_mpc": the plain MPC of this project's first
version (robot_mpc.py, removed in commit ff5ffdc), on the shared leader /
slot layers. No path planner (JPS), corridor or CBF.

    min  W_tra sum_k |p_k - goal|^2 / L^2 + W_u sum_k |u_k|^2
         + W_col sum_k sum_j sigma(p_k, o_j)
    s.t. dynamics, |v_k| <= VMAX, |u_k| <= UMAX,
         |p_k - o_j| >= CORRIDOR_MARGIN  (hard, every LiDAR point within reach)
sigma = 1 / (1 + exp(4 (|p - o|^2 - c^2) / L^2)) over every other LiDAR
point (and the other UAVs), c = L / 2. Goal: the formation slot, or the
predicted target for the leader.

Changes made so that it works at 10 m/s on the 1000 m maps (the original
flew into obstacles there): the hard distance constraint covers every LiDAR
point within the horizon reach, not only the nearest one (the UAV slid along
that one point into the wall), with the corridor margin of the proposed
method (relaxed to the current distance if the UAV is already closer), the
64 nearest points; other UAVs are avoided along their published plans with
UAV_SAFE_DISTANCE, as in every MPC of the comparison (the original treated
them as static LiDAR points, which let two UAVs come 0.5 m close); the
goal in the tracking cost is moved to at most the horizon reach from the UAV
(a goal hundreds of metres away made the tracking cost dwarf the collision
cost).

Scaling from the original (VMAX = 1 m/s, R = 0.25 m, same 0.1 s step):
lengths x L = VMAX / 1 m/s, so its sigmoid centre sqrt(0.25) m becomes L / 2
and the squared-distance terms are divided by L^2; weights as in the original
config (W_tra 1, W_u 0.4, W_col 1.5). Changes: the simulator's UAV model;
the control cost sums over the horizon (the original kept only the last
step, a bug); the sigmoid is written with tanh (same function, no overflow
for far points); a failed solve flies the previous plan, then brakes (the
original raised an exception). Built once and re-solved with parameters.
"""

import time

import numpy as np
import casadi as ca

import dynamics


class PureMPC:

    def __init__(self, robot, n_others, cfg):
        self.robot, self.cfg = robot, cfg
        H = cfg["HORIZON_LENGTH"]
        self.H, self.L = H, cfg["VMAX"] / 1.0
        self.M = 96 + n_others                            # obstacle points in the cost (padded)
        self.Mh = 64                                      # nearest points in the hard constraint (padded)
        self.n_others = n_others
        L, R = self.L, cfg["ROBOT_RADIUS"]
        opti = ca.Opti()
        X = opti.variable(H + 1, 6)
        U = opti.variable(H, 2)
        p = {"x0": opti.parameter(1, 6), "goal": opti.parameter(1, 2),
             "hard": opti.parameter(self.Mh, 2), "hmin": opti.parameter(self.Mh, 1),
             "obs": opti.parameter(self.M, 2),
             "oth": [opti.parameter(H + 1, 2) for _ in range(n_others)],
             "omin": opti.parameter(max(n_others, 1), 1)}
        Ad, Bd = dynamics.model(cfg["TIMESTEP"], cfg["D_FRAC"], cfg["ACCEL_TAU"], 2)
        opti.subject_to(X[0, :] == p["x0"])
        for i in range(H):
            opti.subject_to(X[i + 1, :] == ca.mtimes(X[i, :], Ad.T) + ca.mtimes(U[i, :], Bd.T))
            opti.subject_to(ca.sumsqr(X[i + 1, 2:4]) <= cfg["VMAX"] ** 2)
            opti.subject_to(ca.sumsqr(U[i, :]) <= cfg["UMAX"] ** 2)
            dh = ca.sum2((ca.repmat(X[i + 1, :2], self.Mh, 1) - p["hard"]) ** 2)
            opti.subject_to(dh >= p["hmin"])
            for j in range(n_others):                     # other UAVs: their published plans
                opti.subject_to(ca.sumsqr(X[i + 1, :2] - p["oth"][j][i + 1, :]) >= p["omin"][j])
        c2 = (L / 2) ** 2
        cost = 0
        for i in range(H):
            cost += cfg["W_u"] * ca.sumsqr(U[i, :])
            cost += cfg["W_tra"] * ca.sumsqr(X[i, :2] - p["goal"]) / L ** 2
            d2 = ca.sum2((ca.repmat(X[i, :2], self.M, 1) - p["obs"]) ** 2)
            cost += cfg["PURE_MPC_W_COL"] * ca.sum1(0.5 * (1 - ca.tanh(2 * (d2 - c2) / L ** 2)))
        opti.minimize(cost)
        opts = dict(cfg["IPOPT_OPTIONS"])
        opts.setdefault("expand", True)
        opti.solver("ipopt", opts)
        self.opti, self.X, self.U, self.p = opti, X, U, p
        self.Xp = self.Up = None
        self.left = 0
        self.solve_times, self.n_fail = [], 0

    def plan(self, goal, obs, others):
        r, cfg, H = self.robot, self.cfg, self.H
        x0 = np.concatenate([r.state[[0, 1, 3, 4]], r.accel[:2]])
        pts = np.asarray(obs, float).reshape(-1, 2)
        others_xy = np.array([o.state[:2] for o in others]).reshape(-1, 2)
        allp = pts
        far = x0[:2] + 1e4
        reach = cfg["VMAX"] * H * cfg["TIMESTEP"]
        margin = cfg["CORRIDOR_MARGIN"]
        dist = np.linalg.norm(allp - x0[:2], axis=1) if len(allp) else np.zeros(0)
        hard = allp[dist <= reach + margin]
        hd = dist[dist <= reach + margin]
        order = np.argsort(hd)[:self.Mh]
        Hp = np.tile(far, (self.Mh, 1))
        Hm = np.zeros(self.Mh)
        Hp[:len(order)] = hard[order]
        Hm[:len(order)] = np.minimum(margin, 0.999 * hd[order]) ** 2
        cost_pts = np.vstack([pts[1::2], others_xy])     # every other ray, as the original
        g = np.asarray(goal, float)[:2]
        dg = np.linalg.norm(g - x0[:2])
        if dg > reach:                                   # goal at most the horizon reach away
            g = x0[:2] + (g - x0[:2]) * (reach / dg)
        O = np.tile(far, (self.M, 1))
        O[:min(len(cost_pts), self.M)] = cost_pts[:self.M]
        o, p = self.opti, self.p
        o.set_value(p["x0"], x0.reshape(1, 6))
        o.set_value(p["goal"], g.reshape(1, 2))
        o.set_value(p["hard"], Hp)
        o.set_value(p["hmin"], Hm.reshape(-1, 1))
        o.set_value(p["obs"], O)
        Ds = cfg["UAV_SAFE_DISTANCE"]
        omin = np.zeros((max(self.n_others, 1), 1))
        for j, oth in enumerate(others):
            traj = np.asarray(oth.states_prediction, float)[:, :2]
            if len(traj) < H + 1:
                traj = np.vstack([traj, np.repeat(traj[-1:], H + 1 - len(traj), axis=0)])
            o.set_value(p["oth"][j], traj[:H + 1])
            omin[j, 0] = min(Ds, 0.999 * np.linalg.norm(oth.state[:2] - x0[:2])) ** 2
        o.set_value(p["omin"], omin)
        o.set_initial(self.X, np.tile(x0, (H + 1, 1)) if self.Xp is None else np.vstack([self.Xp[1:], self.Xp[-1:]]))
        o.set_initial(self.U, np.zeros((H, 2)) if self.Up is None else np.vstack([self.Up[1:], self.Up[-1:]]))
        t0 = time.perf_counter()
        try:
            sol = o.solve()
            X, U = sol.value(self.X), sol.value(self.U)
            ok = True
        except RuntimeError:
            ok = False
        self.solve_times.append(time.perf_counter() - t0)
        if ok:
            self.Xp, self.Up, self.left = X, U, H - 1
            r.states_prediction = np.column_stack(
                [X[:, :2], np.full(H + 1, r.state[2]), X[:, 2:4], np.zeros(H + 1)])
            r.mpc_fail_streak = 0
            u = U[0]
        else:
            self.n_fail += 1
            r.mpc_fail_streak = getattr(r, "mpc_fail_streak", 0) + 1
            if self.Up is not None and self.left > 0:
                u = self.Up[H - self.left]
                self.left -= 1
            else:
                v = r.state[3:5]
                u = -cfg["UMAX"] * v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-9 else np.zeros(2)
        return np.array([u[0], u[1], 0.0])
