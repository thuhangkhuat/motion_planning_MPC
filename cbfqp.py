"""
cbfqp.py — Baseline: decentralized safety barrier certificates (Wang, Ames,
Egerstedt, "Safety Barrier Certificates for Collisions-Free Multirobot
Systems", IEEE T-RO 33(3):661-674, 2017).

A safety filter on a nominal controller. Each UAV solves (eq. 12)
    u* = argmin |u - u_nom|^2
         s.t. -dp_ij^T u <= alpha_i / (alpha_i + alpha_j) * b_ij   for j in N_i
              -n_ik^T u  <= b_ik                                   for obstacles k
              |u|_inf <= alpha_i
with the pairwise ZCBF (eq. 6)
    h_ij = sqrt(2 (alpha_i + alpha_j) (|dp| - Ds)) + dp^T dv / |dp|
and the condition dh/dt + gamma h^3 >= 0 written as a linear constraint
(eq. 7); obstacles are agents without control (Sec. III, h_ik with the
agent's own alpha). Neighbours: within D_N (eq. 10). If the QP is
infeasible the UAV brakes at alpha_i (hybrid braking controller, eq. 17).
Deadlocks (Sec. VII, Algorithm 1): u = 0, v = 0 while u_nom != 0; an LP gives
the width of the feasible set and whether u = 0 lies on a vertex (type 1:
relax the left barrier constraints, compress the right ones) or an edge
(type 2: perturb u_nom by k_delta R(90 deg) u_nom); type 3: no perturbation.

Adaptations to this simulator (all in the comparison write-up):
  * nominal controller: PD to the goal with the goal velocity as feed-forward,
    u_nom = -k1 (p - r) - k2 (v - r_dot) (the paper's experiments use a PD to a
    fixed goal); goal r = the formation slot from the shared leader / slot
    layers (leader: predicted target), r_dot = target velocity estimate.
    There is no path planner: the paper has none, the filter alone avoids
    obstacles;
  * obstacles = the current LiDAR points, each a static agent with radius 0,
    kept CORRIDOR_MARGIN away (the margin of the proposed method);
    Ds = UAV_SAFE_DISTANCE between UAVs;
  * alpha = UMAX / sqrt(2) per axis (so |u| <= UMAX), beta = VMAX;
    gamma = CBF_QP_GAMMA (not given in the paper); deadlock constants
    k_left = 2, k_right = 0.5, k_delta = 0.5 (not given);
  * the simulator's acceleration lag (ACCEL_TAU) is not in the paper's double
    integrator, so the guarantee is approximate here (variant "cbf_qp"); the
    variant "cbf_qp_lag" (lag_aware=True) enforces every barrier both on the
    current state and on the state predicted T_d = ACCEL_TAU + TIMESTEP ahead
    with the current actual accelerations (p + v T_d + a T_d^2 / 2, v + a T_d),
    a standard delay-robust CBF: it moves the safety distance out by the closing displacement over the
    delay, which the plain filter ignores (it let the UAVs accelerate towards an
    obstacle until h was small, then the lagged braking came too late); the applied command is
    saturated to keep the speed <= VMAX (low-level controller, as in apfc.py);
  * the 2-variable QP is solved exactly as a projection onto the feasible
    polygon (the paper uses MATLAB quadprog), the LP with scipy linprog.
"""

import time

import numpy as np



class CBFQP:
    """One per UAV. plan() returns the filtered acceleration command."""

    def __init__(self, robot, cfg, lag_aware=False):
        self.robot, self.cfg = robot, cfg
        self.Td = (cfg["ACCEL_TAU"] + cfg["TIMESTEP"]) if lag_aware else 0.0
        self.alpha = cfg["UMAX"] / np.sqrt(2.0)
        self.beta = cfg["VMAX"]
        self.gamma = cfg["CBF_QP_GAMMA"]
        self.Ds = cfg["UAV_SAFE_DISTANCE"]
        self.Dob = cfg["CORRIDOR_MARGIN"]
        self.k1, self.k2 = cfg["CBF_QP_KP"], cfg["CBF_QP_KD"]
        a, g = self.alpha, self.gamma
        # neighbourhood radius (eq. 10) with identical agents
        self.DN = self.Ds + (np.cbrt(4 * a / g) + 2 * self.beta) ** 2 / (4 * a)
        self.solve_times, self.n_infeasible, self.n_deadlock = [], 0, 0

    # ---------- barrier rows ----------
    def _pair_row(self, dp, dv, alpha_sum, Ds, share):
        """Row (a, b) of -dp_hat^T u <= share * b for the ZCBF of eq. 6 / 7,
        written with the unit normal (eq. 7 divided by |dp|)."""
        d = np.linalg.norm(dp)
        if d < 1e-9:
            return None
        n = dp / d
        gap = max(d - Ds, 1e-6)
        root = np.sqrt(2 * alpha_sum * gap)
        nv = n @ dv
        h = root + nv
        b = (self.gamma * h ** 3 * self._k_scale
             + (dv @ dv - nv * nv) / d
             + alpha_sum * nv / root)
        return -n, share * b, h

    def _pred(self, robot):
        """Position / velocity T_d ahead with the current actual acceleration
        (T_d = 0 for the plain filter)."""
        p, v = robot.state[:2], robot.state[3:5]
        if self.Td == 0.0:
            return p, v
        a = np.asarray(getattr(robot, "accel", np.zeros(3)), float)[:2]
        T = self.Td
        return p + v * T + 0.5 * a * T * T, v + a * T

    def _rows(self, others, obs):
        rows = self._rows_at(others, obs, predicted=False)
        if self.Td > 0.0:
            # delay-robust: the barriers must hold now AND on the state T_d ahead;
            # with the predicted ones alone the lagged command chattered at the
            # boundary (pushing in while the actual acceleration still braked)
            # and the UAV crept into the margin
            rows += self._rows_at(others, obs, predicted=True)
        return rows

    def _rows_at(self, others, obs, predicted):
        r = self.robot
        if predicted:
            p, v = self._pred(r)
        else:
            p, v = r.state[:2], r.state[3:5]
        rows = []
        for o in others:
            po, vo = self._pred(o) if predicted else (o.state[:2], o.state[3:5])
            dp = p - po
            if np.linalg.norm(r.state[:2] - o.state[:2]) > self.DN:
                continue
            self._k_scale = 1.0
            row = self._pair_row(dp, v - vo, 2 * self.alpha, self.Ds, 0.5)
            if row is not None:
                rows.append(row)
        reach = self.Dob + (self.beta ** 2) / (2 * self.alpha) + self.beta
        if len(obs):
            d = np.linalg.norm(obs - p, axis=1)
            for q in obs[d <= reach]:
                self._k_scale = 1.0
                row = self._pair_row(p - q, v, self.alpha, self.Dob, 1.0)
                if row is not None:
                    rows.append(row)
        return rows

    def _solve_qp(self, u_nom, A, b):
        """Exact solution of the 2-variable QP: the projection of u_nom onto
        the convex polygon {A u <= b, |u|_inf <= alpha}. The optimum is u_nom,
        or its projection onto one boundary line, or a vertex (intersection of
        two lines); the feasible candidate closest to u_nom is returned, and
        an empty polygon is reported as infeasible. (CasADi's qrqp took up to
        8 s on the near-parallel rows of the points of one wall.)"""
        a = self.alpha
        A = np.vstack([np.asarray(A, float).reshape(-1, 2),
                       [[1, 0], [-1, 0], [0, 1], [0, -1]]])
        b = np.concatenate([np.asarray(b, float).ravel(), [a, a, a, a]])
        nrm = np.linalg.norm(A, axis=1)
        keep = nrm > 1e-12
        A, b = A[keep] / nrm[keep, None], b[keep] / nrm[keep]
        tol = 1e-7

        def feasible(U):
            return np.all(U @ A.T <= b + tol, axis=1)

        u0 = np.asarray(u_nom, float)
        if feasible(u0[None])[0]:
            return True, u0
        proj = u0[None] - (A @ u0 - b)[:, None] * A                        # onto each line
        cand = [proj[feasible(proj)]]
        i, j = np.triu_indices(len(b), 1)
        det = A[i, 0] * A[j, 1] - A[i, 1] * A[j, 0]
        ok = np.abs(det) > 1e-12
        i, j, det = i[ok], j[ok], det[ok]
        V = np.c_[(b[i] * A[j, 1] - b[j] * A[i, 1]) / det,
                  (A[i, 0] * b[j] - A[j, 0] * b[i]) / det]                 # vertices
        cand.append(V[feasible(V)])
        C = np.vstack(cand)
        if not len(C):
            return False, np.zeros(2)
        k = int(np.argmin(np.linalg.norm(C - u0, axis=1)))
        return True, C[k]

    # ---------- deadlock (Sec. VII) ----------
    def _deadlock_type(self, A, b):
        from scipy.optimize import linprog
        n = len(b)
        if n == 0:
            return 3, None
        res = linprog(c=[0, 0, 1], A_ub=np.c_[A, -np.ones(n)], b_ub=b,
                      bounds=[(-self.alpha, self.alpha)] * 2 + [(None, None)], method="highs")
        delta = res.x[2] if res.success else np.inf
        active = np.flatnonzero(np.abs(b) < 1e-3 * max(1.0, np.abs(b).max()))
        if delta < 0 and len(active) >= 2:
            return 1, active
        if delta < 0 and len(active) == 1:
            return 2, active
        return 3, active

    # ---------- one step ----------
    def plan(self, goal, goal_vel, obs, others):
        t0 = time.perf_counter()
        r, cfg = self.robot, self.cfg
        p, v = r.state[:2], r.state[3:5]
        u_nom = -self.k1 * (p - goal) - self.k2 * (v - goal_vel)
        rows = self._rows(others, np.asarray(obs, float).reshape(-1, 2))
        A = np.array([a for a, _, _ in rows]).reshape(-1, 2)
        b = np.array([bb for _, bb, _ in rows])
        ok, u = self._solve_qp(u_nom, A, b)
        # deadlock detection / resolution (Algorithm 1)
        if (ok and np.linalg.norm(u_nom) > 0.1 and np.linalg.norm(u) < 0.05
                and np.linalg.norm(v) < 0.05):
            kind, active = self._deadlock_type(A, b)
            if kind == 1:
                self.n_deadlock += 1
                left = np.array([u_nom[0] * a[1] - u_nom[1] * a[0] > 0 for a in A])
                b2 = b.copy()
                for i in active:                       # relax left (k > 1), compress right (k < 1)
                    b2[i] = b[i] * (2.0 if left[i] else 0.5) if b[i] > 0 else b[i] + (0.1 if left[i] else -0.1)
                ok2, u2 = self._solve_qp(u_nom, A, b2)
                if ok2:
                    u = u2
            elif kind == 2:
                self.n_deadlock += 1
                u_pert = u_nom + 0.5 * np.array([-u_nom[1], u_nom[0]])
                ok2, u2 = self._solve_qp(u_pert, A, b)
                if ok2:
                    u = u2
        if not ok:                                     # hybrid braking (eq. 17)
            self.n_infeasible += 1
            r.mpc_fail_streak = getattr(r, "mpc_fail_streak", 0) + 1
            sp = np.linalg.norm(v)
            u = -self.alpha * v / sp if sp > 1e-9 else np.zeros(2)
        else:
            r.mpc_fail_streak = 0
        x0 = np.concatenate([r.state[[0, 1, 3, 4]], r.accel[:2]])
        from apfc import speed_limit
        u = speed_limit(cfg, x0, u)
        # prediction for the other UAVs: constant velocity over the horizon
        H = cfg["HORIZON_LENGTH"]
        tt = np.arange(H + 1)[:, None] * cfg["TIMESTEP"]
        r.states_prediction = np.column_stack(
            [p + tt * v, np.full(H + 1, r.state[2]), np.repeat(v[None], H + 1, axis=0), np.zeros(H + 1)])
        self.solve_times.append(time.perf_counter() - t0)
        return np.array([u[0], u[1], 0.0])
