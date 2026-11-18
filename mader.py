"""
mader.py — Baseline: MADER (Tordesillas, How, "MADER: Trajectory Planner in
Multiagent and Dynamic Environments", IEEE T-RO 38(1):463-476, 2022,
arXiv:2010.11061; reference code github.com/mit-acl/mader).

Per UAV and replanning step, a nonconvex problem over a clamped uniform cubic
B-spline (m intervals of length dt_s, horizon T = m dt_s):
    min  sum_k |j_k|^2 dt_s + W_f |q_m - G|^2
    s.t. p(0), v(0), a(0) = current state;  v(T) = a(T) = 0 (ends at rest)
         |V_i| <= VMAX (velocity control points)
         |a + tau j| <= UMAX at the interval ends (the command, see below)
         for every interval k and obstacle polygon O_c:
             n_ck . q - d_ck >= margin  for the 4 control points of interval k
             n_ck . o - d_ck <= 0       for the vertices o of O_c,   |n_ck| <= 1
         for every interval k and other agent j (its committed trajectory):
             n_jk . q - d_jk >=  Ds / 2,  n_jk . o - d_jk <= -Ds / 2
The separating planes (n, d) are decision variables, as in MADER; a B-spline
interval lies in the convex hull of its control points, so a separated hull
means a separated trajectory. After the solve the result is checked against
every LiDAR point and the other agents' committed trajectories ("check");
only then it is committed. On a failed solve or check the UAV keeps flying its
last committed trajectory, which ends at rest (MADER's safety argument).

Adaptations to this simulator (all in the comparison write-up):
  * goal G = the point at MADER_RA metres along this code's JPS path to the
    formation slot (leader: predicted target, at the baselines' stand-off),
    as IMPC-OB's tractive point; MADER projects a terminal goal onto a sphere
    of radius R_a and finds an initial guess with its "octopus search" — here
    the initial guess is the previous trajectory (shifted) or the JPS path;
  * planar (fixed altitude), velocity / input bounds as norms (the proposed
    method's bounds), the trajectory confined to a box of MADER_BOX metres
    around the UAV (the corridor box of the proposed method);
  * outer polyhedra: the B-spline control polygon, not the MINVO basis (MADER
    offers B-spline / Bezier / MINVO; MINVO is less conservative);
  * obstacles = the current LiDAR points, split into short chunks along the
    scan (consecutive points, at most MADER_CHUNK_LEN long), each chunk's
    convex hull an obstacle polygon; the MADER_MAX_CHUNKS chunks nearest to the
    initial guess (within MADER_CHUNK_RANGE of it) enter the optimisation, the
    check uses every point;
    margin = CORRIDOR_MARGIN, the proposed method's;
  * agents: the hull of the other agent's control points over each interval
    of this agent's plan (its trajectory held at rest after its end);
    Ds = UAV_SAFE_DISTANCE;
  * synchronous simulator: the UAVs plan one after the other within a step
    and each commits at once, so a later UAV sees the earlier ones' new
    trajectories; the check-recheck of the asynchronous setting reduces to the
    check;
  * the UAV model of this simulator has an acceleration lag (ACCEL_TAU): the
    trajectory is flown by the command that reaches the planned acceleration
    at the end of the step (exact for the lag), plus a PD on the position /
    velocity error to the committed trajectory;
  * solved with IPOPT through CasADi (the reference uses NLopt); built once
    per number of obstacle polygons and re-solved with parameters; the
    reference stops the optimisation at a time budget, here at
    MADER_MAX_ITER iterations (reproducible across machines), and the UAV
    then keeps its committed trajectory.
"""

import time

import numpy as np
import casadi as ca
from scipy.interpolate import BSpline

from impc_ob import _hull

BIG = 1e4                    # relaxes an inactive constraint (local frame, |q| < 100 m)
W_SLACK = 1e4                # first interval: slack on the separation (the UAV may already be closer)
N_OTHER_VERTS = 5            # control points of another agent over one interval (<= 2 of its intervals)


def knots(m, dt):
    """Clamped uniform cubic knot vector, m intervals of length dt."""
    return np.r_[np.zeros(3), np.arange(m + 1) * dt, np.full(3, m * dt)]


def deriv_mats(t, nq):
    """MV (nq-1, nq), MA (nq-2, nq-1): velocity / acceleration control points."""
    MV = np.zeros((nq - 1, nq))
    for i in range(nq - 1):
        c = 3.0 / (t[i + 4] - t[i + 1])
        MV[i, i], MV[i, i + 1] = -c, c
    MA = np.zeros((nq - 2, nq - 1))
    for i in range(nq - 2):
        c = 2.0 / (t[i + 4] - t[i + 2])
        MA[i, i], MA[i, i + 1] = -c, c
    return MV, MA


def initial_points(t, v0, a0):
    """q0, q1, q2 (local frame, p0 = 0) from the initial velocity / acceleration."""
    q0 = np.zeros(2)
    q1 = q0 + np.asarray(v0) * (t[4] - t[1]) / 3.0
    V1 = np.asarray(v0) + np.asarray(a0) * (t[4] - t[2]) / 2.0
    q2 = q1 + V1 * (t[5] - t[2]) / 3.0
    return q0, q1, q2


# ============================================================
# The optimisation problem (built once, parameters per solve)
# ============================================================
class MADERProblem:

    def __init__(self, n_others, cfg, n_chunks):
        m, D = cfg["MADER_N_SEG"], cfg["MADER_SEG_DT"]
        self.m, self.D, self.t = m, D, knots(m, D)
        self.nq = m + 3
        self.M, self.K = n_chunks, cfg["MADER_CHUNK_VERTS"]
        self.n_oth = max(n_others, 1)
        MV, MA = deriv_mats(self.t, self.nq)
        tau, margin, Ds = cfg["ACCEL_TAU"], cfg["CORRIDOR_MARGIN"], cfg["UAV_SAFE_DISTANCE"]
        box = cfg["MADER_BOX"]

        opti = ca.Opti()
        p = {"v0": opti.parameter(1, 2), "a0": opti.parameter(1, 2), "G": opti.parameter(1, 2),
             "obs": [opti.parameter(self.K, 2) for _ in range(self.M)],
             "act_o": opti.parameter(self.M, 1),
             "oth": [[opti.parameter(N_OTHER_VERTS, 2) for _ in range(m)] for _ in range(self.n_oth)],
             "act_a": opti.parameter(self.n_oth, 1)}
        Qf = opti.variable(m - 2, 2)                       # q3 .. q_m
        # q0, q1, q2 from the initial state (local frame: q0 = 0), linear in v0, a0
        t = self.t
        q0 = ca.MX.zeros(1, 2)
        q1 = q0 + p["v0"] * (t[4] - t[1]) / 3.0
        q2 = q1 + (p["v0"] + p["a0"] * (t[4] - t[2]) / 2.0) * (t[5] - t[2]) / 3.0
        qm = Qf[m - 3, :]
        Q = ca.vertcat(q0, q1, q2, Qf, qm, qm)            # q_{m+1} = q_{m+2} = q_m: ends at rest
        V = ca.mtimes(ca.DM(MV), Q)
        A = ca.mtimes(ca.DM(MA), V)

        No = opti.variable(self.M * m, 2)
        Do = opti.variable(self.M * m, 1)
        So = opti.variable(self.M, 1)
        Na = opti.variable(self.n_oth * m, 2)
        Da = opti.variable(self.n_oth * m, 1)
        Sa = opti.variable(self.n_oth, 1)
        opti.subject_to(So >= 0)
        opti.subject_to(Sa >= 0)

        for c in range(self.M):
            off = BIG * (1 - p["act_o"][c])
            for k in range(m):
                i = c * m + k
                n, d = No[i, :], Do[i]
                s = So[c] if k == 0 else 0
                opti.subject_to(ca.mtimes(Q[k:k + 4, :], n.T) - d >= margin - off - s)
                opti.subject_to(ca.mtimes(p["obs"][c], n.T) - d <= off)
                opti.subject_to(ca.sumsqr(n) <= 1)
        for j in range(self.n_oth):
            off = BIG * (1 - p["act_a"][j])
            for k in range(m):
                i = j * m + k
                n, d = Na[i, :], Da[i]
                s = Sa[j] if k == 0 else 0
                opti.subject_to(ca.mtimes(Q[k:k + 4, :], n.T) - d >= Ds / 2 - off - s)
                opti.subject_to(ca.mtimes(p["oth"][j][k], n.T) - d <= -Ds / 2 + off)
                opti.subject_to(ca.sumsqr(n) <= 1)

        for i in range(2, m):                              # V0, V1 fixed by the state; V_m = V_{m+1} = 0
            opti.subject_to(ca.sumsqr(V[i, :]) <= cfg["VMAX"] ** 2)
        cost = 0
        for k in range(m):
            J = (A[k + 1, :] - A[k, :]) / D
            opti.subject_to(ca.sumsqr(A[k, :] + tau * J) <= cfg["UMAX"] ** 2)
            opti.subject_to(ca.sumsqr(A[k + 1, :] + tau * J) <= cfg["UMAX"] ** 2)
            cost += ca.sumsqr(J) * D
        opti.subject_to(ca.vec(Qf) <= box)
        opti.subject_to(ca.vec(Qf) >= -box)
        cost += cfg["MADER_W_FINAL"] * ca.sumsqr(qm - p["G"])
        cost += W_SLACK * (ca.sum1(So) + ca.sum1(Sa) + ca.sumsqr(So) + ca.sumsqr(Sa))
        opti.minimize(cost)
        opts = dict(cfg["IPOPT_OPTIONS"])
        opts["ipopt.max_iter"] = cfg["MADER_MAX_ITER"]    # MADER's optimisation time budget
        opts.setdefault("expand", True)
        opti.solver("ipopt", opts)
        self.opti, self.p, self.Qf = opti, p, Qf
        self.No, self.Do, self.So, self.Na, self.Da, self.Sa = No, Do, So, Na, Da, Sa
        self.solve_times = []

    def solve(self, v0, a0, G, obs, act_o, oth, act_a, Q_init, No0, Do0, Na0, Da0):
        o, p = self.opti, self.p
        o.set_value(p["v0"], np.reshape(v0, (1, 2)))
        o.set_value(p["a0"], np.reshape(a0, (1, 2)))
        o.set_value(p["G"], np.reshape(G, (1, 2)))
        for c in range(self.M):
            o.set_value(p["obs"][c], obs[c])
        o.set_value(p["act_o"], np.reshape(act_o, (-1, 1)))
        for j in range(self.n_oth):
            for k in range(self.m):
                o.set_value(p["oth"][j][k], oth[j][k])
        o.set_value(p["act_a"], np.reshape(act_a, (-1, 1)))
        o.set_initial(self.Qf, Q_init[3:self.m + 1])
        o.set_initial(self.No, No0)
        o.set_initial(self.Do, Do0)
        o.set_initial(self.Na, Na0)
        o.set_initial(self.Da, Da0)
        o.set_initial(self.So, 0)
        o.set_initial(self.Sa, 0)
        t0 = time.perf_counter()
        try:
            sol = o.solve()
            Qf = np.asarray(sol.value(self.Qf)).reshape(-1, 2)
            ok = True
        except RuntimeError:
            Qf, ok = None, False
        self.last_status = o.stats().get("return_status", "?")
        self.last_iter = o.stats().get("iter_count", -1)
        self.solve_times.append(time.perf_counter() - t0)
        return ok, Qf


# ============================================================
# Per-UAV planner
# ============================================================
class MADER:
    """One per UAV. plan() returns the command for this step and leaves
    self.traj = the committed trajectory (read by the other UAVs) and
    robot.states_prediction = that trajectory sampled at the control period."""

    def __init__(self, robot, n_others, cfg):
        self.robot, self.cfg = robot, cfg
        self.n_others = n_others
        self._probs = {}                  # problem per number of obstacle polygons (built when needed)
        self.m, self.D = cfg["MADER_N_SEG"], cfg["MADER_SEG_DT"]
        self.t = knots(self.m, self.D)
        self.K = cfg["MADER_CHUNK_VERTS"]
        self.solve_times = []
        self.T = self.m * self.D
        self.traj = None                  # committed: {"t0": time, "Q": (m+3, 2) world frame}
        self.n_solve = self.n_fail = self.n_check_fail = self.n_dropped = 0
        self._prev_planes = None

    # ---------- trajectory evaluation ----------
    def _spl(self, Q):
        return BSpline(self.t, np.asarray(Q, float), 3, extrapolate=False)

    def eval(self, traj, s, nu=0):
        """Derivative nu of a committed trajectory at absolute times s (held at rest after T)."""
        s = np.atleast_1d(np.asarray(s, float))
        tau = np.clip(s - traj["t0"], 0.0, self.T)
        out = traj.setdefault("_spl", {})
        if nu not in out:
            out[nu] = self._spl(traj["Q"]).derivative(nu) if nu else self._spl(traj["Q"])
        y = out[nu](tau)
        if nu:
            y[s - traj["t0"] >= self.T] = 0.0
        return y

    def _segment_points(self, traj, s0, s1):
        """Control points of traj over [s0, s1] (absolute), N_OTHER_VERTS rows."""
        a, b = s0 - traj["t0"], s1 - traj["t0"]
        Q = traj["Q"]
        if a >= self.T:
            P = Q[-1:]
        else:
            k0 = int(np.clip(np.floor(max(a, 0.0) / self.D), 0, self.m - 1))
            k1 = int(np.clip(np.floor((min(b, self.T) - 1e-9) / self.D), k0, self.m - 1))
            P = Q[k0:k1 + 4]
        P = P[:N_OTHER_VERTS]
        return np.vstack([P, np.repeat(P[-1:], N_OTHER_VERTS - len(P), axis=0)])

    # ---------- inputs ----------
    def _local_goal(self, goal, path, p0):
        Ra = self.cfg["MADER_RA"]
        P = [p0]
        if path is not None and len(path):
            P += list(np.asarray(path, float)[1:, :2])
        else:
            P.append(np.asarray(goal, float)[:2])
        P = np.asarray(P)
        s = 0.0
        for a, b in zip(P[:-1], P[1:]):
            L = np.linalg.norm(b - a)
            if s + L >= Ra:
                return a + (b - a) * (Ra - s) / max(L, 1e-9)
            s += L
        return P[-1]

    def _problem(self, n):
        """The problem with the smallest padded size >= n obstacle polygons."""
        cap = self.cfg["MADER_MAX_CHUNKS"]
        size = min([s for s in (8, 16, 32, 48, 64) if s >= min(n, cap)] + [cap])
        if size not in self._probs:
            self._probs[size] = MADERProblem(self.n_others, self.cfg, size)
        return self._probs[size]

    def _chunks(self, obs_local):
        """LiDAR points (scan order) -> list of convex polygons (<= K vertices)."""
        cl, Lmax, K = [], self.cfg["MADER_CHUNK_LEN"], self.K
        cur = []
        for q in obs_local:
            if cur and (np.linalg.norm(q - cur[-1]) > 1.0 or np.linalg.norm(q - cur[0]) > Lmax):
                cl.append(np.array(cur))
                cur = []
            cur.append(q)
        if cur:
            cl.append(np.array(cur))
        out = []

        def add(P):
            H = _hull(P)
            if len(H) <= K or len(P) <= 2:
                out.append(H[:K])
            else:
                h = len(P) // 2
                add(P[:h + 1])
                add(P[h:])
        for P in cl:
            add(P)
        return out

    def _guess(self, v0, a0, G, now):
        """Initial control points (local frame): the previous trajectory shifted, else towards G."""
        q0, q1, q2 = initial_points(self.t, v0, a0)
        nq = self.m + 3
        grev = np.array([(self.t[i + 1] + self.t[i + 2] + self.t[i + 3]) / 3.0 for i in range(nq)])
        p0 = self.robot.state[:2]
        if self.traj is not None:
            Q = self.eval(self.traj, now + grev) - p0
            far = np.linalg.norm(Q[-1] - G)
            if far > 1e-3:                                 # pull the tail towards the new goal
                w = (grev / self.T) ** 2
                Q = Q + w[:, None] * (G - Q[-1])
        else:
            s = grev / self.T
            w = s * s * (3 - 2 * s)
            Q = w[:, None] * G[None, :]
        Q[0], Q[1], Q[2] = q0, q1, q2
        Q[-2] = Q[-1] = Q[-3]
        return Q

    # ---------- one step ----------
    def plan(self, goal, path, obs, others, now):
        """goal: planning goal (2,); path: JPS path (N, 2) from the UAV;
        obs: LiDAR points (M, 2) in scan order; others: the other robots."""
        r, cfg = self.robot, self.cfg
        self.n_solve += 1
        p0 = r.state[:2].copy()
        v0, a0 = r.state[3:5].copy(), r.accel[:2].copy()
        G = self._local_goal(goal, path, p0) - p0
        box, margin = cfg["MADER_BOX"], cfg["CORRIDOR_MARGIN"]
        obs = np.asarray(obs, float).reshape(-1, 2)
        loc = obs - p0
        near = np.all(np.abs(loc) <= box + margin + cfg["MADER_CHUNK_LEN"], axis=1)
        loc = loc[near]
        Qg = self._guess(v0, a0, G, now)

        # obstacle polygons within MADER_CHUNK_RANGE of the initial guess, nearest
        # first, at most MADER_MAX_CHUNKS (the check below uses every point)
        polys = self._chunks(loc) if len(loc) else []
        if polys:
            curve = self._spl(Qg)(np.linspace(0, self.T, 41))
            dist = np.array([np.min(np.linalg.norm(P[:, None, :] - curve[None], axis=2)) for P in polys])
            order = [i for i in np.argsort(dist) if dist[i] <= cfg["MADER_CHUNK_RANGE"]]
            self.n_dropped += max(0, len(order) - cfg["MADER_MAX_CHUNKS"])
            polys = [polys[i] for i in order[:cfg["MADER_MAX_CHUNKS"]]]
        prob = self._problem(len(polys))
        self.last_prob = prob
        obs_p = np.full((prob.M, prob.K, 2), 1e3)
        act_o = np.zeros(prob.M)
        No0 = np.tile([1.0, 0.0], (prob.M * self.m, 1))
        Do0 = np.full((prob.M * self.m, 1), -BIG / 2)
        for c, P in enumerate(polys):
            obs_p[c, :len(P)] = P
            obs_p[c, len(P):] = P[-1]
            act_o[c] = 1
            oc = P.mean(axis=0)
            for k in range(self.m):
                qc = Qg[k:k + 4].mean(axis=0)
                n = qc - oc
                n = n / max(np.linalg.norm(n), 1e-9)
                No0[c * self.m + k] = n
                Do0[c * self.m + k] = np.max(P @ n)

        # other agents: their committed trajectories over each interval of this plan
        n_oth = prob.n_oth
        oth = [[np.full((N_OTHER_VERTS, 2), 1e3) for _ in range(self.m)] for _ in range(n_oth)]
        act_a = np.zeros(n_oth)
        Na0 = np.tile([1.0, 0.0], (n_oth * self.m, 1))
        Da0 = np.full((n_oth * self.m, 1), -BIG / 2)
        reach = 2 * box + cfg["UAV_SAFE_DISTANCE"]
        for j, o in enumerate(others):
            if np.linalg.norm(o.state[:2] - p0) > reach:
                continue
            tr = getattr(getattr(o, "mader", None), "traj", None)
            act_a[j] = 1
            for k in range(self.m):
                if tr is None:
                    P = np.repeat(o.state[None, :2], N_OTHER_VERTS, axis=0)
                else:
                    P = self._segment_points(tr, now + k * self.D, now + (k + 1) * self.D)
                P = P - p0
                oth[j][k] = P
                oc, qc = P.mean(axis=0), Qg[k:k + 4].mean(axis=0)
                n = qc - oc
                n = n / max(np.linalg.norm(n), 1e-9)
                Na0[j * self.m + k] = n
                Da0[j * self.m + k] = np.max(P @ n) + cfg["UAV_SAFE_DISTANCE"] / 2

        t_s = time.perf_counter()
        ok, Qf = prob.solve(v0, a0, G, obs_p, act_o, oth, act_a, Qg, No0, Do0, Na0, Da0)
        self.solve_times.append(time.perf_counter() - t_s)
        if ok:
            q0, q1, q2 = initial_points(self.t, v0, a0)
            Q = np.vstack([q0, q1, q2, Qf, Qf[-1:], Qf[-1:]]) + p0
            new = {"t0": now, "Q": Q}
            if self._check(new, obs, others, now):
                self.traj = new
            else:
                self.n_check_fail += 1
                ok = False
        else:
            self.n_fail += 1
        if not ok:
            r.mpc_fail_streak = getattr(r, "mpc_fail_streak", 0) + 1
            if not hasattr(r, "infeasible_log"):
                r.infeasible_log = []
            r.infeasible_log.append({"t": now, "kind": "mader"})
        else:
            r.mpc_fail_streak = 0
        return self.control(now)

    def _check(self, traj, obs, others, now):
        """MADER's check: the trajectory itself against every LiDAR point and
        the other agents' committed trajectories (sampled), and the bounds."""
        cfg = self.cfg
        s = now + np.arange(0.0, self.T + 1e-9, 0.02)
        P = self.eval(traj, s)
        if len(obs):
            p0 = self.robot.state[:2]
            d_now = np.linalg.norm(obs - p0, axis=1).min()
            thr = min(cfg["CORRIDOR_MARGIN"], d_now) - 0.02
            near = obs[np.linalg.norm(obs - p0, axis=1) <= 2 * cfg["MADER_BOX"]]
            for i in range(0, len(P), 50):
                if len(near) and np.min(np.linalg.norm(P[i:i + 50, None, :] - near[None], axis=2)) < thr:
                    return False
        Ds = cfg["UAV_SAFE_DISTANCE"]
        for o in others:
            tr = getattr(getattr(o, "mader", None), "traj", None)
            Po = self.eval(tr, s) if tr is not None else np.repeat(o.state[None, :2], len(s), axis=0)
            d = np.linalg.norm(P - Po, axis=1)
            thr = min(Ds, d[0]) - 0.05
            if d.min() < thr:
                return False
        return True

    def control(self, now):
        """Command that flies the committed trajectory: reaches its acceleration
        at the end of the step through the lag, plus a PD on the tracking error;
        publishes the trajectory as states_prediction."""
        r, cfg = self.robot, self.cfg
        dt, tau, H = cfg["TIMESTEP"], cfg["ACCEL_TAU"], cfg["HORIZON_LENGTH"]
        p, v, a = r.state[:2], r.state[3:5], r.accel[:2]
        if self.traj is None:                              # nothing committed yet: brake
            u = -v / (tau + dt)
            n = np.linalg.norm(u)
            u = u * (cfg["UMAX"] / n) if n > cfg["UMAX"] else u
            return np.array([u[0], u[1], 0.0])
        s = now + np.arange(H + 1) * dt
        P, V = self.eval(self.traj, s), self.eval(self.traj, s, 1)
        r.states_prediction = np.column_stack([P, np.full(H + 1, r.state[2]), V, np.zeros(H + 1)])
        a_end = self.eval(self.traj, now + dt, 2)[0]
        a_end = a_end + 1.0 * (P[0] - p) + 2.0 * (V[0] - v)
        e = np.exp(-dt / tau) if tau > 0 else 0.0
        u = (a_end - a * e) / (1.0 - e)
        n = np.linalg.norm(u)
        u = u * (cfg["UMAX"] / n) if n > cfg["UMAX"] else u
        return np.array([u[0], u[1], 0.0])
