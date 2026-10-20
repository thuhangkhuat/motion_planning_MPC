"""
impc_ob.py — Baseline: IMPC-OB (Chen, Wang, Guo, Li, "Multi-Robot Trajectory
Planning with Feasibility Guarantee and Deadlock Resolution: An Obstacle-Dense
Environment", IEEE RA-L 2023, arXiv:2210.04231; reference code
github.com/PKU-MACDLab/IMPC-OB, files uav.py, run.py, inter_avoid.py,
obstacle_corridor.py).

Per UAV and step, a convex QCQP over the horizon K:
    min  sum_{k>=c} Q_tar |p_k - p_tractive|^2 + sum_k (k/K)^2 |p_k - p_{k-1}|^2
         + sum_j rho_ij (w_ij - eps)^2 + 2e5 |s|^2
    s.t. dynamics, |v_k| <= VMAX, |u_k| <= UMAX, v_K = a_K = 0 (terminal rest),
         inter-robot MBVC:    a_kj . p_k >= b_kj            (k < K)
         warning band (WB):   a_Kj . p_K >= b_Kj + w_j,  0 <= w_j <= eps
         obstacle corridor:   A_k p_k <= b_k + s,  0 <= s <= buffer
a_kj, b_kj: bisecting plane of the predetermined trajectories (previous plan,
shifted) of the two robots, shifted by r_min / 2. The obstacle corridor comes
from the predetermined trajectory extended by the tractive point (farthest
reference-path vertex visible from its end), split into segments whose convex
hulls are obstacle free, each separated from the obstacles by planes.
Deadlocks: terminal-overlap detection, growing eta, dynamic priority (1 at
goal, 2 normal, 3 winner) and the right-hand rule through rho_ij.

Adaptations to this simulator (all in the comparison write-up):
  * goal = the formation slot from the shared leader / slot layers (leader:
    predicted target); reference path = this code's JPS path, not ABIT*;
  * obstacles = the current LiDAR points (not a known convex map); a point is
    separated from a segment by the max-margin plane, shifted by the corridor
    margin (CORRIDOR_MARGIN, the same as the proposed method);
  * the UAV model of this simulator (exact discretisation with acceleration
    lag, dynamics.py) so the plan is what the UAV flies; the terminal rest then
    also needs a_K = 0;
  * K = HORIZON_LENGTH, h = TIMESTEP, r_min = UAV_SAFE_DISTANCE; length
    thresholds of the reference code (r_a = 0.3 m robots) scaled by
    L = r_min / (2 * 0.3);
  * solved with IPOPT through CasADi (the reference uses CVXOPT coneqp; no
    conic solver is installed here). The problem is convex, so the optimum is
    the same; the problem is built once and re-solved with parameters;
  * priority 1 ("reached the goal") is re-evaluated every step, since the
    slots move (the reference sets it once, for fixed goals).
"""

import numpy as np
import casadi as ca

import dynamics

BIG = 1e6


# ============================================================
# Geometry on point sets (LiDAR points are the obstacles)
# ============================================================
def _hull(V):
    """Convex hull (counter-clockwise, no repeated point) of 2D points; works
    for 1 or 2 points and for collinear sets (returns the extreme points)."""
    P = np.unique(np.round(np.asarray(V, float), 9), axis=0)
    if len(P) <= 2:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in P[::-1]:
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


def _closest_on_hull(H, Q):
    """Closest point of the convex polygon H (hull vertices) to each point of
    Q, and the distance (0 inside)."""
    Q = np.asarray(Q, float).reshape(-1, 2)
    if len(H) == 1:
        C = np.repeat(H, len(Q), axis=0)
        return C, np.linalg.norm(Q - C, axis=1)
    best_d = np.full(len(Q), np.inf)
    best_c = np.zeros_like(Q)
    n = len(H)
    edges = [(H[i], H[(i + 1) % n]) for i in range(n if n > 2 else 1)]
    for a, b in edges:
        ab = b - a
        t = np.clip(((Q - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
        C = a + t[:, None] * ab
        d = np.linalg.norm(Q - C, axis=1)
        m = d < best_d
        best_d[m], best_c[m] = d[m], C[m]
    if n >= 3:                                            # inside the polygon
        inside = np.ones(len(Q), bool)
        for a, b in edges:
            inside &= (b[0] - a[0]) * (Q[:, 1] - a[1]) - (b[1] - a[1]) * (Q[:, 0] - a[0]) >= -1e-9
        best_d[inside] = 0.0
        best_c[inside] = Q[inside]
    return best_c, best_d


def _segments(points_aug, obs, margin):
    """Reference code get_segment_list: from the last point (the tractive
    point) backwards, grow a segment while the convex hull of its points stays
    farther than `margin` from every obstacle point; the next segment starts at
    the last point of the previous one. Returns lists of indices into
    points_aug, or None if a single step already collides."""
    n = len(points_aug)
    segs, begin = [], n - 1
    for _ in range(n + 1):
        seg = [begin]
        for pt in range(begin - 1, -1, -1):
            H = _hull(points_aug[pt:begin + 1])
            if len(obs) and _closest_on_hull(H, obs)[1].min() < margin:
                if pt == begin - 1:
                    return None                           # p_pt -> p_begin itself collides
                begin = pt + 1
                segs.append(seg)
                break
            seg.append(pt)
            if pt == 0:
                segs.append(seg)
        else:
            if not segs or segs[-1] is not seg:
                segs.append(seg)
        if segs[-1][-1] == 0:
            return segs
    return None


def _planes(H, obs, margin, max_faces):
    """Reference code get_polyhedron with points: obstacles from the nearest
    on; a point already beyond a plane (by the margin) is skipped, otherwise
    the max-margin plane between the segment hull and the point is added,
    shifted by `margin`. Returns (A, b) with A p <= b."""
    if not len(obs):
        return np.zeros((0, 2)), np.zeros(0)
    C, d = _closest_on_hull(H, obs)
    A, b = [], []
    for i in np.argsort(d):
        q = obs[i]
        if any(a @ q >= bb + margin - 1e-9 for a, bb in zip(A, b)):
            continue
        if d[i] < 1e-9:
            continue                                      # on the hull: cannot separate
        n = (q - C[i]) / d[i]
        A.append(n)
        b.append(n @ q - margin)
        if len(A) >= max_faces:
            break
    return np.array(A).reshape(-1, 2), np.array(b)


# ============================================================
# The QCQP, built once per UAV and re-solved with parameters
# ============================================================
class IMPCOBProblem:

    def __init__(self, n_others, n_faces, cfg):
        self.cfg, self.n_others, self.n_faces = cfg, n_others, n_faces
        K = cfg["HORIZON_LENGTH"]
        dt = cfg["TIMESTEP"]
        opti = ca.Opti()
        X = opti.variable(K + 1, 6)                       # [px, py, vx, vy, ax, ay]
        U = opti.variable(K, 2)
        W = opti.variable(max(n_others, 1), 1)            # warning-band slack w_ij
        S = opti.variable(K, n_faces)                     # obstacle slack (<= buffer)
        p = {"x0": opti.parameter(1, 6),
             "G": opti.parameter(1, 2),                   # tractive point
             "sig": opti.parameter(K, 1),                 # Q_tar on step k (0 before cost_index)
             "A": [opti.parameter(n_faces, 2) for _ in range(K)],
             "b": [opti.parameter(n_faces, 1) for _ in range(K)],
             "Aij": [opti.parameter(K, 2) for _ in range(n_others)],
             "bij": [opti.parameter(K, 1) for _ in range(n_others)],
             "rho": opti.parameter(max(n_others, 1), 1),
             "on": opti.parameter(max(n_others, 1), 1)}   # 1 = constraint active
        Ad, Bd = dynamics.model(dt, cfg["D_FRAC"], cfg["ACCEL_TAU"], 2)
        opti.subject_to(X[0, :] == p["x0"])
        for i in range(K):
            opti.subject_to(X[i + 1, :] == ca.mtimes(X[i, :], Ad.T) + ca.mtimes(U[i, :], Bd.T))
            opti.subject_to(ca.sumsqr(X[i + 1, 2:4]) <= cfg["VMAX"] ** 2)
            opti.subject_to(ca.sumsqr(U[i, :]) <= cfg["UMAX"] ** 2)
        opti.subject_to(X[K, 2:6] == 0)                   # terminal rest
        eps, buf = cfg["IMPC_EPSILON"], cfg["IMPC_BUFFER"]
        for i in range(K):
            opti.subject_to(ca.mtimes(p["A"][i], X[i + 1, :2].T) <= p["b"][i] + S[i, :].T)
        opti.subject_to(ca.vec(S) >= 0)
        opti.subject_to(ca.vec(S) <= buf)
        for j in range(n_others):
            for k in range(K):
                lhs = ca.mtimes(p["Aij"][j][k, :], X[k + 1, :2].T) - p["bij"][j][k]
                if k == K - 1:
                    lhs = lhs - W[j]
                opti.subject_to(p["on"][j] * lhs >= 0)
        opti.subject_to(W >= 0)
        opti.subject_to(W <= eps)
        cost = 0
        for k in range(K):
            cost += p["sig"][k] * ca.sumsqr(X[k + 1, :2] - p["G"])
        for k in range(1, K):
            cost += (k / K) ** 2 * ca.sumsqr(X[k + 1, :2] - X[k, :2])
        cost += ca.sum1(p["rho"] * (W - eps) ** 2)
        cost += 2e5 * ca.sumsqr(S)
        opti.minimize(cost)
        opts = dict(cfg["IPOPT_OPTIONS"])
        opts.setdefault("expand", True)
        opti.solver("ipopt", opts)
        self.opti, self.X, self.U, self.W, self.S, self.p = opti, X, U, W, S, p
        self.K = K
        self.solve_times, self.n_fail = [], 0

    def solve(self, x0, G, sig, planes, Aij, bij, on, rho, X_init, U_init):
        import time
        t0 = time.perf_counter()
        o, p, K, F = self.opti, self.p, self.K, self.n_faces
        o.set_value(p["x0"], np.asarray(x0, float).reshape(1, 6))
        o.set_value(p["G"], np.asarray(G, float).reshape(1, 2))
        o.set_value(p["sig"], np.asarray(sig, float).reshape(K, 1))
        for k in range(K):
            A, b = planes[k]
            Ap, bp = np.zeros((F, 2)), np.full((F, 1), BIG)
            Ap[:len(A)], bp[:len(b), 0] = A, b
            o.set_value(p["A"][k], Ap)
            o.set_value(p["b"][k], bp)
        for j in range(self.n_others):
            o.set_value(p["Aij"][j], Aij[j])
            o.set_value(p["bij"][j], bij[j].reshape(K, 1))
        n = max(self.n_others, 1)
        o.set_value(p["on"], np.asarray(on, float).reshape(-1, 1)[:n] if self.n_others else np.zeros((1, 1)))
        o.set_value(p["rho"], np.asarray(rho, float).reshape(-1, 1)[:n] if self.n_others else np.zeros((1, 1)))
        o.set_initial(self.X, X_init)
        o.set_initial(self.U, U_init)
        o.set_initial(self.W, self.cfg["IMPC_EPSILON"])
        o.set_initial(self.S, 0)
        try:
            sol = o.solve()
            out = True, sol.value(self.X), sol.value(self.U), np.atleast_1d(sol.value(self.W))
        except RuntimeError:
            self.n_fail += 1
            out = False, None, None, None
        self.solve_times.append(time.perf_counter() - t0)
        return out


# ============================================================
# Per-UAV planner state (tractive point, corridor, deadlock resolution)
# ============================================================
class IMPCOB:
    """One per UAV. plan() returns the control to apply this step and leaves
    robot.states_prediction = the new predetermined trajectory."""

    def __init__(self, robot, n_others, cfg):
        self.robot, self.cfg = robot, cfg
        self.K = cfg["HORIZON_LENGTH"]
        self.L = cfg["UAV_SAFE_DISTANCE"] / (2 * 0.3)    # length scale vs the reference code
        self.eps = cfg["IMPC_EPSILON"]
        self.rho0 = 10.0
        self.max_faces = 24                               # per step: up to 2 segments x 10 planes + border
        self.n_faces = 8                                  # padded size; the problem grows when needed
        self.prob = IMPCOBProblem(n_others, self.n_faces, cfg)
        self.n_others = n_others
        # deadlock state (reference uav.py)
        self.term_overlap = False
        self.term_overlap_again = False
        self.term_last_p = None
        self.term_index = 0
        self.eta = 1.0
        self.W = 0.4 * self.eps * np.ones(max(n_others, 1))
        self.E = 0.4 * self.eps * np.ones(max(n_others, 1))
        self.priority = 2
        self.contest = False
        self.distance = np.inf
        self.cost_index = self.K
        self.U = None                                     # last plan (fallback: fly its rest)
        self.left = 0
        self.n_corridor_fail = 0

    # ---------- inputs from the shared layers ----------
    def _tractive_point(self, pbar_K, path, gmap):
        """Farthest reference-path vertex visible (inflated grid) from the end
        of the predetermined trajectory; that end itself if none is."""
        if path is None or len(path) == 0:
            return pbar_K.copy()
        for q in np.asarray(path, float)[::-1, :2]:
            if gmap is None or gmap.segment_clear(pbar_K, q):
                return q.copy()
        return pbar_K.copy()

    def _corridors(self, pbar, G, obs):
        """Planes for each step k = 1..K (plus the map border)."""
        cfg, K = self.cfg, self.K
        margin = cfg["CORRIDOR_MARGIN"]
        reach = cfg["VMAX"] * K * cfg["TIMESTEP"] + margin
        if len(obs):
            obs = obs[np.linalg.norm(obs - pbar[0], axis=1) <= reach + np.linalg.norm(G - pbar[0])]
        aug = np.vstack([pbar[1:K + 1], G[None]])        # p_1..p_K, tractive point
        segs = _segments(aug, obs, margin)
        if segs is None:
            return None
        segs[0] = segs[0][1:]                             # the tractive point is no decision variable
        x0, y0, x1, y1 = cfg["WORLD_BOUNDS"]
        R = cfg["ROBOT_RADIUS"]
        border_A = np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]])
        border_b = np.array([x1 - R, -(x0 + R), y1 - R, -(y0 + R)])
        # a point shared by two segments gets the planes of both (reference code)
        acc = [[border_A], [border_b]]
        planes = [([border_A], [border_b]) for _ in range(K)]
        for seg in segs:
            if not seg:
                continue
            A, b = _planes(_hull(aug[sorted(seg)]), obs, margin, 10)
            for k in seg:
                if k < K:
                    planes[k][0].append(A)
                    planes[k][1].append(b)
        out = []
        for As, bs in planes:
            A, b = np.vstack(As), np.concatenate(bs)
            out.append((A[:self.max_faces], b[:self.max_faces]))
        need = max(len(b) for _, b in out)
        if need > self.n_faces:                           # rebuild once with more faces
            self.n_faces = 8 * int(np.ceil(need / 8))
            times, fails = self.prob.solve_times, self.prob.n_fail
            self.prob = IMPCOBProblem(self.n_others, self.n_faces, self.cfg)
            self.prob.solve_times, self.prob.n_fail = times, fails
        return out

    def _inter(self, pbar, others_pbar, others_prio, target):
        """MBVC / MBVC-WB planes and rho_ij (reference inter_avoid.py)."""
        K, eps, rmin = self.K, self.eps, self.cfg["UAV_SAFE_DISTANCE"]
        Aij, bij, on, rho = [], [], [], []
        for j, Pj in enumerate(others_pbar):
            A = np.zeros((K, 2))
            b = np.zeros(K)
            ok = True
            for k in range(1, K + 1):
                pi, pj = pbar[k], Pj[min(k, len(Pj) - 1), :2]
                d = np.linalg.norm(pi - pj)
                if d < 1e-9:
                    ok = False
                    break
                a = (pi - pj) / d
                A[k - 1], b[k - 1] = a, a @ (pi + pj) / 2 + rmin / 2
            Aij.append(A)
            bij.append(b)
            on.append(1.0 if ok else 0.0)
            # rho at the terminal step: right-hand rule + priorities
            pi, pj = pbar[K], Pj[min(K, len(Pj) - 1), :2]
            nj, nt = pj - pi, target - pi
            r = self.rho0
            if self.term_overlap and np.linalg.norm(nj) > 1e-10 and np.linalg.norm(nt) > 1e-10:
                nj, nt = nj / np.linalg.norm(nj), nt / np.linalg.norm(nt)
                r = self.rho0 * np.exp(self.eta * (nj[1] * nt[0] - nj[0] * nt[1]))
            if others_prio[j] > self.priority:
                r = 10 * self.rho0 / eps ** 2
            elif others_prio[j] < self.priority:
                r = 0.01 * self.rho0 / eps ** 2
            else:
                r /= self.W[j] / 2 / eps
            rho.append(r)
        return Aij, bij, on, rho

    # ---------- deadlock resolution (reference uav.py, deadlock_resolution) ----------
    def _deadlock(self, P, target, team):
        L = self.L
        term_p, term_second = P[-1], P[-2]
        if self.term_last_p is None:
            self.term_last_p = term_p.copy()
        if self.term_overlap:
            if (np.linalg.norm(term_p - self.term_last_p) < 0.001 * L and
                    np.linalg.norm(term_p - term_second) < 0.005 * L and
                    np.linalg.norm(term_p - target) > 0.02 * L):
                self.term_overlap_again = True
        else:
            if (np.linalg.norm(term_p - self.term_last_p) < 0.005 * L and
                    np.linalg.norm(term_p - term_second) < 0.01 * L and
                    np.linalg.norm(term_p - target) > 0.02 * L):
                self.term_overlap = True
        if np.all(self.eps - self.E < 1e-3 * L):
            self.term_overlap = self.term_overlap_again = False
            self.eta = 1.0
            self.contest = False
            if self.priority == 3:
                self.priority = 2
        self.term_index += 1
        if self.term_overlap_again and self.eta < 4.0:
            self.term_overlap_again = False
            self.eta += 0.3
            self.term_index = 0
        self.term_last_p = term_p.copy()
        no_supremacy = all(m.priority != 3 for m in team if m is not self)
        min_distance = all(not (m.distance > self.distance and m.contest) for m in team if m is not self)
        if np.linalg.norm(term_p - target) <= 0.01 * L * 10:
            self.priority = 1                             # at its (moving) goal this step
        else:
            if self.priority == 1:
                self.priority = 2
            if self.eta > 3.7:
                if self.contest and no_supremacy and min_distance:
                    self.priority = 3
                self.contest = True

    # ---------- one step ----------
    def plan(self, target, path, obs, others, team):
        """target: goal (2,); path: reference path (N, 2) from the UAV;
        obs: LiDAR points (M, 2); others: the other robots (states_prediction
        read), team: the IMPCOB objects of all UAVs (shared data)."""
        r, K, cfg = self.robot, self.K, self.cfg
        x0 = np.concatenate([r.state[[0, 1, 3, 4]], r.accel[:2]])
        pbar = np.asarray(r.states_prediction, float)[:, :2].copy()  # row k = time t + k
        pbar[0] = x0[:2]
        if len(pbar) < K + 1:
            pbar = np.vstack([pbar, np.repeat(pbar[-1:], K + 1 - len(pbar), axis=0)])
        G = self._tractive_point(pbar[K], path, getattr(getattr(r, "planner", None), "gmap", None))
        planes = self._corridors(pbar, G, np.asarray(obs, float).reshape(-1, 2))
        if planes is None:                                # predetermined trajectory collides
            self.n_corridor_fail += 1
            return self._fallback()
        others_pbar = [np.asarray(o.states_prediction, float)[:, :2] for o in others]
        others_prio = [getattr(getattr(o, "impc", None), "priority", 2) for o in others]
        Aij, bij, on, rho = self._inter(pbar, others_pbar, others_prio, target)
        dist = np.linalg.norm(pbar[K] - target)
        Q_tar = 10.0 * 2.0 * self.L / dist if dist > 2.0 * self.L else 10.0
        for i in range(K, -1, -1):
            if np.linalg.norm(pbar[i] - target) > 0.01 * self.L:
                break
        self.cost_index = i
        sig = np.zeros(K)
        sig[max(self.cost_index - 1, 0):] = Q_tar
        Xi = getattr(self, "_X", None)
        Ui = getattr(self, "_Uw", None)
        if Xi is None:
            Xi, Ui = np.tile(x0, (K + 1, 1)), np.zeros((K, 2))
        ok, X, U, w = self.prob.solve(x0, G, sig, planes, Aij, bij, on, rho, Xi, Ui)
        if not ok:
            return self._fallback()
        self._X = np.vstack([X[1:], X[-1:]])
        self._Uw = np.vstack([U[1:], np.zeros((1, 2))])
        self.U, self.left = U, K - 1
        r.states_prediction = np.column_stack(
            [X[:, :2], np.full(K + 1, r.state[2]), X[:, 2:4], np.zeros(K + 1)])
        r.controls_prediction = np.column_stack([U, np.zeros(K)])
        r.mpc_fail_streak = 0
        # warning-band bookkeeping and deadlock resolution (after the solve)
        self.E = w[:max(self.n_others, 1)]
        self.W = np.maximum(0.2 * self.E + 0.8 * self.W, 0.005 * self.L)
        self.distance = np.linalg.norm(target - X[-1, :2])
        self._deadlock(X[1:, :2], target, team)
        return np.array([U[0, 0], U[0, 1], 0.0])

    def _fallback(self):
        """Solve failed: fly the rest of the last plan (it ends at rest)."""
        r = self.robot
        r.mpc_fail_streak = getattr(r, "mpc_fail_streak", 0) + 1
        if self.U is not None and self.left > 0:
            u = self.U[self.K - self.left]
            self.left -= 1
            return np.array([u[0], u[1], 0.0])
        return np.zeros(3)
