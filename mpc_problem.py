"""
mpc_problem.py — The UAV's NMPC built ONCE and re-solved with new parameters.

robot_jps.py used to create a new casadi.Opti, add every constraint and cost
term and let CasADi re-derive the NLP at every control step (~35-40 ms per UAV
per step, about as long as IPOPT itself). Here the problem contains every term
that any mode can use; what changes from step to step is passed as parameters:

  * corridor polytope      -> A, b padded to a fixed number of faces
                              (padding rows: A = 0, b = BIG, i.e. inactive)
  * neighbour constraints  -> one slot per other UAV + a 0/1 "active" flag
                              (|p - p_j| >= UAV_SAFE_DISTANCE, slack with a large penalty)
  * map bounds             -> WORLD_BOUNDS shrunk by R, slack with a large penalty
  * leader CBF             -> always present, relaxed by BIG when not leader
  * mode-dependent costs   -> weights set to 0 when the term is not used
  * tracking cost branch   -> 0/1 switch between "guide" and "standoff" forms

With every switch at the value the old code implied, the cost and the feasible
set are the same as before (up to constant offsets from padded rows).

The UAV flies at a fixed altitude, so the MPC is planar. Dynamics: dynamics.py
(exact discretisation, drag D_FRAC, first-order acceleration lag ACCEL_TAU).
    state   X[k] = [px, py, vx, vy, ax, ay]   (a = actual acceleration)
    control U[k] = [ux, uy]                   (commanded acceleration)

Cost scaling (COST_LENGTH_SCALE = L, COST_ACCEL_SCALE = U):
    distances are divided by L and controls by U before they enter the cost,
    so weights tuned on a small map keep their meaning on a large one.
    L = U = 1 reproduces the original cost exactly.
"""

import time

import casadi as ca
import numpy as np

import dynamics

BIG = 1e6


class MPCProblem:

    def __init__(self, n_others, n_faces, cfg):
        """
        n_others : number of OTHER UAVs (fixed for a run)
        n_faces  : max number of corridor faces; the problem is rebuilt with
                   more faces if a corridor ever needs them
        cfg      : dict of the config constants used below
        """
        self.cfg = cfg
        self.n_others = n_others
        self.n_faces = n_faces
        H = cfg["HORIZON_LENGTH"]
        dt = cfg["TIMESTEP"]
        R = cfg["ROBOT_RADIUS"]
        Ls = cfg["COST_LENGTH_SCALE"]
        Us = cfg["COST_ACCEL_SCALE"]

        opti = ca.Opti()
        X = opti.variable(H + 1, 6)                   # [px, py, vx, vy, ax, ay]
        U = opti.variable(H, 2)                       # commanded acceleration
        S = opti.variable(H, 4)                       # leader CBF slack (m)
        Sc = opti.variable(H, max(n_others, 1))       # UAV-UAV distance slack (m²)
        Sb = opti.variable(H, 1)                      # map-bounds slack (m)

        p = {}
        p["x0"] = opti.parameter(1, 6)
        p["A_hard"] = opti.parameter(n_faces, 2)
        p["b_hard"] = opti.parameter(n_faces, 1)
        p["corr_relax"] = opti.parameter(n_faces, 1)  # current intrusion into the R margin, per face
        p["brake_mask"] = opti.parameter(n_faces, 1)  # 1 = face lies on an obstacle (braking constraint)
        p["A_cost"] = opti.parameter(n_faces, 2)
        p["b_cost"] = opti.parameter(n_faces, 1)
        p["w_corr"] = opti.parameter()
        p["others"] = [opti.parameter(H + 1, 2) for _ in range(n_others)]
        p["nb_flag"] = opti.parameter(max(n_others, 1), 1)
        p["tgt"] = opti.parameter(H + 1, 2)           # predicted target position at each step
        p["w_search"] = opti.parameter()
        p["slot"] = opti.parameter(H + 1, 2)          # slot position at each horizon step
        p["w_slot"] = opti.parameter()
        p["leader"] = opti.parameter()
        p["ref"] = opti.parameter(1, 2)               # next reference waypoint
        p["trk_goal"] = opti.parameter(1, 2)          # predicted target (lead pursuit)
        p["s_ref"] = opti.parameter()                 # 1: guide to ref, 0: standoff
        p["w_tra"] = opti.parameter()                 # standoff weight (0 for satellites)

        # ── dynamics ──
        Ad, Bd = dynamics.model(dt, cfg["D_FRAC"], cfg["ACCEL_TAU"], 2)
        opti.subject_to(X[0, :] == p["x0"])
        for i in range(H):
            opti.subject_to(X[i + 1, :] == ca.mtimes(X[i, :], Ad.T) + ca.mtimes(U[i, :], Bd.T))

        # ── corridor (hard) ──
        # X[0] is the measured state and is not constrained: a UAV already inside
        # the R margin of a face (by e) made the problem infeasible at every step,
        # it braked, stayed there and failed again (a 65 s deadlock in scen2).
        # Instead the UAV may stay inside the margin by at most what it can have
        # covered by then: allow(t) = max(0, e - s(t)), with s(t) = 0 for
        # t <= CORRIDOR_RECOVER_HOLD (acceleration lag) and then the distance
        # travelled from rest at CORRIDOR_RECOVER_ACCEL_RATIO * UMAX. It never goes
        # deeper than now, a UAV at rest can always comply (half of UMAX), and the
        # time to get out grows with e: ~0.7 s for 0.25 m, ~1.6 s for 2.25 m (a
        # fixed 1 s deadline was infeasible for 2.25 m, another deadlock in scen2).
        # A UAV outside the margin has e = 0: plain A p <= b - R at every step.
        a_rec = cfg["CORRIDOR_RECOVER_ACCEL_RATIO"] * cfg["UMAX"]
        t_hold = cfg["CORRIDOR_RECOVER_HOLD"]
        for i in range(1, H + 1):
            s_i = 0.5 * a_rec * max(0.0, i * dt - t_hold) ** 2
            allow = ca.fmax(0, p["corr_relax"] - s_i)
            opti.subject_to(ca.mtimes(p["A_hard"], X[i, :2].T) <= p["b_hard"] - R + allow)

        # ── braking room at the end of the horizon (obstacle faces only) ──
        # The corridor alone lets the plan end next to a face while still flying
        # at it; one step later, or after the corridor is rebuilt, no plan can
        # stop in time and the MPC fails. At step H the UAV must be able to stop
        # before every obstacle face: v_n * t_lag + v_n^2 / (2 a_b) <= distance,
        # v_n = speed towards the face, a_b = CORRIDOR_BRAKE_ACCEL_RATIO * UMAX,
        # t_lag = ACCEL_TAU + dt. Faces of pydecomp's bounding box carry no
        # obstacle and are left out, otherwise the UAV would slow down in open space.
        if cfg["CORRIDOR_BRAKE"]:
            a_b = cfg["CORRIDOR_BRAKE_ACCEL_RATIO"] * cfg["UMAX"]
            t_lag = cfg["ACCEL_TAU"] + dt
            # smooth max(v_n, 0) (>= it, so conservative): fmax has a kink at
            # v_n = 0, which is exactly where the plan ends when it stops, and
            # IPOPT then ran out of iterations on feasible problems
            v_raw = ca.mtimes(p["A_hard"], X[H, 2:4].T)
            vn = 0.5 * (v_raw + ca.sqrt(v_raw ** 2 + 0.1 ** 2))
            s_H = 0.5 * a_rec * max(0.0, H * dt - t_hold) ** 2
            room = p["b_hard"] - R + ca.fmax(0, p["corr_relax"] - s_H) - ca.mtimes(p["A_hard"], X[H, :2].T)
            opti.subject_to(p["brake_mask"] * (vn * t_lag + vn ** 2 / (2 * a_b)) <= room)

        # ── neighbours: |p - p_j| >= UAV_SAFE_DISTANCE, active only if flagged ──
        # Soft-hard: a slack with a large penalty keeps the problem feasible when
        # another UAV is already too close (an infeasible MPC falls back to
        # braking, which made the near-misses worse).
        Dsafe = cfg["UAV_SAFE_DISTANCE"]
        for i in range(H):
            for j in range(n_others):
                d2 = ca.sumsqr(X[i + 1, :2] - p["others"][j][i + 1, :])
                opti.subject_to(d2 >= Dsafe ** 2 * p["nb_flag"][j] - Sc[i, j])
        opti.subject_to(ca.vec(Sc) >= 0)

        # ── map bounds (soft-hard, as above) ──
        bx0, by0, bx1, by1 = cfg["WORLD_BOUNDS"]
        for i in range(H):
            px, py = X[i + 1, 0], X[i + 1, 1]
            opti.subject_to(px >= bx0 + R - Sb[i])
            opti.subject_to(px <= bx1 - R + Sb[i])
            opti.subject_to(py >= by0 + R - Sb[i])
            opti.subject_to(py <= by1 - R + Sb[i])
        opti.subject_to(Sb >= 0)

        # ── leader visibility CBF (relaxed by BIG when not leader) ──
        Lcbf = cfg["CBF_BOX_RATIO"] * cfg["VIEWING_RADIUS"]
        gam = cfg["DT_CBF_GAMMA"]
        relax = BIG * (1 - p["leader"])
        # h(x_k, t_k) = Lcbf - |x_k - t_k| per axis, with t_k the target predicted
        # at step k: a fixed t_k lets a moving target drift out of the box by
        # ~v*dt per step, which matters once the box is close to the FOV.
        def h_box(x, t):
            return ca.vertcat(Lcbf - (x[0] - t[0]), Lcbf - (t[0] - x[0]),
                              Lcbf - (x[1] - t[1]), Lcbf - (t[1] - x[1]))
        for i in range(H):
            h_cur = h_box(X[i, :2], p["tgt"][i, :])
            h_nxt = h_box(X[i + 1, :2], p["tgt"][i + 1, :])
            for d in range(4):
                opti.subject_to(h_nxt[d] - (1 - gam) * h_cur[d] >= -S[i, d] - relax)
        opti.subject_to(ca.vec(S) >= 0)

        # ── bounds ──
        for i in range(H):
            opti.subject_to(ca.sumsqr(X[i + 1, 2:4]) <= cfg["VMAX"] ** 2)
            opti.subject_to(ca.sumsqr(U[i, :]) <= cfg["UMAX"] ** 2)

        # ── cost ──
        cost = 0
        # control effort
        for i in range(H):
            cost += cfg["W_u"] * ca.sumsqr(U[i, :] / Us)
        # tracking: guide to the next waypoint, or keep the standoff distance
        d_ref = ca.sumsqr(X[H, :2] - p["ref"]) / Ls ** 2
        d_goal = ca.sumsqr(X[H, :2] - p["trk_goal"]) / Ls ** 2
        standoff = (cfg["STANDOFF_DISTANCE"] / Ls) ** 2
        cost += cfg["W_gui"] * p["s_ref"] * d_ref ** 2
        cost += p["w_tra"] * (1 - p["s_ref"]) * (d_goal - standoff) ** 2
        # corridor barrier. dist is clamped at 0: outside the margin (a UAV
        # recovering into the corridor, or a cost polytope that does not contain
        # it) 1/(dist + eps) has a pole at dist = -eps, so every path back inside
        # had infinite cost and IPOPT failed (U5 at rest 2.2 m outside, scen2).
        # The hard corridor with its recovery rule handles that case.
        eps = cfg["CORRIDOR_BARRIER_EPS"]
        for i in range(H):
            dist = p["b_cost"] - ca.mtimes(p["A_cost"], X[i, :2].T) - R
            cost += p["w_corr"] * ca.sum1(Ls / (ca.fmax(dist, 0) + eps))
        # soft collision margin to every other UAV
        d_safe = cfg["COLLISION_AVOID_DISTANCE"]
        for k in range(H + 1):
            for j in range(n_others):
                d = ca.sqrt(ca.sumsqr(X[k, :2] - p["others"][j][k, :]) + 1e-6)
                v = ca.fmax(0, d_safe - d) / Ls
                cost += cfg["W_collision_avoid"] * v * v
        # SEARCH: pull towards the target / satellite: pull towards the slot
        for k in range(H + 1):
            cost += p["w_search"] * ca.sumsqr(X[k, :2] - p["tgt"][0, :]) / Ls ** 2
            cost += p["w_slot"] * ca.sumsqr(X[k, :2] - p["slot"][k, :]) / Ls ** 2
        # leader CBF slack
        cost += cfg["W_leader_slack"] * ca.sum1(ca.sum2((S / Ls) ** 3))
        # UAV-UAV and map-bounds slacks: linear (exact penalty) + quadratic
        sc = Sc / Ls ** 2
        cost += cfg["W_uav_slack"] * (ca.sum1(ca.sum2(sc)) + ca.sumsqr(sc))
        sb = Sb / Ls
        cost += cfg["W_bounds_slack"] * (ca.sum1(sb) + ca.sumsqr(sb))

        opti.minimize(cost)
        solver = cfg.get("MPC_SOLVER", "ipopt")
        opts = dict(cfg["SQP_OPTIONS" if solver == "sqp" else "IPOPT_OPTIONS"])
        opts.setdefault("expand", True)      # SX graph: much cheaper function evaluations
        opti.solver("sqpmethod" if solver == "sqp" else "ipopt", opts)
        self.opti, self.X, self.U, self.S, self.p = opti, X, U, S, p
        self.Sc, self.Sb = Sc, Sb
        self.solve_times = []                # wall time of every solve() call, fallback included (s)
        self.n_fail = 0
        self.n_fallback = 0                  # SQP failures re-solved with IPOPT
        self.fallback = None
        if solver == "sqp" and cfg.get("SQP_FALLBACK_IPOPT", True):
            self.fallback = MPCProblem(n_others, n_faces, {**cfg, "MPC_SOLVER": "ipopt"})

    # ------------------------------------------------------------
    def _pad(self, A, b):
        A = np.zeros((0, 2)) if A is None else np.asarray(A, float).reshape(-1, 2)
        b = np.zeros(0) if b is None else np.asarray(b, float).reshape(-1)
        Ap = np.zeros((self.n_faces, 2))
        bp = np.full((self.n_faces, 1), BIG)
        Ap[:len(A)] = A
        bp[:len(b), 0] = b
        return Ap, bp

    def solve(self, *args, **kwargs):
        """Returns (ok, X, U) in the MPC layout above. Arguments: see _solve."""
        t0 = time.perf_counter()
        try:
            ok, X, U = self._solve(*args, **kwargs)
            if not ok and self.fallback is not None:
                self.n_fallback += 1
                ok, X, U = self.fallback._solve(*args, **kwargs)
            self.n_fail += not ok
            return ok, X, U
        finally:
            self.solve_times.append(time.perf_counter() - t0)

    def _solve(self, x0, A_hard, b_hard, A_cost, b_cost, use_corr_cost, others, nb_flags,
               tgt, w_search, slot, w_slot, leader, ref, trk_goal, s_ref,
               X_init, U_init, w_tra=None, brake_mask=None):
        o, p, cfg = self.opti, self.p, self.cfg
        o.set_value(p["x0"], np.asarray(x0, float).reshape(1, 6))   # [px, py, vx, vy, ax, ay]
        Ah, bh = self._pad(A_hard, b_hard)
        Ac, bc = self._pad(A_cost, b_cost)
        o.set_value(p["A_hard"], Ah)
        o.set_value(p["b_hard"], bh)
        x_now = np.asarray(x0, float).reshape(-1)[:2]
        o.set_value(p["corr_relax"], np.maximum(0.0, Ah @ x_now[:, None] - (bh - cfg["ROBOT_RADIUS"])))
        mask = np.zeros((self.n_faces, 1))
        if brake_mask is not None:
            mk = np.asarray(brake_mask, float).reshape(-1)
            mask[:len(mk), 0] = mk
        o.set_value(p["brake_mask"], mask)
        o.set_value(p["A_cost"], Ac)
        o.set_value(p["b_cost"], bc)
        o.set_value(p["w_corr"], cfg["W_corridor"] if use_corr_cost else 0.0)
        for j in range(self.n_others):
            o.set_value(p["others"][j], np.asarray(others[j], float)[:, :2])
        flags = np.zeros((max(self.n_others, 1), 1))
        flags[:len(nb_flags), 0] = nb_flags
        o.set_value(p["nb_flag"], flags)
        tgt = np.asarray(tgt, float)                  # (2,) fixed point or (H+1, 2) trajectory
        o.set_value(p["tgt"], np.broadcast_to(tgt[..., :2], (cfg["HORIZON_LENGTH"] + 1, 2)))
        o.set_value(p["w_search"], w_search)
        slot = np.asarray(slot, float)                # (2,) fixed point or (H+1, 2) trajectory
        o.set_value(p["slot"], np.broadcast_to(slot[..., :2], (cfg["HORIZON_LENGTH"] + 1, 2)))
        o.set_value(p["w_slot"], w_slot)
        o.set_value(p["leader"], float(leader))
        o.set_value(p["ref"], np.asarray(ref, float)[:2].reshape(1, 2))
        o.set_value(p["trk_goal"], np.asarray(trk_goal, float)[:2].reshape(1, 2))
        o.set_value(p["s_ref"], float(s_ref))
        o.set_value(p["w_tra"], cfg["W_tra"] if w_tra is None else float(w_tra))
        o.set_initial(self.X, X_init)
        o.set_initial(self.U, U_init)
        o.set_initial(self.S, 0)
        o.set_initial(self.Sc, 0)
        o.set_initial(self.Sb, 0)
        try:
            sol = o.solve()
            return True, sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            return False, None, None
