"""
mpc_problem.py — The UAV's NMPC built ONCE and re-solved with new parameters.

robot_jps.py used to create a new casadi.Opti, add every constraint and cost
term and let CasADi re-derive the NLP at every control step (~35-40 ms per UAV
per step, about as long as IPOPT itself). Here the problem contains every term
that any mode can use; what changes from step to step is passed as parameters:

  * corridor polytope      -> A, b padded to a fixed number of faces
                              (padding rows: A = 0, b = BIG, i.e. inactive)
  * neighbour constraints  -> one slot per other UAV + a 0/1 "active" flag
  * leader CBF             -> always present, relaxed by BIG when not leader
  * mode-dependent costs   -> weights set to 0 when the term is not used
  * tracking cost branch   -> 0/1 switch between "guide" and "standoff" forms

With every switch at the value the old code implied, the cost and the feasible
set are the same as before (up to constant offsets from padded rows).

Cost scaling (COST_LENGTH_SCALE = L, COST_ACCEL_SCALE = U):
    distances are divided by L and controls by U before they enter the cost,
    so weights tuned on a small map keep their meaning on a large one.
    L = U = 1 reproduces the original cost exactly.
"""

import casadi as ca
import numpy as np

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
        X = opti.variable(H + 1, 6)
        U = opti.variable(H, 3)
        S = opti.variable(H, 4)                       # leader CBF slack (m)

        p = {}
        p["x0"] = opti.parameter(1, 6)
        p["A_hard"] = opti.parameter(n_faces, 2)
        p["b_hard"] = opti.parameter(n_faces, 1)
        p["A_cost"] = opti.parameter(n_faces, 2)
        p["b_cost"] = opti.parameter(n_faces, 1)
        p["w_corr"] = opti.parameter()
        p["others"] = [opti.parameter(H + 1, 2) for _ in range(n_others)]
        p["nb_flag"] = opti.parameter(max(n_others, 1), 1)
        p["tgt"] = opti.parameter(1, 2)               # current target position
        p["w_search"] = opti.parameter()
        p["slot"] = opti.parameter(1, 2)
        p["w_slot"] = opti.parameter()
        p["leader"] = opti.parameter()
        p["ref"] = opti.parameter(1, 2)               # next reference waypoint
        p["trk_goal"] = opti.parameter(1, 2)          # predicted target (lead pursuit)
        p["s_ref"] = opti.parameter()                 # 1: guide to ref, 0: standoff

        # ── dynamics ──
        opti.subject_to(X[0, :] == p["x0"])
        for i in range(H):
            opti.subject_to(X[i + 1, :] == X[i, :] + ca.horzcat(X[i, 3:], U[i, :]) * dt)

        # ── corridor (hard) ──
        for i in range(H + 1):
            opti.subject_to(ca.mtimes(p["A_hard"], X[i, :2].T) <= p["b_hard"] - R)

        # ── neighbours (hard), active only if flagged ──
        for i in range(H):
            for j in range(n_others):
                d2 = ca.sumsqr(X[i, :2] - p["others"][j][i, :])
                opti.subject_to(d2 >= (2 * R) ** 2 * p["nb_flag"][j])

        # ── leader visibility CBF (relaxed by BIG when not leader) ──
        Lcbf = cfg["CBF_BOX_RATIO"] * cfg["VIEWING_RADIUS"]
        gam = cfg["DT_CBF_GAMMA"]
        relax = BIG * (1 - p["leader"])
        tgt = p["tgt"]
        for i in range(H):
            c, n = X[i, :2], X[i + 1, :2]
            h_cur = ca.vertcat(Lcbf - (c[0] - tgt[0]), Lcbf - (tgt[0] - c[0]),
                               Lcbf - (c[1] - tgt[1]), Lcbf - (tgt[1] - c[1]))
            h_nxt = ca.vertcat(Lcbf - (n[0] - tgt[0]), Lcbf - (tgt[0] - n[0]),
                               Lcbf - (n[1] - tgt[1]), Lcbf - (tgt[1] - n[1]))
            for d in range(4):
                opti.subject_to(h_nxt[d] - (1 - gam) * h_cur[d] >= -S[i, d] - relax)
        opti.subject_to(ca.vec(S) >= 0)

        # ── bounds ──
        for i in range(H):
            opti.subject_to(ca.sumsqr(X[i + 1, 3:]) <= cfg["VMAX"] ** 2)
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
        cost += cfg["W_tra"] * (1 - p["s_ref"]) * (d_goal - standoff) ** 2
        # corridor barrier
        eps = cfg["CORRIDOR_BARRIER_EPS"]
        for i in range(H):
            dist = p["b_cost"] - ca.mtimes(p["A_cost"], X[i, :2].T) - R
            cost += p["w_corr"] * ca.sum1(Ls / (dist + eps))
        # soft collision margin to every other UAV
        d_safe = cfg["COLLISION_AVOID_DISTANCE"]
        for k in range(H + 1):
            for j in range(n_others):
                d = ca.sqrt(ca.sumsqr(X[k, :2] - p["others"][j][k, :]) + 1e-6)
                v = ca.fmax(0, d_safe - d) / Ls
                cost += cfg["W_collision_avoid"] * v * v
        # SEARCH: pull towards the target / satellite: pull towards the slot
        for k in range(H + 1):
            cost += p["w_search"] * ca.sumsqr(X[k, :2] - p["tgt"]) / Ls ** 2
            cost += p["w_slot"] * ca.sumsqr(X[k, :2] - p["slot"]) / Ls ** 2
        # leader CBF slack
        cost += cfg["W_leader_slack"] * ca.sum1(ca.sum2((S / Ls) ** 3))

        opti.minimize(cost)
        opts = dict(cfg["IPOPT_OPTIONS"])
        opts.setdefault("expand", True)      # SX graph: much cheaper function evaluations
        opti.solver("ipopt", opts)
        self.opti, self.X, self.U, self.S, self.p = opti, X, U, S, p

    # ------------------------------------------------------------
    def _pad(self, A, b):
        A = np.zeros((0, 2)) if A is None else np.asarray(A, float).reshape(-1, 2)
        b = np.zeros(0) if b is None else np.asarray(b, float).reshape(-1)
        Ap = np.zeros((self.n_faces, 2))
        bp = np.full((self.n_faces, 1), BIG)
        Ap[:len(A)] = A
        bp[:len(b), 0] = b
        return Ap, bp

    def solve(self, x0, A_hard, b_hard, A_cost, b_cost, use_corr_cost, others, nb_flags,
              tgt, w_search, slot, w_slot, leader, ref, trk_goal, s_ref,
              X_init, U_init):
        """Returns (ok, X, U)."""
        o, p, cfg = self.opti, self.p, self.cfg
        o.set_value(p["x0"], np.asarray(x0, float).reshape(1, 6))
        Ah, bh = self._pad(A_hard, b_hard)
        Ac, bc = self._pad(A_cost, b_cost)
        o.set_value(p["A_hard"], Ah)
        o.set_value(p["b_hard"], bh)
        o.set_value(p["A_cost"], Ac)
        o.set_value(p["b_cost"], bc)
        o.set_value(p["w_corr"], cfg["W_corridor"] if use_corr_cost else 0.0)
        for j in range(self.n_others):
            o.set_value(p["others"][j], np.asarray(others[j], float)[:, :2])
        flags = np.zeros((max(self.n_others, 1), 1))
        flags[:len(nb_flags), 0] = nb_flags
        o.set_value(p["nb_flag"], flags)
        o.set_value(p["tgt"], np.asarray(tgt, float)[:2].reshape(1, 2))
        o.set_value(p["w_search"], w_search)
        o.set_value(p["slot"], np.asarray(slot, float)[:2].reshape(1, 2))
        o.set_value(p["w_slot"], w_slot)
        o.set_value(p["leader"], float(leader))
        o.set_value(p["ref"], np.asarray(ref, float)[:2].reshape(1, 2))
        o.set_value(p["trk_goal"], np.asarray(trk_goal, float)[:2].reshape(1, 2))
        o.set_value(p["s_ref"], float(s_ref))
        o.set_initial(self.X, X_init)
        o.set_initial(self.U, U_init)
        o.set_initial(self.S, 0)
        try:
            sol = o.solve()
            return True, sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            return False, None, None
