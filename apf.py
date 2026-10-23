"""
apf.py — Baseline "apf": the artificial potential field method of this
project's first version (robot_apf.py, removed in commit ff5ffdc), on the
shared leader / slot layers. No path planner, corridor or CBF.

    v_des = 1.0 v_tar + 1.5 v_col + 0.01 v_rand
    u     = v_des - v           (clamped to UMAX; speed kept <= VMAX)
v_tar: towards the goal (the formation slot, or the predicted target for
the leader) at VMAX, proportional to the distance inside L;
v_col: mean of the repulsions VMAX (1 - d/L) / (1 - R/L) away from every
LiDAR point (and other UAV) within L; v_rand: uniform in [0, 1)^2 * VMAX.

Scaling from the original (VMAX = 1 m/s, R = 0.25 m, same 0.1 s step):
speeds x VMAX, the goal threshold x L = VMAX / 1 m/s. The repulsion range is
not scaled the same way: at 1 m/s the braking distance (0.1 m) was a tenth
of the 1 m range, at 10 m/s it is ~10 m, so with L the UAVs could not stop
and flew into the obstacles; it is set to twice the braking distance,
L_col = 2 * VMAX^2 / (2 UMAX) (20 m here). Other UAVs are obstacle points, as
the original LiDAR saw the robots.
"""

import numpy as np


class APF:

    def __init__(self, robot, cfg, rng=None):
        self.robot, self.cfg = robot, cfg
        self.L = cfg["VMAX"] / 1.0
        self.Lc = max(self.L, 2 * cfg["VMAX"] ** 2 / (2 * cfg["UMAX"]))   # repulsion range
        self.rng = rng if rng is not None else np.random.default_rng(1000 + robot.index)

    def plan(self, goal, obs, others):
        r, cfg = self.robot, self.cfg
        p, v = r.state[:2], r.state[3:5]
        vmax, L, Lc, R = cfg["VMAX"], self.L, self.Lc, cfg["ROBOT_RADIUS"]
        d = goal - p
        dist = np.linalg.norm(d)
        v_tar = vmax * (d / L if dist < L else d / dist)
        pts = np.asarray(obs, float).reshape(-1, 2)
        if others:
            pts = np.vstack([pts, [o.state[:2] for o in others]])
        v_col = np.zeros(2)
        if len(pts):
            rel = p - pts
            dis = np.linalg.norm(rel, axis=1) - R
            near = dis <= Lc
            if near.any():
                dirs = rel[near] / np.maximum(dis[near], 1e-6)[:, None]   # as the original: rel / dis
                v_col = (vmax * (1 - dis[near] / Lc) / (1 - R / Lc))[:, None] * dirs
                v_col = v_col.sum(axis=0) / near.sum()
        v_ran = vmax * self.rng.random(2)
        v_des = 1.0 * v_tar + 1.5 * v_col + 1e-2 * v_ran
        u = v_des - v
        n = np.linalg.norm(u)
        if n > cfg["UMAX"]:
            u = u * cfg["UMAX"] / n
        from apfc import speed_limit
        x0 = np.concatenate([r.state[[0, 1, 3, 4]], r.accel[:2]])
        u = speed_limit(cfg, x0, u)
        H = cfg["HORIZON_LENGTH"]
        tt = np.arange(H + 1)[:, None] * cfg["TIMESTEP"]
        r.states_prediction = np.column_stack(
            [p + tt * v, np.full(H + 1, r.state[2]), np.repeat(v[None], H + 1, axis=0), np.zeros(H + 1)])
        r.mpc_fail_streak = 0
        return np.array([u[0], u[1], 0.0])
