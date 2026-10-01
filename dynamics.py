"""
dynamics.py — UAV translational model, shared by the MPC and the simulator.

Per axis (x, y, z are decoupled), with u the commanded acceleration:

    p' = v
    v' = a - D * v          D   = D_FRAC     (linear drag, 1/s)
    a' = (u - a) / tau      tau = ACCEL_TAU  (attitude/thrust response, s)

The first-order lag on a models that a multirotor has to tilt before its
acceleration changes. tau = 0 means a follows u instantly (double integrator).

The model is discretised exactly for a control held constant over one step
(zero-order hold), so the simulator and the MPC predict the same motion.
State per axis: s = [p, v, a];  s[k+1] = Ad s[k] + Bd u[k].
"""

import numpy as np
from scipy.linalg import expm


def _zoh(Ac, Bc, dt):
    n, m = Bc.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = Ac, Bc
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def axis_model(dt, drag, tau):
    """(Ad 3x3, Bd 3x1) for one axis, state [p, v, a]."""
    if tau > 0:
        Ac = np.array([[0.0, 1.0, 0.0],
                       [0.0, -drag, 1.0],
                       [0.0, 0.0, -1.0 / tau]])
        Bc = np.array([[0.0], [0.0], [1.0 / tau]])
        return _zoh(Ac, Bc, dt)
    # tau = 0: a = u during the step, the a-state just stores the last command
    A2, B2 = _zoh(np.array([[0.0, 1.0], [0.0, -drag]]), np.array([[0.0], [1.0]]), dt)
    Ad = np.zeros((3, 3))
    Ad[:2, :2] = A2
    Bd = np.vstack([B2, [[1.0]]])
    return Ad, Bd


def model(dt, drag, tau, n_axes):
    """
    (A, B) for n_axes decoupled axes, state ordered [p(n), v(n), a(n)]
    and control u(n):  s[k+1] = A s[k] + B u[k].
    """
    Ad, Bd = axis_model(dt, drag, tau)
    I = np.eye(n_axes)
    return np.kron(Ad, I), np.kron(Bd, I)
