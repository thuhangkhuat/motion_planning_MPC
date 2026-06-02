"""
Kalman filter để estimate target velocity từ noisy position observations.
Constant velocity (CV) model.

State:   x = [px, py, vx, vy]^T
Process: x_{k+1} = F @ x_k + w   (w ~ N(0, Q))
Observation: z = H @ x + v        (v ~ N(0, R), chỉ đo position)

Dùng cho lead-pursuit: UAV không biết trajectory target, chỉ đo position
qua sensor mỗi frame -> Kalman estimate velocity -> predict future target pos.
"""

import numpy as np


class KalmanTargetTracker:
    """
    Constant-velocity Kalman filter cho 2D target tracking.

    Usage:
        tracker = KalmanTargetTracker(dt=0.1)
        tracker.initialize(initial_pos)
        # mỗi frame:
        tracker.update(observed_pos)
        velocity = tracker.get_velocity()
        future_pos = tracker.predict_future(lookahead_time=1.0)
    """

    def __init__(self, dt=0.1,
                 process_noise_std=1.0,   # std của acceleration unmodeled (m/s²)
                 obs_noise_std=0.3):       # std của sensor noise (m)
        """
        Args:
            dt: timestep giữa các update (s)
            process_noise_std: target có thể tăng tốc bất ngờ tới mức này
                               (lớn → tracker tin observation hơn)
            obs_noise_std: sensor đo position sai khoảng này
                           (lớn → tracker tin model hơn)
        """
        self.dt = dt
        self.initialized = False

        # State transition F: constant velocity
        # [px']   [1 0 dt  0] [px]
        # [py'] = [0 1  0 dt] [py]
        # [vx']   [0 0  1  0] [vx]
        # [vy']   [0 0  0  1] [vy]
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Observation H: chỉ đo position
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Process noise Q: từ continuous-time white noise acceleration model
        # Đối với CV model với acceleration noise std = q
        # Q = q^2 * [[dt^4/4, 0, dt^3/2, 0],
        #            [0, dt^4/4, 0, dt^3/2],
        #            [dt^3/2, 0, dt^2, 0],
        #            [0, dt^3/2, 0, dt^2]]
        q = process_noise_std
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        self.Q = q * q * np.array([
            [dt4 / 4, 0,       dt3 / 2, 0      ],
            [0,       dt4 / 4, 0,       dt3 / 2],
            [dt3 / 2, 0,       dt2,     0      ],
            [0,       dt3 / 2, 0,       dt2    ],
        ])

        # Observation noise R
        r = obs_noise_std
        self.R = r * r * np.eye(2)

        # State estimate và covariance
        self.x = np.zeros(4)        # [px, py, vx, vy]
        self.P = np.eye(4) * 100    # initial uncertainty cao

    def initialize(self, initial_pos):
        """Khởi tạo state với position đầu tiên, velocity = 0."""
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])
        # Position uncertainty thấp, velocity uncertainty cao (chưa biết)
        self.P = np.diag([1.0, 1.0, 100.0, 100.0])
        self.initialized = True

    def predict(self):
        """Predict step: x = F @ x, P = F @ P @ F.T + Q."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, observed_pos):
        """
        Full predict + update với observation mới.
        observed_pos: array-like [x, y]
        """
        if not self.initialized:
            self.initialize(observed_pos)
            return

        # Predict
        self.predict()

        # Update
        z = np.asarray(observed_pos[:2], dtype=float)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_position(self):
        """Trả về estimate position hiện tại [x, y]."""
        return self.x[:2].copy()

    def get_velocity(self):
        """Trả về estimate velocity [vx, vy]."""
        return self.x[2:].copy()

    def get_speed(self):
        """Trả về tốc độ scalar."""
        return float(np.linalg.norm(self.x[2:]))

    def predict_future(self, lookahead_time):
        """
        Dự đoán target position sau lookahead_time giây (giả định constant velocity).
        KHÔNG modify state hiện tại.
        """
        return self.x[:2] + self.x[2:] * lookahead_time

    def get_velocity_uncertainty(self):
        """
        Trả về std của velocity estimate (sqrt diagonal của P).
        Cao = chưa tin được velocity (mới bắt đầu track hoặc target maneuver).
        """
        return float(np.sqrt(self.P[2, 2] + self.P[3, 3]))