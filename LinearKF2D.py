import numpy as np

class LinearKF2D:
    def __init__(self, process_var=1.0, meas_var=1.0):
        self.process_var = process_var
        self.meas_var = meas_var

        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1e3  

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = meas_var * np.eye(2)

    def initialize(self, p0_xy, v0_xy=None, P0_scale=1.0):
        p0_xy = np.asarray(p0_xy).reshape(2)
        if v0_xy is None:
            v0_xy = np.zeros(2)
        else:
            v0_xy = np.asarray(v0_xy).reshape(2)

        self.x = np.hstack([p0_xy, v0_xy]).reshape(4, 1)
        self.P = np.eye(4) * P0_scale
        # print("self.P\n", self.P)

    def _compute_matrices(self, dt):
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        B = np.array([
            [0.5 * dt2,         0.0       ],
            [0.0,               0.5 * dt2 ],
            [dt,                0.0       ],
            [0.0,               dt        ],
        ])

        q = self.process_var
        Q_pos = q * dt4 / 4.0
        Q_pos_vel = q * dt3 / 2.0
        Q_vel = q * dt2

        Q = np.array([
            [Q_pos,     0.0,        Q_pos_vel, 0.0      ],
            [0.0,       Q_pos,      0.0,       Q_pos_vel],
            [Q_pos_vel, 0.0,        Q_vel,     0.0      ],
            [0.0,       Q_pos_vel,  0.0,       Q_vel    ],
        ])

        return A, B, Q

    def predict(self, a_world_xy, dt):
        a_world_xy = np.asarray(a_world_xy).reshape(2, 1)
        A, B, Q = self._compute_matrices(dt)

        # x_k+1 = A x_k + B u_k
        self.x = A @ self.x + B @ a_world_xy

        # P_k+1 = A P_k A^T + Q
        self.P = A @ self.P @ A.T + Q

        # print("self.P\n", self.P)

    def update(self, z_xy):
        z = np.asarray(z_xy).reshape(2, 1)

        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # print("K", K)
        # print("self.x", self.x)
        
        self.x = self.x + K @ y
        I4 = np.eye(4)
        self.P = (I4 - K @ self.H) @ self.P

    @property
    def position(self):
        return self.x[0:2, 0]   # [x, y]

    @property
    def velocity(self):
        return self.x[2:4, 0]   # [vx, vy]
