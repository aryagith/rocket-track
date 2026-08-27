"""Linear Kalman filter over a constant-acceleration box in pixel space.

State is ``[cx, cy, vx, vy, ax, ay, w, h]``: centre in px, velocity in px/s,
acceleration in px/s^2, and box size as a slow random walk. Prediction takes
elapsed seconds, never a frame count, so irregular capture intervals are exact.

Motion and measurement models are separate objects exposing Jacobians. Today
both are linear, so the Jacobian is the constant matrix and this is a plain
Kalman filter; the same structure accepts the nonlinear angular measurement
model needed once the camera itself rotates.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# State indices.
CX, CY, VX, VY, AX, AY, W, H = range(8)
STATE_DIM = 8
MEAS_DIM = 4

_AXIS_X = (CX, VX, AX)
_AXIS_Y = (CY, VY, AY)


def box_to_measurement(xyxy: Sequence[float]) -> np.ndarray:
    """``[x1,y1,x2,y2]`` -> measurement ``[cx, cy, w, h]``."""
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float64)


def measurement_to_box(z: Sequence[float]) -> Tuple[float, float, float, float]:
    """``[cx, cy, w, h]`` -> ``[x1,y1,x2,y2]``."""
    cx, cy, w, h = (float(v) for v in z[:4])
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class ConstantAccelerationBox:
    """Constant acceleration on the centre; width/height as a random walk.

    ``jerk_psd`` is the continuous white-noise-jerk spectral density driving the
    centre (px^2/s^5); ``size_psd`` drives box size (px^2/s).
    """

    def __init__(self, jerk_psd: float = 1.0e4, size_psd: float = 50.0):
        self.jerk_psd = float(jerk_psd)
        self.size_psd = float(size_psd)

    def transition(self, dt: float) -> np.ndarray:
        dt = float(dt)
        F = np.eye(STATE_DIM, dtype=np.float64)
        for pos, vel, acc in (_AXIS_X, _AXIS_Y):
            F[pos, vel] = dt
            F[pos, acc] = 0.5 * dt * dt
            F[vel, acc] = dt
        return F

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Linear model: the Jacobian is the transition matrix itself."""
        return self.transition(dt)

    def process_noise(self, dt: float) -> np.ndarray:
        dt = float(dt)
        q = self.jerk_psd
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        block = q * np.array(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
                [dt3 / 6.0, dt2 / 2.0, dt],
            ],
            dtype=np.float64,
        )
        Q = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        for axis in (_AXIS_X, _AXIS_Y):
            idx = np.array(axis)
            Q[np.ix_(idx, idx)] = block
        Q[W, W] = self.size_psd * dt
        Q[H, H] = self.size_psd * dt
        return Q


class BoxMeasurement:
    """Observes centre and size directly: ``z = [cx, cy, w, h]``."""

    def __init__(self, pos_var: float = 4.0, size_var: float = 4.0):
        self.pos_var = float(pos_var)
        self.size_var = float(size_var)
        self._H = np.zeros((MEAS_DIM, STATE_DIM), dtype=np.float64)
        self._H[0, CX] = 1.0
        self._H[1, CY] = 1.0
        self._H[2, W] = 1.0
        self._H[3, H] = 1.0

    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        return self._H @ x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Linear model: the Jacobian is the constant observation matrix."""
        return self._H

    def noise(self) -> np.ndarray:
        return np.diag(
            np.array([self.pos_var, self.pos_var, self.size_var, self.size_var], dtype=np.float64)
        )


class TrackFilter:
    """Predict/update over a single target, driven by absolute timestamps."""

    def __init__(
        self,
        motion: ConstantAccelerationBox,
        measurement: BoxMeasurement,
        init_vel_var: float = 250.0**2,
        init_acc_var: float = 200.0**2,
    ):
        self.motion = motion
        self.measurement = measurement
        self.init_vel_var = float(init_vel_var)
        self.init_acc_var = float(init_acc_var)
        self.x = np.zeros(STATE_DIM, dtype=np.float64)
        self.P = np.eye(STATE_DIM, dtype=np.float64)
        self.t = 0.0
        self._diverged = False

    @property
    def diverged(self) -> bool:
        """True once the filter saw a state it cannot trust. Never auto-clears."""
        return self._diverged

    def initialize(self, z: Sequence[float], t: float) -> None:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        self.x = np.zeros(STATE_DIM, dtype=np.float64)
        self.x[CX], self.x[CY] = z[0], z[1]
        self.x[W], self.x[H] = z[2], z[3]

        P = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        P[CX, CX] = P[CY, CY] = self.measurement.pos_var
        P[VX, VX] = P[VY, VY] = self.init_vel_var
        P[AX, AX] = P[AY, AY] = self.init_acc_var
        P[W, W] = P[H, H] = self.measurement.size_var
        self.P = P
        self.t = float(t)
        self._diverged = False

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        F = self.motion.jacobian(self.x, dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.motion.process_noise(dt)
        self.t += float(dt)

    def predict_to(self, t: float) -> None:
        self.predict(float(t) - self.t)

    def innovation(self, z: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Residual ``y`` and its covariance ``S`` for measurement ``z``."""
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        H = self.measurement.jacobian(self.x)
        y = z - self.measurement.predict_measurement(self.x)
        S = H @ self.P @ H.T + self.measurement.noise()
        return y, S

    def center_mahalanobis2(self, z: Sequence[float]) -> float:
        """Squared Mahalanobis distance on the centre only (2 dof).

        Scales with the filter's own uncertainty, so the acceptance gate tightens
        after a good update and widens while coasting. Returns infinity when the
        innovation covariance is unusable, which reads as "reject".
        """
        y, S = self.innovation(z)
        y_c, S_c = y[:2], S[:2, :2]
        if not (np.all(np.isfinite(y_c)) and np.all(np.isfinite(S_c))):
            return float("inf")
        try:
            d2 = float(y_c @ np.linalg.solve(S_c, y_c))
        except np.linalg.LinAlgError:
            return float("inf")
        return d2 if np.isfinite(d2) else float("inf")

    def update(self, z: Sequence[float]) -> None:
        """Correct with ``z``. Refuses unusable input and flags divergence."""
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(z)):
            self._diverged = True
            return

        H = self.measurement.jacobian(self.x)
        R = self.measurement.noise()
        y, S = self.innovation(z)
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            self._diverged = True
            return

        x_new = self.x + K @ y
        I_KH = np.eye(STATE_DIM) - K @ H
        P_new = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        if not (np.all(np.isfinite(x_new)) and np.all(np.isfinite(P_new))):
            self._diverged = True
            return

        self.x = x_new
        self.P = P_new

    def state_at(self, dt_ahead: float) -> np.ndarray:
        """State extrapolated ``dt_ahead`` seconds without consuming time."""
        if dt_ahead <= 0.0:
            return self.x.copy()
        return self.motion.transition(dt_ahead) @ self.x

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.x[CX]), float(self.x[CY])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[VX]), float(self.x[VY])

    @property
    def size(self) -> Tuple[float, float]:
        return float(self.x[W]), float(self.x[H])
