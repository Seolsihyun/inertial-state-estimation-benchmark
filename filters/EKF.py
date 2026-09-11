from __future__ import annotations

from typing import Iterable

import numpy as np

from filters.UKF import _quat_from_rot, _quat_from_rotvec, _quat_multiply, _quat_normalize, _rot_from_quat
from models import Hoon_invariant_inekf as lie
from utils.filter_math import diagonal_covariance, kalman_update
from utils.math_utils import fit_diag, fit_vector


class QuaternionExtendedKalmanFilter15D:
    """Additive quaternion-state EKF.

    State order:
        [p, v, q, bg, ba]

    The quaternion is part of the Kalman state and is corrected additively by
    the EKF update. After every predict/update, q is normalized to keep the
    unit-quaternion constraint.

    Note:
        A quaternion attitude has four stored components, so the internal
        Kalman state is 16D: 3 + 3 + 4 + 3 + 3. The file name keeps the project
        convention, but this is not a 15D error-state filter.
    """

    state_dim = 16

    def __init__(
        self,
        pose_type: str = "3d",
        mode: str = "fused",
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
        initialization_config: dict | None = None,
    ) -> None:
        if pose_type == "6d":
            pose_type = "3d"
        if pose_type != "3d":
            raise ValueError("QuaternionExtendedKalmanFilter15D supports only 3d pose.")

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}
        init_cfg = initialization_config or {}

        self.pose_type = pose_type
        self.mode = mode
        self.translation_input_frame = str(motion_cfg.get("translation_input_frame", "body"))
        self.translation_input_type = str(motion_cfg.get("translation_input_type", "acceleration"))
        self.rotation_input_type = str(motion_cfg.get("rotation_input_type", "rate"))
        self.gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.gyro_bias = fit_vector(motion_cfg.get("gyro_bias", [0.0, 0.0, 0.0]), 3)
        self.accel_bias = fit_vector(motion_cfg.get("accel_bias", [0.0, 0.0, 0.0]), 3)
        self.process_noise_diag = _map_common_noise_to_quaternion_state_order(
            motion_cfg.get(
                "process_noise_diag",
                [1e-5, 1e-5, 1e-5, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 1e-7, 1e-7, 1e-7, 1e-5, 1e-5, 1e-5],
            )
        )
        self.measurement_noise_diag = fit_diag(meas_cfg.get("measurement_noise_diag", [1.0, 1.0, 1.0]), 3)
        self.velocity_measurement_noise_diag = fit_diag(
            meas_cfg.get("velocity_measurement_noise_diag", self.measurement_noise_diag),
            3,
        )
        self.innovation_gate_m = float(meas_cfg.get("innovation_gate_m", 0.0))
        self.mahalanobis_gate = float(meas_cfg.get("mahalanobis_gate", 0.0))
        self.covariance_floor = float(motion_cfg.get("covariance_floor", 1.0e-12))
        self.covariance_ceiling = float(motion_cfg.get("covariance_ceiling", 1.0e8))

        self.x = np.zeros(self.state_dim, dtype=float)
        self.P = np.eye(self.state_dim, dtype=float)
        self.Phi = np.eye(self.state_dim, dtype=float)
        self.Q = diagonal_covariance(self.process_noise_diag)
        self.H = np.zeros((3, self.state_dim), dtype=float)
        self.H[:, 0:3] = np.eye(3)
        self.Rm = diagonal_covariance(self.measurement_noise_diag)
        self.innovation = np.zeros(3, dtype=float)
        self.K = np.zeros((self.state_dim, 3), dtype=float)
        self.initialized = False
        self.initialize(init_cfg.get("mean"), init_cfg.get("cov_diag"), init_cfg.get("velocity_mean"))

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "QuaternionExtendedKalmanFilter15D":
        cfg = compare_config.get(
            "quaternion_ekf_15d",
            compare_config.get("Quaternion_ekf_15d", compare_config.get("euler_ekf_15d", compare_config)),
        )
        return cls(
            pose_type=dataset_config.get("pose_type", cfg.get("pose_type", "3d")),
            mode=dataset_config.get("mode", cfg.get("mode", "fused")),
            motion_config=cfg.get("motion_model", {}),
            measurement_config=cfg.get("measurement_model", {}),
            initialization_config=cfg.get("initialization", {}),
        )

    def initialize(
        self,
        mean: Iterable[float] | None = None,
        cov_diag: Iterable[float] | None = None,
        velocity_mean: Iterable[float] | None = None,
    ) -> None:
        pose = fit_vector(np.zeros(6) if mean is None else np.asarray(mean, dtype=float).reshape(-1), 6)
        p0, Rot0 = lie.pose_to_state(pose)
        self.x[:] = 0.0
        self.x[0:3] = p0
        self.x[3:6] = fit_vector(np.zeros(3) if velocity_mean is None else np.asarray(velocity_mean, dtype=float).reshape(-1), 3)
        self.x[6:10] = _quat_from_rot(Rot0)
        self.x[10:13] = self.gyro_bias
        self.x[13:16] = self.accel_bias
        self.P = diagonal_covariance(_map_common_noise_to_quaternion_state_order(cov_diag))
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()
        if control is None:
            return self.estimate_pose()
        u = np.asarray(control, dtype=float).reshape(-1)
        if u.size < 6:
            raise ValueError("Quaternion EKF control must contain [ax, ay, az, gx, gy, gz].")

        x_prev = self.x.copy()
        self.x = self._propagate_vector(x_prev, u, float(dt))
        self.Phi = self._finite_difference_jacobian(x_prev, u, float(dt))
        self.Q = diagonal_covariance(self.process_noise_diag)
        predicted = self.Phi @ self.P @ self.Phi.T + self.Phi @ self.Q @ self.Phi.T * max(float(dt), 1e-9)
        self.P = self._stabilize(predicted)
        return self.estimate_pose()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        z = np.asarray(measurement, dtype=float).reshape(3)
        self.innovation = z - self.x[0:3]
        self.H = np.zeros((3, self.state_dim), dtype=float)
        self.H[:, 0:3] = np.eye(3)
        self.Rm = diagonal_covariance(self.measurement_noise_diag)
        if self._reject_measurement(self.innovation, self.H, self.Rm):
            return self.estimate_pose()
        self.x, P_update, self.innovation, self.S, self.K = kalman_update(self.x, self.P, z, self.H, self.Rm)
        self._normalize_state_quaternion()
        self.P = self._stabilize(P_update)
        return self.estimate_pose()

    def velocity_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        z = np.asarray(measurement, dtype=float).reshape(3)
        innovation = z - self.x[3:6]
        H = np.zeros((3, self.state_dim), dtype=float)
        H[:, 3:6] = np.eye(3)
        Rm = diagonal_covariance(self.velocity_measurement_noise_diag)
        if self._reject_measurement(innovation, H, Rm):
            return self.estimate_pose()
        self.x, P_update, self.innovation, self.S, self.K = kalman_update(self.x, self.P, z, H, Rm)
        self._normalize_state_quaternion()
        self.P = self._stabilize(P_update)
        return self.estimate_pose()

    def step(
        self,
        control: Iterable[float] | None,
        measurement: Iterable[float] | None,
        dt: float,
        mode: str | None = None,
    ) -> np.ndarray:
        run_mode = self.mode if mode is None else mode
        if run_mode in {"imu_only", "fused"}:
            self.predict(control, dt)
        if run_mode in {"gnss_only", "fused"}:
            self.measurement_update(measurement)
        return self.estimate_pose()

    def estimate_pose(self) -> np.ndarray:
        return lie.pose_from_state(_rot_from_quat(self.x[6:10]), self.x[0:3])

    def _propagate_vector(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        dt = max(float(dt), 0.0)
        out = np.asarray(x, dtype=float).reshape(self.state_dim).copy()
        p = out[0:3].copy()
        v = out[3:6].copy()
        q = _quat_normalize(out[6:10])
        bg = out[10:13]
        ba = out[13:16]
        R_prev = _rot_from_quat(q)
        w = u[3:6] - bg
        a = u[0:3] - ba
        phi = w * dt if self.rotation_input_type == "rate" else w
        q_next = _quat_multiply(q, _quat_from_rotvec(phi))

        if self.translation_input_type != "acceleration":
            raise ValueError("Quaternion EKF currently supports acceleration input only.")
        if self.translation_input_frame == "body":
            accel_world = R_prev @ a + self.gravity
        elif self.translation_input_frame == "world":
            accel_world = a + self.gravity
        else:
            raise ValueError(f"Unsupported translation_input_frame: {self.translation_input_frame}")

        out[0:3] = p + v * dt + 0.5 * accel_world * dt * dt
        out[3:6] = v + accel_world * dt
        out[6:10] = _quat_normalize(q_next)
        return out

    def _finite_difference_jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        eps = 1.0e-6
        J = np.zeros((self.state_dim, self.state_dim), dtype=float)
        for idx in range(self.state_dim):
            xp = np.asarray(x, dtype=float).copy()
            xm = np.asarray(x, dtype=float).copy()
            xp[idx] += eps
            xm[idx] -= eps
            xp[6:10] = _quat_normalize(xp[6:10])
            xm[6:10] = _quat_normalize(xm[6:10])
            fp = self._propagate_vector(xp, u, dt)
            fm = self._propagate_vector(xm, u, dt)
            J[:, idx] = (fp - fm) / (2.0 * eps)
        return J

    def _reject_measurement(self, innovation: np.ndarray, H: np.ndarray, Rm: np.ndarray) -> bool:
        if self.innovation_gate_m > 0.0 and np.linalg.norm(innovation) > self.innovation_gate_m:
            return True
        S = H @ self.P @ H.T + Rm + 1e-12 * np.eye(3)
        self.S = S
        if self.mahalanobis_gate <= 0.0:
            return False
        maha = float(innovation.T @ np.linalg.solve(S, innovation))
        return maha > self.mahalanobis_gate

    def _normalize_state_quaternion(self) -> None:
        self.x[6:10] = _quat_normalize(self.x[6:10])

    def _stabilize(self, P: np.ndarray) -> np.ndarray:
        P = np.nan_to_num(np.asarray(P, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        P = 0.5 * (P + P.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.clip(eigvals, self.covariance_floor, self.covariance_ceiling)
            P = (eigvecs * eigvals) @ eigvecs.T
            return 0.5 * (P + P.T)
        except np.linalg.LinAlgError:
            diag = np.clip(np.diag(P), self.covariance_floor, self.covariance_ceiling)
            return np.diag(diag)


# Common 15D noise order in this project is [dtheta, dv, dp, dbg, dba].
# Additive quaternion-state EKF order is [p, v, q, bg, ba].
def _map_common_noise_to_quaternion_state_order(value: Iterable[float] | None) -> np.ndarray:
    if value is None:
        return np.ones(16, dtype=float) * 1e-3
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 16:
        return fit_diag(arr, 16)
    if arr.size != 15:
        return fit_diag(arr, 16)
    q_noise = np.ones(4, dtype=float) * float(np.mean(arr[0:3]))
    return np.concatenate([arr[6:9], arr[3:6], q_noise, arr[9:12], arr[12:15]])

# Short name used by the runners and filter registry.
EKF = QuaternionExtendedKalmanFilter15D
__all__ = ["EKF", "QuaternionExtendedKalmanFilter15D"]
