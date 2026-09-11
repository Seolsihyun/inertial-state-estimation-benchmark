from __future__ import annotations

import numpy as np

from typing import Iterable

from models import Hoon_invariant_inekf as lie
from utils.filter_math import diagonal_covariance, kalman_update
from utils.math_utils import fit_diag, fit_vector
from models.Hoon_lie_group_utils import exp_so3, gamma2_so3, hat_so3, left_jacobian_so3, log_so3, minus_right, plus_right, symmetrize_covariance



class InEKF15D:
    """SE_2(3) InEKF with 15D error state [dR, dv, dp, dbg, dba].

    The group-state error uses the right-perturbation convention
    ``X_plus = X @ Exp(delta_xi)``. Process/measurement Jacobians and every
    group correction in this class use the same local coordinates.
    """

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
            raise ValueError("InEKF15D supports only 3d pose.")

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}
        init_cfg = initialization_config or {}

        self.pose_type = pose_type
        self.mode = mode
        self.error_dim = 15
        self.use_imu_velocity = bool(motion_cfg.get("use_imu_velocity", True))
        self.use_imu_rotation = bool(motion_cfg.get("use_imu_rotation", True))
        self.translation_input_frame = str(motion_cfg.get("translation_input_frame", "body"))
        self.translation_input_type = str(motion_cfg.get("translation_input_type", "acceleration"))
        self.rotation_input_type = str(motion_cfg.get("rotation_input_type", "rate"))
        self.rotation_representation = str(motion_cfg.get("rotation_representation", "rotvec"))
        self.velocity_blend = float(motion_cfg.get("velocity_blend", 0.0))
        self.update_biases = bool(motion_cfg.get("update_biases", True))
        self.covariance_floor = float(motion_cfg.get("covariance_floor", 1.0e-12))
        self.covariance_ceiling = float(motion_cfg.get("covariance_ceiling", 1.0e6))
        self.max_delta_norm = float(motion_cfg.get("max_delta_norm", 100.0))
        self.gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.process_noise_diag = fit_diag(
            motion_cfg.get(
                "process_noise_diag",
                [1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5],
            ),
            self.error_dim,
        )
        self.process_noise_scale = float(motion_cfg.get("process_noise_scale", 1.0))
        self.process_noise_diag = self.process_noise_diag * max(self.process_noise_scale, 0.0)
        self.measurement_noise_diag = fit_diag(
            meas_cfg.get("measurement_noise_diag", [1.0, 1.0, 1.0]),
            3,
        )
        self.innovation_gate_m = float(meas_cfg.get("innovation_gate_m", 0.0))
        self.mahalanobis_gate = float(meas_cfg.get("mahalanobis_gate", 0.0))

        self.Rot = np.eye(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.p = np.zeros(3, dtype=float)
        self.gyro_bias = fit_vector(motion_cfg.get("gyro_bias", [0.0, 0.0, 0.0]), 3)
        self.accel_bias = fit_vector(motion_cfg.get("accel_bias", [0.0, 0.0, 0.0]), 3)
        self.P = np.eye(self.error_dim, dtype=float)
        self.Phi = np.eye(self.error_dim, dtype=float)
        self.Q = diagonal_covariance(self.process_noise_diag)
        self.H = self._position_measurement_jacobian()
        self.Rm = diagonal_covariance(self.measurement_noise_diag)
        self.innovation = np.zeros(3, dtype=float)
        self.K = np.zeros((self.error_dim, 3), dtype=float)
        self.delta = np.zeros(self.error_dim, dtype=float)
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        self.initialized = False
        self.initialize(init_cfg.get("mean"), init_cfg.get("cov_diag"), init_cfg.get("velocity_mean"))

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "InEKF15D":
        cfg = compare_config.get("Hoon_invariant_kalman_filter_15d", compare_config.get("invariant_kalman_filter_15d", compare_config.get("invariant_kalman_filter", compare_config)))
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
        self.p, self.Rot = lie.pose_to_state(pose)
        self.v = fit_vector(np.zeros(3) if velocity_mean is None else np.asarray(velocity_mean, dtype=float).reshape(-1), 3)
        cov = fit_diag(np.ones(self.error_dim) * 1e-3 if cov_diag is None else cov_diag, self.error_dim)
        self.P = diagonal_covariance(cov)
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()
        if control is None:
            return self.estimate_pose()

        u = np.asarray(control, dtype=float).reshape(-1)
        if u.size < 6:
            raise ValueError("3D InEKF control must contain [ax, ay, az, gx, gy, gz].")
        dt = float(dt)
        dt = max(dt, 0.0)
        X_prev = lie.as_matrix(self.Rot, self.v, self.p)
        gyro_bias_prev = self.gyro_bias.copy()
        accel_bias_prev = self.accel_bias.copy()

        self.Rot, self.v, self.p = self._propagate_nominal(
            self.Rot,
            self.v,
            self.p,
            gyro_bias_prev,
            accel_bias_prev,
            u,
            dt,
        )
        X_pred = lie.as_matrix(self.Rot, self.v, self.p)

        # Hoon path: compute the discrete error transition by perturbing the
        # previous SE_2(3) state with plus_right and measuring the propagated
        # residual with minus_right. This uses the Lie utilities rather than a
        # hand-written continuous-time A matrix.
        self.Phi = self._process_jacobian(X_prev, gyro_bias_prev, accel_bias_prev, u, dt, X_pred)
        self.Q = diagonal_covariance(self.process_noise_diag)
        predicted = self.Phi @ self.P @ self.Phi.T + self.Phi @ self.Q @ self.Phi.T * max(dt, 1e-9)
        self.P = self._stabilize_covariance(predicted)
        self.X = X_pred
        return self.estimate_pose()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        z = np.asarray(measurement, dtype=float).reshape(3)
        self.innovation = z - self.p
        if self._reject_measurement():
            return self.estimate_pose()

        self.H = self._position_measurement_jacobian()
        self.Rm = diagonal_covariance(self.measurement_noise_diag)
        _, P_update, self.innovation, self.S, self.K = kalman_update(
            np.zeros(self.error_dim, dtype=float), self.P, self.innovation, self.H, self.Rm
        )
        self.delta = self._bounded_delta(self.K @ self.innovation)
        self.Rot, self.v, self.p = lie.from_matrix(plus_right(lie.as_matrix(self.Rot, self.v, self.p), self.delta[:9]))
        if self.update_biases and self.use_imu_rotation and self.rotation_input_type == "rate":
            self.gyro_bias = self.gyro_bias + self.delta[9:12]
        if self.update_biases and self.translation_input_type == "acceleration":
            self.accel_bias = self.accel_bias + self.delta[12:15]
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        self.P = self._stabilize_covariance(P_update)
        return self.estimate_pose()

    def velocity_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        z = np.asarray(measurement, dtype=float).reshape(3)
        self.innovation = z - self.v
        if self.innovation_gate_m > 0.0 and np.linalg.norm(self.innovation) > self.innovation_gate_m:
            return self.estimate_pose()

        H = self._velocity_measurement_jacobian()
        Rm = diagonal_covariance(fit_diag(self.measurement_noise_diag, 3))
        S = H @ self.P @ H.T + Rm + 1e-12 * np.eye(3)
        if self.mahalanobis_gate > 0.0:
            maha = float(self.innovation.T @ np.linalg.solve(S, self.innovation))
            if maha > self.mahalanobis_gate:
                return self.estimate_pose()

        _, P_update, self.innovation, self.S, self.K = kalman_update(
            np.zeros(self.error_dim, dtype=float), self.P, self.innovation, H, Rm
        )
        self.delta = self._bounded_delta(self.K @ self.innovation)
        self.Rot, self.v, self.p = lie.from_matrix(plus_right(lie.as_matrix(self.Rot, self.v, self.p), self.delta[:9]))
        if self.update_biases and self.use_imu_rotation and self.rotation_input_type == "rate":
            self.gyro_bias = self.gyro_bias + self.delta[9:12]
        if self.update_biases and self.translation_input_type == "acceleration":
            self.accel_bias = self.accel_bias + self.delta[12:15]
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        self.P = self._stabilize_covariance(P_update)
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
        return lie.pose_from_state(self.Rot, self.p)

    def _propagate_nominal(
        self,
        Rot: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        gyro_bias: np.ndarray,
        accel_bias: np.ndarray,
        control: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Rot = np.asarray(Rot, dtype=float).reshape(3, 3)
        v = np.asarray(v, dtype=float).reshape(3).copy()
        p = np.asarray(p, dtype=float).reshape(3).copy()
        u = np.asarray(control, dtype=float).reshape(-1)
        dt = max(float(dt), 0.0)
        dt2 = dt * dt

        R_prev = Rot
        v_prev = v.copy()
        use_accel_bias = self.translation_input_type == "acceleration"
        use_gyro_bias = self.use_imu_rotation and self.rotation_input_type == "rate"
        w = u[3:6] - gyro_bias if use_gyro_bias else u[3:6]
        a = u[0:3] - accel_bias if use_accel_bias else u[0:3]
        phi = w * dt
        G1 = left_jacobian_so3(phi)
        G2 = gamma2_so3(phi)

        if self.use_imu_rotation:
            if self.rotation_input_type == "rate":
                rot_vec = phi
            elif self.rotation_input_type == "increment":
                rot_vec = w
            else:
                raise ValueError(f"Unsupported rotation_input_type: {self.rotation_input_type}")

            if self.rotation_representation == "rotvec":
                Rot = R_prev @ exp_so3(rot_vec)
            elif self.rotation_representation == "euler":
                Rot = lie.rpy_to_rot(lie.rot_to_rpy(R_prev) + rot_vec)
            else:
                raise ValueError(f"Unsupported rotation_representation: {self.rotation_representation}")

        if self.use_imu_velocity:
            if self.translation_input_frame == "body":
                trans_world = R_prev @ G1 @ a if self.translation_input_type == "acceleration" else R_prev @ a
            elif self.translation_input_frame == "world":
                trans_world = a
            else:
                raise ValueError(f"Unsupported translation_input_frame: {self.translation_input_frame}")

            if self.translation_input_type == "acceleration":
                accel_world = trans_world + self.gravity
                if self.translation_input_frame == "body":
                    p = p + v_prev * dt + R_prev @ G2 @ a * dt2 + 0.5 * self.gravity * dt2
                    v = v + accel_world * dt
                else:
                    p = p + v_prev * dt + 0.5 * accel_world * dt2
                    v = v + accel_world * dt
            elif self.translation_input_type == "velocity":
                v = self.velocity_blend * v + (1.0 - self.velocity_blend) * trans_world
                p = p + v * dt
            elif self.translation_input_type == "increment":
                p = p + trans_world
                if dt > 1e-12:
                    inferred_v = trans_world / dt
                    v = self.velocity_blend * v + (1.0 - self.velocity_blend) * inferred_v
            else:
                raise ValueError(f"Unsupported translation_input_type: {self.translation_input_type}")
        else:
            p = p + v * dt
        return Rot, v, p

    def _process_jacobian(
        self,
        X_prev: np.ndarray,
        gyro_bias_prev: np.ndarray,
        accel_bias_prev: np.ndarray,
        control: np.ndarray,
        dt: float,
        X_pred: np.ndarray,
        eps: float = 1.0e-6,
    ) -> np.ndarray:
        def residual(delta: np.ndarray) -> np.ndarray:
            X_pert = plus_right(X_prev, delta[:9])
            Rot_pert, v_pert, p_pert = lie.from_matrix(X_pert)
            gyro_bias_pert = gyro_bias_prev + delta[9:12]
            accel_bias_pert = accel_bias_prev + delta[12:15]
            Rot_next, v_next, p_next = self._propagate_nominal(
                Rot_pert,
                v_pert,
                p_pert,
                gyro_bias_pert,
                accel_bias_pert,
                control,
                dt,
            )
            X_next = lie.as_matrix(Rot_next, v_next, p_next)
            out = np.zeros(self.error_dim, dtype=float)
            out[:9] = minus_right(X_next, X_pred)
            out[9:12] = gyro_bias_pert - gyro_bias_prev
            out[12:15] = accel_bias_pert - accel_bias_prev
            return out

        return _finite_difference_jacobian(residual, self.error_dim, eps)

    def _position_measurement_jacobian(self, eps: float = 1.0e-6) -> np.ndarray:
        X = lie.as_matrix(self.Rot, self.v, self.p)
        base = self.p.copy()

        def measurement(delta: np.ndarray) -> np.ndarray:
            _Rot, _v, p = lie.from_matrix(plus_right(X, delta[:9]))
            return p - base

        return _finite_difference_jacobian(measurement, self.error_dim, eps)

    def _velocity_measurement_jacobian(self, eps: float = 1.0e-6) -> np.ndarray:
        X = lie.as_matrix(self.Rot, self.v, self.p)
        base = self.v.copy()

        def measurement(delta: np.ndarray) -> np.ndarray:
            _Rot, v, _p = lie.from_matrix(plus_right(X, delta[:9]))
            return v - base

        return _finite_difference_jacobian(measurement, self.error_dim, eps)

    def _reject_measurement(self) -> bool:
        if self.innovation_gate_m > 0.0 and np.linalg.norm(self.innovation) > self.innovation_gate_m:
            return True
        self.H = self._position_measurement_jacobian()
        self.Rm = diagonal_covariance(self.measurement_noise_diag)
        self.S = self.H @ self.P @ self.H.T + self.Rm + 1e-12 * np.eye(3)
        if self.mahalanobis_gate <= 0.0:
            return False
        maha = float(self.innovation.T @ np.linalg.solve(self.S, self.innovation))
        return maha > self.mahalanobis_gate

    def _bounded_delta(self, delta: np.ndarray) -> np.ndarray:
        delta = np.nan_to_num(np.asarray(delta, dtype=float).reshape(self.error_dim), nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(delta))
        if self.max_delta_norm > 0.0 and norm > self.max_delta_norm:
            delta = delta * (self.max_delta_norm / norm)
        return delta

    def _stabilize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        return symmetrize_covariance(covariance, floor=self.covariance_floor, ceiling=self.covariance_ceiling)


def _finite_difference_jacobian(fn, dim: int, eps: float) -> np.ndarray:
    zero = np.zeros(dim, dtype=float)
    output_dim = np.asarray(fn(zero), dtype=float).reshape(-1).size
    J = np.zeros((output_dim, dim), dtype=float)
    for col in range(dim):
        step = np.zeros(dim, dtype=float)
        step[col] = eps
        J[:, col] = (fn(step) - fn(-step)) / (2.0 * eps)
    return np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

def _matrix_exp(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norm = float(np.linalg.norm(matrix, ord=np.inf))
    scale = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0.5 else 0
    A = matrix / (2**scale)
    result = np.eye(A.shape[0], dtype=float)
    term = np.eye(A.shape[0], dtype=float)
    for order in range(1, 24):
        term = term @ A / float(order)
        result = result + term
        if np.linalg.norm(term, ord=np.inf) < 1e-14:
            break
    for _ in range(scale):
        result = result @ result
    return result

class InEKFAnalytic15D(InEKF15D):
    """InEKF 15D with selectable finite/analytic Jacobians.

    State error order is [dR, dv, dp, dbg, dba]. Perturbations follow the
    existing Hoon right convention: X_plus = X @ Exp(delta).
    """

    def __init__(self, *args, **kwargs) -> None:
        motion_config = kwargs.get("motion_config") or {}
        self.jacobian_mode = str(motion_config.get("jacobian_mode", "analytic")).lower()
        if self.jacobian_mode not in {"finite", "analytic"}:
            raise ValueError("jacobian_mode must be 'finite' or 'analytic'.")
        super().__init__(*args, **kwargs)

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "InEKFAnalytic15D":
        cfg = compare_config.get(
            "Hoon_invariant_kalman_analytic_15d",
            compare_config.get("Hoon_invariant_kalman_filter_15d", compare_config.get("invariant_kalman_filter_15d", compare_config)),
        )
        return cls(
            pose_type=dataset_config.get("pose_type", cfg.get("pose_type", "3d")),
            mode=dataset_config.get("mode", cfg.get("mode", "fused")),
            motion_config=cfg.get("motion_model", {}),
            measurement_config=cfg.get("measurement_model", {}),
            initialization_config=cfg.get("initialization", {}),
        )

    def _process_jacobian(
        self,
        X_prev: np.ndarray,
        gyro_bias_prev: np.ndarray,
        accel_bias_prev: np.ndarray,
        control: np.ndarray,
        dt: float,
        X_pred: np.ndarray,
        eps: float = 1.0e-6,
    ) -> np.ndarray:
        if self.jacobian_mode == "finite":
            return super()._process_jacobian(X_prev, gyro_bias_prev, accel_bias_prev, control, dt, X_pred, eps=eps)
        return self._analytic_process_jacobian(X_prev, gyro_bias_prev, accel_bias_prev, control, dt)

    def _position_measurement_jacobian(self, eps: float = 1.0e-6) -> np.ndarray:
        if getattr(self, "jacobian_mode", "finite") == "finite":
            return super()._position_measurement_jacobian(eps=eps)
        H = np.zeros((3, self.error_dim), dtype=float)
        # For right perturbation X @ Exp([dR,dv,dp]), p_plus = p + R * dp at first order.
        H[:, 6:9] = self.Rot
        return H

    def _velocity_measurement_jacobian(self, eps: float = 1.0e-6) -> np.ndarray:
        if getattr(self, "jacobian_mode", "finite") == "finite":
            return super()._velocity_measurement_jacobian(eps=eps)
        H = np.zeros((3, self.error_dim), dtype=float)
        # For right perturbation X @ Exp([dR,dv,dp]), v_plus = v + R * dv at first order.
        H[:, 3:6] = self.Rot
        return H

    def finite_process_jacobian_for_validation(self, X_prev, gyro_bias_prev, accel_bias_prev, control, dt, X_pred, eps=1e-6):
        return super()._process_jacobian(X_prev, gyro_bias_prev, accel_bias_prev, control, dt, X_pred, eps=eps)

    def finite_position_jacobian_for_validation(self, eps=1e-6):
        return super()._position_measurement_jacobian(eps=eps)

    def finite_velocity_jacobian_for_validation(self, eps=1e-6):
        return super()._velocity_measurement_jacobian(eps=eps)

    def _analytic_process_jacobian(
        self,
        X_prev: np.ndarray,
        gyro_bias_prev: np.ndarray,
        accel_bias_prev: np.ndarray,
        control: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        Rot, _v, _p = lie.from_matrix(X_prev)
        u = np.asarray(control, dtype=float).reshape(-1)
        dt = max(float(dt), 0.0)
        dt2 = dt * dt
        use_accel_bias = self.translation_input_type == "acceleration"
        use_gyro_bias = self.use_imu_rotation and self.rotation_input_type == "rate"
        w = u[3:6] - gyro_bias_prev if use_gyro_bias else u[3:6]
        a = u[0:3] - accel_bias_prev if use_accel_bias else u[0:3]
        phi = w * dt if self.rotation_input_type == "rate" else w
        E = exp_so3(phi)
        A = E.T
        G1 = left_jacobian_so3(phi)
        G2 = gamma2_so3(phi)
        acc_v = G1 @ a * dt
        acc_p = G2 @ a * dt2

        Phi = np.eye(self.error_dim, dtype=float)
        Phi[:9, :9] = 0.0
        Phi[0:3, 0:3] = A
        if self.use_imu_velocity and self.translation_input_frame == "body" and self.translation_input_type == "acceleration":
            Phi[3:6, 0:3] = -A @ hat_so3(acc_v)
            Phi[3:6, 3:6] = A
            Phi[6:9, 0:3] = -A @ hat_so3(acc_p)
            Phi[6:9, 3:6] = A * dt
            Phi[6:9, 6:9] = A
            Phi[3:6, 12:15] = -A @ G1 * dt
            Phi[6:9, 12:15] = -A @ G2 * dt2
        elif self.use_imu_velocity and self.translation_input_frame == "world" and self.translation_input_type == "acceleration":
            Phi[3:6, 3:6] = A
            Phi[6:9, 3:6] = A * dt
            Phi[6:9, 6:9] = A
            Phi[3:6, 12:15] = -A * dt
            Phi[6:9, 12:15] = -0.5 * A * dt2
        else:
            Phi[3:6, 3:6] = A
            Phi[6:9, 3:6] = A * dt
            Phi[6:9, 6:9] = A

        if use_gyro_bias:
            dtheta_dbg, dv_dbg, dp_dbg = self._gyro_bias_coupling_blocks(phi, a, dt)
            Phi[0:3, 9:12] = dtheta_dbg
            if self.use_imu_velocity and self.translation_input_frame == "body" and self.translation_input_type == "acceleration":
                Phi[3:6, 9:12] = dv_dbg
                Phi[6:9, 9:12] = dp_dbg
        Phi[9:12, 9:12] = np.eye(3)
        Phi[12:15, 12:15] = np.eye(3)
        return np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)

    def _gyro_bias_coupling_blocks(self, phi: np.ndarray, accel_body: np.ndarray, dt: float, eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local SO(3) derivative of right-error residuals wrt gyro bias.

        This avoids a full 15D process finite difference while preserving the
        right-perturbation residual convention used by the Hoon filter.
        """
        phi = np.asarray(phi, dtype=float).reshape(3)
        accel_body = np.asarray(accel_body, dtype=float).reshape(3)
        E = exp_so3(phi)
        A = E.T
        base_v = left_jacobian_so3(phi) @ accel_body * dt
        base_p = gamma2_so3(phi) @ accel_body * dt * dt
        dtheta_dbg = np.zeros((3, 3), dtype=float)
        dv_dbg = np.zeros((3, 3), dtype=float)
        dp_dbg = np.zeros((3, 3), dtype=float)
        for col in range(3):
            step = np.zeros(3, dtype=float)
            step[col] = eps
            # gyro bias perturbation db changes phi to phi - db * dt.
            phi_p = phi - step * dt
            phi_m = phi + step * dt
            dtheta_p = log_so3(A @ exp_so3(phi_p))
            dtheta_m = log_so3(A @ exp_so3(phi_m))
            v_p = A @ (left_jacobian_so3(phi_p) @ accel_body * dt - base_v)
            v_m = A @ (left_jacobian_so3(phi_m) @ accel_body * dt - base_v)
            p_p = A @ (gamma2_so3(phi_p) @ accel_body * dt * dt - base_p)
            p_m = A @ (gamma2_so3(phi_m) @ accel_body * dt * dt - base_p)
            dtheta_dbg[:, col] = (dtheta_p - dtheta_m) / (2.0 * eps)
            dv_dbg[:, col] = (v_p - v_m) / (2.0 * eps)
            dp_dbg[:, col] = (p_p - p_m) / (2.0 * eps)
        return dtheta_dbg, dv_dbg, dp_dbg


def analytic_position_measurement_jacobian_from_state(Rot: np.ndarray, error_dim: int = 15) -> np.ndarray:
    H = np.zeros((3, error_dim), dtype=float)
    H[:, 6:9] = np.asarray(Rot, dtype=float).reshape(3, 3)
    return H


def finite_position_measurement_jacobian_from_state(X: np.ndarray, error_dim: int = 15, eps: float = 1e-6) -> np.ndarray:
    base = lie.from_matrix(X)[2]

    def measurement(delta: np.ndarray) -> np.ndarray:
        _Rot, _v, p = lie.from_matrix(plus_right(X, delta[:9]))
        return p - base

    return _finite_difference_jacobian(measurement, error_dim, eps)


# Backward-compatible aliases for old experiment scripts.
HoonInvariantKalmanFilter15D = InEKF15D
HoonInvariantKalmanAnalytic15D = InEKFAnalytic15D

# Short name used by the runners and filter registry.
InEKF = InEKFAnalytic15D
__all__ = ["InEKF", "InEKF15D", "InEKFAnalytic15D"]
