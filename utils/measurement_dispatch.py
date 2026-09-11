from __future__ import annotations

import numpy as np

from utils.filter_math import diagonal_covariance, diagonal_gaussian_logpdf, kalman_update


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    s = np.sum(np.sin(values) * weights[:, None], axis=0)
    c = np.sum(np.cos(values) * weights[:, None], axis=0)
    return np.arctan2(s, c)


def apply_optional_measurements(estimator, *, position=None, velocity=None, attitude=None, noise_config=None) -> int:
    """Apply optional position, velocity, and attitude measurements.

    Position is the default measurement in every filter. Velocity is applied by
    an existing ``velocity_update`` method when available; for PF it is a
    velocity likelihood. Attitude is handled as an RPY measurement: Kalman-style
    filters update the attitude error state, while PF multiplies an attitude
    likelihood into the particle weights.
    """

    noise_config = noise_config or {}
    count = 0
    if position is not None:
        estimator.measurement_update(position)
        count += 1
    if velocity is not None:
        _velocity_update(estimator, velocity, noise_config)
        count += 1
    if attitude is not None:
        _attitude_update(estimator, attitude, noise_config)
        count += 1
    return count


def _velocity_update(estimator, measurement, noise_config: dict) -> None:
    if hasattr(estimator, "velocity_update"):
        estimator.velocity_update(measurement)
        return
    if hasattr(estimator, "particles_v") and hasattr(estimator, "weights"):
        z = np.asarray(measurement, dtype=float).reshape(3)
        var = np.asarray(noise_config.get("velocity_measurement_noise_diag", [1.0, 1.0, 1.0]), dtype=float)
        innovation = estimator.particles_v - z[None, :]
        log_likelihood = diagonal_gaussian_logpdf(innovation, np.clip(var, 1e-12, None))
        estimator.weights *= np.exp(log_likelihood - np.max(log_likelihood))
        estimator.normalize()
        if estimator.effective_sample_size() < estimator.threshold:
            estimator.resample()
        return
    raise NotImplementedError(f"{type(estimator).__name__} does not support velocity measurement updates.")


def _attitude_update(estimator, measurement, noise_config: dict) -> None:
    z = np.asarray(measurement, dtype=float).reshape(3)
    var = np.asarray(noise_config.get("attitude_measurement_noise_diag", [1.0e-2, 1.0e-2, 1.0e-2]), dtype=float)
    if hasattr(estimator, "particles_q") and hasattr(estimator, "weights"):
        _pf_attitude_update(estimator, z, var)
    elif hasattr(estimator, "q") and hasattr(estimator, "_update_linear_measurement"):
        _ukf_attitude_update(estimator, z, var)
    elif hasattr(estimator, "x") and getattr(estimator, "state_dim", None) == 16:
        _additive_quaternion_ekf_attitude_update(estimator, z, var)
    elif hasattr(estimator, "Rot") and hasattr(estimator, "P") and hasattr(estimator, "error_dim"):
        _rotation_error_state_attitude_update(estimator, z, var)
    else:
        raise NotImplementedError(f"{type(estimator).__name__} does not support attitude measurement updates.")


def _pf_attitude_update(estimator, z: np.ndarray, var: np.ndarray) -> None:
    from filters.UKF import _rot_from_quat
    from models import Hoon_invariant_inekf as lie

    values = np.array([lie.rot_to_rpy(_rot_from_quat(q)) for q in estimator.particles_q], dtype=float)
    innovation = wrap_angle(values - z[None, :])
    log_likelihood = diagonal_gaussian_logpdf(innovation, np.clip(var, 1e-12, None))
    estimator.weights *= np.exp(log_likelihood - np.max(log_likelihood))
    estimator.normalize()
    if estimator.effective_sample_size() < estimator.threshold:
        estimator.resample()


def _ukf_attitude_update(estimator, z: np.ndarray, var: np.ndarray) -> None:
    from filters.UKF import _quat_from_rotvec, _quat_multiply, _rot_from_quat, _weighted_outer
    from models import Hoon_invariant_inekf as lie
    from utils.sigma_points import merwe_sigma_points

    sigmas, Wm, Wc, _ = merwe_sigma_points(np.zeros(estimator.error_dim), estimator.P, estimator.alpha, estimator.beta, estimator.kappa)
    q, v, p, bg, ba = estimator._compose_batch(sigmas)
    values = np.array([lie.rot_to_rpy(_rot_from_quat(qi)) for qi in q], dtype=float)
    z_mean = circular_mean(values, Wm)
    errors = estimator._state_errors_batch(q, v, p, bg, ba)
    dz = wrap_angle(values - z_mean[None, :])
    S = _weighted_outer(dz, Wc) + diagonal_covariance(var) + 1e-12 * np.eye(3)
    Pxz = np.einsum("n,ni,nj->ij", Wc, errors, dz)
    estimator.K = Pxz @ np.linalg.inv(S)
    estimator.innovation = wrap_angle(z - z_mean)
    delta = estimator._bounded_delta(estimator.K @ estimator.innovation)
    estimator.q = _quat_multiply(estimator.q, _quat_from_rotvec(delta[0:3]))
    estimator.v = estimator.v + delta[3:6]
    estimator.p = estimator.p + delta[6:9]
    estimator.gyro_bias = estimator.gyro_bias + delta[9:12]
    estimator.accel_bias = estimator.accel_bias + delta[12:15]
    estimator.P = estimator._stabilize(estimator.P - estimator.K @ S @ estimator.K.T)


def _additive_quaternion_ekf_attitude_update(estimator, z: np.ndarray, var: np.ndarray) -> None:
    from filters.UKF import _quat_normalize

    h0 = estimator.estimate_pose()[3:6]
    innovation = wrap_angle(z - h0)
    H = np.zeros((3, estimator.state_dim), dtype=float)
    eps = 1.0e-6
    x0 = estimator.x.copy()
    for idx in range(estimator.state_dim):
        xp = x0.copy()
        xp[idx] += eps
        xp[6:10] = _quat_normalize(xp[6:10])
        old = estimator.x.copy()
        estimator.x = xp
        hp = estimator.estimate_pose()[3:6]
        estimator.x = old
        H[:, idx] = wrap_angle(hp - h0) / eps
    dx, P_update, estimator.innovation, estimator.S, estimator.K = kalman_update(
        np.zeros(estimator.state_dim), estimator.P, innovation, H, diagonal_covariance(var)
    )
    estimator.x = x0 + dx
    estimator._normalize_state_quaternion()
    estimator.P = estimator._stabilize(P_update)


def _rotation_error_state_attitude_update(estimator, z: np.ndarray, var: np.ndarray) -> None:
    from models import Hoon_invariant_inekf as lie

    h0 = estimator.estimate_pose()[3:6]
    innovation = wrap_angle(z - h0)
    H = np.zeros((3, estimator.error_dim), dtype=float)
    eps = 1.0e-6
    for idx in range(3):
        delta = np.zeros(estimator.error_dim, dtype=float)
        delta[idx] = eps
        R_plus = estimator.Rot @ lie.so3_exp(delta[0:3])
        hp = lie.pose_from_state(R_plus, estimator.p)[3:6]
        H[:, idx] = wrap_angle(hp - h0) / eps
    _, P_update, estimator.innovation, estimator.S, estimator.K = kalman_update(
        np.zeros(estimator.error_dim), estimator.P, innovation, H, diagonal_covariance(var)
    )
    delta = estimator.K @ estimator.innovation
    if hasattr(estimator, "_bounded_delta"):
        delta = estimator._bounded_delta(delta)
    if hasattr(estimator, "_inject_error"):
        estimator._inject_error(delta)
    else:
        from models import Hoon_invariant_inekf as group_lie
        from models.Hoon_lie_group_utils import plus_right
        estimator.Rot, estimator.v, estimator.p = group_lie.from_matrix(
            plus_right(group_lie.as_matrix(estimator.Rot, estimator.v, estimator.p), delta[:9])
        )
        if getattr(estimator, "update_biases", True):
            estimator.gyro_bias = estimator.gyro_bias + delta[9:12]
            estimator.accel_bias = estimator.accel_bias + delta[12:15]
        estimator.X = group_lie.as_matrix(estimator.Rot, estimator.v, estimator.p)
    if hasattr(estimator, "_stabilize_covariance"):
        estimator.P = estimator._stabilize_covariance(P_update)
    else:
        estimator.P = 0.5 * (P_update + P_update.T)
