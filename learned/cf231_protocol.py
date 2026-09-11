"""Calibration and data-split helpers for the CF231 Run-5 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import polar
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from datasets.cf231 import GRAVITY_MAGNITUDE, SynchronizedRun


@dataclass(frozen=True)
class TrainingCalibration:
    gyro_sample_variance: np.ndarray
    accel_sample_variance: np.ndarray
    gyro_between_run_variance: np.ndarray
    accel_between_run_variance: np.ndarray
    sample_period_s: float


@dataclass(frozen=True)
class TestCalibration:
    propagation_imu: np.ndarray
    initial_imu_rotation: np.ndarray
    body_from_imu: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    gyro_intrinsic_matrix: np.ndarray
    stationary_duration_s: float


def network_imu(run: SynchronizedRun) -> np.ndarray:
    """Input used by the reported Small TCN.

    Only the mean of the first 100 gyroscope samples is removed. The network
    otherwise receives the six raw SI-unit channels.
    """
    values = run.imu.copy()
    values[:, 3:6] -= np.mean(values[: min(100, len(values)), 3:6], axis=0)
    return values


def training_calibration(
    runs: list[SynchronizedRun], static_seconds: float = 1.0
) -> TrainingCalibration:
    gyro_bias, accel_bias = [], []
    gyro_variance, accel_variance, periods = [], [], []
    expected = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
    for run in runs:
        static = run.time <= static_seconds
        if np.count_nonzero(static) < 20:
            raise ValueError(f"Run {run.run_id} has too few static samples")
        gyro_mean = np.mean(run.imu[static, 3:6], axis=0)
        accel_mean = np.mean(run.imu[static, 0:3], axis=0)
        gyro_bias.append(gyro_mean)
        accel_bias.append(accel_mean - expected)
        gyro_variance.append(_robust_variance(run.imu[static, 3:6], gyro_mean))
        accel_variance.append(_robust_variance(run.imu[static, 0:3], accel_mean))
        periods.append(float(np.median(np.diff(run.time[static]))))
    gyro_array = np.asarray(gyro_bias)
    accel_array = np.asarray(accel_bias)
    return TrainingCalibration(
        gyro_sample_variance=np.median(np.asarray(gyro_variance), axis=0),
        accel_sample_variance=np.median(np.asarray(accel_variance), axis=0),
        gyro_between_run_variance=np.var(gyro_array, axis=0, ddof=1),
        accel_between_run_variance=np.var(accel_array, axis=0, ddof=1),
        sample_period_s=float(np.median(periods)),
    )


def test_calibration(
    test: SynchronizedRun,
    calibration_runs: list[SynchronizedRun],
    static_seconds: float = 1.0,
) -> TestCalibration:
    intrinsic, body_from_imu = gyro_intrinsic_matrix(calibration_runs, static_seconds)
    duration = detect_stationary_duration(test)
    static = test.time <= duration
    gyro_bias = np.mean(test.imu[static, 3:6], axis=0)
    accel_mean = np.mean(test.imu[static, 0:3], axis=0)
    expected_force = (
        body_from_imu.T
        @ test.rotation[0].as_matrix().T
        @ np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
    )
    accel_bias = accel_mean - expected_force
    propagation = test.imu.copy()
    propagation[:, 0:3] -= accel_bias
    propagation[:, 3:6] = (intrinsic @ (test.imu[:, 3:6] - gyro_bias).T).T
    return TestCalibration(
        propagation_imu=propagation,
        initial_imu_rotation=test.rotation[0].as_matrix() @ body_from_imu,
        body_from_imu=body_from_imu,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        gyro_intrinsic_matrix=intrinsic,
        stationary_duration_s=duration,
    )


def detect_stationary_duration(run: SynchronizedRun) -> float:
    seed = run.time <= min(0.5, float(run.time[-1]))
    seed_bias = np.median(run.imu[seed, 3:6], axis=0)
    moving = np.linalg.norm(run.imu[:, 3:6] - seed_bias, axis=1) > 0.01
    sustained = np.convolve(moving.astype(int), np.ones(5, dtype=int), mode="valid")
    crossings = np.flatnonzero(sustained == 5)
    detected = float(run.time[crossings[0]]) if crossings.size else 3.0
    return float(np.clip(detected - 0.4, 0.8, 3.0))


def gyro_intrinsic_matrix(
    runs: list[SynchronizedRun], static_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    measured, truth = [], []
    for run in runs:
        static = run.time <= static_seconds
        bias = np.mean(run.imu[static, 3:6], axis=0)
        x = run.imu[:, 3:6] - bias
        y = run.angular_velocity_body
        angular_rate = np.linalg.norm(y, axis=1)
        valid = (
            (angular_rate >= 0.03)
            & (angular_rate <= 2.0)
            & np.all(np.isfinite(x), axis=1)
            & np.all(np.isfinite(y), axis=1)
        )
        measured.append(x[valid])
        truth.append(y[valid])
    x_all, y_all = np.vstack(measured), np.vstack(truth)

    def residual(parameters: np.ndarray) -> np.ndarray:
        matrix = parameters.reshape(3, 3)
        return (x_all @ matrix.T - y_all).ravel()

    fit = least_squares(
        residual, np.eye(3).ravel(), loss="soft_l1", f_scale=0.05, max_nfev=100
    )
    affine = fit.x.reshape(3, 3)
    frame_rotation, intrinsic = polar(affine)
    if np.linalg.det(frame_rotation) < 0.0:
        raise ValueError("gyro fit produced a reflected frame transform")
    return intrinsic, frame_rotation


def calibration_to_dict(value: TestCalibration) -> dict[str, object]:
    return {
        "stationary_duration_s": value.stationary_duration_s,
        "gyro_bias_rad_s": value.gyro_bias.tolist(),
        "accel_bias_m_s2": value.accel_bias.tolist(),
        "gyro_intrinsic_matrix": value.gyro_intrinsic_matrix.tolist(),
    }


def _robust_variance(samples: np.ndarray, center: np.ndarray) -> np.ndarray:
    mad = np.median(np.abs(samples - center[None, :]), axis=0)
    return np.square(1.4826 * mad)
