from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.math_utils import wrap_angle
from utils.rotation_utils import rpy_to_rot


@dataclass
class SyntheticInertialDataset:
    dataset: list[dict[str, Any]]
    gt: np.ndarray
    gt_velocity: np.ndarray
    rotations: np.ndarray
    timestamps_ns: np.ndarray
    gnss_mask: np.ndarray
    dropout_mask: np.ndarray
    outlier_mask: np.ndarray
    high_turn_mask: np.ndarray
    metadata: dict[str, Any]


def generate_synthetic_inertial_dataset(config: dict[str, Any]) -> SyntheticInertialDataset:
    """Generate a controllable 3D inertial/GNSS stress-test dataset.

    The filter control vector follows the existing benchmark convention:
    [specific_force_body_x, y, z, angular_rate_body_x, y, z].
    """

    trajectory_cfg = config.get("trajectory", {})
    imu_cfg = config.get("imu", {})
    gnss_cfg = config.get("gnss", {})
    attitude_cfg = config.get("attitude", {})
    seed = int(config.get("random_seed", 7))
    rng = np.random.default_rng(seed)

    dt = float(trajectory_cfg.get("dt", 0.02))
    duration = float(trajectory_cfg.get("duration_sec", 20.0))
    if dt <= 0.0:
        raise ValueError("trajectory.dt must be positive")
    if duration <= 0.0:
        raise ValueError("trajectory.duration_sec must be positive")

    steps = int(np.floor(duration / dt)) + 1
    t = np.arange(steps, dtype=float) * dt
    gravity = np.asarray(imu_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)

    p, v, a_world = _figure_eight_trajectory(t, trajectory_cfg)
    rotations, rpy = _attitude_profile(t, v, attitude_cfg)
    omega_body = _angular_rate_body(rotations, dt)
    specific_force = np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), a_world - gravity)

    accel_bias_true, gyro_bias_true = _bias_random_walk(steps, dt, imu_cfg, rng)
    accel_noise_std = _as_vec3(imu_cfg.get("accel_noise_std", [0.03, 0.03, 0.03]))
    gyro_noise_std = _as_vec3(imu_cfg.get("gyro_noise_std", [0.002, 0.002, 0.002]))
    noisy_accel = specific_force + accel_bias_true + rng.normal(0.0, accel_noise_std, size=(steps, 3))
    noisy_gyro = omega_body + gyro_bias_true + rng.normal(0.0, gyro_noise_std, size=(steps, 3))

    measurements, gnss_mask, dropout_mask, outlier_mask = _gnss_measurements(p, gnss_cfg, rng)
    high_turn_threshold = float(config.get("metrics", {}).get("high_turn_threshold_rad_s", 0.8))
    high_turn_mask = np.linalg.norm(omega_body, axis=1) >= high_turn_threshold

    dataset: list[dict[str, Any]] = []
    for idx in range(steps):
        dataset.append(
            {
                "dt": dt if idx > 0 else dt,
                "control": np.concatenate([noisy_accel[idx], noisy_gyro[idx]]),
                "raw_control": np.concatenate([noisy_accel[idx], noisy_gyro[idx]]),
                "measurement": None if not gnss_mask[idx] else measurements[idx],
                "gt_velocity": v[idx],
                "gt_rotation": rotations[idx],
                "timestamp_ns": int(round(t[idx] * 1.0e9)),
            }
        )

    gt = np.column_stack([p, rpy])
    return SyntheticInertialDataset(
        dataset=dataset,
        gt=gt,
        gt_velocity=v,
        rotations=rotations,
        timestamps_ns=np.asarray([sample["timestamp_ns"] for sample in dataset], dtype=np.int64),
        gnss_mask=gnss_mask,
        dropout_mask=dropout_mask,
        outlier_mask=outlier_mask,
        high_turn_mask=high_turn_mask,
        metadata={
            "seed": seed,
            "steps": steps,
            "dt": dt,
            "duration_sec": duration,
            "gravity": gravity.tolist(),
            "gnss_updates": int(np.sum(gnss_mask)),
            "dropout_steps": int(np.sum(dropout_mask)),
            "outlier_updates": int(np.sum(outlier_mask)),
            "high_turn_steps": int(np.sum(high_turn_mask)),
            "true_accel_bias_final": accel_bias_true[-1].tolist(),
            "true_gyro_bias_final": gyro_bias_true[-1].tolist(),
        },
    )


def _figure_eight_trajectory(t: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax = float(cfg.get("amplitude_x", 8.0))
    ay = float(cfg.get("amplitude_y", 5.0))
    az = float(cfg.get("amplitude_z", 1.5))
    z0 = float(cfg.get("z0", 1.2))
    omega = float(cfg.get("omega_rad_s", 0.35))
    x = ax * np.sin(omega * t)
    y = ay * np.sin(2.0 * omega * t)
    z = z0 + az * np.sin(0.5 * omega * t)

    vx = ax * omega * np.cos(omega * t)
    vy = 2.0 * ay * omega * np.cos(2.0 * omega * t)
    vz = 0.5 * az * omega * np.cos(0.5 * omega * t)

    ax_world = -ax * omega**2 * np.sin(omega * t)
    ay_world = -4.0 * ay * omega**2 * np.sin(2.0 * omega * t)
    az_world = -0.25 * az * omega**2 * np.sin(0.5 * omega * t)
    return np.column_stack([x, y, z]), np.column_stack([vx, vy, vz]), np.column_stack([ax_world, ay_world, az_world])


def _attitude_profile(t: np.ndarray, velocity: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    yaw_offset = np.deg2rad(float(cfg.get("yaw_offset_deg", 0.0)))
    roll_amp = np.deg2rad(float(cfg.get("roll_amp_deg", 8.0)))
    pitch_amp = np.deg2rad(float(cfg.get("pitch_amp_deg", 5.0)))
    roll_freq = float(cfg.get("roll_freq_rad_s", 0.7))
    pitch_freq = float(cfg.get("pitch_freq_rad_s", 0.5))
    yaw = np.arctan2(velocity[:, 1], velocity[:, 0]) + yaw_offset
    yaw = np.unwrap(yaw)
    roll = roll_amp * np.sin(roll_freq * t)
    pitch = pitch_amp * np.sin(pitch_freq * t + 0.4)
    rpy = np.column_stack([roll, pitch, yaw])
    rotations = np.stack([rpy_to_rot(row) for row in rpy], axis=0)
    rpy_wrapped = np.column_stack(
        [
            np.asarray([wrap_angle(value) for value in roll], dtype=float),
            np.asarray([wrap_angle(value) for value in pitch], dtype=float),
            np.asarray([wrap_angle(value) for value in yaw], dtype=float),
        ]
    )
    return rotations, rpy_wrapped


def _angular_rate_body(rotations: np.ndarray, dt: float) -> np.ndarray:
    omega = np.zeros((len(rotations), 3), dtype=float)
    if len(rotations) < 2:
        return omega
    for idx in range(len(rotations) - 1):
        delta_r = rotations[idx].T @ rotations[idx + 1]
        omega[idx] = _so3_log(delta_r) / dt
    omega[-1] = omega[-2]
    return omega


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1.0e-8:
        return np.array(
            [
                0.5 * (rotation[2, 1] - rotation[1, 2]),
                0.5 * (rotation[0, 2] - rotation[2, 0]),
                0.5 * (rotation[1, 0] - rotation[0, 1]),
            ],
            dtype=float,
        )
    return theta / (2.0 * np.sin(theta)) * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=float,
    )


def _bias_random_walk(steps: int, dt: float, cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    accel_bias = np.zeros((steps, 3), dtype=float)
    gyro_bias = np.zeros((steps, 3), dtype=float)
    accel_bias[0] = _as_vec3(cfg.get("accel_bias_initial", [0.08, -0.04, 0.05]))
    gyro_bias[0] = _as_vec3(cfg.get("gyro_bias_initial", [0.003, -0.002, 0.001]))
    accel_rw_std = _as_vec3(cfg.get("accel_bias_rw_std", [0.002, 0.002, 0.002]))
    gyro_rw_std = _as_vec3(cfg.get("gyro_bias_rw_std", [0.00005, 0.00005, 0.00005]))
    scale = np.sqrt(max(dt, 1.0e-12))
    for idx in range(1, steps):
        accel_bias[idx] = accel_bias[idx - 1] + rng.normal(0.0, accel_rw_std * scale)
        gyro_bias[idx] = gyro_bias[idx - 1] + rng.normal(0.0, gyro_rw_std * scale)
    return accel_bias, gyro_bias


def _gnss_measurements(
    positions: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = len(positions)
    stride = max(1, int(cfg.get("stride", 25)))
    offset = max(0, int(cfg.get("offset", 0)))
    noise_std = _as_vec3(cfg.get("noise_std", [0.15, 0.15, 0.25]))
    dropout_windows = cfg.get("dropout_windows", [])
    outlier_probability = float(cfg.get("outlier_probability", 0.0))
    outlier_std = _as_vec3(cfg.get("outlier_std", [4.0, 4.0, 2.0]))

    gnss_mask = np.zeros(steps, dtype=bool)
    gnss_mask[offset::stride] = True
    dropout_mask = np.zeros(steps, dtype=bool)
    for window in dropout_windows:
        start = max(0, int(window[0]))
        end = min(steps, int(window[1]))
        if end > start:
            dropout_mask[start:end] = True
    gnss_mask[dropout_mask] = False

    measurements = positions + rng.normal(0.0, noise_std, size=positions.shape)
    outlier_mask = np.zeros(steps, dtype=bool)
    update_indices = np.flatnonzero(gnss_mask)
    if outlier_probability > 0.0 and update_indices.size > 0:
        outlier_draws = rng.random(update_indices.size) < outlier_probability
        outlier_indices = update_indices[outlier_draws]
        if outlier_indices.size > 0:
            measurements[outlier_indices] += rng.normal(0.0, outlier_std, size=(outlier_indices.size, 3))
            outlier_mask[outlier_indices] = True
    return measurements, gnss_mask, dropout_mask, outlier_mask


def _as_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.repeat(arr[0], 3)
    if arr.size < 3:
        return np.pad(arr, (0, 3 - arr.size), mode="edge")
    return arr[:3].copy()


def perturb_pose(pose: np.ndarray, perturbation_cfg: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).reshape(6).copy()
    position = _as_vec3(perturbation_cfg.get("position_m", [0.0, 0.0, 0.0]))
    rpy_deg = _as_vec3(perturbation_cfg.get("rpy_deg", [0.0, 0.0, 0.0]))
    pose[0:3] += position
    pose[3:6] = np.asarray([wrap_angle(value) for value in pose[3:6] + np.deg2rad(rpy_deg)], dtype=float)
    return pose


def perturb_velocity(velocity: np.ndarray, perturbation_cfg: dict[str, Any]) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=float).reshape(3).copy()
    velocity += _as_vec3(perturbation_cfg.get("velocity_mps", [0.0, 0.0, 0.0]))
    return velocity
