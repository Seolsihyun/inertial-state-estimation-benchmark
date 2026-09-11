from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.common import CommonDataset
from models import Hoon_invariant_inekf as lie
from utils.rotation_utils import quat_to_rpy


def _load_csv_numeric(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data[~np.isnan(data).all(axis=1)]
    return data


def _nearest_indices(source_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(source_t, query_t, side="left")
    idx = np.clip(idx, 1, len(source_t) - 1)
    left = idx - 1
    choose_left = np.abs(query_t - source_t[left]) <= np.abs(source_t[idx] - query_t)
    return np.where(choose_left, left, idx)


def load_euroc_dataset(dataset_cfg: dict) -> CommonDataset:
    """Load EuRoC MAV data into the common benchmark representation.

    Control order is [ax, ay, az, gx, gy, gz]. Ground truth pose is
    [px, py, pz, roll, pitch, yaw]. Position measurements are sparse GT
    pseudo-measurements controlled by ``measurement_stride``.
    """

    root = Path(dataset_cfg.get("root", "data/euroc/V1_01_easy/mav0"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    imu_csv = Path(dataset_cfg.get("imu_csv", root / "imu0" / "data.csv"))
    gt_csv = Path(dataset_cfg.get("gt_csv", root / "state_groundtruth_estimate0" / "data.csv"))

    imu = _load_csv_numeric(imu_csv)
    gt = _load_csv_numeric(gt_csv)
    if imu.shape[1] < 7:
        raise ValueError(f"EuRoC IMU CSV must have at least 7 columns: {imu_csv}")
    if gt.shape[1] < 17:
        raise ValueError(f"EuRoC GT CSV must have at least 17 columns: {gt_csv}")

    imu_stride = max(1, int(dataset_cfg.get("imu_stride", 1)))
    imu = imu[::imu_stride]
    imu_t = imu[:, 0].astype(float) * 1.0e-9
    gt_t = gt[:, 0].astype(float) * 1.0e-9
    nearest_gt = _nearest_indices(gt_t, imu_t)
    gt_rows = gt[nearest_gt]

    # EuRoC columns: gyro wx,wy,wz then accel ax,ay,az in IMU CSV.
    gyro = imu[:, 1:4]
    accel = imu[:, 4:7]
    controls = np.column_stack([accel, gyro])

    positions = gt_rows[:, 1:4]
    quats = gt_rows[:, 4:8]  # qw, qx, qy, qz
    rpy = np.array([quat_to_rpy(q[0], q[1], q[2], q[3]) for q in quats], dtype=float)
    gt_pose = np.column_stack([positions, rpy])

    dt = np.diff(imu_t, prepend=imu_t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.clip(dt, 1.0e-6, None)

    measurement_stride = max(1, int(dataset_cfg.get("measurement_stride", 20)))
    mask = np.zeros(len(imu_t), dtype=bool)
    mask[::measurement_stride] = True
    position_noise_std = float(dataset_cfg.get("position_noise_std", 0.0))
    rng = np.random.default_rng(dataset_cfg.get("seed", 0))
    measurements = positions.copy()
    if position_noise_std > 0.0:
        measurements = measurements + rng.normal(0.0, position_noise_std, size=measurements.shape)

    velocity_measurements = gt_rows[:, 8:11] if gt.shape[1] >= 11 else None
    attitude_measurements = rpy.copy()
    use_velocity_measurement = bool(dataset_cfg.get("use_velocity_measurement", False))
    use_attitude_measurement = bool(dataset_cfg.get("use_attitude_measurement", False))
    velocity_mask = mask.copy() if use_velocity_measurement and velocity_measurements is not None else None
    attitude_mask = mask.copy() if use_attitude_measurement else None

    initial_velocity = gt_rows[0, 8:11] if gt.shape[1] >= 11 else np.zeros(3)
    gyro_bias = gt_rows[0, 11:14] if gt.shape[1] >= 14 else np.zeros(3)
    accel_bias = gt_rows[0, 14:17] if gt.shape[1] >= 17 else np.zeros(3)

    return CommonDataset(
        name=str(dataset_cfg.get("name", "euroc")),
        sequence=str(dataset_cfg.get("sequence", root.parent.name if root.name == "mav0" else root.name)),
        timestamps=imu_t,
        controls=controls,
        dt=dt,
        position_measurements=measurements,
        position_measurement_mask=mask,
        ground_truth=gt_pose,
        velocity_measurements=velocity_measurements,
        velocity_measurement_mask=velocity_mask,
        attitude_measurements=attitude_measurements,
        attitude_measurement_mask=attitude_mask,
        initial_velocity=np.asarray(initial_velocity, dtype=float),
        gyro_bias=np.asarray(gyro_bias, dtype=float),
        accel_bias=np.asarray(accel_bias, dtype=float),
        gravity=np.asarray(dataset_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float),
        metadata={
            "source": "EuRoC MAV",
            "imu_csv": str(imu_csv),
            "gt_csv": str(gt_csv),
            "measurement_stride": measurement_stride,
            "position_noise_std": position_noise_std,
            "use_velocity_measurement": use_velocity_measurement,
            "use_attitude_measurement": use_attitude_measurement,
        },
    )
