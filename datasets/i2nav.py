from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.common import CommonDataset
from utils.dataset_geometry import (
    interpolate_pose,
    load_numeric_table,
    load_gnss_as_local_ned,
    project_measurements_to_imu,
)


def load_i2nav_dataset(dataset_cfg: dict) -> CommonDataset:
    """Load i2NAV robot dataset.

    Expected files for each sequence:
        *_ADIS16465_IMU.txt or *_MID360_IMU.txt: t, dtheta_xyz, dvel_xyz
        *_groundtruth.nav: t, pos_xyz, vel_xyz, attitude_deg_xyz
        optional *_GNSS.pos: t, lat, lon, alt, ...
    """

    sequence = str(dataset_cfg.get("sequence", "street00"))
    root = Path(dataset_cfg.get("root", f"data/i2nav_robot/{sequence}"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    repo_root = Path(__file__).resolve().parents[1]
    imu_path = Path(dataset_cfg.get("imu", root / f"{sequence}_ADIS16465_IMU.txt"))
    gt_path = Path(dataset_cfg.get("groundtruth", root / f"{sequence}_groundtruth.nav"))
    gnss_value = dataset_cfg.get("gnss", root / f"{sequence}_F9P_GNSS.pos")
    gnss_path = Path(gnss_value) if gnss_value else None
    if not imu_path.is_absolute():
        imu_path = repo_root / imu_path
    if not gt_path.is_absolute():
        gt_path = repo_root / gt_path
    if gnss_path is not None and not gnss_path.is_absolute():
        gnss_path = repo_root / gnss_path

    imu_t_raw, controls = _load_i2nav_imu(imu_path)
    gt_t_raw, gt, gt_vel = _load_i2nav_gt_nav(gt_path)

    gnss_t_raw = None
    gnss_ned = None
    if gnss_path is not None and gnss_path.exists():
        gnss_t_raw, gnss_ned = load_gnss_as_local_ned(gnss_path, gt_t_raw, gt[:, :3])

    start = max(float(imu_t_raw[0]), float(gt_t_raw[0]), float(gnss_t_raw[0]) if gnss_t_raw is not None else float(gt_t_raw[0]))
    end = min(float(imu_t_raw[-1]), float(gt_t_raw[-1]), float(gnss_t_raw[-1]) if gnss_t_raw is not None else float(gt_t_raw[-1]))
    duration_sec = float(dataset_cfg.get("duration_sec", 0.0))
    if duration_sec > 0.0:
        end = min(end, start + duration_sec)

    keep = (imu_t_raw >= start) & (imu_t_raw <= end)
    imu_t_raw = imu_t_raw[keep]
    controls = controls[keep]
    imu_t = imu_t_raw - imu_t_raw[0]

    gt_interp = interpolate_pose(gt_t_raw - imu_t_raw[0], gt, imu_t)
    vel_interp = _interpolate_vector(gt_t_raw - imu_t_raw[0], gt_vel, imu_t)

    if gnss_t_raw is not None:
        gnss_keep = (gnss_t_raw >= start) & (gnss_t_raw <= end)
        gnss_t = gnss_t_raw[gnss_keep] - imu_t_raw[0]
        measurements, mask, update_hz = project_measurements_to_imu(
            gnss_t,
            gnss_ned[gnss_keep],
            imu_t,
            every=max(1, int(dataset_cfg.get("gnss_every", 1))),
        )
        position_source = "GNSS converted to local NED"
    else:
        measurements, mask, update_hz = _pseudo_measurements_from_gt(
            imu_t,
            gt_interp[:, :3],
            hz=float(dataset_cfg.get("pseudo_gnss_hz", 1.0)),
        )
        position_source = "pseudo-GNSS from GT position"

    dt = np.zeros(len(imu_t), dtype=float)
    if len(imu_t) > 1:
        dt[1:] = np.diff(imu_t)
        dt[0] = np.median(dt[1:])
    dt = np.clip(dt, 1.0e-6, 1.0)

    use_velocity = bool(dataset_cfg.get("use_velocity_measurement", False))
    use_attitude = bool(dataset_cfg.get("use_attitude_measurement", False))

    return CommonDataset(
        name=str(dataset_cfg.get("name", "i2nav")),
        sequence=sequence,
        timestamps=imu_t,
        controls=controls,
        dt=dt,
        position_measurements=measurements,
        position_measurement_mask=mask,
        ground_truth=gt_interp,
        velocity_measurements=vel_interp,
        velocity_measurement_mask=mask.copy() if use_velocity else None,
        attitude_measurements=gt_interp[:, 3:6].copy(),
        attitude_measurement_mask=mask.copy() if use_attitude else None,
        initial_velocity=vel_interp[0],
        gyro_bias=np.zeros(3),
        accel_bias=np.zeros(3),
        gravity=np.asarray(dataset_cfg.get("gravity", [0.0, 0.0, 9.81]), dtype=float),
        metadata={
            "source": "i2NAV Robot",
            "imu": str(imu_path),
            "groundtruth": str(gt_path),
            "gnss": None if gnss_path is None else str(gnss_path),
            "position_source": position_source,
            "velocity_source": "groundtruth.nav velocity",
            "attitude_source": "groundtruth.nav attitude",
            "update_hz": update_hz,
        },
    )


def _load_i2nav_imu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_numeric_table(path)
    if data.shape[1] < 7:
        raise ValueError(f"{path} must contain at least 7 columns: t dtheta_xyz dvel_xyz")
    t = data[:, 0].astype(float)
    dt = np.zeros_like(t)
    dt[1:] = np.diff(t)
    dt[0] = np.median(dt[1:])
    dt = np.clip(dt, 1.0e-6, None)
    gyro = data[:, 1:4] / dt[:, None]
    accel = data[:, 4:7] / dt[:, None]
    return t, np.column_stack([accel, gyro])


def _load_i2nav_gt_nav(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_numeric_table(path)
    if data.shape[1] < 10:
        raise ValueError(f"{path} must contain at least 10 columns: t pos vel att")
    t = data[:, 0].astype(float)
    pos = data[:, 1:4]
    vel = data[:, 4:7]
    att_rad = np.deg2rad(data[:, 7:10])
    return t, np.column_stack([pos, att_rad]), vel


def _interpolate_vector(src_t: np.ndarray, src_values: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(dst_t, src_t, src_values[:, j]) for j in range(src_values.shape[1])])


def _pseudo_measurements_from_gt(imu_t: np.ndarray, gt_pos: np.ndarray, hz: float) -> tuple[np.ndarray, np.ndarray, float]:
    period = 1.0 / max(float(hz), 1.0e-9)
    mask = np.zeros(len(imu_t), dtype=bool)
    next_t = imu_t[0]
    for i, t in enumerate(imu_t):
        if t + 1.0e-9 >= next_t:
            mask[i] = True
            next_t += period
    measurements = np.full((len(imu_t), 3), np.nan, dtype=float)
    measurements[mask] = gt_pos[mask]
    duration = max(float(imu_t[-1] - imu_t[0]), 1.0e-9)
    return measurements, mask, float(np.count_nonzero(mask) / duration)
