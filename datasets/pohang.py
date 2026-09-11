from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.common import CommonDataset
from utils.dataset_geometry import interpolate_pose, project_measurements_to_imu, quat_xyzw_to_rpy, wrap_angles


def load_pohang_dataset(dataset_cfg: dict) -> CommonDataset:
    """Load Pohang Canal dataset navigation files.

    Expected layout:
        root/navigation/ahrs.txt
        root/navigation/baseline.txt

    AHRS columns are interpreted as t, qx, qy, qz, qw, gyro_xyz, accel_xyz.
    Baseline columns are interpreted as t, qx, qy, qz, qw, pos_xyz.
    """

    root = Path(dataset_cfg.get("root", "data/pohang_canal/pohang05"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    ahrs_path = Path(dataset_cfg.get("ahrs", root / "navigation" / "ahrs.txt"))
    baseline_path = Path(dataset_cfg.get("baseline", root / "navigation" / "baseline.txt"))

    ahrs_t, controls, _ = _load_ahrs(ahrs_path)
    base_t, base_pos, base_rpy = _load_baseline(baseline_path)

    duration_sec = float(dataset_cfg.get("duration_sec", 0.0))
    start = max(float(ahrs_t[0]), float(base_t[0]))
    end = min(float(ahrs_t[-1]), float(base_t[-1]))
    if duration_sec > 0.0:
        end = min(end, start + duration_sec)

    keep = (ahrs_t >= start) & (ahrs_t <= end)
    ahrs_t = ahrs_t[keep]
    controls = controls[keep]
    rel_t = ahrs_t - ahrs_t[0]

    base_keep = (base_t >= start) & (base_t <= end)
    base_rel_t = base_t[base_keep] - ahrs_t[0]
    base_pos_win = base_pos[base_keep]
    base_rpy_win = base_rpy[base_keep]

    gt = interpolate_pose(base_rel_t, np.column_stack([base_pos_win, base_rpy_win]), rel_t)
    measurements, mask, update_hz = project_measurements_to_imu(base_rel_t, base_pos_win, rel_t, every=max(1, int(dataset_cfg.get("measurement_every", 1))))

    dt = np.zeros(len(rel_t), dtype=float)
    if len(rel_t) > 1:
        dt[1:] = np.diff(rel_t)
        dt[0] = np.median(dt[1:])
    dt = np.clip(dt, 1.0e-6, 1.0)

    velocity_measurements = _differentiate_position(gt[:, :3], rel_t)
    attitude_measurements = gt[:, 3:6].copy()
    use_velocity = bool(dataset_cfg.get("use_velocity_measurement", False))
    use_attitude = bool(dataset_cfg.get("use_attitude_measurement", False))

    return CommonDataset(
        name=str(dataset_cfg.get("name", "pohang")),
        sequence=str(dataset_cfg.get("sequence", root.name)),
        timestamps=rel_t,
        controls=controls,
        dt=dt,
        position_measurements=measurements,
        position_measurement_mask=mask,
        ground_truth=gt,
        velocity_measurements=velocity_measurements,
        velocity_measurement_mask=mask.copy() if use_velocity else None,
        attitude_measurements=attitude_measurements,
        attitude_measurement_mask=mask.copy() if use_attitude else None,
        initial_velocity=velocity_measurements[0],
        gyro_bias=np.zeros(3),
        accel_bias=np.zeros(3),
        gravity=np.asarray(dataset_cfg.get("gravity", [0.0, 0.0, 9.81]), dtype=float),
        metadata={
            "source": "Pohang Canal",
            "ahrs": str(ahrs_path),
            "baseline": str(baseline_path),
            "update_hz": update_hz,
            "position_source": "baseline",
            "attitude_source": "baseline quaternion",
            "velocity_source": "finite difference of baseline position",
        },
    )


def _load_ahrs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    t = data[:, 0]
    quat = data[:, 1:5]
    gyro = data[:, 5:8]
    accel = data[:, 8:11]
    rpy = quat_xyzw_to_rpy(quat)
    return t, np.column_stack([accel, gyro]), rpy


def _load_baseline(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    t = data[:, 0]
    quat = data[:, 1:5]
    pos = data[:, 5:8]
    pos = pos - pos[0]
    rpy = quat_xyzw_to_rpy(quat)
    return t, pos, wrap_angles(rpy)


def _differentiate_position(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = np.gradient(t)
    dt = np.clip(dt, 1e-6, None)
    return np.column_stack([np.gradient(pos[:, i]) / dt for i in range(3)])
