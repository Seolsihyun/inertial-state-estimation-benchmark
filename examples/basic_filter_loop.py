"""Minimal data-free example of the common filter API."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from state_estimation import create_filter


def main() -> None:
    estimator = create_filter(
        "inekf",
        mode="fused",
        motion_config={
            "gravity": [0.0, 0.0, -9.81],
            "gyro_bias": [0.0, 0.0, 0.0],
            "accel_bias": [0.0, 0.0, 0.0],
        },
        measurement_config={"measurement_noise_diag": [0.01, 0.01, 0.01]},
        initialization_config={
            "mean": [0.0] * 6,
            "velocity_mean": [0.0] * 3,
            "cov_diag": [1.0e-3] * 15,
        },
    )

    dt = 0.01
    stationary_imu = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
    for index in range(500):
        estimator.predict(stationary_imu, dt)
        if index % 100 == 0:
            estimator.measurement_update(np.zeros(3))

    print("[px, py, pz, roll, pitch, yaw]")
    print(np.array2string(estimator.estimate_pose(), precision=6))


if __name__ == "__main__":
    main()
