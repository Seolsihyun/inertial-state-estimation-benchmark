from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CommonDataset:
    """Common in-memory representation consumed by all filters."""

    name: str
    sequence: str
    timestamps: np.ndarray
    controls: np.ndarray
    dt: np.ndarray
    position_measurements: np.ndarray
    position_measurement_mask: np.ndarray
    ground_truth: np.ndarray
    velocity_measurements: np.ndarray | None = None
    velocity_measurement_mask: np.ndarray | None = None
    attitude_measurements: np.ndarray | None = None
    attitude_measurement_mask: np.ndarray | None = None
    initial_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    metadata: dict[str, Any] = field(default_factory=dict)

    def limited(self, max_steps: int | None) -> "CommonDataset":
        if max_steps is None or int(max_steps) <= 0 or int(max_steps) >= len(self.timestamps):
            return self
        n = int(max_steps)
        return CommonDataset(
            name=self.name,
            sequence=self.sequence,
            timestamps=self.timestamps[:n],
            controls=self.controls[:n],
            dt=self.dt[:n],
            position_measurements=self.position_measurements[:n],
            position_measurement_mask=self.position_measurement_mask[:n],
            ground_truth=self.ground_truth[:n],
            velocity_measurements=None if self.velocity_measurements is None else self.velocity_measurements[:n],
            velocity_measurement_mask=None if self.velocity_measurement_mask is None else self.velocity_measurement_mask[:n],
            attitude_measurements=None if self.attitude_measurements is None else self.attitude_measurements[:n],
            attitude_measurement_mask=None if self.attitude_measurement_mask is None else self.attitude_measurement_mask[:n],
            initial_velocity=self.initial_velocity.copy(),
            gyro_bias=self.gyro_bias.copy(),
            accel_bias=self.accel_bias.copy(),
            gravity=self.gravity.copy(),
            metadata={**self.metadata, "max_steps": n},
        )
