"""Common API for the state-estimation filters."""

from __future__ import annotations

from typing import Any

from filters.registry import get_canonical_name, get_filter_class


def available_filters() -> tuple[str, ...]:
    """Return the supported filter names."""
    return ("ekf", "ukf", "pf", "eskf", "inekf")


def create_filter(
    name: str,
    *,
    mode: str = "fused",
    pose_type: str = "3d",
    motion_config: dict[str, Any] | None = None,
    measurement_config: dict[str, Any] | None = None,
    initialization_config: dict[str, Any] | None = None,
    **filter_options: Any,
):
    """Construct one filter using the shared state and measurement contract.

    Parameters follow the existing filter constructors. All estimators expose
    ``predict([ax, ay, az, gx, gy, gz], dt)``, ``measurement_update(position)``
    and ``estimate_pose() -> [px, py, pz, roll, pitch, yaw]``.
    """
    canonical = get_canonical_name(name)
    filter_class = get_filter_class(canonical)
    return filter_class(
        pose_type=pose_type,
        mode=mode,
        motion_config=motion_config,
        measurement_config=measurement_config,
        initialization_config=initialization_config,
        **filter_options,
    )
