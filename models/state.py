"""Shared state order and units for all filters."""

CONTROL_ORDER = ("ax", "ay", "az", "gx", "gy", "gz")
POSE_OUTPUT_ORDER = ("px", "py", "pz", "roll", "pitch", "yaw")
NOMINAL_STATE_ORDER = ("position", "velocity", "attitude", "gyro_bias", "accel_bias")
ERROR_STATE_ORDER = ("dtheta", "dvelocity", "dposition", "dgyro_bias", "daccel_bias")

STATE_DIM_15D = 15
POSE_DIM = 6
CONTROL_DIM = 6
