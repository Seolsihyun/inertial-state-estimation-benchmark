from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw rows to body-to-world rotation matrices."""
    values = np.asarray(rpy, dtype=float).reshape(-1, 3)
    roll, pitch, yaw = values.T
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotations = np.empty((len(values), 3, 3), dtype=float)
    rotations[:, 0, 0] = cy * cp
    rotations[:, 0, 1] = cy * sp * sr - sy * cr
    rotations[:, 0, 2] = cy * sp * cr + sy * sr
    rotations[:, 1, 0] = sy * cp
    rotations[:, 1, 1] = sy * sp * sr + cy * cr
    rotations[:, 1, 2] = sy * sp * cr - cy * sr
    rotations[:, 2, 0] = -sp
    rotations[:, 2, 1] = cp * sr
    rotations[:, 2, 2] = cp * cr
    return rotations


def compute_metrics(estimates: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    est = np.asarray(estimates, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    n = min(len(est), len(gt))
    if n == 0:
        raise ValueError("Cannot compute metrics for empty arrays.")
    est = est[:n]
    gt = gt[:n]
    pos_err_vec = est[:, :3] - gt[:, :3]
    pos_err = np.linalg.norm(pos_err_vec, axis=1)
    attitude_err = wrap_angle(est[:, 3:6] - gt[:, 3:6])
    heading_err = np.abs(wrap_angle(est[:, 5] - gt[:, 5]))
    est_rotation = _rpy_to_rotation_matrix(est[:, 3:6])
    gt_rotation = _rpy_to_rotation_matrix(gt[:, 3:6])
    relative = np.einsum("nji,njk->nik", gt_rotation, est_rotation)
    cos_angle = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    so3_error = np.arccos(cos_angle)
    return {
        "samples": int(n),
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "position_max_error_m": float(np.max(pos_err)),
        "position_final_error_m": float(pos_err[-1]),
        "roll_rmse_deg": float(np.degrees(np.sqrt(np.mean(attitude_err[:, 0] ** 2)))),
        "pitch_rmse_deg": float(np.degrees(np.sqrt(np.mean(attitude_err[:, 1] ** 2)))),
        "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(heading_err**2)))),
        "heading_final_error_deg": float(np.degrees(heading_err[-1])),
        "attitude_so3_rmse_deg": float(np.degrees(np.sqrt(np.mean(so3_error**2)))),
        "attitude_so3_final_error_deg": float(np.degrees(so3_error[-1])),
    }
