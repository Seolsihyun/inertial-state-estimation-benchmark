"""Velocity-measurement update for InEKF."""

from __future__ import annotations

import numpy as np

from models import Hoon_invariant_inekf as lie
from models.Hoon_lie_group_utils import plus_right
from utils.filter_math import kalman_update


def inekf_world_velocity_update(
    estimator,
    measured_velocity_world: np.ndarray,
    covariance_world: np.ndarray,
) -> np.ndarray:
    """Fuse a learned world velocity without using position or ground truth.

    For the right perturbation X Exp(delta), v_plus is approximately
    v + R*delta_v, hence H[:, 3:6] = R.
    """
    measurement = np.asarray(measured_velocity_world, dtype=float).reshape(3)
    covariance = np.asarray(covariance_world, dtype=float).reshape(3, 3)
    innovation = measurement - estimator.v
    H = np.zeros((3, estimator.error_dim), dtype=float)
    H[:, 3:6] = estimator.Rot
    _, updated_covariance, estimator.innovation, estimator.S, estimator.K = kalman_update(
        np.zeros(estimator.error_dim), estimator.P, innovation, H, covariance
    )
    estimator.delta = estimator._bounded_delta(estimator.K @ innovation)
    estimator.Rot, estimator.v, estimator.p = lie.from_matrix(
        plus_right(
            lie.as_matrix(estimator.Rot, estimator.v, estimator.p),
            estimator.delta[:9],
        )
    )
    estimator.X = lie.as_matrix(estimator.Rot, estimator.v, estimator.p)
    estimator.P = estimator._stabilize_covariance(updated_covariance)
    return estimator.estimate_pose()
