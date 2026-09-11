import numpy as np

from filters.InEKF import InEKF
from models.Hoon_lie_group_utils import plus_right


def test_measurement_update_uses_right_group_correction() -> None:
    estimator = InEKF(
        mode="fused",
        motion_config={"update_biases": False},
        measurement_config={"measurement_noise_diag": [0.1, 0.1, 0.1]},
        initialization_config={
            "mean": [1.0, -0.5, 0.2, 0.1, -0.2, 0.3],
            "velocity_mean": [0.4, -0.1, 0.2],
            "cov_diag": [0.01] * 15,
        },
    )
    state_before = estimator.X.copy()
    estimator.measurement_update([1.2, -0.3, 0.4])
    expected = plus_right(state_before, estimator.delta[:9])
    np.testing.assert_allclose(estimator.X, expected, atol=1.0e-10)


def test_analytic_measurement_jacobians_match_finite_difference() -> None:
    estimator = InEKF(
        motion_config={"jacobian_mode": "analytic"},
        initialization_config={
            "mean": [1.0, 2.0, 3.0, 0.2, -0.3, 0.4],
            "velocity_mean": [0.5, -0.2, 0.1],
        },
    )
    np.testing.assert_allclose(
        estimator._position_measurement_jacobian(),
        estimator.finite_position_jacobian_for_validation(),
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        estimator._velocity_measurement_jacobian(),
        estimator.finite_velocity_jacobian_for_validation(),
        atol=1.0e-7,
    )


def test_analytic_process_jacobian_matches_finite_difference() -> None:
    estimator = InEKF(
        motion_config={"jacobian_mode": "analytic"},
        initialization_config={
            "mean": [1.0, 2.0, 3.0, 0.2, -0.3, 0.4],
            "velocity_mean": [0.5, -0.2, 0.1],
        },
    )
    control = np.array([0.3, -0.1, 9.7, 0.02, -0.03, 0.04])
    dt = 0.01
    previous = estimator.X.copy()
    rotation, velocity, position = estimator._propagate_nominal(
        estimator.Rot,
        estimator.v,
        estimator.p,
        estimator.gyro_bias,
        estimator.accel_bias,
        control,
        dt,
    )
    predicted = previous.copy()
    predicted[:3, :3] = rotation
    predicted[:3, 3] = velocity
    predicted[:3, 4] = position
    analytic = estimator._analytic_process_jacobian(
        previous, estimator.gyro_bias, estimator.accel_bias, control, dt
    )
    finite = estimator.finite_process_jacobian_for_validation(
        previous,
        estimator.gyro_bias,
        estimator.accel_bias,
        control,
        dt,
        predicted,
    )
    np.testing.assert_allclose(analytic, finite, atol=1.0e-7)
