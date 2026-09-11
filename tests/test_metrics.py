import numpy as np

from evaluation.metrics import compute_metrics


def test_so3_metric_is_zero_for_identical_attitude() -> None:
    states = np.zeros((4, 6))
    states[:, 3:6] = [[0.1, -0.2, 0.3], [0.2, 0.1, -0.4], [0.0, 0.0, 1.0], [-0.2, 0.3, 0.4]]
    metrics = compute_metrics(states, states)
    assert metrics["attitude_so3_rmse_deg"] < 1.0e-6
    assert metrics["attitude_so3_final_error_deg"] < 1.0e-6


def test_so3_metric_reports_known_yaw_error() -> None:
    truth = np.zeros((3, 6))
    estimate = truth.copy()
    estimate[:, 5] = np.deg2rad(10.0)
    metrics = compute_metrics(estimate, truth)
    assert np.isclose(metrics["attitude_so3_rmse_deg"], 10.0)
    assert np.isclose(metrics["attitude_so3_final_error_deg"], 10.0)
