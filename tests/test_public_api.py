import numpy as np
import pytest

from state_estimation import available_filters, create_filter


@pytest.mark.parametrize("name", available_filters())
def test_common_filter_contract(name: str) -> None:
    options = {"num_particles": 64, "seed": 3} if name == "pf" else {}
    estimator = create_filter(
        name,
        mode="fused",
        motion_config={"gravity": [0.0, 0.0, -9.81]},
        measurement_config={"measurement_noise_diag": [0.1, 0.1, 0.1]},
        initialization_config={"mean": [0.0] * 6, "cov_diag": [1.0e-3] * 15},
        **options,
    )
    pose = estimator.predict([0.0, 0.0, 9.81, 0.0, 0.0, 0.0], 0.01)
    assert np.asarray(pose).shape == (6,)
    updated = estimator.measurement_update([0.0, 0.0, 0.0])
    assert np.asarray(updated).shape == (6,)
    assert np.all(np.isfinite(updated))
