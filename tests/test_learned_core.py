import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learned.small_tcn import SmallTCN, TCNNormalization, make_windows


def test_small_tcn_contract() -> None:
    model = SmallTCN(channels=32)
    inputs = torch.zeros((4, 6, 100), dtype=torch.float32)
    assert model(inputs).shape == (4, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) == 36003


def test_windowing_is_causal() -> None:
    imu = np.arange(60, dtype=np.float32).reshape(10, 6)
    target = np.arange(30, dtype=np.float32).reshape(10, 3)
    windows, labels, indices = make_windows(
        imu, target, window_samples=5, stride=2, downsample=1
    )
    np.testing.assert_array_equal(indices, np.array([4, 6, 8]))
    np.testing.assert_array_equal(windows[0], imu[0:5])
    np.testing.assert_array_equal(labels, target[indices])


def test_normalization_uses_provided_training_data() -> None:
    windows = np.ones((5, 20, 6), dtype=np.float32)
    targets = np.ones((5, 3), dtype=np.float32)
    stats = TCNNormalization.fit(windows, targets)
    assert np.all(stats.input_std >= 1.0e-5)
    assert np.all(stats.target_std >= 1.0e-4)
