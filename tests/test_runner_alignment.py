from argparse import Namespace
import json

import numpy as np

from datasets.common import CommonDataset
from runners import run_filter as runner


class _CountingFilter:
    def __init__(self) -> None:
        self.pose = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @classmethod
    def from_configs(cls, _dataset_config, _filter_config):
        return cls()

    def estimate_pose(self):
        return self.pose.copy()

    def predict(self, _control, _dt):
        self.pose[0] += 1.0
        return self.estimate_pose()

    def measurement_update(self, _measurement):
        return self.estimate_pose()


def test_initial_state_is_not_propagated_twice(monkeypatch, tmp_path) -> None:
    dataset = CommonDataset(
        name="alignment",
        sequence="three_samples",
        timestamps=np.array([0.0, 1.0, 2.0]),
        controls=np.zeros((3, 6)),
        dt=np.ones(3),
        position_measurements=np.zeros((3, 3)),
        position_measurement_mask=np.ones(3, dtype=bool),
        ground_truth=np.column_stack([np.array([10.0, 11.0, 12.0]), np.zeros((3, 5))]),
    )
    monkeypatch.setattr(runner, "load_dataset", lambda _config: dataset)
    monkeypatch.setattr(runner, "get_filter_class", lambda _name: _CountingFilter)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("pose_type: 3d\nmode: imu_only\n", encoding="utf-8")
    filter_path = tmp_path / "ekf.yaml"
    filter_path.write_text("extended_kalman_filter_15d: {}\n", encoding="utf-8")
    args = Namespace(
        filter="ekf", config=str(config_path), filter_config=str(filter_path),
        output_root=str(tmp_path / "outputs"), max_steps=0, particles=None, no_plots=True,
    )
    monkeypatch.setattr(runner, "_git_commit", lambda: "test-commit")
    runner.run_filter(args)
    output = np.genfromtxt(
        tmp_path / "outputs/alignment/three_samples/ekf/estimate.csv",
        delimiter=",", names=True,
    )
    assert output["est_px"].tolist() == [10.0, 11.0, 12.0]
    manifest_path = tmp_path / "outputs/alignment/three_samples/ekf/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["software"]["git_commit"] == "test-commit"


def test_git_commit_falls_back_outside_git(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise runner.subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(runner.subprocess, "run", fail)
    assert runner._git_commit() == "unknown"
