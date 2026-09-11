"""End-to-end held-out CF231 Small-TCN and InEKF experiment.

Runs 3/4/9/10 supervise the network and Run 5 is held out. During Run-5
inference, the estimator receives one initial state, one fixed calibration
from the initial stationary IMU interval, and subsequent IMU samples only.
Run-5 ground truth is consumed only by ``score`` after all trajectories exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.cf231 import GRAVITY_MAGNITUDE, SynchronizedRun, load_cf231_run, synchronize_to_imu
from learned.cf231_protocol import (
    TrainingCalibration,
    calibration_to_dict,
    network_imu,
    test_calibration,
    training_calibration,
)
from learned.small_tcn import fit_small_tcn, make_windows, predict_velocity
from learned.velocity_update import inekf_world_velocity_update
from state_estimation import create_filter


@dataclass(frozen=True)
class InferenceInput:
    time: np.ndarray
    corrected_imu: np.ndarray
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    initial_imu_rotation: np.ndarray
    body_from_imu: np.ndarray
    calibration: TrainingCalibration


@dataclass(frozen=True)
class Trajectory:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray


def parse_ids(text: str) -> list[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})


def heading_velocity(run: SynchronizedRun) -> np.ndarray:
    yaw = run.rotation.as_euler("xyz")[:, 2]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.column_stack(
        [
            cosine * run.velocity[:, 0] + sine * run.velocity[:, 1],
            -sine * run.velocity[:, 0] + cosine * run.velocity[:, 1],
            run.velocity[:, 2],
        ]
    )


def quality_mask(run: SynchronizedRun) -> np.ndarray:
    speed = np.linalg.norm(run.velocity, axis=1)
    acceleration = np.linalg.norm(run.acceleration, axis=1)
    angular_rate = np.linalg.norm(run.angular_velocity_body, axis=1)
    position_step = np.r_[0.0, np.linalg.norm(np.diff(run.position, axis=0), axis=1)]
    return (
        np.all(np.isfinite(run.imu), axis=1)
        & np.all(np.isfinite(run.velocity), axis=1)
        & (speed < 4.0)
        & (acceleration < 15.0)
        & (angular_rate < 4.0)
        & (position_step < 0.05)
    )


def training_windows(
    run: SynchronizedRun,
    window_samples: int,
    stride: int,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    imu = network_imu(run)
    target = heading_velocity(run)
    windows, labels, indices = make_windows(
        imu,
        target,
        window_samples=window_samples,
        stride=stride,
        downsample=downsample,
    )
    quality = quality_mask(run)
    valid = np.asarray(
        [
            quality[index]
            and np.mean(quality[index - window_samples + 1 : index + 1]) >= 0.90
            and np.all(np.isfinite(labels[row]))
            for row, index in enumerate(indices)
        ]
    )
    return windows[valid], labels[valid]


def prepare_inference(
    run: SynchronizedRun,
    calibration_runs: list[SynchronizedRun],
    base_calibration: TrainingCalibration,
) -> tuple[InferenceInput, dict[str, object]]:
    calibrated = test_calibration(run, calibration_runs)
    return (
        InferenceInput(
            time=run.time.copy(),
            corrected_imu=calibrated.propagation_imu,
            initial_position=run.position[0].copy(),
            initial_velocity=run.velocity[0].copy(),
            initial_imu_rotation=calibrated.initial_imu_rotation,
            body_from_imu=calibrated.body_from_imu,
            calibration=base_calibration,
        ),
        calibration_to_dict(calibrated),
    )


def make_inekf(data: InferenceInput):
    initial_rpy = Rotation.from_matrix(data.initial_imu_rotation).as_euler("xyz")
    process_noise = np.zeros(15)
    process_noise[0:3] = np.maximum(
        data.calibration.gyro_sample_variance,
        1.0e-12,
    )
    process_noise[3:6] = np.maximum(
        data.calibration.accel_sample_variance,
        1.0e-12,
    )
    return create_filter(
        "inekf",
        mode="imu_only",
        motion_config={
            "gravity": [0.0, 0.0, -GRAVITY_MAGNITUDE],
            "gyro_bias": [0.0, 0.0, 0.0],
            "accel_bias": [0.0, 0.0, 0.0],
            "update_biases": False,
            "process_noise_diag": process_noise.tolist(),
        },
        initialization_config={
            "mean": [*data.initial_position, *initial_rpy],
            "velocity_mean": data.initial_velocity.tolist(),
            "cov_diag": [np.deg2rad(0.5) ** 2] * 3
            + [0.05**2] * 3
            + [1.0e-6] * 3
            + [0.0] * 6,
        },
    )


def heading_to_world(heading_velocity_value: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            cosine * heading_velocity_value[0] - sine * heading_velocity_value[1],
            sine * heading_velocity_value[0] + cosine * heading_velocity_value[1],
            heading_velocity_value[2],
        ]
    )


def propagate(
    data: InferenceInput,
    update_indices: np.ndarray | None = None,
    learned_heading_velocity: np.ndarray | None = None,
    learned_covariance: np.ndarray | None = None,
) -> Trajectory:
    estimator = make_inekf(data)
    positions = [estimator.p.copy()]
    velocities = [estimator.v.copy()]
    rotations = [estimator.Rot @ data.body_from_imu.T]
    lookup = {} if update_indices is None else {
        int(index): row for row, index in enumerate(update_indices)
    }
    for index in range(1, data.time.size):
        dt = float(data.time[index] - data.time[index - 1])
        if not 0.0 < dt < 0.5:
            raise ValueError(f"unexpected dt={dt} at sample {index}")
        estimator.predict(data.corrected_imu[index - 1], dt)
        row = lookup.get(index)
        if row is not None:
            body_rotation = estimator.Rot @ data.body_from_imu.T
            yaw = Rotation.from_matrix(body_rotation).as_euler("xyz")[2]
            velocity_world = heading_to_world(learned_heading_velocity[row], yaw)
            cosine, sine = np.cos(yaw), np.sin(yaw)
            heading_rotation = np.array(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
            covariance_world = heading_rotation @ learned_covariance @ heading_rotation.T
            inekf_world_velocity_update(estimator, velocity_world, covariance_world)
        positions.append(estimator.p.copy())
        velocities.append(estimator.v.copy())
        rotations.append(estimator.Rot @ data.body_from_imu.T)
    return Trajectory(np.asarray(positions), np.asarray(velocities), np.asarray(rotations))


def loose_integration(
    data: InferenceInput,
    pure: Trajectory,
    update_indices: np.ndarray,
    predicted_heading_velocity: np.ndarray,
) -> Trajectory:
    yaw = Rotation.from_matrix(pure.rotation).as_euler("xyz")[:, 2]
    velocity = pure.velocity.copy()
    velocity[update_indices] = np.asarray(
        [
            heading_to_world(value, yaw[index])
            for index, value in zip(update_indices, predicted_heading_velocity)
        ]
    )
    position = np.empty_like(velocity)
    position[0] = data.initial_position
    for index in range(1, data.time.size):
        dt = float(data.time[index] - data.time[index - 1])
        position[index] = position[index - 1] + 0.5 * (
            velocity[index - 1] + velocity[index]
        ) * dt
    return Trajectory(position, velocity, pure.rotation.copy())


def score(trajectory: Trajectory, truth: SynchronizedRun) -> dict[str, object]:
    position_error = np.linalg.norm(trajectory.position - truth.position, axis=1)
    relative = np.einsum("nji,njk->nik", truth.rotation.as_matrix(), trajectory.rotation)
    so3 = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    estimate_rpy = Rotation.from_matrix(trajectory.rotation).as_euler("xyz", degrees=True)
    truth_rpy = truth.rotation.as_euler("xyz", degrees=True)
    rpy_error = (estimate_rpy - truth_rpy + 180.0) % 360.0 - 180.0
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_final_error_m": float(position_error[-1]),
        "position_max_error_m": float(np.max(position_error)),
        "so3_rmse_deg": float(np.sqrt(np.mean(so3**2))),
        "so3_final_error_deg": float(so3[-1]),
        "rpy_rmse_deg": np.sqrt(np.mean(rpy_error**2, axis=0)).tolist(),
    }


def plot_results(
    output: Path,
    truth: SynchronizedRun,
    trajectories: dict[str, Trajectory],
) -> None:
    colors = {"fixed_bias_imu": "0.65", "small_tcn_loose": "#f58518", "small_tcn_inekf": "#54a24b"}
    labels = {"fixed_bias_imu": "Fixed-bias IMU DR", "small_tcn_loose": "Small TCN loose", "small_tcn_inekf": "Small TCN + InEKF"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, horizon in zip(axes[:2], (20.0, 60.0)):
        mask = truth.time <= min(horizon, truth.time[-1])
        axis.plot(truth.position[mask, 0], truth.position[mask, 1], "k", lw=2, label="GT")
        for name, trajectory in trajectories.items():
            axis.plot(trajectory.position[mask, 0], trajectory.position[mask, 1], color=colors[name], lw=1.1, label=labels[name])
        axis.set_title(f"First {horizon:.0f} s")
        axis.axis("equal")
        axis.grid(alpha=0.25)
    axes[2].plot(truth.position[:, 0], truth.position[:, 1], "k", lw=2, label="GT")
    for name in ("small_tcn_loose", "small_tcn_inekf"):
        axes[2].plot(trajectories[name].position[:, 0], trajectories[name].position[:, 1], color=colors[name], lw=1.1, label=labels[name])
    axes[2].set_title("Full run (learned methods)")
    axes[2].axis("equal")
    axes[2].grid(alpha=0.25)
    for axis in axes:
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "trajectory_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/cf231_leave_one_out/csv")
    parser.add_argument("--training-runs", default="3,4,9,10")
    parser.add_argument("--bias-runs", default="3,9,10")
    parser.add_argument("--test-run", type=int, default=5)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--training-stride", type=int, default=5)
    parser.add_argument("--update-stride", type=int, default=10)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--cv-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/cf231_small_tcn")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training_ids = parse_ids(args.training_runs)
    bias_ids = parse_ids(args.bias_runs)
    if args.test_run in training_ids:
        raise ValueError("test run must not appear in training runs")
    runs = {
        run_id: synchronize_to_imu(load_cf231_run(args.dataset, run_id))
        for run_id in sorted({*training_ids, *bias_ids, args.test_run})
    }
    sets = {
        run_id: training_windows(
            runs[run_id], args.window_samples, args.training_stride, args.downsample
        )
        for run_id in training_ids
    }

    residuals = []
    cross_validation = {}
    for fold, held_out in enumerate(training_ids):
        fold_train = [run_id for run_id in training_ids if run_id != held_out]
        cv_x = np.concatenate([sets[run_id][0] for run_id in fold_train])
        cv_y = np.concatenate([sets[run_id][1] for run_id in fold_train])
        if args.max_train_windows > 0 and cv_x.shape[0] > args.max_train_windows:
            rng = np.random.default_rng(args.seed + fold)
            selected = np.sort(
                rng.choice(cv_x.shape[0], args.max_train_windows, replace=False)
            )
            cv_x, cv_y = cv_x[selected], cv_y[selected]
        cv_model, cv_stats, _ = fit_small_tcn(
            cv_x,
            cv_y,
            epochs=args.cv_epochs,
            batch_size=args.batch_size,
            seed=args.seed + held_out,
        )
        prediction = predict_velocity(
            cv_model, cv_stats, sets[held_out][0], args.batch_size
        )
        residual = prediction - sets[held_out][1]
        residuals.append(residual)
        cross_validation[str(held_out)] = {
            "samples": int(residual.shape[0]),
            "velocity_rmse_m_s": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        }
    pooled_residual = np.vstack(residuals)
    learned_covariance = np.diag(
        np.maximum(np.mean(pooled_residual**2, axis=0), 0.02**2)
    )

    train_x = np.concatenate([sets[run_id][0] for run_id in training_ids])
    train_y = np.concatenate([sets[run_id][1] for run_id in training_ids])
    if args.max_train_windows > 0 and train_x.shape[0] > args.max_train_windows:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(train_x.shape[0], args.max_train_windows, replace=False))
        train_x, train_y = train_x[selected], train_y[selected]
    model, stats, history = fit_small_tcn(
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    base_calibration = training_calibration([runs[run_id] for run_id in bias_ids])
    test_input, test_calibration = prepare_inference(
        runs[args.test_run],
        [runs[run_id] for run_id in bias_ids],
        base_calibration,
    )
    test_windows, _, update_indices = make_windows(
        network_imu(runs[args.test_run]),
        None,
        window_samples=args.window_samples,
        stride=1,
        downsample=args.downsample,
    )
    test_prediction = predict_velocity(model, stats, test_windows, args.batch_size)
    training_targets = np.concatenate([sets[run_id][1] for run_id in training_ids])
    lower = np.quantile(training_targets, 0.002, axis=0)
    upper = np.quantile(training_targets, 0.998, axis=0)
    test_prediction = np.clip(test_prediction, lower, upper)

    fixed = propagate(test_input)
    loose = loose_integration(test_input, fixed, update_indices, test_prediction)
    tight_mask = update_indices % args.update_stride == 0
    tight = propagate(
        test_input,
        update_indices[tight_mask],
        test_prediction[tight_mask],
        learned_covariance,
    )
    trajectories = {
        "fixed_bias_imu": fixed,
        "small_tcn_loose": loose,
        "small_tcn_inekf": tight,
    }

    # Ground truth is used for the first time here after inference is complete.
    metrics = {name: score(value, runs[args.test_run]) for name, value in trajectories.items()}
    summary = {
        "protocol": {
            "training_runs": training_ids,
            "bias_runs": bias_ids,
            "test_run": args.test_run,
            "test_gt_during_inference": False,
            "runtime_sensors": ["IMU"],
            "initial_external_information": ["position", "velocity", "roll", "pitch", "yaw"],
            "window_samples": args.window_samples,
            "update_stride": args.update_stride,
            "model_parameters": int(sum(value.numel() for value in model.parameters())),
        },
        "test_fixed_calibration": test_calibration,
        "cross_validation": cross_validation,
        "validation_velocity_covariance_m2_s2": learned_covariance.tolist(),
        "final_training_loss": history[-1],
        "metrics": metrics,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "trajectories.npz",
        time=runs[args.test_run].time,
        ground_truth=runs[args.test_run].position,
        fixed_bias_imu=fixed.position,
        small_tcn_loose=loose.position,
        small_tcn_inekf=tight.position,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "normalization": stats,
            "protocol": summary["protocol"],
        },
        args.output / "small_tcn.pt",
    )
    plot_results(args.output, runs[args.test_run], trajectories)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
