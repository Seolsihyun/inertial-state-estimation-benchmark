"""Controlled IMU/GNSS benchmark with increasing three-dimensional rotation.

This experiment complements, rather than replaces, real-dataset evaluation.
Every regime uses identical duration, sample rate, IMU noise, bias process and
GNSS schedule.  Only the trajectory frequency, vertical motion and roll/pitch
amplitudes change from mobile-robot-like to surface-vessel-like to drone-like.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.synthetic_inertial_dataset import generate_synthetic_inertial_dataset
from evaluation.metrics import compute_metrics
from filters.registry import get_canonical_name, get_filter_class, get_filter_config_key


FILTERS = ("ekf", "ukf", "pf", "inekf")
COLORS = {"ekf": "#4c78a8", "ukf": "#f58518", "pf": "#999999", "inekf": "#6f4e9c"}


def deep_update(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def make_filter(name: str, mode: str, synthetic, cfg: dict, particles: int):
    filter_cfg = load_yaml(REPO_ROOT / "config" / f"{name}.yaml")
    key = get_filter_config_key(name)
    core = deepcopy(filter_cfg.get(key, filter_cfg))
    core["mode"] = mode
    core.setdefault("initialization", {})["mean"] = synthetic.gt[0].copy().tolist()
    core["initialization"]["velocity_mean"] = synthetic.gt_velocity[0].copy().tolist()
    core.setdefault("motion_model", {})["gravity"] = cfg["imu"]["gravity"]
    # Apply the one best initial bias and leave the generated random walk unseen.
    core["motion_model"]["gyro_bias"] = cfg["imu"]["gyro_bias_initial"]
    core["motion_model"]["accel_bias"] = cfg["imu"]["accel_bias_initial"]
    if name == "pf":
        core["num_particles"] = particles
    compare = {key: core}
    if name == "inekf":
        compare["Hoon_invariant_kalman_filter_15d"] = core
    return get_filter_class(name).from_configs(
        {"pose_type": "3d", "mode": mode}, compare
    )


def run_once(name: str, mode: str, synthetic, cfg: dict, particles: int) -> tuple[dict, np.ndarray]:
    estimator = make_filter(name, mode, synthetic, cfg, particles)
    estimates = np.zeros_like(synthetic.gt)
    started = time.perf_counter()
    updates = 0
    for index, sample in enumerate(synthetic.dataset):
        estimator.predict(sample["control"], float(sample["dt"]))
        if mode == "fused" and sample["measurement"] is not None:
            estimator.measurement_update(sample["measurement"])
            updates += 1
        estimates[index] = estimator.estimate_pose()
    elapsed = time.perf_counter() - started
    metrics = compute_metrics(estimates, synthetic.gt)
    rpy_error = (estimates[:, 3:6] - synthetic.gt[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    so3_proxy = np.linalg.norm(rpy_error, axis=1)
    metrics.update(
        {
            "filter": get_canonical_name(name),
            "mode": mode,
            "updates": updates,
            "runtime_s": elapsed,
            "attitude_rpy_vector_rmse_deg": float(np.degrees(np.sqrt(np.mean(so3_proxy**2)))),
            "high_turn_heading_rmse_deg": float(
                np.degrees(
                    np.sqrt(np.mean(rpy_error[synthetic.high_turn_mask, 2] ** 2))
                )
            )
            if np.any(synthetic.high_turn_mask)
            else float("nan"),
        }
    )
    return metrics, estimates


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict], labels: dict[str, str]) -> None:
    regimes = list(labels)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3))
    panels = [
        ("imu_only", "heading_rmse_deg", "IMU-only heading RMSE [deg]", False),
        ("fused", "heading_rmse_deg", "IMU + GNSS heading RMSE [deg]", False),
        ("imu_only", "position_rmse_m", "IMU-only position RMSE [m]", True),
        ("fused", "position_rmse_m", "IMU + GNSS position RMSE [m]", True),
    ]
    x = np.arange(len(regimes))
    width = 0.19
    for axis, (mode, field, title, log_scale) in zip(axes.ravel(), panels):
        for offset, name in enumerate(FILTERS):
            values = [
                next(row[field] for row in rows if row["regime"] == regime and row["mode"] == mode and row["filter"] == name)
                for regime in regimes
            ]
            axis.bar(x + (offset - 1.5) * width, values, width, color=COLORS[name], label=name.upper())
        axis.set_xticks(x, [labels[regime] for regime in regimes])
        axis.set_title(title)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25, which="both")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Controlled motion-regime benchmark (same noise and GNSS schedule)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/motion_regimes.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/motion_regimes")
    parser.add_argument("--particles", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_yaml(args.config)
    defaults = source["defaults"]
    rows: list[dict] = []
    labels: dict[str, str] = {}
    estimates_dir = args.output / "estimates"
    estimates_dir.mkdir(parents=True, exist_ok=True)
    for index, (regime, override) in enumerate(source["regimes"].items()):
        label = str(override.get("label", regime))
        labels[regime] = label
        config = deep_update(defaults, {key: value for key, value in override.items() if key != "label"})
        config["random_seed"] = int(defaults["seed"]) + index
        config.setdefault("metrics", {})["high_turn_threshold_rad_s"] = 0.7
        synthetic = generate_synthetic_inertial_dataset(config)
        angular_rate = np.array([sample["control"][3:6] for sample in synthetic.dataset])
        angular_rate_rms = float(np.sqrt(np.mean(np.sum(angular_rate**2, axis=1))))
        for mode in ("imu_only", "fused"):
            for name in FILTERS:
                metrics, estimates = run_once(name, mode, synthetic, config, args.particles)
                row = {
                    "regime": regime,
                    "label": label,
                    "angular_rate_rms_rad_s": angular_rate_rms,
                    "vertical_amplitude_m": config["trajectory"]["amplitude_z"],
                    "roll_amplitude_deg": config["attitude"]["roll_amp_deg"],
                    "pitch_amplitude_deg": config["attitude"]["pitch_amp_deg"],
                    **metrics,
                }
                rows.append(row)
                np.savez_compressed(
                    estimates_dir / f"{regime}_{mode}_{name}.npz",
                    estimate=estimates,
                    ground_truth=synthetic.gt,
                    timestamps_ns=synthetic.timestamps_ns,
                )
                print(json.dumps(row, ensure_ascii=False), flush=True)
    write_csv(args.output / "metrics.csv", rows)
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_summary(args.output / "motion_regime_summary.png", rows, labels)


if __name__ == "__main__":
    main()
