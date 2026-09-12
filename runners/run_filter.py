from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.loader import load_dataset
from evaluation.metrics import compute_metrics
from evaluation.runtime import timer
from evaluation.visualization import save_error_plot, save_trajectory_plot
from filters.registry import get_canonical_name, get_filter_class, get_filter_config_key
from state_estimation import __version__
from utils.measurement_dispatch import apply_optional_measurements


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: str | Path | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_absolute():
        cwd_path = Path.cwd() / p
        p = cwd_path if cwd_path.exists() else REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_filter_config(filter_name: str, common_cfg: dict, filter_cfg: dict, dataset) -> tuple[dict, dict]:
    canonical = get_canonical_name(filter_name)
    config_key = get_filter_config_key(filter_name)
    dataset_config = {
        "pose_type": common_cfg.get("pose_type", "3d"),
        "mode": common_cfg.get("mode", "fused"),
    }
    cfg = _deep_update(filter_cfg.get(config_key, filter_cfg), {})
    cfg.setdefault("pose_type", dataset_config["pose_type"])
    cfg.setdefault("mode", dataset_config["mode"])
    motion = cfg.setdefault("motion_model", {})
    init = cfg.setdefault("initialization", {})
    if common_cfg.get("use_dataset_initialization", True):
        init["mean"] = dataset.ground_truth[0, :6].tolist()
        init["velocity_mean"] = dataset.initial_velocity.tolist()
        motion["gyro_bias"] = dataset.gyro_bias.tolist()
        motion["accel_bias"] = dataset.accel_bias.tolist()
        motion["gravity"] = dataset.gravity.tolist()
    # InEKF has two accepted legacy keys. Provide both so existing from_configs paths keep working.
    compare_config = {config_key: cfg}
    if canonical == "inekf":
        compare_config["Hoon_invariant_kalman_filter_15d"] = cfg
    return dataset_config, compare_config


def _write_estimates(path: Path, timestamps: np.ndarray, estimates: np.ndarray, ground_truth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp", "est_px", "est_py", "est_pz", "est_roll", "est_pitch", "est_yaw",
        "gt_px", "gt_py", "gt_pz", "gt_roll", "gt_pitch", "gt_yaw",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in zip(timestamps, estimates, ground_truth):
            t, est, gt = row
            writer.writerow([float(t), *map(float, est[:6]), *map(float, gt[:6])])


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def run_filter(args: argparse.Namespace) -> dict:
    common_cfg = _load_yaml(args.config)
    filter_cfg = _load_yaml(args.filter_config) if args.filter_config else _load_yaml(REPO_ROOT / "config" / f"{get_canonical_name(args.filter)}.yaml")
    dataset = load_dataset(common_cfg).limited(args.max_steps)
    if args.particles is not None:
        key = get_filter_config_key(args.filter)
        filter_cfg.setdefault(key, {})["num_particles"] = int(args.particles)
    dataset_config, compare_config = _make_filter_config(args.filter, common_cfg, filter_cfg, dataset)
    filter_cls = get_filter_class(args.filter)
    estimator = filter_cls.from_configs(dataset_config, compare_config)
    measurement_cfg = common_cfg.get("measurements", {})
    use_position = bool(measurement_cfg.get("use_position", True))
    use_velocity = bool(measurement_cfg.get("use_velocity", False))
    use_attitude = bool(measurement_cfg.get("use_attitude", False))
    estimates = np.zeros((len(dataset.timestamps), 6), dtype=float)
    updates = 0
    with timer() as runtime:
        # State zero belongs to timestamps[0]. controls[i] and dt[i] describe
        # the interval ending at timestamps[i], so propagation starts at i=1.
        estimates[0] = estimator.estimate_pose()
        for i in range(1, len(dataset.timestamps)):
            if dataset_config["mode"] in {"imu_only", "fused"}:
                estimator.predict(dataset.controls[i], float(dataset.dt[i]))
            if dataset_config["mode"] in {"gnss_only", "fused"}:
                position = dataset.position_measurements[i] if use_position and dataset.position_measurement_mask[i] else None
                velocity = None
                attitude = None
                if use_velocity and dataset.velocity_measurements is not None and dataset.velocity_measurement_mask is not None:
                    velocity = dataset.velocity_measurements[i] if dataset.velocity_measurement_mask[i] else None
                if use_attitude and dataset.attitude_measurements is not None and dataset.attitude_measurement_mask is not None:
                    attitude = dataset.attitude_measurements[i] if dataset.attitude_measurement_mask[i] else None
                updates += apply_optional_measurements(
                    estimator,
                    position=position,
                    velocity=velocity,
                    attitude=attitude,
                    noise_config=measurement_cfg,
                )
            estimates[i] = estimator.estimate_pose()
    metrics = compute_metrics(estimates, dataset.ground_truth)
    metrics.update({
        "filter": get_canonical_name(args.filter),
        "dataset": dataset.name,
        "sequence": dataset.sequence,
        "updates": int(updates),
        "predict_hz_mean": float(1.0 / np.mean(dataset.dt)),
        "runtime_s": float(runtime["elapsed_s"]),
        "duration_s": float(dataset.timestamps[-1] - dataset.timestamps[0]),
        "mode": dataset_config["mode"],
        "initial_gyro_bias_rad_s": dataset.gyro_bias.tolist(),
        "initial_accel_bias_m_s2": dataset.accel_bias.tolist(),
    })
    output_root = Path(args.output_root or common_cfg.get("output_root", "outputs"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    out_dir = output_root / dataset.name / dataset.sequence / get_canonical_name(args.filter)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_estimates(out_dir / "estimate.csv", dataset.timestamps, estimates, dataset.ground_truth)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    runtime_payload = {"runtime_s": float(runtime["elapsed_s"]), "samples": int(len(estimates)), "updates": int(updates)}
    (out_dir / "runtime.json").write_text(json.dumps(runtime_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "software": {
            "project_version": __version__,
            "git_commit": _git_commit(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "command": sys.argv,
        },
        "dataset_config": common_cfg,
        "filter_config": filter_cfg,
        "dataset_metadata": dataset.metadata,
        "resolved_run": {
            "filter": get_canonical_name(args.filter),
            "mode": dataset_config["mode"],
            "samples": len(dataset.timestamps),
            "duration_s": metrics["duration_s"],
            "position_updates": updates,
            "initial_position_m": dataset.ground_truth[0, :3],
            "initial_rpy_rad": dataset.ground_truth[0, 3:6],
            "initial_velocity_m_s": dataset.initial_velocity,
            "initial_gyro_bias_rad_s": dataset.gyro_bias,
            "initial_accel_bias_m_s2": dataset.accel_bias,
            "gravity_m_s2": dataset.gravity,
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not args.no_plots:
        title = f"{dataset.name}/{dataset.sequence} - {get_canonical_name(args.filter)}"
        save_trajectory_plot(estimates, dataset.ground_truth, out_dir / "trajectory.png", title=title)
        save_error_plot(dataset.timestamps, estimates, dataset.ground_truth, out_dir / "error.png", title=title)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one state-estimation filter on one dataset.")
    parser.add_argument("--filter", required=True, help="ekf, ukf, pf, eskf, or inekf")
    parser.add_argument("--config", default="config/euroc.yaml", help="Dataset/common YAML config")
    parser.add_argument("--filter-config", default=None, help="Optional filter YAML config")
    parser.add_argument("--output-root", default=None, help="Output root directory")
    parser.add_argument("--max-steps", type=int, default=0, help="Limit samples for quick checks; 0 means full sequence")
    parser.add_argument("--particles", type=int, default=None, help="Override PF particle count")
    parser.add_argument("--no-plots", action="store_true", help="Skip trajectory/error PNG generation")
    return parser.parse_args(argv)


def main() -> None:
    result = run_filter(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
