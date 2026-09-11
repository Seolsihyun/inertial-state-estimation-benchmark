from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import wrap_angle


def save_trajectory_plot(estimates: np.ndarray, ground_truth: np.ndarray, output_path: str | Path, title: str = "Trajectory") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    est = np.asarray(estimates)
    gt = np.asarray(ground_truth)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2.0, label="GT")
    ax.plot(est[:, 0], est[:, 1], color="#f28e2b", linewidth=1.5, label="Estimate")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_error_plot(timestamps: np.ndarray, estimates: np.ndarray, ground_truth: np.ndarray, output_path: str | Path, title: str = "Error") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    est = np.asarray(estimates)
    gt = np.asarray(ground_truth)
    t = np.asarray(timestamps) - float(timestamps[0])
    pos_err = np.linalg.norm(est[:, :3] - gt[:, :3], axis=1)
    heading_err = np.degrees(np.abs(wrap_angle(est[:, 5] - gt[:, 5])))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, pos_err, color="#4e79a7")
    axes[0].set_ylabel("position [m]")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, heading_err, color="#e15759")
    axes[1].set_ylabel("heading [deg]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
