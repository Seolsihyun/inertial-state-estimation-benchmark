"""Compact TCN used to turn fixed-length 6-axis IMU windows into velocity.

The dataset adapter is deliberately separate: callers must build windows from
training runs only and must never fit normalization with the held-out run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


@dataclass(frozen=True)
class TCNNormalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, windows: np.ndarray, targets: np.ndarray) -> "TCNNormalization":
        windows = np.asarray(windows, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        return cls(
            input_mean=windows.mean(axis=(0, 1)),
            input_std=np.maximum(windows.std(axis=(0, 1)), 1.0e-5),
            target_mean=targets.mean(axis=0),
            target_std=np.maximum(targets.std(axis=0), 1.0e-4),
        )


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(4, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.net(value))


class SmallTCN(nn.Module):
    """Stem + three residual temporal blocks + 3D velocity head."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, channels, 7, padding=3),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            TemporalBlock(channels, 1),
            TemporalBlock(channels, 2),
            TemporalBlock(channels, 4),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * channels, 48),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 3),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature = self.blocks(self.stem(value))
        pooled = torch.cat((feature.mean(dim=-1), feature[:, :, -1]), dim=1)
        return self.head(pooled)


def make_windows(
    imu: np.ndarray,
    targets: np.ndarray | None,
    window_samples: int = 200,
    stride: int = 5,
    downsample: int = 2,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Create causal windows; the returned index is each window's end sample."""
    imu = np.asarray(imu, dtype=np.float32)
    if imu.ndim != 2 or imu.shape[1] != 6:
        raise ValueError("imu must have shape [N, 6]")
    indices = np.arange(window_samples - 1, imu.shape[0], stride, dtype=np.int64)
    windows = np.stack(
        [imu[i - window_samples + 1 : i + 1 : downsample] for i in indices]
    )
    if targets is None:
        return windows, None, indices
    target_array = np.asarray(targets, dtype=np.float32)
    return windows, target_array[indices], indices


class _TrainingWindows(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        targets: np.ndarray,
        stats: TCNNormalization,
        augment: bool,
    ) -> None:
        self.windows = np.asarray(windows, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.stats = stats
        self.augment = augment

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int):
        window = self.windows[index].copy()
        if self.augment:
            window[:, 0:3] += np.random.normal(0.0, 0.015, (1, 3))
            window[:, 3:6] += np.random.normal(0.0, 2.0e-4, (1, 3))
            window[:, 0:3] += np.random.normal(0.0, 0.01, window[:, 0:3].shape)
            window[:, 3:6] += np.random.normal(0.0, 1.0e-4, window[:, 3:6].shape)
        x = (window - self.stats.input_mean) / self.stats.input_std
        y = (self.targets[index] - self.stats.target_mean) / self.stats.target_std
        return (
            torch.from_numpy(x.T.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )


def fit_small_tcn(
    windows: np.ndarray,
    targets: np.ndarray,
    *,
    channels: int = 32,
    epochs: int = 32,
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[SmallTCN, TCNNormalization, list[float]]:
    """Train only on caller-provided training runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    stats = TCNNormalization.fit(windows, targets)
    loader = DataLoader(
        _TrainingWindows(windows, targets, stats, augment=True),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = SmallTCN(channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for inputs, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            point = nn.functional.smooth_l1_loss(prediction, target)
            mean = torch.mean(torch.square(torch.mean(prediction - target, dim=0)))
            loss = point + 0.15 * mean
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * inputs.shape[0]
            count += inputs.shape[0]
        scheduler.step()
        history.append(total / max(count, 1))
    return model, stats, history


def predict_velocity(
    model: SmallTCN,
    stats: TCNNormalization,
    windows: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    x = (np.asarray(windows, dtype=np.float32) - stats.input_mean) / stats.input_std
    tensor = torch.from_numpy(x.transpose(0, 2, 1).astype(np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (inputs,) in loader:
            predictions.append(model(inputs).cpu().numpy())
    normalized = np.vstack(predictions)
    return normalized * stats.target_std + stats.target_mean
