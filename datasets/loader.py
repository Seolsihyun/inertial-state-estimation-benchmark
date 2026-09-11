from __future__ import annotations

from datasets.common import CommonDataset
from datasets.euroc import load_euroc_dataset
from datasets.pohang import load_pohang_dataset
from datasets.i2nav import load_i2nav_dataset


def load_dataset(config: dict) -> CommonDataset:
    dataset_cfg = dict(config.get("dataset", config))
    measurement_cfg = config.get("measurements", {}) if isinstance(config, dict) else {}
    if measurement_cfg:
        dataset_cfg.setdefault("use_velocity_measurement", bool(measurement_cfg.get("use_velocity", False)))
        dataset_cfg.setdefault("use_attitude_measurement", bool(measurement_cfg.get("use_attitude", False)))
    dataset_type = str(dataset_cfg.get("type", "euroc")).lower()
    if dataset_type == "euroc":
        return load_euroc_dataset(dataset_cfg)
    if dataset_type in {"pohang", "pohang05", "pohang_canal"}:
        return load_pohang_dataset(dataset_cfg)
    if dataset_type in {"i2nav", "i2nav_robot"}:
        return load_i2nav_dataset(dataset_cfg)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
