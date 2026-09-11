from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def wrap_angles(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def quat_xyzw_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=float).reshape(-1, 4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    norm = np.where(norm > 0.0, norm, 1.0)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    sinp = 2.0 * (w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return np.column_stack([roll, pitch, yaw])


def interpolate_pose(src_t: np.ndarray, src_pose: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    src_t = np.asarray(src_t, dtype=float).reshape(-1)
    src_pose = np.asarray(src_pose, dtype=float)
    dst_t = np.asarray(dst_t, dtype=float).reshape(-1)
    out = np.zeros((len(dst_t), 6), dtype=float)
    for j in range(3):
        out[:, j] = np.interp(dst_t, src_t, src_pose[:, j])
    for j in range(3, 6):
        out[:, j] = np.interp(dst_t, src_t, np.unwrap(src_pose[:, j]))
    out[:, 3:6] = wrap_angles(out[:, 3:6])
    return out


def project_measurements_to_imu(meas_t: np.ndarray, meas_pos: np.ndarray, imu_t: np.ndarray, every: int = 1) -> tuple[np.ndarray, np.ndarray, float]:
    selected = np.arange(0, len(meas_t), max(1, int(every)))
    meas_t = np.asarray(meas_t, dtype=float)[selected]
    meas_pos = np.asarray(meas_pos, dtype=float)[selected]
    measurements = np.full((len(imu_t), 3), np.nan, dtype=float)
    mask = np.zeros(len(imu_t), dtype=bool)
    indices = np.searchsorted(imu_t, meas_t)
    indices = np.clip(indices, 0, len(imu_t) - 1)
    for src_i, dst_i in enumerate(indices):
        measurements[dst_i] = meas_pos[src_i]
        mask[dst_i] = True
    duration = max(float(imu_t[-1] - imu_t[0]), 1.0e-9)
    return measurements, mask, float(np.count_nonzero(mask) / duration)


def load_numeric_table(path: Path) -> np.ndarray:
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    return np.asarray(rows, dtype=float)


def load_csv_with_optional_header(path: Path) -> tuple[list[str], np.ndarray]:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    tokens = [x.strip() for x in first.replace(" ", ",").split(",") if x.strip()]
    has_header = False
    try:
        [float(x) for x in tokens]
    except ValueError:
        has_header = True
    if has_header:
        header = [h.strip().lower() for h in first.split(",")]
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
    else:
        header = []
        data = np.genfromtxt(path, delimiter=",")
        if np.ndim(data) == 1 or np.isnan(data).all():
            data = load_numeric_table(path)
    return header, np.atleast_2d(np.asarray(data, dtype=float))


def pick_column(header: list[str], data: np.ndarray, names: list[str], fallback_idx: int) -> np.ndarray:
    if header:
        simplified = [h.replace("/", "_").replace(".", "_").lower() for h in header]
        for name in names:
            name = name.lower()
            for i, h in enumerate(simplified):
                if h == name or h.endswith("_" + name) or name in h:
                    return data[:, i]
    return data[:, fallback_idx]


def load_gnss_as_local_ned(path: Path, gt_t: np.ndarray, gt_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    header, data = load_csv_with_optional_header(path)
    t_raw = pick_column(header, data, ["time", "timestamp", "gps_time", "gpstime", "sow"], 0)
    lat = pick_column(header, data, ["lat", "latitude"], 1)
    lon = pick_column(header, data, ["lon", "longitude"], 2)
    alt = pick_column(header, data, ["alt", "altitude", "height"], 3)
    enu = lla_to_enu(lat, lon, alt, lat[0], lon[0], alt[0])
    ned = np.column_stack([enu[:, 1], enu[:, 0], -enu[:, 2]])
    t = np.asarray(t_raw, dtype=float).reshape(-1)
    gt_at_first = np.column_stack([np.interp(t[:1], gt_t, gt_pos[:, j]) for j in range(3)]).reshape(3)
    return t, ned + (gt_at_first - ned[0])


def lla_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    xyz = np.column_stack([ecef_from_lla(la, lo, al) for la, lo, al in zip(lat, lon, alt)]).T
    origin = ecef_from_lla(lat0, lon0, alt0)
    d = xyz - origin[None, :]
    la = math.radians(lat0)
    lo = math.radians(lon0)
    R = np.array([
        [-math.sin(lo), math.cos(lo), 0.0],
        [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
        [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
    ])
    return d @ R.T


def ecef_from_lla(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    a = 6378137.0
    e2 = 6.69437999014e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1.0 - e2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)
