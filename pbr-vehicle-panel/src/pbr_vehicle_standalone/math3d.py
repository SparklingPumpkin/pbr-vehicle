from __future__ import annotations

import math

import numpy as np


def _blackbody_rgb(kelvin: float) -> np.ndarray:
    temperature = float(np.clip(kelvin, 1000.0, 40000.0)) / 100.0
    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
        blue = 0.0 if temperature <= 19.0 else 138.5177312231 * math.log(temperature - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * (temperature - 60.0) ** -0.1332047592
        green = 288.1221695283 * (temperature - 60.0) ** -0.0755148492
        blue = 255.0
    return np.clip(np.asarray([red, green, blue], dtype=np.float32), 0.0, 255.0) / 255.0


_NEUTRAL_6500 = _blackbody_rgb(6500.0)


def color_temperature_to_rgb(kelvin: float) -> np.ndarray:
    return np.clip(_blackbody_rgb(kelvin) / np.clip(_NEUTRAL_6500, 1e-6, None), 0.0, 3.0).astype(np.float32)


def rgb_to_color_temperature(rgb, default: float = 6500.0) -> float:
    values = np.asarray(rgb, dtype=np.float32).reshape(-1)[:3]
    if len(values) < 3 or not np.isfinite(values).all() or float(np.linalg.norm(values)) < 1e-6:
        return float(default)
    target = values / np.linalg.norm(values)
    candidates = np.arange(2000.0, 12001.0, 25.0, dtype=np.float32)
    gains = np.stack([color_temperature_to_rgb(value) for value in candidates])
    gains /= np.clip(np.linalg.norm(gains, axis=1, keepdims=True), 1e-6, None)
    return float(candidates[int(np.argmin(np.sum((gains - target[None, :]) ** 2, axis=1)))])


def adjust_rgb_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    luminance = np.sum(values * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1, keepdims=True)
    return np.clip(luminance + float(saturation) * (values - luminance), 0.0, 1.0).astype(np.float32)


def normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), eps, None)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def quaternion_to_rotation(quaternions: np.ndarray) -> np.ndarray:
    quat = normalize(np.asarray(quaternions, dtype=np.float32))
    w, x, y, z = quat.T
    rotations = np.empty((len(quat), 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def euler_to_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return normalize(np.asarray([[
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]], dtype=np.float32))[0]


def sun_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    return normalize(np.asarray([[
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ]], dtype=np.float32))[0]
