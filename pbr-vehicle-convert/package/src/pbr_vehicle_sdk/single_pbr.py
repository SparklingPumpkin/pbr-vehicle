"""Minimal self-contained PBR Gaussian PLY construction."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np

from .ply import normalize, read_ply, require_standard_gaussian, write_ply


def _rotations(quaternions: np.ndarray) -> np.ndarray:
    q = normalize(quaternions.astype(np.float32))
    w, x, y, z = q.T
    result = np.empty((len(q), 3, 3), dtype=np.float32)
    result[:, 0, 0] = 1 - 2 * (y*y + z*z); result[:, 0, 1] = 2 * (x*y - z*w); result[:, 0, 2] = 2 * (x*z + y*w)
    result[:, 1, 0] = 2 * (x*y + z*w); result[:, 1, 1] = 1 - 2 * (x*x + z*z); result[:, 1, 2] = 2 * (y*z - x*w)
    result[:, 2, 0] = 2 * (x*z - y*w); result[:, 2, 1] = 2 * (y*z + x*w); result[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return result


def gaussian_normals(properties: Mapping[str, np.ndarray]) -> np.ndarray:
    scales = np.stack([properties[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)
    rotations = _rotations(np.stack([properties[f"rot_{i}"] for i in range(4)], axis=1))
    axis = np.argmin(scales, axis=1)
    normals = rotations[np.arange(len(axis)), :, axis]
    points = np.stack([properties[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)
    normals[np.sum(normals * (points - np.median(points, axis=0)), axis=1) < 0.0] *= -1.0
    return normalize(normals).astype(np.float32)


def build_single_pbr_ply(input_ply: str | Path, output_ply: str | Path) -> Path:
    source = read_ply(input_ply)
    require_standard_gaussian(source)
    if any(f"normal_{i}" in source for i in range(3)):
        raise ValueError("Input already has normal_0/1/2; refusing ambiguous PBR conversion")
    result: OrderedDict[str, np.ndarray] = OrderedDict(source)
    normals = gaussian_normals(source)
    for index in range(3):
        result[f"normal_{index}"] = normals[:, index]
    return write_ply(output_ply, result)
