"""Ordinary scene Gaussian loading, vehicle integration, and scene baking."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .asset import PBRAsset
from .config import RenderConfig
from .ply import SH_C0, dc_rgb, positions, read_ply, require_standard_gaussian, write_ply
from .shading import RelightResult, relight


@dataclass(frozen=True)
class GaussianScene:
    source_path: Path
    properties: OrderedDict[str, np.ndarray]

    @property
    def gaussian_count(self) -> int:
        return len(next(iter(self.properties.values())))


@dataclass(frozen=True)
class SceneBakeResult:
    output_ply: Path
    scene_gaussian_count: int
    vehicle_gaussian_count: int
    shadowed_gaussian_count: int
    relight: RelightResult


def load_scene_ply(path: str | Path) -> GaussianScene:
    source = Path(path).resolve()
    properties = read_ply(source)
    require_standard_gaussian(properties)
    return GaussianScene(source, properties)


def _yaw_rotation(yaw_deg: float) -> np.ndarray:
    angle = math.radians(float(yaw_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _yaw_quaternion(yaw_deg: float) -> np.ndarray:
    half_angle = math.radians(float(yaw_deg)) * 0.5
    return np.asarray([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)], dtype=np.float32)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.broadcast_to(np.asarray(left, dtype=np.float32), right.shape)
    lw, lx, ly, lz = left.T
    rw, rx, ry, rz = right.T
    result = np.stack([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], axis=1)
    return result / np.clip(np.linalg.norm(result, axis=1, keepdims=True), 1e-8, None)


def transformed_vehicle_properties(
    asset: PBRAsset,
    colors: np.ndarray,
    config: RenderConfig,
) -> OrderedDict[str, np.ndarray]:
    properties = asset.raw.copy()
    rotation = _yaw_rotation(config.placement.yaw_deg)
    translation = np.asarray(config.placement.translation, dtype=np.float32)
    transformed = positions(properties) * float(config.placement.scale) @ rotation.T + translation
    for channel, name in enumerate(("x", "y", "z")):
        properties[name] = transformed[:, channel].astype(np.float32)
    for channel in range(3):
        properties[f"f_dc_{channel}"] = ((colors[:, channel] - 0.5) / SH_C0).astype(np.float32)
    for name in properties:
        if name.startswith("f_rest_"):
            properties[name] = np.zeros_like(properties[name], dtype=np.float32)
        elif name.startswith("scale_"):
            properties[name] = (properties[name] + math.log(config.placement.scale)).astype(np.float32)
    quaternions = np.stack([properties[f"rot_{index}"] for index in range(4)], axis=1)
    quaternions = _quaternion_multiply(_yaw_quaternion(config.placement.yaw_deg), quaternions)
    for channel in range(4):
        properties[f"rot_{channel}"] = quaternions[:, channel].astype(np.float32)
    return properties


def _apply_vehicle_shadow(
    scene: Mapping[str, np.ndarray],
    vehicle: Mapping[str, np.ndarray],
    config: RenderConfig,
) -> tuple[OrderedDict[str, np.ndarray], int]:
    result = OrderedDict((name, np.asarray(values).copy()) for name, values in scene.items())
    if not config.integration.vehicle_shadow:
        return result, 0
    scene_points = positions(scene)
    vehicle_points = positions(vehicle)
    center_xy = np.percentile(vehicle_points[:, :2], 50.0, axis=0)
    local_xy = (scene_points[:, :2] - center_xy) @ _yaw_rotation(-config.placement.yaw_deg)[:2, :2].T
    vehicle_local = (vehicle_points[:, :2] - center_xy) @ _yaw_rotation(-config.placement.yaw_deg)[:2, :2].T
    vehicle_span = np.maximum(
        np.percentile(vehicle_local, 99.0, axis=0) - np.percentile(vehicle_local, 1.0, axis=0),
        1e-3,
    )
    half_extent = 0.5 * vehicle_span * np.asarray(
        [config.shadow.length_scale, config.shadow.width_scale], dtype=np.float32
    )
    normalized = np.abs(local_xy) / half_extent
    signed_distance = np.max(normalized, axis=1)
    edge = max(config.shadow.edge_softness, 1e-4)
    footprint = np.clip((1.0 + edge - signed_distance) / edge, 0.0, 1.0)
    ground_z = float(np.percentile(vehicle_points[:, 2], 1.0))
    ground_weight = np.exp(-0.5 * np.square((scene_points[:, 2] - ground_z) / config.shadow.ground_band))
    alpha = footprint * ground_weight * config.shadow.strength
    scene_rgb = dc_rgb(scene) * (1.0 - alpha[:, None])
    for channel in range(3):
        result[f"f_dc_{channel}"] = ((scene_rgb[:, channel] - 0.5) / SH_C0).astype(np.float32)
    return result, int(np.count_nonzero(alpha > 1e-3))


def _standard_union(
    scene: Mapping[str, np.ndarray], vehicle: Mapping[str, np.ndarray] | None
) -> OrderedDict[str, np.ndarray]:
    if vehicle is None:
        return OrderedDict((name, np.asarray(values).copy()) for name, values in scene.items())
    required_order = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    rest_names = sorted(
        {name for source in (scene, vehicle) for name in source if name.startswith("f_rest_")},
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    standard_names = required_order[:6] + rest_names + required_order[6:]
    extra_names = [name for source in (scene, vehicle) for name in source if name not in standard_names]
    names = standard_names + list(dict.fromkeys(extra_names))
    merged: OrderedDict[str, np.ndarray] = OrderedDict()
    for name in names:
        sample = scene.get(name, vehicle.get(name))
        dtype = np.asarray(sample).dtype
        scene_values = np.asarray(scene.get(name, np.zeros(len(next(iter(scene.values()))), dtype=dtype)))
        vehicle_values = np.asarray(vehicle.get(name, np.zeros(len(next(iter(vehicle.values()))), dtype=dtype)))
        merged[name] = np.concatenate([scene_values, vehicle_values]).astype(dtype)
    return merged


def compose_scene_properties(
    scene: GaussianScene,
    asset: PBRAsset,
    config: RenderConfig | None = None,
) -> tuple[OrderedDict[str, np.ndarray], RelightResult, int]:
    settings = config or RenderConfig()
    vehicle_relight = relight(asset, settings)
    vehicle = transformed_vehicle_properties(asset, vehicle_relight.raw_rgb, settings)
    scene_properties, shadowed_count = _apply_vehicle_shadow(scene.properties, vehicle, settings)
    projected_vehicle = vehicle if settings.integration.vehicle_projection else None
    return _standard_union(scene_properties, projected_vehicle), vehicle_relight, shadowed_count


def bake_scene(
    scene: GaussianScene,
    asset: PBRAsset,
    output_ply: str | Path,
    config: RenderConfig | None = None,
) -> SceneBakeResult:
    settings = config or RenderConfig()
    combined, vehicle_relight, shadowed_count = compose_scene_properties(scene, asset, settings)
    output = write_ply(output_ply, combined)
    projected_count = asset.raw_count if settings.integration.vehicle_projection else 0
    metadata = {
        "schema": "pbr-scene-bake/1.0",
        "output_ply": output.name,
        "scene_gaussian_count": scene.gaussian_count,
        "vehicle_gaussian_count": projected_count,
        "shadowed_gaussian_count": shadowed_count,
        "integration": settings.to_dict()["integration"],
        "render_config": settings.to_dict(),
    }
    output.with_suffix(output.suffix + ".bake.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SceneBakeResult(output, scene.gaussian_count, projected_count, shadowed_count, vehicle_relight)