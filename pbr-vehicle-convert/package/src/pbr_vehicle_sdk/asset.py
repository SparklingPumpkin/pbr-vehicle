"""Versioned dual-layer PBR Gaussian asset conversion and loading."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .config import ConversionConfig
from .ply import dc_rgb, gaussian_normals, normalize, positions, read_ply, require_standard_gaussian, write_ply


ASSET_SCHEMA = "pbr-vehicle-asset/1.0"


def _canonical_panel_config(asset_id: str, files: Mapping[str, Mapping[str, str]]) -> dict:
    """Emit the shared config-folder contract for the general KNN conversion path."""
    return {
        "schema_version": 9,
        "asset_contract": "pbr-vehicle-config-folder-v4",
        "asset_id": asset_id,
        "files": {
            "original": files["raw"]["path"],
            "proxy": files["proxy"]["path"],
            "mapping": files["mapping"]["path"],
            "config": f"configs/config_{asset_id}.json",
        },
        "mapping": {"mode": "raw_to_proxy_knn", "weights": "normalized"},
        "material": {
            "roughness": 0.32,
            "metallic": 0.02,
            "reflectance": 0.04,
            "clearcoat": 0.35,
            "clearcoat_roughness": 0.12,
            "exposure": 1.0,
            "ambient_fill": 0.35,
            "relight_strength": 1.0,
            "specular_gain": 1.0,
            "environment_reflection": 0.8,
        },
        "light": {
            "sun_enabled": True,
            "sun_azimuth_degrees": 126.0,
            "sun_elevation_degrees": 45.0,
            "intensity": 1.0,
            "color_rgb": [1.0, 0.98, 0.92],
            "sun_color_rgb": [1.0, 0.98, 0.92],
            "environment_color_rgb": [0.55, 0.62, 0.72],
        },
        "projection": {
            "enabled": True,
            "dynamic": True,
            "runtime": "receiver_space_mask_v1",
            "shape_model": "procedural-mask-contact-silhouette-v4",
            "composition": "receiver_alpha_blend_to_grayscale_v1",
            "contact": {"shape": "rounded_rectangle", "alignment": "vehicle_local_xy", "length_scale": 0.92, "width_scale": 0.88, "offset_xy_m": [0.0, 0.0], "corner_radius_m": 0.22, "edge_softness_m": 0.08, "opacity": 0.11, "brightness": 0.18},
            "extension": {"shape": "dynamic_vehicle_outline", "source": "parallel_light_projected_proxy_outline", "opacity": 0.28, "opacity_distance_decay": 0.55, "opacity_distance_exponent": 1.25, "brightness": 0.32, "brightness_distance_to_white": 0.65, "brightness_distance_exponent": 1.35, "edge_softness_m": 0.10, "edge_softness_distance_growth_m": 0.24, "edge_softness_distance_exponent": 1.15, "distance_scale_m": 1.25},
            "anchor": {"mode": "gaussian_bottom_surface", "bottom_surface_percentile": 1.0, "surface_sigma": 1.0, "z_offset_m": 0.0},
        },
        "source_asset_schema": ASSET_SCHEMA,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indexed(properties: Mapping[str, np.ndarray], prefix: str, count: int) -> np.ndarray:
    return np.stack([properties[f"{prefix}_{index}"] for index in range(count)], axis=1).astype(np.float32)


def _proxy_normals(points: np.ndarray, source_normals: np.ndarray, neighbors: int) -> np.ndarray:
    count = len(points)
    if count < 3:
        return source_normals.copy()
    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(neighbors, count))
    indices = np.atleast_2d(indices)
    normals = np.empty_like(points, dtype=np.float32)
    for index, local_indices in enumerate(indices):
        local = points[local_indices]
        covariance = np.cov((local - local.mean(axis=0)).T)
        normal = np.linalg.eigh(covariance)[1][:, 0]
        if np.dot(normal, source_normals[index]) < 0.0:
            normal = -normal
        normals[index] = normal
    return normalize(normals)


def _smooth_normals(points: np.ndarray, normals: np.ndarray, iterations: int = 6) -> np.ndarray:
    if len(points) < 3:
        return normals
    tree = cKDTree(points)
    _, neighbor_indices = tree.query(points, k=min(48, len(points)))
    neighbor_indices = np.atleast_2d(neighbor_indices)
    angular_sigma = np.deg2rad(32.0)
    result = normals.copy()
    for _ in range(iterations):
        neighbors = result[neighbor_indices]
        dot = np.clip(np.sum(neighbors * result[:, None, :], axis=2), -1.0, 1.0)
        signs = np.where(dot[..., None] < 0.0, -1.0, 1.0)
        angles = np.arccos(np.abs(dot))
        weights = np.exp(-0.5 * np.square(angles / angular_sigma))
        averaged = np.sum(neighbors * signs * weights[..., None], axis=1)
        result = normalize(0.28 * result + 0.72 * normalize(averaged))
    return result.astype(np.float32)


def _frame_quaternions(normals: np.ndarray) -> np.ndarray:
    quaternions = np.empty((len(normals), 4), dtype=np.float32)
    for index, normal in enumerate(normals):
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(reference, normal))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        tangent_x = normalize(np.cross(reference, normal)[None])[0]
        tangent_y = normalize(np.cross(normal, tangent_x)[None])[0]
        rotation = np.stack([tangent_x, tangent_y, normal], axis=1)
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array([
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ])
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                quaternion = np.array([(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale])
            elif axis == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                quaternion = np.array([(rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale])
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                quaternion = np.array([(rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale])
        quaternions[index] = normalize(quaternion[None])[0]
    return quaternions


def _build_proxy(source: Mapping[str, np.ndarray], config: ConversionConfig) -> OrderedDict[str, np.ndarray]:
    source_points = positions(source)
    source_rgb = dc_rgb(source)
    source_normals = gaussian_normals(source)
    origin = source_points.min(axis=0)
    voxel_keys = np.floor((source_points - origin) / config.voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)
    proxy_count = len(counts)
    proxy_points = np.zeros((proxy_count, 3), dtype=np.float64)
    proxy_rgb = np.zeros((proxy_count, 3), dtype=np.float64)
    proxy_source_normals = np.zeros((proxy_count, 3), dtype=np.float64)
    np.add.at(proxy_points, inverse, source_points)
    np.add.at(proxy_rgb, inverse, source_rgb)
    np.add.at(proxy_source_normals, inverse, source_normals)
    proxy_points = (proxy_points / counts[:, None]).astype(np.float32)
    proxy_rgb = (proxy_rgb / counts[:, None]).astype(np.float32)
    proxy_source_normals = normalize(proxy_source_normals)
    proxy_normals = _smooth_normals(
        proxy_points,
        _proxy_normals(proxy_points, proxy_source_normals, config.normal_neighbors),
    )
    quaternions = _frame_quaternions(proxy_normals)
    tangent_scale = max(config.voxel_size * config.surfel_overlap, 1e-5)
    normal_scale = max(config.voxel_size * 0.1, 1e-6)
    proxy = OrderedDict()
    for channel, axis in enumerate(("x", "y", "z")):
        proxy[axis] = proxy_points[:, channel]
    for channel in range(3):
        proxy[f"f_dc_{channel}"] = ((proxy_rgb[:, channel] - 0.5) / 0.28209479177387814).astype(np.float32)
    for channel in range(45):
        proxy[f"f_rest_{channel}"] = np.zeros(proxy_count, dtype=np.float32)
    proxy["opacity"] = np.full(proxy_count, 4.0, dtype=np.float32)
    for channel, scale in enumerate((tangent_scale, tangent_scale, normal_scale)):
        proxy[f"scale_{channel}"] = np.full(proxy_count, np.log(scale), dtype=np.float32)
    for channel in range(4):
        proxy[f"rot_{channel}"] = quaternions[:, channel]
    for channel in range(3):
        proxy[f"pbr_albedo_{channel}"] = proxy_rgb[:, channel]
        proxy[f"pbr_normal_{channel}"] = proxy_normals[:, channel]
    proxy["pbr_roughness"] = np.full(proxy_count, config.roughness, dtype=np.float32)
    proxy["pbr_metallic"] = np.full(proxy_count, config.metallic, dtype=np.float32)
    return proxy


def _build_mapping(
    source: Mapping[str, np.ndarray], proxy: Mapping[str, np.ndarray], neighbors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_points = positions(source)
    raw_normals = gaussian_normals(source)
    proxy_points = positions(proxy)
    proxy_normals = _indexed(proxy, "pbr_normal", 3)
    k = min(neighbors, len(proxy_points))
    distances, indices = cKDTree(proxy_points).query(raw_points, k=k)
    distances = np.asarray(distances, dtype=np.float32).reshape(len(raw_points), k)
    indices = np.asarray(indices, dtype=np.int64).reshape(len(raw_points), k)
    normal_dot = np.abs(np.sum(raw_normals[:, None, :] * proxy_normals[indices], axis=2))
    spatial_scale = max(float(np.median(distances[:, -1])), 1e-5)
    angular_sigma = np.deg2rad(28.0)
    angles = np.arccos(np.clip(normal_dot, 0.0, 1.0))
    weights = np.exp(-np.square(distances / spatial_scale)) * np.exp(-0.5 * np.square(angles / angular_sigma))
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    return indices, weights.astype(np.float32), distances


@dataclass(frozen=True)
class PBRAsset:
    root: Path
    manifest: dict
    raw: OrderedDict[str, np.ndarray]
    proxy: OrderedDict[str, np.ndarray]
    mapping_indices: np.ndarray
    mapping_weights: np.ndarray

    @property
    def raw_count(self) -> int:
        return len(next(iter(self.raw.values())))

    @property
    def proxy_count(self) -> int:
        return len(next(iter(self.proxy.values())))


def load_asset(path: str | Path, verify_hashes: bool = True) -> PBRAsset:
    root = Path(path).resolve()
    manifest_path = root / "asset.json"
    if not manifest_path.is_file():
        raise ValueError(f"PBR asset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ASSET_SCHEMA:
        raise ValueError(f"Unsupported PBR asset schema: {manifest.get('schema')!r}")
    files = manifest["files"]
    resolved = {name: root / entry["path"] for name, entry in files.items()}
    if any(root not in file_path.resolve().parents for file_path in resolved.values()):
        raise ValueError("Asset manifest paths must stay inside the asset directory")
    for name, file_path in resolved.items():
        if not file_path.is_file():
            raise ValueError(f"Missing asset file {name!r}: {file_path}")
        if verify_hashes and _sha256(file_path) != files[name]["sha256"]:
            raise ValueError(f"Checksum mismatch for asset file {name!r}")
    raw = read_ply(resolved["raw"])
    proxy = read_ply(resolved["proxy"])
    require_standard_gaussian(raw)
    required_proxy = {f"pbr_albedo_{channel}" for channel in range(3)} | {
        f"pbr_normal_{channel}" for channel in range(3)
    } | {"pbr_roughness", "pbr_metallic"}
    if missing := sorted(required_proxy.difference(proxy)):
        raise ValueError("PBR proxy is missing fields: " + ", ".join(missing))
    with np.load(resolved["mapping"], allow_pickle=False) as mapping:
        indices = np.asarray(mapping["raw_to_proxy_knn_idx"], dtype=np.int64)
        weights = np.asarray(mapping["raw_to_proxy_knn_weight"], dtype=np.float32)
    if indices.shape != weights.shape or indices.shape[0] != len(next(iter(raw.values()))):
        raise ValueError("Raw-to-proxy mapping shape does not match the raw Gaussian layer")
    if indices.min() < 0 or indices.max() >= len(next(iter(proxy.values()))):
        raise ValueError("Raw-to-proxy mapping contains an invalid proxy index")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Raw-to-proxy mapping weights are not normalized")
    return PBRAsset(root, manifest, raw, proxy, indices, weights)


def convert_asset(
    source_ply: str | Path,
    output_dir: str | Path,
    config: ConversionConfig | None = None,
    overwrite: bool = False,
) -> PBRAsset:
    settings = config or ConversionConfig()
    source = Path(source_ply).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {destination}")
    raw = read_ply(source)
    require_standard_gaussian(raw)
    proxy = _build_proxy(raw, settings)
    indices, weights, distances = _build_mapping(raw, proxy, settings.mapping_neighbors)
    raw_path = destination / "raw" / "gaussians.ply"
    proxy_path = destination / "proxy" / "gaussians_pbr.ply"
    mapping_path = destination / "mapping" / "raw_to_proxy_knn.npz"
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, raw_path)
    write_ply(proxy_path, proxy)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mapping_path,
        raw_to_proxy_knn_idx=indices,
        raw_to_proxy_knn_weight=weights,
        raw_to_proxy_knn_distance=distances,
    )
    files = {
        "raw": {"path": "raw/gaussians.ply", "sha256": _sha256(raw_path)},
        "proxy": {"path": "proxy/gaussians_pbr.ply", "sha256": _sha256(proxy_path)},
        "mapping": {"path": "mapping/raw_to_proxy_knn.npz", "sha256": _sha256(mapping_path)},
    }
    asset_id = re.sub(r"[^A-Za-z0-9_-]+", "_", destination.name).strip("_") or "vehicle"
    config_directory = destination / "configs"
    config_directory.mkdir(parents=True, exist_ok=True)
    canonical_config_path = config_directory / f"config_{asset_id}.json"
    manifest = {
        "schema": ASSET_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": source.name,
        "raw_gaussian_count": len(next(iter(raw.values()))),
        "proxy_gaussian_count": len(next(iter(proxy.values()))),
        "conversion": asdict(settings),
        "files": files,
    }
    (destination / "asset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    canonical_config_path.write_text(
        json.dumps(_canonical_panel_config(asset_id, files), indent=2) + "\n", encoding="utf-8"
    )
    return load_asset(destination)
