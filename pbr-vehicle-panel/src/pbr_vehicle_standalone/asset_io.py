from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

from .math3d import normalize, quaternion_to_rotation, sigmoid
from .types import GaussianLayer, VehicleAsset


SH_C0 = 0.28209479177387814
SUPPORTED_CONTRACTS = {"pbr-vehicle-config-folder-v2", "pbr-vehicle-config-folder-v4", "pbr-vehicle-single-ply-v1"}


def _read_vertices(path: Path) -> np.ndarray:
    return PlyData.read(str(path))["vertex"].data


def _has(data: np.ndarray, names: list[str]) -> bool:
    return all(name in (data.dtype.names or ()) for name in names)


def _stack(data: np.ndarray, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(data[name], dtype=np.float32) for name in names], axis=1)


def _colors(data: np.ndarray) -> np.ndarray:
    if _has(data, ["f_dc_0", "f_dc_1", "f_dc_2"]):
        return np.clip(SH_C0 * _stack(data, ["f_dc_0", "f_dc_1", "f_dc_2"]) + 0.5, 0.0, 1.0)
    if _has(data, ["red", "green", "blue"]):
        return np.clip(_stack(data, ["red", "green", "blue"]) / 255.0, 0.0, 1.0)
    return np.full((len(data), 3), 0.5, dtype=np.float32)


def _covariances(data: np.ndarray) -> np.ndarray:
    required = ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    if not _has(data, required):
        return np.tile(np.eye(3, dtype=np.float32)[None], (len(data), 1, 1)) * 0.0025
    scales = np.exp(np.clip(_stack(data, required[:3]), -20.0, 20.0))
    rotations = quaternion_to_rotation(_stack(data, required[3:]))
    return np.einsum("nij,nj,nkj->nik", rotations, np.square(scales), rotations).astype(np.float32)


def _normals(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    for names in (["r3gw_normal_0", "r3gw_normal_1", "r3gw_normal_2"], ["normal_0", "normal_1", "normal_2"]):
        if _has(data, list(names)):
            result = normalize(_stack(data, list(names)))
            outward = centers - np.median(centers, axis=0, keepdims=True)
            signs = np.sign(np.sum(result * outward, axis=1, keepdims=True))
            signs[signs == 0] = 1
            return (result * signs).astype(np.float32)
    required = ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    if not _has(data, required):
        result = np.zeros((len(data), 3), dtype=np.float32)
        result[:, 2] = 1.0
        return result
    scales = np.exp(np.clip(_stack(data, required[:3]), -20.0, 20.0))
    rotations = quaternion_to_rotation(_stack(data, required[3:]))
    result = rotations[np.arange(len(data)), :, np.argmin(scales, axis=1)]
    outward = centers - np.median(centers, axis=0, keepdims=True)
    signs = np.sign(np.sum(result * outward, axis=1, keepdims=True))
    signs[signs == 0] = 1
    return (normalize(result) * signs).astype(np.float32)


def _albedo(data: np.ndarray) -> np.ndarray:
    if _has(data, ["r3gw_albedo_0", "r3gw_albedo_1", "r3gw_albedo_2"]):
        return np.clip(_stack(data, ["r3gw_albedo_0", "r3gw_albedo_1", "r3gw_albedo_2"]), 0.0, 1.0)
    if _has(data, ["albedo_0", "albedo_1", "albedo_2"]):
        return np.clip(sigmoid(_stack(data, ["albedo_0", "albedo_1", "albedo_2"])), 0.0, 1.0)
    return _colors(data)


def _material_channel(data: np.ndarray, direct: str, logit: str, default: float, low: float, high: float) -> np.ndarray:
    if direct in (data.dtype.names or ()):
        values = np.asarray(data[direct], dtype=np.float32)[:, None]
    elif logit in (data.dtype.names or ()):
        values = sigmoid(np.asarray(data[logit], dtype=np.float32))[:, None]
    else:
        values = np.full((len(data), 1), default, dtype=np.float32)
    return np.clip(values, low, high)


def load_gaussian_layer(path: str | Path, opacity_scale: float = 1.0,
                        center: bool = False) -> GaussianLayer:
    source = Path(path).expanduser().resolve()
    data_all = _read_vertices(source)
    all_opacities = np.ones((len(data_all), 1), dtype=np.float32)
    if "opacity" in (data_all.dtype.names or ()):
        all_opacities = np.clip(sigmoid(np.asarray(data_all["opacity"], dtype=np.float32))[:, None] * opacity_scale, 0, 1)
    indices = np.arange(len(data_all), dtype=np.int64)
    data = data_all[indices]
    centers = _stack(data, ["x", "y", "z"])
    covariances = _covariances(data)
    normals = _normals(data, centers)
    if center:
        low, high = np.percentile(centers, [1.0, 99.0], axis=0)
        centers = centers - (low + high)[None, :] * 0.5
        centers[:, 2] -= np.percentile(centers[:, 2], 1.0)
    arrays = (centers, covariances, all_opacities[indices])
    if not all(np.isfinite(item).all() for item in arrays):
        raise ValueError(f"Non-finite Gaussian data in {source}")
    return GaussianLayer(
        centers=np.ascontiguousarray(centers, dtype=np.float32),
        covariances=np.ascontiguousarray(covariances, dtype=np.float32),
        colors=np.ascontiguousarray(_colors(data), dtype=np.float32),
        opacities=np.ascontiguousarray(all_opacities[indices], dtype=np.float32),
        normals=np.ascontiguousarray(normals, dtype=np.float32),
        albedo=np.ascontiguousarray(_albedo(data), dtype=np.float32),
        roughness=np.ascontiguousarray(_material_channel(data, "r3gw_roughness", "roughness", 0.4, 0.02, 0.98)),
        metallic=np.ascontiguousarray(_material_channel(data, "r3gw_metallic", "metalness", 0.0, 0.0, 1.0)),
        source_indices=indices,
        total_splats=len(data_all),
        path=str(source),
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return payload


def resolve_asset_folder(folder: str | Path) -> dict[str, Any]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    asset_configs = []
    for path in sorted((root / "configs").glob("config*.json")):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("asset_contract") in SUPPORTED_CONTRACTS:
            asset_configs.append((path, payload))
    if not asset_configs:
        raise ValueError(f"No canonical PBR asset config under {root / 'configs'}")
    canonical_path, canonical = next(
        ((path, payload) for path, payload in asset_configs
         if (payload.get("files") or {}).get("config") and
         (root / payload["files"]["config"]).resolve() == path.resolve()),
        asset_configs[0],
    )
    files = canonical.get("files", {})
    if canonical.get("asset_contract") == "pbr-vehicle-single-ply-v1":
        required = {"pbr"}
        if not required.issubset(files):
            raise ValueError(f"Canonical config is missing files {sorted(required - set(files))}: {canonical_path}")
        pbr = (root / files["pbr"]).resolve()
        if not pbr.is_file():
            raise FileNotFoundError(f"Missing vehicle asset file: {pbr}")
        return {"root": root, "config_path": canonical_path.resolve(), "config": canonical, "pbr": pbr}
    required = {"original", "proxy", "mapping"}
    if not required.issubset(files):
        raise ValueError(f"Canonical config is missing files {sorted(required - set(files))}: {canonical_path}")
    resolved = {key: (root / files[key]).resolve() for key in required}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing vehicle asset files: {missing}")
    return {"root": root, "config_path": canonical_path.resolve(), "config": canonical, **resolved}


def load_vehicle_asset(folder: str | Path) -> VehicleAsset:
    resolved = resolve_asset_folder(folder)
    if resolved["config"].get("asset_contract") == "pbr-vehicle-single-ply-v1":
        layer = load_gaussian_layer(resolved["pbr"], center=True)
        material = resolved["config"].get("material", {})
        albedo = np.asarray(material.get("albedo_rgb", [0.82, 0.82, 0.82]), dtype=np.float32)
        if albedo.shape != (3,):
            raise ValueError("material.albedo_rgb must contain three values")
        layer.albedo = np.broadcast_to(albedo[None], (len(layer.centers), 3)).copy()
        layer.roughness = np.full((len(layer.centers), 1), float(material.get("roughness", 0.32)), dtype=np.float32)
        layer.metallic = np.full((len(layer.centers), 1), float(material.get("metallic", 0.02)), dtype=np.float32)
        count = len(layer.centers)
        identity = np.arange(count, dtype=np.int64)
        projection = copy.deepcopy(resolved["config"].get("projection", {}))
        return VehicleAsset(
            root=resolved["root"], asset_id=str(resolved["config"].get("asset_id") or resolved["root"].name),
            original=layer, proxy=layer, original_to_proxy=identity,
            mapping_indices=identity[:, None], mapping_weights=np.ones((count, 1), dtype=np.float32),
            canonical_config_path=resolved["config_path"], canonical_config=resolved["config"], projection=projection,
        )
    proxy = load_gaussian_layer(resolved["proxy"], center=True)
    original = load_gaussian_layer(resolved["original"], center=True)
    mapping = np.load(str(resolved["mapping"]))
    source_indices = original.source_indices
    if source_indices is None or source_indices.size == 0:
        raise ValueError("Original layer indices are incompatible with mapping")
    if "raw_to_proxy_knn_idx" in mapping and "raw_to_proxy_knn_weight" in mapping:
        indices = np.asarray(mapping["raw_to_proxy_knn_idx"], dtype=np.int64)
        weights = np.asarray(mapping["raw_to_proxy_knn_weight"], dtype=np.float32)
    elif "clean_to_proxy_idx" in mapping:
        indices = np.asarray(mapping["clean_to_proxy_idx"], dtype=np.int64)[:, None]
        weights = np.ones_like(indices, dtype=np.float32)
    else:
        raise ValueError(
            "Mapping needs raw_to_proxy_knn_idx/raw_to_proxy_knn_weight or clean_to_proxy_idx: "
            f"{resolved['mapping']}"
        )
    if indices.ndim != 2 or weights.shape != indices.shape or int(source_indices.max()) >= len(indices):
        raise ValueError("Original layer indices are incompatible with mapping")
    mapping_indices = np.ascontiguousarray(indices[source_indices], dtype=np.int64)
    mapping_weights = np.ascontiguousarray(weights[source_indices], dtype=np.float32)
    mapping_weights /= np.clip(mapping_weights.sum(axis=1, keepdims=True), 1e-8, None)
    if mapping_indices.min() < 0 or mapping_indices.max() >= len(proxy.centers):
        raise ValueError("Mapping contains proxy indices outside the loaded proxy layer")
    original_to_proxy = np.ascontiguousarray(mapping_indices[:, 0], dtype=np.int64)
    projection = copy.deepcopy(resolved["config"].get("projection", {}))
    stats = resolved["config"].get("projection_stats", {})
    if isinstance(stats, dict) and np.isfinite(float(stats.get("ground_z", np.nan))):
        # Layers are centered independently; re-resolve the fixed anchor from the loaded proxy.
        projection["_fixed_anchor"] = copy.deepcopy(projection.get("anchor", {}))
        anchor = projection.get("anchor", {})
        bottom = proxy.centers[:, 2] - float(anchor.get("surface_sigma", 1.0)) * np.sqrt(
            np.clip(proxy.covariances[:, 2, 2], 0.0, None)
        )
        projection["_fixed_ground_z"] = float(np.percentile(bottom, float(anchor.get("bottom_surface_percentile", 1.0)))) + float(anchor.get("z_offset_m", 0.0))
    return VehicleAsset(
        root=resolved["root"],
        asset_id=str(resolved["config"].get("asset_id") or resolved["root"].name),
        original=original,
        proxy=proxy,
        original_to_proxy=original_to_proxy,
        mapping_indices=mapping_indices,
        mapping_weights=mapping_weights,
        canonical_config_path=resolved["config_path"],
        canonical_config=resolved["config"],
        projection=projection,
    )


def _checkpoint_background(state: Any) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    for models in (state.get("models"), (state.get("state_dict") or {}).get("models") if isinstance(state.get("state_dict"), dict) else None):
        if isinstance(models, dict) and isinstance(models.get("Background"), dict):
            return models["Background"]
    for prefix in ("models.Background.", "Background."):
        found = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if found:
            return found
    return None


def materialize_pth_scene(path: str | Path, cache_dir: str | Path) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PTH scene support requires: pip install 'pbr-vehicle-panel[pth]'") from exc
    source = Path(path).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"{source.stem}_{hashlib.sha256(str(source).encode()).hexdigest()[:12]}.ply"
    if output.is_file() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return output
    paired = source.with_suffix(".ply")
    if paired.is_file():
        output.write_bytes(paired.read_bytes())
        return output
    state = torch.load(str(source), map_location="cpu", weights_only=False)
    background = _checkpoint_background(state)
    if background is None:
        raise ValueError("Checkpoint has no serialized Background model")
    def array(key: str, columns: int) -> np.ndarray:
        item = background.get(key)
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        result = np.asarray(item, dtype=np.float32).reshape(len(background["_means"]), -1)
        return result[:, :columns]
    means = array("_means", 3)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")]
    fields += [(f"f_dc_{i}", "f4") for i in range(3)] + [(f"scale_{i}", "f4") for i in range(3)] + [(f"rot_{i}", "f4") for i in range(4)]
    vertex = np.empty(len(means), dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = means.T
    vertex["opacity"] = array("_opacities", 1)[:, 0]
    for i, values in enumerate(array("_features_dc", 3).T): vertex[f"f_dc_{i}"] = values
    for i, values in enumerate(array("_scales", 3).T): vertex[f"scale_{i}"] = values
    for i, values in enumerate(array("_quats", 4).T): vertex[f"rot_{i}"] = values
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(output))
    return output


def load_scene(path: str | Path,
               cache_dir: str | Path = ".cache/pbr_vehicle_scenes") -> GaussianLayer:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".pth":
        source = materialize_pth_scene(source, cache_dir)
    if source.suffix.lower() != ".ply":
        raise ValueError(f"Scene must be .ply or .pth: {source}")
    return load_gaussian_layer(source, center=False)
