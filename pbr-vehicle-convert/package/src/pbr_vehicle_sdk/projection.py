"""Mask-runtime vehicle projection contract and non-Gaussian outline cache."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .ply import gaussian_covariances


MASK_RUNTIME = "receiver_space_mask_v1"
MASK_MODEL = "procedural-mask-contact-silhouette-v4"


def _sun_direction(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    elevation = math.radians(float(elevation_degrees))
    azimuth = math.radians(float(azimuth_degrees))
    if not 1.0 <= float(elevation_degrees) <= 89.0:
        raise ValueError("sun elevation must be in [1, 89] degrees")
    return np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float32,
    )


def _vehicle_bottom_anchor(proxy: Mapping[str, np.ndarray], anchor: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """Resolve the receiver plane from the lower Gaussian support surface."""
    mode = anchor.get("mode", "gaussian_bottom_surface")
    if mode != "gaussian_bottom_surface":
        raise ValueError("projection.anchor.mode must be gaussian_bottom_surface")
    percentile = float(anchor.get("bottom_surface_percentile", 1.0))
    sigma = float(anchor.get("surface_sigma", 1.0))
    z_offset = float(anchor.get("z_offset_m", 0.0))
    if not 0.0 <= percentile <= 10.0:
        raise ValueError("projection.anchor.bottom_surface_percentile must be in [0, 10]")
    if sigma < 0.0:
        raise ValueError("projection.anchor.surface_sigma must be non-negative")
    covariance_z = gaussian_covariances(proxy)[:, 2, 2]
    lower_surface = np.asarray(proxy["z"], dtype=np.float32) - sigma * np.sqrt(np.clip(covariance_z, 0.0, None))
    base_z = float(np.percentile(lower_surface, percentile))
    return base_z + z_offset, {
        "mode": mode,
        "bottom_surface_percentile": percentile,
        "surface_sigma": sigma,
        "z_offset_m": z_offset,
        "bottom_surface_z": base_z,
    }


def _require_unit_interval(value: Any, label: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _require_non_negative(value: Any, label: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def validate_projection_mask_config(projection: Mapping[str, Any]) -> None:
    """Validate the renderer-independent two-layer receiver-mask contract."""
    if projection.get("runtime") != MASK_RUNTIME:
        raise ValueError(f"projection.runtime must be {MASK_RUNTIME}")
    if projection.get("shape_model") != MASK_MODEL:
        raise ValueError(f"projection.shape_model must be {MASK_MODEL}")

    contact = projection["contact"]
    if contact.get("shape") != "rounded_rectangle":
        raise ValueError("projection.contact.shape must be rounded_rectangle")
    if contact.get("alignment") != "vehicle_local_xy":
        raise ValueError("projection.contact.alignment must be vehicle_local_xy")
    if float(contact["length_scale"]) <= 0.0 or float(contact["width_scale"]) <= 0.0:
        raise ValueError("projection.contact length_scale and width_scale must be positive")
    offset = contact["offset_xy_m"]
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        raise ValueError("projection.contact.offset_xy_m must contain [x, y] meters")
    _require_non_negative(contact["corner_radius_m"], "projection.contact.corner_radius_m")
    _require_non_negative(contact["edge_softness_m"], "projection.contact.edge_softness_m")
    _require_unit_interval(contact["opacity"], "projection.contact.opacity")
    _require_unit_interval(contact["brightness"], "projection.contact.brightness")

    extension = projection["extension"]
    if extension.get("shape") != "dynamic_vehicle_outline":
        raise ValueError("projection.extension.shape must be dynamic_vehicle_outline")
    if extension.get("source") != "parallel_light_projected_pbr_gaussian_outline":
        raise ValueError("projection.extension.source must be parallel_light_projected_pbr_gaussian_outline")
    _require_unit_interval(extension["opacity"], "projection.extension.opacity")
    _require_unit_interval(extension["brightness"], "projection.extension.brightness")
    _require_unit_interval(extension["opacity_distance_decay"], "projection.extension.opacity_distance_decay")
    _require_unit_interval(extension["brightness_distance_to_white"], "projection.extension.brightness_distance_to_white")
    if float(extension["opacity_distance_exponent"]) <= 0.0 or float(extension["brightness_distance_exponent"]) <= 0.0:
        raise ValueError("projection.extension distance exponents must be positive")
    _require_non_negative(extension["edge_softness_m"], "projection.extension.edge_softness_m")
    _require_non_negative(extension["edge_softness_distance_growth_m"], "projection.extension.edge_softness_distance_growth_m")
    if float(extension["edge_softness_distance_exponent"]) <= 0.0:
        raise ValueError("projection.extension.edge_softness_distance_exponent must be positive")
    if float(extension["distance_scale_m"]) <= 0.0:
        raise ValueError("projection.extension.distance_scale_m must be positive")


def _outline_vertices(points_xy: np.ndarray) -> np.ndarray:
    """Return ordered outer boundary samples without creating Gaussian splats."""
    unique = np.unique(np.asarray(points_xy, dtype=np.float32), axis=0)
    if len(unique) < 3:
        return unique
    try:
        hull = ConvexHull(unique)
        return unique[hull.vertices]
    except QhullError:
        center = unique.mean(axis=0)
        angles = np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0])
        return unique[np.argsort(angles)]


def projection_mask_descriptor(
    proxy: Mapping[str, np.ndarray],
    light: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe current mask layers for a receiver or screen-space renderer."""
    validate_projection_mask_config(projection)
    points = np.stack([proxy[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)
    ground_z, anchor = _vehicle_bottom_anchor(proxy, projection["anchor"])
    direction = _sun_direction(light["sun_azimuth_degrees"], light["sun_elevation_degrees"])
    height = np.maximum(points[:, 2] - ground_z, 0.0)
    cast = points - (height / max(float(direction[2]), 1e-5))[:, None] * direction[None, :]
    cast[:, 2] = ground_z

    contact = projection["contact"]
    # Use stable asset-local x/y axes so contact geometry does not rotate as a
    # consequence of a projected outline or point-density variation.
    low, high = np.quantile(points[:, :2], [0.01, 0.99], axis=0)
    full_size = np.maximum(high - low, 1e-4)
    center = (low + high) * 0.5 + np.asarray(contact["offset_xy_m"], dtype=np.float32)
    size = full_size * np.asarray([contact["length_scale"], contact["width_scale"]], dtype=np.float32)
    outline_xy = _outline_vertices(cast[:, :2])
    outline_xyz = np.column_stack([outline_xy, np.full(len(outline_xy), ground_z, dtype=np.float32)]).astype(np.float32)
    return {
        "runtime": MASK_RUNTIME,
        "model": MASK_MODEL,
        "geometry": "analytic-rounded-rectangle-plus-parallel-light-projected-pbr-gaussian-outline",
        "uses_gaussian_splats": False,
        "ground_z": ground_z,
        "vehicle_z_anchor": anchor,
        "sun_direction": direction.astype(float).tolist(),
        "contact_footprint": {
            "center_xy_m": center.astype(float).tolist(),
            "size_xy_m": size.astype(float).tolist(),
            "corner_radius_m": float(contact["corner_radius_m"]),
            "edge_softness_m": float(contact["edge_softness_m"]),
            "opacity": float(contact["opacity"]),
            "brightness": float(contact["brightness"]),
        },
        "extension_mask": {
            "outline_vertex_count": int(len(outline_xyz)),
            "opacity": float(projection["extension"]["opacity"]),
            "brightness": float(projection["extension"]["brightness"]),
            "edge_softness_m": float(projection["extension"]["edge_softness_m"]),
        },
        "cast_outline_xyz": outline_xyz,
    }
