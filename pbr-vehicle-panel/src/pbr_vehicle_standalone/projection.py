from __future__ import annotations

import copy
import math
from typing import Any

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .math3d import normalize
from .types import GaussianLayer


def validate_projection(config: dict[str, Any]) -> None:
    if config.get("runtime") != "receiver_space_mask_v1":
        raise ValueError("projection.runtime must be receiver_space_mask_v1")
    if config.get("shape_model") != "procedural-mask-contact-silhouette-v4":
        raise ValueError("Unsupported projection shape_model")
    if config.get("composition") != "receiver_alpha_blend_to_grayscale_v1":
        raise ValueError("Unsupported projection composition")
    for section in ("contact", "extension", "anchor"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"projection.{section} must be an object")


def _rounded_rectangle(half_extents: np.ndarray, radius_m: float, samples: int = 24) -> np.ndarray:
    half_length, half_width = map(float, half_extents)
    radius = min(max(float(radius_m), 0.0), half_length, half_width)
    corners = (
        (half_length - radius, half_width - radius, 0.0),
        (-half_length + radius, half_width - radius, 90.0),
        (-half_length + radius, -half_width + radius, 180.0),
        (half_length - radius, -half_width + radius, 270.0),
    )
    parts = []
    for cx, cy, start in corners:
        angles = np.deg2rad(np.linspace(start, start + 90.0, samples, endpoint=False))
        parts.append(np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)]))
    return np.concatenate(parts).astype(np.float32)


def _signed_distance(xy: np.ndarray, center: np.ndarray, half_extents: np.ndarray, radius_m: float) -> np.ndarray:
    radius = min(max(float(radius_m), 0.0), float(np.min(half_extents)))
    q = np.abs(xy - center) - half_extents + radius
    return (np.linalg.norm(np.maximum(q, 0.0), axis=-1)
            + np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0) - radius).astype(np.float32)


def _outline(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float32), axis=0)
    if len(unique) < 3:
        return unique
    try:
        return unique[ConvexHull(unique).vertices]
    except QhullError:
        center = unique.mean(axis=0)
        return unique[np.argsort(np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0]))]


def _geometry(points: np.ndarray, padding: float):
    low = points.min(axis=0) - padding
    high = points.max(axis=0) + padding
    extent = np.maximum(high - low, 1e-3)
    ppm = min(192.0, 1024.0 / float(np.max(extent)))
    width = max(64, int(math.ceil(float(extent[0]) * ppm)))
    height = max(64, int(math.ceil(float(extent[1]) * ppm)))
    actual = np.asarray([width, height], dtype=np.float32) / ppm
    return low, low + actual, actual, ppm, (height, width)


def _polygon_mask(polygon: np.ndarray, low: np.ndarray, ppm: float, shape: tuple[int, int], sigma: float) -> np.ndarray:
    pixels = (polygon - low[None, :]) * ppm
    pixels[:, 1] = shape[0] - 1 - pixels[:, 1]
    mask = np.zeros(shape, dtype=np.float32)
    if len(pixels) >= 3:
        cv2.fillPoly(mask, [np.round(pixels).astype(np.int32).reshape(-1, 1, 2)], 1.0, lineType=cv2.LINE_AA)
        if sigma * ppm > 1e-3:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma * ppm, sigmaY=sigma * ppm, borderType=cv2.BORDER_CONSTANT)
    return np.clip(mask, 0.0, 1.0)


def _rgba(brightness: float | np.ndarray, alpha: np.ndarray) -> np.ndarray:
    image = np.empty((*alpha.shape, 4), dtype=np.uint8)
    gray = np.clip(np.asarray(brightness) * 255.0, 0.0, 255.0).astype(np.uint8)
    image[..., :3] = gray[..., None] if gray.ndim == 2 else gray
    image[..., 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    return image


def build_projection_masks(proxy: GaussianLayer, local_sun_direction: np.ndarray,
                           config: dict[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(config)
    validate_projection(projection)
    direction = normalize(np.asarray(local_sun_direction, dtype=np.float32).reshape(1, 3))[0]
    if direction[2] < math.sin(math.radians(1.0)):
        raise ValueError("Projection requires sun elevation of at least 1 degree")
    points = proxy.centers
    anchor = projection["anchor"]
    percentile = float(anchor.get("bottom_surface_percentile", 1.0))
    sigma = float(anchor.get("surface_sigma", 1.0))
    fixed_anchor = projection.get("_fixed_anchor", {})
    fixed_z = projection.get("_fixed_ground_z")
    if fixed_z is not None and percentile == float(fixed_anchor.get("bottom_surface_percentile", percentile)) and sigma == float(fixed_anchor.get("surface_sigma", sigma)):
        ground_z = float(fixed_z) + float(anchor.get("z_offset_m", 0.0)) - float(fixed_anchor.get("z_offset_m", 0.0))
    else:
        bottom = points[:, 2] - sigma * np.sqrt(np.clip(proxy.covariances[:, 2, 2], 0.0, None))
        ground_z = float(np.percentile(bottom, percentile)) + float(anchor.get("z_offset_m", 0.0))
    contact, extension = projection["contact"], projection["extension"]
    low, high = np.quantile(points[:, :2], [0.01, 0.99], axis=0)
    full_size = np.maximum(high - low, 1e-4)
    center = (low + high) * 0.5 + np.asarray(contact["offset_xy_m"], dtype=np.float32)
    half = np.maximum(full_size * np.asarray([contact["length_scale"], contact["width_scale"]]) * 0.5, 0.05)
    contact_polygon = center + _rounded_rectangle(half, contact["corner_radius_m"])

    contact_padding = max(float(contact["edge_softness_m"]), 0.05) * 4.0
    clow, chigh, csize, cppm, cshape = _geometry(contact_polygon, contact_padding)
    contact_mask = _polygon_mask(contact_polygon, clow, cppm, cshape, float(contact["edge_softness_m"]))

    heights = np.maximum(points[:, 2] - ground_z, 0.0)
    cast = points - heights[:, None] * direction[None, :] / float(direction[2])
    cast_outline = _outline(cast[:, :2])
    padding = max(float(contact["edge_softness_m"]), float(extension["edge_softness_m"]) + float(extension["edge_softness_distance_growth_m"]), 0.05) * 4.0
    low2, high2, size2, ppm2, shape2 = _geometry(np.concatenate([contact_polygon, cast_outline]), padding)
    base = _polygon_mask(cast_outline, low2, ppm2, shape2, float(extension["edge_softness_m"]))
    far = _polygon_mask(cast_outline, low2, ppm2, shape2, float(extension["edge_softness_m"]) + float(extension["edge_softness_distance_growth_m"]))
    yy, xx = np.meshgrid(
        high2[1] - (np.arange(shape2[0], dtype=np.float32) + 0.5) / ppm2,
        low2[0] + (np.arange(shape2[1], dtype=np.float32) + 0.5) / ppm2,
        indexing="ij",
    )
    distance = np.maximum(_signed_distance(np.stack([xx, yy], axis=-1), center, half, contact["corner_radius_m"]), 0.0)
    normalized = np.clip(distance / max(float(extension["distance_scale_m"]), 1e-5), 0.0, 1.0)
    blur_mix = np.power(normalized, float(extension["edge_softness_distance_exponent"]))
    extension_mask = base * (1.0 - blur_mix) + far * blur_mix
    falloff = np.clip(1.0 - float(extension["opacity_distance_decay"]) * np.power(normalized, float(extension["opacity_distance_exponent"])), 0.0, 1.0)
    brightness = float(extension["brightness"]) + (1.0 - float(extension["brightness"])) * float(extension["brightness_distance_to_white"]) * np.power(normalized, float(extension["brightness_distance_exponent"]))
    return {
        "contact": {
            "rgba": _rgba(float(contact["brightness"]), contact_mask * float(contact["opacity"])),
            "center_xy": ((clow + chigh) * 0.5).astype(np.float32),
            "size_xy": csize,
        },
        "extension": {
            "rgba": _rgba(brightness, extension_mask * float(extension["opacity"]) * falloff),
            "center_xy": ((low2 + high2) * 0.5).astype(np.float32),
            "size_xy": size2,
        },
        "ground_z": ground_z,
        "cast_outline_xyz": np.column_stack([cast_outline, np.full(len(cast_outline), ground_z)]).astype(np.float32),
    }

