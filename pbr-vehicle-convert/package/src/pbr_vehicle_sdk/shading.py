"""View-dependent GGX car-paint shading and KNN8 transfer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .asset import PBRAsset
from .config import RenderConfig
from .ply import SH_C0, dc_rgb, normalize, positions, write_ply


def _indexed(properties, prefix: str, count: int) -> np.ndarray:
    return np.stack([properties[f"{prefix}_{index}"] for index in range(count)], axis=1).astype(np.float32)


def _fresnel_schlick(cos_theta: np.ndarray, f0: np.ndarray) -> np.ndarray:
    return f0 + (1.0 - f0) * np.power(1.0 - np.clip(cos_theta, 0.0, 1.0), 5.0)


def _aces_film(rgb: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = rgb * (a * rgb + b) / np.clip(rgb * (c * rgb + d) + e, 1e-8, None)
    return np.clip(mapped, 0.0, 1.0)


def _sh_basis(directions: np.ndarray) -> np.ndarray:
    x, y, z = normalize(directions).T
    return np.stack([
        np.ones_like(x) * SH_C0,
        -0.4886025119029199 * y,
        0.4886025119029199 * z,
        -0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        -1.0925484305920792 * y * z,
        0.31539156525252005 * (3.0 * z * z - 1.0),
        -1.0925484305920792 * x * z,
        0.5462742152960396 * (x * x - y * y),
    ], axis=1).astype(np.float32)


def _environment(directions: np.ndarray, config: RenderConfig) -> np.ndarray:
    if not config.integration.lighting_color:
        if config.light.environment_sh is None:
            intensity = float(np.mean(config.light.environment_color))
        else:
            coefficients = np.asarray(config.light.environment_sh, dtype=np.float32)
            intensity = float(np.mean(np.maximum(coefficients[0], 0.0)))
        return np.full(directions.shape, intensity, dtype=np.float32)
    environment_sh = config.light.environment_sh
    if environment_sh is None:
        return np.broadcast_to(np.asarray(config.light.environment_color, dtype=np.float32), directions.shape).copy()
    coefficients = np.asarray(environment_sh, dtype=np.float32)
    if len(coefficients) == 1:
        return np.broadcast_to(np.maximum(coefficients[0], 0.0), directions.shape).copy()
    return np.maximum(np.sum(_sh_basis(directions)[:, :, None] * coefficients[None], axis=1), 0.0)


def _sun_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return normalize(np.array([[
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ]], dtype=np.float32))[0]


def _lighting_color_gain(config: RenderConfig) -> np.ndarray:
    if not config.integration.lighting_color:
        return np.ones((1, 3), dtype=np.float32)
    if config.light.environment_sh is None:
        color = np.asarray(config.light.environment_color, dtype=np.float32)
    else:
        color = np.maximum(np.asarray(config.light.environment_sh, dtype=np.float32)[0], 0.0)
    return (color / np.clip(float(np.mean(color)), 1e-4, None))[None]


def _ggx(normals, view_dirs, light_dirs, roughness, f0):
    half_dirs = normalize(view_dirs + light_dirs)
    ndotv = np.clip(np.sum(normals * view_dirs, axis=1, keepdims=True), 1e-4, 1.0)
    ndotl = np.clip(np.sum(normals * light_dirs, axis=1, keepdims=True), 0.0, 1.0)
    ndoth = np.clip(np.sum(normals * half_dirs, axis=1, keepdims=True), 0.0, 1.0)
    vdoth = np.clip(np.sum(view_dirs * half_dirs, axis=1, keepdims=True), 0.0, 1.0)
    alpha2 = np.power(np.clip(roughness, 0.025, 1.0), 4.0)
    distribution = alpha2 / np.clip(math.pi * np.square(ndoth * ndoth * (alpha2 - 1.0) + 1.0), 1e-6, None)
    k = np.square(roughness + 1.0) / 8.0
    geometry = ndotv / np.clip(ndotv * (1.0 - k) + k, 1e-6, None)
    geometry *= ndotl / np.clip(ndotl * (1.0 - k) + k, 1e-6, None)
    brdf = distribution * geometry * _fresnel_schlick(vdoth, f0)
    return brdf * ndotl / np.clip(4.0 * ndotv * np.maximum(ndotl, 1e-4), 1e-5, None)


@dataclass(frozen=True)
class RelightResult:
    raw_rgb: np.ndarray
    proxy_rgb: np.ndarray
    raw_relight_ratio: np.ndarray


def relight(asset: PBRAsset, config: RenderConfig | None = None) -> RelightResult:
    settings = config or RenderConfig()
    original_rgb = dc_rgb(asset.raw)
    if not settings.integration.pbr_properties:
        raw_rgb = np.clip(original_rgb * _lighting_color_gain(settings) * settings.exposure, 0.0, 1.0).astype(np.float32)
        raw_ratio = raw_rgb / np.clip(original_rgb, 0.03, 1.0)
        return RelightResult(raw_rgb, original_rgb.copy(), raw_ratio.astype(np.float32))
    proxy_points = positions(asset.proxy)
    albedo = _indexed(asset.proxy, "pbr_albedo", 3)
    normals = normalize(_indexed(asset.proxy, "pbr_normal", 3))
    view_dirs = normalize(np.asarray(settings.view.camera_position, dtype=np.float32)[None] - proxy_points)
    count = len(proxy_points)
    roughness = np.full((count, 1), settings.material.roughness, dtype=np.float32)
    metallic = np.full((count, 1), settings.material.metallic, dtype=np.float32)
    f0 = settings.material.reflectance * (1.0 - metallic) + albedo * metallic
    ndotv = np.clip(np.sum(normals * view_dirs, axis=1, keepdims=True), 1e-4, 1.0)
    ambient = _environment(normals, settings) * settings.light.ambient_fill
    diffuse = albedo * ambient * (1.0 - metallic)
    reflected = normalize(2.0 * ndotv * normals - view_dirs)
    reflected_light = _environment(reflected, settings)
    specular = reflected_light * _fresnel_schlick(ndotv, f0)
    specular *= np.square(1.0 - 0.72 * roughness) * settings.light.environment_reflection

    coat_amount = settings.material.clearcoat
    coat_roughness = np.full((count, 1), settings.material.clearcoat_roughness, dtype=np.float32)
    coat_f0 = np.full_like(albedo, 0.04)
    if coat_amount > 0.0:
        coat_environment = reflected_light * _fresnel_schlick(ndotv, coat_f0)
        coat_environment *= np.square(1.0 - 0.65 * coat_roughness)
        specular = specular * (1.0 - 0.25 * coat_amount)
        specular += coat_amount * coat_environment * settings.light.environment_reflection

    if settings.light.sun_enabled:
        light_dirs = np.broadcast_to(
            _sun_direction(settings.light.sun_azimuth_deg, settings.light.sun_elevation_deg)[None], normals.shape
        )
        sun_radiance = _environment(light_dirs, settings) * settings.light.sun_intensity
        ndotl = np.clip(np.sum(normals * light_dirs, axis=1, keepdims=True), 0.0, 1.0)
        diffuse += albedo * sun_radiance * ndotl * (1.0 - metallic)
        specular += sun_radiance * _ggx(normals, view_dirs, light_dirs, roughness, f0)
        if coat_amount > 0.0:
            specular += coat_amount * sun_radiance * _ggx(normals, view_dirs, light_dirs, coat_roughness, coat_f0)

    proxy_rgb = _aces_film(np.maximum((diffuse + specular * settings.material.specular_gain) * settings.exposure, 0.0))
    proxy_ratio = proxy_rgb / np.clip(albedo, 0.03, 1.0)
    mapped_ratio = np.sum(
        proxy_ratio[asset.mapping_indices] * asset.mapping_weights[..., None], axis=1
    )
    raw_ratio = 1.0 + settings.relight_strength * (mapped_ratio - 1.0)
    raw_rgb = np.clip(original_rgb * raw_ratio, 0.0, 1.0).astype(np.float32)
    return RelightResult(raw_rgb, proxy_rgb.astype(np.float32), raw_ratio.astype(np.float32))


def bake(
    asset: PBRAsset,
    output_ply: str,
    config: RenderConfig | None = None,
    manifest_path: str | None = None,
) -> RelightResult:
    settings = config or RenderConfig()
    result = relight(asset, settings)
    baked = asset.raw.copy()
    for channel in range(3):
        baked[f"f_dc_{channel}"] = ((result.raw_rgb[:, channel] - 0.5) / SH_C0).astype(np.float32)
    for name in baked:
        if name.startswith("f_rest_"):
            baked[name] = np.zeros_like(baked[name], dtype=np.float32)
    output = write_ply(output_ply, baked)
    metadata = {
        "schema": "pbr-vehicle-bake/1.0",
        "output_ply": output.name,
        "source_asset": asset.root.name,
        "view_dependent": True,
        "higher_order_sh": "zeroed",
        "render_config": settings.to_dict(),
    }
    target_manifest = output.with_suffix(output.suffix + ".bake.json") if manifest_path is None else output.parent / manifest_path
    target_manifest.write_text(__import__("json").dumps(metadata, indent=2), encoding="utf-8")
    return result
