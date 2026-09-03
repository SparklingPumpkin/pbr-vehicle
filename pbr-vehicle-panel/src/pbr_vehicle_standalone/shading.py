from __future__ import annotations

from dataclasses import replace

import numpy as np

from .asset_io import SH_C0
from .math3d import adjust_rgb_saturation, normalize, sun_direction
from .types import GaussianLayer, LightingState, MaterialState, VehicleAsset


def default_environment_sh() -> np.ndarray:
    result = np.zeros((9, 3), dtype=np.float32)
    result[0] = 1.0 / SH_C0
    return result


def evaluate_sh_basis(directions: np.ndarray) -> np.ndarray:
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


def shade_environment(directions: np.ndarray, env_sh: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(env_sh, dtype=np.float32)[:9]
    basis = evaluate_sh_basis(directions)
    return np.maximum(np.sum(basis[:, :, None] * coeffs[None, :, :], axis=1), 0.0).astype(np.float32)


def apply_environment_gain(env_sh: np.ndarray | None, lighting: LightingState) -> np.ndarray:
    base = default_environment_sh() if env_sh is None else np.asarray(env_sh, dtype=np.float32)
    gains = np.asarray(lighting.environment_rgb, dtype=np.float32)[None, :]
    return base * float(lighting.environment_intensity) * gains


def shade_proxy(proxy: GaussianLayer, material: MaterialState, lighting: LightingState,
                env_sh: np.ndarray | None = None) -> np.ndarray:
    if proxy.albedo is None or proxy.normals is None:
        raise ValueError("Proxy layer has no PBR albedo/normals")
    normals = normalize(proxy.normals)
    albedo = adjust_rgb_saturation(proxy.albedo, material.saturation)
    if material.use_asset_material:
        roughness = np.clip(proxy.roughness + (material.roughness - 0.40), 0.02, 0.98)
        metallic = np.clip(proxy.metallic + material.metallic, 0.0, 1.0)
    else:
        roughness = np.full((len(albedo), 1), material.roughness, dtype=np.float32)
        metallic = np.full((len(albedo), 1), material.metallic, dtype=np.float32)
    environment = apply_environment_gain(env_sh, lighting)
    view_dir = normalize(np.asarray([[0.0, -1.0, 0.6]], dtype=np.float32))[0]
    ambient_light = shade_environment(normals, environment) * max(float(material.ambient_fill), 0.0)
    if lighting.sun_enabled:
        light_dir = sun_direction(lighting.sun_azimuth_deg, lighting.sun_elevation_deg)
        light_dirs = np.broadcast_to(light_dir[None, :], normals.shape)
        ndotl = np.maximum(np.sum(normals * light_dirs, axis=1, keepdims=True), 0.0)
        direct = (
            np.asarray(lighting.sun_rgb, dtype=np.float32)[None, :]
            * ndotl * max(float(lighting.sun_intensity), 0.0)
            * np.clip(float(lighting.visibility), 0.0, 1.0)
            + ambient_light
        )
        specular_angle = np.maximum(
            np.sum(normals * normalize(light_dirs + view_dir[None, :]), axis=1, keepdims=True), 0.0
        )
    else:
        light_dirs = normalize(normals + view_dir[None, :])
        direct = ambient_light
        specular_angle = np.maximum(np.sum(normals * view_dir[None, :], axis=1, keepdims=True), 0.0)
    roughness = np.clip(roughness, 0.02, 0.98)
    metallic = np.clip(metallic, 0.0, 1.0)
    reflectance = np.full((len(albedo), 1), material.reflectance, dtype=np.float32)
    f0 = reflectance * (1.0 - metallic) + albedo * metallic
    diffuse = albedo * direct * (1.0 - metallic)
    gloss = np.square(1.0 - roughness)
    broad = np.power(specular_angle, 1.0 + 3.0 * gloss)
    tight = np.power(specular_angle, 4.0 + 40.0 * gloss)
    specular = direct * f0 * (0.45 + 14.0 * gloss) * (0.85 * broad + 2.5 * tight)
    return np.clip((diffuse + specular) * material.exposure, 0.0, 1.0).astype(np.float32)


def shade_vehicle(asset: VehicleAsset, material: MaterialState, lighting: LightingState,
                  env_sh: np.ndarray | None = None, mode: str = "Relight Original") -> tuple[GaussianLayer, np.ndarray]:
    proxy_material = replace(material, saturation=1.0) if mode == "Relight Original" else material
    proxy_lit = shade_proxy(asset.proxy, proxy_material, lighting, env_sh)
    if mode == "Proxy Lit":
        return asset.proxy, proxy_lit
    if mode == "Original SH":
        return asset.original, asset.original.colors
    if mode == "Albedo":
        return asset.proxy, adjust_rgb_saturation(asset.proxy.albedo, material.saturation)
    if mode == "Normal":
        return asset.proxy, np.clip(asset.proxy.normals * 0.5 + 0.5, 0.0, 1.0)
    if mode == "Roughness":
        return asset.proxy, np.repeat(asset.proxy.roughness, 3, axis=1)
    if mode == "Reflectance":
        values = np.full((len(asset.proxy.centers), 3), material.reflectance, dtype=np.float32)
        return asset.proxy, values
    if mode == "Metallic":
        return asset.proxy, np.repeat(asset.proxy.metallic, 3, axis=1)
    if mode == "SH RGB":
        return asset.proxy, asset.proxy.colors
    ratio = proxy_lit / np.clip(asset.proxy.albedo, 0.03, 1.0)
    mapped = np.sum(ratio[asset.mapping_indices] * asset.mapping_weights[:, :, None], axis=1)
    mixed = 1.0 + material.relight_strength * (mapped - 1.0)
    original_albedo = adjust_rgb_saturation(asset.original.colors, material.saturation)
    return asset.original, np.clip(original_albedo * mixed, 0.0, 1.0).astype(np.float32)
