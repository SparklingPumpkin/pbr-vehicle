"""Torch implementation of GGX proxy shading and mapping transfer."""

from __future__ import annotations

import math

import numpy as np

from .ply import SH_C0, dc_rgb, positions


def _indexed(properties, prefix: str, count: int) -> np.ndarray:
    return np.stack([properties[f"{prefix}_{index}"] for index in range(count)], axis=1).astype(np.float32)


def relight_cuda(asset, settings, device: str):
    import torch

    def tensor(value, dtype=None):
        return torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=dtype)

    def normalize(value):
        return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-8)

    def fresnel(cosine, f0):
        return f0 + (1.0 - f0) * torch.pow(1.0 - cosine.clamp(0.0, 1.0), 5.0)

    def sh_basis(directions):
        x, y, z = normalize(directions).unbind(dim=1)
        return torch.stack((
            torch.ones_like(x) * SH_C0,
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z * z - 1.0),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        ), dim=1)

    def environment(directions):
        if not settings.integration.lighting_color:
            if settings.light.environment_sh is None:
                intensity = float(np.mean(settings.light.environment_color))
            else:
                coefficients = np.asarray(settings.light.environment_sh, dtype=np.float32)
                intensity = float(np.mean(np.maximum(coefficients[0], 0.0)))
            return torch.full_like(directions, intensity)
        if settings.light.environment_sh is None:
            color = torch.tensor(settings.light.environment_color, dtype=torch.float32, device=device)
            return torch.broadcast_to(color[None], directions.shape)
        coefficients = torch.tensor(settings.light.environment_sh, dtype=torch.float32, device=device)
        if len(coefficients) == 1:
            return torch.broadcast_to(coefficients[0].clamp_min(0.0), directions.shape)
        return torch.sum(sh_basis(directions)[:, :, None] * coefficients[None], dim=1).clamp_min(0.0)

    def ggx(normals, views, light_dirs, roughness, f0):
        half_dirs = normalize(views + light_dirs)
        ndotv = (normals * views).sum(dim=1, keepdim=True).clamp(1e-4, 1.0)
        ndotl = (normals * light_dirs).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        ndoth = (normals * half_dirs).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        vdoth = (views * half_dirs).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        alpha2 = torch.pow(roughness.clamp(0.025, 1.0), 4.0)
        distribution = alpha2 / (
            math.pi * torch.square(ndoth * ndoth * (alpha2 - 1.0) + 1.0)
        ).clamp_min(1e-6)
        k = torch.square(roughness + 1.0) / 8.0
        geometry = ndotv / (ndotv * (1.0 - k) + k).clamp_min(1e-6)
        geometry *= ndotl / (ndotl * (1.0 - k) + k).clamp_min(1e-6)
        return distribution * geometry * fresnel(vdoth, f0) * ndotl / (
            4.0 * ndotv * ndotl.clamp_min(1e-4)
        ).clamp_min(1e-5)

    def aces(rgb):
        return (rgb * (2.51 * rgb + 0.03) / (rgb * (2.43 * rgb + 0.59) + 0.14).clamp_min(1e-8)).clamp(0.0, 1.0)

    with torch.inference_mode():
        original_rgb = tensor(dc_rgb(asset.raw))
        if not settings.integration.pbr_properties:
            if not settings.integration.lighting_color:
                gain = torch.ones((1, 3), dtype=torch.float32, device=device)
            else:
                color = (
                    np.asarray(settings.light.environment_color, dtype=np.float32)
                    if settings.light.environment_sh is None
                    else np.maximum(np.asarray(settings.light.environment_sh, dtype=np.float32)[0], 0.0)
                )
                gain = tensor((color / np.clip(float(np.mean(color)), 1e-4, None))[None])
            raw_rgb = (original_rgb * gain * float(settings.exposure)).clamp(0.0, 1.0)
            raw_ratio = raw_rgb / original_rgb.clamp(0.03, 1.0)
            return tuple(
                value.detach().cpu().numpy().astype(np.float32, copy=False)
                for value in (raw_rgb, original_rgb, raw_ratio)
            )

        proxy_points = tensor(positions(asset.proxy))
        albedo = tensor(_indexed(asset.proxy, "pbr_albedo", 3))
        normals = normalize(tensor(_indexed(asset.proxy, "pbr_normal", 3)))
        camera = torch.tensor(settings.view.camera_position, dtype=torch.float32, device=device)
        views = normalize(camera[None] - proxy_points)
        count = len(proxy_points)
        roughness = torch.full((count, 1), float(settings.material.roughness), device=device)
        metallic = torch.full((count, 1), float(settings.material.metallic), device=device)
        f0 = float(settings.material.reflectance) * (1.0 - metallic) + albedo * metallic
        ndotv = (normals * views).sum(dim=1, keepdim=True).clamp(1e-4, 1.0)
        ambient = environment(normals) * float(settings.light.ambient_fill)
        diffuse = albedo * ambient * (1.0 - metallic)
        reflected = normalize(2.0 * ndotv * normals - views)
        reflected_light = environment(reflected)
        specular = reflected_light * fresnel(ndotv, f0)
        specular *= torch.square(1.0 - 0.72 * roughness) * float(
            settings.light.environment_reflection
        )
        coat = float(settings.material.clearcoat)
        coat_roughness = torch.full(
            (count, 1), float(settings.material.clearcoat_roughness), device=device
        )
        coat_f0 = torch.full_like(albedo, 0.04)
        if coat > 0.0:
            coat_environment = reflected_light * fresnel(ndotv, coat_f0)
            coat_environment *= torch.square(1.0 - 0.65 * coat_roughness)
            specular = specular * (1.0 - 0.25 * coat)
            specular += coat * coat_environment * float(settings.light.environment_reflection)
        if settings.light.sun_enabled:
            azimuth = math.radians(settings.light.sun_azimuth_deg)
            elevation = math.radians(settings.light.sun_elevation_deg)
            direction = torch.tensor((
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ), dtype=torch.float32, device=device)
            light_dirs = torch.broadcast_to(normalize(direction[None]), normals.shape)
            radiance = environment(light_dirs) * float(settings.light.sun_intensity)
            ndotl = (normals * light_dirs).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            diffuse += albedo * radiance * ndotl * (1.0 - metallic)
            specular += radiance * ggx(normals, views, light_dirs, roughness, f0)
            if coat > 0.0:
                specular += coat * radiance * ggx(
                    normals, views, light_dirs, coat_roughness, coat_f0
                )
        proxy_rgb = aces(
            ((diffuse + specular * float(settings.material.specular_gain))
             * float(settings.exposure)).clamp_min(0.0)
        )
        proxy_ratio = proxy_rgb / albedo.clamp(0.03, 1.0)
        indices = tensor(asset.mapping_indices, dtype=torch.long)
        weights = tensor(asset.mapping_weights)
        mapped = (proxy_ratio[indices] * weights[..., None]).sum(dim=1)
        raw_ratio = 1.0 + float(settings.relight_strength) * (mapped - 1.0)
        raw_rgb = (original_rgb * raw_ratio).clamp(0.0, 1.0)
        return tuple(
            value.detach().cpu().numpy().astype(np.float32, copy=False)
            for value in (raw_rgb, proxy_rgb, raw_ratio)
        )
