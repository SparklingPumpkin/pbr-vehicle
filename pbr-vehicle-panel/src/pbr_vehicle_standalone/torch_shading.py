"""Torch CUDA implementation of the interactive vehicle shading hot path."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .asset_io import SH_C0
from .math3d import sun_direction


def _tensor(torch, value, device: str, cache: dict | None = None, label: str | None = None,
            dtype=None):
    array = np.ascontiguousarray(value)
    key = None
    if cache is not None and label is not None:
        key = (label, id(value), array.shape, array.dtype.str, device, str(dtype))
        cached = cache.get(key)
        if cached is not None:
            return cached
    result = torch.from_numpy(array).to(device=device, dtype=dtype)
    if key is not None:
        cache[key] = result
    return result


def _normalize(torch, values):
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-8)


def _saturation(torch, rgb, saturation: float):
    weights = rgb.new_tensor((0.2126, 0.7152, 0.0722))
    luminance = (rgb * weights).sum(dim=-1, keepdim=True)
    return (luminance + float(saturation) * (rgb - luminance)).clamp(0.0, 1.0)


def _sh_basis(torch, directions):
    x, y, z = _normalize(torch, directions).unbind(dim=1)
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


def _environment_diffuse(torch, environment_map, normals, device, cache):
    coefficients = _tensor(
        torch, environment_map.diffuse_sh, device, cache, f"env-{id(environment_map)}-diffuse"
    )
    return (_sh_basis(torch, normals) @ coefficients).clamp_min(0.0)


def _sample_environment(torch, environment_map, directions, roughness, device, cache):
    directions = _normalize(torch, directions.reshape(-1, 3))
    rotations = _tensor(
        torch, environment_map.rotations, device, cache, f"env-{id(environment_map)}-rotations"
    )
    face_indices = torch.argmax(directions @ rotations[:, :, 2].T, dim=1)
    roughness = torch.broadcast_to(roughness.reshape(-1), (len(directions),))
    lod = roughness.clamp(0.0, 1.0) * (len(environment_map.mips) - 1)
    lower = torch.floor(lod).to(torch.long)
    upper = torch.minimum(lower + 1, torch.full_like(lower, len(environment_map.mips) - 1))

    def sample_levels(levels):
        result = torch.empty((len(directions), 3), dtype=directions.dtype, device=device)
        for face_index in range(6):
            face_mask = face_indices == face_index
            if not bool(torch.any(face_mask)):
                continue
            local = directions[face_mask] @ rotations[face_index]
            local_xy = local[:, :2] / local[:, 2:3].clamp_min(1e-8)
            target_levels = levels[face_mask]
            target_indices = torch.nonzero(face_mask, as_tuple=False).reshape(-1)
            for level in range(len(environment_map.mips)):
                selected = target_levels == level
                if not bool(torch.any(selected)):
                    continue
                image = _tensor(
                    torch, environment_map.mips[level][face_index], device, cache,
                    f"env-{id(environment_map)}-mip-{level}-face-{face_index}",
                )
                height, width = image.shape[:2]
                uv = (local_xy[selected] * 0.5 + 0.5).clamp(0.0, 1.0)
                x, y = uv[:, 0] * (width - 1), uv[:, 1] * (height - 1)
                x0, y0 = torch.floor(x).long(), torch.floor(y).long()
                x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
                wx, wy = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)
                sampled = (
                    image[y0, x0] * (1.0 - wx) * (1.0 - wy)
                    + image[y0, x1] * wx * (1.0 - wy)
                    + image[y1, x0] * (1.0 - wx) * wy
                    + image[y1, x1] * wx * wy
                )
                result[target_indices[selected]] = sampled
        return result

    mix = (lod - lower).unsqueeze(1)
    return sample_levels(lower) * (1.0 - mix) + sample_levels(upper) * mix


def _shade_proxy_tensor(proxy, material, lighting, env_sh, environment_map, world_normals,
                        view_directions, device, cache):
    import torch

    albedo_source = _tensor(torch, proxy.albedo, device, cache, "proxy-albedo")
    albedo = _saturation(torch, albedo_source, material.saturation)
    normals = _normalize(torch, _tensor(torch, proxy.normals, device, cache, "proxy-normals"))
    count = len(albedo)
    if material.use_asset_material:
        roughness = (
            _tensor(torch, proxy.roughness, device, cache, "proxy-roughness")
            + (float(material.roughness) - 0.40)
        ).clamp(0.02, 0.98)
        metallic = (
            _tensor(torch, proxy.metallic, device, cache, "proxy-metallic")
            + float(material.metallic)
        ).clamp(0.0, 1.0)
    else:
        roughness = torch.full((count, 1), float(material.roughness), device=device)
        metallic = torch.full((count, 1), float(material.metallic), device=device)
    reflectance = torch.full((count, 1), float(material.reflectance), device=device)
    f0 = reflectance * (1.0 - metallic) + albedo * metallic

    if environment_map is not None:
        shading_normals = normals if world_normals is None else _normalize(
            torch, _tensor(torch, world_normals, device)
        )
        if view_directions is None:
            default_view = _normalize(torch, torch.tensor([[0.0, -1.0, 0.6]], device=device))[0]
            views = torch.broadcast_to(default_view, shading_normals.shape)
        else:
            views = _normalize(torch, _tensor(torch, view_directions, device))
        gain = torch.tensor(lighting.environment_rgb, dtype=torch.float32, device=device)[None]
        gain *= float(lighting.environment_intensity)
        fill = max(float(material.ambient_fill), 0.0)
        environment_diffuse = _environment_diffuse(
            torch, environment_map, shading_normals, device, cache
        ) * gain * fill
        reflected = (
            2.0 * (shading_normals * views).sum(dim=1, keepdim=True) * shading_normals - views
        )
        environment_specular = _sample_environment(
            torch, environment_map, reflected, roughness[:, 0], device, cache
        ) * gain * fill
        sun_light = torch.zeros_like(environment_diffuse)
        sun_specular = torch.zeros_like(environment_diffuse)
        if lighting.sun_enabled:
            direction = torch.tensor(
                sun_direction(lighting.sun_azimuth_deg, lighting.sun_elevation_deg),
                dtype=torch.float32, device=device,
            )
            light_dirs = torch.broadcast_to(direction[None], shading_normals.shape)
            ndotl = (shading_normals * light_dirs).sum(dim=1, keepdim=True).clamp_min(0.0)
            sun_light = torch.tensor(lighting.sun_rgb, dtype=torch.float32, device=device)[None]
            sun_light = sun_light * ndotl * max(float(lighting.sun_intensity), 0.0)
            sun_light *= min(max(float(lighting.visibility), 0.0), 1.0)
            half_dirs = _normalize(torch, light_dirs + views)
            specular_angle = (shading_normals * half_dirs).sum(dim=1, keepdim=True).clamp_min(0.0)
            gloss = torch.square(1.0 - roughness)
            broad = torch.pow(specular_angle, 1.0 + 3.0 * gloss)
            tight = torch.pow(specular_angle, 4.0 + 40.0 * gloss)
            sun_specular = sun_light * f0 * (0.45 + 14.0 * gloss) * (0.85 * broad + 2.5 * tight)
        ndotv = (shading_normals * views).sum(dim=1, keepdim=True).clamp_min(0.0)
        fresnel = f0 + (1.0 - f0) * torch.pow(1.0 - ndotv, 5.0)
        environment_specular *= fresnel * (1.0 - 0.45 * roughness)
        diffuse = albedo * (environment_diffuse + sun_light) * (1.0 - metallic)
        return ((diffuse + environment_specular + sun_specular) * float(material.exposure)).clamp(0.0, 1.0)

    if env_sh is None:
        env_sh = np.zeros((9, 3), dtype=np.float32)
        env_sh[0] = 1.0 / SH_C0
    coefficients = _tensor(torch, env_sh, device)
    coefficients = coefficients * float(lighting.environment_intensity)
    coefficients = coefficients * torch.tensor(
        lighting.environment_rgb, dtype=torch.float32, device=device
    )[None]
    view_dir = _normalize(torch, torch.tensor([[0.0, -1.0, 0.6]], device=device))[0]
    ambient = (_sh_basis(torch, normals) @ coefficients).clamp_min(0.0)
    ambient *= max(float(material.ambient_fill), 0.0)
    if lighting.sun_enabled:
        direction = torch.tensor(
            sun_direction(lighting.sun_azimuth_deg, lighting.sun_elevation_deg),
            dtype=torch.float32, device=device,
        )
        light_dirs = torch.broadcast_to(direction[None], normals.shape)
        ndotl = (normals * light_dirs).sum(dim=1, keepdim=True).clamp_min(0.0)
        direct = torch.tensor(lighting.sun_rgb, dtype=torch.float32, device=device)[None]
        direct = direct * ndotl * max(float(lighting.sun_intensity), 0.0)
        direct *= min(max(float(lighting.visibility), 0.0), 1.0)
        direct += ambient
        specular_angle = (
            normals * _normalize(torch, light_dirs + view_dir[None])
        ).sum(dim=1, keepdim=True).clamp_min(0.0)
    else:
        direct = ambient
        specular_angle = (normals * view_dir[None]).sum(dim=1, keepdim=True).clamp_min(0.0)
    diffuse = albedo * direct * (1.0 - metallic)
    gloss = torch.square(1.0 - roughness)
    broad = torch.pow(specular_angle, 1.0 + 3.0 * gloss)
    tight = torch.pow(specular_angle, 4.0 + 40.0 * gloss)
    specular = direct * f0 * (0.45 + 14.0 * gloss) * (0.85 * broad + 2.5 * tight)
    return ((diffuse + specular) * float(material.exposure)).clamp(0.0, 1.0)


def shade_proxy_cuda(proxy, material, lighting, env_sh=None, environment_map=None,
                     world_normals=None, view_directions=None, device="cuda:0", cache=None):
    import torch

    with torch.inference_mode():
        colors = _shade_proxy_tensor(
            proxy, material, lighting, env_sh, environment_map, world_normals,
            view_directions, device, cache,
        )
    return colors.detach().cpu().numpy().astype(np.float32, copy=False)


def shade_vehicle_cuda(asset, material, lighting, env_sh=None, mode="Relight Original",
                       environment_map=None, world_normals=None, view_directions=None,
                       use_proxy_relighting=True, device="cuda:0", cache=None):
    import torch

    proxy_material = material
    if mode == "Relight Original" and use_proxy_relighting:
        from dataclasses import replace
        proxy_material = replace(material, saturation=1.0)
    with torch.inference_mode():
        proxy_lit = _shade_proxy_tensor(
            asset.proxy, proxy_material, lighting, env_sh, environment_map,
            world_normals, view_directions, device, cache,
        )
        if mode == "Proxy Lit" or (mode == "Relight Original" and not use_proxy_relighting):
            return asset.proxy, proxy_lit.detach().cpu().numpy().astype(np.float32, copy=False)
        if mode != "Relight Original":
            raise ValueError(f"CUDA shading does not handle diagnostic mode {mode!r}")
        albedo = _tensor(torch, asset.proxy.albedo, device, cache, "proxy-albedo")
        ratio = proxy_lit / albedo.clamp(0.03, 1.0)
        indices = _tensor(
            torch, asset.mapping_indices, device, cache, "mapping-indices", dtype=torch.long
        )
        weights = _tensor(torch, asset.mapping_weights, device, cache, "mapping-weights")
        mapped = (ratio[indices] * weights[:, :, None]).sum(dim=1)
        mixed = 1.0 + float(material.relight_strength) * (mapped - 1.0)
        original = _tensor(torch, asset.original.colors, device, cache, "original-colors")
        original = _saturation(torch, original, material.saturation)
        colors = (original * mixed).clamp(0.0, 1.0)
    return asset.original, colors.detach().cpu().numpy().astype(np.float32, copy=False)
