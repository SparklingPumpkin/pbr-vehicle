"""Optional gsplat image renderer for relit vehicle assets."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .asset import PBRAsset
from .config import RenderConfig
from .ply import activated_opacity, dc_rgb, gaussian_covariances, normalize, positions
from .scene import GaussianScene, compose_scene_properties
from .shading import relight


def _camera_matrices(config: RenderConfig) -> tuple[np.ndarray, np.ndarray]:
    eye = np.asarray(config.view.camera_position, dtype=np.float32)
    target = np.asarray(config.view.target, dtype=np.float32)
    up = normalize(np.asarray(config.view.up, dtype=np.float32)[None])[0]
    forward = normalize((target - eye)[None])[0]
    right = normalize(np.cross(forward, up)[None])[0]
    camera_up = np.cross(right, forward)
    rotation = np.stack([right, -camera_up, forward], axis=0)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    focal = 0.5 * config.height / math.tan(math.radians(config.view.vertical_fov_deg) * 0.5)
    intrinsic = np.array([
        [focal, 0.0, config.width * 0.5],
        [0.0, focal, config.height * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return view, intrinsic


def _render_properties(properties, output_png: str | Path, settings: RenderConfig) -> np.ndarray:
    try:
        import torch
        from gsplat.rendering import rasterization
    except ImportError as error:
        raise RuntimeError("Image rendering requires `pip install pbr-vehicle-sdk[render]`") from error

    device = torch.device(settings.device)
    view, intrinsic = _camera_matrices(settings)
    with torch.inference_mode():
        rgb, _, _ = rasterization(
            means=torch.from_numpy(positions(properties)).to(device),
            quats=None,
            scales=None,
            opacities=torch.from_numpy(activated_opacity(properties)).to(device),
            colors=torch.from_numpy(dc_rgb(properties)).to(device),
            viewmats=torch.from_numpy(view[None]).to(device),
            Ks=torch.from_numpy(intrinsic[None]).to(device),
            width=settings.width,
            height=settings.height,
            packed=False,
            render_mode="RGB",
            rasterize_mode="antialiased",
            backgrounds=torch.tensor(settings.background_rgb, dtype=torch.float32, device=device)[None],
            covars=torch.from_numpy(gaussian_covariances(properties)).to(device),
        )
    image = np.clip(rgb[0].detach().cpu().numpy(), 0.0, 1.0)
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(image * 255.0).astype(np.uint8), mode="RGB").save(output)
    return image


def render(
    asset: PBRAsset,
    output_png: str | Path,
    config: RenderConfig | None = None,
) -> np.ndarray:
    settings = config or RenderConfig()
    properties = asset.raw.copy()
    colors = relight(asset, settings).raw_rgb
    for channel in range(3):
        properties[f"f_dc_{channel}"] = ((colors[:, channel] - 0.5) / 0.28209479177387814).astype(np.float32)
    return _render_properties(properties, output_png, settings)


def render_scene(
    scene: GaussianScene,
    asset: PBRAsset,
    output_png: str | Path,
    config: RenderConfig | None = None,
) -> np.ndarray:
    settings = config or RenderConfig()
    combined, _, _ = compose_scene_properties(scene, asset, settings)
    return _render_properties(combined, output_png, settings)
