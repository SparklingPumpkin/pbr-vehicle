from __future__ import annotations

import io

import numpy as np
import trimesh
from PIL import Image


def pack_gaussian_buffer(centers: np.ndarray, covariances: np.ndarray,
                         colors: np.ndarray, opacities: np.ndarray) -> np.ndarray:
    count = len(centers)
    upper = covariances.reshape((-1, 9))[:, [0, 1, 2, 4, 5, 8]]
    result = np.concatenate([
        centers.astype(np.float32, copy=False).view(np.uint8),
        np.zeros((count, 4), dtype=np.uint8),
        upper.astype(np.float16).copy().view(np.uint8),
        np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8),
        np.clip(opacities * 255.0, 0.0, 255.0).astype(np.uint8),
    ], axis=-1).view(np.uint32)
    if result.shape != (count, 8):
        raise ValueError(f"Invalid Gaussian buffer shape: {result.shape}")
    return np.ascontiguousarray(result)


def rgba_plane_glb(rgba: np.ndarray, width: float, height: float) -> bytes:
    """Build a glTF BLEND plane; GLTFLoader disables depthWrite for this material."""
    vertices = np.asarray([
        [-width * 0.5, -height * 0.5, 0.0],
        [width * 0.5, -height * 0.5, 0.0],
        [width * 0.5, height * 0.5, 0.0],
        [-width * 0.5, height * 0.5, 0.0],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv = np.asarray([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.asarray(rgba, dtype=np.uint8)),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode="BLEND",
        doubleSided=True,
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh))
