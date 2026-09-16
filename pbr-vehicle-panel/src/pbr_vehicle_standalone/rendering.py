from __future__ import annotations

import numpy as np


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


def projection_rgba_to_gaussians(
    rgba: np.ndarray,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    z: float,
    *,
    max_splats: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a receiver-space RGBA mask into a bounded planar Gaussian layer."""
    image = np.asarray(rgba, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Projection mask must be HxWx4 RGBA, got {image.shape}")
    height, width = image.shape[:2]
    alpha = image[..., 3]
    rows, columns = np.nonzero(alpha > 0)
    if len(rows) == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, np.empty((0, 3, 3), dtype=np.float32), empty.copy(), np.empty((0, 1), dtype=np.float32)

    stride = 1
    while np.count_nonzero((rows % stride == 0) & (columns % stride == 0)) > max_splats:
        stride += 1
    selected = (rows % stride == 0) & (columns % stride == 0)
    rows, columns = rows[selected], columns[selected]

    center = np.asarray(center_xy, dtype=np.float32)
    size = np.asarray(size_xy, dtype=np.float32)
    x = center[0] + ((columns.astype(np.float32) + 0.5) / width - 0.5) * size[0]
    y = center[1] + (0.5 - (rows.astype(np.float32) + 0.5) / height) * size[1]
    centers = np.column_stack([x, y, np.full(len(rows), float(z), dtype=np.float32)])

    sigma_x = max(float(size[0]) / width * stride * 0.7, 1e-4)
    sigma_y = max(float(size[1]) / height * stride * 0.7, 1e-4)
    sigma_z = min(sigma_x, sigma_y, 0.002)
    covariance = np.diag(np.square([sigma_x, sigma_y, sigma_z])).astype(np.float32)
    covariances = np.broadcast_to(covariance, (len(rows), 3, 3)).copy()
    colors = image[rows, columns, :3].astype(np.float32) / 255.0
    opacities = image[rows, columns, 3:4].astype(np.float32) / 255.0
    return tuple(np.ascontiguousarray(array, dtype=np.float32) for array in (
        centers, covariances, colors, opacities,
    ))
