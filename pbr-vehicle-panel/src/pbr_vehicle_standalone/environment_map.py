"""Small CPU-side cubemap and image-based-lighting helpers for Viser previews."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SH_C0 = 0.28209479177387814


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def _look_rotation(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _normalize(np.asarray(forward, dtype=np.float32)[None])[0]
    right = _normalize(np.cross(forward, np.asarray(up, dtype=np.float32))[None])[0]
    down = _normalize(np.cross(forward, right)[None])[0]
    return np.stack([right, down, forward], axis=1).astype(np.float32)


def cube_face_rotations() -> np.ndarray:
    specifications = (
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    )
    return np.stack([_look_rotation(forward, up) for forward, up in specifications])


def evaluate_sh_basis(directions: np.ndarray) -> np.ndarray:
    x, y, z = _normalize(directions).T
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


def _float_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Cubemap face must have shape HxWx3/4, got {image.shape}")
    result = image[..., :3].astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        result /= float(np.iinfo(image.dtype).max)
    return np.clip(result, 0.0, None)


def _downsample_2x(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height == 1 and width == 1:
        return image
    if height == 1:
        width -= width % 2
        return 0.5 * (image[:, 0:width:2] + image[:, 1:width:2])
    if width == 1:
        height -= height % 2
        return 0.5 * (image[0:height:2] + image[1:height:2])
    height -= height % 2
    width -= width % 2
    return 0.25 * (
        image[0:height:2, 0:width:2] + image[1:height:2, 0:width:2]
        + image[0:height:2, 1:width:2] + image[1:height:2, 1:width:2]
    )


def _project_diffuse_sh(faces: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    resolution = faces.shape[1]
    coordinates = (2.0 * (np.arange(resolution, dtype=np.float32) + 0.5) / resolution) - 1.0
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    local = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    solid_angle = (2.0 / resolution) ** 2 / np.power(1.0 + grid_x**2 + grid_y**2, 1.5)
    coefficients = np.zeros((9, 3), dtype=np.float64)
    for face, rotation in zip(faces, rotations):
        directions = _normalize(local.reshape(-1, 3) @ rotation.T)
        basis = evaluate_sh_basis(directions).astype(np.float64)
        weights = solid_angle.reshape(-1, 1).astype(np.float64)
        coefficients += basis.T @ (face.reshape(-1, 3).astype(np.float64) * weights)
    coefficients *= np.asarray([1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
                                0.25, 0.25, 0.25, 0.25, 0.25])[:, None]
    return coefficients.astype(np.float32)


@dataclass(frozen=True)
class EnvironmentMap:
    faces: np.ndarray
    rotations: np.ndarray
    diffuse_sh: np.ndarray
    mips: tuple[np.ndarray, ...]

    @classmethod
    def from_faces(cls, face_images, rotations: np.ndarray | None = None) -> "EnvironmentMap":
        faces = np.stack([_float_rgb(face) for face in face_images]).astype(np.float32)
        if faces.shape[0] != 6 or faces.shape[1] != faces.shape[2]:
            raise ValueError(f"Cubemap requires six square faces, got {faces.shape}")
        rotations = cube_face_rotations() if rotations is None else np.asarray(rotations, dtype=np.float32)
        if rotations.shape != (6, 3, 3):
            raise ValueError(f"Cubemap rotations must have shape (6,3,3), got {rotations.shape}")
        levels = [faces]
        while levels[-1].shape[1] > 1 or levels[-1].shape[2] > 1:
            levels.append(np.stack([_downsample_2x(face) for face in levels[-1]]).astype(np.float32))
        return cls(faces, rotations, _project_diffuse_sh(faces, rotations), tuple(levels))

    def diffuse(self, normals: np.ndarray) -> np.ndarray:
        return np.maximum(evaluate_sh_basis(normals) @ self.diffuse_sh, 0.0).astype(np.float32)

    def sample(self, directions: np.ndarray, roughness: np.ndarray | float = 0.0) -> np.ndarray:
        directions = _normalize(np.asarray(directions, dtype=np.float32).reshape(-1, 3))
        roughness = np.broadcast_to(np.asarray(roughness, dtype=np.float32).reshape(-1), (len(directions),))
        lod = np.clip(roughness, 0.0, 1.0) * (len(self.mips) - 1)
        lower = np.floor(lod).astype(np.int32)
        upper = np.minimum(lower + 1, len(self.mips) - 1)
        mix = (lod - lower).reshape(-1, 1)
        return self._sample_levels(directions, lower) * (1.0 - mix) + self._sample_levels(directions, upper) * mix

    def _sample_levels(self, directions: np.ndarray, levels: np.ndarray) -> np.ndarray:
        face_indices = np.argmax(directions @ self.rotations[:, :, 2].T, axis=1)
        result = np.empty((len(directions), 3), dtype=np.float32)
        for face_index in range(6):
            face_mask = face_indices == face_index
            if not np.any(face_mask):
                continue
            local = directions[face_mask] @ self.rotations[face_index]
            local_xy = local[:, :2] / np.maximum(local[:, 2:3], 1e-8)
            target_levels = levels[face_mask]
            target_indices = np.flatnonzero(face_mask)
            for level in np.unique(target_levels):
                selected = target_levels == level
                image = self.mips[int(level)][face_index]
                height, width = image.shape[:2]
                uv = np.clip(local_xy[selected] * 0.5 + 0.5, 0.0, 1.0)
                x, y = uv[:, 0] * (width - 1), uv[:, 1] * (height - 1)
                x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
                x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
                wx, wy = (x - x0)[:, None], (y - y0)[:, None]
                sampled = (
                    image[y0, x0] * (1.0 - wx) * (1.0 - wy)
                    + image[y0, x1] * wx * (1.0 - wy)
                    + image[y1, x0] * (1.0 - wx) * wy
                    + image[y1, x1] * wx * wy
                )
                result[target_indices[selected]] = sampled
        return result

    def reflection(self, normals: np.ndarray, view_directions: np.ndarray,
                   roughness: np.ndarray | float) -> np.ndarray:
        normals = _normalize(normals)
        view_directions = _normalize(view_directions)
        reflected = 2.0 * np.sum(normals * view_directions, axis=1, keepdims=True) * normals - view_directions
        return self.sample(reflected, roughness)

    def to_equirectangular(self, height: int | None = None, width: int | None = None) -> np.ndarray:
        """Resample the cubemap to an RGB equirectangular image for previewing."""
        face_resolution = int(self.faces.shape[1])
        height = max(int(height or face_resolution * 2), 8)
        width = max(int(width or face_resolution * 4), 16)
        longitude = (
            (np.arange(width, dtype=np.float32) + 0.5) / float(width) * (2.0 * np.pi)
            - np.pi
        )
        colatitude = (np.arange(height, dtype=np.float32) + 0.5) / float(height) * np.pi
        longitude_grid, colatitude_grid = np.meshgrid(longitude, colatitude, indexing="xy")
        directions = np.stack(
            [
                np.sin(colatitude_grid) * np.cos(longitude_grid),
                np.sin(colatitude_grid) * np.sin(longitude_grid),
                np.cos(colatitude_grid),
            ],
            axis=-1,
        )
        return self.sample(directions.reshape(-1, 3), 0.0).reshape(height, width, 3)


def textured_environment_sphere(environment: EnvironmentMap, latitudes: int = 32, longitudes: int = 64):
    """Build a textured Trimesh sphere whose surface direction indexes the cubemap."""
    from PIL import Image
    import trimesh

    latitudes = max(int(latitudes), 8)
    longitudes = max(int(longitudes), 16)
    theta = np.linspace(0.0, np.pi, latitudes + 1, dtype=np.float32)
    phi = np.linspace(-np.pi, np.pi, longitudes + 1, dtype=np.float32)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="xy")
    vertices = np.stack(
        [
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)
    uv = np.stack(
        [
            (phi_grid + np.pi) / (2.0 * np.pi),
            1.0 - theta_grid / np.pi,
        ],
        axis=-1,
    ).reshape(-1, 2)
    row_width = longitudes + 1
    faces = []
    for row in range(latitudes):
        for column in range(longitudes):
            top_left = row * row_width + column
            bottom_left = (row + 1) * row_width + column
            faces.append((top_left, bottom_left, top_left + 1))
            faces.append((top_left + 1, bottom_left, bottom_left + 1))

    texture_rgb = np.clip(environment.to_equirectangular() * 255.0, 0.0, 255.0).astype(np.uint8)
    texture = Image.fromarray(texture_rgb)
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[0, 0, 0, 1],
        emissiveTexture=texture,
        emissiveFactor=[1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(
        vertices=vertices.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        visual=visual,
        process=False,
    )
