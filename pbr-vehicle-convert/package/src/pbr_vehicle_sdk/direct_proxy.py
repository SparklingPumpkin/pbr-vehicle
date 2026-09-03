"""Identity-preserving direct white 2DGS conversion from P-v3/P-v2-c."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .ply import SH_C0, read_ply, require_standard_gaussian, write_ply


@dataclass(frozen=True)
class DirectProxyConfig:
    white_level: float = 0.82
    roughness: float = 0.32
    metallic: float = 0.02
    thin_ratio: float = 0.02
    minimum_thickness: float = 1e-5

    def validate(self) -> None:
        if not 0.02 <= self.white_level <= 0.98:
            raise ValueError("white_level must be in [0.02, 0.98]")
        if not 0.02 <= self.roughness <= 0.98:
            raise ValueError("roughness must be in [0.02, 0.98]")
        if not 0.0 <= self.metallic <= 1.0:
            raise ValueError("metallic must be in [0, 1]")
        if not 0.0 < self.thin_ratio < 1.0 or self.minimum_thickness <= 0.0:
            raise ValueError("thin_ratio must be in (0, 1) and minimum_thickness must be positive")


@dataclass(frozen=True)
class DirectProxyResult:
    root: Path
    proxy_ply: Path
    mapping_npz: Path
    manifest_json: Path
    gaussian_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def _rotations(quaternions: np.ndarray) -> np.ndarray:
    q = _normalize(quaternions.astype(np.float32))
    w, x, y, z = q.T
    out = np.empty((len(q), 3, 3), dtype=np.float32)
    out[:, 0, 0] = 1 - 2 * (y*y + z*z); out[:, 0, 1] = 2 * (x*y - z*w); out[:, 0, 2] = 2 * (x*z + y*w)
    out[:, 1, 0] = 2 * (x*y + z*w); out[:, 1, 1] = 1 - 2 * (x*x + z*z); out[:, 1, 2] = 2 * (y*z - x*w)
    out[:, 2, 0] = 2 * (x*z - y*w); out[:, 2, 1] = 2 * (y*z + x*w); out[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return out


def _quaternions(m: np.ndarray) -> np.ndarray:
    # Stable branch conversion, scalar-first convention.
    q = np.empty((len(m), 4), dtype=np.float32)
    for i, r in enumerate(m):
        t = float(np.trace(r))
        if t > 0:
            s = np.sqrt(t + 1.0) * 2; q[i] = [0.25*s, (r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s]
        else:
            j = int(np.argmax(np.diag(r)))
            if j == 0:
                s = np.sqrt(1+r[0,0]-r[1,1]-r[2,2])*2; q[i] = [(r[2,1]-r[1,2])/s, .25*s, (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s]
            elif j == 1:
                s = np.sqrt(1+r[1,1]-r[0,0]-r[2,2])*2; q[i] = [(r[0,2]-r[2,0])/s, (r[0,1]+r[1,0])/s, .25*s, (r[1,2]+r[2,1])/s]
            else:
                s = np.sqrt(1+r[2,2]-r[0,0]-r[1,1])*2; q[i] = [(r[1,0]-r[0,1])/s, (r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s, .25*s]
    return _normalize(q).astype(np.float32)


def build_direct_white_2dgs(input_ply: str | Path, output_dir: str | Path, config: DirectProxyConfig = DirectProxyConfig(), overwrite: bool = False) -> DirectProxyResult:
    config.validate()
    source_path, root = Path(input_ply), Path(output_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source = read_ply(source_path); require_standard_gaussian(source)
    points = np.stack([source[k] for k in ("x", "y", "z")], axis=1).astype(np.float32)
    scales = np.stack([source[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)
    rotations = _rotations(np.stack([source[f"rot_{i}"] for i in range(4)], axis=1))
    order = np.argsort(scales, axis=1); normal_axis, tangent_axes = order[:, 0], order[:, 1:]
    normals = rotations[np.arange(len(points)), :, normal_axis]
    flip = np.sum(normals * (points - np.median(points, axis=0)), axis=1) < 0
    normals[flip] *= -1; normals = _normalize(normals).astype(np.float32)
    frames = np.empty_like(rotations)
    for i in range(len(points)):
        frames[i] = np.stack([rotations[i, :, tangent_axes[i, 0]], rotations[i, :, tangent_axes[i, 1]], normals[i]], axis=1)
        if np.linalg.det(frames[i]) < 0: frames[i, :, 1] *= -1
    tangent_logs = np.take_along_axis(scales, tangent_axes, axis=1)
    tangent = np.exp(np.clip(tangent_logs, -20, 20))
    thickness = np.maximum(np.sqrt(tangent[:, 0] * tangent[:, 1]) * config.thin_ratio, config.minimum_thickness)
    out_scales = np.column_stack([tangent_logs, np.log(thickness)]).astype(np.float32)
    out_rot = _quaternions(frames)
    white = np.full((len(points), 3), config.white_level, dtype=np.float32)
    logit = lambda x: np.log(np.clip(x, 1e-6, 1-1e-6) / np.clip(1-x, 1e-6, 1))
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    for k in ("x", "y", "z"): out[k] = np.asarray(source[k], dtype=np.float32)
    for i in range(3): out[f"f_dc_{i}"] = (white[:, i] - .5) / SH_C0
    for i in range(45): out[f"f_rest_{i}"] = np.zeros(len(points), np.float32)
    out["opacity"] = np.asarray(source["opacity"], dtype=np.float32)
    for i in range(3): out[f"scale_{i}"] = out_scales[:, i]
    for i in range(4): out[f"rot_{i}"] = out_rot[:, i]
    for i in range(3):
        out[f"r3gw_albedo_{i}"] = white[:, i]; out[f"albedo_{i}"] = logit(white[:, i]).astype(np.float32)
        out[f"r3gw_normal_{i}"] = normals[:, i]; out[f"normal_{i}"] = normals[:, i]
    out["r3gw_roughness"] = np.full(len(points), config.roughness, np.float32); out["roughness"] = np.full(len(points), logit(config.roughness), np.float32)
    out["r3gw_metallic"] = np.full(len(points), config.metallic, np.float32); out["metalness"] = np.full(len(points), logit(np.clip(config.metallic, .02, .98)), np.float32)
    out["is_sky"] = np.zeros(len(points), np.uint8); out["r3gw_asset_class"] = np.ones(len(points), np.uint8)
    proxy = write_ply(root / "direct_white_2dgs_proxy.ply", out)
    mapping = root / "direct_white_2dgs_identity_mapping.npz"
    idx = np.arange(len(points), dtype=np.int64)
    np.savez_compressed(mapping, clean_to_proxy_idx=idx, raw_to_proxy_idx=idx, raw_to_proxy_knn_idx=idx[:, None], raw_to_proxy_knn_weight=np.ones((len(points), 1), np.float32), raw_to_proxy_knn_distance=np.zeros((len(points), 1), np.float32))
    manifest = {"schema_version": 1, "mainline": "P-v3", "source_candidate": "P-v2-c", "input": {"filename": source_path.name, "sha256": _sha256(source_path)}, "outputs": {"proxy": proxy.name, "mapping": mapping.name, "proxy_sha256": _sha256(proxy), "mapping_sha256": _sha256(mapping)}, "contract": {"point_relocation": False, "downsampling": False, "gap_filling": False, "normal_smoothing": False, "mapping": "one-to-one identity"}, "gaussian_count": len(points), "outward_flipped_count": int(flip.sum()), "config": asdict(config), "limitations": ["Validated on simulation2 asset 10010", "Center-relative orientation does not prove local signed-normal smoothness"]}
    manifest_path = root / "asset.json"; manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return DirectProxyResult(root, proxy, mapping, manifest_path, len(points))
