"""Binary little-endian PLY support for standard 3D Gaussian assets."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np


SH_C0 = 0.28209479177387814
PLY_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "<i2",
    "uint16": "<u2",
    "int32": "<i4",
    "uint32": "<u4",
    "float32": "<f4",
    "float64": "<f8",
}


def read_ply(path: str | Path) -> OrderedDict[str, np.ndarray]:
    source = Path(path)
    with source.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {source}")
        ply_format = None
        vertex_count = 0
        in_vertex = False
        fields: list[tuple[str, str]] = []
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError(f"PLY header has no end_header: {source}")
            line = raw_line.decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format":
                ply_format = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("List-valued vertex properties are not supported")
                if parts[1] not in PLY_TO_DTYPE:
                    raise ValueError(f"Unsupported PLY property type: {parts[1]}")
                fields.append((parts[2], PLY_TO_DTYPE[parts[1]]))
        if ply_format != "binary_little_endian":
            raise ValueError(f"Only binary_little_endian PLY is supported, got {ply_format!r}")
        if vertex_count < 1 or not fields:
            raise ValueError(f"PLY has no vertex payload: {source}")
        dtype = np.dtype(fields, align=False)
        vertices = np.fromfile(stream, dtype=dtype, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError(f"Truncated PLY payload: expected {vertex_count}, got {len(vertices)}")
    return OrderedDict((name, np.asarray(vertices[name]).copy()) for name, _ in fields)


def write_ply(path: str | Path, properties: Mapping[str, np.ndarray]) -> Path:
    destination = Path(path)
    if not properties:
        raise ValueError("Cannot write a PLY without properties")
    arrays = OrderedDict((name, np.asarray(values)) for name, values in properties.items())
    vertex_count = len(next(iter(arrays.values())))
    if vertex_count < 1 or any(values.ndim != 1 or len(values) != vertex_count for values in arrays.values()):
        raise ValueError("Every PLY property must be a one-dimensional array with the same nonzero length")

    dtype_fields: list[tuple[str, np.dtype]] = []
    property_types: list[str] = []
    for name, values in arrays.items():
        if values.dtype.kind == "f":
            dtype_fields.append((name, np.dtype("<f4")))
            property_types.append("float")
        elif values.dtype.kind in "uib":
            if values.dtype.itemsize == 1:
                signed = values.dtype.kind == "i"
                dtype_fields.append((name, np.dtype("i1" if signed else "u1")))
                property_types.append("char" if signed else "uchar")
            else:
                dtype_fields.append((name, np.dtype("<i4")))
                property_types.append("int")
        else:
            raise ValueError(f"Unsupported dtype for property {name!r}: {values.dtype}")

    vertices = np.empty(vertex_count, dtype=np.dtype(dtype_fields, align=False))
    for name, values in arrays.items():
        vertices[name] = values
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        stream.write(b"ply\nformat binary_little_endian 1.0\n")
        stream.write(f"element vertex {vertex_count}\n".encode("ascii"))
        for (name, _), property_type in zip(dtype_fields, property_types):
            stream.write(f"property {property_type} {name}\n".encode("ascii"))
        stream.write(b"end_header\n")
        vertices.tofile(stream)
    return destination


def require_standard_gaussian(properties: Mapping[str, np.ndarray]) -> None:
    required = {
        "x", "y", "z", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "f_dc_0", "f_dc_1", "f_dc_2",
    }
    missing = sorted(required.difference(properties))
    if missing:
        raise ValueError("Input is not a supported standard 3DGS PLY; missing: " + ", ".join(missing))


def positions(properties: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.stack([properties[axis] for axis in ("x", "y", "z")], axis=1).astype(np.float32)


def dc_rgb(properties: Mapping[str, np.ndarray]) -> np.ndarray:
    coefficients = np.stack([properties[f"f_dc_{channel}"] for channel in range(3)], axis=1)
    return np.clip(SH_C0 * coefficients + 0.5, 0.0, 1.0).astype(np.float32)


def quaternion_to_rotation(quaternions: np.ndarray) -> np.ndarray:
    normalized = np.asarray(quaternions, dtype=np.float32)
    normalized = normalized / np.clip(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-8, None)
    w, x, y, z = normalized.T
    rotations = np.empty((len(normalized), 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def gaussian_normals(properties: Mapping[str, np.ndarray]) -> np.ndarray:
    scales = np.stack([properties[f"scale_{index}"] for index in range(3)], axis=1)
    quaternions = np.stack([properties[f"rot_{index}"] for index in range(4)], axis=1)
    minimum_axes = np.argmin(np.exp(np.clip(scales, -20.0, 20.0)), axis=1)
    rotations = quaternion_to_rotation(quaternions)
    normals = rotations[np.arange(len(rotations)), :, minimum_axes]
    return normalize(normals)


def gaussian_covariances(properties: Mapping[str, np.ndarray]) -> np.ndarray:
    scales = np.exp(np.clip(np.stack([properties[f"scale_{index}"] for index in range(3)], axis=1), -20.0, 20.0))
    quaternions = np.stack([properties[f"rot_{index}"] for index in range(4)], axis=1)
    rotations = quaternion_to_rotation(quaternions)
    return np.einsum("nij,nj,nkj->nik", rotations, np.square(scales), rotations).astype(np.float32)


def activated_opacity(properties: Mapping[str, np.ndarray]) -> np.ndarray:
    logits = np.asarray(properties["opacity"], dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).astype(np.float32)


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8, None)
