#!/usr/bin/env python3
"""Extract mask-visible vehicle Gaussian geometry from an InfiniDepth PLY.

The PLY remains a source-camera reconstruction. This adapter selects visible
Gaussian centres inside the same-frame vehicle mask, transforms centres and
covariances to the Argoverse local world frame, and deliberately performs no
kNN outlier removal, top-view rendering, crop, or shadow inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData


def camera_to_world(data_root: Path, timestep: int, camera: int) -> np.ndarray:
    ego_zero = np.loadtxt(data_root / "ego_pose/000.txt")
    ego_current = np.loadtxt(data_root / "ego_pose" / f"{timestep:03d}.txt")
    camera_to_ego = np.loadtxt(data_root / "extrinsics" / f"{camera}.txt")
    return np.linalg.inv(ego_zero) @ ego_current @ camera_to_ego


def quaternion_to_rotation(quaternions: np.ndarray) -> np.ndarray:
    q = quaternions.astype(np.float64, copy=False)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    w, x, y, z = q.T
    rotations = np.empty((len(q), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - w * z)
    rotations[:, 0, 2] = 2 * (x * z + w * y)
    rotations[:, 1, 0] = 2 * (x * y + w * z)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - w * x)
    rotations[:, 2, 0] = 2 * (x * z - w * y)
    rotations[:, 2, 1] = 2 * (y * z + w * x)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values.astype(np.float64), -30.0, 30.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--vehicle-mask", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.25)
    parser.add_argument("--min-opacity", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mask = cv2.imread(str(args.vehicle_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.vehicle_mask)
    mask = mask > 0
    image_height, image_width = mask.shape
    intrinsic_values = np.loadtxt(args.data_root / "intrinsics" / f"{args.camera}.txt")
    fx, fy, cx, cy = [float(value) for value in intrinsic_values[:4]]
    intrinsic = np.array(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))
    c2w = camera_to_world(args.data_root, args.timestep, args.camera)

    vertices = PlyData.read(args.ply, mmap=True)["vertex"].data
    required = {
        "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = required - set(vertices.dtype.names or ())
    if missing:
        raise RuntimeError(f"InfiniDepth PLY lacks required Gaussian fields: {sorted(missing)}")
    means_ply = np.stack([np.asarray(vertices[name]) for name in ("x", "y", "z")], axis=1).astype(np.float64)
    negative_depth_fraction = float(np.mean(means_ply[:, 2] < -0.1))
    # InfiniDepth unprojects metric camera-z with K^-1[u,v,1] and exports in
    # the same OpenCV camera frame when shift_to_center is disabled.  A sign
    # inferred from the point distribution is invalid: it flips x/y together
    # with z and turns a scene translation into an apparent axis convention.
    means_camera = means_ply
    depth = means_camera[:, 2]
    opacity = sigmoid(np.asarray(vertices["opacity"]))

    pixel_x = np.full(len(means_camera), -1, dtype=np.int32)
    pixel_y = np.full(len(means_camera), -1, dtype=np.int32)
    projectable = (depth > 0.1) & (opacity >= args.min_opacity)
    pixel_x[projectable] = np.rint(fx * means_camera[projectable, 0] / depth[projectable] + cx).astype(np.int32)
    pixel_y[projectable] = np.rint(fy * means_camera[projectable, 1] / depth[projectable] + cy).astype(np.int32)
    projectable &= (
        (pixel_x >= 0) & (pixel_x < image_width)
        & (pixel_y >= 0) & (pixel_y < image_height)
    )
    projectable_indices = np.flatnonzero(projectable)
    linear = pixel_y[projectable_indices].astype(np.int64) * image_width + pixel_x[projectable_indices]
    z_buffer = np.full(image_height * image_width, np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, linear, depth[projectable_indices].astype(np.float32))
    visible = depth[projectable_indices] <= z_buffer[linear] + args.depth_tolerance_m
    visible_indices = projectable_indices[visible]
    selected = visible_indices[mask[pixel_y[visible_indices], pixel_x[visible_indices]]]
    if len(selected) < 100:
        raise RuntimeError(f"only {len(selected)} mask-visible InfiniDepth Gaussians selected")
    selected_camera = means_camera[selected]
    positions_world = selected_camera @ c2w[:3, :3].T + c2w[:3, 3]
    log_scales = np.stack([np.asarray(vertices[f"scale_{axis}"])[selected] for axis in range(3)], axis=1)
    quaternions = np.stack([np.asarray(vertices[f"rot_{axis}"])[selected] for axis in range(4)], axis=1)
    scales = np.exp(np.clip(log_scales.astype(np.float64), -20.0, 10.0))
    rotations_camera = quaternion_to_rotation(quaternions)
    covariance_camera = np.einsum(
        "nij,nj,nkj->nik", rotations_camera, scales * scales, rotations_camera
    )
    world_rotation = c2w[:3, :3]
    covariances_world = np.einsum(
        "ij,njk,lk->nil", world_rotation, covariance_camera, world_rotation
    )

    np.savez_compressed(
        args.output_dir / "visible_infinidepth_vehicle_gaussians.npz",
        positions_world=positions_world.astype(np.float32),
        covariances_world=covariances_world.astype(np.float32),
        log_scales=log_scales.astype(np.float32),
        quaternions_camera=quaternions.astype(np.float32),
        opacity=opacity[selected].astype(np.float32),
        depths_camera=depth[selected].astype(np.float32),
        pixels_xy=np.c_[pixel_x[selected], pixel_y[selected]],
        camera_to_world=c2w,
        intrinsics=intrinsic,
    )

    source = cv2.imread(str(args.data_root / "images" / f"{args.timestep:03d}_{args.camera}.jpg"))
    if source is not None:
        overlay = source.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array((255, 255, 0))).astype(np.uint8)
        stride = max(1, len(selected) // 150000)
        for x, y in zip(pixel_x[selected][::stride], pixel_y[selected][::stride]):
            cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
        cv2.putText(overlay, "InfiniDepth mask-visible vehicle Gaussians", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, "InfiniDepth mask-visible vehicle Gaussians", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.output_dir / "01_infinidepth_vehicle_selection_review.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 96])

    manifest = {
        "source_ply": str(args.ply.resolve()),
        "vehicle_mask": str(args.vehicle_mask.resolve()),
        "timestep": args.timestep,
        "camera": args.camera,
        "ply_total_gaussians": int(len(means_camera)),
        "projectable_opacity_filtered": int(len(projectable_indices)),
        "center_depth_visible": int(len(visible_indices)),
        "mask_visible_selected": int(len(selected)),
        "coordinate_contract": {
            "negative_ply_depth_fraction": negative_depth_fraction,
            "ply_to_cv_coordinate": "identity; OpenCV +z source-camera frame",
            "requires_uncentered_ply": True,
            "world_transform": "inverse(ego_pose_000) @ ego_pose_t @ camera_to_ego",
        },
        "selection_contract": "same-frame vehicle-mask to InfiniDepth Gaussian selection by source-camera centre projection, opacity gate, and centre-depth visibility tolerance; no external instance-box gate",
        "depth_tolerance_m": args.depth_tolerance_m,
        "min_opacity": args.min_opacity,
        "outlier_removal": "none",
        "top_view_processing": "none",
        "output": "visible_infinidepth_vehicle_gaussians.npz",
    }
    (args.output_dir / "infinidepth_vehicle_geometry_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
