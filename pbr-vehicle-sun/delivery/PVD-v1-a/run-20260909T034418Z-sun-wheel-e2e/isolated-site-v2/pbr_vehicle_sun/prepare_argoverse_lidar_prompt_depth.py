#!/usr/bin/env python3
"""Project one Argoverse LiDAR sweep into a sparse camera-z depth map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image_path = args.data_root / "images" / f"{args.timestep:03d}_{args.camera}.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    fx, fy, cx, cy, *_ = np.loadtxt(args.data_root / "intrinsics" / f"{args.camera}.txt")
    intrinsic = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
    points_ego = np.fromfile(args.data_root / "lidar" / f"{args.timestep:03d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    camera_to_ego = np.loadtxt(args.data_root / "extrinsics" / f"{args.camera}.txt")
    points_camera = (np.linalg.inv(camera_to_ego) @ np.c_[points_ego, np.ones(len(points_ego))].T).T[:, :3]
    points_camera = points_camera[points_camera[:, 2] > 0.1]
    projected = (intrinsic @ points_camera.T).T
    x = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int32)
    y = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    depth = np.full(height * width, np.inf, np.float32)
    linear = y[valid].astype(np.int64) * width + x[valid].astype(np.int64)
    np.minimum.at(depth, linear, points_camera[valid, 2].astype(np.float32))
    depth[~np.isfinite(depth)] = 0
    depth = depth.reshape(height, width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, depth, allow_pickle=False)
    values = depth[depth > 0]
    manifest = {
        "image": str(image_path.resolve()), "lidar": str((args.data_root / 'lidar' / f'{args.timestep:03d}.bin').resolve()),
        "timestep": args.timestep, "camera": args.camera, "shape_hw": [height, width],
        "semantics": "sparse metric OpenCV camera-z; nearest LiDAR point per rounded pixel; zero is missing",
        "valid_pixels": int(len(values)), "valid_depth_min_m": float(values.min()), "valid_depth_max_m": float(values.max()),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
