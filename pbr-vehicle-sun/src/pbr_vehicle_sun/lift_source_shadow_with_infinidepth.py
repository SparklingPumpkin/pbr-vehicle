#!/usr/bin/env python3
"""Lift a source-view shadow mask with per-pixel InfiniDepth metric depth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


VEHICLE_EVIDENCE_EXIT_CODE = 42


class VehicleEvidenceError(RuntimeError):
    pass


def camera_to_world(data_root: Path, timestep: int, camera: int) -> np.ndarray:
    return np.linalg.inv(np.loadtxt(data_root / "ego_pose/000.txt")) @ np.loadtxt(
        data_root / "ego_pose" / f"{timestep:03d}.txt"
    ) @ np.loadtxt(data_root / "extrinsics" / f"{camera}.txt")


def unproject(mask: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0.1))
    z = depth[y, x].astype(np.float64)
    camera = np.c_[
        (x - intrinsic[0, 2]) * z / intrinsic[0, 0],
        (y - intrinsic[1, 2]) * z / intrinsic[1, 1],
        z,
    ]
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    return world, np.c_[x, y]


def fit_plane(points: np.ndarray) -> np.ndarray:
    design = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
    coefficients, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    return coefficients


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 1200), 52), (0, 0, 0), -1)
    cv2.putText(result, text, (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--shadow-mask", type=Path, required=True)
    parser.add_argument("--road-mask", type=Path, required=True)
    parser.add_argument("--vehicle-mask", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--road-residual-m", type=float, default=0.20)
    parser.add_argument("--max-plane-points", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    depth = np.load(args.depth, allow_pickle=False).astype(np.float64)
    shadow = cv2.imread(str(args.shadow_mask), cv2.IMREAD_GRAYSCALE) > 0
    road = cv2.imread(str(args.road_mask), cv2.IMREAD_GRAYSCALE) > 0
    vehicle = cv2.imread(str(args.vehicle_mask), cv2.IMREAD_GRAYSCALE) > 0
    source = cv2.imread(str(args.source_image), cv2.IMREAD_COLOR)
    if source is None or depth.shape != shadow.shape or shadow.shape != road.shape or road.shape != vehicle.shape:
        raise ValueError("depth, source RGB, shadow, road and vehicle masks must share the original image dimensions")
    values = np.loadtxt(args.data_root / "intrinsics" / f"{args.camera}.txt")
    fx, fy, cx, cy = map(float, values[:4])
    intrinsic = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
    c2w = camera_to_world(args.data_root, args.timestep, args.camera)

    road_pixels = road & ~vehicle
    road_world, _ = unproject(road_pixels, depth, intrinsic, c2w)
    if len(road_world) < 1000:
        raise VehicleEvidenceError("insufficient valid predicted-depth road pixels")
    rng = np.random.default_rng(args.seed)
    sample = road_world[rng.choice(len(road_world), min(len(road_world), args.max_plane_points), replace=False)]
    plane = fit_plane(sample)
    for _ in range(4):
        residual = sample[:, 2] - (plane[0] * sample[:, 0] + plane[1] * sample[:, 1] + plane[2])
        limit = np.quantile(np.abs(residual), 0.70)
        plane = fit_plane(sample[np.abs(residual) <= limit])

    shadow_world, shadow_xy = unproject(shadow, depth, intrinsic, c2w)
    shadow_residual = shadow_world[:, 2] - (
        plane[0] * shadow_world[:, 0] + plane[1] * shadow_world[:, 1] + plane[2]
    )
    near = np.abs(shadow_residual) <= args.road_residual_m
    near_world = shadow_world[near]
    near_xy = shadow_xy[near]
    if len(near_world) < 1000:
        raise VehicleEvidenceError(f"only {len(near_world)} lifted shadow pixels lie near the fitted road")
    np.savez_compressed(
        args.output_dir / "infinidepth_shadow_geometry.npz",
        road_points_world=sample,
        shadow_points_world=shadow_world,
        near_ground_shadow_points_world=near_world,
        shadow_pixels_xy=shadow_xy,
        near_ground_shadow_pixels_xy=near_xy,
        shadow_plane_residual_m=shadow_residual,
        plane_z_ax_by_c=plane,
        camera_to_world=c2w,
        intrinsics=intrinsic,
    )

    valid = np.isfinite(depth) & (depth > 0.1)
    lo, hi = np.quantile(depth[valid], (0.01, 0.99))
    depth_view = cv2.applyColorMap(np.rint(np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    source_shadow = source.copy(); source_shadow[shadow] = (0.30 * source_shadow[shadow] + 0.70 * np.array((0, 0, 255))).astype(np.uint8)
    lifted = source.copy(); lifted[shadow] = (0.35 * lifted[shadow] + 0.65 * np.array((0, 165, 255))).astype(np.uint8)
    near_mask = np.zeros_like(shadow); near_mask[near_xy[:, 1], near_xy[:, 0]] = True
    lifted[near_mask] = (0.20 * lifted[near_mask] + 0.80 * np.array((0, 255, 0))).astype(np.uint8)
    residual_view = np.zeros_like(source)
    residual_image = np.full(shadow.shape, np.nan, dtype=np.float32)
    residual_image[shadow_xy[:, 1], shadow_xy[:, 0]] = np.abs(shadow_residual)
    residual_norm = np.nan_to_num(np.clip(residual_image / 1.0, 0, 1), nan=0.0)
    residual_view = cv2.applyColorMap(np.rint(residual_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    residual_view[~shadow] = 0
    review = cv2.vconcat((
        cv2.hconcat((label(source_shadow, "source-view SSISv2 associated shadow"), label(depth_view, "InfiniDepth metric camera-z"))),
        cv2.hconcat((label(lifted, "orange=all lifted; green=within 0.20m of road"), label(residual_view, "absolute residual to fitted road (0-1m)"))),
    ))
    cv2.imwrite(str(args.output_dir / "01_source_shadow_depth_lift_review.jpg"), review, [cv2.IMWRITE_JPEG_QUALITY, 96])
    manifest = {
        "timestep": args.timestep, "camera": args.camera,
        "depth_semantics": "same-frame InfiniDepth metric camera-z; one depth value per retained source-view shadow pixel",
        "source_shadow_contract": "official SSISv2 associated shadow mask, unchanged before geometric lifting",
        "shadow_pixels": int(shadow.sum()), "lifted_shadow_points": int(len(shadow_world)),
        "near_ground_shadow_points": int(len(near_world)), "near_ground_fraction": float(near.mean()),
        "road_pixels_with_valid_depth": int(len(road_world)), "road_plane_fit_points": int(len(sample)),
        "road_residual_gate_m": args.road_residual_m, "plane_z_ax_by_c": plane.tolist(),
        "shadow_residual_abs_quantiles_m": {str(q): float(np.quantile(np.abs(shadow_residual), q)) for q in (0.5, 0.9, 0.99)},
        "output_npz": "infinidepth_shadow_geometry.npz", "review": "01_source_shadow_depth_lift_review.jpg",
    }
    (args.output_dir / "lift_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except VehicleEvidenceError as error:
        print(f"vehicle evidence rejected: {error}", file=sys.stderr)
        raise SystemExit(VEHICLE_EVIDENCE_EXIT_CODE)
