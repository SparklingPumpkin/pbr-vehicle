#!/usr/bin/env python3
"""Fit sun angles with the SSE-v3 contour objective and a depth-lifted shadow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError

try:
    from .fit_sun_from_dense_shadow_boundary import (
        basis, candidate_score, contour_xy, evaluate, footprint_mask, plane_xy,
        points_to_px, unique_basins,
    )
except ImportError:
    from fit_sun_from_dense_shadow_boundary import (
        basis, candidate_score, contour_xy, evaluate, footprint_mask, plane_xy,
        points_to_px, unique_basins,
    )


def rasterize_points(xy: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: int) -> np.ndarray:
    pixels = points_to_px(xy, lo, hi, size)
    pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < size) & (pixels[:, 1] >= 0) & (pixels[:, 1] < size)]
    result = np.zeros((size, size), dtype=np.uint8)
    result[pixels[:, 1], pixels[:, 0]] = 1
    result = cv2.dilate(result, np.ones((3, 3), np.uint8))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    keep = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)[:2]
    return np.isin(labels, keep)


def quaternion_to_rotation(quaternions: np.ndarray) -> np.ndarray:
    """Convert gsplat wxyz quaternions to row-major rotation matrices."""
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


def gaussian_covariances(log_scales: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    scales = np.exp(np.clip(log_scales.astype(np.float64), -20.0, 10.0))
    rotations = quaternion_to_rotation(quaternions)
    return np.einsum("nij,nj,nkj->nik", rotations, scales * scales, rotations)


def support_gaussian_indices(points: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Retain 3D hull centres plus the largest splats that may cross that hull."""
    try:
        hull_indices = ConvexHull(points).vertices
    except QhullError:
        hull_indices = np.arange(len(points))
    largest_count = min(512, len(points))
    largest = np.argpartition(np.linalg.eigvalsh(covariances)[:, -1], -largest_count)[-largest_count:]
    return np.unique(np.concatenate((hull_indices, largest)))


def gaussian_support_mask(
    projected_centres: np.ndarray,
    covariances_world: np.ndarray,
    support_indices: np.ndarray,
    world_to_plane: np.ndarray,
    support_sigma: float,
    circle_samples: int,
    lo: np.ndarray,
    hi: np.ndarray,
    size: int,
) -> np.ndarray:
    """Rasterize the convex envelope of projected Gaussian ellipse supports."""
    selected_covariances = covariances_world[support_indices]
    covariances_2d = np.einsum(
        "ia,nij,jb->nab", world_to_plane, selected_covariances, world_to_plane
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariances_2d)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    covariance_sqrt = np.einsum(
        "nij,nj,nkj->nik", eigenvectors, np.sqrt(eigenvalues), eigenvectors
    )
    angles = np.linspace(0.0, 2.0 * np.pi, circle_samples, endpoint=False)
    circle = np.c_[np.cos(angles), np.sin(angles)]
    offsets = np.einsum("md,ndk->nmk", circle, covariance_sqrt)
    support_points = projected_centres[support_indices, None, :] + support_sigma * offsets
    return footprint_mask(support_points.reshape(-1, 2), lo, hi, size)


def gaussian_union_mask(
    projected_centres: np.ndarray,
    covariances_world: np.ndarray,
    world_to_plane: np.ndarray,
    support_sigma: float,
    lo: np.ndarray,
    hi: np.ndarray,
    size: int,
) -> np.ndarray:
    """Rasterize a non-convex union of projected Gaussian support disks.

    Each disk radius is the exact major eigen-radius of that Gaussian after
    projection to the road plane. Pixel radii are quantized upward only to
    amortize rasterization; this is conservative against silhouette loss.
    """
    covariances_2d = np.einsum(
        "ia,nij,jb->nab", world_to_plane, covariances_world, world_to_plane
    )
    scale_x = (size - 1) / max(float(hi[0] - lo[0]), 1e-9)
    scale_y = (size - 1) / max(float(hi[1] - lo[1]), 1e-9)
    pixel_scale = np.array((scale_x, scale_y))
    covariances_px = covariances_2d * pixel_scale[None, :, None] * pixel_scale[None, None, :]
    trace = covariances_px[:, 0, 0] + covariances_px[:, 1, 1]
    determinant = covariances_px[:, 0, 0] * covariances_px[:, 1, 1] - covariances_px[:, 0, 1] ** 2
    major_variance = 0.5 * (trace + np.sqrt(np.clip(trace * trace - 4.0 * determinant, 0.0, None)))
    radius_pixels = support_sigma * np.sqrt(np.clip(major_variance, 1e-12, None))
    # The source checkpoint contains a handful of metre-scale support outliers.
    # Keep 99.5% of the real scale distribution and cap only that extreme tail.
    radius_pixels = np.clip(radius_pixels, 1.0, np.quantile(radius_pixels, 0.995))
    bins = np.array((1, 2, 3, 4, 6, 8, 12, 16, 24, 32), dtype=np.int32)
    bin_indices = np.clip(np.searchsorted(bins, np.ceil(radius_pixels).astype(np.int32)), 0, len(bins) - 1)
    pixels = points_to_px(projected_centres, lo, hi, size)
    valid = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < size)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < size)
    )
    pixels, bin_indices = pixels[valid], bin_indices[valid]
    result = np.zeros((size, size), dtype=np.uint8)
    for index in np.unique(bin_indices):
        radius = int(bins[index])
        impulses = np.zeros_like(result)
        selected = pixels[bin_indices == index]
        impulses[selected[:, 1], selected[:, 0]] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        result |= cv2.dilate(impulses, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return result.astype(bool)


def evaluate_gaussian_support(
    vehicle_xy: np.ndarray,
    heights: np.ndarray,
    vehicle_floor: np.ndarray,
    covariances_world: np.ndarray,
    support_indices: np.ndarray,
    direction: float,
    elevation: float,
    n: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    size: int,
    observed_boundary: np.ndarray,
    centre: np.ndarray,
    contact_radius: float,
    support_sigma: float,
    circle_samples: int,
    footprint_mode: str,
) -> dict | None:
    horizontal = np.array((np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))))
    cotangent = 1.0 / np.tan(np.deg2rad(elevation))
    projected_xy = vehicle_xy - heights[:, None] * cotangent * horizontal[None, :]
    plane_basis = np.c_[e1, e2]
    world_to_shadow_plane = plane_basis - n[:, None] * cotangent * horizontal[None, :]
    if footprint_mode == "gaussian_union":
        raw_predicted_mask = gaussian_union_mask(
            projected_xy, covariances_world, world_to_shadow_plane,
            support_sigma, lo, hi, size,
        )
    else:
        raw_predicted_mask = gaussian_support_mask(
            projected_xy, covariances_world, support_indices, world_to_shadow_plane,
            support_sigma, circle_samples, lo, hi, size,
        )
    predicted_mask = raw_predicted_mask.copy()
    predicted_mask &= ~cv2.dilate(
        vehicle_floor.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    predicted_boundary = contour_xy(predicted_mask, lo, hi)
    shadow_vector = -horizontal
    metrics = candidate_score(
        predicted_boundary, observed_boundary, centre, shadow_vector, contact_radius
    )
    if metrics is None:
        return None
    return metrics | {
        "azimuth_deg": float(direction),
        "elevation_deg": float(elevation),
        "predicted_boundary_points": int(len(predicted_boundary)),
        "predicted_xy": projected_xy,
        "pred_mask": predicted_mask,
        "raw_predicted_mask_area_pixels": int(raw_predicted_mask.sum()),
        "predicted_mask_area_pixels": int(predicted_mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle-geometry", type=Path, required=True)
    parser.add_argument("--shadow-geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raster-size", type=int, default=900)
    parser.add_argument("--max-vehicle-points", type=int, default=50000)
    parser.add_argument("--coarse-az-step", type=int, default=5)
    parser.add_argument("--coarse-elev-min", type=int, default=20)
    parser.add_argument("--coarse-elev-max", type=int, default=50)
    parser.add_argument("--local-basins", type=int, default=3)
    parser.add_argument("--local-radius-deg", type=int, default=5)
    parser.add_argument("--local-step-deg", type=int, default=1)
    parser.add_argument("--contact-radius-m", type=float, default=0.20)
    parser.add_argument("--footprint-mode", choices=("center_hull", "gaussian_support", "gaussian_union"), default="center_hull")
    parser.add_argument("--gaussian-support-sigma", type=float, default=2.0)
    parser.add_argument("--support-circle-samples", type=int, default=16)
    parser.add_argument("--min-vehicle-height-m", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vd, sd = np.load(args.vehicle_geometry), np.load(args.shadow_geometry)
    plane = sd["plane_z_ax_by_c"].astype(np.float64)
    n, e1, e2, anchor = basis(plane)
    vehicle_all = vd["positions_world"].astype(np.float64)
    heights_all = (vehicle_all - anchor) @ n
    keep = (heights_all >= args.min_vehicle_height_m) & (heights_all <= 5.0)
    vehicle, heights = vehicle_all[keep], heights_all[keep]
    covariances_world = None
    support_indices = None
    if args.footprint_mode != "center_hull":
        if "covariances_world" in vd.files:
            covariances_world = vd["covariances_world"][keep].astype(np.float64)
        else:
            missing = {"log_scales", "quaternions"} - set(vd.files)
            if missing:
                raise RuntimeError(f"vehicle geometry lacks Gaussian support fields: {sorted(missing)}")
            covariances_world = gaussian_covariances(vd["log_scales"][keep], vd["quaternions"][keep])
    if len(vehicle) > args.max_vehicle_points:
        indices = np.random.default_rng(args.seed).choice(len(vehicle), args.max_vehicle_points, replace=False)
        vehicle, heights = vehicle[indices], heights[indices]
        if covariances_world is not None:
            covariances_world = covariances_world[indices]
    shadow = sd["near_ground_shadow_points_world"].astype(np.float64)
    vehicle_xy = plane_xy(vehicle, e1, e2, anchor)
    shadow_xy = plane_xy(shadow, e1, e2, anchor)
    centre = np.median(vehicle_xy, axis=0)
    # Robust display/search bounds: retain the measured shadow but reject rare
    # depth spikes before rasterization.
    relative = shadow_xy - centre
    radius = np.linalg.norm(relative, axis=1)
    shadow_xy = shadow_xy[radius <= np.quantile(radius, 0.995)]
    stacked = np.vstack((vehicle_xy, shadow_xy))
    lo, hi = np.quantile(stacked, (0.002, 0.998), axis=0)
    pad = np.maximum((hi - lo) * 0.08, 0.6)
    lo, hi = lo - pad, hi + pad
    dense = rasterize_points(shadow_xy, lo, hi, args.raster_size)
    if args.footprint_mode != "center_hull":
        support_indices = support_gaussian_indices(vehicle, covariances_world)
        plane_basis = np.c_[e1, e2]
        if args.footprint_mode == "gaussian_union":
            vehicle_floor = gaussian_union_mask(
                vehicle_xy, covariances_world, plane_basis,
                args.gaussian_support_sigma, lo, hi, args.raster_size,
            )
        else:
            vehicle_floor = gaussian_support_mask(
                vehicle_xy, covariances_world, support_indices, plane_basis,
                args.gaussian_support_sigma, args.support_circle_samples,
                lo, hi, args.raster_size,
            )
    else:
        vehicle_floor = footprint_mask(vehicle_xy, lo, hi, args.raster_size)
    observed_boundary = contour_xy(dense, lo, hi)
    if len(observed_boundary) < 8:
        raise RuntimeError("depth-lifted shadow did not yield a usable boundary")

    coarse = []
    for elevation in range(args.coarse_elev_min, args.coarse_elev_max + 1, 5):
        for azimuth in range(0, 360, args.coarse_az_step):
            if args.footprint_mode != "center_hull":
                row = evaluate_gaussian_support(
                    vehicle_xy, heights, vehicle_floor, covariances_world, support_indices,
                    azimuth, elevation, n, e1, e2, lo, hi, args.raster_size,
                    observed_boundary, centre, args.contact_radius_m,
                    args.gaussian_support_sigma, args.support_circle_samples,
                    args.footprint_mode,
                )
            else:
                row = evaluate(vehicle_xy, vehicle_floor, heights, azimuth, elevation, n, e1, e2, anchor, lo, hi, args.raster_size, observed_boundary, centre, args.contact_radius_m)
            if row:
                coarse.append(row)
    basins = unique_basins(coarse, args.local_basins)
    refined = []
    for basin in basins:
        for elevation in range(int(basin["elevation_deg"]) - args.local_radius_deg, int(basin["elevation_deg"]) + args.local_radius_deg + 1, args.local_step_deg):
            if elevation <= 1:
                continue
            for delta in range(-args.local_radius_deg, args.local_radius_deg + 1, args.local_step_deg):
                azimuth = (int(round(basin["azimuth_deg"])) + delta) % 360
                if args.footprint_mode != "center_hull":
                    row = evaluate_gaussian_support(
                        vehicle_xy, heights, vehicle_floor, covariances_world, support_indices,
                        azimuth, elevation, n, e1, e2, lo, hi, args.raster_size,
                        observed_boundary, centre, args.contact_radius_m,
                        args.gaussian_support_sigma, args.support_circle_samples,
                        args.footprint_mode,
                    )
                else:
                    row = evaluate(vehicle_xy, vehicle_floor, heights, azimuth, elevation, n, e1, e2, anchor, lo, hi, args.raster_size, observed_boundary, centre, args.contact_radius_m)
                if row:
                    row["basin_seed_azimuth_deg"] = basin["azimuth_deg"]
                    row["basin_seed_elevation_deg"] = basin["elevation_deg"]
                    refined.append(row)
    best = max(refined or coarse, key=lambda row: row["score"])
    for rows, filename in ((coarse, "coarse_scores.csv"), (refined, "refined_scores.csv")):
        fields = [key for key, value in rows[0].items() if np.isscalar(value) and key not in ("pred_mask", "predicted_xy")]
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            writer.writerows([{key: row[key] for key in fields} for row in rows])

    def panel(row: dict, title: str) -> np.ndarray:
        image = np.full((700, 700, 3), 250, np.uint8)
        observed = cv2.resize(dense.astype(np.uint8), (700, 700), interpolation=cv2.INTER_NEAREST).astype(bool)
        image[observed] = (90, 90, 90)
        if args.footprint_mode != "center_hull":
            horizontal = np.array((np.cos(np.deg2rad(row["azimuth_deg"])), np.sin(np.deg2rad(row["azimuth_deg"]))))
            cotangent = 1.0 / np.tan(np.deg2rad(row["elevation_deg"]))
            plane_basis = np.c_[e1, e2]
            world_to_shadow_plane = plane_basis - n[:, None] * cotangent * horizontal[None, :]
            if args.footprint_mode == "gaussian_union":
                raw_predicted = gaussian_union_mask(
                    row["predicted_xy"], covariances_world, world_to_shadow_plane,
                    args.gaussian_support_sigma, lo, hi, args.raster_size,
                )
            else:
                raw_predicted = gaussian_support_mask(
                    row["predicted_xy"], covariances_world, support_indices,
                    world_to_shadow_plane, args.gaussian_support_sigma,
                    args.support_circle_samples, lo, hi, args.raster_size,
                )
        else:
            raw_predicted = footprint_mask(row["predicted_xy"], lo, hi, args.raster_size)
        display_vehicle = cv2.resize(vehicle_floor.astype(np.uint8), (700, 700), interpolation=cv2.INTER_NEAREST)
        display_raw_predicted = cv2.resize(raw_predicted.astype(np.uint8), (700, 700), interpolation=cv2.INTER_NEAREST)
        display_predicted = cv2.resize(row["pred_mask"].astype(np.uint8), (700, 700), interpolation=cv2.INTER_NEAREST)
        vehicle_contours, _ = cv2.findContours(display_vehicle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        raw_predicted_contours, _ = cv2.findContours(display_raw_predicted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        predicted_contours, _ = cv2.findContours(display_predicted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, vehicle_contours, -1, (255, 80, 0), 3)
        # Red is the complete projected Gaussian support envelope requested for
        # visual audit. Yellow is the exact shadow-only boundary used by the
        # score after subtracting the vehicle contact footprint.
        cv2.drawContours(image, raw_predicted_contours, -1, (0, 0, 255), 3)
        cv2.drawContours(image, predicted_contours, -1, (0, 180, 255), 1)
        cv2.putText(image, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "blue vehicle | red full support | yellow scored", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return image

    candidates = sorted(refined or coarse, key=lambda row: row["score"], reverse=True)[:25]
    while len(candidates) < 25:
        candidates.append(best)
    panels = [panel(row, f"az {row['azimuth_deg']:.0f} | el {row['elevation_deg']:.0f} | {row['score']:.3f}") for row in candidates]
    sheet = cv2.vconcat([cv2.hconcat(panels[index:index + 5]) for index in range(0, 25, 5)])
    cv2.imwrite(str(args.output_dir / "top25_depth_lifted_boundary_candidates.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(args.output_dir / "best_depth_lifted_boundary_fit.png"), panel(best, f"BEST az {best['azimuth_deg']:.2f} | elev {best['elevation_deg']:.2f} | score {best['score']:.4f}"))
    result = {
        "best": {key: value for key, value in best.items() if key not in ("pred_mask", "predicted_xy")},
        "coarse_count": len(coarse), "refined_count": len(refined),
        "coarse_basins": [{key: value for key, value in basin.items() if key not in ("pred_mask", "predicted_xy")} for basin in basins],
        "search": {"coarse_azimuth": [0, 355, args.coarse_az_step], "coarse_elevation": [args.coarse_elev_min, args.coarse_elev_max, 5], "local_radius_deg": args.local_radius_deg, "local_step_deg": args.local_step_deg},
        "objective": "continuous projected vehicle boundary versus observed dense shadow boundary with directional half-plane and weak reverse support",
        "footprint": {
            "mode": args.footprint_mode,
            "gaussian_support_sigma": args.gaussian_support_sigma if args.footprint_mode != "center_hull" else None,
            "support_circle_samples": args.support_circle_samples if args.footprint_mode == "gaussian_support" else None,
            "support_gaussians": int(len(vehicle)) if args.footprint_mode == "gaussian_union" else (int(len(support_indices)) if support_indices is not None else None),
            "min_vehicle_height_m": args.min_vehicle_height_m,
            "visualization": "red=complete projected support envelope; yellow=post-contact-subtraction scored boundary; blue=vehicle support envelope",
        },
        "observed_geometry": "source-view postprocessed MTMT pixels independently unprojected by same-frame InfiniDepth metric camera-z, then gated to 0.20m from the fitted road",
        "inputs": {"vehicle_geometry": str(args.vehicle_geometry.resolve()), "shadow_geometry": str(args.shadow_geometry.resolve())},
        "plane_z_ax_by_c": plane.tolist(), "plane_bounds": {"min": lo.tolist(), "max": hi.tolist()},
        "vehicle_points": int(len(vehicle)), "shadow_points": int(len(shadow_xy)), "observed_boundary_points": int(len(observed_boundary)),
    }
    (args.output_dir / "fit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
