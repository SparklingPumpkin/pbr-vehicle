#!/usr/bin/env python3
"""Fit one shared sun angle for one or more camera-visible 2D contour arcs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

try:
    from .fit_sun_fixed_full_contour import gaussian_union_mask
    from .fit_sun_fixed_shadow_joint import robust, setup
    from .fit_sun_from_dense_shadow_boundary import contour_xy, plane_xy
except ImportError:  # direct script execution by the packaged dispatcher
    from fit_sun_fixed_full_contour import gaussian_union_mask
    from fit_sun_fixed_shadow_joint import robust, setup
    from fit_sun_from_dense_shadow_boundary import contour_xy, plane_xy


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ray_hits(contour: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """Return (ray distance, cyclic contour position, point) intersections."""
    hits = []
    for index in range(len(contour)):
        a = contour[index]
        segment = contour[(index + 1) % len(contour)] - a
        denominator = cross2(direction, segment)
        if abs(denominator) < 1e-10:
            continue
        offset = a - origin
        ray_distance = cross2(offset, segment) / denominator
        fraction = cross2(offset, direction) / denominator
        if ray_distance >= 0.0 and -1e-7 <= fraction <= 1.0 + 1e-7:
            fraction = float(np.clip(fraction, 0.0, 1.0))
            hits.append((ray_distance, index + fraction, origin + ray_distance * direction))
    hits.sort(key=lambda item: item[0])
    return hits


def unique_ray_hits(contour: np.ndarray, origin: np.ndarray,
                    direction: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """Return geometric ray/contour hits with shared-vertex duplicates merged."""
    hits = ray_hits(contour, origin, direction)
    if not hits:
        return []
    scale = max(float(np.ptp(contour[:, 0])), float(np.ptp(contour[:, 1])), 1.0)
    tolerance = 1e-7 * scale
    unique = [hits[0]]
    for hit in hits[1:]:
        if np.linalg.norm(hit[2] - unique[-1][2]) <= tolerance:
            # A ray through a polygon vertex is reported by both adjacent
            # segments. Preserve one exact cyclic position and one point.
            continue
        unique.append(hit)
    return unique


def arc_forward(contour: np.ndarray, start: tuple[float, float, np.ndarray], end: tuple[float, float, np.ndarray]) -> np.ndarray:
    """Ordered cyclic arc from an exact start intersection to end."""
    _, start_pos, start_point = start
    _, end_pos, end_point = end
    length = len(contour)
    end_unwrapped = end_pos
    if end_unwrapped <= start_pos:
        end_unwrapped += length
    indices = np.arange(int(np.floor(start_pos)) + 1, int(np.floor(end_unwrapped)) + 1)
    middle = contour[indices % length] if len(indices) else np.empty((0, 2))
    return np.vstack((start_point, middle, end_point))


def arc_camera_distance(points: np.ndarray, camera_xy: np.ndarray) -> float:
    """Arc-length-weighted camera distance, insensitive to contour sampling."""
    if len(points) < 2:
        return float("inf")
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    total = float(lengths.sum())
    if total <= 1e-12:
        return float(np.linalg.norm(points[0] - camera_xy))
    midpoint_distance = np.linalg.norm((points[:-1] + points[1:]) * 0.5 - camera_xy, axis=1)
    return float(np.dot(lengths, midpoint_distance) / total)


def camera_visible_arc(contour: np.ndarray, camera_xy: np.ndarray, left_angle: float, right_angle: float) -> tuple[np.ndarray, dict]:
    """Clip a closed contour at the two camera rays and retain its near arc."""
    left_direction = np.array((np.cos(left_angle), np.sin(left_angle)))
    right_direction = np.array((np.cos(right_angle), np.sin(right_angle)))
    left_hits = ray_hits(contour, camera_xy, left_direction)
    right_hits = ray_hits(contour, camera_xy, right_direction)
    if not left_hits or not right_hits:
        return np.empty((0, 2)), {"valid": False, "left_hits": len(left_hits), "right_hits": len(right_hits)}
    # The first hit on each ray bounds the camera-facing outline. The two hits
    # split the closed contour into two arcs; retain the arc closer to camera.
    left, right = left_hits[0], right_hits[0]
    first = arc_forward(contour, left, right)
    second = arc_forward(contour, right, left)
    first_distance = arc_camera_distance(first, camera_xy)
    second_distance = arc_camera_distance(second, camera_xy)
    selected = first if first_distance <= second_distance else second[::-1]
    return selected, {
        "valid": True,
        "left_hits": len(left_hits),
        "right_hits": len(right_hits),
        "left_near_distance_m": float(left[0]),
        "right_near_distance_m": float(right[0]),
        "selected_points": int(len(selected)),
        "alternate_median_camera_distance_m": max(first_distance, second_distance),
        "selected_median_camera_distance_m": min(first_distance, second_distance),
    }


def camera_angular_near_edge(contour: np.ndarray, camera_xy: np.ndarray, bins: int) -> tuple[np.ndarray, dict]:
    """Retain the nearest contour sample in each camera-centred angular bin.

    This is the camera-facing radial envelope of a closed 2D contour.  Unlike
    the legacy two-ray clip it does not require the vehicle's extreme rays to
    intersect the observed shadow contour, so the same visibility operator can
    be applied independently to both observed and predicted boundaries.
    """
    if len(contour) < 8:
        return np.empty((0, 2)), {"valid": False, "reason": "contour_too_small", "input_points": int(len(contour))}
    vectors = contour - camera_xy
    radii = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(radii) & (radii > 1e-6)
    vectors, radii, contour = vectors[valid], radii[valid], contour[valid]
    if len(contour) < 8:
        return np.empty((0, 2)), {"valid": False, "reason": "too_few_finite_points", "input_points": int(len(contour))}
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi)
    ordered = np.sort(angles)
    circular_gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    seam = float(ordered[(int(np.argmax(circular_gaps)) + 1) % len(ordered)])
    unwrapped = np.mod(angles - seam, 2.0 * np.pi)
    span = float(np.max(unwrapped) - np.min(unwrapped))
    if span <= 1e-9:
        return np.empty((0, 2)), {"valid": False, "reason": "zero_angular_span", "input_points": int(len(contour))}
    bin_count = max(8, min(int(bins), len(contour)))
    bin_ids = np.minimum((unwrapped / span * bin_count).astype(np.int32), bin_count - 1)
    selected_indices = []
    for bin_id in np.unique(bin_ids):
        members = np.flatnonzero(bin_ids == bin_id)
        selected_indices.append(int(members[np.argmin(radii[members])]))
    selected_indices = np.asarray(selected_indices, dtype=np.int32)
    selected_indices = selected_indices[np.argsort(unwrapped[selected_indices])]
    selected = contour[selected_indices]
    audit = {
        "valid": len(selected) >= 8,
        "operator": "angular_near_edge",
        "input_points": int(len(contour)),
        "requested_bins": int(bins),
        "effective_bins": int(bin_count),
        "occupied_bins": int(len(selected)),
        "selected_points": int(len(selected)),
        "angular_span_deg": float(np.rad2deg(span)),
        "seam_angle_deg": float(np.rad2deg(seam)),
        "median_camera_distance_m": float(np.median(radii[selected_indices])),
    }
    return selected if audit["valid"] else np.empty((0, 2)), audit


def visible_edge(contour: np.ndarray, frame: dict, mode: str, angular_bins: int) -> tuple[np.ndarray, dict]:
    if mode == "ray_arc":
        return camera_visible_arc(contour, frame["camera_xy"], frame["left_angle"], frame["right_angle"])
    if mode == "angular_near_edge":
        return camera_angular_near_edge(contour, frame["camera_xy"], angular_bins)
    raise ValueError(f"unsupported visibility mode: {mode}")


def contour_components_xy(binary: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                          max_components: int = 2,
                          min_relative_area: float = 0.50) -> tuple[list[np.ndarray], list[dict]]:
    """Keep only contract-sized external components, largest first.

    Depth lifting and Gaussian rasterisation can split an upstream component
    into tiny islands. Reapply the two-component/50%-of-largest gate on the
    actual fitting raster so numerical islands cannot become scored arcs.
    """
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_area = float(cv2.contourArea(ranked[0])) if ranked else 0.0
    output, audit = [], []
    for rank, contour in enumerate(ranked, 1):
        area = float(cv2.contourArea(contour))
        relative_area = area / largest_area if largest_area > 0 else 0.0
        retained = (rank <= max_components and relative_area >= min_relative_area
                    and len(contour) >= 8)
        audit.append({"rank": rank, "area_pixels": area,
                      "relative_to_largest": relative_area,
                      "contour_points": int(len(contour)), "retained": retained})
        if not retained:
            continue
        pixels = contour[:, 0, :].astype(np.float64)
        x = lo[0] + pixels[:, 0] / (binary.shape[1] - 1) * (hi[0] - lo[0])
        y = hi[1] - pixels[:, 1] / (binary.shape[0] - 1) * (hi[1] - lo[1])
        output.append(np.c_[x, y])
    return output, audit


def camera_tangent_near_arc(contour: np.ndarray, camera_xy: np.ndarray) -> tuple[np.ndarray, dict]:
    """Find the two camera-cone tangencies and retain the intervening near arc."""
    if len(contour) < 8:
        return np.empty((0, 2)), {"valid": False, "reason": "contour_too_small"}
    vectors = contour - camera_xy
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2*np.pi)
    ordered = np.sort(angles)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2*np.pi])
    seam = float(ordered[(int(np.argmax(gaps)) + 1) % len(ordered)])
    unwrapped = np.mod(angles - seam, 2*np.pi)
    left, right = int(np.argmin(unwrapped)), int(np.argmax(unwrapped))

    def cyclic(start: int, end: int) -> np.ndarray:
        ids = np.arange(start, end + 1) if end >= start else np.r_[np.arange(start, len(contour)), np.arange(end + 1)]
        return contour[ids]

    first, second = cyclic(left, right), cyclic(right, left)[::-1]
    first_distance = arc_camera_distance(first, camera_xy)
    second_distance = arc_camera_distance(second, camera_xy)
    selected = first if first_distance <= second_distance else second
    return selected, {"valid": len(selected) >= 8, "operator": "component_tangent_near_arc",
        "input_points": int(len(contour)), "selected_points": int(len(selected)),
        "left_tangent_xy": contour[left].tolist(), "right_tangent_xy": contour[right].tolist(),
        "angular_span_deg": float(np.rad2deg(unwrapped[right]-unwrapped[left])),
        "selected_median_camera_distance_m": min(first_distance, second_distance),
        "alternate_median_camera_distance_m": max(first_distance, second_distance)}


def maximal_camera_cone_middle_arc(contour: np.ndarray,
                                   camera_xy: np.ndarray) -> tuple[np.ndarray, dict]:
    """Take the camera-facing contour between the two maximal cone rays.

    The two rays are the angular support boundaries of this *individual*
    closed contour as seen from the real camera location.  Their exact
    intersections, rather than nearest sampled indices or vehicle-derived
    rays, split the contour into two arcs.  The complete nearer arc is kept;
    no footprint-distance trimming or longest-run truncation is applied.
    """
    if len(contour) < 3:
        return np.empty((0, 2)), {"valid": False, "reason": "contour_too_small"}
    contour = np.asarray(contour, dtype=np.float64)
    if np.linalg.norm(contour[0] - contour[-1]) < 1e-10:
        contour = contour[:-1]
    vectors = contour - camera_xy
    radii = np.linalg.norm(vectors, axis=1)
    finite = np.isfinite(contour).all(axis=1) & np.isfinite(radii) & (radii > 1e-8)
    contour, vectors, radii = contour[finite], vectors[finite], radii[finite]
    if len(contour) < 3:
        return np.empty((0, 2)), {"valid": False, "reason": "too_few_finite_points"}

    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi)
    ordered = np.sort(angles)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    seam = float(ordered[(int(np.argmax(gaps)) + 1) % len(ordered)])
    unwrapped = np.mod(angles - seam, 2.0 * np.pi)
    left_index = int(np.argmin(unwrapped))
    right_index = int(np.argmax(unwrapped))
    left_angle = float(angles[left_index])
    right_angle = float(angles[right_index])

    def boundary_hit(angle: float, fallback_index: int) -> tuple[list, tuple, bool]:
        direction = np.array((np.cos(angle), np.sin(angle)), dtype=np.float64)
        hits = unique_ray_hits(contour, camera_xy, direction)
        used_fallback = False
        if not hits:
            # The support angle is defined by a contour sample. Floating point
            # cancellation can still miss the exact segment intersection.
            point = contour[fallback_index]
            hits = [(float(np.linalg.norm(point - camera_xy)),
                     float(fallback_index), point.copy())]
            used_fallback = True
        return hits, hits[0], used_fallback

    left_hits, left, left_fallback = boundary_hit(left_angle, left_index)
    right_hits, right, right_fallback = boundary_hit(right_angle, right_index)
    hit_contract = 1 <= len(left_hits) <= 2 and 1 <= len(right_hits) <= 2
    if not hit_contract:
        return np.empty((0, 2)), {
            "valid": False, "reason": "boundary_ray_hit_contract_failed",
            "left_hit_count": len(left_hits), "right_hit_count": len(right_hits),
            "left_angle_deg": float(np.rad2deg(left_angle)),
            "right_angle_deg": float(np.rad2deg(right_angle)),
        }

    first = arc_forward(contour, left, right)
    second = arc_forward(contour, right, left)
    first_distance = arc_camera_distance(first, camera_xy)
    second_distance = arc_camera_distance(second, camera_xy)
    selected = first if first_distance <= second_distance else second[::-1]
    angular_span = float(np.max(unwrapped) - np.min(unwrapped))
    return selected, {
        "valid": len(selected) >= 3,
        "operator": "maximal_camera_cone_middle_arc",
        "input_points": int(len(contour)),
        "selected_points": int(len(selected)),
        "left_angle_deg": float(np.rad2deg(left_angle)),
        "right_angle_deg": float(np.rad2deg(right_angle)),
        "maximal_cone_span_deg": float(np.rad2deg(angular_span)),
        "left_hit_count": int(len(left_hits)),
        "right_hit_count": int(len(right_hits)),
        "left_hits_xy": [hit[2].tolist() for hit in left_hits],
        "right_hits_xy": [hit[2].tolist() for hit in right_hits],
        "selected_left_hit_xy": left[2].tolist(),
        "selected_right_hit_xy": right[2].tolist(),
        "left_numeric_fallback": left_fallback,
        "right_numeric_fallback": right_fallback,
        "hit_contract_1_or_2_each": hit_contract,
        "selected_median_camera_distance_m": min(first_distance, second_distance),
        "alternate_median_camera_distance_m": max(first_distance, second_distance),
        "scored_arc_length_m": polyline_length(selected),
    }


def longest_true_run(points: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return the longest continuous retained section of an ordered open arc."""
    best = np.empty((0, 2), dtype=points.dtype)
    start = None
    for index, value in enumerate(np.r_[keep, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = points[start:index]
            if len(candidate) > len(best):
                best = candidate
            start = None
    return best


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def component_tangent_edges(binary: np.ndarray, frame: dict,
                            contact_exclusion_m: float = 0.0) -> tuple[list[np.ndarray], dict]:
    components, component_audit = contour_components_xy(binary, frame["lo"], frame["hi"])
    arcs, audits = [], []
    floor_boundary = contour_xy(frame["floor"], frame["lo"], frame["hi"])
    floor_tree = cKDTree(floor_boundary) if contact_exclusion_m > 0 and len(floor_boundary) else None
    for component in components:
        arc, audit = camera_tangent_near_arc(component, frame["camera_xy"])
        if floor_tree is not None and len(arc):
            distances = floor_tree.query(arc, k=1, workers=-1)[0]
            filtered = longest_true_run(arc, distances > contact_exclusion_m)
            audit.update({"contact_exclusion_m": float(contact_exclusion_m),
                          "points_before_contact_exclusion": int(len(arc)),
                          "points_after_contact_exclusion": int(len(filtered)),
                          "minimum_floor_boundary_distance_m": float(np.min(distances))})
            arc = filtered
        audit["scored_arc_length_m"] = polyline_length(arc)
        audits.append(audit)
        if len(arc) >= 8:
            arcs.append(arc)
    return arcs, {"valid": bool(arcs), "operator": "component_tangent_near_arc",
                  "input_components": len(component_audit), "contract_components": len(components),
                  "retained_components": len(arcs),
                  "contact_exclusion_m": float(contact_exclusion_m),
                  "component_gate": {"max_components": 2, "min_relative_area": 0.50,
                                     "components": component_audit},
                  "selected_points": int(sum(len(arc) for arc in arcs)), "components": audits}


def component_camera_cone_middle_edges(binary: np.ndarray,
                                       frame: dict,
                                       trim_image_edge: bool = False) -> tuple[list[np.ndarray], dict]:
    """Apply the same maximal-camera-cone contract to each retained island."""
    components, component_audit = contour_components_xy(binary, frame["lo"], frame["hi"])
    arcs, audits = [], []
    for component in components:
        arc, audit = maximal_camera_cone_middle_arc(component, frame["camera_xy"])
        if audit.get("valid") and len(arc) >= 3 and trim_image_edge:
            world = (frame["anchor"][None] + arc[:, 0, None] * frame["e1"][None]
                     + arc[:, 1, None] * frame["e2"][None])
            w2c = np.linalg.inv(frame["camera_to_world"])
            camera = world @ w2c[:3, :3].T + w2c[:3, 3]
            k = frame["intrinsics"]
            pixels = np.c_[k[0, 0] * camera[:, 0] / camera[:, 2] + k[0, 2],
                           k[1, 1] * camera[:, 1] / camera[:, 2] + k[1, 2]]
            h_img, w_img = frame["source_image_shape"]
            margin = 8.0
            keep = ((camera[:, 2] > 1e-5) & (pixels[:, 0] > margin) &
                    (pixels[:, 0] < w_img - 1 - margin) & (pixels[:, 1] > margin) &
                    (pixels[:, 1] < h_img - 1 - margin))
            # Edge contact is expected at arc ends. If an interior section is
            # also touched, split there rather than drawing across it.
            runs = []
            start = None
            for i, value in enumerate(np.r_[keep, False]):
                if value and start is None: start = i
                elif not value and start is not None:
                    if i - start >= 3: runs.append(arc[start:i])
                    start = None
            audit.update({"image_edge_trim_pixels": margin,
                          "points_before_image_edge_trim": int(len(arc)),
                          "points_after_image_edge_trim": int(sum(len(x) for x in runs)),
                          "image_edge_points_removed": int((~keep).sum())})
            audit["selected_points"] = int(sum(len(x) for x in runs))
            audit["scored_arc_length_m"] = float(sum(polyline_length(x) for x in runs))
            arcs.extend(runs)
        elif audit.get("valid") and len(arc) >= 3:
            arcs.append(arc)
        audits.append(audit)
    return arcs, {
        "valid": bool(arcs),
        "operator": "maximal_camera_cone_middle_arc",
        "input_components": len(component_audit),
        "contract_components": len(components),
        "retained_components": len(arcs),
        "component_gate": {"max_components": 2, "min_relative_area": 0.50,
                           "components": component_audit},
        "selected_points": int(sum(len(arc) for arc in arcs)),
        "components": audits,
    }


def add_camera_contract(frame: dict, vehicle_data: np.lib.npyio.NpzFile,
                        visibility_mode: str, angular_bins: int,
                        contact_exclusion_m: float = 0.0) -> None:
    camera_world = vehicle_data["camera_to_world"][:3, 3].astype(float)
    frame["camera_to_world"] = vehicle_data["camera_to_world"].astype(float)
    frame["intrinsics"] = vehicle_data["intrinsics"].astype(float)
    frame["camera_xy"] = plane_xy(camera_world[None], frame["e1"], frame["e2"], frame["anchor"])[0]
    vectors = frame["vxy"] - frame["camera_xy"]
    center_angle = float(np.arctan2(np.median(vectors[:, 1]), np.median(vectors[:, 0])))
    relative = wrap(np.arctan2(vectors[:, 1], vectors[:, 0]) - center_angle)
    # Robust geometric extremes avoid allowing one isolated Gaussian centre to
    # rotate a camera side ray, while retaining 99.8% of selected geometry.
    low, high = np.quantile(relative, (0.001, 0.999))
    frame["left_angle"] = center_angle + float(low)
    frame["right_angle"] = center_angle + float(high)
    observed_full = contour_xy(frame["obs_mask"], frame["lo"], frame["hi"])
    if visibility_mode == "component_tangent_arcs":
        observed_arcs, audit = component_tangent_edges(
            frame["obs_mask"], frame, contact_exclusion_m)
        observed_arc = np.vstack(observed_arcs) if observed_arcs else np.empty((0, 2))
    elif visibility_mode == "camera_cone_middle":
        observed_arcs, audit = component_camera_cone_middle_edges(
            frame["obs_mask"], frame, trim_image_edge=frame.get("trim_image_edge", False))
        observed_arc = np.vstack(observed_arcs) if observed_arcs else np.empty((0, 2))
    else:
        observed_arc, audit = visible_edge(observed_full, frame, visibility_mode, angular_bins)
        observed_arcs = [observed_arc] if len(observed_arc) else []
    if len(observed_arc) < 8:
        raise RuntimeError(f"camera visibility operator did not yield a usable observed edge: {audit}")
    frame["observed_full_boundary"] = observed_full
    frame["obs_boundary"] = observed_arc
    frame["observed_arcs"] = observed_arcs
    frame["observed_arc_audit"] = audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle-geometry", nargs="+", type=Path, required=True)
    parser.add_argument("--shadow-geometry", nargs="+", type=Path, required=True)
    parser.add_argument("--shadow-mask", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raster-size", type=int, default=900)
    parser.add_argument("--max-vehicle-points", type=int, default=100000)
    parser.add_argument("--coarse-az-step", type=int, default=5)
    parser.add_argument("--coarse-elev-min", type=int, default=20)
    parser.add_argument("--coarse-elev-max", type=int, default=70)
    parser.add_argument("--local-radius-deg", type=int, default=5)
    parser.add_argument("--local-basins", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--visibility-mode", choices=("ray_arc", "angular_near_edge", "component_tangent_arcs", "camera_cone_middle"), default="ray_arc")
    parser.add_argument("--angular-bins", type=int, default=512)
    parser.add_argument("--no-observed-floor-subtraction", action="store_true")
    parser.add_argument("--no-predicted-floor-subtraction", action="store_true",
                        help="keep the full projected vehicle support; do not cut it by the floor footprint")
    parser.add_argument("--exclude-image-edge-components", action="store_true",
                        help="exclude shadow components touching any source-image edge")
    parser.add_argument("--contact-exclusion-m", type=float, default=0.0)
    args = parser.parse_args()
    counts = {len(args.vehicle_geometry), len(args.shadow_geometry), len(args.shadow_mask)}
    if len(counts) != 1:
        raise ValueError("vehicle geometry, shadow geometry, and shadow mask lists must have equal lengths")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for index, (vehicle_path, shadow_path, mask_path) in enumerate(zip(args.vehicle_geometry, args.shadow_geometry, args.shadow_mask)):
        vehicle_data, shadow_data = np.load(vehicle_path), np.load(shadow_path)
        frame = setup(vehicle_data, shadow_data, mask_path, args.raster_size, args.max_vehicle_points,
                      args.seed + index, observed_components=2 if args.visibility_mode in ("component_tangent_arcs", "camera_cone_middle") else 1,
                      subtract_observed_floor=not args.no_observed_floor_subtraction,
                      exclude_image_edge=False)
        frame["trim_image_edge"] = args.exclude_image_edge_components
        add_camera_contract(frame, vehicle_data, args.visibility_mode, args.angular_bins,
                            args.contact_exclusion_m)
        frames.append(frame)

    def evaluate_one(frame: dict, azimuth: float, elevation: float) -> dict | None:
        horizontal = np.array((np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))))
        cotangent = 1.0 / np.tan(np.deg2rad(elevation))
        projected = frame["vxy"] - frame["h"][:, None] * cotangent * horizontal[None]
        world_to_plane = np.c_[frame["e1"], frame["e2"]] - frame["n"][:, None] * cotangent * horizontal[None]
        raw_mask = gaussian_union_mask(projected, frame["cov"], world_to_plane, 2.0, frame["lo"], frame["hi"], frame["size"])
        # The observed shadow has already undergone source-view shadow-minus-
        # vehicle subtraction. Candidate floor subtraction is optional because
        # a Boolean difference creates a synthetic boundary identical to the
        # blue footprint; camera-cone mode normally scores the raw projection.
        predicted_mask = (raw_mask if args.no_predicted_floor_subtraction else raw_mask & ~cv2.dilate(
            frame["floor"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
        full_boundary = contour_xy(predicted_mask, frame["lo"], frame["hi"])
        if args.visibility_mode == "component_tangent_arcs":
            visible_arcs, arc_audit = component_tangent_edges(
                predicted_mask, frame, args.contact_exclusion_m)
            visible_arc = np.vstack(visible_arcs) if visible_arcs else np.empty((0, 2))
        elif args.visibility_mode == "camera_cone_middle":
            visible_arcs, arc_audit = component_camera_cone_middle_edges(predicted_mask, frame)
            visible_arc = np.vstack(visible_arcs) if visible_arcs else np.empty((0, 2))
        else:
            visible_arc, arc_audit = visible_edge(full_boundary, frame, args.visibility_mode, args.angular_bins)
            visible_arcs = [visible_arc] if len(visible_arc) else []
        if len(visible_arc) < 8:
            return None
        metrics = robust(visible_arc, frame["obs_boundary"])
        if metrics is None:
            return None
        intersection = np.count_nonzero(raw_mask & frame["obs_mask"])
        union = np.count_nonzero(raw_mask | frame["obs_mask"])
        result = dict(metrics)
        result.update({
            "azimuth_deg": float(azimuth), "elevation_deg": float(elevation),
            "raw_mask": raw_mask, "predicted_mask": predicted_mask,
            "full_boundary": full_boundary, "visible_arc": visible_arc, "visible_arcs": visible_arcs,
            "arc_audit": arc_audit,
            "iou_diagnostic": float(intersection / union) if union else 0.0,
        })
        return result

    def evaluate(azimuth: float, elevation: float) -> dict | None:
        per_frame = [evaluate_one(frame, azimuth, elevation) for frame in frames]
        if any(result is None for result in per_frame):
            return None
        return {"score": float(np.mean([result["score"] for result in per_frame])),
                "azimuth_deg": float(azimuth), "elevation_deg": float(elevation), "frames": per_frame}

    coarse = []
    for elevation in range(args.coarse_elev_min, args.coarse_elev_max + 1, 5):
        for azimuth in range(0, 360, args.coarse_az_step):
            result = evaluate(azimuth, elevation)
            if result is not None:
                coarse.append(result)
    seeds = sorted(coarse, key=lambda row: row["score"], reverse=True)[:args.local_basins]
    refined = []
    for seed in seeds:
        for elevation in range(max(2, int(seed["elevation_deg"]) - args.local_radius_deg), int(seed["elevation_deg"]) + args.local_radius_deg + 1):
            for delta in range(-args.local_radius_deg, args.local_radius_deg + 1):
                result = evaluate((int(seed["azimuth_deg"]) + delta) % 360, elevation)
                if result is not None:
                    refined.append(result)
    best = max(refined or coarse, key=lambda row: row["score"])

    metric_names = ("score", "pred_to_observed_p90_m", "pred_to_observed_median_m",
                    "observed_to_predicted_p90_m", "pred_boundary_support_0p35m", "iou_diagnostic")
    for rows, filename in ((coarse, "coarse_scores.csv"), (refined, "refined_scores.csv")):
        fields = ["score", "azimuth_deg", "elevation_deg"] + [f"frame{i}_{name}" for i in range(len(frames)) for name in metric_names]
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = {"score": row["score"], "azimuth_deg": row["azimuth_deg"], "elevation_deg": row["elevation_deg"]}
                flat.update({f"frame{i}_{name}": row["frames"][i][name]
                             for i in range(len(frames)) for name in metric_names})
                writer.writerow(flat)

    def to_pixels(points: np.ndarray, frame: dict) -> np.ndarray:
        q = (points - frame["lo"]) / (frame["hi"] - frame["lo"])
        return np.rint(np.c_[q[:, 0] * 699, (1.0 - q[:, 1]) * 699]).astype(np.int32)

    def panel(frame: dict, result: dict, title: str) -> np.ndarray:
        image = np.full((700, 700, 3), 250, np.uint8)
        observed = cv2.resize(frame["obs_mask"].astype(np.uint8), (700, 700), interpolation=cv2.INTER_NEAREST).astype(bool)
        image[observed] = (90, 90, 90)
        vehicle = cv2.resize(frame["floor"].astype(np.uint8), (700, 700), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(vehicle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, contours, -1, (255, 80, 0), 2)
        # Thin pale line audits the rejected full contour. Thick red is the only
        # predicted arc used in the score; green is the fixed observed arc.
        full_px = to_pixels(result["full_boundary"], frame)
        pred_arcs_px = [to_pixels(arc, frame) for arc in result.get("visible_arcs", [result["visible_arc"]])]
        obs_arcs_px = [to_pixels(arc, frame) for arc in frame.get("observed_arcs", [frame["obs_boundary"]])]
        if len(full_px) > 1: cv2.polylines(image, [full_px], True, (185, 185, 255), 1, cv2.LINE_AA)
        for pred_px in pred_arcs_px:
            if len(pred_px) > 1: cv2.polylines(image, [pred_px], False, (0, 0, 255), 4, cv2.LINE_AA)
        # Observed evidence is drawn last so exact overlap remains visible.
        for obs_px in obs_arcs_px:
            if len(obs_px) > 1: cv2.polylines(image, [obs_px], False, (0, 210, 0), 2, cv2.LINE_AA)
        camera_px = to_pixels(frame["camera_xy"][None], frame)[0]
        extent = 30.0
        if args.visibility_mode == "ray_arc":
            for angle in (frame["left_angle"], frame["right_angle"]):
                endpoint = frame["camera_xy"] + extent * np.array((np.cos(angle), np.sin(angle)))
                cv2.line(image, tuple(camera_px), tuple(to_pixels(endpoint[None], frame)[0]), (180, 80, 180), 1, cv2.LINE_AA)
        elif args.visibility_mode == "camera_cone_middle":
            # Draw the actually solved per-component cone boundaries. Solid
            # purple audits the observed contour; pale purple audits candidate.
            for audit, color in ((frame["observed_arc_audit"], (170, 60, 170)),
                                 (result["arc_audit"], (220, 155, 220))):
                for component in audit.get("components", []):
                    for key in ("left_angle_deg", "right_angle_deg"):
                        if key not in component:
                            continue
                        angle = np.deg2rad(component[key])
                        endpoint = frame["camera_xy"] + extent * np.array((np.cos(angle), np.sin(angle)))
                        cv2.line(image, tuple(camera_px), tuple(to_pixels(endpoint[None], frame)[0]), color, 1, cv2.LINE_AA)
        cv2.putText(image, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 2, cv2.LINE_AA)
        return image

    chosen = sorted(refined or coarse, key=lambda row: row["score"], reverse=True)[:25]
    for index, frame in enumerate(frames):
        cv2.imwrite(str(args.output_dir / f"frame{index}_best_camera_visible_arc.png"),
                    panel(frame, best["frames"][index], f"frame {index} | shared az {best['azimuth_deg']:.1f} el {best['elevation_deg']:.1f}"))
        observed_arcs = frame.get("observed_arcs", [frame["obs_boundary"]])
        predicted_arcs = best["frames"][index].get("visible_arcs", [best["frames"][index]["visible_arc"]])
        def packed(arcs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            offsets = np.cumsum([0] + [len(arc) for arc in arcs]).astype(np.int32)
            points = np.vstack(arcs) if arcs else np.empty((0, 2), np.float64)
            return points, offsets
        observed_points, observed_offsets = packed(observed_arcs)
        predicted_points, predicted_offsets = packed(predicted_arcs)
        np.savez_compressed(
            args.output_dir / f"frame{index}_best_contours.npz",
            observed_points_xy=observed_points, observed_offsets=observed_offsets,
            predicted_points_xy=predicted_points, predicted_offsets=predicted_offsets,
            plane_e1=frame["e1"], plane_e2=frame["e2"], plane_anchor=frame["anchor"],
            camera_to_world=np.load(args.vehicle_geometry[index])["camera_to_world"],
            intrinsics=np.load(args.vehicle_geometry[index])["intrinsics"],
        )
    cells = [cv2.hconcat([panel(frames[i], row["frames"][i], f"az {row['azimuth_deg']:.0f} el {row['elevation_deg']:.0f} score {row['score']:.3f}") for i in range(len(frames))]) for row in chosen]
    while len(cells) < 25:
        cells.append(cells[-1])
    sheet = cv2.vconcat([cv2.hconcat(cells[index:index + 5]) for index in range(0, 25, 5)])
    cv2.imwrite(str(args.output_dir / "top25_camera_visible_arc_candidates.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])

    scalar_frame = lambda result: {key: value for key, value in result.items() if key not in ("raw_mask", "predicted_mask", "full_boundary", "visible_arc", "visible_arcs")}
    output = {
        "best": {"azimuth_deg": best["azimuth_deg"], "elevation_deg": best["elevation_deg"],
                 "joint_score": best["score"], "frames": [scalar_frame(result) for result in best["frames"]]},
        "objective": (("per-connected-component maximal camera cone boundary intersections and complete intervening near contour arcs, identically applied to observed shadow and candidate projection" if args.visibility_mode == "camera_cone_middle" else "per-connected-component camera tangent intersections and intervening near contour arcs, identically applied to observed shadow and candidate projection" if args.visibility_mode == "component_tangent_arcs" else "camera-angular nearest contour edge for both fixed observed shadow and candidate vehicle projection" if args.visibility_mode == "angular_near_edge" else "camera-visible near contour arcs between fixed left/right camera rays") + "; mean bidirectional robust contour distance across frames; IoU diagnostic only"),
        "visibility_mode": args.visibility_mode,
        "angular_bins": args.angular_bins if args.visibility_mode == "angular_near_edge" else None,
        "observed_floor_subtraction": not args.no_observed_floor_subtraction,
        "contact_exclusion_m": args.contact_exclusion_m,
        "predicted_floor_subtraction": not args.no_predicted_floor_subtraction,
        "exclude_image_edge_components": args.exclude_image_edge_components,
        "visualization": "gray=observed mask; green=scored observed camera-facing edge; blue=vehicle floor; pale-red=rejected full candidate contour; red=scored candidate camera-facing edge; purple=legacy side rays when ray_arc is selected",
        "top25_same_objective": True,
        "execution_mode": "single_frame" if len(frames) == 1 else "multi_frame_shared_angle",
        "frame_count": len(frames),
        "camera_contract": [{"camera_xy": frame["camera_xy"].tolist(), "left_angle_deg": float(np.rad2deg(frame["left_angle"])),
                             "right_angle_deg": float(np.rad2deg(frame["right_angle"])), "observed_arc": frame["observed_arc_audit"]} for frame in frames],
        "inputs": [{"vehicle_geometry": str(v.resolve()), "shadow_geometry": str(s.resolve()), "shadow_mask": str(m.resolve())}
                   for v, s, m in zip(args.vehicle_geometry, args.shadow_geometry, args.shadow_mask)],
        "search": {"coarse_az_step": args.coarse_az_step, "coarse_elevation": [args.coarse_elev_min, args.coarse_elev_max, 5],
                   "local_radius_deg": args.local_radius_deg, "local_basins": args.local_basins},
    }
    (args.output_dir / "fit_result.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
