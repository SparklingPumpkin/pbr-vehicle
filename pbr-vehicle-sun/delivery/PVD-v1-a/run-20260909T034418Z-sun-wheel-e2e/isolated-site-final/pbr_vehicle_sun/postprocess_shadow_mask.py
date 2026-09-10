#!/usr/bin/env python3
"""Remove vehicle pixels and small disconnected islands from a shadow mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (min(900, result.shape[1]), 50), (0, 0, 0), -1)
    cv2.putText(result, text, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    result[mask] = (0.35 * result[mask] + 0.65 * np.asarray(color)).astype(np.uint8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-mask", type=Path, required=True)
    parser.add_argument("--vehicle-mask", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-component-area-ratio",
        type=float,
        default=0.002,
        help="Discard components smaller than this fraction of vehicle-mask area.",
    )
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument(
        "--vehicle-adjacent-only", action="store_true",
        help="Retain only components touching the vehicle boundary after subtraction.",
    )
    parser.add_argument("--adjacent-distance-px", type=float, default=1.5)
    parser.add_argument("--require-centroid-below-vehicle", action="store_true")
    parser.add_argument(
        "--second-component-min-ratio", type=float, default=0.0,
        help="Retain a second component only if its area is at least this fraction of the first.",
    )
    parser.add_argument(
        "--detector-label", default="shadow",
        help="Human-readable origin of the input shadow candidate, used only in evidence labels and manifest.",
    )
    args = parser.parse_args()

    raw = cv2.imread(str(args.shadow_mask), cv2.IMREAD_GRAYSCALE)
    vehicle = cv2.imread(str(args.vehicle_mask), cv2.IMREAD_GRAYSCALE)
    source = cv2.imread(str(args.source_image), cv2.IMREAD_COLOR)
    if raw is None or vehicle is None or source is None:
        raise FileNotFoundError("shadow mask, vehicle mask, and source image are all required")
    if raw.shape != vehicle.shape or raw.shape != source.shape[:2]:
        raise ValueError("shadow mask, vehicle mask, and source image must have identical dimensions")
    raw = raw > 0
    vehicle = vehicle > 0
    difference = raw & ~vehicle
    count, components, stats, _ = cv2.connectedComponentsWithStats(difference.astype(np.uint8), connectivity=8)
    min_area = max(1, int(round(args.min_component_area_ratio * int(vehicle.sum()))))
    vehicle_distance = cv2.distanceTransform((~vehicle).astype(np.uint8), cv2.DIST_L2, 3)
    vehicle_y = float(np.mean(np.nonzero(vehicle)[0])) if vehicle.any() else 0.0
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        component = components == index
        if area < min_area:
            continue
        minimum_distance = float(vehicle_distance[component].min())
        centroid_y = float(np.mean(np.nonzero(component)[0]))
        if args.vehicle_adjacent_only and minimum_distance > args.adjacent_distance_px:
            continue
        if args.require_centroid_below_vehicle and centroid_y < vehicle_y:
            continue
        candidates.append({"label": int(index), "area_pixels": area, "minimum_vehicle_distance_px": minimum_distance, "centroid_y": centroid_y})
    candidates.sort(key=lambda item: item["area_pixels"], reverse=True)
    selected = candidates[:1]
    if selected:
        for item in candidates[1:args.max_components]:
            if item["area_pixels"] >= args.second_component_min_ratio * selected[0]["area_pixels"]:
                selected.append(item)
    result = np.zeros_like(difference, dtype=bool)
    for item in selected:
        result |= components == item["label"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "01_raw_shadow_mask.png"), raw.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output_dir / "02_vehicle_mask.png"), vehicle.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output_dir / "03_shadow_minus_vehicle.png"), difference.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output_dir / "04_shadow_postprocessed.png"), result.astype(np.uint8) * 255)
    review = cv2.hconcat(
        [
            label(overlay(source, raw, (0, 0, 255)), f"raw {args.detector_label} shadow mask"),
            label(overlay(source, vehicle, (255, 255, 0)), "predicted vehicle mask"),
            label(overlay(source, difference, (0, 165, 255)), "shadow - vehicle"),
            label(overlay(source, result, (0, 255, 0)), "postprocessed connected shadow"),
        ]
    )
    cv2.imwrite(str(args.output_dir / "postprocess_review.jpg"), review, [cv2.IMWRITE_JPEG_QUALITY, 96])
    manifest = {
        "operation_order": [
            "binary shadow candidate", "set difference: shadow AND NOT vehicle", "8-connected components",
            "remove components below min_component_area", "retain the largest at most max_components components",
        ],
        "input_shadow_detector": args.detector_label,
        "input_shadow_pixels": int(raw.sum()),
        "vehicle_pixels": int(vehicle.sum()),
        "difference_pixels": int(difference.sum()),
        "connected_components_before_area_filter": int(count - 1),
        "min_component_area_pixels": min_area,
        "components_passing_area_filter": candidates,
        "components_retained": selected,
        "vehicle_adjacency_contract": {
            "enabled": args.vehicle_adjacent_only, "maximum_distance_px": args.adjacent_distance_px,
            "require_centroid_below_vehicle": args.require_centroid_below_vehicle,
            "second_component_min_area_ratio_to_first": args.second_component_min_ratio,
        },
        "output_shadow_pixels": int(result.sum()),
        "output_component_count": len(selected),
        "files": {
            "raw_shadow": "01_raw_shadow_mask.png",
            "vehicle": "02_vehicle_mask.png",
            "difference": "03_shadow_minus_vehicle.png",
            "postprocessed": "04_shadow_postprocessed.png",
            "review": "postprocess_review.jpg",
        },
    }
    (args.output_dir / "postprocess_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
