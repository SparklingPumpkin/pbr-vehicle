#!/usr/bin/env python3
"""Fail-closed shadow evidence gates from MTMT probability and local masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return image > 0


def boundary(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)) > 0


def distance_to(edge: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probability", type=Path, required=True)
    ap.add_argument("--shadow-mask", type=Path, required=True)
    ap.add_argument("--vehicle-mask", type=Path, required=True)
    ap.add_argument("--road-mask", type=Path, required=True)
    ap.add_argument("--source-image", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--low-probability", type=float, default=.35)
    ap.add_argument("--high-probability", type=float, default=.65)
    ap.add_argument("--roi-scale", type=float, default=3.0)
    ap.add_argument("--min-natural-edge-pixels", type=int, default=64)
    ap.add_argument("--min-levelset-coverage", type=float, default=.60)
    ap.add_argument("--max-blur-p90-height-ratio", type=float, default=.050)
    ap.add_argument("--min-edge-contrast", type=float, default=.20)
    ap.add_argument("--max-road-shadow-ratio", type=float, default=.65)
    ap.add_argument("--max-vehicle-fraction", type=float, default=.20)
    ap.add_argument("--min-shadow-vehicle-ratio", type=float, default=3.0)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    probability = np.load(args.probability).astype(np.float32)
    shape = probability.shape
    shadow = read_mask(args.shadow_mask, shape)
    vehicle = read_mask(args.vehicle_mask, shape)
    road = read_mask(args.road_mask, shape)
    source = cv2.imread(str(args.source_image), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(args.source_image)
    if source.shape[:2] != shape:
        source = cv2.resize(source, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

    ys, xs = np.nonzero(vehicle)
    if not len(xs):
        raise RuntimeError("vehicle mask is empty")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    vehicle_height = max(1, y1 - y0); vehicle_width = max(1, x1 - x0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rw, rh = args.roi_scale * vehicle_width, args.roi_scale * vehicle_height
    rx0, rx1 = max(0, int(cx-rw/2)), min(shape[1], int(cx+rw/2))
    ry0, ry1 = max(0, int(cy-rh/2)), min(shape[0], int(cy+rh/2))
    roi = np.zeros(shape, bool); roi[ry0:ry1, rx0:rx1] = True

    # Only natural MTMT boundaries are eligible for sharpness: exclude seams
    # introduced by vehicle subtraction, road-mask clipping, and image borders.
    exclusion_px = max(5, int(round(.02 * vehicle_height)))
    vehicle_near = cv2.dilate(vehicle.astype(np.uint8),
                              np.ones((2*exclusion_px+1, 2*exclusion_px+1), np.uint8)) > 0
    road_edge_near = cv2.dilate(boundary(road).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    border = np.zeros(shape, bool); border[:5] = border[-5:] = True; border[:, :5] = border[:, -5:] = True
    natural_edge = boundary(shadow) & roi & ~vehicle_near & ~road_edge_near & ~border

    low_edge = boundary(probability >= args.low_probability)
    high_edge = boundary(probability >= args.high_probability)
    dl, dh = distance_to(low_edge), distance_to(high_edge)
    max_search_px = max(8.0, .20 * vehicle_height)
    matched = natural_edge & (dl <= max_search_px) & (dh <= max_search_px)
    widths = (dl + dh)[matched]
    edge_count = int(natural_edge.sum())
    coverage = float(matched.sum() / edge_count) if edge_count else 0.0
    blur_median = float(np.median(widths) / vehicle_height) if len(widths) else float("inf")
    blur_p90 = float(np.quantile(widths, .90) / vehicle_height) if len(widths) else float("inf")

    inner_distance = cv2.distanceTransform(shadow.astype(np.uint8), cv2.DIST_L2, 5)
    outer_distance = cv2.distanceTransform((~shadow).astype(np.uint8), cv2.DIST_L2, 5)
    ring_hi = max(6, int(round(.04 * vehicle_height)))
    inner = shadow & roi & ~vehicle_near & (inner_distance >= 2) & (inner_distance <= ring_hi)
    outer = ~shadow & roi & ~vehicle_near & (outer_distance >= 2) & (outer_distance <= ring_hi)
    inside_probability = float(np.median(probability[inner])) if inner.any() else 0.0
    outside_probability = float(np.median(probability[outer])) if outer.any() else 1.0
    contrast = inside_probability - outside_probability

    road_pixels = int((road & roi).sum())
    vehicle_pixels = int((vehicle & roi).sum())
    shadow_road_pixels = int((shadow & road & roi).sum())
    denominator = int(((road | vehicle) & roi).sum())
    road_shadow_ratio = shadow_road_pixels / road_pixels if road_pixels else 1.0
    vehicle_fraction = vehicle_pixels / denominator if denominator else 0.0
    shadow_vehicle_ratio = shadow_road_pixels / vehicle_pixels if vehicle_pixels else float("inf")
    excessive_road_shadow = (road_shadow_ratio > args.max_road_shadow_ratio and
                             vehicle_fraction < args.max_vehicle_fraction and
                             shadow_vehicle_ratio > args.min_shadow_vehicle_ratio)

    failures = []
    if edge_count < args.min_natural_edge_pixels: failures.append("insufficient_natural_edge")
    if coverage < args.min_levelset_coverage: failures.append("insufficient_levelset_coverage")
    if blur_p90 > args.max_blur_p90_height_ratio: failures.append("probability_edge_too_blurry")
    if contrast < args.min_edge_contrast: failures.append("insufficient_inside_outside_contrast")
    if excessive_road_shadow: failures.append("excessive_local_road_shadow_for_vehicle_scale")
    accepted = not failures

    heat = cv2.applyColorMap(np.rint(np.clip(probability, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = source.copy(); overlay[shadow] = (.45*overlay[shadow] + .55*np.array((0,0,255))).astype(np.uint8)
    levels = source.copy()
    for mask, color in ((low_edge, (0,255,255)), (high_edge, (255,0,255)), (natural_edge, (0,255,0))):
        levels[mask] = color
    region = source.copy(); region[road & roi] = (.6*region[road & roi] + .4*np.array((255,0,0))).astype(np.uint8); region[vehicle] = (255,255,0); region[shadow & road & roi] = (0,0,255)
    cv2.rectangle(region, (rx0,ry0), (max(rx0,rx1-1),max(ry0,ry1-1)), (255,255,255), 3)
    panels=[]
    for image, title in ((overlay,"cleaned shadow"),(heat,"MTMT probability"),(levels,"green=evaluated edge; yellow=.35; magenta=.65"),(region,"local ROI: blue=road cyan=vehicle red=shadow-road")):
        image=image.copy(); cv2.rectangle(image,(0,0),(image.shape[1],48),(0,0,0),-1); cv2.putText(image,title,(12,33),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA); panels.append(cv2.resize(image,(768,581),interpolation=cv2.INTER_AREA))
    review=cv2.vconcat([cv2.hconcat(panels[:2]),cv2.hconcat(panels[2:])]); cv2.putText(review,"ACCEPT" if accepted else "REJECT: "+",".join(failures),(12,review.shape[0]-18),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,180,0) if accepted else (0,0,255),2,cv2.LINE_AA)
    cv2.imwrite(str(args.output_dir/"shadow_evidence_review.jpg"),review,[cv2.IMWRITE_JPEG_QUALITY,96])
    result={"accepted":accepted,"failure_reasons":failures,
            "probability_edge":{"low":args.low_probability,"high":args.high_probability,
                "natural_edge_pixels":edge_count,"levelset_coverage":coverage,
                "blur_median_vehicle_height_ratio":blur_median,"blur_p90_vehicle_height_ratio":blur_p90,
                "inside_probability_median":inside_probability,"outside_probability_median":outside_probability,
                "contrast":contrast},
            "local_area":{"roi_xyxy":[rx0,ry0,rx1,ry1],"roi_scale":args.roi_scale,
                "road_shadow_ratio":road_shadow_ratio,"vehicle_fraction":vehicle_fraction,
                "shadow_vehicle_ratio":shadow_vehicle_ratio,"excessive_road_shadow":excessive_road_shadow},
            "thresholds":{"min_natural_edge_pixels":args.min_natural_edge_pixels,
                "min_levelset_coverage":args.min_levelset_coverage,
                "max_blur_p90_height_ratio":args.max_blur_p90_height_ratio,
                "min_edge_contrast":args.min_edge_contrast,
                "max_road_shadow_ratio":args.max_road_shadow_ratio,
                "max_vehicle_fraction":args.max_vehicle_fraction,
                "min_shadow_vehicle_ratio":args.min_shadow_vehicle_ratio},
            "inputs":{"probability":str(args.probability.resolve()),"shadow_mask":str(args.shadow_mask.resolve()),
                "vehicle_mask":str(args.vehicle_mask.resolve()),"road_mask":str(args.road_mask.resolve()),
                "source_image":str(args.source_image.resolve())},"review":"shadow_evidence_review.jpg"}
    (args.output_dir/"shadow_evidence.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
