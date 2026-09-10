#!/usr/bin/env python3
"""Detect vehicles in every Argoverse frame/camera and attach physical IDs.

YOLO supplies every detection.  The Argoverse instance projection is used only
as the dataset adapter that associates detections of the same physical vehicle
across frames/cameras; it is never used as a vehicle mask or fit geometry.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from .audit_argoverse_single_vehicle_views import (
        box_corners, camera_to_world, load_annotations, project_box,
    )
except ImportError:
    from audit_argoverse_single_vehicle_views import (
        box_corners, camera_to_world, load_annotations, project_box,
    )

COCO_VEHICLES = {2: "car", 5: "bus", 7: "truck"}


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = max(0.0, ax1-ax0)*max(0.0, ay1-ay0) + max(0.0, bx1-bx0)*max(0.0, by1-by0) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--cameras", nargs="+", type=int, default=list(range(7)))
    ap.add_argument("--confidence", type=float, default=.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-oracle-iou", type=float, default=.20)
    ap.add_argument("--min-yolo-area-ratio", type=float, default=.003)
    ap.add_argument("--border-margin", type=int, default=8)
    args = ap.parse_args(); started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    annotations, present = load_annotations(args.data_root)
    frame_ids = sorted(int(p.stem) for p in (args.data_root / "ego_pose").glob("*.txt"))
    intrinsics = {}
    dimensions = {}
    for camera in args.cameras:
        fx, fy, cx, cy, *_ = np.loadtxt(args.data_root / "intrinsics" / f"{camera}.txt")
        intrinsics[camera] = np.array(((fx,0,cx),(0,fy,cy),(0,0,1)), float)
        first = next((args.data_root / "images" / f"{t:03d}_{camera}.jpg" for t in frame_ids
                      if (args.data_root / "images" / f"{t:03d}_{camera}.jpg").is_file()), None)
        image = cv2.imread(str(first), cv2.IMREAD_COLOR) if first else None
        if image is None: raise RuntimeError(f"no readable image for camera {camera}")
        dimensions[camera] = (image.shape[1], image.shape[0])

    oracle_by_view = defaultdict(list)
    for timestep in frame_ids:
        for camera in args.cameras:
            width, height = dimensions[camera]
            w2c = np.linalg.inv(camera_to_world(args.data_root, timestep, camera))
            for instance_id in present.get(timestep, ()):
                pose, size = annotations[instance_id]["frames"][timestep]
                corners = box_corners(size)
                world = (pose @ np.c_[corners, np.ones(8)].T).T[:, :3]
                projected = project_box(world, intrinsics[camera], w2c, width, height)
                if projected is None: continue
                _, bbox, area = projected
                oracle_by_view[(timestep, camera)].append({"instance_id": instance_id,
                    "class_name": annotations[instance_id]["class_name"], "bbox_xyxy": list(bbox),
                    "box_area_ratio": float(area)})

    inputs = []
    for timestep in frame_ids:
        for camera in args.cameras:
            image = args.data_root / "images" / f"{timestep:03d}_{camera}.jpg"
            road = args.data_root / "road_masks" / f"{timestep:03d}_{camera}.png"
            if image.is_file() and road.is_file(): inputs.append(image)
    model = YOLO(str(args.weights))
    matched = []
    raw_count = 0
    for result in model.predict(source=[str(p) for p in inputs], stream=True, device=args.device,
                                classes=sorted(COCO_VEHICLES), conf=args.confidence,
                                imgsz=args.imgsz, batch=args.batch, verbose=False):
        path = Path(result.path); ts, cam = map(int, path.stem.split("_")); h, w = result.orig_shape
        by_instance = {}
        for xyxy, conf, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            raw_count += 1; box = list(map(float, xyxy)); area = (box[2]-box[0])*(box[3]-box[1])/float(w*h)
            touches = box[0] <= args.border_margin or box[1] <= args.border_margin or box[2] >= w-1-args.border_margin or box[3] >= h-1-args.border_margin
            if touches or area < args.min_yolo_area_ratio: continue
            candidates = oracle_by_view[(ts, cam)]
            oracle = max(candidates, key=lambda q: iou(box, q["bbox_xyxy"]), default=None)
            overlap = iou(box, oracle["bbox_xyxy"]) if oracle else 0.0
            if oracle is None or overlap < args.min_oracle_iou: continue
            record = {"instance_id": oracle["instance_id"], "oracle_class_name": oracle["class_name"],
                      "timestep": ts, "camera": cam, "image": str(path.resolve()),
                      "bbox_xyxy": box, "yolo_class_id": int(class_id),
                      "yolo_class_name": COCO_VEHICLES[int(class_id)], "yolo_confidence": float(conf),
                      "yolo_area_ratio": float(area), "oracle_iou": float(overlap)}
            old = by_instance.get(oracle["instance_id"])
            if old is None or (overlap, conf) > (old["oracle_iou"], old["yolo_confidence"]):
                by_instance[oracle["instance_id"]] = record
        matched.extend(by_instance.values())
    matched.sort(key=lambda q: (q["timestep"], q["camera"], q["instance_id"]))
    instances = defaultdict(int)
    for record in matched: instances[record["instance_id"]] += 1
    output = {"contract": "all frames and requested cameras; YOLO detections grouped by Argoverse physical instance ID; no same-frame top3 constraint",
              "identity_adapter": "Argoverse projected instance boxes used only for cross-frame/camera physical identity association",
              "frame_count": len(frame_ids), "camera_ids": args.cameras, "image_count": len(inputs),
              "raw_yolo_detections": raw_count, "matched_eligible_detections": len(matched),
              "physical_instances_considered": len(instances),
              "thresholds": {"confidence": args.confidence, "min_yolo_area_ratio": args.min_yolo_area_ratio,
                             "min_oracle_iou": args.min_oracle_iou, "border_margin_px": args.border_margin},
              "detections": matched, "views_per_instance": dict(instances),
              "elapsed_s": time.perf_counter()-started}
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({k: output[k] for k in ("frame_count","image_count","raw_yolo_detections","matched_eligible_detections","physical_instances_considered","elapsed_s")}, indent=2))

if __name__ == "__main__": main()
