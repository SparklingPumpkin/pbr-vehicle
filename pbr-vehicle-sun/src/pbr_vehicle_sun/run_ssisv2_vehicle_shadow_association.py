#!/usr/bin/env python3
"""Run official SSISv2 and bind one associated shadow to a SAM2 vehicle mask.

SSISv2 returns object and shadow instances in pairs through
``pred_associations``.  This adapter deliberately keeps vehicle identity from
the SSE YOLO/SAM2 stage: among all SSIS associations, it chooses the member
with the largest IoU to that selected vehicle mask, then returns the other,
opposite-class member of the same association as the only shadow candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .device import resolve_device
except ImportError:
    from device import resolve_device


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = .52) -> np.ndarray:
    result = image.copy()
    result[mask] = ((1.0 - alpha) * result[mask] + alpha * np.asarray(color)).astype(np.uint8)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssis-root", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source-image", type=Path, required=True)
    ap.add_argument("--vehicle-mask", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--confidence-threshold", type=float, default=.10)
    ap.add_argument("--minimum-object-iou", type=float, default=.05)
    ap.add_argument("--object-class", type=int, default=0,
                    help="Official SSISv2 semantic class used for the casting object; paired opposite class is the shadow.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    compute_device = resolve_device(args.device)
    print(f"SSISv2 compute device: {compute_device.description}", flush=True)
    root = args.ssis_root.resolve()
    sys.path.insert(0, str(root))
    from adet.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor

    source = cv2.imread(str(args.source_image), cv2.IMREAD_COLOR)
    vehicle_u8 = cv2.imread(str(args.vehicle_mask), cv2.IMREAD_GRAYSCALE)
    if source is None or vehicle_u8 is None:
        raise FileNotFoundError("source image and vehicle mask are required")
    vehicle = vehicle_u8 > 0
    if source.shape[:2] != vehicle.shape:
        raise ValueError("source image and vehicle mask must have equal H/W")

    cfg = get_cfg()
    cfg.merge_from_file(str(root / "configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml"))
    cfg.MODEL.WEIGHTS = str(args.weights.resolve())
    cfg.MODEL.DEVICE = compute_device.torch
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.freeze()
    output = DefaultPredictor(cfg)(source)[0]["instances"].to("cpu")
    masks = output.pred_masks.numpy().astype(bool)
    classes = np.asarray(output.pred_classes.numpy(), dtype=np.int64)
    scores = np.asarray(output.scores.numpy(), dtype=np.float64)
    boxes = np.asarray(output.pred_boxes.tensor.numpy(), dtype=np.float64)
    association_ids = np.asarray(output.pred_associations, dtype=np.int64)
    ious = np.asarray([mask_iou(mask, vehicle) for mask in masks], dtype=np.float64)

    candidates = []
    for association_id in sorted(set(association_ids.tolist())):
        members = np.flatnonzero(association_ids == association_id)
        if len(members) != 2 or classes[members[0]] == classes[members[1]]:
            continue
        object_members = members[classes[members] == args.object_class]
        shadow_members = members[classes[members] != args.object_class]
        if len(object_members) != 1 or len(shadow_members) != 1:
            continue
        object_index = int(object_members[0])
        shadow_index = int(shadow_members[0])
        candidates.append({
            "association_id": int(association_id), "object_index": object_index,
            "shadow_index": shadow_index, "object_class": int(classes[object_index]),
            "shadow_class": int(classes[shadow_index]), "object_score": float(scores[object_index]),
            "shadow_score": float(scores[shadow_index]), "object_vehicle_iou": float(ious[object_index]),
            "object_vehicle_intersection_pixels": int(np.count_nonzero(masks[object_index] & vehicle)),
            "shadow_pixels": int(masks[shadow_index].sum()),
        })
    candidates.sort(key=lambda x: (x["object_vehicle_iou"], x["object_score"], x["shadow_score"]), reverse=True)
    selected = candidates[0] if candidates and candidates[0]["object_vehicle_iou"] >= args.minimum_object_iou else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "01_sam2_target_vehicle.png"), vehicle.astype(np.uint8) * 255)
    all_view = source.copy()
    palette = ((255, 100, 0), (0, 155, 255), (255, 0, 180), (0, 220, 0))
    for index, mask in enumerate(masks):
        all_view = overlay(all_view, mask, palette[index % len(palette)], .28)
        x1, y1, _, _ = boxes[index].astype(int)
        cv2.putText(all_view, f"{index}:c{classes[index]} a{association_ids[index]} s{scores[index]:.2f}",
                    (max(0, x1), max(26, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(all_view, f"{index}:c{classes[index]} a{association_ids[index]} s{scores[index]:.2f}",
                    (max(0, x1), max(26, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.output_dir / "02_all_ssisv2_pairs.jpg"), all_view, [cv2.IMWRITE_JPEG_QUALITY, 96])

    if selected is not None:
        obj = masks[selected["object_index"]]
        shadow = masks[selected["shadow_index"]]
        cv2.imwrite(str(args.output_dir / "03_associated_object_mask.png"), obj.astype(np.uint8) * 255)
        cv2.imwrite(str(args.output_dir / "04_associated_shadow_mask.png"), shadow.astype(np.uint8) * 255)
        review = overlay(source, shadow, (0, 0, 255), .52)
        review = overlay(review, obj, (255, 120, 0), .40)
        review = overlay(review, vehicle, (0, 255, 0), .35)
        cv2.putText(review, "red=SSISv2 associated shadow; blue=SSIS object; green=SAM2 target vehicle", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(review, "red=SSISv2 associated shadow; blue=SSIS object; green=SAM2 target vehicle", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.output_dir / "05_target_association_review.jpg"), review, [cv2.IMWRITE_JPEG_QUALITY, 96])
    manifest = {
        "method": "official SSISv2 instance shadow association",
        "binding_contract": "select SSIS association member maximizing IoU with preselected YOLO/SAM2 vehicle; use the opposite-class paired member as shadow",
        "source_image": str(args.source_image.resolve()), "vehicle_mask": str(args.vehicle_mask.resolve()),
        "weights": str(args.weights.resolve()), "confidence_threshold": args.confidence_threshold,
        "minimum_object_iou": args.minimum_object_iou, "official_object_class": args.object_class,
        "instances": int(len(masks)),
        "candidates": candidates, "selected": selected,
        "accepted": selected is not None,
        "files": {"target_vehicle": "01_sam2_target_vehicle.png", "all_pairs": "02_all_ssisv2_pairs.jpg",
                  "associated_object": "03_associated_object_mask.png", "associated_shadow": "04_associated_shadow_mask.png",
                  "review": "05_target_association_review.jpg"},
    }
    (args.output_dir / "ssisv2_association_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
