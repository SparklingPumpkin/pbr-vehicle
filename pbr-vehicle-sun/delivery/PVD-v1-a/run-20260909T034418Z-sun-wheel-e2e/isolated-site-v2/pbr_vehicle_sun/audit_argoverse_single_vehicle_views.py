#!/usr/bin/env python3
"""Rank same-identity Argoverse vehicle views using annotated 3D boxes.

This is an oracle audit for the single-vehicle shadow experiment.  It does not
infer a shadow or a sun angle: its output establishes which camera frames are
large, visible views of the *same* vehicle before an automatic detector is
introduced.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


VEHICLE_CLASSES = {"REGULAR_VEHICLE", "BOX_TRUCK", "SCHOOL_BUS", "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER"}
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def box_corners(size: np.ndarray) -> np.ndarray:
    length, width, height = size
    return np.array(
        [
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
        ],
        dtype=np.float64,
    )


def camera_to_world(data_root: Path, timestep: int, camera: int) -> np.ndarray:
    ego_start = np.loadtxt(data_root / "ego_pose" / "000.txt")
    ego_current = np.loadtxt(data_root / "ego_pose" / f"{timestep:03d}.txt")
    camera_to_ego = np.loadtxt(data_root / "extrinsics" / f"{camera}.txt")
    return np.linalg.inv(ego_start) @ ego_current @ camera_to_ego


def project_box(
    corners_world: np.ndarray, intrinsic: np.ndarray, world_to_camera: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, tuple[int, int, int, int], float] | None:
    corners_h = np.concatenate((corners_world, np.ones((8, 1))), axis=1)
    points_camera = (world_to_camera @ corners_h.T).T[:, :3]
    positive = points_camera[:, 2] > 0.1
    if positive.sum() < 4:
        return None
    uv_full = np.full((8, 2), np.nan, dtype=np.float64)
    uv_valid_h = (intrinsic @ points_camera[positive].T).T
    uv_full[positive] = uv_valid_h[:, :2] / uv_valid_h[:, 2:3]
    uv = uv_full[positive]
    x0 = max(0, int(np.floor(uv[:, 0].min())))
    x1 = min(width - 1, int(np.ceil(uv[:, 0].max())))
    y0 = max(0, int(np.floor(uv[:, 1].min())))
    y1 = min(height - 1, int(np.ceil(uv[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return None
    visible_area = (x1 - x0 + 1) * (y1 - y0 + 1) / float(width * height)
    return uv_full, (x0, y0, x1, y1), visible_area


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    canvas = image.copy()
    selected = mask > 0
    canvas[selected] = (canvas[selected] * (1.0 - alpha) + np.asarray(color) * alpha).astype(np.uint8)
    return canvas


def draw_box(image: np.ndarray, uv: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> np.ndarray:
    canvas = image.copy()
    # The 3D-box edges are drawn only when both endpoints survived positive-depth clipping.
    for start, end in EDGES:
        if np.isfinite(uv[start]).all() and np.isfinite(uv[end]).all():
            p0 = tuple(np.rint(uv[start]).astype(int))
            p1 = tuple(np.rint(uv[end]).astype(int))
            cv2.line(canvas, p0, p1, (0, 255, 0), 3, cv2.LINE_AA)
    x0, y0, x1, y1 = bbox
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (x0, max(32, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (x0, max(32, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def load_annotations(data_root: Path) -> tuple[dict[str, dict], dict[int, set[str]]]:
    info = json.loads((data_root / "instances" / "instances_info.json").read_text())
    world_normalizer = np.linalg.inv(np.loadtxt(data_root / "ego_pose" / "000.txt"))
    annotations: dict[str, dict] = {}
    present: dict[int, set[str]] = defaultdict(set)
    for instance_id, instance in info.items():
        if instance["class_name"] not in VEHICLE_CLASSES:
            continue
        per_frame = {}
        frames = instance["frame_annotations"]["frame_idx"]
        poses = instance["frame_annotations"]["obj_to_world"]
        sizes = instance["frame_annotations"]["box_size"]
        for frame, pose, size in zip(frames, poses, sizes):
            per_frame[int(frame)] = (
                world_normalizer @ np.asarray(pose, dtype=np.float64),
                np.asarray(size, dtype=np.float64),
            )
            present[int(frame)].add(instance_id)
        annotations[instance_id] = {"class_name": instance["class_name"], "uuid": instance["id"], "frames": per_frame}
    return annotations, present


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=56)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-instances", type=int, default=5)
    parser.add_argument("--top-views", type=int, default=5)
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir
    review_dir = output_dir / "visual-review"
    review_dir.mkdir(parents=True, exist_ok=True)
    frame_count = len(list((data_root / "ego_pose").glob("*.txt")))
    available = np.arange(frame_count)
    rng = np.random.default_rng(args.seed)
    sampled = np.sort(rng.choice(available, size=min(args.sample_count, frame_count), replace=False)).astype(int).tolist()
    annotations, present = load_annotations(data_root)

    intrinsics = {}
    for camera in range(7):
        fx, fy, cx, cy, *_ = np.loadtxt(data_root / "intrinsics" / f"{camera}.txt")
        intrinsics[camera] = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)

    records: list[dict] = []
    for timestep in sampled:
        for camera in range(7):
            image_path = data_root / "images" / f"{timestep:03d}_{camera}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            world_to_camera = np.linalg.inv(camera_to_world(data_root, timestep, camera))
            for instance_id in present[timestep]:
                pose, size = annotations[instance_id]["frames"][timestep]
                corners_local = box_corners(size)
                corners_world = (pose @ np.concatenate((corners_local, np.ones((8, 1))), axis=1).T).T[:, :3]
                result = project_box(corners_world, intrinsics[camera], world_to_camera, width, height)
                if result is None:
                    continue
                uv, bbox, area_ratio = result
                records.append(
                    {
                        "instance_id": instance_id,
                        "uuid": annotations[instance_id]["uuid"],
                        "class_name": annotations[instance_id]["class_name"],
                        "timestep": timestep,
                        "camera": camera,
                        "bbox_xyxy": bbox,
                        "box_area_ratio": area_ratio,
                        "touches_image_border": bbox[0] <= 8 or bbox[1] <= 8 or bbox[2] >= width - 9 or bbox[3] >= height - 9,
                        "has_road_mask": (data_root / "road_masks" / f"{timestep:03d}_{camera}.png").exists(),
                        "corners_uv": uv.tolist(),
                    }
                )

    per_instance: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        per_instance[record["instance_id"]].append(record)
    ranking = []
    for instance_id, views in per_instance.items():
        views.sort(key=lambda item: item["box_area_ratio"], reverse=True)
        eligible_views = [view for view in views if not view["touches_image_border"] and view["has_road_mask"]]
        if not eligible_views:
            continue
        ranking.append(
            {
                "instance_id": instance_id,
                "uuid": views[0]["uuid"],
                "class_name": views[0]["class_name"],
                "visible_views": len(views),
                "eligible_views": len(eligible_views),
                "max_box_area_ratio": eligible_views[0]["box_area_ratio"],
                "top_views": eligible_views[: args.top_views],
            }
        )
    ranking.sort(key=lambda item: item["max_box_area_ratio"], reverse=True)
    selected = ranking[: args.top_instances]

    for instance_rank, item in enumerate(selected, start=1):
        for view_rank, view in enumerate(item["top_views"], start=1):
            timestep, camera = view["timestep"], view["camera"]
            image = cv2.imread(str(data_root / "images" / f"{timestep:03d}_{camera}.jpg"), cv2.IMREAD_COLOR)
            fine_mask_path = data_root / "fine_dynamic_masks" / "vehicle" / f"{timestep:03d}_{camera}.png"
            road_mask_path = data_root / "road_masks" / f"{timestep:03d}_{camera}.png"
            fine_mask = cv2.imread(str(fine_mask_path), cv2.IMREAD_GRAYSCALE) if fine_mask_path.exists() else None
            road_mask = cv2.imread(str(road_mask_path), cv2.IMREAD_GRAYSCALE) if road_mask_path.exists() else None
            if fine_mask is None:
                fine_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if road_mask is None:
                road_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            box_view = draw_box(image, np.asarray(view["corners_uv"]), tuple(view["bbox_xyxy"]), f"id={item['instance_id']} {item['class_name']}")
            dynamic_view = overlay_mask(image, fine_mask, (255, 0, 255), 0.45)
            road_view = overlay_mask(image, road_mask, (0, 180, 0), 0.45)
            combined = overlay_mask(overlay_mask(box_view, road_mask, (0, 180, 0), 0.28), fine_mask, (255, 0, 255), 0.42)
            title = f"rank {instance_rank} view {view_rank} | t={timestep:03d} cam={camera} | area={view['box_area_ratio']:.3%}"
            cv2.putText(combined, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(combined, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            if not road_mask_path.exists():
                cv2.putText(combined, "road mask unavailable", (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(combined, "road mask unavailable", (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            panel = cv2.hconcat((box_view, dynamic_view, road_view, combined))
            output = review_dir / f"instance-{item['instance_id']}_rank-{instance_rank:02d}_view-{view_rank:02d}_t{timestep:03d}_cam{camera}.jpg"
            cv2.imwrite(str(output), panel, [cv2.IMWRITE_JPEG_QUALITY, 96])

    manifest = {
        "data_root": str(data_root),
        "frame_count": frame_count,
        "sample_count": len(sampled),
        "sample_seed": args.seed,
        "sampled_timesteps": sampled,
        "vehicle_classes": sorted(VEHICLE_CLASSES),
        "records": len(records),
        "selection_rule": "rank only projected boxes that do not touch an image border and whose camera frame has a road mask; both full vehicle geometry and road intersection are required for shadow geometry.",
        "top_instances": selected,
        "notes": [
            "Bounding boxes are annotation-oracle projections, not YOLO detections.",
            "Fine dynamic vehicle masks are union masks and are visualized only as a contextual cue.",
            "No shadow mask, 3D shadow points, or sun estimate is produced by this audit.",
        ],
    }
    (output_dir / "oracle_view_ranking.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "all_oracle_projected_views.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"records": len(records), "top_instances": selected}, indent=2))


if __name__ == "__main__":
    main()
