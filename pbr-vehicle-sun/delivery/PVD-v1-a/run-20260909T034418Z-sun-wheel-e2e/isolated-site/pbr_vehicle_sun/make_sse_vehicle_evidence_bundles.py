#!/usr/bin/env python3
"""Create one high-resolution evidence sheet per fitted SSE vehicle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                 max(1, round(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 248, np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def panel(image: np.ndarray, title: str, subtitle: str = "",
          width: int = 1200, height: int = 930) -> np.ndarray:
    header = 86
    body = letterbox(image, width, height - header)
    output = np.full((height, width, 3), 255, np.uint8)
    output[header:] = body
    cv2.putText(output, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                .78, (0, 0, 0), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(output, subtitle, (18, 69), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (45, 45, 45), 1, cv2.LINE_AA)
    return output


def read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def rgb_contour_overlay(source_path: Path, contour_npz: Path) -> np.ndarray:
    """Project top-down plane contours back into the source camera image."""
    image = read(source_path).copy()
    data = np.load(contour_npz)
    c2w = data["camera_to_world"].astype(np.float64)
    k = data["intrinsics"].astype(np.float64)
    r_w2c = np.linalg.inv(c2w)[:3, :3]
    t_w2c = np.linalg.inv(c2w)[:3, 3]
    anchor, e1, e2 = data["plane_anchor"], data["plane_e1"], data["plane_e2"]
    def project(points: np.ndarray) -> np.ndarray:
        world = anchor[None, :] + points[:, 0, None] * e1[None, :] + points[:, 1, None] * e2[None, :]
        cam = world @ r_w2c.T + t_w2c[None, :]
        valid = cam[:, 2] > 1e-5
        px = np.zeros((len(points), 2), np.int32)
        px[valid, 0] = np.rint(k[0, 0] * cam[valid, 0] / cam[valid, 2] + k[0, 2]).astype(np.int32)
        px[valid, 1] = np.rint(k[1, 1] * cam[valid, 1] / cam[valid, 2] + k[1, 2]).astype(np.int32)
        return px
    def draw(points_key: str, offsets_key: str, color: tuple[int, int, int], width: int) -> None:
        points, offsets = data[points_key], data[offsets_key]
        for start, end in zip(offsets[:-1], offsets[1:]):
            if end - start < 2: continue
            px = project(points[start:end])
            visible = (px[:, 0] >= 0) & (px[:, 0] < image.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < image.shape[0])
            if visible.sum() >= 2:
                cv2.polylines(image, [px[visible].reshape(-1, 1, 2)], False, color, width, cv2.LINE_AA)
    draw("observed_points_xy", "observed_offsets", (0, 220, 0), 5)
    draw("predicted_points_xy", "predicted_offsets", (0, 0, 255), 5)
    cv2.putText(image, "green=observed shadow visible arc; red=predicted vehicle projection arc", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, "green=observed shadow visible arc; red=predicted vehicle projection arc", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    output = (args.output_dir or run / "vehicle_evidence_bundles").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "batch_manifest.json").read_text())
    records = []
    for vehicle in manifest["vehicles"]:
        if vehicle["fit_status"] != "completed":
            continue
        source = Path(vehicle["source"])
        fit_path = Path(vehicle["fit_image"])
        gate = vehicle["gate"]
        source_image = Path(gate["inputs"]["source_image"])
        probability_path = Path(gate["inputs"]["probability"])
        probability = np.load(probability_path).astype(np.float32)
        heatmap = cv2.applyColorMap(
            np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO)
        mask = cv2.imread(str(source / "shadow_postprocess/04_shadow_postprocessed.png"),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(source / "shadow_postprocess/04_shadow_postprocessed.png")
        fit = json.loads((fit_path.parent / "fit_result.json").read_text())
        pred = fit["best"]["frames"][0]["arc_audit"]["components"][0]
        obs = fit["camera_contract"][0]["observed_arc"]["components"][0]
        angle = (f'az={fit["best"]["azimuth_deg"]:.0f} deg  '
                 f'el={fit["best"]["elevation_deg"]:.0f} deg  '
                 f'score={fit["best"]["joint_score"]:.4f}')
        if pred.get("operator") == "maximal_camera_cone_middle_arc":
            pred_stats = (f'pred cone: hits {pred.get("left_hit_count", 0)}/'
                          f'{pred.get("right_hit_count", 0)}, {pred.get("selected_points", 0)} pts, '
                          f'{pred.get("scored_arc_length_m", 0):.3f} m')
            obs_stats = (f'obs cone: hits {obs.get("left_hit_count", 0)}/'
                         f'{obs.get("right_hit_count", 0)}, {obs.get("selected_points", 0)} pts, '
                         f'{obs.get("scored_arc_length_m", 0):.3f} m')
        else:
            pred_stats = (f'pred tangent/contact: {pred.get("points_before_contact_exclusion", 0)}'
                          f' -> {pred.get("points_after_contact_exclusion", 0)} pts, '
                          f'{pred.get("scored_arc_length_m", 0):.3f} m')
            obs_stats = (f'obs tangent/contact: {obs.get("points_before_contact_exclusion", 0)}'
                         f' -> {obs.get("points_after_contact_exclusion", 0)} pts, '
                         f'{obs.get("scored_arc_length_m", 0):.3f} m')
        blur = gate["probability_edge"]
        gate_stats = (f'P35-P65 blur p90={blur["blur_p90_vehicle_height_ratio"]:.3f} vehicle-H; '
                      f'contrast={blur["contrast"]:.3f}; gate=ACCEPT')
        contour_file = fit_path.parent / "frame0_best_contours.npz"
        rgb_overlay = rgb_contour_overlay(source_image, contour_file) if contour_file.exists() else read(source_image)
        sheets = [
            panel(read(source_image), "1. Source RGB", f'{vehicle["scene"]} t{vehicle["timestep"]:03d} cam{vehicle["camera"]} {vehicle["vehicle"]}'),
            panel(heatmap, "2. MTMT shadow probability", gate_stats),
            panel(rgb_overlay, "3. RGB contour projection", "green=observed shadow arc; red=predicted vehicle projection arc"),
            panel(read(source / "vehicle_geometry/01_infinidepth_vehicle_selection_review.jpg"),
                  "4. InfiniDepth PLY vehicle selection", "green=mask-visible vehicle Gaussian centres in source camera"),
            panel(read(source / "shadow_geometry/01_source_shadow_depth_lift_review.jpg"),
                  "5. Shadow depth lift evidence", "cleaned source shadow + InfiniDepth depth + road-plane lift"),
            panel(read(fit_path), "6. Top-down PLY projection and contour fit",
                  angle + " | " + pred_stats + " | " + obs_stats),
        ]
        sheet = cv2.vconcat([cv2.hconcat(sheets[:3]), cv2.hconcat(sheets[3:])])
        name = f'scene-{vehicle["scene"]}_t{vehicle["timestep"]:03d}_cam{vehicle["camera"]}_{vehicle["vehicle"]}_evidence.jpg'
        target = output / name
        cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])
        records.append({"scene": vehicle["scene"], "timestep": vehicle["timestep"],
                        "camera": vehicle["camera"], "vehicle": vehicle["vehicle"],
                        "bundle": str(target), "predicted_arc": pred,
                        "observed_arc": obs})
    (output / "manifest.json").write_text(json.dumps({"bundles": records}, indent=2) + "\n")
    print(json.dumps({"count": len(records), "output_dir": str(output),
                      "files": [record["bundle"] for record in records]}, indent=2))


if __name__ == "__main__":
    main()
