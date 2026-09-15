#!/usr/bin/env python3
"""Run the generic SSE-v8 scene pipeline on the globally largest vehicles.

The Argoverse adapter scans all requested frames/cameras, groups observations
by physical vehicle, and selects one maximum-area SAM2 mask per vehicle.
SSISv2 then supplies the object-associated shadow mask. That official mask is
lifted directly with same-frame InfiniDepth depth and is never rewritten by
subtraction, connected-component cleanup, morphology, or a detector-specific
probability gate. Successfully prepared vehicles enter the common 1..5
confidence-gated shared-angle fitter together.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .device import resolve_device
except ImportError:
    from device import resolve_device


SCRIPTS = Path(__file__).resolve().parent


class VehicleMethodRejection(RuntimeError):
    """The selected vehicle lacks usable method evidence, not a process failure."""


VEHICLE_EVIDENCE_EXIT_CODE = 42


def is_vehicle_evidence_failure(error: subprocess.CalledProcessError) -> bool:
    return error.returncode == VEHICLE_EVIDENCE_EXIT_CODE


def run_stage(name: str, command: list[str], timings: list[dict], env: dict | None = None) -> None:
    started = time.perf_counter()
    subprocess.run(command, check=True, env=env)
    timings.append({"stage": name, "elapsed_s": time.perf_counter() - started, "command": command})


def prepend_pythonpath(env: dict[str, str], path: Path) -> dict[str, str]:
    result = env.copy()
    old = result.get("PYTHONPATH")
    result["PYTHONPATH"] = str(path.resolve()) + (os.pathsep + old if old else "")
    return result


def select_ranked_vehicles(ranking: dict, offset: int, count: int) -> list[dict]:
    return ranking["selected"][offset:offset + count]


def ssisv2_command(args: argparse.Namespace, image: Path, vehicle_mask: Path,
                   output_dir: Path) -> list[str]:
    return [
        str(args.ssis_python), str(SCRIPTS / "run_ssisv2_vehicle_shadow_association.py"),
        "--ssis-root", str(args.ssis_root), "--weights", str(args.ssis_weights),
        "--source-image", str(image), "--vehicle-mask", str(vehicle_mask),
        "--output-dir", str(output_dir), "--device", args.ssis_device,
        "--confidence-threshold", str(args.ssis_confidence_threshold),
        "--minimum-object-iou", str(args.minimum_ssis_object_iou),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="scene identity used only in manifests")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--sam2-root", type=Path, required=True)
    ap.add_argument("--sam2-checkpoint", type=Path, required=True)
    ap.add_argument("--sam2-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--ssis-root", type=Path, required=True)
    ap.add_argument("--ssis-weights", type=Path, required=True)
    ap.add_argument("--ssis-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--ssis-device", default="auto")
    ap.add_argument("--ssis-confidence-threshold", type=float, default=.10)
    ap.add_argument("--minimum-ssis-object-iou", type=float, default=.05)
    ap.add_argument("--infinidepth-root", type=Path, required=True)
    ap.add_argument("--infinidepth-python", type=Path, default=Path(sys.executable))
    ap.add_argument("--infinidepth-depth-checkpoint", type=Path)
    ap.add_argument("--infinidepth-gs-checkpoint", type=Path)
    ap.add_argument("--moge2-pretrained", type=Path)
    ap.add_argument("--sky-checkpoint", type=Path)
    ap.add_argument("--infinidepth-sample-points", type=int, default=2_000_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--yolo-batch", type=int, default=16)
    ap.add_argument("--top-vehicles", type=int, choices=range(1, 6), default=3)
    ap.add_argument("--selection-pool-size", type=int)
    ap.add_argument("--vehicle-rank-offset", type=int, default=0)
    ap.add_argument("--min-mask-area-ratio", type=float, default=.01)
    ap.add_argument("--min-vehicle-confidence", type=float, default=.50)
    ap.add_argument("--min-vehicle-support", type=float, default=.75)
    ap.add_argument("--cameras", nargs="+", type=int, default=list(range(7)))
    ap.add_argument("--existing-detection", type=Path)
    ap.add_argument("--existing-ranking", type=Path)
    args = ap.parse_args()
    compute_device = resolve_device(args.device)
    ssis_device = resolve_device(args.ssis_device)
    args.ssis_device = ssis_device.torch
    print(f"SSE compute device: {compute_device.description}", flush=True)
    if args.vehicle_rank_offset < 0:
        ap.error("--vehicle-rank-offset must be non-negative")
    selection_pool_size = args.selection_pool_size or args.top_vehicles + args.vehicle_rank_offset
    if selection_pool_size < args.top_vehicles + args.vehicle_rank_offset:
        ap.error("--selection-pool-size must cover the requested vehicle rank range")

    started = time.perf_counter()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    timings: list[dict] = []
    py = sys.executable

    depth_checkpoint = args.infinidepth_depth_checkpoint or args.infinidepth_root / "checkpoints/depth/infinidepth.ckpt"
    gs_checkpoint = args.infinidepth_gs_checkpoint or args.infinidepth_root / "checkpoints/gs/infinidepth_gs.ckpt"
    moge2 = args.moge2_pretrained or args.infinidepth_root / "checkpoints/moge-2-vitl-normal/model.pt"
    sky_checkpoint = args.sky_checkpoint or args.infinidepth_root / "checkpoints/sky/skyseg.onnx"

    detection = out / "selection/all_vehicle_detections.json"
    if args.existing_detection:
        detection = args.existing_detection.resolve()
    else:
        run_stage("all_frame_all_camera_yolo_identity", [
            py, str(SCRIPTS / "detect_sse_scene_all_vehicle_views.py"),
            "--data-root", str(args.data_root), "--output", str(detection),
            "--weights", str(args.yolo_weights), "--device", compute_device.ultralytics,
            "--batch", str(args.yolo_batch),
            "--cameras", *map(str, args.cameras),
        ], timings)

    ranking_dir = out / "selection/sam2_global_ranking"
    if args.existing_ranking:
        ranking_manifest = args.existing_ranking.resolve()
        ranking_dir = ranking_manifest.parent
    else:
        ranking_manifest = ranking_dir / "scene_global_vehicle_ranking.json"
        sam_env = prepend_pythonpath(os.environ, args.sam2_root)
        run_stage("all_candidate_sam2_global_mask_ranking", [
            str(args.sam2_python), str(SCRIPTS / "rank_sse_scene_vehicles_by_sam2_mask.py"),
            "--detections", str(detection), "--output-dir", str(ranking_dir),
            "--checkpoint", str(args.sam2_checkpoint), "--config", args.sam2_config,
            "--device", compute_device.torch,
            "--top-vehicles", str(selection_pool_size),
            "--min-mask-area-ratio", str(args.min_mask_area_ratio),
        ], timings, sam_env)
    ranking = json.loads(ranking_manifest.read_text())
    selected = select_ranked_vehicles(ranking, args.vehicle_rank_offset, args.top_vehicles)

    depth_env = prepend_pythonpath(os.environ, args.infinidepth_root)
    common: dict[tuple[int, int], tuple[Path, Path]] = {}
    prepared: list[dict] = []
    results: list[dict] = []
    for vehicle in selected:
        rank = int(vehicle["vehicle_rank"])
        timestep = int(vehicle["timestep"])
        camera = int(vehicle["camera"])
        image = Path(vehicle["image"])
        vehicle_mask = ranking_dir / vehicle["mask"]
        key = (timestep, camera)
        view_dir = out / f"t{timestep:03d}_cam{camera}"
        vehicle_dir = view_dir / f"vehicle_{rank:02d}_instance-{vehicle['instance_id']}"
        record = {
            "vehicle_rank": rank, "instance_id": vehicle["instance_id"],
            "timestep": timestep, "camera": camera,
            "mask_area_ratio": vehicle["mask_area_ratio"],
            "vehicle_mask": str(vehicle_mask.resolve()),
        }
        try:
            if key not in common:
                depth_dir = view_dir / "depth"
                run_stage(f"view_t{timestep:03d}_cam{camera}_rgb_depth", [
                    str(args.infinidepth_python), str(SCRIPTS / "infer_infinidepth_dense_depth.py"),
                    "--image", str(image), "--output-dir", str(depth_dir),
                    "--device", compute_device.torch, "--model-type", "InfiniDepth",
                    "--infinidepth-root", str(args.infinidepth_root),
                    "--depth-checkpoint", str(depth_checkpoint), "--moge2-pretrained", str(moge2),
                ], timings, depth_env)
                ply_dir = view_dir / "infinidepth_ply"
                ply_dir.mkdir(parents=True, exist_ok=True)
                fx, fy, cx, cy, *_ = map(float, np.loadtxt(args.data_root / "intrinsics" / f"{camera}.txt"))
                run_stage(f"view_t{timestep:03d}_cam{camera}_rgb_gaussian", [
                    str(args.infinidepth_python), str(args.infinidepth_root / "inference_gs.py"),
                    "--input-image-path", str(image), "--output-ply-dir", str(ply_dir),
                    "--output-ply-name", "scene_gaussians.ply", "--model-type", "InfiniDepth",
                    "--depth-model-path", str(depth_checkpoint), "--gs-model-path", str(gs_checkpoint),
                    "--moge2-pretrained", str(moge2), "--sky-model-ckpt-path", str(sky_checkpoint),
                    "--fx-org", str(fx), "--fy-org", str(fy), "--cx-org", str(cx), "--cy-org", str(cy),
                    "--sample-point-num", str(args.infinidepth_sample_points), "--no-render-novel-video",
                ], timings, depth_env)
                common[key] = (depth_dir, ply_dir / "scene_gaussians.ply")
            depth_dir, ply = common[key]

            association_dir = vehicle_dir / "ssisv2_association"
            run_stage(f"vehicle{rank}_ssisv2_association",
                      ssisv2_command(args, image, vehicle_mask, association_dir), timings)
            association_manifest = association_dir / "ssisv2_association_manifest.json"
            association = json.loads(association_manifest.read_text())
            if not association.get("accepted"):
                raise VehicleMethodRejection("SSISv2 found no object-shadow pair passing the vehicle IoU gate")
            shadow_mask = association_dir / "04_associated_shadow_mask.png"

            shadow_geometry_dir = vehicle_dir / "shadow_geometry"
            run_stage(f"vehicle{rank}_shadow_depth_lift", [
                py, str(SCRIPTS / "lift_source_shadow_with_infinidepth.py"),
                "--depth", str(depth_dir / "predicted_depth_original.npy"),
                "--shadow-mask", str(shadow_mask),
                "--road-mask", str(args.data_root / "road_masks" / f"{timestep:03d}_{camera}.png"),
                "--vehicle-mask", str(vehicle_mask), "--source-image", str(image),
                "--data-root", str(args.data_root), "--timestep", str(timestep),
                "--camera", str(camera), "--output-dir", str(shadow_geometry_dir),
            ], timings)
            vehicle_geometry_dir = vehicle_dir / "vehicle_geometry"
            run_stage(f"vehicle{rank}_vehicle_geometry", [
                py, str(SCRIPTS / "extract_infinidepth_vehicle_geometry.py"),
                "--ply", str(ply), "--vehicle-mask", str(vehicle_mask),
                "--data-root", str(args.data_root), "--timestep", str(timestep),
                "--camera", str(camera), "--output-dir", str(vehicle_geometry_dir),
            ], timings)
            prepared.append({
                **record,
                "association_manifest": str(association_manifest.resolve()),
                "shadow_mask": str(shadow_mask.resolve()),
                "shadow_geometry": str((shadow_geometry_dir / "infinidepth_shadow_geometry.npz").resolve()),
                "vehicle_geometry": str((vehicle_geometry_dir / "visible_infinidepth_vehicle_gaussians.npz").resolve()),
            })
            results.append({**record, "status": "prepared", "association_manifest": str(association_manifest.resolve())})
        except VehicleMethodRejection as error:
            results.append({**record, "status": "rejected", "reason": str(error)})
        except subprocess.CalledProcessError as error:
            if is_vehicle_evidence_failure(error):
                results.append({
                    **record,
                    "status": "rejected",
                    "reason": "vehicle geometric evidence did not satisfy the preparation contract",
                    "failed_command": error.cmd,
                    "returncode": error.returncode,
                })
                continue
            failure = {
                "scene": args.scene,
                "status": "process_failed",
                "failed_vehicle": record,
                "error": repr(error),
                "vehicle_preparation_results": results,
                "timings": timings,
                "total_elapsed_s": time.perf_counter() - started,
            }
            (out / "scene_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise
        except Exception as error:
            failure = {
                "scene": args.scene,
                "status": "process_failed",
                "failed_vehicle": record,
                "error": repr(error),
                "vehicle_preparation_results": results,
                "timings": timings,
                "total_elapsed_s": time.perf_counter() - started,
            }
            (out / "scene_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise

    aggregate_result = None
    if prepared:
        fit_dir = out / "confidence_gated_joint_fit"
        command = [
            py, str(SCRIPTS / "run_sse_mainline.py"), "--geometry-source", "feedforward_infinidepth",
            "--aggregation-mode", "vehicles",
            "--vehicle-geometry", *[row["vehicle_geometry"] for row in prepared],
            "--shadow-geometry", *[row["shadow_geometry"] for row in prepared],
            "--shadow-mask", *[row["shadow_mask"] for row in prepared],
            "--output-dir", str(fit_dir), "--min-vehicle-confidence", str(args.min_vehicle_confidence),
            "--min-vehicle-support", str(args.min_vehicle_support),
        ]
        run_stage("confidence_gated_one_to_five_vehicle_joint_fit", command, timings)
        aggregate_result = json.loads((fit_dir / "multi_vehicle_fit_result.json").read_text())

    summary = {
        "scene": args.scene,
        "mainline": "SSE-v8",
        "compute_device": {
            "requested": str(args.device),
            "torch": compute_device.torch,
            "ultralytics": compute_device.ultralytics,
            "name": compute_device.name,
        },
        "protocol_basis": "SSE-v6-b official SSISv2 associated raw shadow mask",
        "status": aggregate_result["status"] if aggregate_result else "no_valid_sun_information",
        "selection_contract": ranking["contract"],
        "vehicles_requested": args.top_vehicles,
        "selection_pool_size": selection_pool_size,
        "vehicle_rank_offset": args.vehicle_rank_offset,
        "vehicles_selected": len(selected),
        "vehicles_prepared": len(prepared),
        "uses_lidar": False,
        "shadow_detector": "SSISv2 object-shadow association",
        "source_mask_postprocessing": "none",
        "prepared_vehicle_inputs": prepared,
        "vehicle_preparation_results": results,
        "aggregate_result": aggregate_result,
        "timings": timings,
        "total_elapsed_s": time.perf_counter() - started,
        "contract_checks": {
            "all_frames_all_cameras": True,
            "physical_vehicle_deduplication": True,
            "global_max_sam_mask_ranking": True,
            "top_vehicle_count_between_one_and_five": 1 <= args.top_vehicles <= 5,
            "selected_vehicle_rank_range": [
                args.vehicle_rank_offset + 1,
                args.vehicle_rank_offset + args.top_vehicles,
            ],
            "official_ssisv2_pair_binding": True,
            "direct_ssisv2_shadow_mask": True,
            "shadow_vehicle_subtraction": False,
            "connected_component_cleanup": False,
            "source_mask_morphology": False,
            "legacy_detector_probability_gate": False,
            "confidence_gate_before_joint_fit": True,
        },
    }
    (out / "scene_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
