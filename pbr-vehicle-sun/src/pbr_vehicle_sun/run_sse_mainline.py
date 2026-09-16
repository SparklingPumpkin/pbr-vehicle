#!/usr/bin/env python3
"""Unified scene-sun-estimation mainline dispatcher.

The scene-sun estimator has two deliberate geometry branches:

* ``native_gaussian``: the scene/vehicle already supplies Gaussian geometry;
  retain the previous complete-support contour branch.
* ``feedforward_infinidepth``: no scene-native Gaussian is available; use the
  same-frame InfiniDepth visible-geometry adapter and the promoted SSE-v6-b
  maximal-camera-cone contour branch.

Both branches consume the official shadow member paired with the selected
vehicle by SSISv2.  The mask is never rewritten by vehicle subtraction,
connected-component cleanup, or morphology.  This file is orchestration only;
numerical fitting remains in the branch scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
BRANCHES = {
    "native_gaussian": {
        "fit_script": "run_sse_native_gaussian_branch.py",
        "objective": "unchanged SSE-v4 per-frame depth-lifted-shadow contour fit",
        "geometry_contract": "caller-provided scene-native Gaussian geometry; no feed-forward reconstruction",
    },
    "feedforward_infinidepth": {
        "fit_script": "fit_sun_camera_visible_arc.py",
        "objective": "source-edge-trimmed observed shadow and full candidate projection, each clipped to the complete near arc inside its own maximal same-frame camera cone",
        "geometry_contract": "same-frame InfiniDepth feed-forward visible vehicle geometry; no scene-native Gaussian input required",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-source", choices=sorted(BRANCHES), required=True)
    parser.add_argument("--aggregation-mode", choices=("frames", "vehicles"), default="frames",
                        help="frames share one observation sequence; vehicles are 1..5 independent vehicle observations with confidence gating")
    parser.add_argument("--vehicle-geometry", nargs="+", type=Path, required=True)
    parser.add_argument("--shadow-geometry", nargs="+", type=Path, required=True)
    parser.add_argument("--shadow-mask", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raster-size", type=int, default=900)
    parser.add_argument("--max-vehicle-points", type=int)
    parser.add_argument("--coarse-az-step", type=int, default=5)
    parser.add_argument("--coarse-elev-min", type=int, default=20)
    parser.add_argument("--coarse-elev-max", type=int)
    parser.add_argument("--local-radius-deg", type=int, default=5)
    parser.add_argument("--local-basins", type=int, default=3)
    parser.add_argument("--feedforward-visibility-mode", choices=("ray_arc", "angular_near_edge", "component_tangent_arcs", "camera_cone_middle"), default="camera_cone_middle")
    parser.add_argument("--angular-bins", type=int, default=512)
    parser.add_argument("--distance-percentile", type=float, default=.95,
                        help="robust contour tail percentile for feed-forward fitting")
    parser.add_argument("--structure-aware", action="store_true",
                        help="include optional smoothed tangent/curvature/bump line correspondence")
    parser.add_argument("--top-candidates", type=int, default=25)
    parser.add_argument("--per-vehicle-top-candidates", type=int, default=0,
                        help="candidate sheets for independent gate fits; best evidence is always written")
    parser.add_argument("--fit-workers", type=int, default=0,
                        help="parallel independent vehicle fits; 0 selects up to three")
    parser.add_argument("--candidate-workers", type=int, default=8,
                        help="parallel angle candidates inside each fit")
    parser.add_argument("--min-vehicle-confidence", type=float, default=.50)
    parser.add_argument("--min-vehicle-support", type=float, default=.75)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    branch = BRANCHES[args.geometry_source]
    max_vehicle_points = args.max_vehicle_points or (50000 if args.geometry_source == "native_gaussian" else 100000)
    coarse_elev_max = args.coarse_elev_max or (50 if args.geometry_source == "native_gaussian" else 70)
    for path in [*args.vehicle_geometry, *args.shadow_geometry, *args.shadow_mask]:
        if not path.is_file():
            raise FileNotFoundError(path)
    counts = {len(args.vehicle_geometry), len(args.shadow_geometry), len(args.shadow_mask)}
    if len(counts) != 1:
        raise ValueError("vehicle geometry, shadow geometry, and SSISv2 shadow mask lists must have equal lengths")
    frame_count = len(args.vehicle_geometry)

    if args.aggregation_mode == "vehicles":
        if args.geometry_source != "feedforward_infinidepth":
            raise ValueError("confidence-gated vehicle aggregation is defined for feedforward_infinidepth geometry")
        if not 1 <= frame_count <= 5:
            raise ValueError("vehicle aggregation requires one to five aligned vehicle tuples")
        if args.feedforward_visibility_mode != "camera_cone_middle":
            raise ValueError("vehicle aggregation requires the promoted camera_cone_middle visibility operator")
        command = [
            sys.executable, str(SCRIPTS / "run_sse_confidence_gated_joint.py"),
            "--vehicle-geometry", *map(str, args.vehicle_geometry),
            "--shadow-geometry", *map(str, args.shadow_geometry),
            "--shadow-mask", *map(str, args.shadow_mask),
            "--output-dir", str(args.output_dir),
            "--raster-size", str(args.raster_size),
            "--max-vehicle-points", str(max_vehicle_points),
            "--coarse-az-step", str(args.coarse_az_step),
            "--coarse-elev-min", str(args.coarse_elev_min),
            "--coarse-elev-max", str(coarse_elev_max),
            "--local-radius-deg", str(args.local_radius_deg),
            "--local-basins", str(args.local_basins),
            "--distance-percentile", str(args.distance_percentile),
            "--top-candidates", str(args.top_candidates),
            "--per-vehicle-top-candidates", str(args.per_vehicle_top_candidates),
            "--fit-workers", str(args.fit_workers),
            "--candidate-workers", str(args.candidate_workers),
            "--min-confidence", str(args.min_vehicle_confidence),
            "--min-support", str(args.min_vehicle_support),
        ]
        if args.structure_aware:
            command.append("--structure-aware")
        manifest = {
            "mainline": "SSE-v9",
            "geometry_source": args.geometry_source,
            "aggregation_mode": "vehicles",
            "vehicle_count": frame_count,
            "selected_fit_script": "run_sse_confidence_gated_joint.py",
            "inputs": {"vehicle_geometry": [str(path.resolve()) for path in args.vehicle_geometry],
                       "shadow_geometry": [str(path.resolve()) for path in args.shadow_geometry],
                       "shadow_masks": [str(path.resolve()) for path in args.shadow_mask]},
            "hard_gate": {"min_confidence": args.min_vehicle_confidence,
                          "min_support_0p35m": args.min_vehicle_support},
            "distance_percentile": args.distance_percentile,
            "structure_aware": args.structure_aware,
            "fit_workers": args.fit_workers,
            "candidate_workers": args.candidate_workers,
            "per_vehicle_top_candidates": args.per_vehicle_top_candidates,
            "shadow_mask_contract": "official SSISv2 associated shadow mask, consumed directly without source-mask postprocessing",
            "all_rejected_behavior": "no_valid_sun_information",
            "command": command,
        }
        (args.output_dir / "sse_mainline_branch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
            return
        subprocess.run(command, check=True)
        result_path = args.output_dir / "multi_vehicle_fit_result.json"
        if not result_path.is_file():
            raise RuntimeError(f"vehicle aggregator did not produce {result_path}")
        print(result_path.read_text())
        return

    # A single native-Gaussian observation uses the original numerical fitter
    # directly. Multi-frame native inputs go through the stability adapter.
    # The InfiniDepth fitter itself implements both one-frame and shared-angle
    # multi-frame modes with the identical camera-visible-arc objective.
    fit_script = branch["fit_script"]
    if args.geometry_source == "native_gaussian" and frame_count == 1:
        fit_script = "fit_sun_from_depth_lifted_shadow.py"

    command = [
        sys.executable,
        str(SCRIPTS / fit_script),
        "--vehicle-geometry", *map(str, args.vehicle_geometry),
        "--shadow-geometry", *map(str, args.shadow_geometry),
        "--output-dir", str(args.output_dir),
        "--raster-size", str(args.raster_size),
        "--max-vehicle-points", str(max_vehicle_points),
        "--coarse-az-step", str(args.coarse_az_step),
        "--coarse-elev-min", str(args.coarse_elev_min),
        "--coarse-elev-max", str(coarse_elev_max),
        "--local-radius-deg", str(args.local_radius_deg),
        "--local-basins", str(args.local_basins),
    ]
    # The original native single-frame fitter consumes its shadow through the
    # depth-lifted NPZ and has no --shadow-mask argument; every other path uses
    # masks explicitly for lineage/camera-visible clipping.
    if not (args.geometry_source == "native_gaussian" and frame_count == 1):
        insert_at = command.index("--output-dir")
        command[insert_at:insert_at] = ["--shadow-mask", *map(str, args.shadow_mask)]
    if args.geometry_source == "feedforward_infinidepth":
        command.extend(["--visibility-mode", args.feedforward_visibility_mode,
                        "--angular-bins", str(args.angular_bins),
                        "--distance-percentile", str(args.distance_percentile),
                        "--top-candidates", str(args.top_candidates)])
        if args.structure_aware:
            command.append("--structure-aware")
        command.append("--direct-shadow-mask")
        if args.feedforward_visibility_mode == "camera_cone_middle":
            command.append("--exclude-image-edge-components")
            command.append("--no-predicted-floor-subtraction")
    manifest = {
        "mainline": "SSE-v9",
        "geometry_source": args.geometry_source,
        "execution_mode": "single_frame" if frame_count == 1 else "multi_frame",
        "frame_count": frame_count,
        "selected_fit_script": fit_script,
        "branch": branch,
        "inputs": {
            "vehicle_geometry": [str(path.resolve()) for path in args.vehicle_geometry],
            "shadow_geometry": [str(path.resolve()) for path in args.shadow_geometry],
            "ssisv2_associated_shadow_masks": [str(path.resolve()) for path in args.shadow_mask],
        },
        "command": command,
        "shadow_mask_contract": "official SSISv2 associated shadow mask, consumed directly without vehicle subtraction, connected-component cleanup, or morphology",
        "top25_contract": "branch fitter writes Top-25 from the same objective used for best selection",
        "feedforward_visibility_mode": args.feedforward_visibility_mode if args.geometry_source == "feedforward_infinidepth" else None,
        "feedforward_exclude_image_edge_shadow": True if args.geometry_source == "feedforward_infinidepth" else None,
        "feedforward_predicted_floor_subtraction": False if args.geometry_source == "feedforward_infinidepth" else None,
        "angular_bins": args.angular_bins if args.geometry_source == "feedforward_infinidepth" and args.feedforward_visibility_mode == "angular_near_edge" else None,
    }
    (args.output_dir / "sse_mainline_branch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    subprocess.run(command, check=True)
    result_path = args.output_dir / "fit_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"branch fitter did not produce {result_path}")
    result = json.loads(result_path.read_text())
    result["mainline_branch"] = args.geometry_source
    result["mainline_version"] = "SSE-v9"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"branch_manifest": str((args.output_dir / 'sse_mainline_branch_manifest.json').resolve()), "fit_result": result}, indent=2))


if __name__ == "__main__":
    main()
