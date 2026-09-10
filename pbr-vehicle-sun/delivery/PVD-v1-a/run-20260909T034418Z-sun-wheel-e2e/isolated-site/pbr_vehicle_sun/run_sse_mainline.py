#!/usr/bin/env python3
"""Unified SSE-v6 mainline dispatcher.

The scene-sun estimator has two deliberate geometry branches:

* ``native_gaussian``: the scene/vehicle already supplies Gaussian geometry;
  retain the previous complete-support contour branch.
* ``feedforward_infinidepth``: no scene-native Gaussian is available; use the
  same-frame InfiniDepth visible-geometry adapter and the promoted SSE-v6
  maximal-camera-cone contour branch.

Both branches consume the same postprocessed source-view shadow masks. This
file is orchestration only; numerical fitting remains in the branch scripts.
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
    parser.add_argument("--feedforward-include-image-edge-shadow", action="store_true",
                        help="legacy opt-in: retain shadow contour sections touching the source-image crop")
    parser.add_argument("--feedforward-subtract-predicted-floor", action="store_true",
                        help="legacy opt-in: Boolean-subtract vehicle footprint from projected mask")
    parser.add_argument("--angular-bins", type=int, default=512)
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
        raise ValueError("vehicle geometry, shadow geometry, and postprocessed shadow mask lists must have equal lengths")
    frame_count = len(args.vehicle_geometry)

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
                        "--angular-bins", str(args.angular_bins)])
        if args.feedforward_visibility_mode == "camera_cone_middle":
            if not args.feedforward_include_image_edge_shadow:
                command.append("--exclude-image-edge-components")
            if not args.feedforward_subtract_predicted_floor:
                command.append("--no-predicted-floor-subtraction")
    manifest = {
        "mainline": "SSE-v6",
        "geometry_source": args.geometry_source,
        "execution_mode": "single_frame" if frame_count == 1 else "multi_frame",
        "frame_count": frame_count,
        "selected_fit_script": fit_script,
        "branch": branch,
        "inputs": {
            "vehicle_geometry": [str(path.resolve()) for path in args.vehicle_geometry],
            "shadow_geometry": [str(path.resolve()) for path in args.shadow_geometry],
            "postprocessed_shadow_masks": [str(path.resolve()) for path in args.shadow_mask],
        },
        "command": command,
        "postprocess_contract": "caller-supplied masks must be the exact shadow AND NOT vehicle result after 8-connected cleanup, at most two retained components",
        "top25_contract": "branch fitter writes Top-25 from the same objective used for best selection",
        "feedforward_visibility_mode": args.feedforward_visibility_mode if args.geometry_source == "feedforward_infinidepth" else None,
        "feedforward_exclude_image_edge_shadow": (not args.feedforward_include_image_edge_shadow) if args.geometry_source == "feedforward_infinidepth" else None,
        "feedforward_predicted_floor_subtraction": args.feedforward_subtract_predicted_floor if args.geometry_source == "feedforward_infinidepth" else None,
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
    result["mainline_version"] = "SSE-v6"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"branch_manifest": str((args.output_dir / 'sse_mainline_branch_manifest.json').resolve()), "fit_result": result}, indent=2))


if __name__ == "__main__":
    main()
