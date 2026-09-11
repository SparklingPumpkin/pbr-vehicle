#!/usr/bin/env python3
"""Generic SSE one-to-five vehicle confidence-gated sun fitting.

Each input tuple is one independently selected vehicle observation.  The
shadow mask in each tuple is the official SSISv2 shadow member associated with
that selected vehicle and is consumed without source-mask postprocessing. The
existing camera-visible-arc fitter is run once per vehicle, using exactly the
same distance percentile and optional contour-structure objective. A hard
confidence gate is applied before aggregation.  Accepted vehicles are then
combined into a confidence-weighted estimate and, when at least two survive,
one shared-angle joint fit.  If no vehicle survives, the output is explicit
``no_valid_sun_information`` and no angle is published.

This is orchestration only: contour rasterization, camera-cone visibility,
line processing, and numerical search remain in fit_sun_camera_visible_arc.py.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parent
FIT = SCRIPTS / "fit_sun_camera_visible_arc.py"
VEHICLE_EVIDENCE_EXIT_CODE = 42


def validate_vehicle_count(count: int) -> None:
    if not 1 <= count <= 5:
        raise ValueError("one to five vehicle tuples are required")


def gate_confidence(confidence: float, support: float, min_confidence: float,
                    min_support: float) -> list[str]:
    reasons = []
    if not math.isfinite(confidence) or confidence < min_confidence:
        reasons.append(f"confidence<{min_confidence:g}")
    if not math.isfinite(support) or support < min_support:
        reasons.append(f"support<{min_support:g}")
    return reasons


def confidence_from_fit(result: dict) -> tuple[float, dict]:
    """Return the established boundary-match confidence and its audit terms."""
    best = result.get("best", {})
    frames = best.get("frames", [])
    if not frames:
        return 0.0, {"reason": "missing_best_frame"}
    frame = frames[0]
    support = float(frame.get("pred_boundary_support_0p35m", 0.0))
    tail = float(frame.get("pred_to_observed_tail_m", frame.get("pred_to_observed_p90_m", math.inf)))
    median = float(frame.get("pred_to_observed_median_m", math.inf))
    reverse = float(frame.get("observed_to_predicted_tail_m", frame.get("observed_to_predicted_p90_m", math.inf)))
    robust_term = 0.40 * tail + 0.25 * median + 0.25 * reverse
    confidence = support * math.exp(-robust_term / 0.35) if math.isfinite(robust_term) else 0.0
    return float(confidence), {
        "support_0p35m": support,
        "pred_to_observed_tail_m": tail,
        "pred_to_observed_median_m": median,
        "observed_to_predicted_tail_m": reverse,
        "robust_distance_term_m": robust_term,
        "confidence_formula": "support_0p35m * exp(-(0.40*tail + 0.25*median + 0.25*reverse)/0.35m)",
    }


def weighted_angle(rows: list[dict]) -> dict:
    weights = np.asarray([row["match_confidence"] for row in rows], dtype=float)
    az = np.deg2rad(np.asarray([row["azimuth_deg"] for row in rows], dtype=float))
    total = float(weights.sum())
    if total <= 0:
        return {"valid": False, "reason": "non_positive_weight_sum"}
    return {
        "valid": True,
        "method": "circular azimuth mean and arithmetic elevation mean weighted by gated boundary confidence",
        "weighted_azimuth_deg": float(np.rad2deg(np.arctan2(np.sum(weights * np.sin(az)), np.sum(weights * np.cos(az)))) % 360.0),
        "weighted_elevation_deg": float(np.sum(weights * np.asarray([row["elevation_deg"] for row in rows])) / total),
        "weight_sum": total,
    }


def publication_status(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["status"] == "accepted"]
    return {
        "status": "ok" if accepted else "no_valid_sun_information",
        "accepted_vehicle_indices": [row["vehicle_index"] for row in accepted],
        "rejected_vehicle_indices": [row["vehicle_index"] for row in rows if row["status"] != "accepted"],
    }


def fit_command(geometry: list[Path], shadow: list[Path], masks: list[Path], output: Path,
                args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(FIT), "--vehicle-geometry", *map(str, geometry),
               "--shadow-geometry", *map(str, shadow), "--shadow-mask", *map(str, masks),
               "--output-dir", str(output), "--raster-size", str(args.raster_size),
               "--max-vehicle-points", str(args.max_vehicle_points), "--coarse-az-step", str(args.coarse_az_step),
               "--coarse-elev-min", str(args.coarse_elev_min), "--coarse-elev-max", str(args.coarse_elev_max),
               "--local-radius-deg", str(args.local_radius_deg), "--local-basins", str(args.local_basins),
               "--visibility-mode", args.visibility_mode, "--distance-percentile", str(args.distance_percentile),
               "--top-candidates", str(args.top_candidates), "--top-page-size", "25", "--direct-shadow-mask",
               "--no-observed-floor-subtraction", "--no-predicted-floor-subtraction", "--exclude-image-edge-components"]
    if args.structure_aware:
        command.append("--structure-aware")
    return command


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vehicle-geometry", nargs="+", type=Path, required=True)
    ap.add_argument("--shadow-geometry", nargs="+", type=Path, required=True)
    ap.add_argument("--shadow-mask", nargs="+", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--min-confidence", type=float, default=0.50,
                    help="hard gate on boundary confidence; default 0.50")
    ap.add_argument("--min-support", type=float, default=0.75,
                    help="hard gate on predicted boundary support within 0.35m")
    ap.add_argument("--distance-percentile", type=float, default=.95)
    ap.add_argument("--structure-aware", action="store_true")
    ap.add_argument("--visibility-mode", default="camera_cone_middle",
                    choices=("ray_arc", "angular_near_edge", "component_tangent_arcs", "camera_cone_middle"))
    ap.add_argument("--raster-size", type=int, default=900)
    ap.add_argument("--max-vehicle-points", type=int, default=100000)
    ap.add_argument("--coarse-az-step", type=int, default=5)
    ap.add_argument("--coarse-elev-min", type=int, default=20)
    ap.add_argument("--coarse-elev-max", type=int, default=70)
    ap.add_argument("--local-radius-deg", type=int, default=5)
    ap.add_argument("--local-basins", type=int, default=3)
    ap.add_argument("--top-candidates", type=int, default=25)
    args = ap.parse_args()
    groups = (args.vehicle_geometry, args.shadow_geometry, args.shadow_mask)
    validate_vehicle_count(len(args.vehicle_geometry))
    if len({len(group) for group in groups}) != 1:
        raise ValueError("vehicle geometry, shadow geometry, and shadow mask counts must match")
    if not 0.5 <= args.distance_percentile <= 1.0:
        raise ValueError("distance percentile must be in [0.50, 1.0]")
    for path in (*args.vehicle_geometry, *args.shadow_geometry, *args.shadow_mask):
        if not path.is_file():
            raise FileNotFoundError(path)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    commands: list[list[str]] = []
    for index, (vehicle, shadow, mask) in enumerate(zip(*groups), start=1):
        vehicle_out = out / f"vehicle_{index:02d}" / "fit"
        vehicle_out.mkdir(parents=True, exist_ok=True)
        command = fit_command([vehicle], [shadow], [mask], vehicle_out, args)
        commands.append(command)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            if error.returncode != VEHICLE_EVIDENCE_EXIT_CODE:
                raise
            rows.append({
                "vehicle_index": index,
                "status": "rejected",
                "gate_reasons": ["unusable_camera_visible_geometry"],
                "match_confidence": 0.0,
                "azimuth_deg": None,
                "elevation_deg": None,
                "joint_score": None,
                "fit_result": None,
                "inputs": {
                    "vehicle_geometry": str(vehicle.resolve()),
                    "shadow_geometry": str(shadow.resolve()),
                    "shadow_mask": str(mask.resolve()),
                },
                "confidence_terms": {"reason": "per_vehicle_fit_failed"},
                "failed_command": command,
                "returncode": error.returncode,
            })
            continue
        fit_path = vehicle_out / "fit_result.json"
        result = json.loads(fit_path.read_text())
        confidence, terms = confidence_from_fit(result)
        best = result["best"]
        frame = best["frames"][0]
        support = float(terms.get("support_0p35m", 0.0))
        reasons = gate_confidence(confidence, support, args.min_confidence, args.min_support)
        accepted = not reasons and math.isfinite(confidence)
        rows.append({
            "vehicle_index": index,
            "status": "accepted" if accepted else "rejected",
            "gate_reasons": reasons,
            "match_confidence": confidence,
            "azimuth_deg": float(best["azimuth_deg"]),
            "elevation_deg": float(best["elevation_deg"]),
            "joint_score": float(best["joint_score"]),
            "fit_result": str(fit_path),
            "inputs": {"vehicle_geometry": str(vehicle.resolve()), "shadow_geometry": str(shadow.resolve()), "shadow_mask": str(mask.resolve())},
            "confidence_terms": terms,
        })

    publication = publication_status(rows)
    accepted = [row for row in rows if row["status"] == "accepted"]
    result: dict = {
        **publication,
        "contract": {
            "vehicle_count": len(rows),
            "allowed_vehicle_count": "1..5",
            "hard_gate": {"min_confidence": args.min_confidence, "min_support_0p35m": args.min_support},
            "all_rejected_behavior": "publish no_valid_sun_information and no solar angle",
            "distance_percentile": args.distance_percentile,
            "structure_aware": args.structure_aware,
            "visibility_mode": args.visibility_mode,
            "shadow_mask_contract": "official SSISv2 associated shadow mask; no vehicle subtraction, connected-component cleanup, or morphology",
        },
        "vehicles": rows,
        "commands": commands,
    }
    if accepted:
        result["weighted_angle"] = weighted_angle(accepted)
        if len(accepted) >= 2:
            joint_out = out / "joint_fit"
            accepted_indices = [row["vehicle_index"] - 1 for row in accepted]
            joint_geometry = [args.vehicle_geometry[i] for i in accepted_indices]
            joint_shadow = [args.shadow_geometry[i] for i in accepted_indices]
            joint_masks = [args.shadow_mask[i] for i in accepted_indices]
            command = fit_command(joint_geometry, joint_shadow, joint_masks, joint_out, args)
            commands.append(command)
            try:
                subprocess.run(command, check=True)
                joint_path = joint_out / "fit_result.json"
                joint = json.loads(joint_path.read_text())
                result["joint_fit"] = {"fit_result": str(joint_path), "azimuth_deg": joint["best"]["azimuth_deg"],
                                        "elevation_deg": joint["best"]["elevation_deg"], "joint_score": joint["best"]["joint_score"],
                                        "vehicle_indices": [row["vehicle_index"] for row in accepted]}
            except subprocess.CalledProcessError as error:
                if error.returncode != VEHICLE_EVIDENCE_EXIT_CODE:
                    raise
                result["joint_fit"] = {
                    "status": "unavailable",
                    "reason": "shared camera-visible geometry was unusable; weighted accepted-vehicle angle remains published",
                    "returncode": error.returncode,
                    "vehicle_indices": [row["vehicle_index"] for row in accepted],
                }
    result["commands"] = commands
    (out / "multi_vehicle_fit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "README.md").write_text(
        "# SSE confidence-gated multi-vehicle fit\n\n"
        "Each vehicle is fitted independently with the same camera-visible contour, robust bidirectional distance, "
        "P95 percentile and optional line-structure settings. Only vehicles passing both hard gates enter the "
        "weighted and shared-angle joint outputs. If none pass, `status` is `no_valid_sun_information`.\n\n"
        "Each shadow input is the official SSISv2 shadow member associated with that vehicle and is consumed "
        "directly without source-mask postprocessing.\n\n"
        "The `vehicle_XX/fit` and `joint_fit` directories contain the complete candidate visualizations generated "
        "by the same objective used for selection.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
