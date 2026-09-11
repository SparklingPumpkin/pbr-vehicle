#!/usr/bin/env python3
"""Multi-frame adapter that preserves the original SSE-v4 Gaussian fitter."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_shadow_lineage(shadow_npz: Path, shadow_mask: Path) -> dict:
    data = np.load(shadow_npz)
    pixels = data["near_ground_shadow_pixels_xy"].astype(int)
    mask = cv2.imread(str(shadow_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(shadow_mask)
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < mask.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < mask.shape[0])
    contained = valid & (mask[np.clip(pixels[:, 1], 0, mask.shape[0] - 1), np.clip(pixels[:, 0], 0, mask.shape[1] - 1)] > 0)
    fraction = float(np.mean(contained))
    if fraction < 0.99:
        raise RuntimeError(f"shadow geometry is not derived from the supplied SSISv2 associated mask: {fraction:.4%}")
    return {"near_ground_points": int(len(pixels)), "mask_containment_fraction": fraction, "mask_sha256": sha256(shadow_mask)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle-geometry", nargs="+", type=Path, required=True)
    parser.add_argument("--shadow-geometry", nargs="+", type=Path, required=True)
    parser.add_argument("--shadow-mask", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raster-size", type=int, default=900)
    parser.add_argument("--max-vehicle-points", type=int, default=50000)
    parser.add_argument("--coarse-az-step", type=int, default=5)
    parser.add_argument("--coarse-elev-min", type=int, default=20)
    parser.add_argument("--coarse-elev-max", type=int, default=50)
    parser.add_argument("--local-radius-deg", type=int, default=5)
    parser.add_argument("--local-basins", type=int, default=3)
    args = parser.parse_args()
    counts = {len(args.vehicle_geometry), len(args.shadow_geometry), len(args.shadow_mask)}
    if len(counts) != 1 or len(args.vehicle_geometry) < 2:
        raise ValueError("native Gaussian multi-frame adapter requires aligned lists of at least two frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    lineage = []
    for index, (vehicle, shadow, mask) in enumerate(zip(args.vehicle_geometry, args.shadow_geometry, args.shadow_mask)):
        lineage.append(verify_shadow_lineage(shadow, mask))
        frame_dir = args.output_dir / f"frame{index}_native_gaussian"
        command = [
            sys.executable, str(SCRIPTS / "fit_sun_from_depth_lifted_shadow.py"),
            "--vehicle-geometry", str(vehicle), "--shadow-geometry", str(shadow),
            "--output-dir", str(frame_dir), "--raster-size", str(args.raster_size),
            "--max-vehicle-points", str(args.max_vehicle_points),
            "--coarse-az-step", str(args.coarse_az_step),
            "--coarse-elev-min", str(args.coarse_elev_min), "--coarse-elev-max", str(args.coarse_elev_max),
            "--local-radius-deg", str(args.local_radius_deg), "--local-basins", str(args.local_basins),
        ]
        subprocess.run(command, check=True)
        results.append(json.loads((frame_dir / "fit_result.json").read_text()))
    azimuths = np.deg2rad([result["best"]["azimuth_deg"] for result in results])
    mean_azimuth = float(np.rad2deg(np.arctan2(np.mean(np.sin(azimuths)), np.mean(np.cos(azimuths)))) % 360.0)
    azimuth_residuals = np.abs((np.rad2deg(azimuths) - mean_azimuth + 180.0) % 360.0 - 180.0)
    elevations = np.array([result["best"]["elevation_deg"] for result in results], dtype=float)
    comparison = {
        "mainline_branch": "native_gaussian",
        "algorithm_contract": "unchanged SSE-v4 per-frame depth-lifted shadow fitter using caller-provided native Gaussian geometry",
        "execution_mode": "multi_frame",
        "frame_count": len(results),
        "frames": [result["best"] for result in results],
        "shadow_lineage_gate": lineage,
        "multi_frame_summary": {
            "circular_mean_azimuth_deg": mean_azimuth,
            "median_elevation_deg": float(np.median(elevations)),
            "max_azimuth_residual_deg": float(np.max(azimuth_residuals)),
            "elevation_range_deg": float(np.ptp(elevations)),
        },
    }
    (args.output_dir / "fit_result.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
