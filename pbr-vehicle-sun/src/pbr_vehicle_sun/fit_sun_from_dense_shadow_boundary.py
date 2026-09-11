#!/usr/bin/env python3
"""Fit sun angles against the continuous boundary of a camera shadow mask.

The previous fitter matched lifted vehicle Gaussian centers to sparse dark-road
centers.  This implementation keeps the vehicle geometry, but turns every
candidate projection into a filled convex footprint and compares its boundary
to the dense, camera-to-road-plane shadow boundary.  A directional half-plane
and a small contact exclusion keep the clipped vehicle footprint and road-band
false positives from dominating the objective.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def basis(plane: np.ndarray):
    n = np.array((-plane[0], -plane[1], 1.0), dtype=np.float64)
    n /= np.linalg.norm(n)
    e1 = np.array((1.0, 0.0, plane[0]), dtype=np.float64)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    a = np.array((0.0, 0.0, plane[2]), dtype=np.float64)
    return n, e1, e2, a


def plane_xy(points, e1, e2, anchor):
    d = points - anchor
    return np.c_[d @ e1, d @ e2]


def project_dense_mask(mask, c2w, intrinsic, e1, e2, anchor, lo, hi, size):
    yy, xx = np.indices((size, size), dtype=np.float64)
    p1 = lo[0] + xx.ravel() / (size - 1) * (hi[0] - lo[0])
    p2 = hi[1] - yy.ravel() / (size - 1) * (hi[1] - lo[1])
    world = anchor + p1[:, None] * e1 + p2[:, None] * e2
    camera = (np.linalg.inv(c2w) @ np.c_[world, np.ones(len(world))].T).T[:, :3]
    q = (intrinsic @ camera.T).T
    valid = camera[:, 2] > 0.1
    uv = np.zeros((len(world), 2), dtype=np.float64)
    uv[valid] = q[valid, :2] / q[valid, 2:3]
    px = np.rint(uv[:, 0]).astype(np.int32)
    py = np.rint(uv[:, 1]).astype(np.int32)
    valid &= np.isfinite(uv).all(axis=1)
    valid &= (px >= 0) & (px < mask.shape[1]) & (py >= 0) & (py < mask.shape[0])
    out = np.zeros(len(world), dtype=np.uint8)
    out[valid] = mask[py[valid], px[valid]]
    return out.reshape(size, size).astype(bool)


def points_to_px(xy, lo, hi, size):
    q = (xy - lo) / (hi - lo)
    return np.c_[np.rint(q[:, 0] * (size - 1)), np.rint((1 - q[:, 1]) * (size - 1))].astype(np.int32)


def footprint_mask(xy, lo, hi, size):
    p = points_to_px(xy, lo, hi, size)
    p = p[(p[:, 0] >= 0) & (p[:, 0] < size) & (p[:, 1] >= 0) & (p[:, 1] < size)]
    out = np.zeros((size, size), dtype=np.uint8)
    if len(p) >= 3:
        hull = cv2.convexHull(p)
        cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def contour_xy(binary, lo, hi):
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    # Subpixel coordinate at the pixel center; the raster is deliberately dense.
    x = lo[0] + c[:, 0] / (binary.shape[1] - 1) * (hi[0] - lo[0])
    y = hi[1] - c[:, 1] / (binary.shape[0] - 1) * (hi[1] - lo[1])
    return np.c_[x, y]


def candidate_score(pred_xy, observed_boundary, centre, shadow_vec, contact_radius):
    if len(pred_xy) < 3 or len(observed_boundary) < 8:
        return None
    pred_rel = pred_xy - centre
    obs_rel = observed_boundary - centre
    # The shadow is downstream of the occluder.  Keep a tolerance around the
    # perpendicular midline because the observed mask can be clipped at contact.
    obs_dir = obs_rel @ shadow_vec
    obs = observed_boundary[(obs_dir >= -0.15) & (np.linalg.norm(obs_rel, axis=1) >= contact_radius)]
    if len(obs) < 8:
        obs = observed_boundary[np.linalg.norm(obs_rel, axis=1) >= contact_radius]
    if len(obs) < 8:
        obs = observed_boundary
    tree = cKDTree(obs)
    d_pred = tree.query(pred_xy, k=1, workers=-1)[0]
    reverse = cKDTree(pred_xy).query(obs, k=1, workers=-1)[0]
    # Boundary agreement is primary; reverse support is deliberately weaker so
    # Broad detector spill should not force a large IoU-style penalty.
    p90 = float(np.quantile(d_pred, 0.90))
    med = float(np.median(d_pred))
    support = float(np.mean(d_pred <= 0.35))
    reverse_p75 = float(np.quantile(reverse, 0.75))
    score = -(0.45 * p90 + 0.25 * med + 0.20 * reverse_p75 + 0.10 * (1.0 - support))
    return {
        "score": score,
        "pred_to_observed_p90_m": p90,
        "pred_to_observed_median_m": med,
        "observed_to_predicted_p75_m": reverse_p75,
        "pred_boundary_support_0p35m": support,
        "observed_boundary_points_used": int(len(obs)),
    }


def evaluate(vehicle_xy0, vehicle_floor, heights, direction, elevation, n, e1, e2, anchor, lo, hi, size, obs_boundary, centre, contact_radius):
    sun = np.cos(np.deg2rad(direction)) * e1 + np.sin(np.deg2rad(direction)) * e2
    shadow_vec = -np.array((np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))))
    projected_xy = vehicle_xy0 - (heights / np.tan(np.deg2rad(elevation)))[:, None] * np.c_[sun @ e1, sun @ e2]
    pred_mask = footprint_mask(projected_xy, lo, hi, size)
    # Mirror the observed-mask contract: it is ``shadow AND NOT vehicle``.
    # Keeping the vehicle contact footprint in the prediction would make the
    # two outlines incomparable and reward a non-shadow edge.
    pred_mask &= ~cv2.dilate(vehicle_floor.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    pred_boundary = contour_xy(pred_mask, lo, hi)
    metrics = candidate_score(pred_boundary, obs_boundary, centre, shadow_vec, contact_radius)
    if metrics is None:
        return None
    return metrics | {"azimuth_deg": float(direction), "elevation_deg": float(elevation), "predicted_boundary_points": int(len(pred_boundary)), "predicted_xy": projected_xy, "pred_mask": pred_mask}


def unique_basins(rows, count):
    selected = []
    for row in sorted(rows, key=lambda x: x["score"], reverse=True):
        if all(abs(((row["azimuth_deg"] - old["azimuth_deg"] + 180) % 360) - 180) >= 12 or abs(row["elevation_deg"] - old["elevation_deg"]) >= 8 for old in selected):
            selected.append(row)
        if len(selected) >= count:
            break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle-geometry", type=Path, required=True)
    ap.add_argument("--road-geometry", type=Path, required=True)
    ap.add_argument("--shadow-mask", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--raster-size", type=int, default=900)
    ap.add_argument("--max-vehicle-points", type=int, default=50000)
    ap.add_argument("--coarse-az-step", type=int, default=5)
    ap.add_argument("--coarse-elev-min", type=int, default=20)
    ap.add_argument("--coarse-elev-max", type=int, default=50)
    ap.add_argument("--local-basins", type=int, default=3)
    ap.add_argument("--local-radius-deg", type=int, default=5)
    ap.add_argument("--local-step-deg", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--contact-radius-m", type=float, default=0.20)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vd, rd = np.load(args.vehicle_geometry), np.load(args.road_geometry)
    mask = cv2.imread(str(args.shadow_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.shadow_mask)
    mask = mask > 0
    plane = rd["plane_z_ax_by_c"].astype(np.float64)
    n, e1, e2, anchor = basis(plane)
    vehicle = vd["positions_world"].astype(np.float64)
    heights = (vehicle - anchor) @ n
    keep = (heights >= 0.35) & (heights <= 5.0)
    vehicle, heights = vehicle[keep], heights[keep]
    if len(vehicle) > args.max_vehicle_points:
        idx = np.random.default_rng(args.seed).choice(len(vehicle), args.max_vehicle_points, replace=False)
        vehicle, heights = vehicle[idx], heights[idx]
    vehicle_xy = plane_xy(vehicle, e1, e2, anchor)
    centre = np.median(vehicle_xy, axis=0)
    # Use the full road extent only to bound the displayed local search region;
    # projected footprints determine the downstream extent.
    road_xy = plane_xy(rd["road_points_world"].astype(np.float64), e1, e2, anchor)
    max_len = float(np.max(heights) / np.tan(np.deg2rad(args.coarse_elev_min)))
    lo = np.min(np.vstack((vehicle_xy, road_xy[np.linalg.norm(road_xy - centre, axis=1) < max_len * 2.5])), axis=0)
    hi = np.max(np.vstack((vehicle_xy, road_xy[np.linalg.norm(road_xy - centre, axis=1) < max_len * 2.5])), axis=0)
    pad = np.maximum((hi - lo) * 0.08, 0.8)
    lo, hi = lo - pad, hi + pad
    dense = project_dense_mask(mask, vd["camera_to_world"].astype(np.float64), vd["intrinsics"].astype(np.float64), e1, e2, anchor, lo, hi, args.raster_size)
    # Close single-pixel projection holes but preserve the measured silhouette.
    dense = cv2.morphologyEx(dense.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    vehicle_floor = footprint_mask(vehicle_xy, lo, hi, args.raster_size)
    obs_boundary = contour_xy(dense, lo, hi)
    if len(obs_boundary) < 8:
        raise RuntimeError("dense shadow mask did not project to a usable road-plane boundary")
    coarse = []
    for elev in range(args.coarse_elev_min, args.coarse_elev_max + 1, 5):
        for az in range(0, 360, args.coarse_az_step):
            row = evaluate(vehicle_xy, vehicle_floor, heights, az, elev, n, e1, e2, anchor, lo, hi, args.raster_size, obs_boundary, centre, args.contact_radius_m)
            if row:
                coarse.append(row)
    basins = unique_basins(coarse, args.local_basins)
    refined = []
    for basin in basins:
        for elev in range(int(basin["elevation_deg"]) - args.local_radius_deg, int(basin["elevation_deg"]) + args.local_radius_deg + 1, args.local_step_deg):
            if elev <= 1:
                continue
            for delta in range(-args.local_radius_deg, args.local_radius_deg + 1, args.local_step_deg):
                az = (int(round(basin["azimuth_deg"])) + delta) % 360
                row = evaluate(vehicle_xy, vehicle_floor, heights, az, elev, n, e1, e2, anchor, lo, hi, args.raster_size, obs_boundary, centre, args.contact_radius_m)
                if row:
                    row["basin_seed_azimuth_deg"] = basin["azimuth_deg"]
                    row["basin_seed_elevation_deg"] = basin["elevation_deg"]
                    refined.append(row)
    best = max(refined or coarse, key=lambda x: x["score"])
    for rows, name in ((coarse, "coarse_scores.csv"), (refined, "refined_scores.csv")):
        with (args.output_dir / name).open("w", newline="") as f:
            fields = [k for k, v in rows[0].items() if np.isscalar(v) and k not in ("pred_mask", "predicted_xy")] if rows else []
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows([{k: r[k] for k in fields} for r in rows])
    # Visual review: observed boundary in gray, projected continuous footprint in red,
    # vehicle footprint in blue, with a small panel for the best coarse basins.
    def panel(row, title):
        im = np.full((700, 700, 3), 250, np.uint8)
        display_dense = cv2.resize(dense.astype(np.uint8), (700, 700), interpolation=cv2.INTER_NEAREST).astype(bool)
        im[display_dense] = (90, 90, 90)
        vp = points_to_px(vehicle_xy, lo, hi, 700); pp = points_to_px(row["predicted_xy"], lo, hi, 700)
        cv2.polylines(im, [cv2.convexHull(vp)], True, (255, 80, 0), 3)
        cv2.polylines(im, [cv2.convexHull(pp)], True, (0, 0, 255), 3)
        cv2.putText(im, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2, cv2.LINE_AA)
        return im
    candidates = sorted(refined or coarse, key=lambda x: x["score"], reverse=True)[:25]
    while len(candidates) < 25: candidates.append(best)
    sheets = []
    for start in range(0, len(candidates), 25):
        ps = [panel(x, f"az {x['azimuth_deg']:.0f} | el {x['elevation_deg']:.0f} | {x['score']:.3f}") for x in candidates[start:start+25]]
        sheets.append(cv2.vconcat([cv2.hconcat(ps[r:r+5]) for r in range(0, 25, 5)]))
    cv2.imwrite(str(args.output_dir / "top25_boundary_candidates.jpg"), sheets[0], [cv2.IMWRITE_JPEG_QUALITY, 96])
    cv2.imwrite(str(args.output_dir / "best_dense_boundary_fit.png"), panel(best, f"BEST az {best['azimuth_deg']:.2f} | elev {best['elevation_deg']:.2f} | score {best['score']:.4f}"))
    result = {
        "best": {k: v for k, v in best.items() if k not in ("pred_mask", "predicted_xy")},
        "coarse_count": len(coarse), "refined_count": len(refined),
        "coarse_basins": [{k: v for k, v in b.items() if k not in ("pred_mask", "predicted_xy")} for b in basins],
        "search": {"coarse_azimuth": [0, 355, args.coarse_az_step], "coarse_elevation": [args.coarse_elev_min, args.coarse_elev_max, 5], "local_radius_deg": args.local_radius_deg, "local_step_deg": args.local_step_deg},
        "objective": "continuous projected convex vehicle-footprint boundary versus dense camera-mask road-plane boundary; directional shadow half-plane and weak reverse support; no whole-mask IoU",
        "inputs": {"vehicle_geometry": str(args.vehicle_geometry), "road_geometry": str(args.road_geometry), "shadow_mask": str(args.shadow_mask)},
        "plane_z_ax_by_c": plane.tolist(), "plane_bounds": {"min": lo.tolist(), "max": hi.tolist()}, "vehicle_points": int(len(vehicle)), "observed_boundary_points": int(len(obs_boundary)),
    }
    (args.output_dir / "fit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
