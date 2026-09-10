#!/usr/bin/env python3
"""Fit sun angles against one fixed observed full shadow contour.

Unlike the legacy objective, the observed target is built once and is never
filtered by the candidate direction.  Each candidate is compared using its
complete predicted Gaussian-support contour, with a symmetric robust contour
distance.  IoU is recorded for audit only and does not select the optimum.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

try:
    from .fit_sun_from_depth_lifted_shadow import gaussian_union_mask, plane_xy, rasterize_points
    from .fit_sun_from_dense_shadow_boundary import basis, contour_xy, points_to_px
except ImportError:
    from fit_sun_from_depth_lifted_shadow import gaussian_union_mask, plane_xy, rasterize_points
    from fit_sun_from_dense_shadow_boundary import basis, contour_xy, points_to_px


def robust_score(pred: np.ndarray, obs: np.ndarray) -> dict:
    if len(pred) < 8 or len(obs) < 8:
        return None
    obs_use = obs
    pred_tree = cKDTree(pred)
    obs_tree = cKDTree(obs_use)
    d_pred = obs_tree.query(pred, k=1, workers=-1)[0]
    d_obs = pred_tree.query(obs_use, k=1, workers=-1)[0]
    p90 = float(np.quantile(d_pred, 0.90))
    med = float(np.median(d_pred))
    rev90 = float(np.quantile(d_obs, 0.90))
    support = float(np.mean(d_pred <= 0.35))
    score = -(0.40 * p90 + 0.25 * med + 0.25 * rev90 + 0.10 * (1.0 - support))
    return {
        "score": score,
        "pred_to_observed_p90_m": p90,
        "pred_to_observed_median_m": med,
        "observed_to_predicted_p90_m": rev90,
        "pred_boundary_support_0p35m": support,
        "observed_boundary_points_used": int(len(obs_use)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle-geometry", type=Path, required=True)
    ap.add_argument("--shadow-geometry", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--raster-size", type=int, default=900)
    ap.add_argument("--max-vehicle-points", type=int, default=100000)
    ap.add_argument("--coarse-az-step", type=int, default=5)
    ap.add_argument("--coarse-elev-min", type=int, default=20)
    ap.add_argument("--coarse-elev-max", type=int, default=70)
    ap.add_argument("--local-basins", type=int, default=3)
    ap.add_argument("--local-radius-deg", type=int, default=5)
    ap.add_argument("--local-step-deg", type=int, default=1)
    ap.add_argument("--contact-radius-m", type=float, default=0.20)
    ap.add_argument("--gaussian-support-sigma", type=float, default=2.0)
    ap.add_argument("--min-vehicle-height-m", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    vd, sd = np.load(args.vehicle_geometry), np.load(args.shadow_geometry)
    plane = sd["plane_z_ax_by_c"].astype(float); n, e1, e2, anchor = basis(plane)
    vehicle_all = vd["positions_world"].astype(float)
    heights_all = (vehicle_all - anchor) @ n
    keep = (heights_all >= args.min_vehicle_height_m) & (heights_all <= 5.0)
    vehicle, heights = vehicle_all[keep], heights_all[keep]
    cov = vd["covariances_world"][keep].astype(float)
    if len(vehicle) > args.max_vehicle_points:
        ii = np.random.default_rng(args.seed).choice(len(vehicle), args.max_vehicle_points, replace=False)
        vehicle, heights, cov = vehicle[ii], heights[ii], cov[ii]
    vehicle_xy = plane_xy(vehicle, e1, e2, anchor)
    shadow_xy = plane_xy(sd["near_ground_shadow_points_world"].astype(float), e1, e2, anchor)
    centre = np.median(vehicle_xy, axis=0)
    rel = shadow_xy - centre; shadow_xy = shadow_xy[np.linalg.norm(rel, axis=1) <= np.quantile(np.linalg.norm(rel, axis=1), .995)]
    stacked = np.vstack((vehicle_xy, shadow_xy)); lo, hi = np.quantile(stacked, (.002, .998), axis=0)
    pad = np.maximum((hi - lo) * .08, .6); lo, hi = lo - pad, hi + pad
    shadow_mask = rasterize_points(shadow_xy, lo, hi, args.raster_size)
    plane_basis = np.c_[e1, e2]
    vehicle_floor = gaussian_union_mask(vehicle_xy, cov, plane_basis, args.gaussian_support_sigma, lo, hi, args.raster_size)
    # The detector mask excludes the vehicle.  Restore the occluder support so
    # the observed and predicted objects are both complete cast-shadow domains.
    obs_mask = shadow_mask | vehicle_floor
    obs_boundary = contour_xy(obs_mask, lo, hi)
    rows = []
    def evaluate(az, elev):
        h = np.array((np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az))))
        cot = 1.0 / np.tan(np.deg2rad(elev))
        proj = vehicle_xy - heights[:, None] * cot * h[None, :]
        w2p = plane_basis - n[:, None] * cot * h[None, :]
        pred_raw = gaussian_union_mask(proj, cov, w2p, args.gaussian_support_sigma, lo, hi, args.raster_size)
        pred_mask = pred_raw & ~cv2.dilate(vehicle_floor.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        pred = contour_xy(pred_raw, lo, hi)
        m = robust_score(pred, obs_boundary)
        if m is None: return None
        # Direction-independent IoU diagnostic; it is intentionally not scored.
        inter = np.count_nonzero(pred_raw & obs_mask); union = np.count_nonzero(pred_raw | obs_mask)
        return m | {"azimuth_deg": float(az), "elevation_deg": float(elev), "predicted_boundary_points": int(len(pred)), "raw_predicted_mask_area_pixels": int(pred_raw.sum()), "predicted_mask_area_pixels": int(pred_mask.sum()), "iou_diagnostic": float(inter / union) if union else 0.0, "pred_mask": pred_mask, "raw_mask": pred_raw}
    for elev in range(args.coarse_elev_min, args.coarse_elev_max + 1, 5):
        for az in range(0, 360, args.coarse_az_step):
            r = evaluate(az, elev)
            if r: rows.append(r)
    seeds = sorted(rows, key=lambda x: x["score"], reverse=True)[:args.local_basins]
    refined = []
    for seed in seeds:
        for elev in range(max(2, int(seed["elevation_deg"])-args.local_radius_deg), int(seed["elevation_deg"])+args.local_radius_deg+1):
            for d in range(-args.local_radius_deg, args.local_radius_deg+1):
                r = evaluate((int(seed["azimuth_deg"])+d)%360, elev)
                if r: r["basin_seed_azimuth_deg"], r["basin_seed_elevation_deg"] = seed["azimuth_deg"], seed["elevation_deg"]; refined.append(r)
    best = max(refined or rows, key=lambda x: x["score"])
    for data, fn in ((rows, "coarse_scores.csv"), (refined, "refined_scores.csv")):
        fields = [k for k,v in data[0].items() if np.isscalar(v) and k not in ("pred_mask", "raw_mask")]
        with (args.output_dir/fn).open("w", newline="") as f: w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k:r[k] for k in fields} for r in data)
    def panel(r, title):
        out=np.full((700,700,3),250,np.uint8); o=cv2.resize(obs_mask.astype(np.uint8),(700,700),interpolation=cv2.INTER_NEAREST).astype(bool); out[o]=(90,90,90)
        raw=cv2.resize(r["raw_mask"].astype(np.uint8),(700,700),interpolation=cv2.INTER_NEAREST).astype(bool)
        pm=cv2.resize(r["pred_mask"].astype(np.uint8),(700,700),interpolation=cv2.INTER_NEAREST).astype(bool); vf=cv2.resize(vehicle_floor.astype(np.uint8),(700,700),interpolation=cv2.INTER_NEAREST).astype(bool)
        for mask,col,th in ((vf,(255,80,0),3),(raw,(0,0,255),3),(pm,(0,180,255),1)):
            cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(out,cs,-1,col,th)
        cv2.putText(out,title,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),2,cv2.LINE_AA); return out
    # Save candidate masks/centres for the panel and make Top-25 from this same run.
    # Re-evaluate candidates so their complete projected contours are available.
    chosen=sorted(refined or rows,key=lambda x:x["score"],reverse=True)[:25]
    for r in chosen:
        az,e=r["azimuth_deg"],r["elevation_deg"]; hh=np.array((np.cos(np.deg2rad(az)),np.sin(np.deg2rad(az)))); cot=1/np.tan(np.deg2rad(e)); r["predicted_xy"]=vehicle_xy-heights[:,None]*cot*hh[None,:]
    while len(chosen)<25: chosen.append(best)
    sheet=cv2.vconcat([cv2.hconcat([panel(r,f"az {r['azimuth_deg']:.0f} | el {r['elevation_deg']:.0f} | {r['score']:.3f} | IoU {r['iou_diagnostic']:.3f}") for r in chosen[i:i+5]]) for i in range(0,25,5)])
    cv2.imwrite(str(args.output_dir/"top25_fixed_full_contour_candidates.jpg"),sheet,[cv2.IMWRITE_JPEG_QUALITY,96])
    # Best panel uses same re-evaluated representation.
    az,e=best["azimuth_deg"],best["elevation_deg"]; hh=np.array((np.cos(np.deg2rad(az)),np.sin(np.deg2rad(az)))); best["predicted_xy"]=vehicle_xy-heights[:,None]*(1/np.tan(np.deg2rad(e)))*hh[None,:]
    cv2.imwrite(str(args.output_dir/"best_fixed_full_contour_fit.png"),panel(best,f"BEST az {az:.2f} | elev {e:.2f} | score {best['score']:.4f} | IoU {best['iou_diagnostic']:.4f}"))
    result={"best":{k:v for k,v in best.items() if k not in ("pred_mask","raw_mask","predicted_xy")},"coarse_count":len(rows),"refined_count":len(refined),"objective":"fixed observed full contour; symmetric robust bidirectional boundary distance; IoU diagnostic only","iou_role":"reported only, never used to select best","inputs":{"vehicle_geometry":str(args.vehicle_geometry.resolve()),"shadow_geometry":str(args.shadow_geometry.resolve())},"search":{"coarse_az_step":args.coarse_az_step,"coarse_elevation":[args.coarse_elev_min,args.coarse_elev_max,5],"local_radius_deg":args.local_radius_deg,"local_step_deg":args.local_step_deg},"footprint":{"mode":"gaussian_union","gaussian_support_sigma":args.gaussian_support_sigma,"support_gaussians":int(len(vehicle))}}
    (args.output_dir/"fit_result.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
