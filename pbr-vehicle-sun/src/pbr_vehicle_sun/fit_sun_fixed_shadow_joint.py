#!/usr/bin/env python3
"""Shared numerical helpers and legacy fixed-contour sun fitting."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
try:
    from .fit_sun_fixed_full_contour import gaussian_union_mask
    from .fit_sun_from_dense_shadow_boundary import basis, contour_xy, plane_xy, points_to_px
except ImportError:
    from fit_sun_fixed_full_contour import gaussian_union_mask
    from fit_sun_from_dense_shadow_boundary import basis, contour_xy, plane_xy, points_to_px


def component_filter(sd, mask_path: Path, max_components: int = 1,
                     exclude_image_edge: bool = False):
    """Map lifted near-ground points to selected source-mask components."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    binary = mask > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n <= 1:
        return sd["near_ground_shadow_points_world"].astype(float), binary, {"components": 0}
    ranked = sorted(range(1, n), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    edge_labels = []
    if exclude_image_edge:
        edge_labels = sorted(set(np.r_[labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]].tolist()) - {0})
        ranked = [label for label in ranked if label not in edge_labels]
    retained = ranked if max_components <= 0 else ranked[:max_components]
    pix = sd["near_ground_shadow_pixels_xy"].astype(int)
    inside = (pix[:, 0] >= 0) & (pix[:, 0] < binary.shape[1]) & (pix[:, 1] >= 0) & (pix[:, 1] < binary.shape[0])
    keep = np.zeros(len(pix), bool)
    keep[inside] = np.isin(labels[pix[inside, 1], pix[inside, 0]], retained)
    pts = sd["near_ground_shadow_points_world"].astype(float)[keep]
    return pts, np.isin(labels, retained), {"components": int(n - 1), "retained_labels": retained,
                                    "excluded_image_edge_labels": edge_labels,
                                    "exclude_image_edge": bool(exclude_image_edge),
                                    "retained_area_pixels": [int(stats[label, cv2.CC_STAT_AREA]) for label in retained],
                                    "points_kept": int(len(pts)), "points_before": int(len(pix))}


def robust(pred, obs, percentile: float = .90):
    if len(pred) < 8 or len(obs) < 8: return None
    dt = cKDTree(pred); ot = cKDTree(obs)
    dp = ot.query(pred, k=1, workers=-1)[0]; do = dt.query(obs, k=1, workers=-1)[0]
    tail, med, reverse_tail = np.quantile(dp, percentile), np.median(dp), np.quantile(do, percentile)
    support = np.mean(dp <= .35)
    score = -(0.40*tail + 0.25*med + 0.25*reverse_tail + 0.10*(1-support))
    return {"score": float(score), "distance_percentile": float(percentile),
            "pred_to_observed_tail_m": float(tail), "observed_to_predicted_tail_m": float(reverse_tail),
            # Legacy keys remain aliases so existing evidence consumers keep
            # working; use the generic tail keys when percentile != 0.90.
            "pred_to_observed_p90_m": float(tail),
            "pred_to_observed_median_m": float(med), "observed_to_predicted_p90_m": float(reverse_tail),
            "pred_boundary_support_0p35m": float(support), "observed_boundary_points_used": int(len(obs))}


def setup(vd, sd, shadow_mask, size, max_points, seed, observed_components=1,
          subtract_observed_floor=True, exclude_image_edge=False,
          projection_elevation_min_deg: float | None = None,
          shadow_point_quantile: float = .995):
    plane = sd["plane_z_ax_by_c"].astype(float); n,e1,e2,anchor = basis(plane)
    v = vd["positions_world"].astype(float); h = (v-anchor) @ n
    keep = (h >= .02) & (h <= 5.0); v,h = v[keep],h[keep]
    cov = vd["covariances_world"].astype(float)[keep]
    if len(v)>max_points:
        ii=np.random.default_rng(seed).choice(len(v),max_points,replace=False); v,h,cov=v[ii],h[ii],cov[ii]
    s, _, comp = component_filter(sd, shadow_mask, observed_components, exclude_image_edge)
    # Keep source-image edge contacts as an auditable exclusion signal while
    # retaining the rest of a component (a component may legitimately extend
    # to the crop boundary but its boundary segment there is not evidence).
    edge_world = np.empty((0, 3), dtype=float)
    source_mask = cv2.imread(str(shadow_mask), cv2.IMREAD_GRAYSCALE)
    if source_mask is not None:
        pix = sd["near_ground_shadow_pixels_xy"].astype(int)
        h_img, w_img = source_mask.shape[:2]
        edge = ((pix[:, 0] <= 1) | (pix[:, 0] >= w_img - 2) |
                (pix[:, 1] <= 1) | (pix[:, 1] >= h_img - 2))
        edge_world = sd["near_ground_shadow_points_world"].astype(float)[edge]
    vxy=plane_xy(v,e1,e2,anchor); sxy=plane_xy(s,e1,e2,anchor)
    centre=np.median(vxy,axis=0); rel=sxy-centre; rr=np.linalg.norm(rel,axis=1)
    if shadow_point_quantile < 1.0:
        sxy=sxy[rr<=np.quantile(rr,shadow_point_quantile)]
    stacked=np.vstack((vxy,sxy)); lo,hi=np.quantile(stacked,(.002,.998),axis=0); pad=np.maximum((hi-lo)*.08,.6); lo-=pad; hi+=pad
    # A candidate was previously rasterised into bounds estimated only from
    # the unprojected vehicle and observed shadow. Low-elevation projections
    # could therefore leave the raster and acquire an artificial straight
    # crop boundary. Cover the complete azimuth search envelope instead.
    max_projection_shift_m = 0.0
    if projection_elevation_min_deg is not None:
        angle = np.deg2rad(max(float(projection_elevation_min_deg), 1e-3))
        max_projection_shift_m = float(np.max(np.maximum(h, 0.0)) / np.tan(angle))
        envelope_lo = np.min(vxy, axis=0) - max_projection_shift_m
        envelope_hi = np.max(vxy, axis=0) + max_projection_shift_m
        lo = np.minimum(lo, envelope_lo - .6)
        hi = np.maximum(hi, envelope_hi + .6)
    dense=np.zeros((size,size),np.uint8); pp=points_to_px(sxy,lo,hi,size); ok=(pp[:,0]>=0)&(pp[:,0]<size)&(pp[:,1]>=0)&(pp[:,1]<size); dense[pp[ok,1],pp[ok,0]]=1
    dense=cv2.dilate(dense,np.ones((3,3),np.uint8)); dense=cv2.morphologyEx(dense,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8));
    # Keep at most the largest two components, matching the upstream contract.
    cn,lab,st,_=cv2.connectedComponentsWithStats(dense,8)
    ids=sorted(range(1,cn),key=lambda i:int(st[i,cv2.CC_STAT_AREA]),reverse=True)
    if observed_components > 0:
        ids=ids[:observed_components]
    dense=np.isin(lab,ids)
    floor=gaussian_union_mask(vxy,cov,np.c_[e1,e2],2.0,lo,hi,size)
    obs=(dense & ~cv2.dilate(floor.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
         if subtract_observed_floor else dense.astype(bool))
    edge_xy = plane_xy(edge_world, e1, e2, anchor) if len(edge_world) else edge_world.reshape(0, 2)
    return {"n":n,"e1":e1,"e2":e2,"anchor":anchor,"vxy":vxy,"h":h,"cov":cov,"floor":floor,"obs_mask":obs,"obs_boundary":contour_xy(obs,lo,hi),"lo":lo,"hi":hi,"size":size,"component":comp,"points":len(s),"image_edge_shadow_xy":edge_xy,"source_image_shape":source_mask.shape[:2],"max_projection_shift_m":max_projection_shift_m,"projection_elevation_min_deg":projection_elevation_min_deg,"shadow_point_quantile":shadow_point_quantile}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--vehicle-geometry",nargs=2,type=Path,required=True); ap.add_argument("--shadow-geometry",nargs=2,type=Path,required=True); ap.add_argument("--shadow-mask",nargs=2,type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--raster-size",type=int,default=700); ap.add_argument("--max-vehicle-points",type=int,default=80000); ap.add_argument("--coarse-az-step",type=int,default=5); ap.add_argument("--coarse-elev-min",type=int,default=20); ap.add_argument("--coarse-elev-max",type=int,default=70); ap.add_argument("--local-radius-deg",type=int,default=5); ap.add_argument("--local-basins",type=int,default=3); ap.add_argument("--seed",type=int,default=20260907); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    frames=[setup(np.load(v),np.load(s),m,args.raster_size,args.max_vehicle_points,args.seed+i) for i,(v,s,m) in enumerate(zip(args.vehicle_geometry,args.shadow_geometry,args.shadow_mask))]
    def eval_one(f,az,el):
        hh=np.array((np.cos(np.deg2rad(az)),np.sin(np.deg2rad(az)))); cot=1/np.tan(np.deg2rad(el)); proj=f["vxy"]-f["h"][:,None]*cot*hh[None,:]; wp=np.c_[f["e1"],f["e2"]]-f["n"][:,None]*cot*hh[None,:]; raw=gaussian_union_mask(proj,f["cov"],wp,2.0,f["lo"],f["hi"],f["size"]); pred=raw & ~cv2.dilate(f["floor"].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool); m=robust(contour_xy(pred,f["lo"],f["hi"]),f["obs_boundary"]); return m|{"azimuth_deg":float(az),"elevation_deg":float(el),"raw_mask":raw,"pred_mask":pred,"proj":proj,"iou_diagnostic":float(np.count_nonzero(raw&f["obs_mask"])/max(np.count_nonzero(raw|f["obs_mask"]),1))}
    def evaluate(az,el):
        rs=[eval_one(f,az,el) for f in frames]; score=float(np.mean([r["score"] for r in rs])); return {"score":score,"azimuth_deg":float(az),"elevation_deg":float(el),"frames":rs}
    coarse=[evaluate(az,el) for el in range(args.coarse_elev_min,args.coarse_elev_max+1,5) for az in range(0,360,args.coarse_az_step)]; seeds=sorted(coarse,key=lambda x:x["score"],reverse=True)[:args.local_basins]; refined=[]
    for q in seeds:
        for el in range(max(2,int(q["elevation_deg"])-args.local_radius_deg),int(q["elevation_deg"])+args.local_radius_deg+1):
            for d in range(-args.local_radius_deg,args.local_radius_deg+1): refined.append(evaluate((int(q["azimuth_deg"])+d)%360,el))
    best=max(refined or coarse,key=lambda x:x["score"])
    scalar=lambda r:{k:v for k,v in r.items() if np.isscalar(v) and k not in ("raw_mask","pred_mask","proj")}
    for rows,name in ((coarse,"coarse_scores.csv"),(refined,"refined_scores.csv")):
        with (args.output_dir/name).open("w",newline="") as h:
            if rows:
                fields=list(scalar(rows[0]).keys())+[f"frame{i}_{k}" for i in range(2) for k in ("score","pred_to_observed_p90_m","pred_to_observed_median_m","observed_to_predicted_p90_m","pred_boundary_support_0p35m","iou_diagnostic")]; w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
                for r in rows: w.writerow({**scalar(r),**{f"frame{i}_{k}":r["frames"][i].get(k) for i in range(2) for k in ("score","pred_to_observed_p90_m","pred_to_observed_median_m","observed_to_predicted_p90_m","pred_boundary_support_0p35m","iou_diagnostic")}})
    def panel(f,r,title):
        out=np.full((700,700,3),250,np.uint8); o=cv2.resize(f["obs_mask"].astype(np.uint8),(700,700),interpolation=cv2.INTER_NEAREST).astype(bool); out[o]=(90,90,90); vf=cv2.resize(f["floor"].astype(np.uint8),(700,700),interpolation=cv2.INTER_NEAREST).astype(bool); raw=cv2.resize(r["raw_mask"].astype(np.uint8),(700,700),interpolation=cv2.INTER_NEAREST).astype(bool); pred=cv2.resize(r["pred_mask"].astype(np.uint8),(700,700),interpolation=cv2.INTER_NEAREST).astype(bool)
        for m,c,t in ((vf,(255,80,0),3),(raw,(0,0,255),3),(pred,(0,180,255),1)):
            cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(out,cs,-1,c,t)
        cv2.putText(out,title,(8,28),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),2,cv2.LINE_AA); return out
    chosen=sorted(refined or coarse,key=lambda x:x["score"],reverse=True)[:25]
    for i,f in enumerate(frames): cv2.imwrite(str(args.output_dir/f"frame{i}_best_fixed_shadow_fit.png"),panel(f,best["frames"][i],f"frame {i} | shared az {best['azimuth_deg']:.1f} el {best['elevation_deg']:.1f}"))
    cells=[]
    for q in chosen:
        cells.append(cv2.hconcat([panel(frames[i],q["frames"][i],f"az {q['azimuth_deg']:.0f} el {q['elevation_deg']:.0f} joint {q['score']:.3f}") for i in range(2)]))
    while len(cells)<25: cells.append(cells[-1])
    sheet=cv2.vconcat([cv2.hconcat(cells[i:i+5]) for i in range(0,25,5)]); cv2.imwrite(str(args.output_dir/"top25_joint_fixed_shadow_candidates.jpg"),sheet,[cv2.IMWRITE_JPEG_QUALITY,96])
    result={"best":{"azimuth_deg":best["azimuth_deg"],"elevation_deg":best["elevation_deg"],"joint_score":best["score"],"frames":[{k:v for k,v in r.items() if k not in ("raw_mask","pred_mask","proj")} for r in best["frames"]]},"objective":"fixed largest postprocessed shadow component minus fixed vehicle floor; mean bidirectional contour distance across two frames; IoU diagnostic only","top25_same_objective":True,"frames":[{"vehicle_geometry":str(p.resolve()),"shadow_geometry":str(s.resolve()),"shadow_mask":str(m.resolve()),"component":f["component"],"shadow_points_used":f["points"]} for p,s,m,f in zip(args.vehicle_geometry,args.shadow_geometry,args.shadow_mask,frames)],"search":{"coarse_az_step":args.coarse_az_step,"coarse_elevation":[args.coarse_elev_min,args.coarse_elev_max,5],"local_radius_deg":args.local_radius_deg,"local_basins":args.local_basins}}
    (args.output_dir/"fit_result.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
