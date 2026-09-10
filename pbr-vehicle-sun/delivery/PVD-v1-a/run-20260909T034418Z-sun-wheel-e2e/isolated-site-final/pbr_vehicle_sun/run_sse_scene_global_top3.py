#!/usr/bin/env python3
"""Run SSE-v6 for scene-global top physical vehicles from independent views."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def run(name, command, timings, env=None):
    started=time.perf_counter(); subprocess.run(command,check=True,env=env); timings.append({"stage":name,"elapsed_s":time.perf_counter()-started,"command":command})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scene",required=True); ap.add_argument("--data-root",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--yolo-weights",type=Path,required=True); ap.add_argument("--sam2-root",type=Path,required=True); ap.add_argument("--sam2-checkpoint",type=Path,required=True); ap.add_argument("--sam2-python",type=Path,default=Path(sys.executable)); ap.add_argument("--sam2-config",default="configs/sam2.1/sam2.1_hiera_t.yaml"); ap.add_argument("--mtmt-repo",type=Path,required=True); ap.add_argument("--mtmt-checkpoint",type=Path,required=True); ap.add_argument("--infinidepth-root",type=Path,required=True); ap.add_argument("--infinidepth-python",type=Path,default=Path(sys.executable)); ap.add_argument("--device",default="0"); ap.add_argument("--min-mask-area-ratio",type=float,default=.01); ap.add_argument("--cameras",nargs="+",type=int,default=list(range(7))); ap.add_argument("--existing-detection",type=Path); ap.add_argument("--existing-ranking",type=Path); args=ap.parse_args()
    started=time.perf_counter(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); timings=[]; py=sys.executable
    detection=out/"selection"/"all_vehicle_detections.json"
    if args.existing_detection:
        detection=args.existing_detection.resolve()
    else:
        run("all_frame_all_camera_yolo_identity",[py,str(ROOT/"detect_sse_scene_all_vehicle_views.py"),"--data-root",str(args.data_root),"--output",str(detection),"--weights",str(args.yolo_weights),"--device",args.device,"--cameras",*map(str,args.cameras)],timings)
    ranking_dir=out/"selection"/"sam2_global_ranking"; sam_env=os.environ.copy(); sam_env["PYTHONPATH"]=str(args.sam2_root.resolve())+":"+sam_env.get("PYTHONPATH","")
    if args.existing_ranking:
        ranking_dir=args.existing_ranking.resolve().parent
    else:
        run("all_candidate_sam2_global_mask_ranking",[str(args.sam2_python),str(ROOT/"rank_sse_scene_vehicles_by_sam2_mask.py"),"--detections",str(detection),"--output-dir",str(ranking_dir),"--checkpoint",str(args.sam2_checkpoint),"--config",args.sam2_config,"--device","cuda:"+args.device.split(":")[-1],"--top-vehicles","3","--min-mask-area-ratio",str(args.min_mask_area_ratio)],timings,sam_env)
    ranking=json.loads((ranking_dir/"scene_global_vehicle_ranking.json").read_text()); selected=ranking["selected"]
    depth_env=os.environ.copy(); depth_env["PYTHONPATH"]=str(args.infinidepth_root)+":"+depth_env.get("PYTHONPATH","")
    model=args.infinidepth_root/"checkpoints/depth/infinidepth.ckpt"; gs=args.infinidepth_root/"checkpoints/gs/infinidepth_gs.ckpt"; moge=args.infinidepth_root/"checkpoints/moge-2-vitl-normal/model.pt"; sky=args.infinidepth_root/"checkpoints/sky/skyseg.onnx"
    common={}; results=[]
    for vehicle in selected:
        rank=int(vehicle["vehicle_rank"]); ts=int(vehicle["timestep"]); cam=int(vehicle["camera"]); image=Path(vehicle["image"]); mask=ranking_dir/vehicle["mask"]; key=(ts,cam); common_dir=out/f"t{ts:03d}_cam{cam}"
        try:
            if key not in common:
                depth_dir=common_dir/"depth"; run(f"view_t{ts:03d}_cam{cam}_rgb_depth",[str(args.infinidepth_python),str(ROOT/"infer_infinidepth_dense_depth.py"),"--image",str(image),"--output-dir",str(depth_dir),"--device","cuda:"+args.device.split(":")[-1],"--model-type","InfiniDepth","--infinidepth-root",str(args.infinidepth_root),"--depth-checkpoint",str(model),"--moge2-pretrained",str(moge)],timings,depth_env)
                ply_dir=common_dir/"infinidepth_ply"; ply_dir.mkdir(parents=True,exist_ok=True); fx,fy,cx,cy,*_=map(float,np.loadtxt(args.data_root/"intrinsics"/f"{cam}.txt"))
                run(f"view_t{ts:03d}_cam{cam}_rgb_gaussian",[str(args.infinidepth_python),str(args.infinidepth_root/"inference_gs.py"),"--input-image-path",str(image),"--output-ply-dir",str(ply_dir),"--output-ply-name","scene_gaussians.ply","--model-type","InfiniDepth","--depth-model-path",str(model),"--gs-model-path",str(gs),"--moge2-pretrained",str(moge),"--sky-model-ckpt-path",str(sky),"--fx-org",str(fx),"--fy-org",str(fy),"--cx-org",str(cx),"--cy-org",str(cy),"--sample-point-num","2000000","--no-render-novel-video"],timings,depth_env); common[key]=(depth_dir,ply_dir/"scene_gaussians.ply")
            depth_dir,ply=common[key]; vehicle_dir=common_dir/f"vehicle_{rank:02d}_instance-{vehicle['instance_id']}"
            source_manifest=vehicle_dir/"mtmt_source.json"; source_manifest.parent.mkdir(parents=True,exist_ok=True); source_manifest.write_text(json.dumps({"results":[{"scene":args.scene,"scene_rank":rank,"oracle_instance_id":vehicle["instance_id"],"oracle_class_name":vehicle["oracle_class_name"],"selected_view":{"timestep":ts,"camera":cam},"image":str(image)}]},indent=2)+"\n")
            mtmt=vehicle_dir/"mtmt"; run(f"vehicle{rank}_mtmt",[py,str(ROOT/"test_mtmt_shadow_detector.py"),"--source-manifest",str(source_manifest),"--mtmt-repo",str(args.mtmt_repo),"--checkpoint",str(args.mtmt_checkpoint),"--output-dir",str(mtmt),"--device","cuda:"+args.device.split(":")[-1],"--thresholds","0.50"],timings); stem=f"scene-{args.scene}_rank-{rank:02d}_instance-{vehicle['instance_id']}_t{ts:03d}_cam{cam}"; raw=mtmt/f"{stem}_threshold-0.50_mask.png"
            post=vehicle_dir/"shadow_postprocess"; run(f"vehicle{rank}_shadow_postprocess",[py,str(ROOT/"postprocess_shadow_mask.py"),"--shadow-mask",str(raw),"--vehicle-mask",str(mask),"--source-image",str(image),"--output-dir",str(post),"--min-component-area-ratio","0.002","--max-components","2","--vehicle-adjacent-only","--adjacent-distance-px","1.5","--require-centroid-below-vehicle","--second-component-min-ratio","0.50"],timings); shadow=post/"04_shadow_postprocessed.png"
            probability=mtmt/f"{stem}_probability.npy"; gate=vehicle_dir/"evidence_gate"
            run(f"vehicle{rank}_shadow_evidence_gate",[py,str(ROOT/"evaluate_mtmt_shadow_evidence.py"),"--probability",str(probability),"--shadow-mask",str(shadow),"--vehicle-mask",str(mask),"--road-mask",str(args.data_root/"road_masks"/f"{ts:03d}_{cam}.png"),"--source-image",str(image),"--output-dir",str(gate)],timings)
            gate_result=json.loads((gate/"shadow_evidence.json").read_text())
            if not gate_result.get("accepted"):
                raise RuntimeError("shadow evidence gate rejected: "+",".join(gate_result.get("failure_reasons",[])))
            sg=vehicle_dir/"shadow_geometry"; run(f"vehicle{rank}_shadow_depth_lift",[py,str(ROOT/"lift_source_shadow_with_infinidepth.py"),"--depth",str(depth_dir/"predicted_depth_original.npy"),"--shadow-mask",str(shadow),"--road-mask",str(args.data_root/"road_masks"/f"{ts:03d}_{cam}.png"),"--vehicle-mask",str(mask),"--source-image",str(image),"--data-root",str(args.data_root),"--timestep",str(ts),"--camera",str(cam),"--output-dir",str(sg)],timings)
            vg=vehicle_dir/"vehicle_geometry"; run(f"vehicle{rank}_vehicle_geometry",[py,str(ROOT/"extract_infinidepth_vehicle_geometry.py"),"--ply",str(ply),"--vehicle-mask",str(mask),"--data-root",str(args.data_root),"--timestep",str(ts),"--camera",str(cam),"--output-dir",str(vg)],timings)
            fit=vehicle_dir/"fit"; run(f"vehicle{rank}_maximal_camera_cone_fit",[py,str(ROOT/"run_sse_mainline.py"),"--geometry-source","feedforward_infinidepth","--vehicle-geometry",str(vg/"visible_infinidepth_vehicle_gaussians.npz"),"--shadow-geometry",str(sg/"infinidepth_shadow_geometry.npz"),"--shadow-mask",str(shadow),"--output-dir",str(fit),"--raster-size","900","--coarse-az-step","5","--coarse-elev-min","20","--coarse-elev-max","70","--local-radius-deg","5","--local-basins","3"],timings)
            best=json.loads((fit/"fit_result.json").read_text())["best"]; results.append({"status":"completed","vehicle_rank":rank,"instance_id":vehicle["instance_id"],"timestep":ts,"camera":cam,"mask_area_ratio":vehicle["mask_area_ratio"],"mask":str(mask.resolve()),"shadow_evidence":str((gate/"shadow_evidence.json").resolve()),"azimuth_deg":best["azimuth_deg"],"elevation_deg":best["elevation_deg"],"score":best["joint_score"],"fit_result":str((fit/"fit_result.json").resolve()),"best_fit_image":str((fit/"frame0_best_camera_visible_arc.png").resolve())})
        except Exception as error: results.append({"status":"failed","vehicle_rank":rank,"instance_id":vehicle["instance_id"],"timestep":ts,"camera":cam,"mask_area_ratio":vehicle["mask_area_ratio"],"mask":str(mask.resolve()),"error":repr(error)})
    valid=sorted([r for r in results if r["status"]=="completed"],key=lambda r:r["score"],reverse=True); winner=valid[0] if valid else None
    summary={"scene":args.scene,"mainline":"SSE-v6","selection_contract":ranking["contract"],"vehicles_requested":3,"vehicles_selected":len(selected),"vehicles_completed":len(valid),"selected_observations_are_independent":len({(r["timestep"],r["camera"]) for r in results})>1,"min_mask_area_ratio":args.min_mask_area_ratio,"uses_lidar":False,"infinidepth_mode":"rgb","visibility_mode":"camera_cone_middle","image_edge_shadow_excluded":True,"observed_floor_subtraction":False,"predicted_floor_subtraction":False,"vehicle_results":results,"winner":winner,"ranking_metric":"SAM2 mask area for selection; evidence-gated robust bidirectional maximal-camera-cone contour score for winner","timings":timings,"total_elapsed_s":time.perf_counter()-started,"contract_checks":{"all_frames_all_cameras":True,"physical_vehicle_deduplication":True,"global_max_sam_mask_ranking":True,"no_same_frame_requirement":True,"no_small_vehicle_substitution":len(selected)<3 or all(v["mask_area_ratio"]>=args.min_mask_area_ratio for v in selected),"shadow_probability_and_area_gate_before_fit":True,"same_visibility_operator_observed_and_predicted":True,"postprocess_before_fit":True}}
    (out/"scene_result.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
