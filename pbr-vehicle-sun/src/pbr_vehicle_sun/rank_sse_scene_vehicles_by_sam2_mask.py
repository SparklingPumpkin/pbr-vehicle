#!/usr/bin/env python3
"""Segment every eligible physical-vehicle observation and rank by max mask area."""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path
from contextlib import nullcontext
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
try:
    from .device import resolve_device
except ImportError:
    from device import resolve_device

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--detections",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--config",default="configs/sam2.1/sam2.1_hiera_t.yaml"); ap.add_argument("--device",default="auto"); ap.add_argument("--top-vehicles",type=int,default=3); ap.add_argument("--min-mask-area-ratio",type=float,default=.01); args=ap.parse_args(); started=time.perf_counter(); args.output_dir.mkdir(parents=True,exist_ok=True)
    compute_device=resolve_device(args.device); print(f"SAM2 compute device: {compute_device.description}",flush=True)
    source=json.loads(args.detections.read_text()); detections=source["detections"]
    by_image=defaultdict(list)
    for record in detections: by_image[record["image"]].append(record)
    predictor=SAM2ImagePredictor(build_sam2(args.config,str(args.checkpoint),device=compute_device.torch)); best={}; evaluated=0
    for image_path, records in by_image.items():
        image=cv2.imread(image_path,cv2.IMREAD_COLOR)
        if image is None: continue
        predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        for record in records:
            box=np.asarray(record["bbox_xyxy"],np.float32)
            autocast=torch.autocast("cuda",dtype=torch.bfloat16) if compute_device.uses_cuda else nullcontext()
            with torch.inference_mode(), autocast:
                masks,scores,_=predictor.predict(box=box[None],multimask_output=True)
            index=int(np.argmax(scores)); mask=masks[index].astype(bool); evaluated+=1
            candidate=dict(record); candidate.update({"sam2_score":float(scores[index]),"mask_area_ratio":float(mask.mean())})
            old=best.get(record["instance_id"])
            if old is None or candidate["mask_area_ratio"]>old[0]["mask_area_ratio"]:
                best[record["instance_id"]]=(candidate,mask.copy())
    ranking=sorted(best.values(),key=lambda pair:pair[0]["mask_area_ratio"],reverse=True)
    eligible=[pair for pair in ranking if pair[0]["mask_area_ratio"]>=args.min_mask_area_ratio]
    selected=eligible[:args.top_vehicles]; panels=[]; selected_records=[]
    for rank,(record,mask) in enumerate(selected,1):
        record=dict(record); record["vehicle_rank"]=rank
        mask_name=f"vehicle_{rank:02d}_instance-{record['instance_id']}_t{record['timestep']:03d}_cam{record['camera']}_mask.png"; review_name=mask_name.replace("_mask.png","_review.jpg")
        cv2.imwrite(str(args.output_dir/mask_name),mask.astype(np.uint8)*255)
        image=cv2.imread(record["image"],cv2.IMREAD_COLOR); image[mask]=(image[mask]*.4+np.array((255,255,0))*.6).astype(np.uint8); x0,y0,x1,y1=np.rint(record["bbox_xyxy"]).astype(int); cv2.rectangle(image,(x0,y0),(x1,y1),(0,0,255),3); title=f"global rank {rank} id={record['instance_id']} t={record['timestep']} cam={record['camera']} mask={record['mask_area_ratio']:.2%}"; cv2.putText(image,title,(20,42),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,0,0),4,cv2.LINE_AA); cv2.putText(image,title,(20,42),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA); cv2.imwrite(str(args.output_dir/review_name),image,[cv2.IMWRITE_JPEG_QUALITY,96]); panels.append(cv2.resize(image,(1024,775))); record.update({"mask":mask_name,"review":review_name}); selected_records.append(record)
    if panels: cv2.imwrite(str(args.output_dir/"scene_global_top_vehicle_masks.jpg"),cv2.vconcat(panels),[cv2.IMWRITE_JPEG_QUALITY,95])
    audit=[item[0] for item in ranking]
    output={"contract":"all eligible YOLO observations were segmented by SAM2; group by physical instance; each instance represented by its maximum mask-area observation; rank instances globally; no same-frame requirement and no small-vehicle substitution","model":"SAM2.1 Hiera Tiny","checkpoint":str(args.checkpoint.resolve()),"detections_evaluated":evaluated,"physical_instances_segmented":len(ranking),"min_mask_area_ratio":args.min_mask_area_ratio,"vehicles_requested":args.top_vehicles,"vehicles_selected":len(selected_records),"selected":selected_records,"full_instance_ranking":audit,"contact_sheet":"scene_global_top_vehicle_masks.jpg" if panels else None,"elapsed_s":time.perf_counter()-started}
    (args.output_dir/"scene_global_vehicle_ranking.json").write_text(json.dumps(output,indent=2)+"\n"); print(json.dumps({k:output[k] for k in ("detections_evaluated","physical_instances_segmented","vehicles_selected","selected","elapsed_s")},indent=2))
if __name__=="__main__": main()
