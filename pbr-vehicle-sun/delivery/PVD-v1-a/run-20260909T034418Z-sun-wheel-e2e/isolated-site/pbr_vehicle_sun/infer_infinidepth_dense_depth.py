#!/usr/bin/env python3
"""Export the dense metric camera-z prediction used by InfiniDepth."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


INPUT_SIZE = (768, 1024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt-depth", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-type", choices=("InfiniDepth", "InfiniDepth_DepthSensor"),
                        default="InfiniDepth_DepthSensor")
    parser.add_argument("--infinidepth-root", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path)
    parser.add_argument("--moge2-pretrained", type=Path)
    args = parser.parse_args()
    if args.model_type == "InfiniDepth_DepthSensor" and args.prompt_depth is None:
        parser.error("--prompt-depth is required for InfiniDepth_DepthSensor")
    if args.model_type == "InfiniDepth" and args.prompt_depth is not None:
        parser.error("pure-RGB InfiniDepth must not receive --prompt-depth")
    infinidepth_root = args.infinidepth_root.resolve()
    depth_checkpoint = (args.depth_checkpoint or infinidepth_root / "checkpoints/depth" /
                        ("infinidepth.ckpt" if args.model_type == "InfiniDepth" else
                         "infinidepth_depthsensor.ckpt")).resolve()
    moge2_pretrained = (args.moge2_pretrained or
                        infinidepth_root / "checkpoints/moge-2-vitl-normal/model.pt").resolve()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(infinidepth_root))
    from InfiniDepth.model.model import _make_dense_query_coord
    from InfiniDepth.utils.inference_utils import prepare_metric_depth_inputs
    from InfiniDepth.utils.io_utils import depth_to_disparity, load_image
    from InfiniDepth.utils.model_utils import build_model

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device(args.device)
    original = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(args.image)
    original_h, original_w = original.shape[:2]
    _, image, _ = load_image(str(args.image), INPUT_SIZE)
    image = image.to(device)
    gt_depth, prompt_depth, gt_mask, _, _ = prepare_metric_depth_inputs(
        input_depth_path=str(args.prompt_depth.resolve()) if args.prompt_depth is not None else None,
        input_size=INPUT_SIZE, image=image, device=device,
        moge2_pretrained=str(moge2_pretrained),
    )
    model = build_model(args.model_type, model_path=str(depth_checkpoint)).to(device).eval()
    gt_disparity = depth_to_disparity(gt_depth)
    prompt_disparity = depth_to_disparity(prompt_depth)
    batch, _, height, width = image.shape
    query = _make_dense_query_coord(batch, height, width, device)
    with torch.inference_mode():
        prediction, _ = model.inference(
            image=image, query_coord=query, gt_depth=gt_disparity, gt_depth_mask=gt_mask,
            prompt_depth=prompt_disparity, prompt_mask=prompt_disparity > 0,
            use_batch_infer=True, return_dino_tokens=False,
        )
        depth_model = model._prepare_dense_depthmap_for_gs(prediction, batch, height, width)[0, 0]
    depth_model = depth_model.detach().float().cpu().numpy()
    depth_original = cv2.resize(depth_model, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    model_path = args.output_dir / "predicted_depth_model.npy"
    original_path = args.output_dir / "predicted_depth_original.npy"
    np.save(model_path, depth_model.astype(np.float32), allow_pickle=False)
    np.save(original_path, depth_original.astype(np.float32), allow_pickle=False)
    valid = np.isfinite(depth_original) & (depth_original > 0)
    lo, hi = np.quantile(depth_original[valid], (0.01, 0.99))
    normalized = np.clip((depth_original - lo) / max(hi - lo, 1e-6), 0, 1)
    heatmap = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0
    cv2.putText(heatmap, f"InfiniDepth metric camera-z | display {lo:.2f}-{hi:.2f} m", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(heatmap, f"InfiniDepth metric camera-z | display {lo:.2f}-{hi:.2f} m", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.output_dir / "predicted_depth_heatmap.png"), heatmap)
    manifest = {
        "model": args.model_type, "checkpoint": str(depth_checkpoint),
        "checkpoint_sha256": sha256(depth_checkpoint), "image": str(args.image.resolve()),
        "uses_lidar": args.prompt_depth is not None,
        "prompt_depth": str(args.prompt_depth.resolve()) if args.prompt_depth is not None else None,
        "depth_prompt_source": "external depth map" if args.prompt_depth is not None else "MoGe-2 metric depth inferred from the same RGB image",
        "moge2_pretrained": str(moge2_pretrained) if args.prompt_depth is None else None,
        "moge2_sha256": sha256(moge2_pretrained) if args.prompt_depth is None else None,
        "input_size_hw": list(INPUT_SIZE),
        "model_depth_shape_hw": list(depth_model.shape), "original_depth_shape_hw": list(depth_original.shape),
        "depth_semantics": "metric camera-z depth in meters after model inference and reciprocal conversion",
        "valid_pixels": int(valid.sum()), "depth_m_quantiles": {str(q): float(np.quantile(depth_original[valid], q)) for q in (0.01, 0.5, 0.99)},
        "elapsed_s": time.perf_counter() - started,
        "outputs": {"model_depth": model_path.name, "original_depth": original_path.name, "heatmap": "predicted_depth_heatmap.png"},
    }
    (args.output_dir / "depth_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
