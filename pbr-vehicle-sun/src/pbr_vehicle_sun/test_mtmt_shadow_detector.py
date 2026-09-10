#!/usr/bin/env python3
"""Run the official MTMT shadow detector on images listed by a prior experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(image: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(image, (416, 416), interpolation=cv2.INTER_LINEAR)
    value = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (value - mean) / std


def colorize_probability(probability: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.rint(probability * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image_bgr.copy()
    tint = np.zeros_like(result)
    tint[:, :, 2] = 255
    result[mask] = (0.35 * result[mask] + 0.65 * tint[mask]).astype(np.uint8)
    return result


def title(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 700), 48), (0, 0, 0), -1)
    cv2.putText(result, text, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mtmt-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.50])
    args = parser.parse_args()

    repo = args.mtmt_repo.resolve()
    sys.path.insert(0, str(repo))
    from networks.MTMT import build_model

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model("resnext101")
    incompatibility = model.load_state_dict(checkpoint, strict=False)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatibility.missing_keys}, unexpected={incompatibility.unexpected_keys}"
        )
    model.to(device).eval()

    source = json.loads(args.source_manifest.read_text())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    rows: list[np.ndarray] = []

    for record in source["results"]:
        scene = str(record["scene"])
        rank = int(record["scene_rank"])
        instance = str(record["oracle_instance_id"])
        view = record["selected_view"]
        timestep = int(view["timestep"])
        camera = int(view["camera"])
        stem = f"scene-{scene}_rank-{rank:02d}_instance-{instance}_t{timestep:03d}_cam{camera}"
        image_path = Path(record["image"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        source_height, source_width = image_bgr.shape[:2]

        with torch.inference_mode():
            tensor = normalize(image_rgb).unsqueeze(0).to(device)
            _, _, _, final_heads = model(tensor)
            probability = torch.sigmoid(final_heads[-1])
            probability = F.interpolate(
                probability, size=(source_height, source_width), mode="bilinear", align_corners=False
            )[0, 0].detach().float().cpu().numpy()

        np.save(output / f"{stem}_probability.npy", probability.astype(np.float32))
        cv2.imwrite(str(output / f"{stem}_probability.png"), np.rint(probability * 255).astype(np.uint8))
        heatmap = colorize_probability(probability)
        panels = [title(image_bgr, "input RGB"), title(heatmap, "MTMT probability (red = high)")]
        binary_paths: dict[str, str] = {}
        area_ratios: dict[str, float] = {}
        for threshold in args.thresholds:
            mask = probability >= threshold
            suffix = f"threshold-{threshold:.2f}"
            mask_path = output / f"{stem}_{suffix}_mask.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            binary_paths[f"{threshold:.2f}"] = mask_path.name
            area_ratios[f"{threshold:.2f}"] = float(mask.mean())
            panels.append(title(overlay(image_bgr, mask), f"mask >= {threshold:.2f} ({mask.mean():.1%})"))

        min_height = min(panel.shape[0] for panel in panels)
        panels = [cv2.resize(panel, (int(panel.shape[1] * min_height / panel.shape[0]), min_height)) for panel in panels]
        review = cv2.hconcat(panels)
        review_path = output / f"{stem}_review.jpg"
        cv2.imwrite(str(review_path), review, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(review)
        records.append(
            {
                "stem": stem,
                "source_image": str(image_path),
                "vehicle_class": record["oracle_class_name"],
                "source_size": [source_width, source_height],
                "probability_mean": float(probability.mean()),
                "probability_max": float(probability.max()),
                "threshold_area_ratios": area_ratios,
                "probability_file": f"{stem}_probability.npy",
                "probability_preview_file": f"{stem}_probability.png",
                "binary_mask_files": binary_paths,
                "review_file": review_path.name,
            }
        )

    sheet_width = max(row.shape[1] for row in rows)
    sheet = cv2.vconcat([cv2.copyMakeBorder(row, 0, 0, 0, sheet_width - row.shape[1], cv2.BORDER_CONSTANT) for row in rows])
    cv2.imwrite(str(output / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "model": "MTMT: A Multi-task Mean Teacher for Semi-supervised Shadow Detection (CVPR 2020)",
                "repository": str(repo),
                "repository_head": "19e22592e87f8ad9b4cab87782775d51f767e297",
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "inference": {
                    "input_resize": [416, 416],
                    "normalization": "ImageNet mean/std from repository test_MT_util.py",
                    "head": "sigmoid(up_sal_final[-1])",
                    "postprocessing": "none; CRF intentionally disabled because it is not bundled by MTMT",
                    "thresholds": args.thresholds,
                },
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
