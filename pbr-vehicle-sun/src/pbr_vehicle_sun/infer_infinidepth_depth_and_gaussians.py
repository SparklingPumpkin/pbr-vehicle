#!/usr/bin/env python3
"""Run InfiniDepth once and export both dense depth and Gaussian geometry.

InfiniDepth ``inference_for_gs`` starts with the same dense, 2D-uniform depth
query used by the standalone depth inference path.  Keeping a separate depth
process therefore loads MoGe-2 and InfiniDepth twice and repeats that complete
query.  This adapter serializes the already-computed dense tensor before
continuing the unmodified 3D-uniform Gaussian path.

The output layout intentionally matches the two legacy commands:

* ``<output-dir>/depth/predicted_depth_{model,original}.npy``
* ``<output-dir>/depth/predicted_depth_heatmap.png``
* ``<output-dir>/depth/depth_manifest.json``
* full compatibility mode: ``<output-dir>/infinidepth_ply/scene_gaussians.ply``
* fast direct-mask mode: one ready-to-fit vehicle NPZ per requested mask
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .device import resolve_device
except ImportError:
    from device import resolve_device


DEFAULT_INPUT_SIZE = (768, 1024)


def save_dense_depth_outputs(
    depth_model: np.ndarray,
    original_shape_hw: tuple[int, int],
    output_dir: Path,
    manifest_fields: dict,
) -> dict:
    """Save the legacy dense-depth artifacts from a GS dense-depth tensor."""
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_model = np.asarray(depth_model, dtype=np.float32)
    if depth_model.ndim != 2:
        raise ValueError(f"dense depth must be HxW, got {depth_model.shape}")
    original_h, original_w = original_shape_hw
    depth_original = cv2.resize(
        depth_model, (original_w, original_h), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32, copy=False)
    model_path = output_dir / "predicted_depth_model.npy"
    original_path = output_dir / "predicted_depth_original.npy"
    np.save(model_path, depth_model, allow_pickle=False)
    np.save(original_path, depth_original, allow_pickle=False)

    valid = np.isfinite(depth_original) & (depth_original > 0)
    if not np.any(valid):
        raise RuntimeError("InfiniDepth returned no finite positive depth pixels")
    lo, hi = np.quantile(depth_original[valid], (0.01, 0.99))
    normalized = np.clip((depth_original - lo) / max(float(hi - lo), 1e-6), 0, 1)
    heatmap = cv2.applyColorMap(
        np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    heatmap[~valid] = 0
    label = f"InfiniDepth metric camera-z | display {lo:.2f}-{hi:.2f} m"
    cv2.putText(
        heatmap, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        heatmap, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    heatmap_name = "predicted_depth_heatmap.png"
    if not cv2.imwrite(str(output_dir / heatmap_name), heatmap):
        raise RuntimeError(f"failed to save depth heatmap under {output_dir}")

    manifest = {
        **manifest_fields,
        "model_depth_shape_hw": list(depth_model.shape),
        "original_depth_shape_hw": list(depth_original.shape),
        "depth_semantics": (
            "metric camera-z depth in meters from the dense 2D-uniform pass "
            "inside inference_for_gs"
        ),
        "valid_pixels": int(valid.sum()),
        "depth_m_quantiles": {
            str(q): float(np.quantile(depth_original[valid], q))
            for q in (0.01, 0.5, 0.99)
        },
        "outputs": {
            "model_depth": model_path.name,
            "original_depth": original_path.name,
            "heatmap": heatmap_name,
        },
    }
    (output_dir / "depth_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--infinidepth-root", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--gs-checkpoint", type=Path, required=True)
    parser.add_argument("--moge2-pretrained", type=Path, required=True)
    parser.add_argument(
        "--sky-checkpoint", type=Path,
        help="required only by full-PLY 3D-uniform sampling mode",
    )
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--sample-points", type=int, default=2_000_000)
    parser.add_argument(
        "--vehicle-mask", type=Path, action="append", default=[],
        help="Mask to extract directly from the dense one-Gaussian-per-pixel output; repeatable",
    )
    parser.add_argument(
        "--vehicle-output-dir", type=Path, action="append", default=[],
        help="Output directory corresponding to each --vehicle-mask",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--timestep", type=int)
    parser.add_argument("--camera", type=int)
    parser.add_argument("--min-opacity", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_SIZE[0])
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_SIZE[1])
    return parser.parse_args()


def quaternion_to_rotation(quaternions: np.ndarray) -> np.ndarray:
    q = quaternions.astype(np.float64, copy=False)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    w, x, y, z = q.T
    result = np.empty((len(q), 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1 - 2 * (y * y + z * z)
    result[:, 0, 1] = 2 * (x * y - w * z)
    result[:, 0, 2] = 2 * (x * z + w * y)
    result[:, 1, 0] = 2 * (x * y + w * z)
    result[:, 1, 1] = 1 - 2 * (x * x + z * z)
    result[:, 1, 2] = 2 * (y * z - w * x)
    result[:, 2, 0] = 2 * (x * z - w * y)
    result[:, 2, 1] = 2 * (y * z + w * x)
    result[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def camera_to_world(data_root: Path, timestep: int, camera: int) -> np.ndarray:
    ego_zero = np.loadtxt(data_root / "ego_pose/000.txt")
    ego_current = np.loadtxt(data_root / "ego_pose" / f"{timestep:03d}.txt")
    camera_to_ego = np.loadtxt(data_root / "extrinsics" / f"{camera}.txt")
    return np.linalg.inv(ego_zero) @ ego_current @ camera_to_ego


def save_direct_vehicle_geometry(
    dense_gaussians,
    source_grid_hw: tuple[int, int],
    vehicle_mask_path: Path,
    output_dir: Path,
    data_root: Path,
    timestep: int,
    camera: int,
    intrinsic_original: np.ndarray,
    min_opacity: float,
) -> dict:
    """Select dense source-view Gaussians without creating or rereading a PLY."""
    mask_original = cv2.imread(str(vehicle_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_original is None:
        raise FileNotFoundError(vehicle_mask_path)
    model_h, model_w = source_grid_hw
    mask_model = cv2.resize(
        mask_original, (model_w, model_h), interpolation=cv2.INTER_NEAREST
    ) > 0
    opacity_all = dense_gaussians.opacities[0].detach().float().cpu().numpy()
    selected = np.flatnonzero(mask_model.reshape(-1) & (opacity_all >= min_opacity))
    if len(selected) < 100:
        raise RuntimeError(f"only {len(selected)} direct mask-visible InfiniDepth Gaussians selected")

    indices = torch.as_tensor(selected, device=dense_gaussians.means.device)
    means_camera = dense_gaussians.means[0, indices].detach().float().cpu().numpy()
    scales = dense_gaussians.scales[0, indices].detach().float().cpu().numpy()
    rotations_camera_q = dense_gaussians.rotations[0, indices].detach().float().cpu().numpy()
    opacity = opacity_all[selected]
    c2w = camera_to_world(data_root, timestep, camera)
    positions_world = means_camera @ c2w[:3, :3].T + c2w[:3, 3]
    rotations_camera = quaternion_to_rotation(rotations_camera_q)
    covariance_camera = np.einsum(
        "nij,nj,nkj->nik", rotations_camera, scales.astype(np.float64) ** 2, rotations_camera
    )
    world_rotation = c2w[:3, :3]
    covariance_world = np.einsum(
        "ij,njk,lk->nil", world_rotation, covariance_camera, world_rotation
    )
    y_model, x_model = np.divmod(selected, model_w)
    original_h, original_w = mask_original.shape
    x_original = (x_model.astype(np.float64) + 0.5) * original_w / model_w - 0.5
    y_original = (y_model.astype(np.float64) + 0.5) * original_h / model_h - 0.5

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "visible_infinidepth_vehicle_gaussians.npz"
    np.savez_compressed(
        output_path,
        positions_world=positions_world.astype(np.float32),
        covariances_world=covariance_world.astype(np.float32),
        log_scales=np.log(np.clip(scales, 1e-20, None)).astype(np.float32),
        quaternions_camera=rotations_camera_q.astype(np.float32),
        opacity=opacity.astype(np.float32),
        depths_camera=means_camera[:, 2].astype(np.float32),
        pixels_xy=np.c_[x_original, y_original].astype(np.float32),
        camera_to_world=c2w,
        intrinsics=intrinsic_original,
    )
    manifest = {
        "vehicle_mask": str(vehicle_mask_path.resolve()),
        "timestep": timestep,
        "camera": camera,
        "dense_gaussians": int(model_h * model_w),
        "mask_visible_selected": int(len(selected)),
        "min_opacity": min_opacity,
        "selection_contract": (
            "same-frame vehicle mask resized by nearest neighbor to the dense "
            "InfiniDepth source grid; one surface Gaussian per pixel; opacity gate; "
            "no 3D-uniform resampling, PLY serialization, or PLY reprojection"
        ),
        "geometry_relation_to_legacy_ply": (
            "same InfiniDepth dense depth, DINO features, Gaussian predictor, camera "
            "intrinsics, and world transform; dense source-pixel sampling replaces "
            "legacy 3D-uniform resampling, so a regression gate is required"
        ),
        "output": output_path.name,
    }
    (output_dir / "infinidepth_vehicle_geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.sample_points <= 0:
        raise ValueError("--sample-points must be positive")
    selection = resolve_device(args.device)
    if not selection.uses_cuda:
        raise RuntimeError("CUDA is required for InfiniDepth Gaussian inference")
    device = torch.device(selection.torch)
    if len(args.vehicle_mask) != len(args.vehicle_output_dir):
        raise ValueError("--vehicle-mask and --vehicle-output-dir counts must match")
    direct_vehicle_mode = bool(args.vehicle_mask)
    if direct_vehicle_mode and (
        args.data_root is None or args.timestep is None or args.camera is None
    ):
        raise ValueError(
            "direct vehicle-mask extraction requires --data-root, --timestep, and --camera"
        )
    root = args.infinidepth_root.resolve()
    sys.path.insert(0, str(root))

    from InfiniDepth.gs import GSPixelAlignPredictor
    from InfiniDepth.utils.inference_utils import (
        build_camera_matrices,
        prepare_metric_depth_inputs,
    )
    from InfiniDepth.utils.io_utils import depth_to_disparity, load_image
    from InfiniDepth.utils.model_utils import build_model
    from InfiniDepth.model.model import _make_dense_query_coord

    required_paths = [
        args.image, args.depth_checkpoint, args.gs_checkpoint, args.moge2_pretrained,
        *args.vehicle_mask,
    ]
    if not direct_vehicle_mode:
        if args.sky_checkpoint is None:
            raise ValueError("full-PLY mode requires --sky-checkpoint")
        required_paths.append(args.sky_checkpoint)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = args.output_dir / "depth"
    ply_dir = args.output_dir / "infinidepth_ply"
    if not direct_vehicle_mode:
        ply_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    input_size = (args.input_height, args.input_width)
    original, image, (original_h, original_w) = load_image(str(args.image), input_size)
    del original
    image = image.to(device)
    batch, _, height, width = image.shape

    gt_depth, prompt_depth, gt_mask, use_gt_depth, _ = prepare_metric_depth_inputs(
        input_depth_path=None,
        input_size=input_size,
        image=image,
        device=device,
        moge2_pretrained=str(args.moge2_pretrained),
    )
    if use_gt_depth:
        raise RuntimeError("pure-RGB InfiniDepth unexpectedly received external depth")
    gt_disparity = depth_to_disparity(gt_depth)
    prompt_disparity = depth_to_disparity(prompt_depth)
    _, _, _, _, intrinsics, extrinsics = build_camera_matrices(
        fx_org=args.fx, fy_org=args.fy, cx_org=args.cx, cy_org=args.cy,
        org_h=original_h, org_w=original_w, h=height, w=width,
        batch=batch, device=device,
    )
    model = build_model("InfiniDepth", model_path=str(args.depth_checkpoint)).to(device).eval()
    if direct_vehicle_mode:
        dense_query = _make_dense_query_coord(batch, height, width, device)
        dense_prediction, _, dino_tokens = model.inference(
            image=image,
            query_coord=dense_query,
            gt_depth=gt_disparity,
            gt_depth_mask=gt_mask,
            prompt_depth=prompt_disparity,
            prompt_mask=prompt_disparity > 0,
            use_batch_infer=True,
            return_dino_tokens=True,
        )
        depthmap = model._prepare_dense_depthmap_for_gs(
            dense_prediction, batch, height, width
        )
        query_coord = predicted_query_depth = None
    else:
        from InfiniDepth.utils.inference_utils import run_optional_sampling_sky_mask
        sky_mask = run_optional_sampling_sky_mask(
            image=image, enable_skyseg_model=True,
            sky_model_ckpt_path=str(args.sky_checkpoint), dilate_px=0,
        )
        depthmap, dino_tokens, query_coord, predicted_query_depth = model.inference_for_gs(
            image=image,
            intrinsics=intrinsics,
            gt_depth=gt_disparity,
            gt_depth_mask=gt_mask,
            prompt_depth=prompt_disparity,
            prompt_mask=prompt_disparity > 0,
            sky_mask=sky_mask,
            sample_point_num=args.sample_points,
            coord_deterministic_sampling=True,
        )
        if query_coord is None or predicted_query_depth is None:
            raise RuntimeError("inference_for_gs did not return 3D-uniform query outputs")

    # This is the exact dense tensor that the former standalone depth process
    # recomputed.  Persist it before the Gaussian predictor consumes it.
    depth_model = depthmap[0, 0].detach().float().cpu().numpy()
    depth_manifest = save_dense_depth_outputs(
        depth_model,
        (original_h, original_w),
        depth_dir,
        {
            "model": "InfiniDepth",
            "checkpoint": str(args.depth_checkpoint.resolve()),
            "image": str(args.image.resolve()),
            "uses_lidar": False,
            "prompt_depth": None,
            "depth_prompt_source": "MoGe-2 metric depth inferred from the same RGB image",
            "moge2_pretrained": str(args.moge2_pretrained.resolve()),
            "input_size_hw": list(input_size),
            "shared_forward_with_gaussian_export": True,
        },
    )

    predictor = GSPixelAlignPredictor(dino_feature_dim=dino_tokens.shape[-1]).to(device)
    predictor.load_from_infinidepth_gs_checkpoint(str(args.gs_checkpoint))
    predictor.eval()
    dense_gaussians = predictor(
        image=image, depthmap=depthmap, dino_tokens=dino_tokens,
        intrinsics=intrinsics, extrinsics=extrinsics,
    )
    direct_manifests = []
    ply_path = None
    gaussian_count = int(dense_gaussians.means.shape[1])
    if direct_vehicle_mode:
        intrinsic_original = np.array(
            ((args.fx, 0.0, args.cx), (0.0, args.fy, args.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        for mask, vehicle_output_dir in zip(args.vehicle_mask, args.vehicle_output_dir):
            direct_manifests.append(save_direct_vehicle_geometry(
                dense_gaussians=dense_gaussians,
                source_grid_hw=(height, width),
                vehicle_mask_path=mask,
                output_dir=vehicle_output_dir,
                data_root=args.data_root,
                timestep=args.timestep,
                camera=args.camera,
                intrinsic_original=intrinsic_original,
                min_opacity=args.min_opacity,
            ))
    else:
        from InfiniDepth.gs import export_ply
        from InfiniDepth.utils.gs_utils import _build_sparse_uniform_gaussians
        from InfiniDepth.utils.inference_utils import (
            filter_gaussians_by_statistical_outlier,
            unpack_gaussians_for_export,
        )
        gaussians = _build_sparse_uniform_gaussians(
            dense_gaussians=dense_gaussians,
            query_3d_uniform_coord=query_coord,
            pred_depth_3d=predicted_query_depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            h=height,
            w=width,
        )
        gaussians = filter_gaussians_by_statistical_outlier(gaussians)
        means, harmonics, opacities, scales, rotations = unpack_gaussians_for_export(gaussians)
        gaussian_count = int(means.shape[0])
        ply_path = ply_dir / "scene_gaussians.ply"
        export_ply(
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            path=ply_path,
            scales=scales,
            rotations=rotations,
            focal_length_px=(args.fx, args.fy),
            principal_point_px=(args.cx, args.cy),
            image_shape=(original_h, original_w),
            extrinsic_matrix=extrinsics[0],
            shift_to_center=False,
        )
    summary = {
        "model": "InfiniDepth",
        "image": str(args.image.resolve()),
        "device": selection.torch,
        "uses_lidar": False,
        "single_shared_depth_forward": True,
        "mode": "direct_vehicle_masks" if direct_vehicle_mode else "full_ply",
        "sample_points_requested": 0 if direct_vehicle_mode else args.sample_points,
        "gaussians_generated": gaussian_count,
        "vehicle_geometries": direct_manifests,
        "depth_manifest": str((depth_dir / "depth_manifest.json").resolve()),
        "ply": str(ply_path.resolve()) if ply_path is not None else None,
        "elapsed_s": time.perf_counter() - started,
    }
    depth_manifest["joint_inference_elapsed_s"] = summary["elapsed_s"]
    (depth_dir / "depth_manifest.json").write_text(json.dumps(depth_manifest, indent=2) + "\n")
    (args.output_dir / "infinidepth_joint_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
