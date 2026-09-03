"""Command-line entry point for the PBR Vehicle SDK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analysis import analyze_scene
from .asset import convert_asset, load_asset
from .config import ConversionConfig, RenderConfig
from .direct_proxy import DirectProxyConfig, build_direct_white_2dgs
from .complete_asset import build_complete_asset, load_complete_asset
from .rendering import render, render_scene
from .scene import bake_scene, load_scene_ply
from .shading import bake


def _json(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _conversion_config(path: str | None) -> ConversionConfig:
    value = _json(path)
    return ConversionConfig(**value.get("conversion", value))


def _render_config(path: str | None) -> RenderConfig:
    value = _json(path)
    return RenderConfig.from_dict(value.get("render", value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pbr-vehicle", description="Convert, relight, render, and bake Gaussian vehicles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Legacy generic KNN conversion; use complete-asset for the formal single-PLY asset.")
    convert_parser.add_argument("input_ply")
    convert_parser.add_argument("output_asset")
    convert_parser.add_argument("--config")
    convert_parser.add_argument("--overwrite", action="store_true")

    direct_parser = subparsers.add_parser("direct-proxy", help="Legacy P-v3 proxy builder; not part of the formal single-PLY asset.")
    direct_parser.add_argument("input_ply")
    direct_parser.add_argument("output_asset")
    direct_parser.add_argument("--config")
    direct_parser.add_argument("--overwrite", action="store_true")

    complete_parser = subparsers.add_parser("complete-asset", help="Build the PBR delivery with a configs/ folder.")
    complete_parser.add_argument("input_ply")
    complete_parser.add_argument("output_asset")
    complete_parser.add_argument("--asset-id", required=True)
    complete_parser.add_argument("--config", required=True)

    complete_inspect = subparsers.add_parser("inspect-complete", help="Validate the PBR asset and configs/ contract.")
    complete_inspect.add_argument("asset")

    inspect_parser = subparsers.add_parser("inspect", help="Validate an asset and print its portable manifest.")
    inspect_parser.add_argument("asset")
    inspect_parser.add_argument("--skip-hash-check", action="store_true")

    inspect_scene_parser = subparsers.add_parser("inspect-scene", help="Validate an ordinary scene 3DGS PLY.")
    inspect_scene_parser.add_argument("scene_ply")

    render_parser = subparsers.add_parser("render", help="Render the vehicle with editable PBR parameters.")
    render_parser.add_argument("asset")
    render_parser.add_argument("output_png")
    render_parser.add_argument("--config")

    render_scene_parser = subparsers.add_parser(
        "render-scene", help="Render a vehicle in an ordinary Gaussian scene using the four integration switches."
    )
    render_scene_parser.add_argument("scene_ply")
    render_scene_parser.add_argument("asset")
    render_scene_parser.add_argument("output_png")
    render_scene_parser.add_argument("--config")

    bake_parser = subparsers.add_parser("bake", help="Bake one configured view to an ordinary 3DGS PLY.")
    bake_parser.add_argument("asset")
    bake_parser.add_argument("output_ply")
    bake_parser.add_argument("--config")

    bake_scene_parser = subparsers.add_parser(
        "bake-scene", help="Bake any combination of lighting, PBR, shadow, and projection to an ordinary scene PLY."
    )
    bake_scene_parser.add_argument("scene_ply")
    bake_scene_parser.add_argument("asset")
    bake_scene_parser.add_argument("output_ply")
    bake_scene_parser.add_argument("--config")

    analyze_parser = subparsers.add_parser(
        "analyze-scene", help="Reserved interface for automatic sun and vehicle PBR estimation."
    )
    analyze_parser.add_argument("scene_ply")
    analyze_parser.add_argument("--asset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "convert":
        asset = convert_asset(args.input_ply, args.output_asset, _conversion_config(args.config), args.overwrite)
        print(json.dumps({"asset": str(asset.root), "raw_count": asset.raw_count, "proxy_count": asset.proxy_count}, indent=2))
    elif args.command == "direct-proxy":
        value = _json(args.config)
        result = build_direct_white_2dgs(args.input_ply, args.output_asset, DirectProxyConfig(**value.get("direct_proxy", value)), args.overwrite)
        print(json.dumps({"asset": str(result.root), "gaussian_count": result.gaussian_count, "manifest": str(result.manifest_json)}, indent=2))
    elif args.command == "complete-asset":
        asset = build_complete_asset(args.input_ply, args.output_asset, args.asset_id, _json(args.config))
        print(json.dumps({"asset": str(asset.root), "asset_id": asset.asset_id, "files": sorted(path.name for path in asset.root.iterdir())}, indent=2))
    elif args.command == "inspect-complete":
        asset = load_complete_asset(args.asset)
        print(json.dumps({"asset": str(asset.root), "asset_id": asset.asset_id, "status": "PASS"}, indent=2))
    elif args.command == "inspect":
        asset = load_asset(args.asset, verify_hashes=not args.skip_hash_check)
        print(json.dumps(asset.manifest, indent=2))
    elif args.command == "inspect-scene":
        scene = load_scene_ply(args.scene_ply)
        print(json.dumps({"scene_ply": scene.source_path.name, "gaussian_count": scene.gaussian_count}, indent=2))
    elif args.command == "render":
        asset = load_asset(args.asset)
        render(asset, args.output_png, _render_config(args.config))
        print(json.dumps({"output_png": str(Path(args.output_png).resolve())}, indent=2))
    elif args.command == "render-scene":
        scene = load_scene_ply(args.scene_ply)
        asset = load_asset(args.asset)
        render_scene(scene, asset, args.output_png, _render_config(args.config))
        print(json.dumps({"output_png": str(Path(args.output_png).resolve())}, indent=2))
    elif args.command == "bake":
        asset = load_asset(args.asset)
        bake(asset, args.output_ply, _render_config(args.config))
        print(json.dumps({
            "output_ply": str(Path(args.output_ply).resolve()),
            "bake_manifest": str(Path(args.output_ply).resolve().with_suffix(Path(args.output_ply).suffix + ".bake.json")),
        }, indent=2))
    elif args.command == "bake-scene":
        scene = load_scene_ply(args.scene_ply)
        asset = load_asset(args.asset)
        result = bake_scene(scene, asset, args.output_ply, _render_config(args.config))
        print(json.dumps({
            "output_ply": str(result.output_ply.resolve()),
            "scene_gaussian_count": result.scene_gaussian_count,
            "vehicle_gaussian_count": result.vehicle_gaussian_count,
            "shadowed_gaussian_count": result.shadowed_gaussian_count,
        }, indent=2))
    elif args.command == "analyze-scene":
        scene = load_scene_ply(args.scene_ply)
        asset = load_asset(args.asset) if args.asset else None
        try:
            analyze_scene(scene, asset)
        except NotImplementedError as error:
            print(json.dumps({"status": "not_implemented", "message": str(error)}, indent=2), file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
