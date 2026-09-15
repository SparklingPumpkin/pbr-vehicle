from __future__ import annotations

import argparse
import time
from pathlib import Path

from .viewer import StandaloneViewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Standalone Gaussian PBR vehicle viewer")
    parser.add_argument("--scene", type=Path, default=None, help="Optional scene Gaussian .ply or DriveStudio .pth")
    parser.add_argument("--vehicle-asset-folder", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Optional asset/viewer JSON config for the first vehicle")
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument(
        "--device", default="auto",
        help="PBR compute device: auto (local CUDA if available), cpu, cuda, or cuda:N.",
    )
    parser.add_argument("--scene-cache-dir", type=Path, default=Path(".cache/pbr_vehicle_scenes"))
    parser.add_argument(
        "--scene-max-splats",
        type=int,
        default=0,
        help="Maximum scene Gaussians sent to the browser; use 0 for the full scene.",
    )
    parser.add_argument("--vehicle-spacing", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--enable-environment-map",
        action="store_true",
        help="Enable per-vehicle scene cubemap IBL initially; it remains switchable in the panel.",
    )
    parser.add_argument(
        "--environment-map-resolution",
        type=int,
        choices=(32, 64, 128, 256),
        default=128,
        help="Resolution of each captured cubemap face.",
    )
    parser.add_argument(
        "--sun-estimator-config",
        type=Path,
        default=None,
        help="Portable JSON config for pbr-vehicle-sun-scene; estimator artifacts use a temporary directory.",
    )
    parser.add_argument("--sun-estimator-scene", default="", help="Optional scene identity passed to the estimator.")
    parser.add_argument("--sun-estimator-data-root", type=Path, default=None, help="Dataset root used by the estimator.")
    parser.add_argument("--sun-estimator-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--auto-fit-config",
        type=Path,
        default=None,
        help="Portable JSON config for pbr-vehicle-auto-fit; artifacts use a temporary directory.",
    )
    parser.add_argument("--auto-fit-timeout", type=float, default=600.0)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    app = StandaloneViewer(args)
    print(
        f"Standalone PBR vehicle viewer (native resolution): "
        f"http://localhost:{args.port}/?fixedDpr=1",
        flush=True,
    )
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
