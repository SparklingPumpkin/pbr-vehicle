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
    parser.add_argument("--scene-cache-dir", type=Path, default=Path(".cache/pbr_vehicle_scenes"))
    parser.add_argument("--vehicle-spacing", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    app = StandaloneViewer(args)
    print(f"Standalone PBR vehicle viewer: http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
