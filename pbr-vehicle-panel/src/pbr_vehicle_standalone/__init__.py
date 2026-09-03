"""Framework-independent runtime for the interactive Gaussian PBR vehicle panel."""

__version__ = "1.0.0"

from .asset_io import load_scene, load_vehicle_asset, resolve_asset_folder
from .projection import build_projection_masks
from .shading import shade_vehicle
from .types import LightingState, MaterialState, TransformState, VehicleAsset

__all__ = [
    "LightingState",
    "MaterialState",
    "TransformState",
    "VehicleAsset",
    "build_projection_masks",
    "load_scene",
    "load_vehicle_asset",
    "resolve_asset_folder",
    "shade_vehicle",
]
