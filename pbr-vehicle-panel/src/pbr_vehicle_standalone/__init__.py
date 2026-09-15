"""Framework-independent runtime for the interactive Gaussian PBR vehicle panel."""

__version__ = "1.9.1"

from .asset_io import load_scene, load_vehicle_asset, resolve_asset_folder
from .auto_fit import read_auto_fit_result, run_vehicle_auto_fit
from .environment_map import EnvironmentMap
from .projection import build_projection_masks
from .shading import shade_vehicle
from .sun_estimation import read_sun_estimate, run_sun_estimator
from .types import LightingState, MaterialState, TransformState, VehicleAsset

__all__ = [
    "LightingState",
    "EnvironmentMap",
    "MaterialState",
    "TransformState",
    "VehicleAsset",
    "build_projection_masks",
    "load_scene",
    "load_vehicle_asset",
    "resolve_asset_folder",
    "read_auto_fit_result",
    "run_vehicle_auto_fit",
    "read_sun_estimate",
    "run_sun_estimator",
    "shade_vehicle",
]
