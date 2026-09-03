"""Public API for the PBR Vehicle SDK."""

from .analysis import (
    SceneAnalysisResult,
    SunLightEstimate,
    SunPositionEstimate,
    VehiclePBREstimate,
    analyze_scene,
    estimate_sun_light_rgb,
    estimate_sun_position,
    estimate_vehicle_pbr,
)
from .asset import ASSET_SCHEMA, PBRAsset, convert_asset, load_asset
from .direct_proxy import DirectProxyConfig, DirectProxyResult, build_direct_white_2dgs
from .complete_asset import CompletePBRAsset, build_complete_asset, load_complete_asset
from .single_pbr import build_single_pbr_ply, gaussian_normals
from .projection import projection_mask_descriptor, validate_projection_mask_config
from .config import (
    ConversionConfig,
    IntegrationMode,
    LightConfig,
    MaterialConfig,
    RenderConfig,
    ShadowConfig,
    VehiclePlacement,
    ViewConfig,
)
from .rendering import render, render_scene
from .scene import GaussianScene, SceneBakeResult, bake_scene, compose_scene_properties, load_scene_ply
from .shading import RelightResult, bake, relight

__all__ = [
    "ASSET_SCHEMA",
    "SceneAnalysisResult",
    "SunLightEstimate",
    "SunPositionEstimate",
    "VehiclePBREstimate",
    "ConversionConfig",
    "CompletePBRAsset",
    "DirectProxyConfig",
    "DirectProxyResult",
    "IntegrationMode",
    "LightConfig",
    "MaterialConfig",
    "PBRAsset",
    "RelightResult",
    "RenderConfig",
    "GaussianScene",
    "SceneBakeResult",
    "ShadowConfig",
    "VehiclePlacement",
    "ViewConfig",
    "bake",
    "build_direct_white_2dgs",
    "build_complete_asset",
    "build_single_pbr_ply",
    "bake_scene",
    "analyze_scene",
    "compose_scene_properties",
    "convert_asset",
    "load_asset",
    "load_complete_asset",
    "gaussian_normals",
    "load_scene_ply",
    "estimate_sun_light_rgb",
    "estimate_sun_position",
    "estimate_vehicle_pbr",
    "relight",
    "render",
    "render_scene",
    "projection_mask_descriptor",
    "validate_projection_mask_config",
]

__version__ = "1.4.0"
