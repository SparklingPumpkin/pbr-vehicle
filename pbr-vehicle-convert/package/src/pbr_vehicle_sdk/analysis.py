"""Reserved scene-analysis interfaces.

The public contracts are frozen in this release. Estimation implementations are
intentionally deferred and must not return fabricated values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asset import PBRAsset
from .scene import GaussianScene, load_scene_ply


_NOT_IMPLEMENTED = (
    "Automatic scene inference is an API placeholder in this release; "
    "no sun or material estimate has been computed."
)


@dataclass(frozen=True)
class SunPositionEstimate:
    azimuth_deg: float
    elevation_deg: float
    confidence: float


@dataclass(frozen=True)
class SunLightEstimate:
    rgb: tuple[float, float, float]
    intensity: float
    confidence: float


@dataclass(frozen=True)
class VehiclePBREstimate:
    roughness: float
    metallic: float
    reflectance: float
    clearcoat: float
    clearcoat_roughness: float
    confidence: float


@dataclass(frozen=True)
class SceneAnalysisResult:
    sun_position: SunPositionEstimate
    sun_light: SunLightEstimate
    vehicle_pbr: VehiclePBREstimate


def estimate_sun_position(scene: GaussianScene | str | Path) -> SunPositionEstimate:
    if not isinstance(scene, GaussianScene):
        load_scene_ply(scene)
    raise NotImplementedError(_NOT_IMPLEMENTED)


def estimate_sun_light_rgb(scene: GaussianScene | str | Path) -> SunLightEstimate:
    if not isinstance(scene, GaussianScene):
        load_scene_ply(scene)
    raise NotImplementedError(_NOT_IMPLEMENTED)


def estimate_vehicle_pbr(
    scene: GaussianScene | str | Path,
    vehicle_asset: PBRAsset | None = None,
) -> VehiclePBREstimate:
    if not isinstance(scene, GaussianScene):
        load_scene_ply(scene)
    raise NotImplementedError(_NOT_IMPLEMENTED)


def analyze_scene(
    scene: GaussianScene | str | Path,
    vehicle_asset: PBRAsset | None = None,
) -> SceneAnalysisResult:
    if not isinstance(scene, GaussianScene):
        load_scene_ply(scene)
    raise NotImplementedError(_NOT_IMPLEMENTED)