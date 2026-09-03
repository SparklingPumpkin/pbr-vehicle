from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class GaussianLayer:
    centers: np.ndarray
    covariances: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    normals: Optional[np.ndarray] = None
    albedo: Optional[np.ndarray] = None
    roughness: Optional[np.ndarray] = None
    metallic: Optional[np.ndarray] = None
    source_indices: Optional[np.ndarray] = None
    total_splats: int = 0
    path: str = ""


@dataclass
class VehicleAsset:
    root: Path
    asset_id: str
    original: GaussianLayer
    proxy: GaussianLayer
    original_to_proxy: np.ndarray
    mapping_indices: np.ndarray
    mapping_weights: np.ndarray
    canonical_config_path: Path
    canonical_config: Dict[str, Any]
    projection: Dict[str, Any]


@dataclass
class TransformState:
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_deg: dict[str, float] = field(
        default_factory=lambda: {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    )
    scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaterialState:
    use_asset_material: bool = True
    roughness: float = 0.40
    reflectance: float = 0.04
    metallic: float = 0.0
    exposure: float = 1.0
    ambient_fill: float = 0.35
    relight_strength: float = 1.0
    saturation: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LightingState:
    environment_intensity: float = 1.0
    environment_rgb: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    environment_temperature_k: float = 6500.0
    sun_enabled: bool = True
    sun_intensity: float = 1.0
    sun_rgb: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    sun_azimuth_deg: float = 45.0
    sun_elevation_deg: float = 35.0
    visibility: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sun_color_rgb"] = result.pop("sun_rgb")
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LightingState":
        from .math3d import rgb_to_color_temperature

        aliases = {
            "intensity": "environment_intensity",
            "sun_azimuth": "sun_azimuth_deg",
            "sun_elevation": "sun_elevation_deg",
        }
        values = {aliases.get(key, key): value for key, value in payload.items()}
        if "red" in values or "green" in values or "blue" in values:
            values["environment_rgb"] = [
                float(values.pop("red", 1.0)),
                float(values.pop("green", 1.0)),
                float(values.pop("blue", 1.0)),
            ]
        if values.get("environment_temperature_k") is None:
            values.pop("environment_temperature_k", None)
        if "environment_temperature" in values and "environment_temperature_k" not in values:
            values["environment_temperature_k"] = values.pop("environment_temperature")
        if "environment_temperature_k" not in values and "environment_rgb" in values:
            values["environment_temperature_k"] = rgb_to_color_temperature(values["environment_rgb"])
        if "sun_color_rgb" in values and "sun_rgb" not in values:
            values["sun_rgb"] = values.pop("sun_color_rgb")
        if "sun_red" in values or "sun_green" in values or "sun_blue" in values:
            values["sun_rgb"] = [
                float(values.pop("sun_red", 1.0)),
                float(values.pop("sun_green", 1.0)),
                float(values.pop("sun_blue", 1.0)),
            ]
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class VehicleState:
    vehicle_id: str
    asset_folder: str
    visible: bool = True
    display_mode: str = "Relight Original"
    transform: TransformState = field(default_factory=TransformState)
    material: MaterialState = field(default_factory=MaterialState)
    use_scene_lighting: bool = True
    lighting: LightingState = field(default_factory=LightingState)
    projection_visible: bool = True
    projection_opacity: float = 1.0
    projection: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["transform"] = self.transform.to_dict()
        result["material"] = self.material.to_dict()
        result["lighting"] = self.lighting.to_dict()
        return result
