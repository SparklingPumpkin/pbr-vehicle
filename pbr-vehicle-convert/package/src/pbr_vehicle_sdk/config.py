"""Validated, JSON-serializable SDK configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def _vec3(name: str, value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return tuple(float(component) for component in value)


def _range(name: str, value: float, minimum: float, maximum: float) -> float:
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {result}")
    return result


@dataclass(frozen=True)
class MaterialConfig:
    roughness: float = 0.38
    metallic: float = 0.02
    reflectance: float = 0.04
    clearcoat: float = 0.75
    clearcoat_roughness: float = 0.16
    specular_gain: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "roughness", _range("roughness", self.roughness, 0.025, 0.98))
        object.__setattr__(self, "metallic", _range("metallic", self.metallic, 0.0, 1.0))
        object.__setattr__(self, "reflectance", _range("reflectance", self.reflectance, 0.0, 1.0))
        object.__setattr__(self, "clearcoat", _range("clearcoat", self.clearcoat, 0.0, 1.0))
        object.__setattr__(
            self,
            "clearcoat_roughness",
            _range("clearcoat_roughness", self.clearcoat_roughness, 0.025, 0.98),
        )
        if self.specular_gain < 0.0:
            raise ValueError("specular_gain must be nonnegative")


@dataclass(frozen=True)
class LightConfig:
    sun_enabled: bool = True
    sun_azimuth_deg: float = 135.0
    sun_elevation_deg: float = 40.0
    sun_intensity: float = 1.0
    ambient_fill: float = 0.65
    environment_reflection: float = 1.0
    environment_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    environment_sh: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment_color", _vec3("environment_color", self.environment_color))
        if self.environment_sh is not None:
            coefficients = tuple(_vec3("environment_sh coefficient", row) for row in self.environment_sh)
            if len(coefficients) not in (1, 9):
                raise ValueError("environment_sh must contain one or nine RGB coefficients")
            object.__setattr__(self, "environment_sh", coefficients)
        for name in ("sun_intensity", "ambient_fill", "environment_reflection"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class ViewConfig:
    camera_position: tuple[float, float, float] = (6.0, -6.0, 3.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.8)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vertical_fov_deg: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_position", _vec3("camera_position", self.camera_position))
        object.__setattr__(self, "target", _vec3("target", self.target))
        object.__setattr__(self, "up", _vec3("up", self.up))
        object.__setattr__(self, "vertical_fov_deg", _range("vertical_fov_deg", self.vertical_fov_deg, 1.0, 179.0))


@dataclass(frozen=True)
class ConversionConfig:
    voxel_size: float = 0.04
    normal_neighbors: int = 24
    mapping_neighbors: int = 8
    surfel_overlap: float = 1.2
    roughness: float = 0.38
    metallic: float = 0.02

    def __post_init__(self) -> None:
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if self.normal_neighbors < 3:
            raise ValueError("normal_neighbors must be at least 3")
        if self.mapping_neighbors < 1:
            raise ValueError("mapping_neighbors must be positive")
        if self.surfel_overlap <= 0.0:
            raise ValueError("surfel_overlap must be positive")
        _range("roughness", self.roughness, 0.025, 0.98)
        _range("metallic", self.metallic, 0.0, 1.0)


@dataclass(frozen=True)
class IntegrationMode:
    lighting_color: bool = True
    pbr_properties: bool = True
    vehicle_shadow: bool = True
    vehicle_projection: bool = True

    def __post_init__(self) -> None:
        for name in ("lighting_color", "pbr_properties", "vehicle_shadow", "vehicle_projection"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class VehiclePlacement:
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", _vec3("translation", self.translation))
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")


@dataclass(frozen=True)
class ShadowConfig:
    strength: float = 0.65
    length_scale: float = 1.2
    width_scale: float = 1.35
    edge_softness: float = 0.12
    ground_band: float = 0.18

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength", _range("strength", self.strength, 0.0, 1.0))
        object.__setattr__(self, "edge_softness", _range("edge_softness", self.edge_softness, 0.0, 1.0))
        for name in ("length_scale", "width_scale", "ground_band"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1280
    height: int = 720
    background_rgb: tuple[float, float, float] = (0.05, 0.05, 0.05)
    device: str = "cuda:0"
    exposure: float = 1.0
    relight_strength: float = 1.0
    material: MaterialConfig = field(default_factory=MaterialConfig)
    light: LightConfig = field(default_factory=LightConfig)
    view: ViewConfig = field(default_factory=ViewConfig)
    integration: IntegrationMode = field(default_factory=IntegrationMode)
    placement: VehiclePlacement = field(default_factory=VehiclePlacement)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        object.__setattr__(self, "background_rgb", _vec3("background_rgb", self.background_rgb))
        if self.exposure <= 0.0:
            raise ValueError("exposure must be positive")
        if self.relight_strength < 0.0:
            raise ValueError("relight_strength must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RenderConfig":
        data = dict(value)
        data["material"] = MaterialConfig(**data.get("material", {}))
        data["light"] = LightConfig(**data.get("light", {}))
        data["view"] = ViewConfig(**data.get("view", {}))
        data["integration"] = IntegrationMode(**data.get("integration", {}))
        data["placement"] = VehiclePlacement(**data.get("placement", {}))
        data["shadow"] = ShadowConfig(**data.get("shadow", {}))
        return cls(**data)