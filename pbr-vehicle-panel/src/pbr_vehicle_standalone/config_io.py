from __future__ import annotations

import copy
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import LightingState, MaterialState, TransformState, VehicleState


SCHEMA_VERSION = 1


def sanitize(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip()).strip("._-")
    return result or "unnamed"


def read_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a JSON object: {source}")
    return payload


def next_config_path(asset_folder: Path, scene_name: str, vehicle_name: str) -> Path:
    folder = asset_folder / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"config_{sanitize(scene_name)}_{sanitize(vehicle_name)}_"
    sequence = 1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_")
    for path in folder.glob(f"{prefix}*.json"):
        match = pattern.match(path.name)
        if match:
            sequence = max(sequence, int(match.group(1)) + 1)
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    return folder / f"{prefix}{sequence:04d}_{timestamp}.json"


def save_viewer_config(path: str | Path, scene: dict[str, Any], scene_lighting: LightingState,
                       vehicle: VehicleState, canonical_asset: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_kind": "pbr-vehicle-standalone-viewer",
        "saved_at": datetime.now().astimezone().isoformat(),
        "scene": scene,
        "scene_lighting": scene_lighting.to_dict(),
        "pbr_asset": canonical_asset,
        "vehicle": vehicle.to_dict(),
    }
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def state_from_config(payload: dict[str, Any], fallback: VehicleState) -> tuple[VehicleState, LightingState | None]:
    if payload.get("config_kind") == "pbr-vehicle-standalone-viewer" or isinstance(payload.get("vehicle"), dict):
        raw = payload["vehicle"]
        transform_raw = raw.get("transform")
        if transform_raw is None and "position" in raw:
            transform_raw = {"position": raw.get("position"), "rotation_deg": raw.get("rotation_deg"), "scale": raw.get("scale", 1.0)}
        material_raw = raw.get("material") or {
            "use_asset_material": raw.get("use_ply_material", True),
            **{key: raw[key] for key in ("roughness", "reflectance", "metallic", "exposure", "ambient_fill", "relight_strength", "saturation") if key in raw},
        }
        lighting_raw = copy.deepcopy(raw.get("lighting") or raw.get("vehicle_lighting", {}))
        has_sun_color = any(
            key in lighting_raw
            for key in ("sun_rgb", "sun_color_rgb", "sun_red", "sun_green", "sun_blue")
        )
        embedded_light = (payload.get("pbr_asset") or {}).get("light", {})
        if not has_sun_color and isinstance(embedded_light, dict) and "color_rgb" in embedded_light:
            lighting_raw["sun_color_rgb"] = embedded_light["color_rgb"]
        projection = copy.deepcopy(raw.get("projection") or fallback.projection)
        state = VehicleState(
            vehicle_id=fallback.vehicle_id,
            asset_folder=str(raw.get("asset_folder", fallback.asset_folder)),
            visible=bool(raw.get("visible", True)),
            display_mode=str(raw.get("display_mode", raw.get("mode", "Relight Original"))),
            transform=TransformState(**{key: value for key, value in (transform_raw or {}).items() if key in TransformState.__dataclass_fields__}),
            material=MaterialState(**{key: value for key, value in material_raw.items() if key in MaterialState.__dataclass_fields__}),
            use_scene_lighting=bool(raw.get("use_scene_lighting", lighting_raw.get("use_scene_lighting", True))),
            lighting=LightingState.from_dict(lighting_raw),
            projection_visible=bool(raw.get("projection_visible", True)),
            projection_opacity=float(raw.get("projection_opacity", 1.0)),
            projection=projection,
        )
        scene_lighting = payload.get("scene_lighting")
        return state, LightingState.from_dict(scene_lighting) if isinstance(scene_lighting, dict) else None
    if payload.get("asset_contract"):
        material = payload.get("material", {})
        light = payload.get("light", {})
        sun_color = light.get("sun_color_rgb", light.get("color_rgb", [1.0, 1.0, 1.0]))
        environment_color = light.get("environment_color_rgb", [1.0, 1.0, 1.0])
        state = copy.deepcopy(fallback)
        state.use_scene_lighting = False
        state.material = MaterialState(
            roughness=float(material.get("roughness", 0.4)),
            reflectance=float(material.get("reflectance", 0.04)),
            metallic=float(material.get("metallic", 0.0)),
            exposure=float(material.get("exposure", 1.0)),
            relight_strength=float(material.get("relight_strength", 1.0)),
        )
        state.lighting = LightingState.from_dict({
            "environment_rgb": list(map(float, environment_color[:3])),
            "environment_temperature_k": light.get("environment_temperature_k"),
            "sun_enabled": bool(light.get("sun_enabled", True)),
            "sun_intensity": float(light.get("intensity", 1.0)),
            "sun_rgb": list(map(float, sun_color[:3])),
            "sun_azimuth_deg": float(light.get("sun_azimuth_degrees", 45.0)),
            "sun_elevation_deg": float(light.get("sun_elevation_degrees", 35.0)),
        })
        state.projection = copy.deepcopy(payload.get("projection") or fallback.projection)
        return state, None
    raise ValueError("Unsupported config kind")
