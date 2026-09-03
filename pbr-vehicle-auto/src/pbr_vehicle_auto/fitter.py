"""Fast constrained PBR DC adaptation for an ordinary Gaussian scene."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
CCT_MIN_KELVIN = 3500.0
CCT_MAX_KELVIN = 7200.0
ENVIRONMENT_ENERGY = 0.612338
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_PATH = PACKAGE_ROOT / "templates/default_vehicle_pbr_viser_template.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8, None)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def rgb_dc(data: np.ndarray) -> np.ndarray:
    return np.clip(0.5 + SH_C0 * np.stack([data[f"f_dc_{i}"] for i in range(3)], axis=1), 0.0, 1.0).astype(np.float32)


def opacity(data: np.ndarray) -> np.ndarray:
    return sigmoid(np.asarray(data["opacity"], dtype=np.float64)) if "opacity" in data.dtype.names else np.ones(len(data), dtype=np.float64)


def luma(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ LUMA


def cdf(values: np.ndarray, weights: np.ndarray, bins: int) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    index = np.minimum((values * (bins - 1)).astype(np.int64), bins - 1)
    hist = np.bincount(index, weights=weights, minlength=bins)
    return np.cumsum(hist) / max(float(hist.sum()), 1e-12)


def cdf_l1(values: np.ndarray, weights: np.ndarray, target: np.ndarray, bins: int) -> float:
    return float(np.mean(np.abs(cdf(values, weights, bins) - target)))


def quantiles(values: np.ndarray, weights: np.ndarray) -> list[float]:
    order = np.argsort(values)
    total = float(weights.sum())
    current = np.cumsum(weights[order])
    return [float(values[order[min(np.searchsorted(current, q * total), len(order) - 1)]]) for q in (0.02, 0.10, 0.50, 0.90, 0.98)]


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def cct_tint(cct_kelvin: float) -> np.ndarray:
    """Approximate CCT in sRGB, then normalize the linear RGB tint by luma."""
    temperature = float(np.clip(cct_kelvin, CCT_MIN_KELVIN, CCT_MAX_KELVIN)) / 100.0
    if temperature <= 66.0:
        red = 1.0
        green = np.clip((99.4708025861 * math.log(temperature) - 161.1195681661) / 255.0, 0.0, 1.0)
        blue = 0.0 if temperature <= 19.0 else np.clip((138.5177312231 * math.log(temperature - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0)
    else:
        red = np.clip((329.698727446 * (temperature - 60.0) ** -0.1332047592) / 255.0, 0.0, 1.0)
        green = np.clip((288.1221695283 * (temperature - 60.0) ** -0.0755148492) / 255.0, 0.0, 1.0)
        blue = 1.0
    result = srgb_to_linear(np.asarray([red, green, blue], dtype=np.float64))
    return result / max(float(luma(result)), 1e-12)


def neutral_chroma(scene_rgb: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    brightness = luma(scene_rgb)
    low, high = np.quantile(brightness, [0.20, 0.80])
    saturation = np.ptp(scene_rgb, axis=1)
    selected = (brightness >= low) & (brightness <= high) & (saturation <= 0.12)
    if int(selected.sum()) < 1024:
        selected = (brightness >= low) & (brightness <= high)
    chroma = scene_rgb[selected] / np.clip(brightness[selected, None], 1e-6, None)
    result = np.sum(chroma * weights[selected, None], axis=0) / max(float(weights[selected].sum()), 1e-12)
    result /= max(float(luma(result)), 1e-12)
    return result, {"samples": int(selected.sum()), "weight": float(weights[selected].sum()), "chroma": result.tolist()}


def choose_cct(target: np.ndarray) -> float:
    grid = np.linspace(CCT_MIN_KELVIN, CCT_MAX_KELVIN, 75)
    return float(grid[np.argmin([np.mean(np.abs(cct_tint(value) - target)) for value in grid])])


def sun_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    return normalize(np.asarray([[
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ]]))[0]


def aces(rgb: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    result = rgb * (a * rgb + b) / np.clip(rgb * (c * rgb + d) + e, 1e-8, None)
    return np.clip(result, 0.0, 1.0)


def fresnel(cosine: np.ndarray, f0: np.ndarray) -> np.ndarray:
    return f0 + (1.0 - f0) * np.power(1.0 - np.clip(cosine, 0.0, 1.0), 5.0)


def ggx(normals: np.ndarray, views: np.ndarray, light_dirs: np.ndarray, roughness: float, f0: np.ndarray) -> np.ndarray:
    half = normalize(views + light_dirs)
    ndotv = np.clip(np.sum(normals * views, axis=1, keepdims=True), 1e-4, 1.0)
    ndotl = np.clip(np.sum(normals * light_dirs, axis=1, keepdims=True), 0.0, 1.0)
    ndoth = np.clip(np.sum(normals * half, axis=1, keepdims=True), 0.0, 1.0)
    vdoth = np.clip(np.sum(views * half, axis=1, keepdims=True), 0.0, 1.0)
    alpha = np.clip(float(roughness), 0.025, 1.0) ** 2
    alpha2 = alpha ** 2
    denominator = (ndoth * ndoth * (alpha2 - 1.0) + 1.0) ** 2
    distribution = alpha2 / np.clip(math.pi * denominator, 1e-6, None)
    k = (float(roughness) + 1.0) ** 2 / 8.0
    geometry = ndotv / np.clip(ndotv * (1.0 - k) + k, 1e-6, None)
    geometry *= ndotl / np.clip(ndotl * (1.0 - k) + k, 1e-6, None)
    return distribution * geometry * fresnel(vdoth, f0) * ndotl / np.clip(4.0 * ndotv * np.maximum(ndotl, 1e-4), 1e-5, None)


def proxy_lighting(albedo: np.ndarray, normals: np.ndarray, view_dirs: np.ndarray, material: Mapping[str, float], light: Mapping[str, Any]) -> np.ndarray:
    normals, view_dirs = normalize(normals), normalize(view_dirs)
    count = len(albedo)
    roughness, metallic = float(material["roughness"]), float(material["metallic"])
    f0 = float(material["reflectance"]) * (1.0 - metallic) + albedo * metallic
    env_rgb = np.asarray(light["environment_color_rgb"], dtype=np.float32)[None, :]
    ambient = np.broadcast_to(env_rgb * float(material["ambient_fill"]), albedo.shape)
    diffuse = albedo * ambient * (1.0 - metallic)
    ndotv = np.clip(np.sum(normals * view_dirs, axis=1, keepdims=True), 1e-4, 1.0)
    reflected = normalize(2.0 * ndotv * normals - view_dirs)
    reflected_light = np.broadcast_to(env_rgb, albedo.shape)
    specular = reflected_light * fresnel(ndotv, f0) * (1.0 - 0.72 * roughness) ** 2 * float(material["environment_reflection"])
    coat = float(material["clearcoat"])
    if coat > 0.0:
        coat_f0 = np.full_like(albedo, 0.04)
        coat_env = reflected_light * fresnel(ndotv, coat_f0) * (1.0 - 0.65 * float(material["clearcoat_roughness"])) ** 2
        specular = specular * (1.0 - 0.25 * coat) + coat * coat_env * float(material["environment_reflection"])
    direction = sun_direction(float(light["sun_azimuth_degrees"]), float(light["sun_elevation_degrees"]))
    directions = np.broadcast_to(direction[None, :], normals.shape)
    radiance = np.broadcast_to(np.asarray(light["sun_color_rgb"], dtype=np.float32)[None, :], albedo.shape) * float(light["intensity"])
    ndotl = np.clip(np.sum(normals * directions, axis=1, keepdims=True), 0.0, 1.0)
    diffuse += albedo * radiance * ndotl * (1.0 - metallic)
    specular += radiance * ggx(normals, view_dirs, directions, roughness, f0)
    if coat > 0.0:
        coat_f0 = np.full_like(albedo, 0.04)
        specular += coat * radiance * ggx(normals, view_dirs, directions, float(material["clearcoat_roughness"]), coat_f0)
    return aces(np.maximum((diffuse + specular * float(material["specular_gain"])) * float(material["exposure"]), 0.0)).astype(np.float32)


def required(data: np.ndarray, names: list[str], label: str) -> None:
    missing = [name for name in names if name not in data.dtype.names]
    if missing:
        raise ValueError(f"{label} lacks fields: {missing}")


def load_asset(asset_dir: Path, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PlyData]:
    files = config["files"]
    if config.get("asset_contract") == "pbr-vehicle-single-ply-v1":
        original_ply = PlyData.read(str(asset_dir / files["pbr"]))
        original = original_ply["vertex"].data
        required(original, ["x", "y", "z", "normal_0", "normal_1", "normal_2"], "pbr Gaussian")
        material = config.get("material", {})
        base_color = np.asarray(material.get("albedo_rgb", [0.82, 0.82, 0.82]), dtype=np.float32)
        if base_color.shape != (3,):
            raise ValueError("material.albedo_rgb must contain three values")
        count = len(original)
        albedo = np.broadcast_to(base_color[None], (count, 3)).copy()
        normals = normalize(np.stack([original[f"normal_{i}"] for i in range(3)], axis=1).astype(np.float32))
        centers = np.stack([original[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)
        indices = np.arange(count, dtype=np.int64)[:, None]
        return albedo, normals, centers, indices, np.ones((count, 1), np.float32), original_ply
    proxy = PlyData.read(str(asset_dir / files["proxy"]))["vertex"].data
    original_ply = PlyData.read(str(asset_dir / files["original"]))
    original = original_ply["vertex"].data
    required(proxy, ["x", "y", "z"], "proxy")
    albedo_prefix = "r3gw_albedo" if all(f"r3gw_albedo_{i}" in proxy.dtype.names for i in range(3)) else "pbr_albedo"
    normal_prefix = "r3gw_normal" if all(f"r3gw_normal_{i}" in proxy.dtype.names for i in range(3)) else "pbr_normal"
    required(proxy, [f"{albedo_prefix}_{i}" for i in range(3)] + [f"{normal_prefix}_{i}" for i in range(3)], "proxy")
    albedo = np.stack([proxy[f"{albedo_prefix}_{i}"] for i in range(3)], axis=1).astype(np.float32)
    normals = normalize(np.stack([proxy[f"{normal_prefix}_{i}"] for i in range(3)], axis=1))
    centers = np.stack([proxy[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)
    mapping = np.load(str(asset_dir / files["mapping"]))
    if "raw_to_proxy_knn_idx" in mapping and "raw_to_proxy_knn_weight" in mapping:
        indices = np.asarray(mapping["raw_to_proxy_knn_idx"], dtype=np.int64)
        weights = np.asarray(mapping["raw_to_proxy_knn_weight"], dtype=np.float32)
    elif "clean_to_proxy_idx" in mapping:
        indices = np.asarray(mapping["clean_to_proxy_idx"], dtype=np.int64)[:, None]
        weights = np.ones_like(indices, dtype=np.float32)
    else:
        raise ValueError("mapping needs raw_to_proxy_knn_idx/raw_to_proxy_knn_weight or clean_to_proxy_idx")
    if len(original) != len(indices) or indices.max() >= len(proxy):
        raise ValueError("original PLY and mapping are incompatible")
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    return albedo, normals, centers, indices, weights, original_ply


def material_defaults(base: Mapping[str, Any]) -> dict[str, float]:
    values = {"roughness": 0.32, "reflectance": 0.04, "metallic": 0.02, "clearcoat": 0.35, "exposure": 1.0, "relight_strength": 1.0, "ambient_fill": 0.4, "specular_gain": 1.0, "environment_reflection": 0.8, "clearcoat_roughness": 0.12, "saturation": 1.0}
    values.update({key: float(value) for key, value in base.get("material", {}).items() if key in values})
    return values


def adjust_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    luminance = np.sum(values * LUMA.astype(np.float32), axis=-1, keepdims=True)
    return np.clip(luminance + float(saturation) * (values - luminance), 0.0, 1.0).astype(np.float32)


def split_template(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    asset, vehicle = payload.get("pbr_asset"), payload.get("vehicle")
    if isinstance(asset, Mapping) and isinstance(vehicle, Mapping):
        if not isinstance(asset.get("files"), Mapping):
            raise ValueError("template pbr_asset must contain files")
        return copy.deepcopy(dict(asset)), copy.deepcopy(dict(vehicle))
    if not isinstance(payload.get("files"), Mapping):
        raise ValueError("template must be a PBR asset config or a Viser state")
    return copy.deepcopy(dict(payload)), None


def template_material(base: Mapping[str, Any], vehicle: Mapping[str, Any] | None) -> dict[str, float]:
    material = material_defaults(base)
    if vehicle is not None:
        for key in material:
            if key in vehicle and vehicle[key] is not None:
                material[key] = float(vehicle[key])
    return material


def fixed_template_light(base: Mapping[str, Any], vehicle: Mapping[str, Any] | None) -> tuple[np.ndarray, float, float]:
    base_light = base.get("light", {})
    vehicle_light = vehicle.get("vehicle_lighting", {}) if vehicle is not None else {}
    if not isinstance(vehicle_light, Mapping):
        vehicle_light = {}
    sun_rgb = np.asarray(vehicle_light.get("sun_color_rgb", base_light.get("sun_color_rgb", base_light.get("color_rgb", [1.0, 1.0, 1.0]))), dtype=np.float64)
    if sun_rgb.shape != (3,) or not np.isfinite(sun_rgb).all():
        raise ValueError("template fixed sun color must have three finite values")
    return sun_rgb, float(vehicle_light.get("sun_azimuth", base_light.get("sun_azimuth_degrees", 90.0))), float(vehicle_light.get("sun_elevation", base_light.get("sun_elevation_degrees", 45.0)))


def emit_config(template: Mapping[str, Any], base: Mapping[str, Any], vehicle: Mapping[str, Any] | None, material: Mapping[str, float], light: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    asset = copy.deepcopy(dict(base))
    asset_material = dict(asset.get("material", {}))
    asset_material["ambient_fill"] = float(material["ambient_fill"])
    asset["material"] = asset_material
    asset_light = dict(asset.get("light", {}))
    asset_light.update({"intensity": float(light["intensity"]), "environment_cct_kelvin": float(light["environment_cct_kelvin"]), "environment_energy": float(light["environment_energy"]), "environment_color_rgb": [float(value) for value in light["environment_color_rgb"]]})
    asset["light"] = asset_light
    asset.pop("appearance_adjustment", None)
    asset["auto_fit"] = dict(metadata)
    if vehicle is None:
        return asset
    result = copy.deepcopy(dict(template))
    result["pbr_asset"] = asset
    active_vehicle = dict(result["vehicle"])
    active_vehicle["ambient_fill"] = float(material["ambient_fill"])
    active_light = dict(active_vehicle.get("vehicle_lighting", {}))
    active_light.update({"sun_intensity": float(light["intensity"]), "sun_color_rgb": [float(value) for value in light["sun_color_rgb"]], "environment_temperature": float(light["environment_cct_kelvin"])})
    active_vehicle["vehicle_lighting"] = active_light
    result["vehicle"] = active_vehicle
    result["auto_fit"] = dict(metadata)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-ply", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, default=None)
    parser.add_argument("--base-config", type=Path, default=None, help="Deprecated alias for --template-config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-config", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=480.0)
    parser.add_argument("--bins", type=int, default=512)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.template_config is not None and args.base_config is not None:
        raise ValueError("use only one of --template-config and --base-config")
    template_argument = args.template_config or args.base_config
    template_path = (template_argument or DEFAULT_TEMPLATE_PATH).resolve()
    template_reference = str(template_argument) if template_argument is not None else "packaged:default_vehicle_pbr_viser_template.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    base, vehicle = split_template(template)
    scene_ply = PlyData.read(str(args.scene_ply))
    scene = scene_ply["vertex"].data
    albedo, normals, centers, indices, weights, original_ply = load_asset(args.asset_dir, base)
    original = original_ply["vertex"].data
    scene_rgb, scene_weight = rgb_dc(scene), opacity(scene)
    original_rgb, original_weight = rgb_dc(original), opacity(original)
    scene_luma, original_luma = luma(scene_rgb), luma(original_rgb)
    target_cdf = cdf(scene_luma, scene_weight, args.bins)
    target_chroma, neutral = neutral_chroma(scene_rgb, scene_weight)
    cct_initial = choose_cct(target_chroma)
    material = template_material(base, vehicle)
    sun_rgb, sun_azimuth, sun_elevation = fixed_template_light(base, vehicle)
    rear_x = float(np.quantile(centers[:, 0], 0.005))
    target = np.quantile(centers, 0.5, axis=0)
    camera = np.asarray([rear_x - 4.0, target[1], target[2] + 0.5], dtype=np.float32)
    views = normalize(camera[None, :] - centers)
    phases: dict[str, float] = {"load_seconds": time.perf_counter() - started}
    records: list[dict[str, Any]] = []

    def evaluate(stage: str, sun_intensity: float, fill: float, cct_kelvin: float) -> dict[str, Any]:
        candidate_material = dict(material, ambient_fill=float(fill))
        light = {"sun_azimuth_degrees": sun_azimuth, "sun_elevation_degrees": sun_elevation, "intensity": float(sun_intensity), "sun_color_rgb": sun_rgb.tolist(), "color_rgb": sun_rgb.tolist(), "environment_cct_kelvin": float(cct_kelvin), "environment_energy": ENVIRONMENT_ENERGY, "environment_color_rgb": (ENVIRONMENT_ENERGY * cct_tint(cct_kelvin)).tolist()}
        saturation = float(candidate_material["saturation"])
        proxy_albedo = adjust_saturation(albedo, saturation)
        proxy = proxy_lighting(proxy_albedo, normals, views, candidate_material, light)
        ratio = proxy / np.clip(proxy_albedo, 0.03, 1.0)
        mapped = np.sum(ratio[indices] * weights[:, :, None], axis=1)
        colors = np.clip(adjust_saturation(original_rgb, saturation) * (1.0 + candidate_material["relight_strength"] * (mapped - 1.0)) * candidate_material["exposure"], 0.0, 1.0)
        value = cdf_l1(luma(colors), original_weight, target_cdf, args.bins)
        return {"stage": stage, "sun_intensity": float(sun_intensity), "ambient_fill": float(fill), "environment_cct_kelvin": float(cct_kelvin), "luminance_cdf_l1": value, "colors": colors, "material": candidate_material, "light": light}

    phase_start = time.perf_counter()
    brightness = []
    for sun in (0.50, 0.75, 1.00, 1.50, 2.00, 3.00):
        for fill in (0.15, 0.30, 0.45, 0.60, 0.85):
            if time.perf_counter() - started > args.max_seconds: break
            item = evaluate("brightness", sun, fill, cct_initial); brightness.append(item); records.append(item)
    if not brightness: raise RuntimeError("time limit expired before brightness search")
    best = min(brightness, key=lambda item: item["luminance_cdf_l1"])
    phases["brightness_seconds"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    colour = []
    for cct_kelvin in np.unique(np.clip(cct_initial + np.arange(-4, 5) * 350.0, CCT_MIN_KELVIN, CCT_MAX_KELVIN)):
        item = evaluate("cct", best["sun_intensity"], best["ambient_fill"], float(cct_kelvin)); item["colour_cost"] = float(0.85 * np.mean(np.abs(cct_tint(cct_kelvin) - target_chroma)) + 0.15 * item["luminance_cdf_l1"]); colour.append(item); records.append(item)
    if colour: best = min(colour, key=lambda item: item["colour_cost"])
    phases["colour_seconds"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    final = []
    for sun_scale in (0.85, 1.0, 1.15):
        for delta in (-0.10, 0.0, 0.10):
            item = evaluate("finish", best["sun_intensity"] * sun_scale, float(np.clip(best["ambient_fill"] + delta, 0.05, 1.20)), best["environment_cct_kelvin"]); final.append(item); records.append(item)
    best = min(final, key=lambda item: item["luminance_cdf_l1"])
    phases["finish_seconds"] = time.perf_counter() - phase_start
    phases["total_seconds"] = time.perf_counter() - started

    raw_distance = cdf_l1(original_luma, original_weight, target_cdf, args.bins)
    candidate_distance = cdf_l1(luma(best["colors"]), original_weight, target_cdf, args.bins)
    accepted = candidate_distance < raw_distance - 1e-9
    final_colors = best["colors"] if accepted else original_rgb
    baked_distance = candidate_distance if accepted else raw_distance
    baked = np.array(original, copy=True)
    for channel in range(3): baked[f"f_dc_{channel}"] = ((final_colors[:, channel] - 0.5) / SH_C0).astype(baked[f"f_dc_{channel}"].dtype)
    for name in baked.dtype.names or ():
        if name.startswith("f_rest_"): baked[name] = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baked_path = args.output_dir / f"{base.get('asset_id', 'vehicle')}_rear_view_cct_pbr_dc_baked.ply"
    PlyData([PlyElement.describe(baked, "vertex")], text=False).write(str(baked_path))
    metadata = {"schema_version": 2, "status": "candidate_only" if accepted else "rejected_no_improvement", "method": "template_cct_environment_brightness_luminance_fit", "contract": {"scene_kind": "ordinary_gaussian_dc", "sun_direction_source": "manual_template_input", "sun_rgb": "fixed_template_value", "environment_cct_kelvin_range": [CCT_MIN_KELVIN, CCT_MAX_KELVIN], "environment_energy": "fixed", "free_material_parameters": [], "manual_only_material_parameters": ["saturation"], "free_brightness_parameters": ["sun_intensity", "ambient_fill"], "free_colour_parameters": ["environment_cct_kelvin"]}, "inputs": {"scene_ply": str(args.scene_ply), "scene_sha256": digest(args.scene_ply), "template_config": template_reference, "template_config_sha256": digest(template_path), "template_kind": "viser_state" if vehicle is not None else "pbr_asset"}, "metric": {"name": "bilateral_opacity_weighted_luminance_cdf_l1_512", "raw": raw_distance, "baked": baked_distance, "improvement_percent": 100.0 * (raw_distance - baked_distance) / raw_distance if raw_distance > 1e-12 else 0.0}, "candidate_metric": {"baked": candidate_distance}, "colour": {"neutral_scene": neutral, "selected_cct_kelvin": best["environment_cct_kelvin"], "selected_tint": cct_tint(best["environment_cct_kelvin"]).tolist(), "environment_energy_fixed": ENVIRONMENT_ENERGY, "vehicle_saturation": "manual_only_not_fitted"}, "search": {"candidates_evaluated": len(records), "phase_times": phases}}
    payload = emit_config(template, base, vehicle, best["material"], best["light"], metadata) if accepted else copy.deepcopy(template)
    if not accepted:
        payload["auto_fit"] = metadata
    args.final_config.parent.mkdir(parents=True, exist_ok=True)
    args.final_config.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = [{key: value for key, value in record.items() if key not in {"colors", "material", "light"}} for record in records]
    (args.output_dir / "candidates.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final_config": str(args.final_config), "baked_ply": str(baked_path), **metadata["metric"], "elapsed_seconds": phases["total_seconds"]}))


if __name__ == "__main__":
    main()
