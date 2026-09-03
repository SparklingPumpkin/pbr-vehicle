"""Build and reload a PBR vehicle asset with a dedicated configuration folder."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .ply import read_ply
from .projection import projection_mask_descriptor, validate_projection_mask_config
from .single_pbr import build_single_pbr_ply

ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CompletePBRAsset:
    root: Path
    asset_id: str
    pbr_ply: Path
    config_json: Path


def load_complete_asset(root: str | Path) -> CompletePBRAsset:
    directory = Path(root)
    config_directory = directory / "configs"
    if not config_directory.is_dir():
        raise ValueError("A complete asset must contain a configs/ directory")
    candidates = sorted(config_directory.glob("config_*.json"))
    if not candidates:
        raise ValueError("A complete asset must contain configs/config_<asset-id>.json")
    primary = [path for path in candidates
               if json.loads(path.read_text(encoding="utf-8")).get("asset_contract") == "pbr-vehicle-single-ply-v1"
               and path.stem == f"config_{json.loads(path.read_text(encoding='utf-8')).get('asset_id')}" ]
    if len(primary) != 1:
        raise ValueError("A complete asset must contain exactly one canonical config_*.json")
    config_path = primary[0]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    files = config["files"]
    result = CompletePBRAsset(directory, config["asset_id"], directory / files["pbr"], config_path)
    expected = {result.pbr_ply.name}
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(f"Asset payload contract mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    expected_config = Path(files["config"])
    if expected_config.parts[:1] != ("configs",) or expected_config.name != config_path.name:
        raise ValueError("Canonical config must be recorded as configs/config_<asset-id>.json")
    if any(not path.name.startswith("config_") or path.suffix != ".json" for path in config_directory.iterdir() if path.is_file()):
        raise ValueError("configs/ may contain only config_*.json files")
    if _sha256(result.pbr_ply) != config["sha256"]["pbr"]:
        raise ValueError("pbr SHA-256 mismatch")
    return result


def build_complete_asset(input_ply: str | Path, output_dir: str | Path, asset_id: str, config: Mapping[str, Any]) -> CompletePBRAsset:
    if not ASSET_ID_RE.fullmatch(asset_id):
        raise ValueError("asset_id must use letters, digits, underscore, or hyphen")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    config_directory = root / "configs"
    config_directory.mkdir()
    pbr = root / f"pbr_{asset_id}.ply"
    config_path = config_directory / f"config_{asset_id}.json"
    build_single_pbr_ply(input_ply, pbr)
    validate_projection_mask_config(config["projection"])
    projection_descriptor = projection_mask_descriptor(read_ply(pbr), config["light"], config["projection"])
    material = dict(config["material"])
    material.setdefault("albedo_rgb", [0.82, 0.82, 0.82])
    payload = {
        "schema_version": 10,
        "asset_contract": "pbr-vehicle-single-ply-v1",
        "asset_id": asset_id,
        "files": {"pbr": pbr.name, "config": str(config_path.relative_to(root))},
        "material": material,
        "light": config["light"],
        "surface": {"geometry": "visible_original_gaussians", "normal_fields": ["normal_0", "normal_1", "normal_2"], "normal_source": "minimum_covariance_axis", "normal_orientation": "center_outward", "normal_version": 1},
        "relighting": {"control_layer": "logical", "geometry_source": "pbr_gaussian", "material_source": "config.material", "transfer": "same_point_multiplicative_ratio", "mapping_required": False},
        "projection": config["projection"],
        "projection_runtime": {
            "dynamic": True,
            "runtime": "receiver_space_mask_v1",
            "recompute_on": ["sun_azimuth_degrees", "sun_elevation_degrees", "vehicle pose", "projection parameters"],
            "algorithm": "render an analytic vehicle-local rounded-rectangle contact mask and a filled, smoothed parallel-light vehicle outline from the current PBR Gaussian and configuration",
        },
        "projection_stats": {
            "model": projection_descriptor["model"],
            "runtime": projection_descriptor["runtime"],
            "geometry": projection_descriptor["geometry"],
            "ground_z": projection_descriptor["ground_z"],
            "vehicle_z_anchor": projection_descriptor["vehicle_z_anchor"],
        },
        "sha256": {"pbr": _sha256(pbr)},
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return load_complete_asset(root)
