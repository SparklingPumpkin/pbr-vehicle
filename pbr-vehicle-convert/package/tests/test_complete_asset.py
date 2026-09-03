from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import pytest

from pbr_vehicle_sdk import (
    build_complete_asset,
    load_complete_asset,
    projection_mask_descriptor,
    validate_projection_mask_config,
)
from pbr_vehicle_sdk.ply import read_ply, write_ply


def gaussian_vehicle(count: int = 48) -> OrderedDict[str, np.ndarray]:
    x = np.linspace(-1.8, 1.8, count, dtype=np.float32)
    y = 0.45 * np.sin(x * 2.0)
    z = np.linspace(0.1, 1.5, count, dtype=np.float32)
    values: OrderedDict[str, np.ndarray] = OrderedDict(x=x, y=y, z=z)
    for index in range(3):
        values[f"f_dc_{index}"] = np.zeros(count, dtype=np.float32)
    values["opacity"] = np.full(count, 3.0, dtype=np.float32)
    values["scale_0"] = np.full(count, -3.2, dtype=np.float32)
    values["scale_1"] = np.full(count, -3.2, dtype=np.float32)
    values["scale_2"] = np.full(count, -4.5, dtype=np.float32)
    values["rot_0"] = np.ones(count, dtype=np.float32)
    for index in range(1, 4):
        values[f"rot_{index}"] = np.zeros(count, dtype=np.float32)
    return values


def default_config(package_root):
    return json.loads((package_root / "examples/configs/config_default.json").read_text())


def test_config_folder_contract_and_runtime_mask_projection(tmp_path):
    package_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    source = write_ply(tmp_path / "vehicle.ply", gaussian_vehicle())
    asset = build_complete_asset(source, tmp_path / "asset", "demo", default_config(package_root))
    assert {path.name for path in asset.root.iterdir()} == {
        "pbr_demo.ply", "configs"
    }
    assert {path.name for path in (asset.root / "configs").iterdir()} == {"config_demo.json"}

    loaded = load_complete_asset(asset.root)
    config = json.loads(loaded.config_json.read_text())
    assert config["asset_contract"] == "pbr-vehicle-single-ply-v1"
    assert config["files"]["config"] == "configs/config_demo.json"
    assert config["projection"]["runtime"] == "receiver_space_mask_v1"
    assert config["projection"]["shape_model"] == "procedural-mask-contact-silhouette-v4"
    assert config["projection"]["contact"]["shape"] == "rounded_rectangle"
    assert config["projection"]["contact"]["offset_xy_m"] == [0.0, 0.0]

    loaded_pbr = read_ply(loaded.pbr_ply)
    assert all(f"normal_{i}" in loaded_pbr for i in range(3))
    assert not any(name.startswith("r3gw_") for name in loaded_pbr)
    descriptor = projection_mask_descriptor(loaded_pbr, config["light"], config["projection"])
    bottom_surface = loaded_pbr["z"] - np.exp(loaded_pbr["scale_2"])
    expected_anchor = np.percentile(bottom_surface, 1.0)
    assert np.isclose(descriptor["ground_z"], expected_anchor)
    assert descriptor["uses_gaussian_splats"] is False
    assert descriptor["contact_footprint"]["size_xy_m"][0] < 3.6
    assert descriptor["contact_footprint"]["opacity"] == 0.11
    assert descriptor["contact_footprint"]["brightness"] == 0.18

    changed_light = dict(config["light"])
    changed_light["sun_azimuth_degrees"] = 260.0
    changed = projection_mask_descriptor(loaded_pbr, changed_light, config["projection"])
    assert not np.array_equal(descriptor["cast_outline_xyz"], changed["cast_outline_xyz"])


def test_contact_and_extension_controls_are_separate(tmp_path):
    package_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = default_config(package_root)
    config["projection"]["contact"].update({
        "length_scale": 0.80,
        "width_scale": 0.70,
        "offset_xy_m": [0.15, -0.12],
        "corner_radius_m": 0.30,
        "edge_softness_m": 0.05,
        "opacity": 0.11,
        "brightness": 0.40,
    })
    config["projection"]["extension"].update({
        "opacity": 0.31,
        "brightness": 0.52,
        "brightness_distance_to_white": 0.77,
        "edge_softness_m": 0.12,
        "edge_softness_distance_growth_m": 0.33,
    })
    source = write_ply(tmp_path / "vehicle.ply", gaussian_vehicle())
    asset = build_complete_asset(source, tmp_path / "asset", "controls", config)
    saved = json.loads(asset.config_json.read_text())
    descriptor = projection_mask_descriptor(read_ply(asset.pbr_ply), saved["light"], saved["projection"])
    assert descriptor["contact_footprint"]["opacity"] == 0.11
    assert descriptor["contact_footprint"]["brightness"] == 0.40
    assert descriptor["extension_mask"]["opacity"] == 0.31
    assert descriptor["extension_mask"]["brightness"] == 0.52


def test_projection_mask_contract_rejects_out_of_range_brightness():
    package_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = default_config(package_root)
    config["projection"]["contact"]["brightness"] = 1.01
    with pytest.raises(ValueError, match="contact.brightness"):
        validate_projection_mask_config(config["projection"])
