from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pbr_vehicle_sdk import (
    ConversionConfig,
    IntegrationMode,
    RenderConfig,
    analyze_scene,
    bake_scene,
    compose_scene_properties,
    convert_asset,
    estimate_sun_light_rgb,
    estimate_sun_position,
    estimate_vehicle_pbr,
    load_scene_ply,
)
from pbr_vehicle_sdk.ply import read_ply, require_standard_gaussian, write_ply

from test_sdk_e2e import ordinary_gaussian


def scene_fixture(tmp_path):
    vehicle_path = write_ply(tmp_path / "vehicle.ply", ordinary_gaussian(64))
    asset = convert_asset(vehicle_path, tmp_path / "asset", ConversionConfig(voxel_size=0.18))
    scene_values = ordinary_gaussian(100)
    grid = np.linspace(-1.5, 1.5, 10)
    grid_x, grid_y = np.meshgrid(grid, grid)
    scene_values["x"] = grid_x.ravel().astype(np.float32)
    scene_values["y"] = grid_y.ravel().astype(np.float32)
    scene_values["z"] = np.zeros(100, dtype=np.float32)
    scene_values.pop("f_rest_8")
    scene_values["nx"] = np.zeros(100, dtype=np.float32)
    scene_values["semantic_id"] = np.full(100, 7, dtype=np.uint8)
    scene = load_scene_ply(write_ply(tmp_path / "scene.ply", scene_values))
    return scene, asset


def mode_from_mask(mask: int) -> IntegrationMode:
    return IntegrationMode(
        lighting_color=bool(mask & 1),
        pbr_properties=bool(mask & 2),
        vehicle_shadow=bool(mask & 4),
        vehicle_projection=bool(mask & 8),
    )


@pytest.mark.parametrize("mask", range(16))
def test_all_scene_bake_mode_combinations(tmp_path, mask):
    scene, asset = scene_fixture(tmp_path)
    config = RenderConfig(integration=mode_from_mask(mask))
    source_dc = np.stack([scene.properties[f"f_dc_{index}"] for index in range(3)], axis=1)
    combined, _, shadowed_count = compose_scene_properties(scene, asset, config)
    output = tmp_path / f"mode_{mask:02d}.ply"
    result = bake_scene(scene, asset, output, config)
    baked = read_ply(output)
    require_standard_gaussian(baked)

    expected_count = scene.gaussian_count + (asset.raw_count if config.integration.vehicle_projection else 0)
    assert len(combined["x"]) == expected_count
    assert len(baked["x"]) == expected_count
    assert result.vehicle_gaussian_count == (asset.raw_count if config.integration.vehicle_projection else 0)
    assert np.isfinite(np.stack(list(baked.values()), axis=1)).all()
    assert "nx" in baked and "semantic_id" in baked
    np.testing.assert_array_equal(baked["semantic_id"][: scene.gaussian_count], 7)

    baked_scene_dc = np.stack(
        [baked[f"f_dc_{index}"][: scene.gaussian_count] for index in range(3)], axis=1
    )
    if config.integration.vehicle_shadow:
        assert shadowed_count > 0
        assert not np.allclose(baked_scene_dc, source_dc)
    else:
        assert shadowed_count == 0
        np.testing.assert_allclose(baked_scene_dc, source_dc)

    manifest = json.loads(output.with_suffix(".ply.bake.json").read_text())
    assert manifest["integration"] == config.to_dict()["integration"]


def test_scene_analysis_interfaces_are_explicit_placeholders(tmp_path):
    scene, asset = scene_fixture(tmp_path)
    calls = (
        lambda: analyze_scene(scene, asset),
        lambda: estimate_sun_position(scene),
        lambda: estimate_sun_light_rgb(scene),
        lambda: estimate_vehicle_pbr(scene, asset),
    )
    for call in calls:
        with pytest.raises(NotImplementedError, match="placeholder"):
            call()


def test_scene_cli_contract(tmp_path):
    scene, asset = scene_fixture(tmp_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)

    inspect_result = subprocess.run(
        [sys.executable, "-m", "pbr_vehicle_sdk.cli", "inspect-scene", str(scene.source_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(inspect_result.stdout)["gaussian_count"] == scene.gaussian_count

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"render": RenderConfig().to_dict()}))
    output_path = tmp_path / "scene_baked.ply"
    bake_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pbr_vehicle_sdk.cli",
            "bake-scene",
            str(scene.source_path),
            str(asset.root),
            str(output_path),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(bake_result.stdout)["vehicle_gaussian_count"] == asset.raw_count
    require_standard_gaussian(read_ply(output_path))

    analyze_result = subprocess.run(
        [sys.executable, "-m", "pbr_vehicle_sdk.cli", "analyze-scene", str(scene.source_path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert analyze_result.returncode == 3
    assert json.loads(analyze_result.stderr)["status"] == "not_implemented"