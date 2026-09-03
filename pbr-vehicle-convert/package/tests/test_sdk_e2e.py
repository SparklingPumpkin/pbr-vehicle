from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import pytest

from pbr_vehicle_sdk import ConversionConfig, IntegrationMode, LightConfig, MaterialConfig, RenderConfig, ViewConfig, bake, convert_asset, relight
from pbr_vehicle_sdk.ply import read_ply, require_standard_gaussian, write_ply
from pbr_vehicle_sdk.rendering import _camera_matrices


def ordinary_gaussian(count: int = 64) -> OrderedDict[str, np.ndarray]:
    side = int(np.ceil(np.sqrt(count)))
    grid_x, grid_y = np.meshgrid(np.linspace(-1.0, 1.0, side), np.linspace(-0.5, 0.5, side))
    points = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(side * side)], axis=1)[:count]
    values: OrderedDict[str, np.ndarray] = OrderedDict()
    for channel, name in enumerate(("x", "y", "z")):
        values[name] = points[:, channel].astype(np.float32)
    for channel in range(3):
        values[f"f_dc_{channel}"] = np.full(count, 0.15 + 0.1 * channel, dtype=np.float32)
    for channel in range(9):
        values[f"f_rest_{channel}"] = np.full(count, 0.02, dtype=np.float32)
    values["opacity"] = np.full(count, 4.0, dtype=np.float32)
    values["scale_0"] = np.full(count, -2.5, dtype=np.float32)
    values["scale_1"] = np.full(count, -2.5, dtype=np.float32)
    values["scale_2"] = np.full(count, -4.0, dtype=np.float32)
    values["rot_0"] = np.ones(count, dtype=np.float32)
    for channel in range(1, 4):
        values[f"rot_{channel}"] = np.zeros(count, dtype=np.float32)
    return values


def test_convert_relight_bake_round_trip(tmp_path):
    source = write_ply(tmp_path / "vehicle.ply", ordinary_gaussian())
    asset = convert_asset(source, tmp_path / "vehicle_pbr", ConversionConfig(voxel_size=0.18))
    assert asset.raw_count == 64
    assert 0 < asset.proxy_count < asset.raw_count
    assert asset.mapping_indices.shape == (64, min(8, asset.proxy_count))
    np.testing.assert_allclose(asset.mapping_weights.sum(axis=1), 1.0, atol=1e-5)

    result = relight(asset, RenderConfig())
    assert result.raw_rgb.shape == (64, 3)
    assert np.isfinite(result.raw_rgb).all()
    assert np.all((result.raw_rgb >= 0.0) & (result.raw_rgb <= 1.0))

    output = tmp_path / "vehicle_baked.ply"
    bake(asset, str(output), RenderConfig())
    baked = read_ply(output)
    require_standard_gaussian(baked)
    for name, values in baked.items():
        if name.startswith("f_rest_"):
            assert np.count_nonzero(values) == 0
    manifest = json.loads(output.with_suffix(".ply.bake.json").read_text())
    assert manifest["view_dependent"] is True
    assert str(tmp_path) not in (asset.root / "asset.json").read_text()


def test_public_config_validation():
    config = RenderConfig()
    assert RenderConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        MaterialConfig(roughness=1.2)
    with pytest.raises(ValueError):
        ViewConfig(vertical_fov_deg=180.0)


def test_all_integration_mode_combinations_round_trip():
    for mask in range(16):
        mode = IntegrationMode(
            lighting_color=bool(mask & 1),
            pbr_properties=bool(mask & 2),
            vehicle_shadow=bool(mask & 4),
            vehicle_projection=bool(mask & 8),
        )
        config = RenderConfig(integration=mode)
        assert RenderConfig.from_dict(config.to_dict()) == config


def test_pbr_switch_bypasses_proxy_shading(tmp_path):
    source = write_ply(tmp_path / "vehicle.ply", ordinary_gaussian())
    asset = convert_asset(source, tmp_path / "vehicle_pbr", ConversionConfig(voxel_size=0.18))
    disabled = RenderConfig(integration=IntegrationMode(pbr_properties=False))
    np.testing.assert_allclose(relight(asset, disabled).raw_rgb, relight(asset, disabled).proxy_rgb)


def test_lighting_color_is_independent_from_pbr(tmp_path):
    source = write_ply(tmp_path / "vehicle.ply", ordinary_gaussian())
    asset = convert_asset(source, tmp_path / "vehicle_pbr", ConversionConfig(voxel_size=0.18))
    colored_light = LightConfig(environment_color=(1.8, 0.7, 0.3), sun_enabled=False)
    outputs = {}
    for lighting_color in (False, True):
        for pbr_properties in (False, True):
            mode = IntegrationMode(lighting_color=lighting_color, pbr_properties=pbr_properties)
            outputs[(lighting_color, pbr_properties)] = relight(
                asset, RenderConfig(light=colored_light, integration=mode)
            ).raw_rgb
    assert not np.allclose(outputs[(False, False)], outputs[(True, False)])
    assert not np.allclose(outputs[(False, True)], outputs[(True, True)])
    assert not np.allclose(outputs[(False, False)], outputs[(False, True)])


def test_rejects_incomplete_gaussian(tmp_path):
    incomplete = OrderedDict((name, np.ones(4, dtype=np.float32)) for name in ("x", "y", "z"))
    source = write_ply(tmp_path / "incomplete.ply", incomplete)
    with pytest.raises(ValueError, match="missing"):
        convert_asset(source, tmp_path / "asset")


def test_gsplat_camera_looks_along_positive_z():
    config = RenderConfig(view=ViewConfig(camera_position=(4.0, -3.0, 2.0), target=(0.0, 0.0, 0.0)))
    view, _ = _camera_matrices(config)
    target_camera = view @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    camera_camera = view @ np.array([4.0, -3.0, 2.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(camera_camera[:3], 0.0, atol=1e-6)
    assert target_camera[2] > 0.0
