from pathlib import Path

import numpy as np

from pbr_vehicle_standalone.math3d import sun_direction
from pbr_vehicle_standalone.math3d import color_temperature_to_rgb
from pbr_vehicle_standalone.shading import shade_proxy, shade_vehicle
from pbr_vehicle_standalone.types import GaussianLayer, LightingState, MaterialState, VehicleAsset


def _proxy():
    normal = sun_direction(45.0, 35.0)[None, :]
    return GaussianLayer(
        centers=np.zeros((1, 3), np.float32),
        covariances=np.eye(3, dtype=np.float32)[None],
        colors=np.full((1, 3), 0.5, np.float32),
        opacities=np.ones((1, 1), np.float32),
        normals=normal,
        albedo=np.full((1, 3), 0.5, np.float32),
        roughness=np.full((1, 1), 0.5, np.float32),
        metallic=np.zeros((1, 1), np.float32),
    )


def test_environment_color_does_not_tint_direct_sun():
    material = MaterialState(use_asset_material=False, ambient_fill=0.0)
    red_environment = LightingState(environment_rgb=[3.0, 0.0, 0.0], sun_rgb=[1.0, 1.0, 1.0])
    blue_environment = LightingState(environment_rgb=[0.0, 0.0, 3.0], sun_rgb=[1.0, 1.0, 1.0])
    assert np.allclose(
        shade_proxy(_proxy(), material, red_environment),
        shade_proxy(_proxy(), material, blue_environment),
    )


def test_sun_color_controls_direct_light_independently():
    material = MaterialState(use_asset_material=False, ambient_fill=0.0)
    red_sun = LightingState(environment_rgb=[1.0, 1.0, 1.0], sun_rgb=[1.0, 0.0, 0.0])
    blue_sun = LightingState(environment_rgb=[1.0, 1.0, 1.0], sun_rgb=[0.0, 0.0, 1.0])
    assert not np.allclose(
        shade_proxy(_proxy(), material, red_sun),
        shade_proxy(_proxy(), material, blue_sun),
    )


def test_neutral_temperature_and_albedo_saturation():
    assert np.allclose(color_temperature_to_rgb(6500.0), [1.0, 1.0, 1.0])
    proxy = _proxy()
    proxy.albedo[:] = [0.8, 0.3, 0.1]
    material_gray = MaterialState(use_asset_material=False, ambient_fill=1.0, saturation=0.0)
    material_color = MaterialState(use_asset_material=False, ambient_fill=1.0, saturation=2.0)
    lighting = LightingState(sun_enabled=False)
    assert not np.allclose(
        shade_proxy(proxy, material_gray, lighting),
        shade_proxy(proxy, material_color, lighting),
    )


def test_saturation_affects_colored_original_with_grayscale_proxy():
    proxy = _proxy()
    original = GaussianLayer(
        centers=np.zeros((1, 3), np.float32),
        covariances=np.eye(3, dtype=np.float32)[None],
        colors=np.array([[0.75, 0.30, 0.10]], np.float32),
        opacities=np.ones((1, 1), np.float32),
    )
    asset = VehicleAsset(
        root=Path("."), asset_id="test", original=original, proxy=proxy,
        original_to_proxy=np.array([0]), mapping_indices=np.array([[0]]),
        mapping_weights=np.array([[1.0]], np.float32), canonical_config_path=Path("config.json"),
        canonical_config={}, projection={},
    )
    lighting = LightingState(sun_enabled=False)
    _, gray = shade_vehicle(asset, MaterialState(saturation=0.0), lighting)
    _, vivid = shade_vehicle(asset, MaterialState(saturation=2.0), lighting)
    assert not np.allclose(gray, vivid)
    assert np.allclose(gray[0, 0], gray[0, 1])
    assert np.allclose(gray[0, 1], gray[0, 2])
