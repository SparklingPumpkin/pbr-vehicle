import json
from importlib.resources import files

import numpy as np

from pbr_vehicle_auto.defaults import normalize_default
from pbr_vehicle_auto.fitter import cct_tint, proxy_lighting


def test_cct_path_is_luminance_normalized_and_bounded():
    assert cct_tint(3500.0).shape == (3,)
    assert np.isclose(np.dot(cct_tint(6650.0), [0.2126, 0.7152, 0.0722]), 1.0)


def test_saved_state_normalization_removes_automatic_metadata():
    state = {
        "pbr_asset": {"material": {"roughness": 0.32}, "light": {"environment_color_rgb": [0.6, 0.6, 0.6]}, "auto_fit": {"old": True}},
        "vehicle": {"roughness": 0.2, "reflectance": 0.04, "metallic": 0.0, "exposure": 1.15, "ambient_fill": 0.3, "relight_strength": 1.0, "vehicle_lighting": {"red": 1.0, "green": 0.98, "blue": 0.92, "sun_azimuth": 56.0, "sun_elevation": 45.0, "sun_intensity": 1.0, "sun_enabled": True}},
    }
    result = normalize_default(state)
    assert "auto_fit" not in result
    assert result["vehicle"]["exposure"] == 1.15
    assert result["vehicle"]["vehicle_lighting"]["sun_color_rgb"] == [1.0, 1.0, 1.0]
    assert result["vehicle"]["vehicle_lighting"]["sun_azimuth"] == 56.0


def test_direct_sun_is_independent_of_environment_colour():
    albedo = np.asarray([[0.6, 0.5, 0.4]], dtype=np.float32)
    normals = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    views = normals.copy()
    material = {"roughness": 0.32, "reflectance": 0.04, "metallic": 0.0, "clearcoat": 0.0, "clearcoat_roughness": 0.12, "ambient_fill": 0.0, "environment_reflection": 0.0, "specular_gain": 1.0, "exposure": 1.0}
    light = {"sun_azimuth_degrees": 0.0, "sun_elevation_degrees": 90.0, "intensity": 1.0, "sun_color_rgb": [1.0, 0.98, 0.92], "environment_color_rgb": [1.0, 0.0, 0.0]}
    red_environment = proxy_lighting(albedo, normals, views, material, light)
    light["environment_color_rgb"] = [0.0, 0.0, 1.0]
    blue_environment = proxy_lighting(albedo, normals, views, material, light)
    assert np.allclose(red_environment, blue_environment)


def test_default_template_keeps_extension_distance_to_white_disabled():
    template = json.loads(
        files("pbr_vehicle_auto")
        .joinpath("templates/default_vehicle_pbr_viser_template.json")
        .read_text(encoding="utf-8")
    )
    assert template["pbr_asset"]["projection"]["extension"]["brightness_distance_to_white"] == 0.0
    assert template["vehicle"]["projection"]["extension"]["brightness_distance_to_white"] == 0.0
