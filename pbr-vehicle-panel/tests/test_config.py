import json
import os
from pathlib import Path

import pytest

from pbr_vehicle_standalone.config_io import read_config, save_viewer_config, state_from_config
from pbr_vehicle_standalone.types import LightingState, VehicleState


def test_viewer_config_round_trip(tmp_path):
    state = VehicleState(vehicle_id="vehicle_001", asset_folder="assets/test", projection={"contact": {}, "extension": {}, "anchor": {}})
    path = save_viewer_config(tmp_path / "config.json", {"name": "scene", "path": "scenes/test.ply"}, LightingState(), state, {})
    restored, scene_light = state_from_config(read_config(path), state)
    assert restored.to_dict() == state.to_dict()
    assert scene_light == LightingState()
    payload = read_config(path)
    assert payload["vehicle"]["lighting"]["sun_color_rgb"] == [1.0, 1.0, 1.0]
    assert "sun_rgb" not in payload["vehicle"]["lighting"]


def test_legacy_lighting_defaults_to_independent_white_sun():
    lighting = LightingState.from_dict({"red": 0.2, "green": 0.4, "blue": 0.6})
    assert lighting.environment_rgb == [0.2, 0.4, 0.6]
    assert lighting.sun_rgb == [1.0, 1.0, 1.0]


ASSET_CONFIG_ENV = "PBR_VEHICLE_TEST_CONFIG"
ASSET_CONFIG = Path(os.environ[ASSET_CONFIG_ENV]) if os.environ.get(ASSET_CONFIG_ENV) else None


def test_asset_config_separates_environment_and_sun_color():
    if ASSET_CONFIG is None or not ASSET_CONFIG.is_file():
        pytest.skip(f"set {ASSET_CONFIG_ENV} to run the real-config test")
    fallback = VehicleState(vehicle_id="vehicle_001", asset_folder=str(ASSET_CONFIG.parents[1]))
    state, _ = state_from_config(json.loads(ASSET_CONFIG.read_text()), fallback)
    assert state.lighting.environment_rgb == [0.55, 0.62, 0.72]
    assert state.lighting.sun_rgb == [1.0, 0.98, 0.92]
