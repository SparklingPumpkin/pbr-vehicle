import inspect

import pytest

from pbr_vehicle_standalone.viewer import VehicleController, _ui_to_value, _value_to_ui


class _Handle:
    def __init__(self, value):
        self.value = value


def test_advanced_panel_has_no_nested_priority_folders():
    source = inspect.getsource(VehicleController._build_gui)
    assert source.count('add_folder("高级"') == 1
    assert "次要参数" not in source
    for category in ("Transform", "Material", "R3GW Lighting", "Projection"):
        assert f'add_folder("{category}"' in source


def test_quick_and_category_controls_are_mirrored():
    controller = object.__new__(VehicleController)
    controller._suspend_updates = False
    controller._removed = False
    controller.handles = {
        "light_sun_intensity": _Handle(1.0),
        "light_sun_intensity_advanced": _Handle(2.5),
        "use_scene_lighting": _Handle(True),
        "light_sun_enabled": _Handle(False),
    }
    updates = []
    controller._control_changed = updates.append

    controller._mirrored_control_changed(
        "light_sun_intensity_advanced",
        "light_sun_intensity",
        "light_sun_intensity",
    )

    assert controller.handles["light_sun_intensity"].value == 2.5
    assert controller.handles["use_scene_lighting"].value is False
    assert controller.handles["light_sun_enabled"].value is True
    assert updates == ["light_sun_intensity"]


@pytest.mark.parametrize(
    ("kind", "low", "neutral", "high"),
    [
        ("sun_intensity", 0.0, 1.0, 8.0),
        ("brightness", 0.0, 0.35, 1.0),
        ("temperature", 2000.0, 6500.0, 12000.0),
        ("saturation", 0.0, 1.0, 2.0),
    ],
)
def test_normalized_exposed_control_mapping(kind, low, neutral, high):
    assert _ui_to_value(-1.0, kind) == pytest.approx(low)
    assert _ui_to_value(0.0, kind) == pytest.approx(neutral)
    assert _ui_to_value(1.0, kind) == pytest.approx(high)
    for value in (-1.0, -0.4, 0.0, 0.4, 1.0):
        assert _value_to_ui(_ui_to_value(value, kind), kind) == pytest.approx(value)
