import inspect
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pbr_vehicle_standalone.viewer as viewer_module
from pbr_vehicle_standalone.viewer import (
    StandaloneViewer,
    VehicleController,
    _scene_display_indices,
    _ui_to_value,
    _value_to_ui,
)
from pbr_vehicle_standalone.environment_map import EnvironmentMap
from pbr_vehicle_standalone.types import TransformState


class _Handle:
    def __init__(self, value):
        self.value = value
        self.visible = value
        self.disabled = False


def test_scene_display_indices_are_bounded_aligned_and_reproducible():
    first = _scene_display_indices(total_splats=100, max_splats=20, seed=7)
    second = _scene_display_indices(total_splats=100, max_splats=20, seed=7)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 20
    assert np.all(first[:-1] < first[1:])
    assert first.min() >= 0
    assert first.max() < 100


def test_scene_display_indices_keep_full_scene_when_limit_is_disabled():
    values = np.arange(12)

    np.testing.assert_array_equal(values[_scene_display_indices(12, 0, 7)], values)
    np.testing.assert_array_equal(values[_scene_display_indices(12, 12, 7)], values)


def test_projection_uses_gaussian_nodes_instead_of_glb(monkeypatch):
    calls = []

    class _Scene:
        def add_gaussian_splats(self, name, **kwargs):
            calls.append((name, kwargs))
            return SimpleNamespace(buffer=None, wxyz=None, position=None, visible=True)

        def add_glb(self, *_args, **_kwargs):
            raise AssertionError("Projection must share the Gaussian render channel")

    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[2:14, 2:14, :3] = 64
    rgba[2:14, 2:14, 3] = 128
    result = {
        "ground_z": -0.02,
        "extension": {"rgba": rgba, "center_xy": np.array([0.0, 0.0]), "size_xy": np.array([4.0, 2.0])},
        "contact": {"rgba": rgba, "center_xy": np.array([0.0, 0.0]), "size_xy": np.array([3.0, 1.5])},
    }
    monkeypatch.setattr(viewer_module, "build_projection_masks", lambda *_args: result)

    controller = object.__new__(VehicleController)
    controller.server = SimpleNamespace(scene=_Scene(), atomic=nullcontext)
    controller.vehicle_id = "vehicle_001"
    controller.asset = SimpleNamespace(proxy=object())
    controller.handles = {
        "projection_visible": _Handle(True),
        "projection_opacity": _Handle(1.0),
    }
    controller.contact_handle = None
    controller.extension_handle = None
    controller.projection = lambda: {}

    controller._update_projection(
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([6.0, 0.0, 0.0], dtype=np.float32),
        scale=1.0,
        lighting=SimpleNamespace(sun_enabled=True, visibility=1.0),
        local_sun=np.array([0.5, 0.5, 0.7], dtype=np.float32),
    )

    assert [name for name, _ in calls] == [
        "/vehicles/vehicle_001/projection/extension",
        "/vehicles/vehicle_001/projection/contact",
    ]
    assert all(kwargs["centers"].shape[1] == 3 for _, kwargs in calls)
    assert calls[0][1]["centers"][:, 2].min() == pytest.approx(0.01)
    assert calls[1][1]["centers"][:, 2].min() == pytest.approx(0.012)


def test_scene_only_cubemap_capture_hides_and_restores_vehicle_nodes():
    class _Client:
        client_id = 7

        def __init__(self):
            self.calls = []

        def flush(self):
            pass

        def get_render(self, height, width, **kwargs):
            self.calls.append((height, width, kwargs))
            return np.full((height, width, 4), 255, dtype=np.uint8)

    client = _Client()
    vehicle_handle = _Handle(True)
    projection_handle = _Handle(False)
    scene_handle = _Handle(False)
    app = object.__new__(StandaloneViewer)
    app.scene_handle = scene_handle
    app.controllers = [SimpleNamespace(
        splat_handle=vehicle_handle,
        contact_handle=projection_handle,
        extension_handle=None,
    )]
    app.server = SimpleNamespace(get_clients=lambda: {client.client_id: client})
    app._environment_client = client
    app._environment_capture_lock = threading.RLock()

    environment = app.capture_environment_map(np.zeros(3, dtype=np.float32), 32)

    assert environment.faces.shape == (6, 32, 32, 3)
    assert len(client.calls) == 6
    assert vehicle_handle.visible is True
    assert projection_handle.visible is False
    assert scene_handle.visible is False


def test_preview_independent_position_does_not_move_vehicle_ibl_sampling_center():
    environment = EnvironmentMap.from_faces(np.ones((6, 8, 8, 3), dtype=np.float32))
    capture_centers = []
    mesh_positions = []

    class _Scene:
        def add_mesh_trimesh(self, _name, _mesh, **kwargs):
            mesh_positions.append(np.asarray(kwargs["position"]))
            return _Handle(True)

    controller = object.__new__(VehicleController)
    controller.asset = SimpleNamespace(proxy=SimpleNamespace(centers=np.asarray([[-1, -1, -1], [1, 1, 1]], np.float32)))
    controller.vehicle_id = "vehicle_001"
    controller.server = SimpleNamespace(scene=_Scene())
    controller.app = SimpleNamespace(
        _environment_generation=0,
        capture_environment_map=lambda center, _resolution: (
            capture_centers.append(np.asarray(center).copy()) or environment
        ),
    )
    controller.handles = {
        "environment_map_enabled": _Handle(True),
        "environment_map_resolution": _Handle("32"),
        "environment_preview_visible": _Handle(True),
        "environment_preview_follow_vehicle": _Handle(False),
        "environment_preview_x": _Handle(10.0),
        "environment_preview_y": _Handle(11.0),
        "environment_preview_z": _Handle(12.0),
        "environment_preview_offset_x": _Handle(0.0),
        "environment_preview_offset_y": _Handle(-4.0),
        "environment_preview_offset_z": _Handle(2.0),
    }
    controller._environment_map_cache = None
    controller._environment_map_cache_key = None
    controller._environment_preview_cache = None
    controller._environment_preview_cache_key = None
    controller._environment_map_error = None
    controller.environment_preview_handle = None
    controller._environment_preview_mesh_key = None

    result = controller._environment_map_for(
        np.eye(3, dtype=np.float32), TransformState(position=[2.0, 3.0, 4.0])
    )

    assert result is environment
    assert np.allclose(capture_centers[0], [2.0, 3.0, 4.0])
    assert np.allclose(capture_centers[1], [10.0, 11.0, 12.0])
    assert np.allclose(mesh_positions[0], [10.0, 11.0, 12.0])


def test_following_preview_uses_vehicle_capture_and_display_offset():
    environment = EnvironmentMap.from_faces(np.ones((6, 8, 8, 3), dtype=np.float32))
    capture_centers = []
    mesh_positions = []

    class _Scene:
        def add_mesh_trimesh(self, _name, _mesh, **kwargs):
            mesh_positions.append(np.asarray(kwargs["position"]))
            return _Handle(True)

    controller = object.__new__(VehicleController)
    controller.asset = SimpleNamespace(proxy=SimpleNamespace(centers=np.asarray([[-1, -1, -1], [1, 1, 1]], np.float32)))
    controller.vehicle_id = "vehicle_001"
    controller.server = SimpleNamespace(scene=_Scene())
    controller.app = SimpleNamespace(
        _environment_generation=0,
        capture_environment_map=lambda center, _resolution: (
            capture_centers.append(np.asarray(center).copy()) or environment
        ),
    )
    controller.handles = {
        "environment_map_enabled": _Handle(False),
        "environment_map_resolution": _Handle("32"),
        "environment_preview_visible": _Handle(True),
        "environment_preview_follow_vehicle": _Handle(True),
        "environment_preview_x": _Handle(10.0),
        "environment_preview_y": _Handle(11.0),
        "environment_preview_z": _Handle(12.0),
        "environment_preview_offset_x": _Handle(1.0),
        "environment_preview_offset_y": _Handle(-4.0),
        "environment_preview_offset_z": _Handle(2.0),
    }
    controller._environment_map_cache = None
    controller._environment_map_cache_key = None
    controller._environment_preview_cache = None
    controller._environment_preview_cache_key = None
    controller._environment_map_error = None
    controller.environment_preview_handle = None
    controller._environment_preview_mesh_key = None

    result = controller._environment_map_for(
        np.eye(3, dtype=np.float32), TransformState(position=[2.0, 3.0, 4.0])
    )

    assert result is None
    assert len(capture_centers) == 1
    assert np.allclose(capture_centers[0], [2.0, 3.0, 4.0])
    assert np.allclose(mesh_positions[0], [3.0, -1.0, 6.0])


def test_advanced_panel_has_no_nested_priority_folders():
    source = inspect.getsource(VehicleController._build_gui)
    assert source.count('add_folder("高级"') == 1
    assert "次要参数" not in source
    assert '"Auto 识别车辆参数"' in source
    assert "绝对 UI 区间 -0.3 到 0.3" in source
    assert '"使用代理重光照"' in source
    for category in ("Transform", "Material", "R3GW Lighting", "Projection"):
        assert f'add_folder("{category}"' in source


def test_auto_fit_button_is_not_bound_as_a_normal_slider_update():
    source = inspect.getsource(VehicleController._build_gui)
    assert '"auto_fit_progress", "auto_fit_status"' in source
    assert '"auto_fit_stage", "auto_fit_result"' in source
    assert 'self.handles["auto_fit"].on_click(self._auto_fit_event)' in source


def test_both_estimators_expose_progress_status_stage_and_result():
    scene_source = inspect.getsource(StandaloneViewer._build_gui)
    vehicle_source = inspect.getsource(VehicleController._build_gui)
    for key in (
        "sun_inference_progress", "sun_inference_status",
        "sun_inference_stage", "sun_inference_result",
    ):
        assert key in scene_source
    for key in (
        "auto_fit_progress", "auto_fit_status", "auto_fit_stage", "auto_fit_result",
    ):
        assert key in vehicle_source


def test_config_panel_exposes_reset_vehicle():
    source = inspect.getsource(VehicleController._build_gui)
    assert 'add_button("Reset vehicle")' in source
    assert "reset_button.on_click" in source


def test_reset_restores_the_vehicle_initial_state():
    controller = object.__new__(VehicleController)
    initial = SimpleNamespace(value=1)
    controller.initial_state = initial
    applied = []
    notifications = []
    controller.apply_state = lambda state: applied.append(state)
    controller.app = SimpleNamespace(_notify=lambda *args: notifications.append(args))

    controller._reset_event(None)

    assert len(applied) == 1
    assert applied[0].value == 1
    assert applied[0] is not initial
    assert notifications[-1][1] == "Vehicle reset"


def test_auto_fit_applies_only_its_three_output_controls(monkeypatch, tmp_path):
    controller = object.__new__(VehicleController)
    controller.asset = SimpleNamespace(root=tmp_path / "asset")
    controller.asset.root.mkdir()
    controller._removed = False
    controller._suspend_updates = False
    controller.server = SimpleNamespace(atomic=nullcontext)
    controller.handles = {
        "auto_fit": _Handle(True),
        "auto_fit_progress": _Handle(0.0),
        "auto_fit_status": _Handle("等待启动"),
        "auto_fit_stage": _Handle("尚未执行"),
        "auto_fit_result": _Handle("尚无结果"),
        "use_scene_lighting": _Handle(True),
        "light_sun_enabled": _Handle(False),
        "light_sun_intensity": _Handle(-0.2),
        "ambient_fill": _Handle(-0.1),
        "light_temperature": _Handle(0.2),
        "light_sun_r": _Handle(0.7),
        "light_sun_g": _Handle(0.8),
        "light_sun_b": _Handle(0.9),
        "light_intensity": _Handle(2.5),
        "light_sun_azimuth": _Handle(81.0),
        "light_sun_elevation": _Handle(27.0),
        "saturation": _Handle(1.4),
        "roughness": _Handle(0.3),
    }
    notifications = []
    controller.app = SimpleNamespace(
        scene_ply_for_auto_fit=lambda: Path("scene.ply"),
        handles={"auto_fit_config": _Handle("")},
        args=SimpleNamespace(auto_fit_timeout=600.0),
        _notify=lambda *args: notifications.append(args),
    )
    controller.auto_fit_template = lambda: {"vehicle": {}}
    controller._sync_mirrored_controls = lambda: None
    controller._sync_lighting_enabled = lambda: None
    updates = []
    controller.update = lambda: updates.append(True)
    monkeypatch.setattr(
        viewer_module,
        "run_vehicle_auto_fit",
        lambda **_: {
            "sun_intensity": 2.0,
            "ambient_fill": 0.6,
            "environment_temperature_k": 5000.0,
            "improvement_percent": 5.0,
            "output_dir": "/tmp/test-auto",
        },
    )

    controller._run_auto_fit(None)

    assert controller.handles["use_scene_lighting"].value is False
    assert controller.handles["light_sun_enabled"].value is True
    assert _ui_to_value(controller.handles["light_sun_intensity"].value, "sun_intensity") == pytest.approx(2.0)
    assert _ui_to_value(controller.handles["ambient_fill"].value, "brightness") == pytest.approx(0.6)
    assert _ui_to_value(controller.handles["light_temperature"].value, "temperature") == pytest.approx(5000.0)
    assert [controller.handles[key].value for key in ("light_sun_r", "light_sun_g", "light_sun_b")] == [0.7, 0.8, 0.9]
    assert controller.handles["light_intensity"].value == 2.5
    assert controller.handles["light_sun_azimuth"].value == 81.0
    assert controller.handles["light_sun_elevation"].value == 27.0
    assert controller.handles["saturation"].value == 1.4
    assert controller.handles["roughness"].value == 0.3
    assert updates == [True]
    assert controller.handles["auto_fit"].disabled is False
    assert controller.handles["auto_fit_progress"].value == pytest.approx(100.0)
    assert controller.handles["auto_fit_status"].value == "成功"
    assert controller.handles["auto_fit_stage"].value == "网格最优参数已回填面板"
    assert "太阳光强度：2.0000" in controller.handles["auto_fit_result"].value
    assert notifications[-1][1] == "Auto 识别完成"


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
    ("kind", "ui_low", "ui_neutral", "ui_high", "value_low", "value_neutral", "value_high"),
    [
        ("sun_intensity", -1.0, 0.0, 1.0, 0.0, 1.0, 8.0),
        ("brightness", -1.0, 0.0, 1.0, 0.0, 0.35, 1.0),
        ("temperature", -0.5, 0.0, 0.5, 2000.0, 6500.0, 12000.0),
        ("saturation", 0.0, 1.0, 2.0, 0.0, 1.0, 2.0),
    ],
)
def test_exposed_control_mapping(kind, ui_low, ui_neutral, ui_high, value_low, value_neutral, value_high):
    assert _ui_to_value(ui_low, kind) == pytest.approx(value_low)
    assert _ui_to_value(ui_neutral, kind) == pytest.approx(value_neutral)
    assert _ui_to_value(ui_high, kind) == pytest.approx(value_high)
    for value in np.linspace(ui_low, ui_high, 5):
        assert _value_to_ui(_ui_to_value(value, kind), kind) == pytest.approx(value)
