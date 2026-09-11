from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from pbr_vehicle_standalone.asset_io import resolve_scene_context
from pbr_vehicle_standalone.viewer import StandaloneViewer


class Handle:
    def __init__(self, value):
        self.value = value


def test_pth_scene_context_resolves_relative_dataset_root(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "data/argoverse/processed/training/033"
    dataset.mkdir(parents=True)
    run = tmp_path / "output/argoverse_033_mv3"
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint_final.pth"
    checkpoint.touch()
    (run / "config.yaml").write_text(
        "data:\n  data_root: data/argoverse/processed/training\n  scene_idx: 33\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert resolve_scene_context(checkpoint) == {
        "name": "argoverse_033_mv3",
        "data_root": str(dataset),
        "config_path": str(run / "config.yaml"),
    }


def test_estimated_sun_angles_update_shared_and_all_vehicle_controls():
    def controller(removed=False, shared=False):
        item = SimpleNamespace(
            _removed=removed,
            _suspend_updates=False,
            handles={
                "light_sun_enabled": Handle(False),
                "light_sun_azimuth": Handle(0.0),
                "light_sun_elevation": Handle(0.0),
                "light_sun_azimuth_advanced": Handle(0.0),
                "light_sun_elevation_advanced": Handle(0.0),
            },
            _sync_lighting_enabled=lambda: None,
            use_scene_lighting=lambda: shared,
        )
        item._sync_mirrored_controls = lambda: [
            setattr(item.handles[advanced], "value", item.handles[canonical].value)
            for canonical, advanced in {
                "light_sun_azimuth": "light_sun_azimuth_advanced",
                "light_sun_elevation": "light_sun_elevation_advanced",
            }.items()
        ]
        item.schedule_update = lambda source: updates.append((item, source))
        return item

    updates = []
    shared = controller(shared=True)
    independent = controller(shared=False)
    removed = controller(removed=True)
    app = object.__new__(StandaloneViewer)
    app.server = SimpleNamespace(atomic=nullcontext)
    app.handles = {
        "sun_enabled": Handle(False),
        "sun_azimuth": Handle(0.0),
        "sun_elevation": Handle(0.0),
    }
    app.controllers = [shared, independent, removed]
    app._shared_lighting_changed = lambda source: updates.append((app, source))

    assert app.apply_estimated_sun_angles(203.0, 70.0) == 2
    assert app.handles["sun_azimuth"].value == -157.0
    for item in (shared, independent):
        assert item.handles["light_sun_enabled"].value is True
        assert item.handles["light_sun_azimuth"].value == -157.0
        assert item.handles["light_sun_azimuth_advanced"].value == -157.0
        assert item.handles["light_sun_elevation"].value == 70.0
        assert item.handles["light_sun_elevation_advanced"].value == 70.0
    assert removed.handles["light_sun_azimuth"].value == 0.0
    assert shared.use_scene_lighting() is True
    assert independent.use_scene_lighting() is False