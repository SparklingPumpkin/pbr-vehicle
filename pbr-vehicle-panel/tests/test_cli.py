from importlib.metadata import entry_points

import pbr_vehicle_standalone
from pbr_vehicle_standalone.cli import build_parser


def test_cli_defaults_are_portable_and_load_full_scene():
    args = build_parser().parse_args([])

    assert args.scene is None
    assert args.vehicle_asset_folder is None
    assert args.scene_cache_dir.as_posix() == ".cache/pbr_vehicle_scenes"
    assert args.scene_max_splats == 0
    assert args.device == "auto"
    assert not args.scene_cache_dir.is_absolute()


def test_installed_distribution_exposes_both_cli_names():
    available = entry_points()
    console_scripts = (
        available.select(group="console_scripts")
        if hasattr(available, "select")
        else available.get("console_scripts", ())
    )
    scripts = {
        entry.name: entry.value
        for entry in console_scripts
        if entry.name.startswith("pbr-vehicle-")
    }

    assert pbr_vehicle_standalone.__version__
    assert scripts["pbr-vehicle-panel"] == "pbr_vehicle_standalone.cli:main"
    assert scripts["pbr-vehicle-viewer"] == "pbr_vehicle_standalone.cli:main"
