import sys
from pathlib import Path

import pbr_vehicle_standalone.optional_command as command_module
from pbr_vehicle_standalone.optional_command import resolve_optional_command


def test_resolves_console_script_next_to_current_python(monkeypatch, tmp_path):
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir()
    interpreter.touch()
    entry = interpreter.parent / "pbr-vehicle-auto-fit"
    entry.touch()
    monkeypatch.setattr(sys, "executable", str(interpreter))
    spec = type("Spec", (), {"origin": str(tmp_path / "installed/__init__.py")})()
    monkeypatch.setattr(command_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(command_module.importlib.util, "find_spec", lambda _: spec)

    command, _ = resolve_optional_command(
        ["pbr-vehicle-auto-fit", "--device", "auto"],
        executable_name="pbr-vehicle-auto-fit",
        module_name="missing_auto_module",
        source_package="pbr-vehicle-auto",
        source_script="fitter.py",
    )

    assert command == [str(entry), "--device", "auto"]


def test_ignores_stale_console_script_when_module_is_missing(monkeypatch, tmp_path):
    interpreter = tmp_path / "env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    stale_entry = interpreter.parent / "pbr-vehicle-auto-fit"
    stale_entry.touch()
    package_root = tmp_path / "pbr-vehicle-panel"
    module_file = package_root / "src" / "pbr_vehicle_standalone" / "optional_command.py"
    module_file.parent.mkdir(parents=True)
    adjacent = tmp_path / "pbr-vehicle-auto" / "src" / "pbr_vehicle_auto" / "fitter.py"
    adjacent.parent.mkdir(parents=True)
    adjacent.touch()
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr(command_module, "__file__", str(module_file))
    monkeypatch.setattr(command_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(command_module.importlib.util, "find_spec", lambda _: None)

    command, _ = resolve_optional_command(
        ["pbr-vehicle-auto-fit"],
        executable_name="pbr-vehicle-auto-fit",
        module_name="pbr_vehicle_auto",
        source_package="pbr-vehicle-auto",
        source_script="fitter.py",
    )

    assert command == [str(interpreter), str(adjacent)]


def test_resolves_installed_module_source(monkeypatch, tmp_path):
    package = tmp_path / "pbr_vehicle_sun"
    package.mkdir()
    init = package / "__init__.py"
    init.touch()
    script = package / "run_sse_scene_two_round.py"
    script.touch()
    spec = type("Spec", (), {"origin": str(init)})()
    monkeypatch.setattr(command_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(command_module.importlib.util, "find_spec", lambda _: spec)

    command, _ = resolve_optional_command(
        ["pbr-vehicle-sun-scene", "--scene", "017"],
        executable_name="pbr-vehicle-sun-scene",
        module_name="pbr_vehicle_sun",
        source_package="pbr-vehicle-sun",
        source_script="run_sse_scene_two_round.py",
    )

    assert command == [sys.executable, str(script), "--scene", "017"]


def test_keeps_explicit_executable_path(tmp_path):
    script = tmp_path / "custom-fit"
    script.touch()

    command, checked = resolve_optional_command(
        [str(script), "--flag"],
        executable_name="pbr-vehicle-auto-fit",
        module_name="pbr_vehicle_auto",
        source_package="pbr-vehicle-auto",
        source_script="fitter.py",
    )

    assert command == [str(script.resolve()), "--flag"]
    assert checked == []