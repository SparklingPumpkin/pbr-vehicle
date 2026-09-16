from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


def resolve_optional_command(
    command: list[str],
    *,
    executable_name: str,
    module_name: str,
    source_package: str,
    source_script: str,
) -> tuple[list[str], list[str]]:
    """Resolve an optional delivery CLI without relying on the shell PATH alone."""
    if not command:
        return command, []
    requested = command[0]
    checked: list[str] = []

    direct = Path(requested).expanduser()
    if direct.is_file():
        return [str(direct.resolve()), *command[1:]], checked

    path_entry = shutil.which(requested)
    checked.append(f"PATH:{requested}")
    interpreter_entry = Path(sys.executable).resolve().parent / executable_name
    checked.append(str(interpreter_entry))

    spec = importlib.util.find_spec(module_name)
    checked.append(f"module:{module_name}")
    if spec is not None and spec.origin:
        if interpreter_entry.is_file():
            return [str(interpreter_entry), *command[1:]], checked
        if path_entry:
            return [path_entry, *command[1:]], checked
        module_path = Path(spec.origin).resolve()
        script_path = module_path.parent / source_script
        checked.append(str(script_path))
        if script_path.is_file():
            return [sys.executable, str(script_path), *command[1:]], checked

    package_root = Path(__file__).resolve().parents[2]
    adjacent_script = package_root.parent / source_package / "src" / module_name / source_script
    checked.append(str(adjacent_script))
    if adjacent_script.is_file():
        return [sys.executable, str(adjacent_script), *command[1:]], checked

    if path_entry:
        return [path_entry, *command[1:]], checked

    return command, checked