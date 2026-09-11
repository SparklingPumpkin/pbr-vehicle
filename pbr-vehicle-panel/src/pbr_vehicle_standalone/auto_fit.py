"""Adapter for applying the delivered PBR Vehicle Auto fitter from Viser."""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


class VehicleAutoFitError(RuntimeError):
    """The optional auto fitter failed or rejected its candidate."""


def _read_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VehicleAutoFitError(f"车辆自动适配配置必须是 JSON 对象: {source}")
    return payload


def _command_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value]
    raise VehicleAutoFitError("车辆自动适配 command 必须是字符串或字符串数组")


def build_auto_fit_command(
    config: dict[str, Any],
    *,
    scene_ply: Path,
    asset_dir: Path,
    template_config: Path,
    output_dir: Path,
    final_config: Path,
    command_override: str | Iterable[str] | None = None,
) -> list[str]:
    context = {
        "scene_ply": str(scene_ply),
        "asset_dir": str(asset_dir),
        "template_config": str(template_config),
        "output_dir": str(output_dir),
        "final_config": str(final_config),
    }
    value = command_override if command_override else config.get("command")
    if value:
        command = [part.format(**context) for part in _command_parts(value)]
    else:
        command = [str(config.get("executable", "pbr-vehicle-auto-fit"))]
    required = {
        "--scene-ply": scene_ply,
        "--asset-dir": asset_dir,
        "--template-config": template_config,
        "--output-dir": output_dir,
        "--final-config": final_config,
    }
    for flag, path in required.items():
        if flag not in command:
            command.extend([flag, str(path)])
    for key, default in (("max_seconds", 480.0), ("bins", 512)):
        flag = "--" + key.replace("_", "-")
        if flag not in command:
            command.extend([flag, str(config.get(key, default))])
    return command


def _finite(mapping: dict[str, Any], key: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise VehicleAutoFitError(f"Auto 输出缺少有效参数: {key}") from exc
    if not math.isfinite(value):
        raise VehicleAutoFitError(f"Auto 输出参数不是有限数: {key}")
    return value


def read_auto_fit_result(final_config: str | Path, metrics_path: str | Path) -> dict[str, Any]:
    metrics_source = Path(metrics_path).expanduser().resolve()
    metrics = json.loads(metrics_source.read_text(encoding="utf-8"))
    status = str(metrics.get("status", ""))
    if status != "candidate_only":
        raise VehicleAutoFitError(f"Auto 候选未改善场景匹配（{status or 'unknown'}），保留当前参数")
    result_source = Path(final_config).expanduser().resolve()
    payload = json.loads(result_source.read_text(encoding="utf-8"))
    vehicle = payload.get("vehicle") if isinstance(payload, dict) else None
    asset = payload.get("pbr_asset", payload) if isinstance(payload, dict) else None
    if isinstance(vehicle, dict):
        material = vehicle
        lighting = vehicle.get("vehicle_lighting", {})
        values = {
            "sun_intensity": _finite(lighting, "sun_intensity"),
            "ambient_fill": _finite(material, "ambient_fill"),
            "environment_temperature_k": _finite(lighting, "environment_temperature"),
        }
    elif isinstance(asset, dict):
        material = asset.get("material", {})
        lighting = asset.get("light", {})
        values = {
            "sun_intensity": _finite(lighting, "intensity"),
            "ambient_fill": _finite(material, "ambient_fill"),
            "environment_temperature_k": _finite(lighting, "environment_cct_kelvin"),
        }
    else:
        raise VehicleAutoFitError("Auto 最终配置格式无效")
    return {
        **values,
        "status": status,
        "improvement_percent": float(metrics.get("metric", {}).get("improvement_percent", 0.0)),
        "final_config": str(result_source),
        "metrics_path": str(metrics_source),
        "payload": payload,
    }


def run_vehicle_auto_fit(
    *,
    scene_ply: str | Path,
    asset_dir: str | Path,
    template_payload: dict[str, Any],
    config_path: str | Path | None = None,
    command_override: str | Iterable[str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Fit three appearance controls and retain all artifacts in /tmp."""
    config = _read_config(config_path)
    root = Path(tempfile.mkdtemp(prefix="pbr_vehicle_auto_"))
    output = root / "output"
    output.mkdir(parents=True)
    template = root / "current_panel_state.json"
    final_config = root / "auto_fit_config.json"
    template.write_text(json.dumps(template_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scene = Path(scene_ply).expanduser().resolve()
    asset = Path(asset_dir).expanduser().resolve()
    if not scene.is_file() or scene.suffix.lower() != ".ply":
        raise VehicleAutoFitError(f"Auto 需要普通 Gaussian PLY 场景: {scene}")
    if not asset.is_dir():
        raise VehicleAutoFitError(f"车辆资产目录不存在: {asset}")
    command = build_auto_fit_command(
        config,
        scene_ply=scene,
        asset_dir=asset,
        template_config=template,
        output_dir=output,
        final_config=final_config,
        command_override=command_override,
    )
    executable = shutil.which(command[0]) or (command[0] if Path(command[0]).exists() else None)
    if executable is None:
        raise VehicleAutoFitError(f"找不到车辆自动适配命令: {command[0]}")
    command[0] = executable
    limit = float(timeout if timeout is not None else config.get("timeout", 600.0))
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True,
            timeout=limit, env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        raise VehicleAutoFitError(f"车辆自动适配超时（{limit:g} 秒），中间结果保留在 {root}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise VehicleAutoFitError(
            f"车辆自动适配失败（退出码 {completed.returncode}），中间结果 {root}: {detail}"
        )
    result = read_auto_fit_result(final_config, output / "metrics.json")
    result.update({"output_dir": str(root), "command": command, "stdout": completed.stdout[-4000:]})
    return result

