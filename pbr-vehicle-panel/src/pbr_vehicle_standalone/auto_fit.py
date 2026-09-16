"""Adapter for applying the delivered PBR Vehicle Auto fitter from Viser."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from .optional_command import resolve_optional_command


class VehicleAutoFitError(RuntimeError):
    """The optional auto fitter failed or returned an unsupported result."""


def _emit_progress(callback, progress: float, stage: str) -> None:
    if callback is not None:
        callback(max(0.0, min(1.0, float(progress))), str(stage))


def _progress_snapshot(output_dir: Path) -> tuple[float, str]:
    progress_path = output_dir / "progress.json"
    if progress_path.is_file():
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            return (
                float(payload.get("progress", 0.0)),
                str(payload.get("stage", "读取 Auto 识别进度")),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if (output_dir / "metrics.json").is_file():
        return 1.0, "车辆参数识别完成"
    return 0.05, "加载场景与车辆资产"


def _run_monitored_process(
    command: list[str], timeout: float, progress_callback, output_dir: Path
) -> tuple[int, str]:
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        while process.poll() is None:
            if time.monotonic() - started > float(timeout):
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(command, timeout)
            _emit_progress(progress_callback, *_progress_snapshot(output_dir))
            time.sleep(0.5)
        _emit_progress(progress_callback, *_progress_snapshot(output_dir))
        log.seek(0)
        return process.returncode, log.read()


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
    device: str = "auto",
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
    if "--device" not in command:
        command.extend(["--device", str(config.get("device", device))])
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
        raise VehicleAutoFitError(f"Auto 输出状态无效或不受支持（{status or 'unknown'}），保留当前参数")
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
    device: str = "auto",
    progress_callback=None,
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
        device=device,
        command_override=command_override,
    )
    command, checked = resolve_optional_command(
        command,
        executable_name="pbr-vehicle-auto-fit",
        module_name="pbr_vehicle_auto",
        source_package="pbr-vehicle-auto",
        source_script="fitter.py",
    )
    if command[0] == "pbr-vehicle-auto-fit":
        raise VehicleAutoFitError(
            f"找不到车辆自动适配命令: {command[0]}；已检查: {', '.join(checked)}"
        )
    limit = float(timeout if timeout is not None else config.get("timeout", 600.0))
    _emit_progress(progress_callback, 0.01, "创建车辆参数识别任务")
    try:
        returncode, process_output = _run_monitored_process(
            command, limit, progress_callback, output,
        )
    except subprocess.TimeoutExpired as exc:
        raise VehicleAutoFitError(f"车辆自动适配超时（{limit:g} 秒），中间结果保留在 {root}") from exc
    if returncode != 0:
        detail = process_output.strip()[-2000:]
        raise VehicleAutoFitError(
            f"车辆自动适配失败（退出码 {returncode}），中间结果 {root}: {detail}"
        )
    result = read_auto_fit_result(final_config, output / "metrics.json")
    _emit_progress(progress_callback, 1.0, "车辆参数识别完成")
    result.update({"output_dir": str(root), "command": command, "stdout": process_output[-4000:]})
    return result
