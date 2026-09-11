"""Portable adapter for the delivered PBR Vehicle Sun estimator.

The estimator itself is intentionally kept as a separate delivery package.  A
panel only needs to launch its public scene entry point, retain the temporary
artifacts, and read the published angle from the result manifest.  This module
does not import torch, YOLO, SAM2, SSISv2, or InfiniDepth, so the viewer still
starts when those optional estimator dependencies are not installed.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class SunEstimationError(RuntimeError):
    """Raised when an estimator cannot be run or publishes no valid angle."""


def _emit_progress(callback, progress: float, stage: str) -> None:
    if callback is not None:
        callback(float(np.clip(progress, 0.0, 1.0)), str(stage))


def _round_progress(output: Path, round_number: int) -> tuple[float, str]:
    round_dir = output / f"round-{round_number}"
    label = "第一轮（车辆排名 1–3）" if round_number == 1 else "第二轮（更换车辆排名 4–6）"
    if not round_dir.is_dir():
        return 0.0, f"{label}：等待启动"
    if (round_dir / "scene_result.json").is_file():
        return 1.0, f"{label}：完成发布门判断"
    selection = round_dir / "selection"
    if round_number == 1 and not (selection / "all_vehicle_detections.json").is_file():
        return 0.08, f"{label}：遍历全部图像并检测车辆"
    if round_number == 1 and not (selection / "sam2_global_ranking/scene_global_vehicle_ranking.json").is_file():
        return 0.18, f"{label}：SAM2 分割并生成全局车辆排名"
    associations = list(round_dir.rglob("ssisv2_association_manifest.json"))
    shadow_geometry = list(round_dir.rglob("lift_manifest.json"))
    vehicle_geometry = list(round_dir.rglob("infinidepth_vehicle_geometry_manifest.json"))
    fit_results = list((round_dir / "confidence_gated_joint_fit").glob("vehicle_*/fit/fit_result.json"))
    prepared = min(len(shadow_geometry), len(vehicle_geometry), 3)
    if prepared < 3:
        completed = min(len(associations), 3) + prepared
        return 0.28 + 0.055 * completed, f"{label}：生成深度、关联阴影并准备车辆几何 ({prepared}/3 辆完成)"
    if len(fit_results) < 3:
        return 0.62 + 0.10 * len(fit_results), f"{label}：逐车拟合相机可见阴影轮廓 ({len(fit_results)}/3 辆完成)"
    return 0.94, f"{label}：聚合置信度并执行发布门判断"


def _progress_snapshot(output: Path) -> tuple[float, str]:
    if (output / "two_round_result.json").is_file():
        return 1.0, "太阳估计完成"
    if (output / "round-2").is_dir():
        progress, stage = _round_progress(output, 2)
        return 0.50 + 0.48 * progress, stage
    progress, stage = _round_progress(output, 1)
    first_result = output / "round-1/scene_result.json"
    if first_result.is_file():
        try:
            if json.loads(first_result.read_text(encoding="utf-8")).get("status") == "no_valid_sun_information":
                return 0.50, "第一轮未发布角度，正在切换替换车辆"
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return 0.48 * progress, stage


def _run_monitored_process(command, timeout: float, progress_callback, output: Path) -> tuple[int, str]:
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        while process.poll() is None:
            if time.monotonic() - started > float(timeout):
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(command, timeout)
            _emit_progress(progress_callback, *_progress_snapshot(output))
            time.sleep(0.5)
        _emit_progress(progress_callback, *_progress_snapshot(output))
        log.seek(0)
        return process.returncode, log.read()


_OPTIONAL_KEYS = (
    "yolo_weights", "sam2_root", "sam2_checkpoint", "sam2_python", "sam2_config",
    "ssis_root", "ssis_weights", "ssis_python", "ssis_device",
    "ssis_confidence_threshold", "minimum_ssis_object_iou", "infinidepth_root",
    "infinidepth_python", "infinidepth_depth_checkpoint", "infinidepth_gs_checkpoint",
    "moge2_pretrained", "sky_checkpoint", "infinidepth_sample_points", "device",
    "top_vehicles", "min_mask_area_ratio", "min_vehicle_confidence",
    "min_vehicle_support", "cameras", "existing_detection", "existing_ranking",
)


def _format_part(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(_MissingPlaceholder(context))
    if isinstance(value, (list, tuple)):
        return [_format_part(item, context) for item in value]
    return value


class _MissingPlaceholder(dict):
    def __missing__(self, key: str) -> str:
        # Keeping an unknown placeholder visible makes a malformed config
        # obvious to the estimator rather than silently changing its command.
        return "{" + key + "}"


def _read_config(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SunEstimationError(f"太阳估计配置必须是 JSON 对象: {path}")
    return payload


def _as_command(command: Any) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, (list, tuple)):
        return [str(part) for part in command]
    raise SunEstimationError("太阳估计 command 必须是字符串或字符串数组")


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, bool):
        if value and flag not in command:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        if flag in command:
            return
        command.extend([flag, *map(str, value)])
        return
    if flag not in command:
        command.extend([flag, str(value)])


def build_sun_estimator_command(
    config: dict[str, Any],
    *,
    scene: str,
    data_root: str,
    output_dir: Path,
    scene_path: str = "",
    command_override: str | Iterable[str] | None = None,
) -> list[str]:
    """Build a subprocess command from the portable JSON contract.

    A config may provide a complete ``command`` (with ``{scene}``,
    ``{data_root}``, ``{output_dir}``, and ``{scene_path}`` placeholders), or
    only resource fields.  The latter is expanded to the official
    ``pbr-vehicle-sun-scene`` CLI and is convenient for both panels.
    """
    context = {
        "scene": str(scene or ""),
        "data_root": str(data_root or ""),
        "output_dir": str(output_dir),
        "scene_path": str(scene_path or ""),
    }
    command_value = command_override if command_override else config.get("command")
    if command_value:
        command = [_format_part(part, context) for part in _as_command(command_value)]
    else:
        executable = str(config.get("executable", "pbr-vehicle-sun-scene"))
        command = [executable]
        _append_option(command, "--scene", scene or config.get("scene"))
        _append_option(command, "--data-root", data_root or config.get("data_root"))
        for key in _OPTIONAL_KEYS:
            _append_option(command, "--" + key.replace("_", "-"), config.get(key))
        if not any(part == "--output-dir" for part in command):
            _append_option(command, "--output-dir", output_dir)
    # A custom command may omit output-dir deliberately only when it already
    # writes the standard manifest into the supplied directory.  Append it in
    # the common case so a short ``["pbr-vehicle-sun-scene"]`` config works.
    if "--output-dir" not in command:
        command.extend(["--output-dir", str(output_dir)])
    return [str(_format_part(part, context)) for part in command]


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _angle_from_mapping(payload: Any) -> tuple[float, float] | None:
    if not isinstance(payload, dict):
        return None
    if "status" in payload and str(payload["status"]) not in {"ok", "accepted", "success"}:
        return None
    # The promoted multi-vehicle contract publishes this first.
    for key in ("joint_fit", "weighted_angle", "best", "aggregate_result"):
        candidate = payload.get(key)
        result = _angle_from_mapping(candidate)
        if result is not None:
            return result
    azimuth = None
    elevation = None
    for key in ("weighted_azimuth_deg", "circular_mean_azimuth_deg", "azimuth_deg"):
        azimuth = _finite_number(payload.get(key))
        if azimuth is not None:
            break
    for key in ("weighted_elevation_deg", "median_elevation_deg", "elevation_deg"):
        elevation = _finite_number(payload.get(key))
        if elevation is not None:
            break
    if azimuth is not None and elevation is not None:
        return azimuth % 360.0, max(-90.0, min(90.0, elevation))
    # A single accepted vehicle is also a valid publication.
    for row in payload.get("vehicles", []) if isinstance(payload.get("vehicles"), list) else []:
        if row.get("status") in {"accepted", "ok", "success"}:
            result = _angle_from_mapping(row)
            if result is not None:
                return result
    return None


def _world_direction(output_dir: Path, azimuth_deg: float, elevation_deg: float) -> list[float] | None:
    """Lift the estimator's road-plane angle into its world coordinates."""
    contour_files = sorted(output_dir.rglob("frame0_best_contours.npz"))
    for path in contour_files:
        try:
            with np.load(path) as data:
                e1 = np.asarray(data["plane_e1"], dtype=np.float64).reshape(3)
                e2 = np.asarray(data["plane_e2"], dtype=np.float64).reshape(3)
            normal = np.cross(e1, e2)
            normal /= np.linalg.norm(normal)
            azimuth = math.radians(float(azimuth_deg))
            elevation = math.radians(float(elevation_deg))
            direction = math.cos(elevation) * (math.cos(azimuth) * e1 + math.sin(azimuth) * e2) + math.sin(elevation) * normal
            direction /= np.linalg.norm(direction)
            if np.isfinite(direction).all():
                return direction.tolist()
        except (OSError, KeyError, ValueError):
            continue
    return None


def read_sun_estimate(output_dir: str | Path) -> dict[str, Any]:
    """Read a published SSE result and return normalized angles."""
    root = Path(output_dir).expanduser().resolve()
    two_round = root / "two_round_result.json"
    published_round = None
    if two_round.is_file():
        payload = json.loads(two_round.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("status") not in {"ok", "accepted", "success"}:
            status = payload.get("status") if isinstance(payload, dict) else "invalid_result"
            raise SunEstimationError(f"太阳估计未通过发布门（{status}），输出目录: {root}")
        try:
            published_round = int(payload["published_round"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SunEstimationError(f"两轮太阳估计结果缺少有效 published_round，输出目录: {root}") from exc
        candidates = [two_round]
    else:
        candidates = []
    scene_result = root / "scene_result.json"
    if not candidates and scene_result.is_file():
        try:
            scene_payload = json.loads(scene_result.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            scene_payload = None
        if isinstance(scene_payload, dict) and scene_payload.get("status") not in (None, "ok", "accepted", "success"):
            raise SunEstimationError(
                f"太阳估计未通过发布门（{scene_payload.get('status')}），输出目录: {root}"
            )
    if not candidates:
        preferred = [
            scene_result,
            root / "confidence_gated_joint_fit" / "multi_vehicle_fit_result.json",
            root / "fit_result.json",
        ]
        candidates = [path for path in preferred if path.is_file()]
        candidates.extend(path for path in sorted(root.rglob("*.json")) if path not in candidates)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status", "ok"))
        if "status" in payload and status not in {"ok", "accepted", "success"}:
            continue
        angle = _angle_from_mapping(payload)
        if angle is None:
            continue
        result = {
            "azimuth_deg": angle[0],
            "elevation_deg": angle[1],
            "status": status,
            "result_path": str(path),
            "payload": payload,
        }
        contour_root = root / f"round-{published_round}" if published_round is not None else root
        world_direction = _world_direction(contour_root, angle[0], angle[1])
        if world_direction is not None:
            result["world_direction"] = world_direction
        if published_round is not None:
            result["published_round"] = published_round
        return result
    raise SunEstimationError(f"太阳估计未发布有效角度，输出目录: {root}")


def run_sun_estimator(
    *,
    config_path: str | Path | None = None,
    scene: str = "",
    data_root: str | Path | None = None,
    scene_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    command_override: str | Iterable[str] | None = None,
    timeout: float = 3600.0,
    progress_callback=None,
) -> dict[str, Any]:
    """Run SSE in a retained temporary directory and parse its published angle."""
    config = _read_config(config_path)
    output = Path(output_dir).expanduser().resolve() if output_dir else Path(
        tempfile.mkdtemp(prefix="pbr_vehicle_sun_")
    )
    output.mkdir(parents=True, exist_ok=True)
    effective_scene = str(scene or config.get("scene") or "")
    effective_data_root = str(data_root or config.get("data_root") or "")
    command = build_sun_estimator_command(
        config,
        scene=effective_scene,
        data_root=effective_data_root,
        output_dir=output,
        scene_path=str(scene_path or ""),
        command_override=command_override,
    )
    if not command:
        raise SunEstimationError("未配置太阳估计命令")
    executable = shutil.which(command[0]) or (command[0] if Path(command[0]).exists() else None)
    if executable is None:
        raise SunEstimationError(f"找不到太阳估计命令: {command[0]}")
    command[0] = executable
    _emit_progress(progress_callback, 0.01, "创建太阳估计任务")
    try:
        returncode, process_output = _run_monitored_process(command, float(timeout), progress_callback, output)
    except subprocess.TimeoutExpired as exc:
        raise SunEstimationError(f"太阳估计超时（{timeout:g} 秒），中间结果保留在 {output}") from exc
    if returncode != 0:
        detail = process_output.strip()[-2000:]
        raise SunEstimationError(
            f"太阳估计命令失败（退出码 {returncode}），输出目录 {output}: {detail}"
        )
    result = read_sun_estimate(output)
    _emit_progress(progress_callback, 1.0, "太阳角度通过发布门")
    result.update({"output_dir": str(output), "command": command, "stdout": process_output[-4000:]})
    return result
