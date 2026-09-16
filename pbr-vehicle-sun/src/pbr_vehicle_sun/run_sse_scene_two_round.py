#!/usr/bin/env python3
"""Run the delivered SSE-v9 scene estimator with one disjoint replacement round."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


ROUND_SCRIPT = Path(__file__).resolve().with_name("run_sse_scene_global_top3.py")
ROUND_ONE_RANKS = [1, 3]
ROUND_TWO_RANKS = [4, 6]


def option_value(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 >= len(arguments):
                raise ValueError(f"{name} requires a value")
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def without_option(arguments: list[str], name: str) -> list[str]:
    result: list[str] = []
    skip_value = False
    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if argument == name:
            skip_value = True
            continue
        if argument.startswith(name + "="):
            continue
        result.append(argument)
    if skip_value:
        raise ValueError(f"{name} requires a value")
    return result


def normalize_forwarded_arguments(arguments: list[str]) -> list[str]:
    top_vehicles = option_value(arguments, "--top-vehicles")
    if top_vehicles is not None and int(top_vehicles) != 3:
        raise ValueError("two-round delivery requires --top-vehicles 3")
    for forbidden in ("--existing-detection", "--existing-ranking"):
        if option_value(arguments, forbidden) is not None:
            raise ValueError(f"{forbidden} is managed by the two-round delivery entry point")
    result = list(arguments)
    for owned in ("--top-vehicles", "--selection-pool-size", "--vehicle-rank-offset"):
        result = without_option(result, owned)
    return result


def build_round_command(arguments: list[str], output_dir: Path, round_number: int) -> list[str]:
    if round_number not in (1, 2):
        raise ValueError("round_number must be 1 or 2")
    command = [
        sys.executable,
        str(ROUND_SCRIPT),
        *arguments,
        "--output-dir",
        str(output_dir / f"round-{round_number}"),
        "--top-vehicles",
        "3",
        "--selection-pool-size",
        "6",
        "--vehicle-rank-offset",
        "0" if round_number == 1 else "3",
    ]
    if round_number == 2:
        selection = output_dir / "round-1" / "selection"
        command.extend([
            "--existing-detection",
            str(selection / "all_vehicle_detections.json"),
            "--existing-ranking",
            str(selection / "sam2_global_ranking" / "scene_global_vehicle_ranking.json"),
        ])
    return command


def extract_angle(payload: dict) -> tuple[float | None, float | None, str]:
    aggregate = payload.get("aggregate_result")
    if not isinstance(aggregate, dict) or aggregate.get("status") != "ok":
        return None, None, ""
    joint = aggregate.get("joint_fit")
    if isinstance(joint, dict):
        azimuth, elevation = joint.get("azimuth_deg"), joint.get("elevation_deg")
        if azimuth is not None and elevation is not None:
            return float(azimuth), float(elevation), "joint_fit"
    weighted = aggregate.get("weighted_angle")
    if isinstance(weighted, dict):
        azimuth = weighted.get("weighted_azimuth_deg")
        elevation = weighted.get("weighted_elevation_deg")
        if azimuth is not None and elevation is not None:
            return float(azimuth), float(elevation), "weighted_angle"
    accepted = [row for row in aggregate.get("vehicles", []) if row.get("status") == "accepted"]
    if len(accepted) == 1:
        return float(accepted[0]["azimuth_deg"]), float(accepted[0]["elevation_deg"]), "single_vehicle"
    return None, None, ""


def summarize_round(round_number: int, execution: dict) -> dict:
    payload = execution.get("payload")
    summary = {
        "round": round_number,
        "rank_range": ROUND_ONE_RANKS if round_number == 1 else ROUND_TWO_RANKS,
        "process_status": execution["process_status"],
        "attempts": execution.get("attempts", []),
        "resumed": execution.get("resumed", False),
        "result_path": execution.get("result_path"),
        "publication_status": "process_failed",
        "published_angle": False,
        "azimuth_deg": None,
        "elevation_deg": None,
        "angle_source": "",
        "selected_vehicle_ranks": [],
        "prepared_vehicle_ranks": [],
        "accepted_vehicle_ranks": [],
    }
    if execution["process_status"] != "completed" or not isinstance(payload, dict):
        return summary
    status = payload.get("status")
    if status not in ("ok", "no_valid_sun_information"):
        summary["process_status"] = "failed"
        summary["contract_error"] = f"unexpected scene status: {status!r}"
        return summary
    azimuth, elevation, source = extract_angle(payload)
    published = (
        status == "ok"
        and azimuth is not None
        and elevation is not None
        and math.isfinite(azimuth)
        and math.isfinite(elevation)
    )
    if status == "ok" and not published:
        summary["process_status"] = "failed"
        summary["contract_error"] = "status ok did not contain a finite published angle"
        return summary
    prepared = [int(row["vehicle_rank"]) for row in payload.get("prepared_vehicle_inputs", [])]
    accepted_indices = (payload.get("aggregate_result") or {}).get("accepted_vehicle_indices", [])
    summary.update({
        "publication_status": status,
        "published_angle": published,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "angle_source": source,
        "selected_vehicle_ranks": [
            int(row["vehicle_rank"]) for row in payload.get("vehicle_preparation_results", [])
        ],
        "prepared_vehicle_ranks": prepared,
        "accepted_vehicle_ranks": [
            prepared[index - 1] for index in accepted_indices if 0 < index <= len(prepared)
        ],
    })
    return summary


def execute_round(command: list[str], round_dir: Path, retries: int) -> dict:
    result_path = round_dir / "scene_result.json"
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("status") in ("ok", "no_valid_sun_information"):
            return {
                "process_status": "completed",
                "payload": payload,
                "attempts": [],
                "resumed": True,
                "result_path": str(result_path.resolve()),
            }
    round_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for attempt in range(1, retries + 2):
        log_path = round_dir / f"attempt-{attempt}.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        attempts.append({
            "attempt": attempt,
            "returncode": completed.returncode,
            "elapsed_s": time.perf_counter() - started,
            "log_path": str(log_path.resolve()),
        })
        if completed.returncode == 0 and result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                return {
                    "process_status": "completed",
                    "payload": payload,
                    "attempts": attempts,
                    "resumed": False,
                    "result_path": str(result_path.resolve()),
                }
    return {
        "process_status": "failed",
        "payload": None,
        "attempts": attempts,
        "resumed": False,
        "result_path": str(result_path.resolve()),
    }


RoundExecutor = Callable[..., dict]


def _path_fingerprint(path: Path) -> dict:
    """Cheap invalidation fingerprint for cache inputs and model assets."""
    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    # Repository and dataset roots can contain millions of files. Their own
    # metadata is intentionally cheap; explicit checkpoint/image arguments
    # still receive file-level size/mtime invalidation above.
    stat = path.stat()
    return {"path": str(path), "directory_mtime_ns": stat.st_mtime_ns}


def _dataset_fingerprint(path: Path) -> dict:
    """Fingerprint scene inputs by metadata without reading every image payload."""
    root = path.expanduser().resolve()
    if not root.is_dir():
        return _path_fingerprint(root)
    files = []
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = child.stat()
        files.append((str(child.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    encoded = json.dumps(files, separators=(",", ":")).encode()
    return {
        "path": str(root),
        "file_count": len(files),
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def cache_key(arguments: list[str]) -> str:
    """Fingerprint the complete invocation and all path-valued arguments."""
    source = Path(__file__).resolve().parent
    payload: dict = {
        "arguments": arguments,
        "paths": [],
        "package_source": [_path_fingerprint(path) for path in sorted(source.glob("*.py"))],
    }
    data_root = option_value(arguments, "--data-root")
    if data_root is not None:
        payload["dataset"] = _dataset_fingerprint(Path(data_root))
    seen = set()
    for token in arguments:
        candidate = Path(token).expanduser()
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        payload["paths"].append(_path_fingerprint(candidate))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _copy_tree_hardlink(source: Path, destination: Path) -> None:
    """Materialize a cache tree cheaply while retaining standalone outputs."""
    destination.mkdir(parents=True, exist_ok=True)
    def link_or_copy(src: str, dst: str) -> str:
        # Manifests are rewritten when a cache tree is restored. Keep them
        # private copies; hard-linking them would mutate the cache entry.
        if Path(src).suffix.lower() in {".json", ".md", ".csv", ".txt"}:
            return shutil.copy2(src, dst)
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy2(src, dst)
    shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=link_or_copy)


def _rewrite_cached_paths(root: Path, old_root: Path, new_root: Path) -> None:
    old, new = str(old_root.resolve()), str(new_root.resolve())
    for pattern in ("*.json", "*.md", "*.csv"):
        for path in root.rglob(pattern):
            try:
                value = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if old in value:
                path.write_text(value.replace(old, new), encoding="utf-8")


def restore_cached_run(cache_entry: Path, output_dir: Path) -> dict | None:
    result_path = cache_entry / "two_round_result.json"
    marker = cache_entry / "cache_manifest.json"
    if not result_path.is_file() or not marker.is_file():
        return None
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    _copy_tree_hardlink(cache_entry, output_dir)
    _rewrite_cached_paths(output_dir, cache_entry, output_dir)
    result = json.loads((output_dir / "two_round_result.json").read_text(encoding="utf-8"))
    result["cache"] = {"hit": True, "entry": str(cache_entry.resolve())}
    (output_dir / "two_round_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def store_cached_run(output_dir: Path, cache_entry: Path, key: str) -> None:
    if cache_entry.exists():
        return
    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_entry.with_name(cache_entry.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    _copy_tree_hardlink(output_dir, temporary)
    _rewrite_cached_paths(temporary, output_dir, cache_entry)
    (temporary / "cache_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "cache_key": key,
        "source_contract": "complete SSE-v9 audited run; invalidated by invocation and input/model path metadata",
    }, indent=2) + "\n")
    try:
        temporary.rename(cache_entry)
    except FileExistsError:
        shutil.rmtree(temporary)


def run_two_round(
    output_dir: Path,
    forwarded_arguments: list[str],
    retries: int,
    executor: RoundExecutor = execute_round,
    cache_dir: Path | None = None,
) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = normalize_forwarded_arguments(forwarded_arguments)
    key = cache_key(arguments)
    if cache_dir is not None and executor is execute_round:
        cache_entry = cache_dir.expanduser().resolve() / "complete-runs" / key
        cached = restore_cached_run(cache_entry, output_dir)
        if cached is not None:
            return cached
    started = time.perf_counter()
    rounds = []
    for round_number in (1, 2):
        round_dir = output_dir / f"round-{round_number}"
        command = build_round_command(arguments, output_dir, round_number)
        execution = executor(command, round_dir, retries)
        summary = summarize_round(round_number, execution)
        summary["command"] = command
        rounds.append(summary)
        if summary["process_status"] != "completed" or summary["published_angle"]:
            break
    final_round = rounds[-1]
    if final_round["process_status"] != "completed":
        status = "process_failed"
    elif final_round["published_angle"]:
        status = "ok"
    else:
        status = "no_valid_sun_information"
    result = {
        "schema_version": 1,
        "mainline": "SSE-v9",
        "status": status,
        "published_angle": status == "ok",
        "published_round": final_round["round"] if status == "ok" else None,
        "azimuth_deg": final_round["azimuth_deg"],
        "elevation_deg": final_round["elevation_deg"],
        "angle_source": final_round["angle_source"],
        "accepted_vehicle_ranks": final_round["accepted_vehicle_ranks"],
        "rounds_executed": len(rounds),
        "replacement_triggered": len(rounds) == 2,
        "round_contract": {
            "round_1_ranks": ROUND_ONE_RANKS,
            "round_2_ranks": ROUND_TWO_RANKS,
            "round_2_trigger": "round_1 completed with no_valid_sun_information",
            "round_2_reuses": ["all_vehicle_detections.json", "scene_global_vehicle_ranking.json"],
            "process_failure_behavior": "stop without replacement and return process_failed",
        },
        "rounds": rounds,
        "total_elapsed_s": time.perf_counter() - started,
        "cache": {"hit": False, "key": key},
    }
    temporary = output_dir / "two_round_result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "two_round_result.json")
    if cache_dir is not None and executor is execute_round:
        store_cached_run(output_dir, cache_entry, key)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "All other scene/model arguments are forwarded to the single-round "
            "SSE-v9 estimator. Selection ranks and cache arguments are managed here."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="parent directory containing round-1, optional round-2, and two_round_result.json")
    parser.add_argument("--process-retries", type=int, default=1,
                        help="additional retries for each failed process round")
    parser.add_argument("--cache-dir", type=Path,
                        help="optional persistent cache for complete audited SSE-v9 runs")
    owned, forwarded = parser.parse_known_args()
    if owned.process_retries < 0:
        parser.error("--process-retries must be non-negative")
    try:
        result = run_two_round(owned.output_dir, forwarded, owned.process_retries,
                               cache_dir=owned.cache_dir)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    if result["status"] == "process_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
