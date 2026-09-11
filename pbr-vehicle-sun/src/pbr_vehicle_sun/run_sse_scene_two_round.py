#!/usr/bin/env python3
"""Run the delivered SSE-v8 scene estimator with one disjoint replacement round."""
from __future__ import annotations

import argparse
import json
import math
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


def run_two_round(
    output_dir: Path,
    forwarded_arguments: list[str],
    retries: int,
    executor: RoundExecutor = execute_round,
) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = normalize_forwarded_arguments(forwarded_arguments)
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
        "mainline": "SSE-v8",
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
    }
    temporary = output_dir / "two_round_result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "two_round_result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "All other scene/model arguments are forwarded to the single-round "
            "SSE-v8 estimator. Selection ranks and cache arguments are managed here."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="parent directory containing round-1, optional round-2, and two_round_result.json")
    parser.add_argument("--process-retries", type=int, default=1,
                        help="additional retries for each failed process round")
    owned, forwarded = parser.parse_known_args()
    if owned.process_retries < 0:
        parser.error("--process-retries must be non-negative")
    try:
        result = run_two_round(owned.output_dir, forwarded, owned.process_retries)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    if result["status"] == "process_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()