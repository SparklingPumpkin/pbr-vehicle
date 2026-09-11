from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pbr_vehicle_sun.fit_sun_camera_visible_arc import maximal_camera_cone_middle_arc
from pbr_vehicle_sun.run_sse_confidence_gated_joint import (
    gate_confidence, publication_status, validate_vehicle_count,
)
from pbr_vehicle_sun.run_sse_scene_global_top3 import (
    VehicleMethodRejection,
    is_vehicle_evidence_failure,
    select_ranked_vehicles,
    ssisv2_command,
)
from pbr_vehicle_sun.run_sse_scene_two_round import run_two_round


ROOT = Path(__file__).resolve().parents[1]


def test_maximal_camera_cone_selects_near_boundary() -> None:
    contour = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    arc, audit = maximal_camera_cone_middle_arc(contour, np.array([2.0, -5.0]))
    assert audit["hit_contract_1_or_2_each"] is True
    assert audit["left_hit_count"] in (1, 2)
    assert audit["right_hit_count"] in (1, 2)
    assert np.isclose(np.min(arc[:, 1]), 0.0, atol=1e-12)


def test_default_dispatch_is_sse_v7_ssisv2_direct_mask(tmp_path: Path) -> None:
    files = []
    for name in ("vehicle.npz", "shadow.npz", "mask.png"):
        path = tmp_path / name
        path.write_bytes(b"placeholder")
        files.append(path)
    output = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([
        sys.executable, "-m", "pbr_vehicle_sun.run_sse_mainline",
        "--geometry-source", "feedforward_infinidepth",
        "--vehicle-geometry", str(files[0]), "--shadow-geometry", str(files[1]),
        "--shadow-mask", str(files[2]), "--output-dir", str(output), "--dry-run",
    ], check=True, capture_output=True, text=True, env=env)
    manifest = json.loads((output / "sse_mainline_branch_manifest.json").read_text())
    assert manifest["mainline"] == "SSE-v8"
    assert "official SSISv2 associated shadow mask" in manifest["shadow_mask_contract"]
    assert "--direct-shadow-mask" in manifest["command"]
    assert manifest["feedforward_exclude_image_edge_shadow"] is True
    assert manifest["feedforward_predicted_floor_subtraction"] is False


def test_one_to_five_gate_and_fail_closed_contract() -> None:
    for count in range(1, 6):
        validate_vehicle_count(count)
    assert gate_confidence(.7, .8, .5, .75) == []
    assert publication_status([{"vehicle_index": 1, "status": "rejected"}])["status"] == "no_valid_sun_information"


def test_scene_orchestrator_uses_official_ssisv2_association() -> None:
    args = argparse.Namespace(
        ssis_python=Path("python"), ssis_root=Path("external/ssis"),
        ssis_weights=Path("weights/model.pth"), ssis_device="cuda:0",
        ssis_confidence_threshold=.1, minimum_ssis_object_iou=.05,
    )
    command = ssisv2_command(args, Path("image.jpg"), Path("vehicle.png"), Path("out"))
    assert command[1].endswith("run_ssisv2_vehicle_shadow_association.py")
    assert "--minimum-object-iou" in command


def test_rank_offset_selects_disjoint_replacement_vehicles() -> None:
    ranking = {"selected": [{"vehicle_rank": rank} for rank in range(1, 7)]}
    first = select_ranked_vehicles(ranking, offset=0, count=3)
    second = select_ranked_vehicles(ranking, offset=3, count=3)
    assert [row["vehicle_rank"] for row in first] == [1, 2, 3]
    assert [row["vehicle_rank"] for row in second] == [4, 5, 6]
    assert {row["vehicle_rank"] for row in first}.isdisjoint(
        row["vehicle_rank"] for row in second
    )


def test_method_rejection_is_distinct_from_process_failure() -> None:
    assert issubclass(VehicleMethodRejection, RuntimeError)
    assert not issubclass(subprocess.CalledProcessError, VehicleMethodRejection)
    geometry_error = subprocess.CalledProcessError(
        42, ["python", "lift_source_shadow_with_infinidepth.py"]
    )
    inference_error = subprocess.CalledProcessError(
        1, ["python", "infer_infinidepth_dense_depth.py"]
    )
    assert is_vehicle_evidence_failure(geometry_error) is True
    assert is_vehicle_evidence_failure(inference_error) is False


def scene_payload(status: str, rank: int | None = None) -> dict:
    aggregate = None
    prepared = []
    preparation = []
    if rank is not None:
        prepared = [{"vehicle_rank": rank}]
        preparation = [{"vehicle_rank": rank, "status": "prepared"}]
        aggregate = {
            "status": status,
            "accepted_vehicle_indices": [1],
            "vehicles": [{
                "vehicle_index": 1,
                "status": "accepted",
                "azimuth_deg": 123.0,
                "elevation_deg": 45.0,
            }],
        }
    return {
        "status": status,
        "aggregate_result": aggregate,
        "prepared_vehicle_inputs": prepared,
        "vehicle_preparation_results": preparation,
    }


def test_two_round_delivery_rescues_with_disjoint_replacement(tmp_path: Path) -> None:
    payloads = [scene_payload("no_valid_sun_information"), scene_payload("ok", rank=4)]
    calls = 0

    def fake_executor(command: list[str], round_dir: Path, retries: int) -> dict:
        nonlocal calls
        payload = payloads[calls]
        calls += 1
        return {"process_status": "completed", "payload": payload}

    result = run_two_round(tmp_path, ["--scene", "example"], 0, fake_executor)
    assert result["status"] == "ok"
    assert result["published_round"] == 2
    assert result["accepted_vehicle_ranks"] == [4]
    first, second = result["rounds"]
    assert first["rank_range"] == [1, 3]
    assert second["rank_range"] == [4, 6]
    assert first["command"][1].endswith("run_sse_scene_global_top3.py")
    assert "--existing-detection" in second["command"]
    assert "--existing-ranking" in second["command"]
    assert calls == 2


def test_two_round_delivery_fail_closed_and_process_failure(tmp_path: Path) -> None:
    def rejected(command: list[str], round_dir: Path, retries: int) -> dict:
        return {"process_status": "completed", "payload": scene_payload("no_valid_sun_information")}

    rejected_result = run_two_round(tmp_path / "rejected", ["--scene", "example"], 0, rejected)
    assert rejected_result["status"] == "no_valid_sun_information"
    assert rejected_result["rounds_executed"] == 2

    calls = 0

    def failed(command: list[str], round_dir: Path, retries: int) -> dict:
        nonlocal calls
        calls += 1
        return {"process_status": "failed", "payload": None}

    failed_result = run_two_round(tmp_path / "failed", ["--scene", "example"], 0, failed)
    assert failed_result["status"] == "process_failed"
    assert failed_result["rounds_executed"] == 1
    assert calls == 1


def test_package_has_no_retired_detector_or_source_mask_cleanup() -> None:
    source_dir = ROOT / "src/pbr_vehicle_sun"
    retired = "mt" + "mt"
    for path in source_dir.glob("*.py"):
        text = path.read_text().lower()
        assert retired not in text
        assert "postprocess_shadow_mask" not in text
    assert not (source_dir / ("test_" + retired + "_shadow_detector.py")).exists()
