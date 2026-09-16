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
    combined_coarse_seeds, fit_command, fit_subprocess_environment,
    geometry_uses_subsampling,
    gate_confidence, publication_status, validate_vehicle_count,
)
from pbr_vehicle_sun.run_sse_scene_global_top3 import (
    VehicleMethodRejection,
    is_vehicle_evidence_failure,
    select_ranked_vehicles,
    ssisv2_command,
)
from pbr_vehicle_sun.device import resolve_device
from pbr_vehicle_sun.sam2_batching import (
    best_multimasks,
    process_sam2_image_batch,
)


def test_portable_device_formats_for_torch_and_ultralytics():
    cpu = resolve_device("cpu")
    assert (cpu.torch, cpu.ultralytics, cpu.uses_cuda) == ("cpu", "cpu", False)
    cuda = resolve_device("cuda:3")
    assert (cuda.torch, cuda.ultralytics, cuda.uses_cuda) == ("cuda:3", "3", True)
    automatic = resolve_device("auto")
    assert automatic.torch in {"cpu", "cuda:0"}
from pbr_vehicle_sun.run_sse_scene_two_round import (
    cache_key,
    restore_cached_run,
    run_two_round,
    store_cached_run,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sam2_multibox_batch_preserves_every_observation() -> None:
    class FakePredictor:
        def __init__(self) -> None:
            self.images = None
            self.box_batch = None

        def set_image_batch(self, images):
            self.images = images

        def predict_batch(self, *, box_batch, multimask_output):
            self.box_batch = box_batch
            assert multimask_output is True
            masks_batch, scores_batch = [], []
            for boxes in box_batch:
                masks = np.zeros((len(boxes), 3, 4, 5), dtype=bool)
                scores = np.tile(np.array([.1, .9, .2]), (len(boxes), 1))
                for index in range(len(boxes)):
                    masks[index, 1, index % 4, index % 5] = True
                masks_batch.append(masks)
                scores_batch.append(scores)
            return masks_batch, scores_batch, [None] * len(box_batch)

    image = np.zeros((4, 5, 3), np.uint8)
    records_a = [{"bbox_xyxy": [0, 0, 2, 2], "id": "a"},
                 {"bbox_xyxy": [1, 1, 3, 3], "id": "b"}]
    records_b = [{"bbox_xyxy": [0, 0, 4, 4], "id": "c"}]
    predictor = FakePredictor()
    result = process_sam2_image_batch(
        predictor, [("a.jpg", image, records_a), ("b.jpg", image, records_b)]
    )
    assert [record["id"] for record, _, _ in result] == ["a", "b", "c"]
    assert [score for _, _, score in result] == [.9, .9, .9]
    assert [boxes.shape for boxes in predictor.box_batch] == [(2, 4), (1, 4)]
    assert len(predictor.images) == 2


def test_sam2_singleton_shape_is_normalized() -> None:
    masks = np.zeros((3, 4, 5), dtype=bool)
    masks[2, 1:3, 1:4] = True
    selected = best_multimasks(masks, np.array([.1, .2, .8]))
    assert len(selected) == 1
    assert selected[0][0].sum() == 6
    assert selected[0][1] == .8


def test_maximal_camera_cone_selects_near_boundary() -> None:
    contour = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    arc, audit = maximal_camera_cone_middle_arc(contour, np.array([2.0, -5.0]))
    assert audit["hit_contract_1_or_2_each"] is True
    assert audit["left_hit_count"] in (1, 2)
    assert audit["right_hit_count"] in (1, 2)
    assert np.isclose(np.min(arc[:, 1]), 0.0, atol=1e-12)


def test_default_dispatch_is_sse_v9_ssisv2_direct_mask(tmp_path: Path) -> None:
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
    assert manifest["mainline"] == "SSE-v9"
    assert "official SSISv2 associated shadow mask" in manifest["shadow_mask_contract"]
    assert "--direct-shadow-mask" in manifest["command"]
    assert manifest["feedforward_exclude_image_edge_shadow"] is True
    assert manifest["feedforward_predicted_floor_subtraction"] is False


def test_one_to_five_gate_and_fail_closed_contract() -> None:
    for count in range(1, 6):
        validate_vehicle_count(count)
    assert gate_confidence(.7, .8, .5, .75) == []
    assert publication_status([{"vehicle_index": 1, "status": "rejected"}])["status"] == "no_valid_sun_information"


def test_joint_coarse_seeds_are_exact_mean_of_independent_scores(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("score,azimuth_deg,elevation_deg\n-1,10,30\n-3,20,30\n")
    second.write_text("score,azimuth_deg,elevation_deg\n-3,10,30\n-0.5,20,30\n")
    assert combined_coarse_seeds([first, second], 2) == [(20.0, 30.0), (10.0, 30.0)]


def test_joint_coarse_reuse_detects_seed_sensitive_subsampling(tmp_path: Path) -> None:
    vehicle = tmp_path / "vehicle.npz"
    shadow = tmp_path / "shadow.npz"
    np.savez(vehicle, positions_world=np.array([[0., 0., .5], [1., 0., .5], [2., 0., 9.]]))
    np.savez(shadow, plane_z_ax_by_c=np.array([0., 0., 0.]))
    assert geometry_uses_subsampling(vehicle, shadow, 1) is True
    assert geometry_uses_subsampling(vehicle, shadow, 2) is False


def test_fit_processes_bound_native_threads_and_defer_independent_top25(tmp_path: Path) -> None:
    args = argparse.Namespace(
        raster_size=900, max_vehicle_points=100000, coarse_az_step=5,
        coarse_elev_min=20, coarse_elev_max=70, local_radius_deg=5,
        local_basins=3, visibility_mode="camera_cone_middle",
        distance_percentile=.95, top_candidates=25, structure_aware=False,
        candidate_workers=8,
    )
    command = fit_command([Path("vehicle.npz")], [Path("shadow.npz")],
                          [Path("mask.png")], tmp_path, args, top_candidates=0,
                          refinement_seeds=[(15.0, 40.0)])
    assert command[command.index("--top-candidates") + 1] == "0"
    assert command[-3:] == ["--refinement-seed", "15.0", "40.0"]
    assert command[command.index("--candidate-workers") + 1] == "8"
    environment = fit_subprocess_environment()
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"


def test_mainline_exposes_parallel_fit_defaults(tmp_path: Path) -> None:
    files = []
    for name in ("vehicle.npz", "shadow.npz", "mask.png"):
        path = tmp_path / name
        path.write_bytes(b"placeholder")
        files.append(path)
    output = tmp_path / "parallel"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([
        sys.executable, "-m", "pbr_vehicle_sun.run_sse_mainline",
        "--geometry-source", "feedforward_infinidepth", "--aggregation-mode", "vehicles",
        "--vehicle-geometry", str(files[0]), "--shadow-geometry", str(files[1]),
        "--shadow-mask", str(files[2]), "--output-dir", str(output), "--dry-run",
    ], check=True, capture_output=True, text=True, env=env)
    command = json.loads((output / "sse_mainline_branch_manifest.json").read_text())["command"]
    assert command[command.index("--candidate-workers") + 1] == "8"
    assert command[command.index("--fit-workers") + 1] == "0"
    assert command[command.index("--per-vehicle-top-candidates") + 1] == "0"


def test_combined_coarse_seed_tie_order_matches_original_grid_order(tmp_path: Path) -> None:
    files = []
    for index in range(2):
        path = tmp_path / f"scores_{index}.csv"
        path.write_text("score,azimuth_deg,elevation_deg\n-1,0,20\n-1,5,20\n-2,0,25\n")
        files.append(path)
    assert combined_coarse_seeds(files, 2) == [(0.0, 20.0), (5.0, 20.0)]


def test_scene_orchestrator_uses_official_ssisv2_association() -> None:
    args = argparse.Namespace(
        ssis_python=Path("python"), ssis_root=Path("external/ssis"),
        ssis_weights=Path("weights/model.pth"), ssis_device="cuda:0",
        ssis_confidence_threshold=.1, minimum_ssis_object_iou=.05,
    )
    command = ssisv2_command(args, Path("image.jpg"), Path("vehicle.png"), Path("out"))
    assert command[1].endswith("run_ssisv2_vehicle_shadow_association.py")
    assert "--minimum-object-iou" in command


def test_scene_orchestrator_exposes_exact_sam2_batching() -> None:
    source = (ROOT / "src/pbr_vehicle_sun/run_sse_scene_global_top3.py").read_text()
    assert '--sam2-image-batch-size' in source
    assert '"--image-batch-size", str(args.sam2_image_batch_size)' in source


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


def test_complete_run_cache_restores_paths_without_mutating_entry(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = {
        "status": "ok",
        "result_path": str((source / "round-1/scene_result.json").resolve()),
    }
    (source / "two_round_result.json").write_text(json.dumps(payload) + "\n")
    binary = source / "evidence.jpg"
    binary.write_bytes(b"evidence")
    entry = tmp_path / "cache" / "complete-runs" / "abc"
    store_cached_run(source, entry, "abc")
    cached_text = (entry / "two_round_result.json").read_text()
    assert str(entry.resolve()) in cached_text
    destination = tmp_path / "restored"
    restored = restore_cached_run(entry, destination)
    assert restored is not None and restored["cache"]["hit"] is True
    assert str(destination.resolve()) in (destination / "two_round_result.json").read_text()
    assert (destination / "evidence.jpg").read_bytes() == b"evidence"
    assert (entry / "two_round_result.json").read_text() == cached_text


def test_cache_key_invalidates_on_nested_dataset_change(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    image = scene / "images/000_0.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"first")
    before = cache_key(["--data-root", str(scene), "--scene", "example"])
    image.write_bytes(b"changed-size")
    after = cache_key(["--data-root", str(scene), "--scene", "example"])
    assert before != after


def test_package_has_no_retired_detector_or_source_mask_cleanup() -> None:
    source_dir = ROOT / "src/pbr_vehicle_sun"
    retired = "mt" + "mt"
    for path in source_dir.glob("*.py"):
        text = path.read_text().lower()
        assert retired not in text
        assert "postprocess_shadow_mask" not in text
    assert not (source_dir / ("test_" + retired + "_shadow_detector.py")).exists()
