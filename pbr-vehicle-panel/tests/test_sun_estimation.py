import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pbr_vehicle_standalone.sun_estimation import (
    SunEstimationError,
    build_sun_estimator_command,
    read_sun_estimate,
    run_sun_estimator,
)


def test_command_uses_public_scene_cli_and_portable_fields(tmp_path):
    command = build_sun_estimator_command(
        {
            "yolo_weights": "models/yolo.pt",
            "sam2_root": "third_party/sam2",
            "sam2_checkpoint": "models/sam2.pt",
            "ssis_root": "third_party/ssis",
            "ssis_weights": "models/ssis.pth",
            "infinidepth_root": "third_party/InfiniDepth",
            "cameras": [0, 2],
        },
        scene="004",
        data_root="datasets/004",
        output_dir=tmp_path,
    )
    assert command[0] == "pbr-vehicle-sun-scene"
    assert command[command.index("--scene") + 1] == "004"
    assert command[command.index("--output-dir") + 1] == str(tmp_path)
    cameras_at = command.index("--cameras")
    assert command[cameras_at + 1:cameras_at + 3] == ["0", "2"]


def test_reads_joint_angle_and_lifts_road_plane_to_world(tmp_path):
    payload = {
        "status": "ok",
        "aggregate_result": {
            "status": "ok",
            "weighted_angle": {"weighted_azimuth_deg": 40.0, "weighted_elevation_deg": 30.0},
            "joint_fit": {"azimuth_deg": 90.0, "elevation_deg": 45.0},
        },
    }
    (tmp_path / "scene_result.json").write_text(json.dumps(payload), encoding="utf-8")
    contour = tmp_path / "confidence_gated_joint_fit" / "joint_fit" / "frame0_best_contours.npz"
    contour.parent.mkdir(parents=True)
    np.savez(contour, plane_e1=np.array([1.0, 0.0, 0.0]), plane_e2=np.array([0.0, 1.0, 0.0]))

    result = read_sun_estimate(tmp_path)

    assert result["azimuth_deg"] == pytest.approx(90.0)
    assert result["elevation_deg"] == pytest.approx(45.0)
    assert result["world_direction"] == pytest.approx([0.0, 2 ** -0.5, 2 ** -0.5])


def test_run_uses_temporary_output_and_rejects_no_angle(tmp_path):
    script = tmp_path / "fake_estimator.py"
    script.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--output-dir', required=True); a=p.parse_args()\n"
        "o=Path(a.output_dir); o.mkdir(parents=True, exist_ok=True)\n"
        "(o/'scene_result.json').write_text(json.dumps({'status':'ok','aggregate_result':{'joint_fit':{'azimuth_deg':12,'elevation_deg':34}}}))\n",
        encoding="utf-8",
    )
    result = run_sun_estimator(command_override=[sys.executable, str(script)])
    assert result["azimuth_deg"] == pytest.approx(12.0)
    assert Path(result["output_dir"]).is_dir()

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "scene_result.json").write_text('{"status":"no_valid_sun_information"}', encoding="utf-8")
    with pytest.raises(SunEstimationError):
        read_sun_estimate(empty)


def test_two_round_result_uses_only_published_round(tmp_path):
    (tmp_path / "two_round_result.json").write_text(json.dumps({
        "status": "ok", "published_round": 2, "azimuth_deg": 203.0, "elevation_deg": 70.0,
    }), encoding="utf-8")
    first = tmp_path / "round-1/confidence_gated_joint_fit/joint_fit"
    second = tmp_path / "round-2/confidence_gated_joint_fit/joint_fit"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    np.savez(first / "frame0_best_contours.npz", plane_e1=[0, 1, 0], plane_e2=[0, 0, 1])
    np.savez(second / "frame0_best_contours.npz", plane_e1=[1, 0, 0], plane_e2=[0, 1, 0])

    result = read_sun_estimate(tmp_path)

    assert result["published_round"] == 2
    assert result["azimuth_deg"] == pytest.approx(203.0)
    assert result["elevation_deg"] == pytest.approx(70.0)


def test_two_round_rejection_never_falls_back_to_round_candidate(tmp_path):
    (tmp_path / "two_round_result.json").write_text(json.dumps({
        "status": "no_valid_sun_information", "published_round": None,
    }), encoding="utf-8")
    first = tmp_path / "round-1"
    first.mkdir()
    (first / "scene_result.json").write_text(json.dumps({
        "status": "ok", "aggregate_result": {"joint_fit": {"azimuth_deg": 12.0, "elevation_deg": 34.0}},
    }), encoding="utf-8")

    with pytest.raises(SunEstimationError, match="no_valid_sun_information"):
        read_sun_estimate(tmp_path)
