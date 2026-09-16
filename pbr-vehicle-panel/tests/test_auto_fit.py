import json
import sys
from pathlib import Path

import pytest

from pbr_vehicle_standalone.auto_fit import (
    VehicleAutoFitError,
    build_auto_fit_command,
    read_auto_fit_result,
    run_vehicle_auto_fit,
)


def _accepted_payload():
    return {
        "vehicle": {
            "ambient_fill": 0.6,
            "vehicle_lighting": {
                "sun_intensity": 1.5,
                "environment_temperature": 5800.0,
            },
        }
    }


def test_default_command_contains_complete_public_contract(tmp_path):
    paths = {
        "scene_ply": tmp_path / "scene.ply",
        "asset_dir": tmp_path / "asset",
        "template_config": tmp_path / "template.json",
        "output_dir": tmp_path / "output",
        "final_config": tmp_path / "final.json",
    }
    command = build_auto_fit_command({}, **paths)
    assert command[0] == "pbr-vehicle-auto-fit"
    for flag, key in (
        ("--scene-ply", "scene_ply"),
        ("--asset-dir", "asset_dir"),
        ("--template-config", "template_config"),
        ("--output-dir", "output_dir"),
        ("--final-config", "final_config"),
    ):
        assert command[command.index(flag) + 1] == str(paths[key])
    assert command[command.index("--max-seconds") + 1] == "480.0"
    assert command[command.index("--bins") + 1] == "512"


def test_command_placeholders_are_expanded_without_duplicate_flags(tmp_path):
    scene = tmp_path / "scene.ply"
    command = build_auto_fit_command(
        {
            "command": [
                "python", "fit.py", "--scene-ply", "{scene_ply}",
                "--asset-dir", "{asset_dir}", "--template-config", "{template_config}",
                "--output-dir", "{output_dir}", "--final-config", "{final_config}",
            ]
        },
        scene_ply=scene,
        asset_dir=tmp_path / "asset",
        template_config=tmp_path / "template.json",
        output_dir=tmp_path / "output",
        final_config=tmp_path / "final.json",
    )
    assert command[command.index("--scene-ply") + 1] == str(scene)
    assert command.count("--scene-ply") == 1


def test_result_reads_only_the_three_fitted_controls(tmp_path):
    final_config = tmp_path / "final.json"
    metrics = tmp_path / "metrics.json"
    final_config.write_text(json.dumps(_accepted_payload()), encoding="utf-8")
    metrics.write_text(
        json.dumps({"status": "candidate_only", "metric": {"improvement_percent": 4.25}}),
        encoding="utf-8",
    )
    result = read_auto_fit_result(final_config, metrics)
    assert result["sun_intensity"] == pytest.approx(1.5)
    assert result["ambient_fill"] == pytest.approx(0.6)
    assert result["environment_temperature_k"] == pytest.approx(5800.0)
    assert result["improvement_percent"] == pytest.approx(4.25)


def test_legacy_rejected_status_is_treated_as_unsupported_output(tmp_path):
    final_config = tmp_path / "final.json"
    metrics = tmp_path / "metrics.json"
    final_config.write_text(json.dumps(_accepted_payload()), encoding="utf-8")
    metrics.write_text(json.dumps({"status": "rejected_no_improvement"}), encoding="utf-8")
    with pytest.raises(VehicleAutoFitError, match="状态无效或不受支持"):
        read_auto_fit_result(final_config, metrics)


def test_grid_best_with_negative_improvement_still_returns_panel_values(tmp_path):
    final_config = tmp_path / "final.json"
    metrics = tmp_path / "metrics.json"
    final_config.write_text(json.dumps(_accepted_payload()), encoding="utf-8")
    metrics.write_text(
        json.dumps({"status": "candidate_only", "metric": {"improvement_percent": -1.5}}),
        encoding="utf-8",
    )
    result = read_auto_fit_result(final_config, metrics)
    assert result["sun_intensity"] == pytest.approx(1.5)
    assert result["improvement_percent"] == pytest.approx(-1.5)


def test_run_keeps_auto_artifacts_in_temporary_directory(tmp_path):
    scene = tmp_path / "scene.ply"
    scene.write_bytes(b"fake ply input")
    asset = tmp_path / "asset"
    asset.mkdir()
    script = tmp_path / "fake_auto.py"
    script.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--output-dir', required=True); "
        "p.add_argument('--final-config', required=True); a,_=p.parse_known_args()\n"
        "o=Path(a.output_dir); o.mkdir(parents=True, exist_ok=True)\n"
        "payload={'vehicle':{'ambient_fill':0.45,'vehicle_lighting':"
        "{'sun_intensity':2.0,'environment_temperature':6100.0}}}\n"
        "Path(a.final_config).write_text(json.dumps(payload))\n"
        "(o/'metrics.json').write_text(json.dumps({'status':'candidate_only',"
        "'metric':{'improvement_percent':3.0}}))\n",
        encoding="utf-8",
    )
    progress = []
    result = run_vehicle_auto_fit(
        scene_ply=scene,
        asset_dir=asset,
        template_payload={"vehicle": {}},
        command_override=[sys.executable, str(script)],
        timeout=10.0,
        progress_callback=lambda value, stage: progress.append((value, stage)),
    )
    output_root = Path(result["output_dir"])
    assert output_root.is_dir()
    assert (output_root / "current_panel_state.json").is_file()
    assert (output_root / "output" / "metrics.json").is_file()
    assert result["sun_intensity"] == pytest.approx(2.0)
    assert progress[0] == (0.01, "创建车辆参数识别任务")
    assert progress[-1] == (1.0, "车辆参数识别完成")
