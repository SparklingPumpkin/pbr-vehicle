from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pbr_vehicle_sun.fit_sun_camera_visible_arc import maximal_camera_cone_middle_arc


def test_maximal_camera_cone_selects_near_boundary() -> None:
    contour = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    arc, audit = maximal_camera_cone_middle_arc(contour, np.array([2.0, -5.0]))
    assert audit["hit_contract_1_or_2_each"] is True
    assert audit["left_hit_count"] in (1, 2)
    assert audit["right_hit_count"] in (1, 2)
    assert np.min(arc[:, 1]) == 0.0


def test_default_dispatch_contract(tmp_path: Path) -> None:
    files = []
    for name in ("vehicle.npz", "shadow.npz", "mask.png"):
        path = tmp_path / name
        path.write_bytes(b"placeholder")
        files.append(path)
    output = tmp_path / "out"
    command = [
        sys.executable, "-m", "pbr_vehicle_sun.run_sse_mainline",
        "--geometry-source", "feedforward_infinidepth",
        "--vehicle-geometry", str(files[0]), "--shadow-geometry", str(files[1]),
        "--shadow-mask", str(files[2]), "--output-dir", str(output), "--dry-run",
    ]
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    manifest = json.loads((output / "sse_mainline_branch_manifest.json").read_text())
    assert manifest["mainline"] == "SSE-v6"
    assert manifest["feedforward_visibility_mode"] == "camera_cone_middle"
    assert manifest["feedforward_exclude_image_edge_shadow"] is True
    assert manifest["feedforward_observed_floor_subtraction"] is False
    assert manifest["feedforward_predicted_floor_subtraction"] is False
    assert "--exclude-image-edge-components" in manifest["command"]
    assert "--no-observed-floor-subtraction" in manifest["command"]
    assert "--no-predicted-floor-subtraction" in manifest["command"]
