from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pbr_vehicle_sun.infer_infinidepth_depth_and_gaussians import (
    quaternion_to_rotation,
    save_dense_depth_outputs,
)


def test_joint_adapter_cli_supports_multiple_vehicle_masks() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/pbr_vehicle_sun/infer_infinidepth_depth_and_gaussians.py"
    ).read_text()
    assert '"--vehicle-mask", type=Path, action="append"' in source
    assert '"--vehicle-output-dir", type=Path, action="append"' in source
    assert "direct_vehicle_masks" in source
    assert "inference_for_gs" in source  # retained full-PLY compatibility branch


def test_joint_adapter_preserves_legacy_dense_depth_artifact_layout(tmp_path: Path) -> None:
    depth_model = np.linspace(1.0, 20.0, 12, dtype=np.float32).reshape(3, 4)
    manifest = save_dense_depth_outputs(
        depth_model,
        (6, 8),
        tmp_path,
        {
            "model": "InfiniDepth",
            "uses_lidar": False,
            "shared_forward_with_gaussian_export": True,
        },
    )

    saved_model = np.load(tmp_path / "predicted_depth_model.npy")
    saved_original = np.load(tmp_path / "predicted_depth_original.npy")
    saved_manifest = json.loads((tmp_path / "depth_manifest.json").read_text())
    heatmap = cv2.imread(str(tmp_path / "predicted_depth_heatmap.png"))

    np.testing.assert_array_equal(saved_model, depth_model)
    assert saved_original.shape == (6, 8)
    assert saved_original.dtype == np.float32
    assert heatmap.shape[:2] == (6, 8)
    assert manifest["shared_forward_with_gaussian_export"] is True
    assert saved_manifest["outputs"]["original_depth"] == "predicted_depth_original.npy"
    assert saved_manifest["depth_semantics"].startswith("metric camera-z depth")


def test_joint_adapter_rejects_invalid_depth_shape(tmp_path: Path) -> None:
    with np.testing.assert_raises_regex(ValueError, "dense depth must be HxW"):
        save_dense_depth_outputs(
            np.ones((1, 3, 4), dtype=np.float32), (6, 8), tmp_path, {}
        )


def test_quaternion_to_rotation_identity_and_quarter_turn() -> None:
    rotations = quaternion_to_rotation(np.array([
        [1.0, 0.0, 0.0, 0.0],
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
    ]))
    np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        rotations[1],
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-12,
    )
