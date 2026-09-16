import json
from pathlib import Path

import numpy as np

from pbr_vehicle_sun.audit_argoverse_single_vehicle_views import load_annotations
from pbr_vehicle_sun.dataset_adapter import discover_camera_ids, resolve_view_image


def test_resolve_view_image_supports_png_and_jpeg(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    png = images / "000_0.png"
    png.touch()
    jpg = images / "000_1.jpg"
    jpg.touch()

    assert resolve_view_image(tmp_path, 0, 0) == png
    assert resolve_view_image(tmp_path, 0, 1) == jpg
    assert resolve_view_image(tmp_path, 0, 2) is None


def test_discover_camera_ids_uses_matching_intrinsics_and_extrinsics(tmp_path):
    (tmp_path / "intrinsics").mkdir()
    (tmp_path / "extrinsics").mkdir()
    for camera in (0, 2, 8):
        (tmp_path / "intrinsics" / f"{camera}.txt").touch()
    for camera in (0, 8):
        (tmp_path / "extrinsics" / f"{camera}.txt").touch()

    assert discover_camera_ids(tmp_path) == [0, 8]


def test_ncore_vehicle_class_is_accepted(tmp_path):
    (tmp_path / "instances").mkdir()
    (tmp_path / "ego_pose").mkdir()
    np.savetxt(tmp_path / "ego_pose" / "000.txt", np.eye(4))
    payload = {
        "0": {
            "id": "vehicle-1",
            "class_name": "vehicle",
            "frame_annotations": {
                "frame_idx": [0],
                "obj_to_world": [np.eye(4).tolist()],
                "box_size": [[4.0, 2.0, 1.5]],
            },
        },
    }
    (tmp_path / "instances" / "instances_info.json").write_text(json.dumps(payload))

    annotations, present = load_annotations(tmp_path)

    assert list(annotations) == ["0"]
    assert present[0] == {"0"}