import json
import struct

import numpy as np

from pbr_vehicle_standalone.projection import build_projection_masks
from pbr_vehicle_standalone.rendering import rgba_plane_glb
from pbr_vehicle_standalone.types import GaussianLayer


def _proxy():
    x, y, z = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-1, 1, 6), np.linspace(0, 1.5, 4))
    centers = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)
    covariances = np.tile(np.eye(3, dtype=np.float32)[None] * 0.0025, (len(centers), 1, 1))
    return GaussianLayer(centers, covariances, np.ones((len(centers), 3), np.float32), np.ones((len(centers), 1), np.float32))


def _config():
    return {
        "runtime": "receiver_space_mask_v1",
        "shape_model": "procedural-mask-contact-silhouette-v4",
        "composition": "receiver_alpha_blend_to_grayscale_v1",
        "contact": {"length_scale": 0.9, "width_scale": 0.85, "offset_xy_m": [0, 0], "corner_radius_m": 0.2, "edge_softness_m": 0.05, "opacity": 0.8, "brightness": 0.0},
        "extension": {"opacity": 0.3, "opacity_distance_decay": 0.5, "opacity_distance_exponent": 1.2, "brightness": 0.2, "brightness_distance_to_white": 0.5, "brightness_distance_exponent": 1.2, "edge_softness_m": 0.08, "edge_softness_distance_growth_m": 0.2, "edge_softness_distance_exponent": 1.1, "distance_scale_m": 1.2},
        "anchor": {"bottom_surface_percentile": 1.0, "surface_sigma": 1.0, "z_offset_m": 0.0},
    }


def test_contact_is_sun_invariant_and_extension_changes():
    proxy, config = _proxy(), _config()
    first = build_projection_masks(proxy, np.array([0.5, 0.2, 0.84]), config)
    second = build_projection_masks(proxy, np.array([-0.3, 0.7, 0.64]), config)
    assert np.array_equal(first["contact"]["rgba"], second["contact"]["rgba"])
    assert np.array_equal(first["contact"]["center_xy"], second["contact"]["center_xy"])
    assert np.array_equal(first["contact"]["size_xy"], second["contact"]["size_xy"])
    assert not np.array_equal(first["extension"]["rgba"], second["extension"]["rgba"])


def test_glb_uses_blend_material():
    glb = rgba_plane_glb(np.full((8, 8, 4), 255, np.uint8), 2.0, 1.0)
    assert glb[:4] == b"glTF"
    json_length, chunk_type = struct.unpack_from("<II", glb, 12)
    assert chunk_type == 0x4E4F534A
    document = json.loads(glb[20:20 + json_length].decode("utf-8"))
    assert document["materials"][0]["alphaMode"] == "BLEND"
