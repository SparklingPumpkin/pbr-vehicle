import numpy as np

from pbr_vehicle_standalone.projection import build_projection_masks
from pbr_vehicle_standalone.rendering import projection_rgba_to_gaussians
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


def test_low_sun_extension_contains_vehicle_footprint_and_casts_opposite_sun():
    proxy, config = _proxy(), _config()
    azimuth = np.deg2rad(66.0)
    elevation = np.deg2rad(7.0)
    sun = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ], dtype=np.float32)
    result = build_projection_masks(proxy, sun, config)
    outline = result["cast_outline_xyz"][:, :2]

    # The swept hull must retain every source footprint point, so the cast
    # cannot start at a detached far-end silhouette.
    from scipy.spatial import Delaunay
    hull = Delaunay(outline)
    assert np.all(hull.find_simplex(proxy.centers[:, :2]) >= 0)

    # Its farthest displacement is opposite the horizontal sun direction.
    source_center = proxy.centers[:, :2].mean(axis=0)
    farthest = outline[np.argmin((outline - source_center) @ sun[:2])]
    assert float((farthest - source_center) @ sun[:2]) < -1.0


def test_projection_mask_converts_to_bounded_planar_gaussians():
    rgba = np.zeros((100, 200, 4), dtype=np.uint8)
    rgba[10:90, 20:180, :3] = 64
    rgba[10:90, 20:180, 3] = 128

    centers, covariances, colors, opacities = projection_rgba_to_gaussians(
        rgba,
        center_xy=np.array([2.0, -1.0], dtype=np.float32),
        size_xy=np.array([4.0, 2.0], dtype=np.float32),
        z=0.03,
        max_splats=2_000,
    )

    assert 0 < len(centers) <= 2_000
    assert covariances.shape == (len(centers), 3, 3)
    assert colors.shape == (len(centers), 3)
    assert opacities.shape == (len(centers), 1)
    assert np.allclose(centers[:, 2], 0.03)
    assert centers[:, 0].min() >= 0.0 and centers[:, 0].max() <= 4.0
    assert centers[:, 1].min() >= -2.0 and centers[:, 1].max() <= 0.0
    assert np.allclose(colors, 64.0 / 255.0)
    assert np.allclose(opacities, 128.0 / 255.0)
    assert np.all(np.diagonal(covariances, axis1=1, axis2=2) > 0.0)


def test_projection_rejects_nonfinite_sun_direction():
    with np.errstate(invalid="ignore"):
        with np.testing.assert_raises_regex(ValueError, "finite sun direction"):
            build_projection_masks(_proxy(), np.array([np.nan, 0.0, 1.0]), _config())
