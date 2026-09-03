import json
import os
from pathlib import Path

import pytest
import numpy as np

from pbr_vehicle_standalone import asset_io
from pbr_vehicle_standalone.asset_io import load_vehicle_asset
from pbr_vehicle_standalone.shading import shade_vehicle
from pbr_vehicle_standalone.types import GaussianLayer, LightingState, MaterialState


ASSET_ENV = "PBR_VEHICLE_TEST_ASSET"
ASSET = Path(os.environ[ASSET_ENV]) if os.environ.get(ASSET_ENV) else None


def test_current_delivery_asset_contract():
    if ASSET is None or not ASSET.is_dir():
        pytest.skip(f"set {ASSET_ENV} to run the real-asset test")
    asset = load_vehicle_asset(ASSET)
    assert asset.original_to_proxy.shape == (len(asset.original.centers),)
    assert asset.original_to_proxy.min() >= 0
    assert asset.original_to_proxy.max() < len(asset.proxy.centers)
    assert asset.projection["runtime"] == "receiver_space_mask_v1"


def test_panel_loads_single_ply_contract(tmp_path, monkeypatch):
    root = tmp_path / "single"
    (root / "configs").mkdir(parents=True)
    (root / "pbr_single.ply").touch()
    (root / "configs" / "config_single.json").write_text(json.dumps({
        "asset_contract": "pbr-vehicle-single-ply-v1", "asset_id": "single",
        "files": {"pbr": "pbr_single.ply", "config": "configs/config_single.json"},
        "material": {"albedo_rgb": [0.82, 0.82, 0.82], "roughness": 0.32, "metallic": 0.02},
        "projection": {"runtime": "receiver_space_mask_v1"},
    }))
    layer = GaussianLayer(
        centers=np.zeros((2, 3), np.float32), covariances=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        colors=np.full((2, 3), 0.5, np.float32), opacities=np.ones((2, 1), np.float32),
        normals=np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (2, 1)), source_indices=np.array([0, 1]),
    )
    monkeypatch.setattr(asset_io, "load_gaussian_layer", lambda path, center: layer)
    asset = load_vehicle_asset(root)
    assert asset.original is asset.proxy
    np.testing.assert_array_equal(asset.mapping_indices[:, 0], [0, 1])
    np.testing.assert_allclose(asset.proxy.albedo, 0.82)


def test_panel_loads_normalized_knn_mapping_contract(tmp_path, monkeypatch):
    root = tmp_path / "knn_asset"
    (root / "configs").mkdir(parents=True)
    for name in ("original.ply", "proxy.ply", "mapping.npz"):
        (root / name).touch()
    (root / "configs" / "config_knn_asset.json").write_text(json.dumps({
        "asset_contract": "pbr-vehicle-config-folder-v4",
        "asset_id": "knn_asset",
        "files": {"original": "original.ply", "proxy": "proxy.ply", "mapping": "mapping.npz", "config": "configs/config_knn_asset.json"},
        "projection": {},
    }))
    np.savez(root / "mapping.npz", raw_to_proxy_knn_idx=np.array([[0, 1], [1, 0]]), raw_to_proxy_knn_weight=np.array([[3.0, 1.0], [3.0, 1.0]], np.float32))
    original = GaussianLayer(
        centers=np.zeros((2, 3), np.float32), covariances=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        colors=np.full((2, 3), 0.5, np.float32), opacities=np.ones((2, 1), np.float32), source_indices=np.array([0, 1]),
    )
    proxy = GaussianLayer(
        centers=np.zeros((2, 3), np.float32), covariances=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        colors=np.full((2, 3), 0.5, np.float32), opacities=np.ones((2, 1), np.float32),
        normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], np.float32),
        albedo=np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], np.float32),
        roughness=np.full((2, 1), 0.4, np.float32), metallic=np.zeros((2, 1), np.float32),
    )
    monkeypatch.setattr(asset_io, "load_gaussian_layer", lambda path, center: proxy if Path(path).name == "proxy.ply" else original)
    asset = load_vehicle_asset(root)
    np.testing.assert_allclose(asset.mapping_weights, [[0.75, 0.25], [0.75, 0.25]])
    _, colors = shade_vehicle(
        asset,
        MaterialState(use_asset_material=False, ambient_fill=0.0),
        LightingState(sun_enabled=True, sun_intensity=0.4, sun_azimuth_deg=0.0, sun_elevation_deg=90.0),
        mode="Relight Original",
    )
    assert colors.shape == (2, 3)
    assert not np.allclose(colors[0], colors[1])
