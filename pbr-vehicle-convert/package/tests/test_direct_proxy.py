from collections import OrderedDict
import numpy as np
from pbr_vehicle_sdk import build_direct_white_2dgs
from pbr_vehicle_sdk.ply import read_ply, write_ply


def test_direct_proxy_preserves_centers_and_identity(tmp_path):
    n = 8
    p = OrderedDict()
    p["x"] = np.linspace(-1, 1, n, dtype=np.float32); p["y"] = np.zeros(n, np.float32); p["z"] = np.ones(n, np.float32)
    for i in range(3): p[f"f_dc_{i}"] = np.zeros(n, np.float32)
    p["opacity"] = np.ones(n, np.float32)
    p["scale_0"] = np.full(n, -3, np.float32); p["scale_1"] = np.full(n, -2, np.float32); p["scale_2"] = np.full(n, -1, np.float32)
    p["rot_0"] = np.ones(n, np.float32)
    for i in range(1, 4): p[f"rot_{i}"] = np.zeros(n, np.float32)
    source = write_ply(tmp_path / "source.ply", p)
    result = build_direct_white_2dgs(source, tmp_path / "out")
    proxy = read_ply(result.proxy_ply); mapping = np.load(result.mapping_npz)
    for axis in ("x", "y", "z"): np.testing.assert_array_equal(proxy[axis], p[axis])
    np.testing.assert_array_equal(mapping["raw_to_proxy_idx"], np.arange(n))
    assert result.gaussian_count == n
