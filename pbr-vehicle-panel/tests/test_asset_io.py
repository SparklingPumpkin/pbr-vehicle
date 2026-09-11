import threading
import time

import numpy as np

from pbr_vehicle_standalone import asset_io


def test_checkpoint_loader_supports_numpy_two_pickle_names_concurrently(monkeypatch, tmp_path):
    entered = []

    class TorchStub:
        @staticmethod
        def load(path, **kwargs):
            entered.append(path)
            assert __import__("numpy._core", fromlist=["multiarray"]) is np.core
            time.sleep(0.05)
            assert __import__("numpy._core.multiarray", fromlist=["_reconstruct"]) is np.core.multiarray
            return path

    monkeypatch.delitem(asset_io.sys.modules, "numpy._core", raising=False)
    monkeypatch.delitem(asset_io.sys.modules, "numpy._core.multiarray", raising=False)
    monkeypatch.delitem(asset_io.sys.modules, "numpy._core.numeric", raising=False)
    results = []
    sources = [tmp_path / "first.pth", tmp_path / "second.pth"]
    threads = [
        threading.Thread(target=lambda source=source: results.append(asset_io._load_torch_checkpoint(TorchStub, source)))
        for source in sources
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(entered) == sorted(map(str, sources))
    assert sorted(results) == sorted(map(str, sources))
    assert "numpy._core" not in asset_io.sys.modules
    assert "numpy._core.multiarray" not in asset_io.sys.modules
    assert "numpy._core.numeric" not in asset_io.sys.modules