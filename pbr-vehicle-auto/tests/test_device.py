from pbr_vehicle_auto.device import resolve_compute_device


def test_device_selection_is_portable():
    cpu = resolve_compute_device("cpu")
    assert cpu.value == "cpu"
    assert not cpu.uses_cuda
    automatic = resolve_compute_device("auto")
    assert automatic.value == "cpu" or automatic.value == "cuda:0"
