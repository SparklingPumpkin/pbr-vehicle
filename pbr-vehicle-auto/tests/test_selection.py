import pytest

from pbr_vehicle_auto.fitter import grid_best_metric


@pytest.mark.parametrize(
    ("raw", "candidate", "expected_change"),
    [
        (0.10, 0.08, 20.0),
        (0.10, 0.10, 0.0),
        (0.10, 0.12, -20.0),
    ],
)
def test_grid_best_is_reported_even_when_it_does_not_beat_raw_dc(
    raw, candidate, expected_change
):
    metric = grid_best_metric(raw, candidate)
    assert metric["raw"] == pytest.approx(raw)
    assert metric["baked"] == pytest.approx(candidate)
    assert metric["improvement_percent"] == pytest.approx(expected_change)
