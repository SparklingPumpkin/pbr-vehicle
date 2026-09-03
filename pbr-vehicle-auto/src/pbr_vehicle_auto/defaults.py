"""Normalize a full Viser state as the template for automatic fitting."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping


def normalize_default(saved_state: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve every saved state field while removing stale automatic output."""
    asset = saved_state.get("pbr_asset")
    vehicle = saved_state.get("vehicle")
    if not isinstance(asset, Mapping) or not isinstance(vehicle, Mapping):
        raise ValueError("saved state must contain pbr_asset and vehicle objects")
    payload = copy.deepcopy(dict(saved_state))
    pbr_asset = copy.deepcopy(dict(asset))
    pbr_asset.pop("auto_fit", None)
    active_vehicle = copy.deepcopy(dict(vehicle))
    vehicle_light = active_vehicle.get("vehicle_lighting")
    if not isinstance(vehicle_light, Mapping):
        raise ValueError("saved state vehicle.vehicle_lighting is required")
    active_light = dict(vehicle_light)
    pbr_light = pbr_asset.get("light", {})
    if "sun_color_rgb" not in active_light:
        active_light["sun_color_rgb"] = [float(value) for value in pbr_light.get("sun_color_rgb", pbr_light.get("color_rgb", [1.0, 1.0, 1.0]))]
    active_vehicle["vehicle_lighting"] = active_light
    payload["pbr_asset"] = pbr_asset
    payload["vehicle"] = active_vehicle
    payload.pop("auto_fit", None)
    payload["default_profile"] = {
        "source": "full saved_viser_state",
        "automatic_fit": False,
        "preserved": "vehicle pose, projection, material, scene state, and non-fitted lighting",
        "sun_color_rgb": "persisted in vehicle_lighting for fresh-process reproducibility",
    }
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    saved = json.loads(args.saved_state.read_text(encoding="utf-8"))
    payload = normalize_default(saved)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
