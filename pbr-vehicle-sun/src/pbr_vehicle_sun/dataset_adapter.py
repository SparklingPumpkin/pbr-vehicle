from __future__ import annotations

from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def resolve_view_image(data_root: Path, timestep: int, camera: int) -> Path | None:
    stem = f"{int(timestep):03d}_{int(camera)}"
    images = Path(data_root) / "images"
    for suffix in IMAGE_SUFFIXES:
        candidate = images / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def discover_camera_ids(data_root: Path) -> list[int]:
    intrinsics = Path(data_root) / "intrinsics"
    cameras = []
    for path in intrinsics.glob("*.txt"):
        try:
            camera = int(path.stem)
        except ValueError:
            continue
        if (Path(data_root) / "extrinsics" / f"{camera}.txt").is_file():
            cameras.append(camera)
    return sorted(set(cameras))