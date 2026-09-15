"""Portable device selection for automatic fitting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeDevice:
    requested: str
    value: str
    uses_cuda: bool
    name: str

    @property
    def description(self) -> str:
        return f"{self.value} ({self.name})" if self.name else self.value


def resolve_compute_device(requested: str = "auto") -> ComputeDevice:
    value = str(requested or "auto").strip().lower()
    if value == "cpu":
        return ComputeDevice(value, "cpu", False, "NumPy CPU fallback")
    try:
        import torch
    except ImportError as exc:
        if value == "auto":
            return ComputeDevice(value, "cpu", False, "PyTorch unavailable")
        raise RuntimeError(f"Device {requested!r} requires PyTorch") from exc
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cpu":
        return ComputeDevice(str(requested), "cpu", False, "NumPy CPU fallback")
    if device.type != "cuda":
        raise ValueError(f"Unsupported compute device: {requested!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    index = int(device.index or 0)
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {value} is not visible; visible device count is {torch.cuda.device_count()}"
        )
    return ComputeDevice(str(requested), f"cuda:{index}", True, torch.cuda.get_device_name(index))
