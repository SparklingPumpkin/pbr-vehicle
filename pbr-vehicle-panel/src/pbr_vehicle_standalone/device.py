"""Portable CUDA selection for the standalone delivery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeDevice:
    requested: str
    value: str
    backend: str
    name: str

    @property
    def uses_cuda(self) -> bool:
        return self.backend == "cuda"

    @property
    def description(self) -> str:
        return f"{self.value} ({self.name})" if self.name else self.value


def resolve_compute_device(requested: str = "auto") -> ComputeDevice:
    """Resolve a logical device without hard-coding a physical GPU index.

    ``cuda:0`` means the first GPU visible to this process, so
    ``CUDA_VISIBLE_DEVICES`` remains authoritative on multi-GPU machines.
    """
    value = str(requested or "auto").strip().lower()
    if value == "cpu":
        return ComputeDevice(value, "cpu", "numpy", "NumPy CPU fallback")
    try:
        import torch
    except ImportError as exc:
        if value == "auto":
            return ComputeDevice(value, "cpu", "numpy", "PyTorch unavailable")
        raise RuntimeError(f"Device {requested!r} requires PyTorch") from exc
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
        index = int(device.index or 0)
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {value} is not visible; visible device count is {torch.cuda.device_count()}"
            )
        return ComputeDevice(str(requested), f"cuda:{index}", "cuda", torch.cuda.get_device_name(index))
    if device.type != "cpu":
        raise ValueError(f"Unsupported compute device: {requested!r}; use auto, cpu, cuda, or cuda:N")
    return ComputeDevice(str(requested), "cpu", "numpy", "NumPy CPU fallback")
