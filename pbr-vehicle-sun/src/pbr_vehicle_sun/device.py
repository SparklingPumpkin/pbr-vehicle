"""Normalize one portable device option for Torch, Detectron2 and Ultralytics."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    torch: str
    ultralytics: str
    uses_cuda: bool
    name: str

    @property
    def description(self) -> str:
        return f"{self.torch} ({self.name})" if self.name else self.torch


def _auto_cuda_available() -> tuple[bool, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip().lower() in {"", "-1", "none"}:
        return False, ""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, ""
    except ImportError:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return False, ""
        completed = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=False, capture_output=True, text=True,
        )
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return completed.returncode == 0 and bool(names), names[0] if names else ""


def resolve_device(requested: str = "auto") -> DeviceSelection:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        available, name = _auto_cuda_available()
        return DeviceSelection(value, "cuda:0", "0", True, name) if available else DeviceSelection(
            value, "cpu", "cpu", False, "CPU fallback"
        )
    if value == "cpu":
        return DeviceSelection(value, "cpu", "cpu", False, "CPU")
    if value == "cuda":
        value = "cuda:0"
    elif value.isdigit():
        value = f"cuda:{value}"
    if not value.startswith("cuda:") or not value.split(":", 1)[1].isdigit():
        raise ValueError(f"Unsupported device {requested!r}; use auto, cpu, cuda, cuda:N, or N")
    index = int(value.split(":", 1)[1])
    return DeviceSelection(str(requested), f"cuda:{index}", str(index), True, "explicit CUDA")
