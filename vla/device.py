"""Device selection for the inference entry points (XPU, then CUDA, then CPU).

An explicit device always wins over detection and is never swapped for another backend.
"""

from typing import Union

import torch

DeviceLike = Union[str, torch.device, None]


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in ("xpu", "cuda"):
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` to the detected default (``cuda:0`` on a CUDA host, as before), anything else verbatim."""
    if device is None or device == "":
        device_type = default_device_type()
        return torch.device("cpu") if device_type == "cpu" else torch.device(device_type, 0)
    return torch.device(device)
