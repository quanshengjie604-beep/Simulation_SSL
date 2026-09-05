"""Tensor / device helpers shared across radar."""

from __future__ import annotations

import numpy as np
import torch


def real_dtype(value: torch.Tensor) -> torch.dtype:
    if value.dtype in {torch.float64, torch.complex128}:
        return torch.float64
    return torch.float32


def to_tensor3(value, *, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.shape != (3,):
        raise ValueError("value must have shape (3,).")
    return tensor


def to_vertex_tensor(value, *, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def to_faces_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    faces = np.asarray(value, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3).")
    return np.ascontiguousarray(faces)


def resolve_scene_device(device: str | None) -> str:
    requested = "cuda" if device is None else device
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Radar scenes default to CUDA, but torch.cuda.is_available() is False. "
            "Pass device='cpu' only for scene construction or non-rendering workflows."
        )
    return str(resolved)
