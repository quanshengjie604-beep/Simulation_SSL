"""Torch-native vector math helpers."""

from __future__ import annotations

import torch


def vec3_tensor(value, *, name: str) -> torch.Tensor:
    """Coerce to a CPU float32 tensor of shape (3,)."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    else:
        tensor = torch.tensor(tuple(float(component) for component in value), dtype=torch.float32)
    if tensor.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")
    return tensor


def optional_vec3_tensor(value, *, name: str) -> torch.Tensor | None:
    if value is None:
        return None
    return vec3_tensor(value, name=name)


def scalar_tensor(value, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).reshape(())
    return torch.tensor(float(value), dtype=torch.float32)


def normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)
