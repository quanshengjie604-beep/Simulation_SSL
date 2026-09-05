"""Shared geometry base classes and tensor helpers."""

from __future__ import annotations

import numpy as np
import torch

from ..math import (
    quat_from_euler,
    quat_identity,
    quat_to_rotation_matrix,
    quat_to_rotation_matrix_np,
)


def _as_position(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor([float(v) for v in value], dtype=torch.float32, device=device)


def _as_rotation(value, *, device=None) -> torch.Tensor:
    """Accept quaternion (4,), Euler tuple (3,), or None -> identity."""
    if value is None:
        return quat_identity(device=device)
    if isinstance(value, torch.Tensor):
        if value.shape == (4,):
            return value.to(device=device, dtype=torch.float32)
        if value.shape == (3,):
            return quat_from_euler(value[0], value[1], value[2], device=device)
        raise ValueError(f"rotation tensor must be shape (4,) quaternion or (3,) Euler, got {value.shape}")
    seq = list(value)
    if len(seq) == 4:
        return torch.tensor([float(v) for v in seq], dtype=torch.float32, device=device)
    if len(seq) == 3:
        return quat_from_euler(float(seq[0]), float(seq[1]), float(seq[2]), device=device)
    raise ValueError(f"rotation must have 3 (Euler) or 4 (quaternion) elements, got {len(seq)}")


def _as_scalar(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).reshape(())
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _as_vec3(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    if isinstance(value, (int, float)):
        scalar = float(value)
        return torch.tensor([scalar, scalar, scalar], dtype=torch.float32, device=device)
    return torch.tensor([float(v) for v in value], dtype=torch.float32, device=device)


def _rotate_coords(xx, yy, zz, position: torch.Tensor, rotation: torch.Tensor):
    """Translate + inverse-rotate 3D grid coordinates into local frame."""
    dx = xx - position[0]
    dy = yy - position[1]
    dz = zz - position[2]
    rotation_matrix = quat_to_rotation_matrix(rotation)
    rotation_inverse = rotation_matrix.T
    dx_rot = rotation_inverse[0, 0] * dx + rotation_inverse[0, 1] * dy + rotation_inverse[0, 2] * dz
    dy_rot = rotation_inverse[1, 0] * dx + rotation_inverse[1, 1] * dy + rotation_inverse[1, 2] * dz
    dz_rot = rotation_inverse[2, 0] * dx + rotation_inverse[2, 1] * dy + rotation_inverse[2, 2] * dz
    return dx_rot, dy_rot, dz_rot


def _axial_split(dx, dy, dz, axis: str):
    """Return (axial, radial_a, radial_b) for a named axis."""
    if axis == "z":
        return dz, dx, dy
    if axis == "y":
        return dy, dx, dz
    return dx, dy, dz


def _apply_rotation_np(vertices: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply a (3,3) rotation matrix to (N,3) vertices."""
    return (vertices @ rotation_matrix.T).astype(np.float32)


def _constant_tensor(data, *, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, device=device)


def _remap_axis_torch(vertices: torch.Tensor, axis: str) -> torch.Tensor:
    """Remap local z-up coordinates to the requested axis."""
    if axis == "y":
        return vertices[:, [0, 2, 1]]
    if axis == "x":
        return vertices[:, [2, 1, 0]]
    return vertices


def _coordinate_spacing(coord: torch.Tensor, dim: int) -> torch.Tensor | None:
    if not isinstance(coord, torch.Tensor) or coord.ndim <= dim or coord.shape[dim] <= 1:
        return None
    delta = torch.diff(coord, dim=dim).abs()
    positive = delta[delta > 0]
    if positive.numel() == 0:
        return None
    return positive.min()


def _default_beta_from_coords(xx, yy, zz, *, reference: torch.Tensor) -> torch.Tensor:
    spacings = [
        spacing
        for spacing in (
            _coordinate_spacing(xx, 0),
            _coordinate_spacing(yy, 1),
            _coordinate_spacing(zz, 2),
        )
        if spacing is not None
    ]
    if not spacings:
        return reference.new_tensor(1.0e-3)

    spacing = spacings[0].to(device=reference.device, dtype=reference.dtype)
    for candidate in spacings[1:]:
        spacing = torch.minimum(spacing, candidate.to(device=reference.device, dtype=reference.dtype))
    return spacing * 0.05


def _resolve_beta(beta, xx, yy, zz, *, reference: torch.Tensor) -> torch.Tensor:
    if beta is None:
        resolved = _default_beta_from_coords(xx, yy, zz, reference=reference)
    else:
        resolved = torch.as_tensor(beta, device=reference.device, dtype=reference.dtype)
    return torch.clamp(resolved, min=torch.finfo(reference.dtype).eps)


def occupancy_from_signed_distance(signed_distance: torch.Tensor, *, xx, yy, zz, offset=0.0, beta=None) -> torch.Tensor:
    offset_tensor = torch.as_tensor(offset, device=signed_distance.device, dtype=signed_distance.dtype)
    beta_tensor = _resolve_beta(beta, xx, yy, zz, reference=signed_distance)
    occupancy = 0.5 * (1.0 - torch.tanh((signed_distance - offset_tensor) / beta_tensor))
    return occupancy.clamp(0.0, 1.0)


class GeometryBase:
    """Base class for all geometry primitives."""

    kind: str = "base"

    def __init__(self, position=(0, 0, 0), rotation=None, *, device=None):
        self.position: torch.Tensor = _as_position(position, device=device)
        self.rotation: torch.Tensor = _as_rotation(rotation, device=device)

    @property
    def device(self):
        return self.position.device

    def _local_coords(self, xx, yy, zz):
        return _rotate_coords(xx, yy, zz, self.position, self.rotation)

    def _rotation_matrix_np(self) -> np.ndarray:
        return quat_to_rotation_matrix_np(self.rotation)

    @staticmethod
    def _validate_axis(axis: str) -> str:
        axis_name = str(axis).lower()
        if axis_name not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'.")
        return axis_name

    def _apply_axis_transform(self, vertices: torch.Tensor, axis: str) -> torch.Tensor:
        return _remap_axis_torch(vertices, axis)

    def _transform_mesh_verts(self, vertices: torch.Tensor) -> torch.Tensor:
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.as_tensor(vertices, dtype=torch.float32, device=self.device)
        rotation_matrix = quat_to_rotation_matrix(self.rotation.to(device=vertices.device, dtype=vertices.dtype))
        position = self.position.to(device=vertices.device, dtype=vertices.dtype)
        return vertices @ rotation_matrix.T + position

    def with_material(self, material, **kwargs):
        from ..material import Structure

        return Structure(geometry=self, material=material, **kwargs)

    def signed_distance(self, xx, yy, zz):
        raise NotImplementedError

    def to_mask(self, xx, yy, zz, offset=0.0, beta=None):
        signed_distance = self.signed_distance(xx, yy, zz)
        return occupancy_from_signed_distance(
            signed_distance,
            xx=xx,
            yy=yy,
            zz=zz,
            offset=offset,
            beta=beta,
        )

    def to_mesh(self, segments=16):
        raise NotImplementedError

    def __repr__(self):
        fields = ", ".join(f"{key}={value}" for key, value in self.__dict__.items() if not key.startswith("_"))
        return f"{type(self).__name__}({fields})"
