"""4x4 transform helpers and local/world conversions for scene geometry."""

from __future__ import annotations

import torch

from witwin.core import GeometryBase, Mesh
from witwin.core.math import quat_to_rotation_matrix


def identity_transform(*, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype)


def translation_transform(translation: torch.Tensor, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    transform = identity_transform(device=device, dtype=dtype)
    transform[:3, 3] = translation
    return transform


def axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-12)
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_minus_c = 1.0 - c
    row0 = torch.stack((c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s))
    row1 = torch.stack((y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s))
    row2 = torch.stack((z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c))
    return torch.stack((row0, row1, row2), dim=0)


def rotation_about_origin_transform(
    origin: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    rotation = axis_angle_rotation(axis.to(device=device, dtype=dtype), angle.to(device=device, dtype=dtype))
    translate_to = translation_transform(origin.to(device=device, dtype=dtype), device=device, dtype=dtype)
    translate_back = translation_transform(-origin.to(device=device, dtype=dtype), device=device, dtype=dtype)
    transform = identity_transform(device=device, dtype=dtype)
    transform[:3, :3] = rotation
    return translate_to @ transform @ translate_back


def apply_transform_to_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.transpose(0, 1) + translation


def apply_transform_to_vectors(vectors: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[:3, :3]
    return vectors @ rotation.transpose(0, 1)


def geometry_local_to_world_points(
    geometry: GeometryBase,
    points: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    points = points.to(device=device, dtype=dtype)
    rotation = quat_to_rotation_matrix(geometry.rotation.to(device=device, dtype=dtype))
    position = geometry.position.to(device=device, dtype=dtype)
    if isinstance(geometry, Mesh):
        scale = geometry.scale.to(device=device, dtype=dtype)
        points = points * scale
    return points @ rotation.transpose(0, 1) + position


def geometry_local_to_world_vectors(
    geometry: GeometryBase,
    vectors: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    vectors = vectors.to(device=device, dtype=dtype)
    rotation = quat_to_rotation_matrix(geometry.rotation.to(device=device, dtype=dtype))
    return vectors @ rotation.transpose(0, 1)
