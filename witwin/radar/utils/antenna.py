"""Antenna-pattern math helpers."""

from __future__ import annotations

import math

import torch


DEFAULT_DIPOLE_ANGLES_DEG = tuple(float(angle) for angle in range(-90, 91))


def half_wave_dipole_power_cut(angle_deg: float) -> float:
    """Normalized half-wave dipole power gain versus off-boresight angle."""
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    if abs(cos_angle) < 1e-8:
        return 0.0

    field = math.cos(0.5 * math.pi * math.sin(angle_rad)) / cos_angle
    return max(field * field, 0.0)


DEFAULT_DIPOLE_VALUES = tuple(half_wave_dipole_power_cut(angle) for angle in DEFAULT_DIPOLE_ANGLES_DEG)


def interp1d_zero_outside(axis: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if query.numel() == 0:
        return torch.empty_like(query, dtype=values.dtype)

    flat_query = query.reshape(-1)
    index_upper = torch.bucketize(flat_query.detach(), axis)
    index_left = torch.clamp(index_upper - 1, 0, axis.numel() - 1)
    index_right = torch.clamp(index_upper, 0, axis.numel() - 1)

    x0 = axis[index_left]
    x1 = axis[index_right]
    y0 = values[index_left]
    y1 = values[index_right]
    denom = torch.clamp(x1 - x0, min=1e-12)
    weight = torch.where(index_left == index_right, torch.zeros_like(flat_query), (flat_query - x0) / denom)
    interpolated = y0 + weight * (y1 - y0)
    inside = (flat_query >= axis[0]) & (flat_query <= axis[-1])
    return torch.where(inside, interpolated, torch.zeros_like(interpolated)).reshape(query.shape)


def interp2d_zero_outside(
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
    values: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
) -> torch.Tensor:
    flat_x = x_query.reshape(-1)
    flat_y = y_query.reshape(-1)

    x_upper = torch.bucketize(flat_x.detach(), x_axis)
    y_upper = torch.bucketize(flat_y.detach(), y_axis)
    x_left = torch.clamp(x_upper - 1, 0, x_axis.numel() - 1)
    x_right = torch.clamp(x_upper, 0, x_axis.numel() - 1)
    y_low = torch.clamp(y_upper - 1, 0, y_axis.numel() - 1)
    y_high = torch.clamp(y_upper, 0, y_axis.numel() - 1)

    x0 = x_axis[x_left]
    x1 = x_axis[x_right]
    y0 = y_axis[y_low]
    y1 = y_axis[y_high]
    tx = torch.where(x_left == x_right, torch.zeros_like(flat_x), (flat_x - x0) / torch.clamp(x1 - x0, min=1e-12))
    ty = torch.where(y_low == y_high, torch.zeros_like(flat_y), (flat_y - y0) / torch.clamp(y1 - y0, min=1e-12))

    v00 = values[y_low, x_left]
    v10 = values[y_low, x_right]
    v01 = values[y_high, x_left]
    v11 = values[y_high, x_right]

    interpolated = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    inside = (
        (flat_x >= x_axis[0])
        & (flat_x <= x_axis[-1])
        & (flat_y >= y_axis[0])
        & (flat_y <= y_axis[-1])
    )
    return torch.where(inside, interpolated, torch.zeros_like(interpolated)).reshape(x_query.shape)


def evaluate_antenna_pattern_xy(
    pattern_kind: str,
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
    x_values: torch.Tensor | None,
    y_values: torch.Tensor | None,
    values_2d: torch.Tensor | None,
    x_angles_deg: torch.Tensor,
    y_angles_deg: torch.Tensor,
) -> torch.Tensor:
    if pattern_kind == "separable":
        return interp1d_zero_outside(x_axis, x_values, x_angles_deg) * interp1d_zero_outside(
            y_axis,
            y_values,
            y_angles_deg,
        )
    return interp2d_zero_outside(x_axis, y_axis, values_2d, x_angles_deg, y_angles_deg)


def evaluate_antenna_pattern_vectors(
    pattern_kind: str,
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
    x_values: torch.Tensor | None,
    y_values: torch.Tensor | None,
    values_2d: torch.Tensor | None,
    vectors: torch.Tensor,
) -> torch.Tensor:
    forward = -vectors[..., 2]
    x_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 0], forward))
    y_angles_deg = torch.rad2deg(torch.atan2(vectors[..., 1], forward))
    return evaluate_antenna_pattern_xy(
        pattern_kind,
        x_axis,
        y_axis,
        x_values,
        y_values,
        values_2d,
        x_angles_deg,
        y_angles_deg,
    )
