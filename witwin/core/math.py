"""Vector, quaternion, and rotation utilities for geometry types."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _vec3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return tuple(float(v) for v in value)


def _positive(value: float, *, name: str) -> float:
    scalar = float(value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return scalar


def _optional_rotation(value):
    if value is None:
        return None
    return _vec3(value, name="rotation")


def _scale3(value) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar, scalar, scalar
    return _vec3(value, name="scale")


# ---------------------------------------------------------------------------
# Euler rotation matrix (numpy, legacy)
# ---------------------------------------------------------------------------

def _rotation_matrix_np(angles) -> np.ndarray:
    if angles is None:
        return np.eye(3, dtype=np.float32)

    roll, pitch, yaw = _vec3(angles, name="rotation")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Quaternion utilities (torch)
# ---------------------------------------------------------------------------

def quat_identity(*, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return identity quaternion [w, x, y, z] = [1, 0, 0, 0]."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)


def quat_from_euler(roll: float, pitch: float, yaw: float,
                    *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Convert Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z]."""
    roll_t = torch.as_tensor(roll, device=device, dtype=dtype)
    pitch_t = torch.as_tensor(pitch, device=device, dtype=dtype)
    yaw_t = torch.as_tensor(yaw, device=device, dtype=dtype)
    hr, hp, hy = roll_t / 2, pitch_t / 2, yaw_t / 2
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z])


def quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix (differentiable)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ]).reshape(3, 3)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions [w, x, y, z]."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_rotation_matrix_np(q: torch.Tensor) -> np.ndarray:
    """Convert quaternion to 3x3 numpy rotation matrix."""
    return quat_to_rotation_matrix(q).detach().cpu().numpy()
