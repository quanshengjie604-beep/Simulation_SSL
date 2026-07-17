#!/usr/bin/env python3
"""Shared SMPL utilities for RT-Pose sequence mesh fitting."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "logs" / ".cache" / "matplotlib"

RT_BONES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (7, 9),
    (9, 10),
    (10, 11),
    (7, 12),
    (12, 13),
    (13, 14),
)

SMPL = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

RT_TO_SMPL = {
    0: "pelvis",
    1: "right_hip",
    2: "right_knee",
    3: "right_ankle",
    4: "left_hip",
    5: "left_knee",
    6: "left_ankle",
    8: "head",
    9: "right_shoulder",
    10: "right_elbow",
    11: "right_wrist",
    12: "left_shoulder",
    13: "left_elbow",
    14: "left_wrist",
}


def import_runtime():
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
        torch = None
    try:
        import smplx
    except ImportError:
        missing.append("smplx")
        smplx = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        missing.append("matplotlib")
        plt = None
    if missing:
        raise RuntimeError(
            "Missing Python packages: "
            + ", ".join(missing)
            + ". Install them in the selected Python environment before running this fitter."
        )
    globals()["torch"] = torch
    return torch, smplx, plt, None, None


def rtpose_to_world(points_xyz: np.ndarray) -> np.ndarray:
    out = np.empty_like(points_xyz, dtype=np.float32)
    out[:, 0] = points_xyz[:, 1]
    out[:, 1] = points_xyz[:, 2]
    out[:, 2] = -points_xyz[:, 0]
    return out


def safe_normalize_t(x, eps: float = 1e-8):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def safe_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = max(float(np.linalg.norm(x)), eps)
    return x / norm


def body_axes_np(kpts: np.ndarray, desired_front: np.ndarray | None = None) -> np.ndarray:
    pelvis = kpts[0]
    shoulder_mid = 0.5 * (kpts[9] + kpts[12])
    upper_mid = 0.5 * (shoulder_mid + kpts[8])
    up = safe_normalize_np(upper_mid - pelvis)
    right_anatomical = safe_normalize_np(0.5 * (kpts[9] + kpts[1]) - 0.5 * (kpts[12] + kpts[4]))
    left = -right_anatomical
    front = safe_normalize_np(np.cross(left, up))
    left = safe_normalize_np(np.cross(up, front))
    if desired_front is not None:
        sign_hint = desired_front - up * float(np.dot(desired_front, up))
        sign_hint = safe_normalize_np(sign_hint)
    else:
        sign_hint = front
    if np.dot(front, sign_hint) < 0:
        front = -front
        left = -left
    return np.stack([left, up, front], axis=1).astype(np.float32)


def matrix_to_axis_angle_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_theta = max(min((trace - 1.0) * 0.5, 1.0), -1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    if abs(math.pi - theta) < 1e-4:
        diag = np.diag(rotation)
        axis_index = int(np.argmax(diag))
        if axis_index == 0:
            x = math.sqrt(max((rotation[0, 0] + 1.0) * 0.5, 1e-8))
            y = (rotation[0, 1] + rotation[1, 0]) / (4.0 * x)
            z = (rotation[0, 2] + rotation[2, 0]) / (4.0 * x)
            axis = np.array([x, y, z], dtype=np.float32)
        elif axis_index == 1:
            y = math.sqrt(max((rotation[1, 1] + 1.0) * 0.5, 1e-8))
            x = (rotation[0, 1] + rotation[1, 0]) / (4.0 * y)
            z = (rotation[1, 2] + rotation[2, 1]) / (4.0 * y)
            axis = np.array([x, y, z], dtype=np.float32)
        else:
            z = math.sqrt(max((rotation[2, 2] + 1.0) * 0.5, 1e-8))
            x = (rotation[0, 2] + rotation[2, 0]) / (4.0 * z)
            y = (rotation[1, 2] + rotation[2, 1]) / (4.0 * z)
            axis = np.array([x, y, z], dtype=np.float32)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
        return (axis * theta).astype(np.float32)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float32,
    )
    axis = axis / (2.0 * math.sin(theta))
    return (axis * theta).astype(np.float32)


def smpl_rtpose_joints(joints, vertices=None, head_vertex_ids=None):
    values = []
    for rt_idx in range(15):
        if rt_idx == 7:
            thorax = (
                0.50 * joints[:, SMPL["neck"]]
                + 0.25 * joints[:, SMPL["left_shoulder"]]
                + 0.25 * joints[:, SMPL["right_shoulder"]]
            )
            values.append(thorax)
        elif rt_idx == 8 and vertices is not None and head_vertex_ids is not None:
            values.append(vertices[:, head_vertex_ids].mean(dim=1))
        else:
            values.append(joints[:, SMPL[RT_TO_SMPL[rt_idx]]])
    return torch.stack(values, dim=1)


def surface_orientation_ids(model, device) -> dict[str, object]:
    vertices = model.v_template.detach().cpu().numpy()

    def split_front_back(indices: np.ndarray, low_q: float = 0.25, high_q: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
        z = vertices[indices, 2]
        front = indices[z >= np.quantile(z, high_q)]
        back = indices[z <= np.quantile(z, low_q)]
        return front.astype(np.int64), back.astype(np.int64)

    head = np.where(vertices[:, 1] >= np.quantile(vertices[:, 1], 0.93))[0]
    right_hand = np.where(vertices[:, 0] <= np.quantile(vertices[:, 0], 0.025))[0]
    left_hand = np.where(vertices[:, 0] >= np.quantile(vertices[:, 0], 0.975))[0]
    low = vertices[:, 1] <= np.quantile(vertices[:, 1], 0.12)
    right_foot = np.where(low & (vertices[:, 0] < 0.0))[0]
    left_foot = np.where(low & (vertices[:, 0] > 0.0))[0]

    ids: dict[str, object] = {}
    for name, region in (
        ("head", head),
        ("right_hand", right_hand),
        ("left_hand", left_hand),
        ("right_foot", right_foot),
        ("left_foot", left_foot),
    ):
        front, back = split_front_back(region)
        ids[f"{name}_front"] = torch.as_tensor(front, dtype=torch.long, device=device)
        ids[f"{name}_back"] = torch.as_tensor(back, dtype=torch.long, device=device)
    return ids


def body_axes_t(j):
    pelvis = j[:, 0]
    shoulder_mid = 0.5 * (j[:, 9] + j[:, 12])
    upper_mid = 0.5 * (shoulder_mid + j[:, 8])
    up = safe_normalize_t(upper_mid - pelvis)
    right_anatomical = safe_normalize_t(0.5 * (j[:, 9] + j[:, 1]) - 0.5 * (j[:, 12] + j[:, 4]))
    left = -right_anatomical
    front = safe_normalize_t(torch.cross(left, up, dim=-1))
    left = safe_normalize_t(torch.cross(up, front, dim=-1))
    return left, up, front


def huber(x, delta: float = 0.05):
    abs_x = x.abs()
    quad = 0.5 * abs_x.square()
    linear = delta * (abs_x - 0.5 * delta)
    return torch.where(abs_x <= delta, quad, linear)


def angle_between(a, b):
    a_n = safe_normalize_t(a)
    b_n = safe_normalize_t(b)
    cos = (a_n * b_n).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return cos.acos()


def point_line_bend_dir(parent, joint, child):
    line = child - parent
    proj = parent + ((joint - parent) * line).sum(dim=-1, keepdim=True) / line.square().sum(dim=-1, keepdim=True).clamp_min(1e-8) * line
    return safe_normalize_t(joint - proj)


def target_limb_bend_dirs(target, body_forward):
    dirs = []
    for parent_i, joint_i, child_i in ((1, 2, 3), (4, 5, 6), (9, 10, 11), (12, 13, 14)):
        bend = point_line_bend_dir(target[:, parent_i], target[:, joint_i], target[:, child_i])
        bend_norm = (target[:, joint_i] - target[:, parent_i]).norm(dim=-1, keepdim=True)
        dirs.append(torch.where(bend_norm > 1e-3, bend, body_forward))
    return torch.stack(dirs, dim=1)


def body_pose_slice(full_joint_index: int) -> slice:
    start = (full_joint_index - 1) * 3
    return slice(start, start + 3)


LOCKED_BODY_POSE_JOINTS = (
    SMPL["neck"],
    SMPL["head"],
    SMPL["left_wrist"],
    SMPL["right_wrist"],
    22,
    23,
)


def locked_body_pose_values(body_pose):
    return torch.cat([body_pose[:, body_pose_slice(j)] for j in LOCKED_BODY_POSE_JOINTS], dim=1)


def zero_locked_body_pose_(body_pose) -> None:
    for joint_index in LOCKED_BODY_POSE_JOINTS:
        body_pose[:, body_pose_slice(joint_index)] = 0.0


def anatomy_prior(body_pose):
    return locked_body_pose_values(body_pose).square().mean()


def joint_rotation_norm(body_pose, full_joint_index: int):
    return body_pose[:, body_pose_slice(full_joint_index)].norm(dim=-1)


def articulation_prior(body_pose):
    terms = []
    for joint_index in (SMPL["left_collar"], SMPL["right_collar"]):
        terms.append(relu(joint_rotation_norm(body_pose, joint_index) - 0.45).square().mean())
    for joint_index in (SMPL["left_shoulder"], SMPL["right_shoulder"]):
        terms.append(relu(joint_rotation_norm(body_pose, joint_index) - 0.90).square().mean())
    for joint_index in (SMPL["left_elbow"], SMPL["right_elbow"]):
        terms.append(relu(joint_rotation_norm(body_pose, joint_index) - 0.45).square().mean())
    return sum(terms) / len(terms)


def limb_angle_loss(j, target_bend_dirs):
    terms = []
    pairs = (
        (1, 2, 3, 0, math.radians(25.0), math.radians(178.0), 0.75),
        (4, 5, 6, 1, math.radians(25.0), math.radians(178.0), 0.75),
        (9, 10, 11, 2, math.radians(20.0), math.radians(178.0), 0.90),
        (12, 13, 14, 3, math.radians(20.0), math.radians(178.0), 0.90),
    )
    for parent_i, joint_i, child_i, target_idx, min_angle, max_angle, bend_weight in pairs:
        parent = j[:, parent_i]
        joint = j[:, joint_i]
        child = j[:, child_i]
        theta = angle_between(parent - joint, child - joint)
        range_loss = relu(theta.new_tensor(min_angle) - theta).square() + relu(theta - theta.new_tensor(max_angle)).square()
        bend_dir = point_line_bend_dir(parent, joint, child)
        target_dir = target_bend_dirs[:, target_idx]
        bend_loss = relu(theta.new_tensor(0.25) - (bend_dir * target_dir).sum(dim=-1)).square()
        terms.append(range_loss.mean() + bend_weight * bend_loss.mean())
    return sum(terms) / len(terms)


def relu(x):
    return torch.relu(x)


def forward_model(model, global_orient, body_pose, betas, transl):
    return model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=transl,
        return_verts=True,
    )
