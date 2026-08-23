#!/usr/bin/env python3
"""Fit MVDoppler-17 keypoints with the repository's SMPL-fit v9 method.

This is an adaptation of the original ``code/SMPL_fit/fit_smpl_sequence.py``:
the 15-point RT-Pose projection, bones, body axes, and temporal joint weights
are replaced with the 17-point layout used by MVDoppler-Pose. It does not
use ``keypoints17_to_smplx_module.py`` or its scale-fitting objective.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE = REPO_ROOT / "logs" / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


DEFAULT_RADAR_H5 = (
    REPO_ROOT
    / "datasets"
    / "MVDoppler-Pose"
    / "Data"
    / "MVDoppler_public"
    / "2022Jun25-0207"
    / "radar_v2"
    / "20220625022344.h5"
)

JOINT_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_foot",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

LEFT_RIGHT_PAIRS = np.asarray(
    ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)),
    dtype=np.int64,
)

BONES = np.asarray(
    (
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    ),
    dtype=np.int64,
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
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

JOINT_WEIGHTS = np.asarray(
    [1.6, 2.8, 1.2, 0.9, 2.8, 1.2, 0.9, 1.2, 1.4, 1.1, 1.1, 1.2, 0.9, 0.7, 1.2, 0.9, 0.7],
    dtype=np.float32,
)
VELOCITY_WEIGHTS = np.asarray(
    [0.3, 0.4, 0.7, 1.0, 0.4, 0.7, 1.0, 0.3, 0.4, 0.5, 0.6, 7.0, 5.0, 4.0, 7.0, 5.0, 4.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Stage:
    name: str
    iters: int
    lr: float
    optimize_pose: bool
    optimize_betas: bool
    w_keypoint: float
    w_pose: float
    w_shape: float
    w_bone_length: float
    w_lower_body_side: float
    w_joint_velocity: float
    w_joint_accel: float
    w_foot_forward: float


# Exact v9 stage schedule and active loss weights from fit_smpl_sequence.py.
V9_STAGES = (
    Stage("stage1_global", 180, 8e-3, False, False, 4.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.40),
    Stage("stage2_pose", 1350, 4e-3, True, False, 8.0, 3e-4, 0.0, 0.10, 0.35, 0.25, 0.08, 3.00),
    Stage("stage3_shape_refine", 750, 1.5e-3, True, True, 8.0, 3e-4, 0.01, 0.10, 0.35, 0.20, 0.06, 3.00),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar-h5", type=Path, default=DEFAULT_RADAR_H5)
    parser.add_argument(
        "--keypoint-archive",
        type=Path,
        default=REPO_ROOT / "datasets" / "MVDoppler-Pose" / "Data" / "MVDoppler_public_keypoint.zip",
    )
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "smpl_models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mvdoppler_pose_smpl_fit17_v9",
    )
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="male")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-fps", type=float, default=25.0)
    parser.add_argument("--gif-fps", type=float, default=12.5)
    parser.add_argument("--max-render-faces", type=int, default=4000)
    parser.add_argument("--iters-scale", type=float, default=1.0)
    parser.add_argument(
        "--source-lr",
        choices=("mirrored", "as-stored"),
        default="mirrored",
        help=(
            "Interpret the source as left-right mirrored (default), or retain its stored "
            "joint semantics. Mirroring all paired limbs reverses SMPL facing without "
            "changing the observed joint geometry."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Debug only; 0 fits the complete sequence.")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def load_keypoints(radar_h5: Path, archive: Path) -> tuple[np.ndarray, str]:
    session = radar_h5.parent.parent.name
    member = f"MVDoppler_public_keypoint/{session}/{radar_h5.stem}/output_3D/keypoints.npy"
    with zipfile.ZipFile(archive) as handle:
        try:
            payload = handle.read(member)
        except KeyError as exc:
            raise FileNotFoundError(f"Could not find {member!r} in {archive}") from exc
    keypoints = np.load(io.BytesIO(payload), allow_pickle=False).astype(np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"Expected (T,17,3) keypoints, got {keypoints.shape}")
    if not np.isfinite(keypoints).all():
        raise ValueError("MVDoppler keypoints contain NaN or Inf")
    return keypoints, member


def safe_normalize_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return values / max(float(np.linalg.norm(values)), eps)


def safe_normalize_t(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def swap_left_right(keypoints: np.ndarray) -> np.ndarray:
    swapped = keypoints.copy()
    for left_index, right_index in LEFT_RIGHT_PAIRS:
        swapped[..., left_index, :] = keypoints[..., right_index, :]
        swapped[..., right_index, :] = keypoints[..., left_index, :]
    return swapped


def body_axes_np(keypoints: np.ndarray) -> np.ndarray:
    pelvis = keypoints[0]
    shoulder_mid = 0.5 * (keypoints[11] + keypoints[14])
    upper_mid = 0.5 * (shoulder_mid + keypoints[8])
    up = safe_normalize_np(upper_mid - pelvis)
    right_anatomical = safe_normalize_np(
        0.5 * (keypoints[14] + keypoints[1]) - 0.5 * (keypoints[11] + keypoints[4])
    )
    left = -right_anatomical
    front = safe_normalize_np(np.cross(left, up))
    left = safe_normalize_np(np.cross(up, front))
    return np.stack((left, up, front), axis=1).astype(np.float32)


def body_axes_t(keypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pelvis = keypoints[:, 0]
    shoulder_mid = 0.5 * (keypoints[:, 11] + keypoints[:, 14])
    upper_mid = 0.5 * (shoulder_mid + keypoints[:, 8])
    up = safe_normalize_t(upper_mid - pelvis)
    right_anatomical = safe_normalize_t(
        0.5 * (keypoints[:, 14] + keypoints[:, 1])
        - 0.5 * (keypoints[:, 11] + keypoints[:, 4])
    )
    left = -right_anatomical
    front = safe_normalize_t(torch.cross(left, up, dim=-1))
    left = safe_normalize_t(torch.cross(up, front, dim=-1))
    return left, up, front


def matrix_to_axis_angle_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(float(cos_theta))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    if abs(math.pi - theta) < 1e-4:
        values, vectors = np.linalg.eigh(rotation)
        axis = vectors[:, int(np.argmin(np.abs(values - 1.0)))]
        return (safe_normalize_np(axis.astype(np.float32)) * theta).astype(np.float32)
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float32,
    ) / (2.0 * math.sin(theta))
    return (axis * theta).astype(np.float32)


def smpl_mvdoppler17_joints(output) -> torch.Tensor:
    joints = output.joints
    midhip = 0.5 * (joints[:, SMPL["left_hip"]] + joints[:, SMPL["right_hip"]])
    shoulder_mid = 0.5 * (
        joints[:, SMPL["left_shoulder"]] + joints[:, SMPL["right_shoulder"]]
    )
    values = (
        midhip,
        joints[:, SMPL["right_hip"]],
        joints[:, SMPL["right_knee"]],
        joints[:, SMPL["right_ankle"]],
        joints[:, SMPL["left_hip"]],
        joints[:, SMPL["left_knee"]],
        joints[:, SMPL["left_ankle"]],
        0.5 * (midhip + shoulder_mid),
        shoulder_mid,
        joints[:, SMPL["neck"]],
        joints[:, SMPL["head"]],
        joints[:, SMPL["left_shoulder"]],
        joints[:, SMPL["left_elbow"]],
        joints[:, SMPL["left_wrist"]],
        joints[:, SMPL["right_shoulder"]],
        joints[:, SMPL["right_elbow"]],
        joints[:, SMPL["right_wrist"]],
    )
    return torch.stack(values, dim=1)


def huber(values: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    absolute = values.abs()
    quadratic = 0.5 * absolute.square()
    linear = delta * (absolute - 0.5 * delta)
    return torch.where(absolute <= delta, quadratic, linear)


def bone_length_loss(joints: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    edges = torch.as_tensor(BONES, dtype=torch.long, device=joints.device)
    pred = (joints[:, edges[:, 0]] - joints[:, edges[:, 1]]).norm(dim=-1)
    truth = (target[:, edges[:, 0]] - target[:, edges[:, 1]]).norm(dim=-1)
    return huber(pred - truth, delta=0.03).mean()


def lower_body_side_loss(joints: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    across = safe_normalize_t(target[:, 1] - target[:, 4]).detach()
    pred_pelvis = joints[:, 0]
    target_pelvis = target[:, 0]
    terms = []
    term_weights = []
    for right, left, pair_weight, min_sep_fraction in (
        (1, 4, 6.0, 0.70),
        (2, 5, 2.0, 0.55),
        (3, 6, 1.0, 0.45),
    ):
        target_sep = ((target[:, right] - target[:, left]) * across).sum(dim=-1)
        pred_sep = ((joints[:, right] - joints[:, left]) * across).sum(dim=-1)
        target_sep_abs = target_sep.abs()
        min_sep = torch.clamp(min_sep_fraction * target_sep_abs, min=0.035)
        sep_match = huber(pred_sep - target_sep, delta=0.04)
        sep_barrier = torch.relu(min_sep - pred_sep).square()
        pred_right = ((joints[:, right] - pred_pelvis) * across).sum(dim=-1)
        pred_left = ((joints[:, left] - pred_pelvis) * across).sum(dim=-1)
        target_right = ((target[:, right] - target_pelvis) * across).sum(dim=-1)
        target_left = ((target[:, left] - target_pelvis) * across).sum(dim=-1)
        side_match = huber(pred_right - target_right, delta=0.04) + huber(
            pred_left - target_left, delta=0.04
        )
        side_margin = torch.clamp(0.30 * target_sep_abs, min=0.022)
        side_barrier = torch.relu(side_margin - pred_right).square() + torch.relu(
            side_margin + pred_left
        ).square()
        terms.append(pair_weight * (sep_match + sep_barrier + 0.5 * side_match + side_barrier).mean())
        term_weights.append(pair_weight)
    return torch.stack(terms).sum() / sum(term_weights)


def joint_velocity_loss(
    joints: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if len(joints) < 2:
        return joints.new_tensor(0.0)
    difference = (joints[1:] - joints[:-1]) - (target[1:] - target[:-1])
    return (huber(difference, delta=0.04).sum(dim=-1) * weights).mean()


def joint_accel_loss(
    joints: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if len(joints) < 3:
        return joints.new_tensor(0.0)
    fit_velocity = joints[1:] - joints[:-1]
    target_velocity = target[1:] - target[:-1]
    difference = (fit_velocity[1:] - fit_velocity[:-1]) - (
        target_velocity[1:] - target_velocity[:-1]
    )
    return (huber(difference, delta=0.03).sum(dim=-1) * weights).mean()


def foot_surface_ids(model, device: torch.device) -> dict[str, torch.Tensor]:
    vertices = model.v_template.detach().cpu().numpy()
    low = vertices[:, 1] <= np.quantile(vertices[:, 1], 0.12)
    regions = {
        "right": np.where(low & (vertices[:, 0] < 0.0))[0],
        "left": np.where(low & (vertices[:, 0] > 0.0))[0],
    }
    ids: dict[str, torch.Tensor] = {}
    for side, region in regions.items():
        z = vertices[region, 2]
        ids[f"{side}_front"] = torch.as_tensor(
            region[z >= np.quantile(z, 0.75)], dtype=torch.long, device=device
        )
        ids[f"{side}_back"] = torch.as_tensor(
            region[z <= np.quantile(z, 0.25)], dtype=torch.long, device=device
        )
    return ids


def foot_forward_loss(
    vertices: torch.Tensor,
    target_axes: torch.Tensor,
    surface_ids: dict[str, torch.Tensor],
) -> torch.Tensor:
    target_front = target_axes[:, :, 2].detach()

    def front_vector(side: str) -> torch.Tensor:
        front = vertices[:, surface_ids[f"{side}_front"]].mean(dim=1)
        back = vertices[:, surface_ids[f"{side}_back"]].mean(dim=1)
        return safe_normalize_t(front - back)

    min_dot = math.cos(math.radians(45.0))
    right_dot = (front_vector("right") * target_front).sum(dim=-1)
    left_dot = (front_vector("left") * target_front).sum(dim=-1)
    return 0.5 * (
        torch.relu(min_dot - right_dot).square().mean()
        + torch.relu(min_dot - left_dot).square().mean()
    )


def body_pose_slice(full_joint_index: int) -> slice:
    start = (full_joint_index - 1) * 3
    return slice(start, start + 3)


def zero_locked_pose(body_pose: torch.Tensor) -> None:
    for joint_index in (SMPL["neck"], SMPL["head"], SMPL["left_wrist"], SMPL["right_wrist"]):
        body_pose[:, body_pose_slice(joint_index)] = 0.0


def sequence_loss(
    model,
    target: torch.Tensor,
    target_axes: torch.Tensor,
    weights: torch.Tensor,
    velocity_weights: torch.Tensor,
    surface_ids: dict[str, torch.Tensor],
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    stage: Stage,
):
    output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas.expand(len(target), -1),
        transl=transl,
        return_verts=True,
    )
    joints = smpl_mvdoppler17_joints(output)
    keypoint = (huber(joints - target).sum(dim=-1) * weights).mean()
    pose = body_pose.square().mean()
    shape = betas.square().mean()
    bone = bone_length_loss(joints, target)
    lower_side = lower_body_side_loss(joints, target)
    velocity = joint_velocity_loss(joints, target, velocity_weights)
    acceleration = joint_accel_loss(joints, target, velocity_weights)
    foot_forward = foot_forward_loss(output.vertices, target_axes, surface_ids)
    loss = (
        stage.w_keypoint * keypoint
        + stage.w_pose * pose
        + stage.w_shape * shape
        + stage.w_bone_length * bone
        + stage.w_lower_body_side * lower_side
        + stage.w_joint_velocity * velocity
        + stage.w_joint_accel * acceleration
        + stage.w_foot_forward * foot_forward
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "keypoint": float(keypoint.detach().cpu()),
        "pose": float(pose.detach().cpu()),
        "shape": float(shape.detach().cpu()),
        "bone_length": float(bone.detach().cpu()),
        "lower_body_side": float(lower_side.detach().cpu()),
        "joint_velocity": float(velocity.detach().cpu()),
        "joint_accel": float(acceleration.detach().cpu()),
        "foot_forward": float(foot_forward.detach().cpu()),
    }
    return loss, parts, output, joints


def collect_params(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    stage: Stage,
) -> list[torch.Tensor]:
    params = [global_orient, transl]
    if stage.optimize_pose:
        params.append(body_pose)
    if stage.optimize_betas:
        params.append(betas)
    for tensor in (global_orient, body_pose, betas, transl):
        tensor.requires_grad_(any(tensor is param for param in params))
    return params


def display_coordinates(points: np.ndarray) -> np.ndarray:
    return points[..., [0, 2, 1]] * np.asarray([1.0, 1.0, -1.0], dtype=np.float32)


def skeleton_segments(joints: np.ndarray) -> np.ndarray:
    return joints[BONES]


def view_bounds(vertices: np.ndarray, keypoints: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.concatenate((vertices.reshape(-1, 3), keypoints.reshape(-1, 3)), axis=0)
    lower = np.percentile(points, 0.05, axis=0)
    upper = np.percentile(points, 99.95, axis=0)
    return 0.5 * (lower + upper), max(0.9, 0.58 * float(np.max(upper - lower)))


def apply_view(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_proj_type("ortho")
    ax.view_init(elev=7, azim=-88)
    ax.set_axis_off()


def add_artists(ax, vertices, faces, source, fitted):
    mesh = Poly3DCollection(
        vertices[faces],
        facecolors="#7fa9c2",
        edgecolors="#42677c",
        linewidths=0.025,
        alpha=0.50,
    )
    source_lines = Line3DCollection(skeleton_segments(source), colors="#ef4444", linewidths=2.2)
    fitted_lines = Line3DCollection(skeleton_segments(fitted), colors="#06b6d4", linewidths=1.35)
    ax.add_collection3d(mesh)
    ax.add_collection3d(source_lines)
    ax.add_collection3d(fitted_lines)
    source_points = ax.scatter(*source.T, color="#dc2626", s=19, depthshade=False)
    fitted_points = ax.scatter(*fitted.T, color="#0891b2", s=9, depthshade=False)
    return mesh, source_lines, fitted_lines, source_points, fitted_points


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#7fa9c2", linewidth=7, alpha=0.65, label="V9-fitted SMPL-X mesh"),
        Line2D([0], [0], color="#ef4444", marker="o", linewidth=2, label="Input MVDoppler-17 keypoints"),
        Line2D([0], [0], color="#06b6d4", marker="o", linewidth=1.5, label="Fitted 17 joints"),
    ]


def render_contact_sheet(
    output: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    source: np.ndarray,
    fitted: np.ndarray,
    source_fps: float,
    mpjpe_mm: float,
) -> None:
    frame_ids = np.linspace(0, len(vertices) - 1, 6, dtype=np.int64)
    center, radius = view_bounds(vertices, source)
    fig = plt.figure(figsize=(12.5, 8.3), dpi=150)
    for panel, frame_id in enumerate(frame_ids, start=1):
        ax = fig.add_subplot(2, 3, panel, projection="3d")
        add_artists(ax, vertices[frame_id], faces, source[frame_id], fitted[frame_id])
        apply_view(ax, center, radius)
        ax.set_title(f"frame {frame_id}  t={frame_id / source_fps:.2f} s", fontsize=10, pad=-3)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"MVDoppler-17 fit using repository SMPL-fit v9  |  MPJPE {mpjpe_mm:.1f} mm")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def render_gif(
    output: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    source: np.ndarray,
    fitted: np.ndarray,
    source_fps: float,
    gif_fps: float,
) -> np.ndarray:
    step = max(1, int(round(source_fps / gif_fps)))
    frame_ids = np.arange(0, len(vertices), step, dtype=np.int64)
    actual_fps = source_fps / step
    center, radius = view_bounds(vertices, source)
    fig = plt.figure(figsize=(6.4, 7.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    artists = add_artists(ax, vertices[0], faces, source[0], fitted[0])
    mesh, source_lines, fitted_lines, source_points, fitted_points = artists
    apply_view(ax, center, radius)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=1, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.14)

    def update(index: int):
        frame_id = int(frame_ids[index])
        mesh.set_verts(vertices[frame_id][faces])
        source_lines.set_segments(skeleton_segments(source[frame_id]))
        fitted_lines.set_segments(skeleton_segments(fitted[frame_id]))
        source_points._offsets3d = tuple(source[frame_id].T)
        fitted_points._offsets3d = tuple(fitted[frame_id].T)
        ax.set_title(
            f"Repository SMPL-fit v9 adapted to MVDoppler-17\nframe {frame_id}/{len(vertices) - 1}, "
            f"t={frame_id / source_fps:.2f} s",
            fontsize=11,
        )
        return artists

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000.0 / actual_fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=actual_fps))
    plt.close(fig)
    return frame_ids


def main() -> int:
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iters_scale <= 0.0:
        raise ValueError("--iters-scale must be positive")

    radar_h5 = args.radar_h5.expanduser().resolve()
    archive = args.keypoint_archive.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_keypoints, keypoint_member = load_keypoints(radar_h5, archive)
    if args.max_frames > 0:
        source_keypoints = source_keypoints[: args.max_frames]
    fit_keypoints = (
        swap_left_right(source_keypoints)
        if args.source_lr == "mirrored"
        else source_keypoints.copy()
    )
    num_frames = len(source_keypoints)
    device = torch.device(args.device)
    print(f"loaded MVDoppler-17 keypoints {source_keypoints.shape} from {keypoint_member}", flush=True)
    print(f"source left/right interpretation: {args.source_lr}", flush=True)
    print(f"running repository SMPL-fit v9 adaptation on {torch.cuda.get_device_name(device)}", flush=True)

    model = smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=args.gender,
        batch_size=num_frames,
        num_betas=10,
        use_pca=False,
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_transl=False,
    ).to(device).eval()
    surface_ids = foot_surface_ids(model, device)
    axes_np = np.stack([body_axes_np(frame) for frame in fit_keypoints])
    target = torch.as_tensor(fit_keypoints, dtype=torch.float32, device=device)
    target_axes = torch.as_tensor(axes_np, dtype=torch.float32, device=device)
    weights = torch.as_tensor(JOINT_WEIGHTS, device=device)
    velocity_weights = torch.as_tensor(VELOCITY_WEIGHTS, device=device)

    global_orient = torch.tensor(
        np.stack([matrix_to_axis_angle_np(axes) for axes in axes_np]),
        dtype=torch.float32,
        device=device,
    )
    body_pose = torch.zeros((num_frames, int(model.NUM_BODY_JOINTS) * 3), device=device)
    betas = torch.zeros((1, 10), device=device)
    transl = torch.as_tensor(fit_keypoints[:, 0], dtype=torch.float32, device=device).clone()
    history = []

    for stage in V9_STAGES:
        stage_iters = max(1, int(round(stage.iters * args.iters_scale)))
        optimizer = torch.optim.Adam(
            collect_params(global_orient, body_pose, betas, transl, stage), lr=stage.lr
        )
        for iteration in range(stage_iters):
            optimizer.zero_grad(set_to_none=True)
            loss, parts, _, _ = sequence_loss(
                model,
                target,
                target_axes,
                weights,
                velocity_weights,
                surface_ids,
                global_orient,
                body_pose,
                betas,
                transl,
                stage,
            )
            loss.backward()
            optimizer.step()
            if stage.optimize_pose:
                with torch.no_grad():
                    zero_locked_pose(body_pose)
            if stage.optimize_betas:
                with torch.no_grad():
                    betas.clamp_(-2.0, 2.0)
            if iteration == 0 or iteration == stage_iters - 1 or (iteration + 1) % 50 == 0:
                record = {"stage": stage.name, "iter": iteration + 1, **parts}
                history.append(record)
                print(f"{stage.name} {iteration + 1:04d}/{stage_iters}: {parts}", flush=True)

    with torch.no_grad():
        _, final_parts, output, fitted = sequence_loss(
            model,
            target,
            target_axes,
            weights,
            velocity_weights,
            surface_ids,
            global_orient,
            body_pose,
            betas,
            transl,
            V9_STAGES[-1],
        )
    fitted_fit_order = fitted.cpu().numpy().astype(np.float32)
    fitted_np = (
        swap_left_right(fitted_fit_order)
        if args.source_lr == "mirrored"
        else fitted_fit_order.copy()
    )
    vertices = output.vertices.cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    errors = np.linalg.norm(fitted_np - source_keypoints, axis=-1)
    velocity_errors = np.linalg.norm(
        np.diff(fitted_np, axis=0) - np.diff(source_keypoints, axis=0), axis=-1
    ) if num_frames > 1 else np.zeros((0, 17), dtype=np.float32)
    acceleration_errors = np.linalg.norm(
        np.diff(fitted_np, n=2, axis=0) - np.diff(source_keypoints, n=2, axis=0), axis=-1
    ) if num_frames > 2 else np.zeros((0, 17), dtype=np.float32)

    stem = radar_h5.stem
    lr_tag = "lr_mirrored" if args.source_lr == "mirrored" else "lr_as_stored"
    fit_path = output_dir / f"{stem}_smplfit_v9_h36m17_{lr_tag}.npz"
    metrics_path = output_dir / f"{stem}_smplfit_v9_h36m17_{lr_tag}.json"
    contact_path = output_dir / f"{stem}_smplfit_v9_h36m17_{lr_tag}_overlay.png"
    gif_path = output_dir / f"{stem}_smplfit_v9_h36m17_{lr_tag}_overlay.gif"
    np.savez_compressed(
        fit_path,
        radar_h5=str(radar_h5),
        keypoint_member=keypoint_member,
        source_fps=np.float32(args.source_fps),
        source17=source_keypoints,
        source17_fit_order=fit_keypoints,
        fitted17=fitted_np,
        fitted17_fit_order=fitted_fit_order,
        vertices=vertices,
        faces=faces,
        global_orient=global_orient.detach().cpu().numpy(),
        body_pose=body_pose.detach().cpu().numpy(),
        betas=betas.detach().cpu().numpy(),
        transl=transl.detach().cpu().numpy(),
        target_axes=axes_np,
    )

    face_stride = max(1, int(math.ceil(len(faces) / args.max_render_faces)))
    render_faces = faces[::face_stride]
    display_vertices = display_coordinates(vertices)
    display_source = display_coordinates(source_keypoints)
    display_fitted = display_coordinates(fitted_np)
    render_contact_sheet(
        contact_path,
        display_vertices,
        render_faces,
        display_source,
        display_fitted,
        args.source_fps,
        float(errors.mean()) * 1000.0,
    )
    gif_frame_ids = np.zeros(0, dtype=np.int64)
    if not args.no_gif:
        gif_frame_ids = render_gif(
            gif_path,
            display_vertices,
            render_faces,
            display_source,
            display_fitted,
            args.source_fps,
            args.gif_fps,
        )

    metrics = {
        "method": "code/SMPL_fit v9 adapted from RT-Pose-15 to MVDoppler-17",
        "stored_joint_semantics": "1:3 right leg, 4:6 left leg, 11:13 left arm, 14:16 right arm",
        "source_lr_interpretation": args.source_lr,
        "source_lr_pairs": LEFT_RIGHT_PAIRS.tolist(),
        "uses_keypoints17_to_smplx_module": False,
        "body_model": "SMPL-X male (available model; SMPL PKL payload is absent)",
        "radar_h5": str(radar_h5),
        "keypoint_archive": str(archive),
        "keypoint_member": keypoint_member,
        "joint_names": list(JOINT_NAMES),
        "num_frames": num_frames,
        "source_fps": args.source_fps,
        "iters_scale": args.iters_scale,
        "stages": [
            {**asdict(stage), "effective_iters": max(1, int(round(stage.iters * args.iters_scale)))}
            for stage in V9_STAGES
        ],
        "mpjpe_m": float(errors.mean()),
        "per_joint_mpjpe_m": {
            name: float(errors[:, index].mean()) for index, name in enumerate(JOINT_NAMES)
        },
        "per_frame_mpjpe_m": errors.mean(axis=1).tolist(),
        "pa_mpjpe_not_computed": True,
        "max_joint_error_m": float(errors.max()),
        "mean_joint_velocity_error_m_per_frame": float(velocity_errors.mean()) if velocity_errors.size else 0.0,
        "mean_joint_acceleration_error_m_per_frame2": float(acceleration_errors.mean()) if acceleration_errors.size else 0.0,
        "final_loss_parts": final_parts,
        "history": history,
        "render_face_stride": face_stride,
        "gif_frame_ids": gif_frame_ids.tolist(),
        "outputs": {
            "fit_npz": str(fit_path),
            "contact_png": str(contact_path),
            "animation_gif": str(gif_path) if not args.no_gif else None,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="ascii")
    print(f"MPJPE={errors.mean() * 1000.0:.3f} mm", flush=True)
    print(f"wrote {fit_path}", flush=True)
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {contact_path}", flush=True)
    if not args.no_gif:
        print(f"wrote {gif_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
