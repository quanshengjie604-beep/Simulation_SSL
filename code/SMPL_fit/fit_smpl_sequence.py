#!/usr/bin/env python3
"""Fit a whole RT-Pose sequence with SMPL, without camera priors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

import smpl_fit_common as single


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SequenceStage:
    name: str
    iters: int
    lr: float
    optimize_pose: bool
    optimize_betas: bool
    w_keypoint: float
    w_orientation: float
    w_angle: float
    w_shape: float
    w_pose: float
    w_anatomy: float
    w_surface: float
    w_articulation: float
    w_temporal_pose: float
    w_temporal_trans: float
    w_motion: float
    w_joint_velocity: float
    w_axis_velocity: float
    w_joint_speed_excess: float
    w_joint_accel: float
    w_pose_speed_barrier: float
    w_pose_accel_barrier: float
    w_prediction: float
    w_bone_length: float
    w_leg_plane: float
    w_lower_body_side: float
    w_foot_forward: float = 0.0


STAGES = (
    SequenceStage(
        name="stage1_global",
        iters=220,
        lr=3e-2,
        optimize_pose=False,
        optimize_betas=False,
        w_keypoint=1.0,
        w_orientation=0.12,
        w_angle=0.0,
        w_shape=0.0,
        w_pose=0.0,
        w_anatomy=0.0,
        w_surface=0.0,
        w_articulation=0.0,
        w_temporal_pose=0.0,
        w_temporal_trans=0.10,
        w_motion=0.02,
        w_joint_velocity=0.0,
        w_axis_velocity=0.0,
        w_joint_speed_excess=0.0,
        w_joint_accel=0.0,
        w_pose_speed_barrier=0.0,
        w_pose_accel_barrier=0.0,
        w_prediction=0.0,
        w_bone_length=0.35,
        w_leg_plane=0.0,
        w_lower_body_side=0.0,
    ),
    SequenceStage(
        name="stage2_pose",
        iters=700,
        lr=1e-2,
        optimize_pose=True,
        optimize_betas=False,
        w_keypoint=1.0,
        w_orientation=0.08,
        w_angle=0.25,
        w_shape=0.0,
        w_pose=0.003,
        w_anatomy=0.15,
        w_surface=0.04,
        w_articulation=0.05,
        w_temporal_pose=0.006,
        w_temporal_trans=0.08,
        w_motion=0.01,
        w_joint_velocity=2.50,
        w_axis_velocity=0.10,
        w_joint_speed_excess=4.00,
        w_joint_accel=1.80,
        w_pose_speed_barrier=0.18,
        w_pose_accel_barrier=0.08,
        w_prediction=0.30,
        w_bone_length=0.70,
        w_leg_plane=1.00,
        w_lower_body_side=5.00,
    ),
    SequenceStage(
        name="stage3_shape_refine",
        iters=320,
        lr=3e-3,
        optimize_pose=True,
        optimize_betas=True,
        w_keypoint=1.0,
        w_orientation=0.08,
        w_angle=0.25,
        w_shape=0.08,
        w_pose=0.003,
        w_anatomy=0.15,
        w_surface=0.04,
        w_articulation=0.05,
        w_temporal_pose=0.006,
        w_temporal_trans=0.08,
        w_motion=0.01,
        w_joint_velocity=2.50,
        w_axis_velocity=0.10,
        w_joint_speed_excess=4.00,
        w_joint_accel=1.80,
        w_pose_speed_barrier=0.18,
        w_pose_accel_barrier=0.08,
        w_prediction=0.30,
        w_bone_length=0.70,
        w_leg_plane=1.00,
        w_lower_body_side=5.00,
    ),
)


@dataclass(frozen=True)
class SequenceLabels:
    frame_keys: list[str]
    radar_frames: list[str]
    keypoints: np.ndarray
    is_keyframe: np.ndarray
    keyframe_frames: list[str]
    keyframe_radar_frames: list[str]
    keyframe_keypoints: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default=str(REPO_ROOT / "datasets" / "Train.json"))
    parser.add_argument("--sequence", default="185")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--gender", default="male", choices=("neutral", "male", "female"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "SMPL_fit" / "sequence_fit"))
    parser.add_argument("--batch-sequences", nargs="+", default=[], help="Run v9 over multiple sequences.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute outputs even if metrics already exist.")
    parser.add_argument("--no-summarize", action="store_true", help="Do not write mpjpe_summary.csv/md in batch mode.")
    parser.add_argument("--iters-scale", type=float, default=1.0)
    parser.add_argument("--start-index", type=int, default=0, help="Debug only. Start index in the sorted annotated sequence.")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug only. 0 means all frames.")
    parser.add_argument("--gif-fps", type=float, default=8.0)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--init-fit", default="", help="Optional previous sequence fit NPZ used to initialize parameters.")
    parser.add_argument("--tag", default="velmatch_strict", help="Output tag used in file names.")
    parser.add_argument("--axis-mode", default="raw", choices=("raw", "smooth"))
    parser.add_argument("--axis-alpha", type=float, default=0.45)
    parser.add_argument(
        "--keypoint-interp",
        default="linear",
        choices=("linear",),
        help="Interpolate RT-Pose keypoints to every radar frame before fitting SMPL.",
    )
    parser.add_argument(
        "--no-keypoint-densify",
        action="store_true",
        help="Debug/ablation only. Fit only annotated keypoint frames, matching the legacy sparse strategy.",
    )
    parser.add_argument("--disable-jump-prior", action="store_true")
    parser.add_argument("--jump-axis-rate-deg", type=float, default=55.0)
    parser.add_argument("--jump-arm-diff-threshold", type=float, default=0.08)
    parser.add_argument("--jump-orientation-scale", type=float, default=0.15)
    parser.add_argument("--jump-surface-scale", type=float, default=0.15)
    parser.add_argument("--disable-lower-body-repair", action="store_true")
    parser.add_argument(
        "--preset",
        default="v9",
        choices=("v9", "v10"),
        help=(
            "Loss-weight preset. v9 is the selected motion-foot constrained fitter; "
            "v10 adds conservative temporal regularization for Doppler-oriented sequences."
        ),
    )
    parser.add_argument(
        "--gif-style",
        default="both",
        choices=("mesh", "pointcloud", "both"),
        help="Rendering style for the diagnostic GIF.",
    )
    parser.add_argument("--gif-vertex-stride", type=int, default=6, help="SMPL vertex stride for point-cloud GIF rendering.")
    parser.add_argument("--gif-dpi", type=int, default=150, help="DPI for diagnostic GIF frames.")
    return parser.parse_args()


def stages_for_preset(preset: str) -> tuple[SequenceStage, ...]:
    if preset == "v9":
        return (
            replace(
                STAGES[0],
                iters=180,
                lr=8e-3,
                w_keypoint=4.0,
                w_orientation=0.0,
                w_angle=0.0,
                w_shape=0.0,
                w_pose=0.0,
                w_anatomy=0.0,
                w_surface=0.0,
                w_articulation=0.0,
                w_temporal_pose=0.0,
                w_temporal_trans=0.0,
                w_motion=0.0,
                w_joint_velocity=0.0,
                w_axis_velocity=0.0,
                w_joint_speed_excess=0.0,
                w_joint_accel=0.0,
                w_pose_speed_barrier=0.0,
                w_pose_accel_barrier=0.0,
                w_prediction=0.0,
                w_bone_length=0.05,
                w_leg_plane=0.0,
                w_lower_body_side=0.0,
                w_foot_forward=0.40,
            ),
            replace(
                STAGES[1],
                iters=1350,
                lr=4e-3,
                w_keypoint=8.0,
                w_orientation=0.0,
                w_angle=0.0,
                w_shape=0.0,
                w_pose=0.0003,
                w_anatomy=0.0,
                w_surface=0.0,
                w_articulation=0.0,
                w_temporal_pose=0.0,
                w_temporal_trans=0.0,
                w_motion=0.0,
                w_joint_velocity=0.25,
                w_axis_velocity=0.0,
                w_joint_speed_excess=0.0,
                w_joint_accel=0.08,
                w_pose_speed_barrier=0.0,
                w_pose_accel_barrier=0.0,
                w_prediction=0.0,
                w_bone_length=0.10,
                w_leg_plane=0.0,
                w_lower_body_side=0.35,
                w_foot_forward=3.00,
            ),
            replace(
                STAGES[2],
                iters=750,
                lr=1.5e-3,
                w_keypoint=8.0,
                w_orientation=0.0,
                w_angle=0.0,
                w_shape=0.01,
                w_pose=0.0003,
                w_anatomy=0.0,
                w_surface=0.0,
                w_articulation=0.0,
                w_temporal_pose=0.0,
                w_temporal_trans=0.0,
                w_motion=0.0,
                w_joint_velocity=0.20,
                w_axis_velocity=0.0,
                w_joint_speed_excess=0.0,
                w_joint_accel=0.06,
                w_pose_speed_barrier=0.0,
                w_pose_accel_barrier=0.0,
                w_prediction=0.0,
                w_bone_length=0.10,
                w_leg_plane=0.0,
                w_lower_body_side=0.35,
                w_foot_forward=3.00,
            ),
        )
    if preset == "v10":
        return (
            replace(
                STAGES[0],
                iters=220,
                lr=7e-3,
                w_keypoint=4.0,
                w_orientation=0.0,
                w_angle=0.0,
                w_shape=0.0,
                w_pose=0.0,
                w_anatomy=0.0,
                w_surface=0.0,
                w_articulation=0.0,
                w_temporal_pose=0.0,
                w_temporal_trans=0.02,
                w_motion=0.0,
                w_joint_velocity=0.0,
                w_axis_velocity=0.0,
                w_joint_speed_excess=0.0,
                w_joint_accel=0.0,
                w_pose_speed_barrier=0.0,
                w_pose_accel_barrier=0.0,
                w_prediction=0.0,
                w_bone_length=0.05,
                w_leg_plane=0.0,
                w_lower_body_side=0.0,
                w_foot_forward=0.40,
            ),
            replace(
                STAGES[1],
                iters=1450,
                lr=3.5e-3,
                w_keypoint=8.0,
                w_orientation=0.0,
                w_angle=0.0,
                w_shape=0.0,
                w_pose=0.0004,
                w_anatomy=0.0,
                w_surface=0.0,
                w_articulation=0.0,
                w_temporal_pose=0.003,
                w_temporal_trans=0.03,
                w_motion=0.0,
                w_joint_velocity=0.40,
                w_axis_velocity=0.03,
                w_joint_speed_excess=1.00,
                w_joint_accel=0.14,
                w_pose_speed_barrier=0.06,
                w_pose_accel_barrier=0.025,
                w_prediction=0.08,
                w_bone_length=0.10,
                w_leg_plane=0.0,
                w_lower_body_side=0.35,
                w_foot_forward=3.00,
            ),
            replace(
                STAGES[2],
                iters=850,
                lr=1.2e-3,
                w_keypoint=8.0,
                w_orientation=0.0,
                w_angle=0.0,
                w_shape=0.01,
                w_pose=0.0004,
                w_anatomy=0.0,
                w_surface=0.0,
                w_articulation=0.0,
                w_temporal_pose=0.004,
                w_temporal_trans=0.04,
                w_motion=0.0,
                w_joint_velocity=0.35,
                w_axis_velocity=0.04,
                w_joint_speed_excess=1.20,
                w_joint_accel=0.12,
                w_pose_speed_barrier=0.08,
                w_pose_accel_barrier=0.035,
                w_prediction=0.10,
                w_bone_length=0.10,
                w_leg_plane=0.0,
                w_lower_body_side=0.35,
                w_foot_forward=3.00,
            ),
        )

    raise ValueError(f"unknown preset: {preset}")


def interpolate_keypoints_by_radar_frame(
    frame_keys: list[str],
    radar_frames: list[str],
    keypoints: np.ndarray,
) -> SequenceLabels:
    radar_ids = np.asarray([int(frame) for frame in radar_frames], dtype=np.int32)
    order = np.argsort(radar_ids)
    radar_ids = radar_ids[order]
    keypoints = keypoints[order].astype(np.float32, copy=False)
    keyframe_frames = [frame_keys[int(idx)] for idx in order]
    keyframe_radar_frames = [radar_frames[int(idx)] for idx in order]

    if np.any(np.diff(radar_ids) <= 0):
        raise ValueError(f"Radar_frameID values must be unique and increasing after sorting: {radar_ids.tolist()}")

    dense_ids = np.arange(int(radar_ids[0]), int(radar_ids[-1]) + 1, dtype=np.int32)
    flat = keypoints.reshape(len(keypoints), -1)
    dense_flat = np.empty((len(dense_ids), flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        dense_flat[:, col] = np.interp(dense_ids, radar_ids, flat[:, col]).astype(np.float32)
    dense_keypoints = dense_flat.reshape((len(dense_ids),) + keypoints.shape[1:])
    is_keyframe = np.isin(dense_ids, radar_ids)
    dense_frames = [f"{int(frame):06d}" for frame in dense_ids]
    return SequenceLabels(
        frame_keys=dense_frames,
        radar_frames=dense_frames,
        keypoints=dense_keypoints,
        is_keyframe=is_keyframe,
        keyframe_frames=keyframe_frames,
        keyframe_radar_frames=keyframe_radar_frames,
        keyframe_keypoints=keypoints.copy(),
    )


def load_sequence(
    train_json: Path,
    sequence: str,
    start_index: int,
    max_frames: int,
    *,
    densify_keypoints: bool,
) -> SequenceLabels:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    seq = data[sequence]
    frame_keys = [key for key in sorted(seq, key=lambda x: int(x)) if seq[key]]
    frame_keys = frame_keys[start_index:]
    if max_frames > 0:
        frame_keys = frame_keys[:max_frames]
    radar_frames = []
    poses = []
    for key in frame_keys:
        obj = seq[key][0]
        radar_frames.append(str(obj["Radar_frameID"]).zfill(6))
        poses.append(single.rtpose_to_world(np.asarray(obj["pose"], dtype=np.float32)))
    keypoints = np.stack(poses, axis=0)
    if densify_keypoints:
        return interpolate_keypoints_by_radar_frame(frame_keys, radar_frames, keypoints)
    return SequenceLabels(
        frame_keys=frame_keys,
        radar_frames=radar_frames,
        keypoints=keypoints,
        is_keyframe=np.ones((len(frame_keys),), dtype=bool),
        keyframe_frames=frame_keys,
        keyframe_radar_frames=radar_frames,
        keyframe_keypoints=keypoints.copy(),
    )


def sequence_motion_targets(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pelvis = keypoints[:, 0].copy()
    velocity = np.zeros_like(pelvis)
    velocity[1:-1] = pelvis[2:] - pelvis[:-2]
    velocity[0] = pelvis[1] - pelvis[0]
    velocity[-1] = pelvis[-1] - pelvis[-2]
    velocity[:, 1] = 0.0
    speed = np.linalg.norm(velocity, axis=1)
    motion = velocity.copy()
    good = speed > 0.02
    motion[good] /= speed[good, None]
    motion[~good] = 0.0
    return motion.astype(np.float32), good.astype(np.float32)


def normalized_frame_dt(frame_keys: list[str]) -> np.ndarray:
    frame_ids = np.asarray([int(key) for key in frame_keys], dtype=np.float32)
    if len(frame_ids) < 2:
        return np.zeros((0,), dtype=np.float32)
    diffs = np.diff(frame_ids)
    nominal = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
    return (diffs / max(nominal, 1.0)).astype(np.float32)


def smooth_sequence_axes_np(raw_axes: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Build a continuous body-orientation prior from joint-label axes only."""
    if len(raw_axes) == 0:
        return raw_axes
    smoothed = np.empty_like(raw_axes)
    up = single.safe_normalize_np(raw_axes[0, :, 1])
    front = raw_axes[0, :, 2] - up * float(np.dot(raw_axes[0, :, 2], up))
    front = single.safe_normalize_np(front)
    left = single.safe_normalize_np(np.cross(up, front))
    smoothed[0] = np.stack([left, up, front], axis=1)
    for idx in range(1, len(raw_axes)):
        raw_up = single.safe_normalize_np(raw_axes[idx, :, 1])
        up = single.safe_normalize_np((1.0 - alpha) * up + alpha * raw_up)

        raw_front = raw_axes[idx, :, 2]
        raw_front = raw_front - up * float(np.dot(raw_front, up))
        raw_front = single.safe_normalize_np(raw_front)
        if float(np.dot(raw_front, front)) < 0.0:
            raw_front = -raw_front

        front = single.safe_normalize_np((1.0 - alpha) * front + alpha * raw_front)
        front = single.safe_normalize_np(front - up * float(np.dot(front, up)))
        left = single.safe_normalize_np(np.cross(up, front))
        smoothed[idx] = np.stack([left, up, front], axis=1)
    return smoothed.astype(np.float32)


def sequence_axes_np(keypoints: np.ndarray, mode: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    raw_axes = np.stack([single.body_axes_np(kpts, desired_front=None) for kpts in keypoints], axis=0)
    if mode == "raw":
        return raw_axes, raw_axes
    return smooth_sequence_axes_np(raw_axes, alpha=alpha), raw_axes


def rotation_angle_np(rotation: np.ndarray) -> float:
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cos_theta)


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)).astype(np.float32)


def matrix_to_quat_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float32,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-8)) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float32,
            )
        elif axis == 1:
            scale = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-8)) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float32,
            )
        else:
            scale = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-8)) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float32,
            )
    return quat / max(float(np.linalg.norm(quat)), 1e-8)


def quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_slerp_np(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-8)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-8)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        quat = (1.0 - alpha) * q0 + alpha * q1
        return quat / max(float(np.linalg.norm(quat)), 1e-8)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    return (math.sin(theta_0 - theta) / sin_theta_0 * q0 + math.sin(theta) / sin_theta_0 * q1).astype(np.float32)


def interpolate_global_np(prev_global: np.ndarray, next_global: np.ndarray) -> np.ndarray:
    return interpolate_axis_angle_np(prev_global, next_global, 0.5)


def interpolate_axis_angle_np(prev_global: np.ndarray, next_global: np.ndarray, alpha: float) -> np.ndarray:
    prev_quat = matrix_to_quat_np(axis_angle_to_matrix_np(prev_global))
    next_quat = matrix_to_quat_np(axis_angle_to_matrix_np(next_global))
    return single.matrix_to_axis_angle_np(quat_to_matrix_np(quat_slerp_np(prev_quat, next_quat, alpha)))


def build_jump_prior_np(
    raw_axes: np.ndarray,
    keypoints: np.ndarray,
    frame_dt: np.ndarray,
    init_global: np.ndarray,
    init_body_pose: np.ndarray,
    init_transl: np.ndarray,
    init_joints: np.ndarray | None,
    axis_rate_threshold_deg: float,
    arm_diff_threshold: float,
    orientation_scale: float,
    surface_scale: float,
) -> dict[str, np.ndarray | list[dict[str, float | int | str]]]:
    n_frames = len(keypoints)
    unreliable = np.zeros(n_frames, dtype=bool)
    transition_records: list[dict[str, float | int | str]] = []
    arm_idx = np.array([9, 10, 11, 12, 13, 14], dtype=np.int64)
    dt = np.maximum(frame_dt.astype(np.float32), 1.0)

    for idx in range(1, n_frames):
        step_dt = float(dt[idx - 1]) if len(dt) else 1.0
        axis_angle_deg = math.degrees(rotation_angle_np(raw_axes[idx - 1].T @ raw_axes[idx]))
        axis_rate_deg = axis_angle_deg / step_dt
        gt_delta = (keypoints[idx, arm_idx] - keypoints[idx - 1, arm_idx]) / step_dt
        gt_arm_speed = float(np.linalg.norm(gt_delta, axis=-1).mean())
        arm_diff = 0.0
        arm_diff_max = 0.0
        if init_joints is not None:
            fit_delta = (init_joints[idx, arm_idx] - init_joints[idx - 1, arm_idx]) / step_dt
            diff = np.linalg.norm(fit_delta - gt_delta, axis=-1)
            arm_diff = float(diff.mean())
            arm_diff_max = float(diff.max())

        axis_jump = axis_rate_deg > axis_rate_threshold_deg and gt_arm_speed < 0.28
        arm_jump = init_joints is not None and (arm_diff > arm_diff_threshold or arm_diff_max > arm_diff_threshold * 2.25)
        if not (axis_jump or arm_jump):
            continue

        mark_index = idx
        if init_joints is not None:
            err_prev = float(np.linalg.norm(init_joints[idx - 1, arm_idx] - keypoints[idx - 1, arm_idx], axis=-1).mean())
            err_curr = float(np.linalg.norm(init_joints[idx, arm_idx] - keypoints[idx, arm_idx], axis=-1).mean())
            mark_index = idx - 1 if err_prev >= err_curr else idx
        unreliable[mark_index] = True
        transition_records.append(
            {
                "transition_start": idx - 1,
                "transition_end": idx,
                "marked_frame": mark_index,
                "axis_angle_deg": axis_angle_deg,
                "axis_rate_deg": axis_rate_deg,
                "gt_arm_speed_m": gt_arm_speed,
                "arm_velocity_diff_m": arm_diff,
                "max_arm_velocity_diff_m": arm_diff_max,
                "reason": "axis+arm" if axis_jump and arm_jump else "axis" if axis_jump else "arm",
            }
        )

    pred_global = init_global.copy()
    pred_body_pose = init_body_pose.copy()
    pred_transl = init_transl.copy()
    for idx in np.flatnonzero(unreliable):
        prev_idx = idx - 1
        while prev_idx >= 0 and unreliable[prev_idx]:
            prev_idx -= 1
        next_idx = idx + 1
        while next_idx < n_frames and unreliable[next_idx]:
            next_idx += 1
        if prev_idx >= 0 and next_idx < n_frames:
            pred_global[idx] = interpolate_global_np(init_global[prev_idx], init_global[next_idx])
            pred_body_pose[idx] = 0.5 * (init_body_pose[prev_idx] + init_body_pose[next_idx])
            pred_transl[idx] = 0.5 * (init_transl[prev_idx] + init_transl[next_idx])
        elif prev_idx >= 0:
            pred_global[idx] = init_global[prev_idx]
            pred_body_pose[idx] = init_body_pose[prev_idx]
            pred_transl[idx] = init_transl[prev_idx]
        elif next_idx < n_frames:
            pred_global[idx] = init_global[next_idx]
            pred_body_pose[idx] = init_body_pose[next_idx]
            pred_transl[idx] = init_transl[next_idx]

    orientation_weights = np.ones(n_frames, dtype=np.float32)
    surface_weights = np.ones(n_frames, dtype=np.float32)
    orientation_weights[unreliable] = orientation_scale
    surface_weights[unreliable] = surface_scale
    return {
        "mask": unreliable.astype(np.float32),
        "orientation_weights": orientation_weights,
        "surface_weights": surface_weights,
        "pred_global": pred_global.astype(np.float32),
        "pred_body_pose": pred_body_pose.astype(np.float32),
        "pred_transl": pred_transl.astype(np.float32),
        "transitions": transition_records,
    }


def huber_mean(x, delta: float = 0.05):
    return single.huber(x, delta=delta).mean()


def temporal_accel_loss(x):
    if x.shape[0] < 3:
        return x.new_tensor(0.0)
    return huber_mean(x[2:] - 2.0 * x[1:-1] + x[:-2])


def temporal_velocity_loss(x):
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    return huber_mean(x[1:] - x[:-1])


def axis_angle_to_matrix_t(axis_angle):
    angle = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(axis_angle.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(axis_angle.shape[:-1] + (3, 3))
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    matrix = eye + sin * skew + (1.0 - cos) * (skew @ skew)
    return torch.where((angle[..., None] < 1e-8), eye, matrix)


def so3_angle_between_t(current_axis_angle, target_axis_angle):
    current = axis_angle_to_matrix_t(current_axis_angle)
    target = axis_angle_to_matrix_t(target_axis_angle)
    relative = target.transpose(-1, -2) @ current
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return cos_theta.acos()


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def surface_orientation_prior_weighted(vertices, target_axes, surface_ids, frame_weights):
    target_front = target_axes[:, :, 2]

    def front_vector(name: str):
        front = vertices[:, surface_ids[f"{name}_front"]].mean(dim=1)
        back = vertices[:, surface_ids[f"{name}_back"]].mean(dim=1)
        return single.safe_normalize_t(front - back)

    head_front = front_vector("head")
    right_hand_front = front_vector("right_hand")
    left_hand_front = front_vector("left_hand")
    right_foot_front = front_vector("right_foot")
    left_foot_front = front_vector("left_foot")

    head = 1.0 - (head_front * target_front).sum(dim=-1)
    right_hand = single.relu(0.15 - (right_hand_front * target_front).sum(dim=-1)).square()
    left_hand = single.relu(0.15 - (left_hand_front * target_front).sum(dim=-1)).square()
    right_foot = single.relu(0.60 - (right_foot_front * target_front).sum(dim=-1)).square()
    left_foot = single.relu(0.60 - (left_foot_front * target_front).sum(dim=-1)).square()
    per_frame = head + 0.5 * (right_hand + left_hand) + right_foot + left_foot
    return weighted_mean(per_frame, frame_weights)


def foot_forward_loss(vertices, target_axes, surface_ids, motion_front, motion_mask, max_angle_deg: float = 45.0):
    target_front = target_axes[:, :, 2].detach()
    motion_norm = motion_front.norm(dim=-1)
    reliable_motion = (motion_mask > 0.5) & (motion_norm > 0.5)
    if bool(reliable_motion.any()):
        direction = single.safe_normalize_t(motion_front.detach())
        frame_mask = reliable_motion
    else:
        direction = target_front
        frame_mask = torch.ones(target_front.shape[0], dtype=torch.bool, device=target_front.device)
    min_dot = math.cos(math.radians(max_angle_deg))

    def front_vector(name: str):
        front = vertices[:, surface_ids[f"{name}_front"]].mean(dim=1)
        back = vertices[:, surface_ids[f"{name}_back"]].mean(dim=1)
        return single.safe_normalize_t(front - back)

    right_dot = (front_vector("right_foot") * direction).sum(dim=-1)[frame_mask]
    left_dot = (front_vector("left_foot") * direction).sum(dim=-1)[frame_mask]
    return 0.5 * (single.relu(min_dot - right_dot).square().mean() + single.relu(min_dot - left_dot).square().mean())


def prediction_prior_loss(global_orient, body_pose, transl, pred_global, pred_body_pose, pred_transl, pred_mask):
    if pred_mask.sum() < 0.5:
        return global_orient.new_tensor(0.0)
    global_angle = so3_angle_between_t(global_orient, pred_global).square()
    transl_term = single.huber(transl - pred_transl, delta=0.03).sum(dim=-1)
    pose_term = single.huber(body_pose - pred_body_pose, delta=0.05).mean(dim=-1)
    per_frame = 0.30 * global_angle + 2.0 * transl_term + 0.45 * pose_term
    return (per_frame * pred_mask).sum() / pred_mask.sum().clamp_min(1.0)


def bone_length_match_loss(joints, target):
    terms = []
    for a, b in single.RT_BONES:
        pred_len = (joints[:, a] - joints[:, b]).norm(dim=-1)
        target_len = (target[:, a] - target[:, b]).norm(dim=-1)
        terms.append(single.huber(pred_len - target_len, delta=0.03).mean())
    return torch.stack(terms).mean()


def leg_plane_loss(joints, target):
    terms = []
    for hip_idx, knee_idx, ankle_idx in ((1, 2, 3), (4, 5, 6)):
        pred_leg = joints[:, ankle_idx] - joints[:, hip_idx]
        pred_knee = joints[:, knee_idx] - joints[:, hip_idx]
        target_leg = target[:, ankle_idx] - target[:, hip_idx]
        target_knee = target[:, knee_idx] - target[:, hip_idx]

        pred_axis = single.safe_normalize_t(pred_leg)
        target_axis = single.safe_normalize_t(target_leg)
        pred_side = pred_knee - (pred_knee * pred_axis).sum(dim=-1, keepdim=True) * pred_axis
        target_side = target_knee - (target_knee * target_axis).sum(dim=-1, keepdim=True) * target_axis
        pred_side_norm = pred_side.norm(dim=-1)
        target_side_norm = target_side.norm(dim=-1)
        reliable = (pred_side_norm > 0.01) & (target_side_norm > 0.01)
        if bool(reliable.any()):
            side_dot = (single.safe_normalize_t(pred_side) * single.safe_normalize_t(target_side)).sum(dim=-1)
            terms.append((single.relu(0.90 - side_dot[reliable]).square()).mean())
    if not terms:
        return joints.new_tensor(0.0)
    return torch.stack(terms).mean()


def lower_body_side_loss(joints, target):
    across = single.safe_normalize_t(target[:, 1] - target[:, 4]).detach()
    pred_pelvis = joints[:, 0]
    target_pelvis = target[:, 0]
    terms = []
    weights = []
    for right_idx, left_idx, pair_weight, min_sep_frac in ((1, 4, 6.0, 0.70), (2, 5, 2.0, 0.55), (3, 6, 1.0, 0.45)):
        target_sep = ((target[:, right_idx] - target[:, left_idx]) * across).sum(dim=-1)
        pred_sep = ((joints[:, right_idx] - joints[:, left_idx]) * across).sum(dim=-1)
        target_sep_abs = target_sep.abs()
        min_sep = torch.clamp(min_sep_frac * target_sep_abs, min=0.035)
        sep_match = single.huber(pred_sep - target_sep, delta=0.04)
        sep_barrier = single.relu(min_sep - pred_sep).square()

        pred_right_side = ((joints[:, right_idx] - pred_pelvis) * across).sum(dim=-1)
        pred_left_side = ((joints[:, left_idx] - pred_pelvis) * across).sum(dim=-1)
        target_right_side = ((target[:, right_idx] - target_pelvis) * across).sum(dim=-1)
        target_left_side = ((target[:, left_idx] - target_pelvis) * across).sum(dim=-1)
        side_match = single.huber(pred_right_side - target_right_side, delta=0.04) + single.huber(
            pred_left_side - target_left_side, delta=0.04
        )
        side_margin = torch.clamp(0.30 * target_sep_abs, min=0.022)
        side_barrier = single.relu(side_margin - pred_right_side).square() + single.relu(side_margin + pred_left_side).square()
        terms.append(pair_weight * (sep_match + sep_barrier + 0.50 * side_match + side_barrier).mean())
        weights.append(joints.new_tensor(pair_weight))
    return torch.stack(terms).sum() / torch.stack(weights).sum()


def joint_velocity_match_loss(joints, target, velocity_weights, frame_dt):
    if joints.shape[0] < 2:
        return joints.new_tensor(0.0)
    dt = frame_dt[:, None, None].clamp_min(1.0)
    fit_delta = (joints[1:] - joints[:-1]) / dt
    target_delta = (target[1:] - target[:-1]) / dt
    diff = fit_delta - target_delta
    return (single.huber(diff, delta=0.04).sum(dim=-1) * velocity_weights).mean()


def joint_speed_excess_loss(joints, target, velocity_weights, frame_dt):
    if joints.shape[0] < 2:
        return joints.new_tensor(0.0)
    dt = frame_dt[:, None].clamp_min(1.0)
    fit_speed = (joints[1:] - joints[:-1]).norm(dim=-1) / dt
    target_speed = (target[1:] - target[:-1]).norm(dim=-1) / dt
    excess = single.relu(fit_speed - target_speed - 0.025)
    return (excess.square() * velocity_weights).mean()


def joint_accel_match_loss(joints, target, velocity_weights, frame_dt):
    if joints.shape[0] < 3:
        return joints.new_tensor(0.0)
    dt = frame_dt[:, None, None].clamp_min(1.0)
    fit_velocity = (joints[1:] - joints[:-1]) / dt
    target_velocity = (target[1:] - target[:-1]) / dt
    diff = (fit_velocity[1:] - fit_velocity[:-1]) - (target_velocity[1:] - target_velocity[:-1])
    return (single.huber(diff, delta=0.03).sum(dim=-1) * velocity_weights).mean()


def pose_speed_barrier_loss(body_pose, frame_dt):
    if body_pose.shape[0] < 2:
        return body_pose.new_tensor(0.0)
    tracked = (
        (single.SMPL["left_collar"], 0.16, 1.0),
        (single.SMPL["right_collar"], 0.16, 1.0),
        (single.SMPL["left_shoulder"], 0.22, 1.8),
        (single.SMPL["right_shoulder"], 0.22, 1.8),
        (single.SMPL["left_elbow"], 0.28, 1.2),
        (single.SMPL["right_elbow"], 0.28, 1.2),
    )
    dt = frame_dt.clamp_min(1.0)
    terms = []
    for joint_index, threshold, weight in tracked:
        sl = single.body_pose_slice(joint_index)
        speed = (body_pose[1:, sl] - body_pose[:-1, sl]).norm(dim=-1) / dt
        terms.append(weight * single.relu(speed - threshold).square().mean())
    return sum(terms) / len(terms)


def pose_accel_barrier_loss(body_pose, frame_dt):
    if body_pose.shape[0] < 3:
        return body_pose.new_tensor(0.0)
    tracked = (
        (single.SMPL["left_collar"], 0.18, 1.0),
        (single.SMPL["right_collar"], 0.18, 1.0),
        (single.SMPL["left_shoulder"], 0.24, 1.8),
        (single.SMPL["right_shoulder"], 0.24, 1.8),
        (single.SMPL["left_elbow"], 0.32, 1.2),
        (single.SMPL["right_elbow"], 0.32, 1.2),
    )
    dt = frame_dt[:, None].clamp_min(1.0)
    velocity = (body_pose[1:] - body_pose[:-1]) / dt
    terms = []
    for joint_index, threshold, weight in tracked:
        sl = single.body_pose_slice(joint_index)
        accel = (velocity[1:, sl] - velocity[:-1, sl]).norm(dim=-1)
        terms.append(weight * single.relu(accel - threshold).square().mean())
    return sum(terms) / len(terms)


def axis_velocity_match_loss(left, up, front, target_axes):
    if front.shape[0] < 2:
        return front.new_tensor(0.0)
    fit_axes = torch.stack([left, up, front], dim=2)
    fit_delta = fit_axes[1:] - fit_axes[:-1]
    target_delta = target_axes[1:] - target_axes[:-1]
    return single.huber(fit_delta - target_delta, delta=0.02).mean()


def collect_params(global_orient, body_pose, betas, transl, stage: SequenceStage):
    params = [global_orient, transl]
    if stage.optimize_pose:
        params.append(body_pose)
    if stage.optimize_betas:
        params.append(betas)
    for tensor in (global_orient, body_pose, betas, transl):
        tensor.requires_grad_(False)
    for tensor in params:
        tensor.requires_grad_(True)
    return params


def sequence_loss(
    model,
    target,
    target_axes,
    orientation_frame_weights,
    surface_frame_weights,
    weights,
    velocity_weights,
    head_vertex_ids,
    surface_ids,
    target_bend_dirs,
    motion_front,
    motion_mask,
    frame_dt,
    global_orient,
    body_pose,
    betas,
    transl,
    pred_global,
    pred_body_pose,
    pred_transl,
    pred_mask,
    stage: SequenceStage,
):
    betas_seq = betas.expand(target.shape[0], -1)
    out = single.forward_model(model, global_orient, body_pose, betas_seq, transl)
    joints = single.smpl_rtpose_joints(out.joints, out.vertices, head_vertex_ids)

    diff = joints - target
    keypoint = (single.huber(diff).sum(dim=-1) * weights).mean()

    left, up, front = single.body_axes_t(joints)
    target_left, target_up, target_front = target_axes[:, :, 0], target_axes[:, :, 1], target_axes[:, :, 2]
    orientation_per_frame = (
        (1.0 - (left * target_left).sum(dim=-1))
        + (1.0 - (up * target_up).sum(dim=-1))
        + (1.0 - (front * target_front).sum(dim=-1))
    ) / 3.0
    orientation = weighted_mean(orientation_per_frame, orientation_frame_weights)

    angle = single.limb_angle_loss(joints, target_bend_dirs)
    bone_length = bone_length_match_loss(joints, target)
    leg_plane = leg_plane_loss(joints, target)
    lower_body_side = lower_body_side_loss(joints, target)
    shape = betas.square().mean()
    pose = body_pose.square().mean()
    anatomy = single.anatomy_prior(body_pose)
    surface = surface_orientation_prior_weighted(out.vertices, target_axes, surface_ids, surface_frame_weights)
    foot_forward = foot_forward_loss(out.vertices, target_axes, surface_ids, motion_front, motion_mask)
    articulation = single.articulation_prior(body_pose)

    temporal_pose = temporal_velocity_loss(body_pose) + 0.5 * temporal_accel_loss(body_pose)
    temporal_trans = temporal_accel_loss(transl) + 0.25 * temporal_accel_loss(global_orient)
    joint_velocity = joint_velocity_match_loss(joints, target, velocity_weights, frame_dt)
    joint_speed_excess = joint_speed_excess_loss(joints, target, velocity_weights, frame_dt)
    joint_accel = joint_accel_match_loss(joints, target, velocity_weights, frame_dt)
    pose_speed_barrier = pose_speed_barrier_loss(body_pose, frame_dt)
    pose_accel_barrier = pose_accel_barrier_loss(body_pose, frame_dt)
    axis_velocity = axis_velocity_match_loss(left, up, front, target_axes)
    prediction = prediction_prior_loss(global_orient, body_pose, transl, pred_global, pred_body_pose, pred_transl, pred_mask)

    motion_dot = (front * motion_front).sum(dim=-1)
    motion = (single.relu(0.20 - motion_dot).square() * motion_mask).sum() / motion_mask.sum().clamp_min(1.0)

    loss = (
        stage.w_keypoint * keypoint
        + stage.w_orientation * orientation
        + stage.w_angle * angle
        + stage.w_bone_length * bone_length
        + stage.w_leg_plane * leg_plane
        + stage.w_lower_body_side * lower_body_side
        + stage.w_shape * shape
        + stage.w_pose * pose
        + stage.w_anatomy * anatomy
        + stage.w_surface * surface
        + stage.w_foot_forward * foot_forward
        + stage.w_articulation * articulation
        + stage.w_temporal_pose * temporal_pose
        + stage.w_temporal_trans * temporal_trans
        + stage.w_motion * motion
        + stage.w_joint_velocity * joint_velocity
        + stage.w_axis_velocity * axis_velocity
        + stage.w_joint_speed_excess * joint_speed_excess
        + stage.w_joint_accel * joint_accel
        + stage.w_pose_speed_barrier * pose_speed_barrier
        + stage.w_pose_accel_barrier * pose_accel_barrier
        + stage.w_prediction * prediction
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "keypoint": float(keypoint.detach().cpu()),
        "orientation": float(orientation.detach().cpu()),
        "angle": float(angle.detach().cpu()),
        "bone_length": float(bone_length.detach().cpu()),
        "leg_plane": float(leg_plane.detach().cpu()),
        "lower_body_side": float(lower_body_side.detach().cpu()),
        "shape": float(shape.detach().cpu()),
        "pose": float(pose.detach().cpu()),
        "anatomy": float(anatomy.detach().cpu()),
        "surface": float(surface.detach().cpu()),
        "foot_forward": float(foot_forward.detach().cpu()),
        "articulation": float(articulation.detach().cpu()),
        "temporal_pose": float(temporal_pose.detach().cpu()),
        "temporal_trans": float(temporal_trans.detach().cpu()),
        "motion": float(motion.detach().cpu()),
        "joint_velocity": float(joint_velocity.detach().cpu()),
        "keypoint_velocity": float(joint_velocity.detach().cpu()),
        "axis_velocity": float(axis_velocity.detach().cpu()),
        "joint_speed_excess": float(joint_speed_excess.detach().cpu()),
        "joint_accel": float(joint_accel.detach().cpu()),
        "keypoint_accel": float(joint_accel.detach().cpu()),
        "pose_speed_barrier": float(pose_speed_barrier.detach().cpu()),
        "pose_accel_barrier": float(pose_accel_barrier.detach().cpu()),
        "prediction": float(prediction.detach().cpu()),
    }
    return loss, parts, out, joints


def make_gif(
    plt,
    out_gif: Path,
    keypoints,
    smpl_joints,
    vertices,
    faces,
    frame_keys,
    fps: float,
    *,
    style: str = "mesh",
    vertex_stride: int = 10,
    dpi: int = 150,
) -> None:
    from PIL import Image
    import matplotlib.tri as mtri

    def mesh_foot_vertex_mask(frame_vertices: np.ndarray, frame_joints: np.ndarray) -> np.ndarray:
        mask = np.zeros(frame_vertices.shape[0], dtype=bool)
        for ankle_idx in (3, 6):
            ankle = frame_joints[ankle_idx]
            dist = np.linalg.norm(frame_vertices - ankle[None], axis=1)
            low = frame_vertices[:, 1] <= ankle[1] + 0.08
            mask |= (dist < 0.30) & low
        return mask

    frame_dir = out_gif.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    point_stride = max(1, int(vertex_stride))
    body_points = np.concatenate([keypoints, smpl_joints, vertices[:, ::point_stride]], axis=1)
    body_span = np.max(body_points.max(axis=1) - body_points.min(axis=1), axis=1)
    radius = max(1.15, 0.62 * float(np.percentile(body_span, 90)) + 0.20)
    mesh_faces = faces[:: max(1, len(faces) // 1300)]
    paths = []
    for idx, frame in enumerate(frame_keys):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        foot_mask = mesh_foot_vertex_mask(vertices[idx], smpl_joints[idx])
        vp = vertices[idx][:, [0, 2, 1]]
        if style in ("mesh", "both"):
            visible_faces = mesh_faces[~foot_mask[mesh_faces].any(axis=1)]
            triangulation = mtri.Triangulation(vp[:, 0], vp[:, 1], triangles=visible_faces)
            ax.plot_trisurf(
                triangulation,
                vp[:, 2],
                color="#b8c0cc",
                alpha=0.22 if style == "both" else 0.32,
                linewidth=0.04,
                edgecolor="#64748b",
            )
        if style in ("pointcloud", "both"):
            point_ids = np.arange(0, len(vp), point_stride)
            point_ids = point_ids[~foot_mask[point_ids]]
            cloud = vp[point_ids]
            ax.scatter(
                cloud[:, 0],
                cloud[:, 1],
                cloud[:, 2],
                c="#475569",
                s=4.2 if style == "pointcloud" else 3.0,
                alpha=0.72 if style == "pointcloud" else 0.55,
                depthshade=False,
            )
        kp = keypoints[idx]
        sj = smpl_joints[idx]
        ax.scatter(kp[:, 0], kp[:, 2], kp[:, 1], c="#dc2626", s=34, depthshade=False)
        ax.scatter(sj[:, 0], sj[:, 2], sj[:, 1], c="#2563eb", s=24, depthshade=False)
        for a, b in single.RT_BONES:
            ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 2], kp[b, 2]], [kp[a, 1], kp[b, 1]], c="#ef4444", linewidth=2.0)
            ax.plot([sj[a, 0], sj[b, 0]], [sj[a, 2], sj[b, 2]], [sj[a, 1], sj[b, 1]], c="#2563eb", linewidth=1.6)
        center = smpl_joints[idx, 0]
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=18, azim=-62)
        ax.set_title(f"sequence {frame_keys[0]}..{frame_keys[-1]}  frame {frame}")
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        path = frame_dir / f"{idx:04d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        paths.append(path)

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in paths]
    duration_ms = int(round(1000.0 / fps))
    images[0].save(out_gif, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=False)


def lower_body_metrics_np(joints: np.ndarray, target: np.ndarray) -> dict[str, float]:
    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    across = normalize(target[:, 1] - target[:, 4])
    metrics: dict[str, float] = {}
    for name, right_idx, left_idx in (("hip", 1, 4), ("knee", 2, 5), ("ankle", 3, 6)):
        target_sep = ((target[:, right_idx] - target[:, left_idx]) * across).sum(axis=-1)
        pred_sep = ((joints[:, right_idx] - joints[:, left_idx]) * across).sum(axis=-1)
        metrics[f"{name}_side_mismatch_rate"] = float((np.sign(pred_sep) != np.sign(target_sep)).mean())
        metrics[f"{name}_pred_sep_mean_m"] = float(pred_sep.mean())
        metrics[f"{name}_target_sep_mean_m"] = float(target_sep.mean())

    for side, hip_idx, knee_idx, ankle_idx in (("right", 1, 2, 3), ("left", 4, 5, 6)):
        pred_leg = joints[:, ankle_idx] - joints[:, hip_idx]
        pred_knee = joints[:, knee_idx] - joints[:, hip_idx]
        target_leg = target[:, ankle_idx] - target[:, hip_idx]
        target_knee = target[:, knee_idx] - target[:, hip_idx]
        pred_axis = normalize(pred_leg)
        target_axis = normalize(target_leg)
        pred_side = pred_knee - (pred_knee * pred_axis).sum(axis=-1, keepdims=True) * pred_axis
        target_side = target_knee - (target_knee * target_axis).sum(axis=-1, keepdims=True) * target_axis
        reliable = (np.linalg.norm(pred_side, axis=-1) > 0.01) & (np.linalg.norm(target_side, axis=-1) > 0.01)
        if reliable.any():
            cosine = (normalize(pred_side[reliable]) * normalize(target_side[reliable])).sum(axis=-1)
            metrics[f"{side}_leg_plane_cos_mean"] = float(cosine.mean())
            metrics[f"{side}_leg_plane_negative_rate"] = float((cosine < 0.0).mean())
        else:
            metrics[f"{side}_leg_plane_cos_mean"] = 0.0
            metrics[f"{side}_leg_plane_negative_rate"] = 0.0
    return metrics


def lower_body_crossing_mask_np(joints: np.ndarray, target: np.ndarray) -> np.ndarray:
    across = target[:, 1] - target[:, 4]
    across = across / np.maximum(np.linalg.norm(across, axis=-1, keepdims=True), 1e-8)
    hip_sep = ((joints[:, 1] - joints[:, 4]) * across).sum(axis=-1)
    knee_sep = ((joints[:, 2] - joints[:, 5]) * across).sum(axis=-1)
    ankle_sep = ((joints[:, 3] - joints[:, 6]) * across).sum(axis=-1)
    target_hip_sep = ((target[:, 1] - target[:, 4]) * across).sum(axis=-1)
    target_knee_sep = ((target[:, 2] - target[:, 5]) * across).sum(axis=-1)
    target_ankle_sep = ((target[:, 3] - target[:, 6]) * across).sum(axis=-1)
    hip_crossed = hip_sep < 0.0 * np.maximum(target_hip_sep, 1e-6)
    knee_ok = knee_sep > 0.25 * np.maximum(target_knee_sep, 1e-6)
    ankle_ok = ankle_sep > 0.20 * np.maximum(target_ankle_sep, 1e-6)
    return hip_crossed & knee_ok & ankle_ok


def repair_crossed_lower_body_params_np(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    crossing_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    repaired = np.flatnonzero(crossing_mask)
    if len(repaired) == 0 or len(repaired) == len(crossing_mask):
        return global_orient, body_pose, transl, []

    out_global = global_orient.copy()
    out_body = body_pose.copy()
    out_transl = transl.copy()
    good = ~crossing_mask
    n_frames = len(crossing_mask)
    for idx in repaired:
        prev_idx = idx - 1
        while prev_idx >= 0 and not good[prev_idx]:
            prev_idx -= 1
        next_idx = idx + 1
        while next_idx < n_frames and not good[next_idx]:
            next_idx += 1

        if prev_idx >= 0 and next_idx < n_frames:
            alpha = float(idx - prev_idx) / float(next_idx - prev_idx)
            out_global[idx] = interpolate_axis_angle_np(global_orient[prev_idx], global_orient[next_idx], alpha)
            out_body[idx] = (1.0 - alpha) * body_pose[prev_idx] + alpha * body_pose[next_idx]
            out_transl[idx] = (1.0 - alpha) * transl[prev_idx] + alpha * transl[next_idx]
        elif prev_idx >= 0:
            out_global[idx] = global_orient[prev_idx]
            out_body[idx] = body_pose[prev_idx]
            out_transl[idx] = transl[prev_idx]
        elif next_idx < n_frames:
            out_global[idx] = global_orient[next_idx]
            out_body[idx] = body_pose[next_idx]
            out_transl[idx] = transl[next_idx]
    return out_global, out_body, out_transl, [int(i) for i in repaired]


def fit_single_sequence(args: argparse.Namespace) -> dict[str, object]:
    torch, smplx, plt, _, _ = single.import_runtime()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    stages = stages_for_preset(args.preset)

    labels = load_sequence(
        Path(args.train_json),
        args.sequence,
        args.start_index,
        args.max_frames,
        densify_keypoints=not args.no_keypoint_densify,
    )
    frame_keys = labels.frame_keys
    radar_frames = labels.radar_frames
    keypoints_np = labels.keypoints
    is_keyframe_np = labels.is_keyframe.astype(bool, copy=False)
    n_frames = len(frame_keys)
    axes_np, raw_axes_np = sequence_axes_np(keypoints_np, mode=args.axis_mode, alpha=args.axis_alpha)
    motion_np, motion_mask_np = sequence_motion_targets(keypoints_np)
    frame_dt_np = normalized_frame_dt(radar_frames)

    model = smplx.create(model_path=args.model_root, model_type="smpl", gender=args.gender, batch_size=n_frames).to(device)
    model.eval()
    head_vertex_ids = torch.topk(model.v_template[:, 1], k=40).indices.to(device=device)
    surface_ids = single.surface_orientation_ids(model, device)

    target = torch.as_tensor(keypoints_np, dtype=torch.float32, device=device)
    target_axes = torch.as_tensor(axes_np, dtype=torch.float32, device=device)
    target_bend_dirs = single.target_limb_bend_dirs(target, target_axes[:, :, 2])
    weights = torch.tensor(
        [1.6, 2.8, 1.2, 0.9, 2.8, 1.2, 0.9, 1.4, 1.1, 1.2, 0.9, 0.7, 1.2, 0.9, 0.7],
        dtype=torch.float32,
        device=device,
    )
    velocity_weights = torch.tensor(
        [0.3, 0.4, 0.7, 1.0, 0.4, 0.7, 1.0, 0.4, 0.6, 7.0, 5.0, 4.0, 7.0, 5.0, 4.0],
        dtype=torch.float32,
        device=device,
    )
    motion_front = torch.as_tensor(motion_np, dtype=torch.float32, device=device)
    motion_mask = torch.as_tensor(motion_mask_np, dtype=torch.float32, device=device)
    frame_dt = torch.as_tensor(frame_dt_np, dtype=torch.float32, device=device)

    if args.init_fit:
        init = np.load(args.init_fit)
        if len(init["frames"]) != n_frames:
            raise ValueError(f"--init-fit has {len(init['frames'])} frames, expected {n_frames}")
        init_global = init["global_orient"].astype(np.float32)
        init_body_pose = init["body_pose"].astype(np.float32)
        init_betas = init["betas"].astype(np.float32)
        init_transl = init["transl"].astype(np.float32)
        init_joints = init["smpl_joints"].astype(np.float32) if "smpl_joints" in init.files else None
    else:
        init_global = np.stack([single.matrix_to_axis_angle_np(axes) for axes in axes_np], axis=0)
        init_body_pose = np.zeros((n_frames, 69), dtype=np.float32)
        init_betas = np.zeros((1, 10), dtype=np.float32)
        init_transl = keypoints_np[:, 0].astype(np.float32)
        init_joints = None
    if args.disable_jump_prior:
        jump_prior = {
            "mask": np.zeros(n_frames, dtype=np.float32),
            "orientation_weights": np.ones(n_frames, dtype=np.float32),
            "surface_weights": np.ones(n_frames, dtype=np.float32),
            "pred_global": init_global.astype(np.float32),
            "pred_body_pose": init_body_pose.astype(np.float32),
            "pred_transl": init_transl.astype(np.float32),
            "transitions": [],
        }
    else:
        jump_prior = build_jump_prior_np(
            raw_axes_np,
            keypoints_np,
            frame_dt_np,
            init_global,
            init_body_pose,
            init_transl,
            init_joints,
            axis_rate_threshold_deg=args.jump_axis_rate_deg,
            arm_diff_threshold=args.jump_arm_diff_threshold,
            orientation_scale=args.jump_orientation_scale,
            surface_scale=args.jump_surface_scale,
        )
    print(
        f"jump_prior frames={int(np.asarray(jump_prior['mask']).sum())} "
        f"transitions={len(jump_prior['transitions'])} disabled={args.disable_jump_prior}",
        flush=True,
    )
    global_orient = torch.tensor(init_global, dtype=torch.float32, device=device)
    body_pose = torch.tensor(init_body_pose, dtype=torch.float32, device=device)
    betas = torch.tensor(init_betas, dtype=torch.float32, device=device)
    transl = torch.tensor(init_transl, dtype=torch.float32, device=device)
    orientation_frame_weights = torch.as_tensor(jump_prior["orientation_weights"], dtype=torch.float32, device=device)
    surface_frame_weights = torch.as_tensor(jump_prior["surface_weights"], dtype=torch.float32, device=device)
    pred_global = torch.as_tensor(jump_prior["pred_global"], dtype=torch.float32, device=device)
    pred_body_pose = torch.as_tensor(jump_prior["pred_body_pose"], dtype=torch.float32, device=device)
    pred_transl = torch.as_tensor(jump_prior["pred_transl"], dtype=torch.float32, device=device)
    pred_mask = torch.as_tensor(jump_prior["mask"], dtype=torch.float32, device=device)

    history = []
    for stage in stages:
        stage_iters = max(1, int(round(stage.iters * args.iters_scale)))
        params = collect_params(global_orient, body_pose, betas, transl, stage)
        optim = torch.optim.Adam(params, lr=stage.lr)
        for it in range(stage_iters):
            optim.zero_grad(set_to_none=True)
            loss, parts, _, _ = sequence_loss(
                model,
                target,
                target_axes,
                orientation_frame_weights,
                surface_frame_weights,
                weights,
                velocity_weights,
                head_vertex_ids,
                surface_ids,
                target_bend_dirs,
                motion_front,
                motion_mask,
                frame_dt,
                global_orient,
                body_pose,
                betas,
                transl,
                pred_global,
                pred_body_pose,
                pred_transl,
                pred_mask,
                stage,
            )
            loss.backward()
            optim.step()
            if stage.optimize_pose:
                with torch.no_grad():
                    single.zero_locked_body_pose_(body_pose)
            if stage.optimize_betas:
                with torch.no_grad():
                    betas.clamp_(-2.0, 2.0)
            if it == 0 or it == stage_iters - 1 or (it + 1) % 50 == 0:
                print(f"{stage.name} {it + 1:04d}/{stage_iters}: {parts}", flush=True)
                history.append({"stage": stage.name, "iter": it + 1, **parts})

    with torch.no_grad():
        _, parts, out, joints = sequence_loss(
            model,
            target,
            target_axes,
            orientation_frame_weights,
            surface_frame_weights,
            weights,
            velocity_weights,
            head_vertex_ids,
            surface_ids,
            target_bend_dirs,
            motion_front,
            motion_mask,
            frame_dt,
            global_orient,
            body_pose,
            betas,
            transl,
            pred_global,
            pred_body_pose,
            pred_transl,
            pred_mask,
            stages[-1],
        )
        smpl_j = single.smpl_rtpose_joints(out.joints, out.vertices, head_vertex_ids).detach().cpu().numpy()
        vertices = out.vertices.detach().cpu().numpy()

    original_global_np = global_orient.detach().cpu().numpy()
    original_body_np = body_pose.detach().cpu().numpy()
    original_transl_np = transl.detach().cpu().numpy()
    original_smpl_j = smpl_j.copy()
    lower_body_repair_mask = lower_body_crossing_mask_np(smpl_j, keypoints_np)
    (
        repaired_global_np,
        repaired_body_np,
        repaired_transl_np,
        lower_body_repaired_indices,
    ) = repair_crossed_lower_body_params_np(
        global_orient.detach().cpu().numpy(),
        body_pose.detach().cpu().numpy(),
        transl.detach().cpu().numpy(),
        lower_body_repair_mask,
    )
    if args.disable_lower_body_repair:
        lower_body_repaired_indices = []
        rejected_indices = []
    elif lower_body_repaired_indices:
        with torch.no_grad():
            global_orient.copy_(torch.as_tensor(repaired_global_np, dtype=torch.float32, device=device))
            body_pose.copy_(torch.as_tensor(repaired_body_np, dtype=torch.float32, device=device))
            transl.copy_(torch.as_tensor(repaired_transl_np, dtype=torch.float32, device=device))
            _, parts, out, joints = sequence_loss(
                model,
                target,
                target_axes,
                orientation_frame_weights,
                surface_frame_weights,
                weights,
                velocity_weights,
                head_vertex_ids,
                surface_ids,
                target_bend_dirs,
                motion_front,
                motion_mask,
                frame_dt,
                global_orient,
                body_pose,
                betas,
                transl,
                pred_global,
                pred_body_pose,
                pred_transl,
                pred_mask,
                stages[-1],
            )
            smpl_j = single.smpl_rtpose_joints(out.joints, out.vertices, head_vertex_ids).detach().cpu().numpy()
            vertices = out.vertices.detach().cpu().numpy()
        original_errors = np.linalg.norm(original_smpl_j - keypoints_np, axis=-1)
        repaired_errors = np.linalg.norm(smpl_j - keypoints_np, axis=-1)
        accepted_indices = []
        rejected_indices = []
        for idx in lower_body_repaired_indices:
            original_mpjpe = float(original_errors[idx].mean())
            repaired_mpjpe = float(repaired_errors[idx].mean())
            original_max = float(original_errors[idx].max())
            repaired_max = float(repaired_errors[idx].max())
            improves_or_close = repaired_mpjpe <= original_mpjpe + 0.035 and repaired_max <= original_max + 0.12
            if improves_or_close:
                accepted_indices.append(idx)
            else:
                rejected_indices.append(idx)
                repaired_global_np[idx] = original_global_np[idx]
                repaired_body_np[idx] = original_body_np[idx]
                repaired_transl_np[idx] = original_transl_np[idx]
        if rejected_indices:
            with torch.no_grad():
                global_orient.copy_(torch.as_tensor(repaired_global_np, dtype=torch.float32, device=device))
                body_pose.copy_(torch.as_tensor(repaired_body_np, dtype=torch.float32, device=device))
                transl.copy_(torch.as_tensor(repaired_transl_np, dtype=torch.float32, device=device))
                _, parts, out, joints = sequence_loss(
                    model,
                    target,
                    target_axes,
                    orientation_frame_weights,
                    surface_frame_weights,
                    weights,
                    velocity_weights,
                    head_vertex_ids,
                    surface_ids,
                    target_bend_dirs,
                    motion_front,
                    motion_mask,
                    frame_dt,
                    global_orient,
                    body_pose,
                    betas,
                    transl,
                    pred_global,
                    pred_body_pose,
                    pred_transl,
                    pred_mask,
                    stages[-1],
                )
                smpl_j = single.smpl_rtpose_joints(out.joints, out.vertices, head_vertex_ids).detach().cpu().numpy()
                vertices = out.vertices.detach().cpu().numpy()
        lower_body_repaired_indices = accepted_indices
    else:
        rejected_indices = []

    errors = np.linalg.norm(smpl_j - keypoints_np, axis=-1)
    keyframe_errors = errors[is_keyframe_np]
    interpolated_errors = errors[~is_keyframe_np]
    per_frame_mpjpe = errors.mean(axis=1)
    keyframe_per_frame_mpjpe = per_frame_mpjpe[is_keyframe_np]
    interpolated_per_frame_mpjpe = per_frame_mpjpe[~is_keyframe_np]
    if n_frames > 1:
        fit_velocity = smpl_j[1:] - smpl_j[:-1]
        target_velocity = keypoints_np[1:] - keypoints_np[:-1]
        velocity_error = np.linalg.norm(fit_velocity - target_velocity, axis=-1)
        arm_idx = np.array([9, 10, 11, 12, 13, 14])
        limb_idx = np.array([2, 3, 5, 6, 10, 11, 13, 14])
    else:
        velocity_error = np.zeros((0, 15), dtype=np.float32)
        arm_idx = np.array([9, 10, 11, 12, 13, 14])
        limb_idx = np.array([2, 3, 5, 6, 10, 11, 13, 14])
    lower_body_metrics = lower_body_metrics_np(smpl_j, keypoints_np)
    stage_weights = [
        {
            "stage": stage.name,
            "iters": stage.iters,
            "base_iters": stage.iters,
            "effective_iters": max(1, int(round(stage.iters * args.iters_scale))),
            "lr": stage.lr,
            "w_keypoint": stage.w_keypoint,
            "w_orientation": stage.w_orientation,
            "w_angle": stage.w_angle,
            "w_shape": stage.w_shape,
            "w_pose": stage.w_pose,
            "w_anatomy": stage.w_anatomy,
            "w_surface": stage.w_surface,
            "w_foot_forward": stage.w_foot_forward,
            "w_articulation": stage.w_articulation,
            "w_temporal_trans": stage.w_temporal_trans,
            "w_motion": stage.w_motion,
            "w_temporal_pose": stage.w_temporal_pose,
            "w_joint_velocity": stage.w_joint_velocity,
            "w_keypoint_velocity": stage.w_joint_velocity,
            "w_joint_speed_excess": stage.w_joint_speed_excess,
            "w_joint_accel": stage.w_joint_accel,
            "w_keypoint_accel": stage.w_joint_accel,
            "w_pose_speed_barrier": stage.w_pose_speed_barrier,
            "w_pose_accel_barrier": stage.w_pose_accel_barrier,
            "w_prediction": stage.w_prediction,
            "w_bone_length": stage.w_bone_length,
            "w_leg_plane": stage.w_leg_plane,
            "w_lower_body_side": stage.w_lower_body_side,
            "w_axis_velocity": stage.w_axis_velocity,
        }
        for stage in stages
    ]
    metrics = {
        "sequence": args.sequence,
        "n_frames": n_frames,
        "n_keyframes": int(is_keyframe_np.sum()),
        "n_interpolated_frames": int((~is_keyframe_np).sum()),
        "frames": frame_keys,
        "radar_frames": radar_frames,
        "is_keyframe": [bool(x) for x in is_keyframe_np],
        "keyframe_frames": labels.keyframe_frames,
        "keyframe_radar_frames": labels.keyframe_radar_frames,
        "keypoint_densify": {
            "enabled": not args.no_keypoint_densify,
            "interp": args.keypoint_interp,
            "source_keyframes": int(len(labels.keyframe_radar_frames)),
            "first_radar_frame": int(radar_frames[0]),
            "last_radar_frame": int(radar_frames[-1]),
        },
        "mpjpe_m": float(errors.mean()),
        "keyframe_mpjpe_m": float(keyframe_errors.mean()) if keyframe_errors.size else 0.0,
        "interpolated_mpjpe_m": float(interpolated_errors.mean()) if interpolated_errors.size else 0.0,
        "max_joint_error_m": float(errors.max()),
        "keyframe_max_joint_error_m": float(keyframe_errors.max()) if keyframe_errors.size else 0.0,
        "interpolated_max_joint_error_m": float(interpolated_errors.max()) if interpolated_errors.size else 0.0,
        "per_frame_mpjpe_m": [float(x) for x in per_frame_mpjpe],
        "keyframe_per_frame_mpjpe_m": [float(x) for x in keyframe_per_frame_mpjpe],
        "interpolated_per_frame_mpjpe_m": [float(x) for x in interpolated_per_frame_mpjpe],
        "mean_joint_velocity_error_m": float(velocity_error.mean()) if len(velocity_error) else 0.0,
        "mean_arm_velocity_error_m": float(velocity_error[:, arm_idx].mean()) if len(velocity_error) else 0.0,
        "mean_limb_velocity_error_m": float(velocity_error[:, limb_idx].mean()) if len(velocity_error) else 0.0,
        "max_joint_velocity_error_m": float(velocity_error.max()) if len(velocity_error) else 0.0,
        "lower_body_metrics": lower_body_metrics,
        "lower_body_postprocess": {
            "enabled": True,
            "disabled_by_arg": args.disable_lower_body_repair,
            "n_repaired_frames": len(lower_body_repaired_indices),
            "repaired_frame_indices": lower_body_repaired_indices,
            "repaired_frames": [frame_keys[idx] for idx in lower_body_repaired_indices],
            "n_rejected_frames": len(rejected_indices),
            "rejected_frame_indices": rejected_indices,
            "rejected_frames": [frame_keys[idx] for idx in rejected_indices],
            "rule": "repair frames whose fitted hip side is crossed while knee and ankle sides remain valid",
        },
        "front_source": f"joint-labels-{args.axis_mode}",
        "camera_prior_used": False,
        "preset": args.preset,
        "iters_scale": args.iters_scale,
        "init_fit": args.init_fit,
        "motion_frames": int(motion_mask_np.sum()),
        "jump_prior": {
            "enabled": not args.disable_jump_prior,
            "n_unreliable_frames": int(np.asarray(jump_prior["mask"]).sum()),
            "unreliable_frames": [
                frame_keys[idx] for idx in np.flatnonzero(np.asarray(jump_prior["mask"], dtype=np.float32) > 0.5)
            ],
            "axis_rate_threshold_deg": args.jump_axis_rate_deg,
            "arm_diff_threshold_m": args.jump_arm_diff_threshold,
            "orientation_scale": args.jump_orientation_scale,
            "surface_scale": args.jump_surface_scale,
            "transitions": jump_prior["transitions"],
        },
        "temporal_priors": {
            "target_axes": f"joint-labels-{args.axis_mode}",
            "target_axes_alpha": args.axis_alpha if args.axis_mode == "smooth" else None,
            "joint_velocity_match": any(stage.w_joint_velocity > 0.0 for stage in stages),
            "axis_velocity_match": any(stage.w_axis_velocity > 0.0 for stage in stages),
            "joint_speed_excess": any(stage.w_joint_speed_excess > 0.0 for stage in stages),
            "joint_acceleration_match": any(stage.w_joint_accel > 0.0 for stage in stages),
            "pose_speed_barrier": any(stage.w_pose_speed_barrier > 0.0 for stage in stages),
            "pose_accel_barrier": any(stage.w_pose_accel_barrier > 0.0 for stage in stages),
            "velocity_weights": [float(x) for x in velocity_weights.detach().cpu().numpy()],
            "frame_dt_normalized": [float(x) for x in frame_dt_np],
            "stage_weights": stage_weights,
            "camera_prior_used": False,
        },
        "gif": {
            "style": args.gif_style,
            "vertex_stride": args.gif_vertex_stride,
            "dpi": args.gif_dpi,
        },
        "final_loss_parts": parts,
        "history": history,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seq{args.sequence}_joint_labels_temporal_{args.tag}"
    out_json = out_dir / f"{stem}_metrics.json"
    out_npz = out_dir / f"{stem}_fit.npz"
    out_gif = out_dir / f"{stem}.gif"
    faces = np.asarray(model.faces, dtype=np.int32)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_npz,
        frames=np.asarray(frame_keys),
        radar_frames=np.asarray(radar_frames),
        is_keyframe=is_keyframe_np,
        vertices=vertices,
        faces=faces,
        smpl_joints=smpl_j,
        rtpose_world_keypoints=keypoints_np,
        keyframe_frames=np.asarray(labels.keyframe_frames),
        keyframe_radar_frames=np.asarray(labels.keyframe_radar_frames),
        keyframe_rtpose_world_keypoints=labels.keyframe_keypoints,
        target_axes=axes_np,
        raw_target_axes=raw_axes_np,
        jump_prior_mask=np.asarray(jump_prior["mask"], dtype=np.float32),
        jump_orientation_weights=np.asarray(jump_prior["orientation_weights"], dtype=np.float32),
        jump_surface_weights=np.asarray(jump_prior["surface_weights"], dtype=np.float32),
        pred_global_orient=np.asarray(jump_prior["pred_global"], dtype=np.float32),
        pred_body_pose=np.asarray(jump_prior["pred_body_pose"], dtype=np.float32),
        pred_transl=np.asarray(jump_prior["pred_transl"], dtype=np.float32),
        motion_front=motion_np,
        motion_mask=motion_mask_np,
        frame_dt=frame_dt_np,
        global_orient=global_orient.detach().cpu().numpy(),
        body_pose=body_pose.detach().cpu().numpy(),
        betas=betas.detach().cpu().numpy(),
        transl=transl.detach().cpu().numpy(),
        preset=np.asarray(args.preset),
    )
    if not args.no_gif:
        make_gif(
            plt,
            out_gif,
            keypoints_np,
            smpl_j,
            vertices,
            faces,
            frame_keys,
            args.gif_fps,
            style=args.gif_style,
            vertex_stride=args.gif_vertex_stride,
            dpi=args.gif_dpi,
        )
        print(f"Wrote GIF: {out_gif}")
    print(f"Wrote metrics: {out_json}")
    print(f"Wrote fit: {out_npz}")
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "n_frames",
                    "n_keyframes",
                    "n_interpolated_frames",
                    "mpjpe_m",
                    "keyframe_mpjpe_m",
                    "max_joint_error_m",
                    "motion_frames",
                )
            },
            indent=2,
        )
    )
    return {
        "metrics": metrics,
        "metrics_path": out_json,
        "fit_path": out_npz,
        "gif_path": out_gif if not args.no_gif else None,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[int(math.ceil(q * len(ordered)) - 1)]


def summarize_fit_metrics(metrics_dir: Path, pattern: str, out_csv: Path, out_md: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    all_per_frame: list[float] = []
    for path in sorted(metrics_dir.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        per_frame = [float(x) for x in data.get("keyframe_per_frame_mpjpe_m", data.get("per_frame_mpjpe_m", []))]
        all_per_frame.extend(per_frame)
        n_frames = int(data["n_frames"])
        n_keyframes = int(data.get("n_keyframes", n_frames))
        rows.append(
            {
                "sequence": str(data["sequence"]),
                "preset": str(data.get("preset", "")),
                "n_frames": n_frames,
                "n_keyframes": n_keyframes,
                "n_interpolated_frames": int(data.get("n_interpolated_frames", 0)),
                "dense_mpjpe_mm": float(data["mpjpe_m"]) * 1000.0,
                "mpjpe_mm": float(data.get("keyframe_mpjpe_m", data["mpjpe_m"])) * 1000.0,
                "per_frame_median_mm": percentile(per_frame, 0.5) * 1000.0,
                "per_frame_p95_mm": percentile(per_frame, 0.95) * 1000.0,
                "max_joint_error_mm": float(data.get("keyframe_max_joint_error_m", data["max_joint_error_m"])) * 1000.0,
                "motion_frames": int(data.get("motion_frames", 0)),
                "n_unreliable_frames": int(data.get("jump_prior", {}).get("n_unreliable_frames", 0)),
                "n_repaired_frames": int(data.get("lower_body_postprocess", {}).get("n_repaired_frames", 0)),
                "n_rejected_frames": int(data.get("lower_body_postprocess", {}).get("n_rejected_frames", 0)),
                "file": str(path),
            }
        )

    total_frames = sum(int(row["n_frames"]) for row in rows)
    total_keyframes = sum(int(row["n_keyframes"]) for row in rows)
    frame_weighted = sum(float(row["mpjpe_mm"]) * int(row["n_keyframes"]) for row in rows) / max(total_keyframes, 1)
    sequence_mean = sum(float(row["mpjpe_mm"]) for row in rows) / max(len(rows), 1)
    overall = {
        "num_sequences": len(rows),
        "total_frames": total_frames,
        "total_keyframes": total_keyframes,
        "keyframe_weighted_mpjpe_mm": frame_weighted,
        "sequence_mean_mpjpe_mm": sequence_mean,
        "per_frame_median_mm": percentile(all_per_frame, 0.5) * 1000.0,
        "per_frame_p95_mm": percentile(all_per_frame, 0.95) * 1000.0,
    }

    fields = [
        "sequence",
        "preset",
        "n_frames",
        "n_keyframes",
        "n_interpolated_frames",
        "mpjpe_mm",
        "dense_mpjpe_mm",
        "per_frame_median_mm",
        "per_frame_p95_mm",
        "max_joint_error_mm",
        "motion_frames",
        "n_unreliable_frames",
        "n_repaired_frames",
        "n_rejected_frames",
        "file",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SMPL fit MPJPE summary",
        "",
        f"- Sequences: {overall['num_sequences']}",
        f"- Total frames: {overall['total_frames']}",
        f"- Total keyframes: {overall['total_keyframes']}",
        f"- Keyframe-weighted MPJPE: {overall['keyframe_weighted_mpjpe_mm']:.3f} mm",
        f"- Sequence-mean MPJPE: {overall['sequence_mean_mpjpe_mm']:.3f} mm",
        f"- Per-frame median: {overall['per_frame_median_mm']:.3f} mm",
        f"- Per-frame P95: {overall['per_frame_p95_mm']:.3f} mm",
        "",
        "| Sequence | Dense frames | Keyframes | Keyframe MPJPE mm | Dense MPJPE mm | Median mm | P95 mm | Max joint mm |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sequence']} | {row['n_frames']} | {row['n_keyframes']} | "
            f"{float(row['mpjpe_mm']):.3f} | {float(row['dense_mpjpe_mm']):.3f} | "
            f"{float(row['per_frame_median_mm']):.3f} | {float(row['per_frame_p95_mm']):.3f} | "
            f"{float(row['max_joint_error_mm']):.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary CSV: {out_csv}")
    print(f"Wrote summary MD: {out_md}")
    return {"overall": overall, "rows": rows}


def clone_args(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def main() -> int:
    args = parse_args()
    if args.batch_sequences:
        for sequence in [str(seq) for seq in args.batch_sequences]:
            sequence_args = clone_args(args, sequence=sequence)
            fit_single_sequence(sequence_args)
        if not args.no_summarize:
            summarize_fit_metrics(
                Path(args.out_dir),
                f"seq*_joint_labels_temporal_{args.tag}_metrics.json",
                Path(args.out_dir) / "mpjpe_summary.csv",
                Path(args.out_dir) / "mpjpe_summary.md",
            )
    else:
        fit_single_sequence(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
