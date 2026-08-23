#!/usr/bin/env python3
"""Fit an mmRadPose skeleton sequence with SMPL-X and render a micro-Doppler spectrum."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = REPO_ROOT / "logs" / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import smplx
import torch

sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code" / "SMPL_fit"))

import m4human_smplx_to_micro_doppler as m4human
import smpl_mesh_to_micro_doppler as pipeline
from fit_mvdoppler_17keypoints import (
    JOINT_NAMES,
    V9_STAGES,
    body_axes_np,
    collect_params,
    display_coordinates,
    foot_surface_ids,
    foot_forward_loss,
    huber,
    matrix_to_axis_angle_np,
    render_contact_sheet,
    render_gif,
)


DEFAULT_SEQUENCE_DIR = (
    REPO_ROOT / "datasets" / "mmRadPose" / "mmRadPose_pointclouds" / "p6" / "angle0" / "7" / "0"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "mmradpose_smplx_micro_doppler"
MMRADPOSE_FPS = 15.0

# mmRadPose stores 26 OMC skeleton points in radar coordinates
# (x=lateral, y=range/depth, z=height).  The 17-point subset is only used
# to initialize body axes and to render a compact overlay; the fit loss uses
# all 26 points.
MMRADPOSE_17_INDICES = {
    "pelvis": 0,
    "right_hip": 6,
    "right_knee": 7,
    "right_foot": 8,
    "left_hip": 1,
    "left_knee": 2,
    "left_foot": 3,
    "spine": 11,
    "thorax": 12,
    "neck": 23,
    "head": 25,
    "left_shoulder": 14,
    "left_elbow": 15,
    "left_wrist": 17,
    "right_shoulder": 19,
    "right_elbow": 20,
    "right_wrist": 22,
}
MMRADPOSE_TO_MVDOPPLER17 = np.asarray(
    [MMRADPOSE_17_INDICES[name] for name in JOINT_NAMES],
    dtype=np.int64,
)

MMRADPOSE26_NAMES = (
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot_front",
    "left_foot_back",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot_front",
    "right_foot_back",
    "spine_low",
    "spine_high",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "neck",
    "head",
    "head_top",
)

MMRADPOSE26_BONES = np.asarray(
    (
        (0, 1), (1, 2), (2, 3), (3, 4), (3, 5),
        (0, 6), (6, 7), (7, 8), (8, 9), (8, 10),
        (0, 11), (11, 12), (12, 23), (23, 24), (24, 25),
        (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
        (12, 18), (18, 19), (19, 20), (20, 21), (21, 22),
    ),
    dtype=np.int64,
)

MMRADPOSE26_WEIGHTS = np.asarray(
    (
        1.8,
        2.5, 1.5, 1.0, 0.7, 0.7,
        2.5, 1.5, 1.0, 0.7, 0.7,
        1.2, 1.4,
        1.2, 2.0, 3.0, 5.0, 6.0,
        1.2, 2.0, 3.0, 5.0, 6.0,
        1.1, 1.0, 0.6,
    ),
    dtype=np.float32,
)

MMRADPOSE26_VELOCITY_WEIGHTS = np.asarray(
    (
        0.4,
        0.5, 0.8, 1.0, 0.8, 0.8,
        0.5, 0.8, 1.0, 0.8, 0.8,
        0.4, 0.5,
        1.0, 7.0, 6.0, 6.0, 6.0,
        1.0, 7.0, 6.0, 6.0, 6.0,
        0.5, 0.6, 0.4,
    ),
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--fit-npz", type=Path, default=None, help="Reuse a saved *_smplfit.npz and run Doppler only.")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "smpl_models")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default="")
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smpl-device", default="", help="Defaults to --device.")
    parser.add_argument("--backend", choices=("dirichlet", "pytorch", "slang"), default="dirichlet")
    parser.add_argument("--radar-profile", choices=tuple(pipeline.RADAR_PROFILES), default=pipeline.RADAR_PROFILE)
    parser.add_argument("--range-bin-mode", choices=("fixed", "tracked"), default="tracked")
    parser.add_argument("--visibility-mode", choices=("linear", "hold"), default="linear")
    parser.add_argument("--doppler-transform", choices=("fft", "nudft"), default="nudft")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=150, help="150 frames is about 10 s at 15 Hz.")
    parser.add_argument("--source-fps", type=float, default=MMRADPOSE_FPS)
    parser.add_argument("--iters-scale", type=float, default=0.20)
    parser.add_argument("--smpl-batch-size", type=int, default=32)
    parser.add_argument("--max-render-faces", type=int, default=4000)
    parser.add_argument("--gif-fps", type=float, default=7.5)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument("--max-radar-frames", type=int, default=0)
    parser.add_argument("--skip-doppler", action="store_true", help="Only run and save the 26-keypoint SMPL-X fit.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sequence_label(sequence_dir: Path, start_frame: int, num_frames: int) -> str:
    parts = sequence_dir.resolve().parts
    try:
        participant, angle, action, recording = parts[-4:]
        return f"mmradpose_{participant}_{angle}_ac{int(action):02d}_r{int(recording):02d}_F{start_frame}_{start_frame + num_frames - 1}"
    except Exception:
        return f"mmradpose_{sequence_dir.name}_F{start_frame}_{start_frame + num_frames - 1}"


def validate_args(args: argparse.Namespace) -> None:
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative")
    if args.num_frames < 2:
        raise ValueError("--num-frames must select at least two frames")
    if args.source_fps <= 0.0:
        raise ValueError("--source-fps must be positive")
    if args.iters_scale <= 0.0:
        raise ValueError("--iters-scale must be positive")
    if args.smpl_batch_size < 1:
        raise ValueError("--smpl-batch-size must be positive")
    if args.max_radar_frames < 0:
        raise ValueError("--max-radar-frames cannot be negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")


def load_mmradpose(sequence_dir: Path, start_frame: int, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    skeleton_path = sequence_dir / "skeleton.npy"
    skeleton = np.load(skeleton_path, allow_pickle=False).astype(np.float32).reshape(-1, 26, 3)
    stop = start_frame + num_frames
    if stop > len(skeleton):
        raise ValueError(f"Requested frames {start_frame}:{stop}, but {skeleton_path} has {len(skeleton)} frames")
    selected26 = skeleton[start_frame:stop]
    selected17 = selected26[:, MMRADPOSE_TO_MVDOPPLER17].copy()
    if not np.isfinite(selected26).all():
        raise ValueError(f"{skeleton_path} contains non-finite values in the selected segment")
    return selected26, selected17


def mmradpose_surface_ids(model, device: torch.device) -> dict[str, torch.Tensor]:
    ids = foot_surface_ids(model, device)
    vertices = model.v_template.detach().cpu().numpy()
    ids["left_hand"] = torch.as_tensor(
        np.where(vertices[:, 0] >= np.quantile(vertices[:, 0], 0.995))[0],
        dtype=torch.long,
        device=device,
    )
    ids["right_hand"] = torch.as_tensor(
        np.where(vertices[:, 0] <= np.quantile(vertices[:, 0], 0.005))[0],
        dtype=torch.long,
        device=device,
    )
    top = vertices[:, 1] >= np.quantile(vertices[:, 1], 0.995)
    ids["head_top"] = torch.as_tensor(np.where(top)[0], dtype=torch.long, device=device)
    return ids


def smpl_mmradpose26_joints(output, surface_ids: dict[str, torch.Tensor]) -> torch.Tensor:
    joints = output.joints
    vertices = output.vertices
    midhip = 0.5 * (joints[:, 1] + joints[:, 2])
    values = (
        midhip,
        joints[:, 1],
        joints[:, 4],
        joints[:, 7],
        vertices[:, surface_ids["left_front"]].mean(dim=1),
        vertices[:, surface_ids["left_back"]].mean(dim=1),
        joints[:, 2],
        joints[:, 5],
        joints[:, 8],
        vertices[:, surface_ids["right_front"]].mean(dim=1),
        vertices[:, surface_ids["right_back"]].mean(dim=1),
        joints[:, 3],
        joints[:, 9],
        joints[:, 13],
        joints[:, 16],
        joints[:, 18],
        joints[:, 20],
        vertices[:, surface_ids["left_hand"]].mean(dim=1),
        joints[:, 14],
        joints[:, 17],
        joints[:, 19],
        joints[:, 21],
        vertices[:, surface_ids["right_hand"]].mean(dim=1),
        joints[:, 12],
        joints[:, 15],
        vertices[:, surface_ids["head_top"]].mean(dim=1),
    )
    return torch.stack(values, dim=1)


def bone_length_loss26(joints: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    edges = torch.as_tensor(MMRADPOSE26_BONES, dtype=torch.long, device=joints.device)
    pred = (joints[:, edges[:, 0]] - joints[:, edges[:, 1]]).norm(dim=-1)
    truth = (target[:, edges[:, 0]] - target[:, edges[:, 1]]).norm(dim=-1)
    return huber(pred - truth, delta=0.03).mean()


def temporal_loss26(
    joints: torch.Tensor,
    target: torch.Tensor,
    velocity_weights: torch.Tensor,
    order: int,
) -> torch.Tensor:
    if len(joints) <= order:
        return joints.new_tensor(0.0)
    pred_delta = torch.diff(joints, n=order, dim=0)
    target_delta = torch.diff(target, n=order, dim=0)
    delta = 0.04 if order == 1 else 0.03
    return (huber(pred_delta - target_delta, delta=delta).sum(dim=-1) * velocity_weights).mean()


def sequence_loss26(
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
    stage,
):
    output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas.expand(len(target), -1),
        transl=transl,
        return_verts=True,
    )
    joints = smpl_mmradpose26_joints(output, surface_ids)
    keypoint = (huber(joints - target).sum(dim=-1) * weights).mean()
    pose = body_pose.square().mean()
    shape = betas.square().mean()
    bone = bone_length_loss26(joints, target)
    velocity = temporal_loss26(joints, target, velocity_weights, order=1)
    acceleration = temporal_loss26(joints, target, velocity_weights, order=2)
    foot_forward = foot_forward_loss(output.vertices, target_axes, surface_ids)
    loss = (
        stage.w_keypoint * keypoint
        + stage.w_pose * pose
        + stage.w_shape * shape
        + stage.w_bone_length * bone
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
        "joint_velocity": float(velocity.detach().cpu()),
        "joint_accel": float(acceleration.detach().cpu()),
        "foot_forward": float(foot_forward.detach().cpu()),
    }
    return loss, parts, output, joints


def fit_smplx(
    source26: np.ndarray,
    source17_for_axes: np.ndarray,
    model_dir: Path,
    gender: str,
    device: torch.device,
    iters_scale: float,
) -> dict[str, object]:
    num_frames = int(source26.shape[0])
    model = smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=gender,
        batch_size=num_frames,
        num_betas=10,
        use_pca=False,
        create_global_orient=False,
        create_body_pose=False,
        create_betas=False,
        create_transl=False,
    ).to(device).eval()
    surface_ids = mmradpose_surface_ids(model, device)
    axes_np = np.stack([body_axes_np(frame) for frame in source17_for_axes]).astype(np.float32)
    target = torch.as_tensor(source26, dtype=torch.float32, device=device)
    target_axes = torch.as_tensor(axes_np, dtype=torch.float32, device=device)
    weights = torch.as_tensor(MMRADPOSE26_WEIGHTS, dtype=torch.float32, device=device)
    velocity_weights = torch.as_tensor(MMRADPOSE26_VELOCITY_WEIGHTS, dtype=torch.float32, device=device)
    global_orient = torch.tensor(
        np.stack([matrix_to_axis_angle_np(axes) for axes in axes_np]),
        dtype=torch.float32,
        device=device,
    )
    body_pose = torch.zeros((num_frames, int(model.NUM_BODY_JOINTS) * 3), dtype=torch.float32, device=device)
    betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
    transl = torch.as_tensor(source26[:, 0], dtype=torch.float32, device=device).clone()
    history: list[dict[str, object]] = []

    for stage in V9_STAGES:
        stage_iters = max(1, int(round(stage.iters * iters_scale)))
        optimizer = torch.optim.Adam(collect_params(global_orient, body_pose, betas, transl, stage), lr=stage.lr)
        for iteration in range(stage_iters):
            optimizer.zero_grad(set_to_none=True)
            loss, parts, _, _ = sequence_loss26(
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
            if stage.optimize_betas:
                with torch.no_grad():
                    betas.clamp_(-2.0, 2.0)
            if iteration == 0 or iteration == stage_iters - 1 or (iteration + 1) % 50 == 0:
                record = {"stage": stage.name, "iter": iteration + 1, **parts}
                history.append(record)
                print(f"{stage.name} {iteration + 1:04d}/{stage_iters}: {parts}", flush=True)

    with torch.no_grad():
        _, final_parts, output, fitted = sequence_loss26(
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
    result = {
        "vertices": output.vertices.detach().cpu().numpy().astype(np.float32),
        "faces": np.asarray(model.faces, dtype=np.int32),
        "fitted26": fitted.detach().cpu().numpy().astype(np.float32),
        "fitted17": fitted.detach().cpu().numpy().astype(np.float32)[:, MMRADPOSE_TO_MVDOPPLER17],
        "global_orient": global_orient.detach().cpu().numpy().astype(np.float32),
        "body_pose": body_pose.detach().cpu().numpy().astype(np.float32),
        "betas": betas.detach().cpu().numpy().astype(np.float32),
        "transl": transl.detach().cpu().numpy().astype(np.float32),
        "target_axes": axes_np,
        "history": history,
        "final_loss_parts": final_parts,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_fit_outputs(
    out_dir: Path,
    label: str,
    args: argparse.Namespace,
    source26: np.ndarray,
    source17: np.ndarray,
    fit: dict[str, object],
) -> tuple[Path, Path, Path | None, dict[str, object]]:
    vertices = fit["vertices"]
    faces = fit["faces"]
    fitted26 = fit["fitted26"]
    fitted17 = fit["fitted17"]
    errors26 = np.linalg.norm(fitted26 - source26, axis=-1)
    errors17 = np.linalg.norm(fitted17 - source17, axis=-1)
    fit_path = out_dir / f"{label}_smplfit.npz"
    metrics_path = out_dir / f"{label}_smplfit.json"
    contact_path = out_dir / f"{label}_smplfit_overlay.png"
    gif_path = None if args.no_gif else out_dir / f"{label}_smplfit_overlay.gif"
    np.savez_compressed(
        fit_path,
        source_sequence_dir=str(args.sequence_dir),
        source_fps=np.float32(args.source_fps),
        mmradpose26=source26,
        source17=source17,
        fitted26=fitted26,
        fitted17=fitted17,
        vertices=vertices,
        faces=faces,
        global_orient=fit["global_orient"],
        body_pose=fit["body_pose"],
        betas=fit["betas"],
        transl=fit["transl"],
        target_axes=fit["target_axes"],
        mmradpose_17_indices=MMRADPOSE_TO_MVDOPPLER17,
    )
    face_stride = max(1, int(math.ceil(len(faces) / args.max_render_faces)))
    render_faces = faces[::face_stride]
    render_contact_sheet(
        contact_path,
        display_coordinates(vertices),
        render_faces,
        display_coordinates(source17),
        display_coordinates(fitted17),
        args.source_fps,
        float(errors17.mean()) * 1000.0,
    )
    gif_frame_ids = np.zeros(0, dtype=np.int64)
    if gif_path is not None:
        gif_frame_ids = render_gif(
            gif_path,
            display_coordinates(vertices),
            render_faces,
            display_coordinates(source17),
            display_coordinates(fitted17),
            args.source_fps,
            args.gif_fps,
        )
    metrics = {
        "method": "mmRadPose all 26 OMC joints -> repository SMPL-fit v9 adaptation with extra SMPL-X foot/hand/head constraints",
        "source_sequence_dir": str(args.sequence_dir),
        "source_fps": float(args.source_fps),
        "num_frames": int(source26.shape[0]),
        "duration_s": float((source26.shape[0] - 1) / args.source_fps),
        "gender": args.gender,
        "joint_names26": list(MMRADPOSE26_NAMES),
        "overlay_joint_names17": list(JOINT_NAMES),
        "mmradpose_17_indices": {name: int(index) for name, index in zip(JOINT_NAMES, MMRADPOSE_TO_MVDOPPLER17)},
        "iters_scale": float(args.iters_scale),
        "stages": [
            {**asdict(stage), "effective_iters": max(1, int(round(stage.iters * args.iters_scale)))}
            for stage in V9_STAGES
        ],
        "mpjpe26_m": float(errors26.mean()),
        "mpjpe17_overlay_m": float(errors17.mean()),
        "per_joint26_mpjpe_m": {name: float(errors26[:, index].mean()) for index, name in enumerate(MMRADPOSE26_NAMES)},
        "per_joint17_overlay_mpjpe_m": {name: float(errors17[:, index].mean()) for index, name in enumerate(JOINT_NAMES)},
        "max_joint26_error_m": float(errors26.max()),
        "max_joint17_overlay_error_m": float(errors17.max()),
        "final_loss_parts": fit["final_loss_parts"],
        "history": fit["history"],
        "outputs": {
            "fit_npz": str(fit_path),
            "contact_png": str(contact_path),
            "animation_gif": str(gif_path) if gif_path is not None else None,
        },
        "gif_frame_ids": gif_frame_ids.tolist(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="ascii")
    return fit_path, metrics_path, gif_path, metrics


def run_micro_doppler(
    label: str,
    args: argparse.Namespace,
    vertices_radar: np.ndarray,
    faces: np.ndarray,
) -> tuple[Path, Path, Path]:
    out_dir = args.out_dir.expanduser().resolve()
    result_path = out_dir / f"{label}_micro_doppler.npz"
    image_path = out_dir / f"{label}_micro_doppler.png"
    summary_path = out_dir / f"{label}_micro_doppler.json"
    existing = [path for path in (result_path, image_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}; pass --overwrite to recompute")

    radar_device = torch.device(args.device)
    world_vertices_np = m4human.m4human_radar_to_witwin(vertices_radar)
    native_vertices = torch.as_tensor(world_vertices_np, dtype=torch.float32, device=radar_device)
    mocap_rate = float(args.source_fps)
    duration_s = (native_vertices.shape[0] - 1) / mocap_rate
    chirp_frequency_hz = pipeline.RADAR_PROFILES[args.radar_profile]
    frame_times, chirp_times_by_frame = pipeline.actual_radar_timing(duration_s, chirp_frequency_hz)
    if args.max_radar_frames:
        frame_times = frame_times[: args.max_radar_frames]
        chirp_times_by_frame = chirp_times_by_frame[: args.max_radar_frames]
    if frame_times.size == 0:
        raise ValueError("Selected mmRadPose segment is too short for one complete radar frame")
    print(
        f"[timing] mmRadPose={native_vertices.shape[0]} frames at {mocap_rate:g} Hz; "
        f"radar={len(frame_times)} complete frames, {chirp_times_by_frame.size} acquired chirps",
        flush=True,
    )

    config = pipeline.radar_config(chirp_frequency_hz)
    derived = pipeline.derived_radar_parameters(config)
    fixed_range_bins = None
    if args.range_bin_mode == "fixed":
        first_center = native_vertices[0].detach().cpu().numpy().mean(axis=0)
        center_range = abs(float(first_center[2]))
        range_bin_count = max(1, int(round(pipeline.RANGE_WINDOW_M / derived["range_resolution_m"])))
        fixed_center_bin = int(round(center_range / derived["range_resolution_m"]))
        fixed_range_bins = pipeline.centered_range_indices(fixed_center_bin, pipeline.ADC_SAMPLES // 2, range_bin_count)
        print(
            f"[range] fixed bins={fixed_range_bins[0]}..{fixed_range_bins[-1]} "
            f"center_range={center_range:.3f}m",
            flush=True,
        )

    Radar, RadarConfig, Scene, Tracer = pipeline.bootstrap_witwin_modules()
    pipeline.apply_radar_equation_patch()
    radar = Radar(
        RadarConfig.from_dict(config),
        backend=args.backend,
        device=args.device,
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        name=f"{args.radar_profile}_mmradpose_smplx",
    )
    scene = Scene(device=args.device)
    scene.add_mesh(name="human", vertices=native_vertices[0].clone(), faces=faces, dynamic=True)
    tracer = Tracer(scene, radar, resolution=1, sampling="triangle", multipath=False, max_reflections=0)
    interpolator = pipeline.FrameLevelScattererInterpolator(
        radar,
        scene,
        tracer,
        native_vertices,
        faces,
        mocap_rate,
        duration_s,
    )
    slow_time_points, chirp_times, strong_ranges, selected_bins = pipeline.simulate_slow_time(
        radar,
        interpolator,
        frame_times,
        chirp_times_by_frame,
        derived,
        fixed_range_bins,
        args.visibility_mode,
    )
    spectrum, stft_times, velocities = pipeline.mean_power_stft(
        slow_time_points,
        chirp_times,
        derived["wavelength_m"],
        chirp_frequency_hz,
        args.doppler_transform,
    )

    visible_per_frame = np.asarray(interpolator.frame_visible_counts, dtype=np.int32)
    np.savez_compressed(
        result_path,
        spectrum=spectrum,
        stft_time_s=stft_times,
        velocity_mps=velocities,
        strong_range_m=strong_ranges,
        radar_frame_time_s=frame_times,
        chirp_time_s=chirp_times,
        visible_triangles_per_frame=visible_per_frame,
        visible_triangles_per_chirp=np.repeat(visible_per_frame, pipeline.CHIRPS_PER_FRAME),
        next_visible_triangles_per_frame=np.asarray(interpolator.frame_next_visible_counts, dtype=np.int32),
        union_visible_triangles_per_frame=np.asarray(interpolator.frame_union_visible_counts, dtype=np.int32),
        selected_range_bins_by_frame=np.asarray(selected_bins, dtype=np.int16),
        mmradpose_source_sequence=str(args.sequence_dir),
        mmradpose_start_frame=np.asarray(args.start_frame, dtype=np.int32),
        mmradpose_num_frames=np.asarray(args.num_frames, dtype=np.int32),
    )
    pipeline.save_plot(image_path, spectrum, stft_times, velocities, label, args.db_floor, chirp_frequency_hz)
    summary = {
        "processing": (
            "mmRadPose OMC skeleton -> repository SMPL-fit v9 adaptation -> "
            "M4Human radar-to-WiTwin coordinate transform -> frame-level visibility -> "
            "per-chirp geometry/intensity -> CUDA MIMO -> STFT"
        ),
        "source_sequence_dir": str(args.sequence_dir),
        "sequence": label,
        "native_motion_frames": int(native_vertices.shape[0]),
        "native_motion_rate_hz": mocap_rate,
        "native_motion_duration_s": duration_s,
        "coordinate_transform": "mmRadPose radar (x=lateral,y=range,z=height) -> WiTwin (x=lateral,y=height,z=-range)",
        "radar": {
            "profile": args.radar_profile,
            "frame_rate_hz": pipeline.FRAME_RATE_HZ,
            "chirp_frequency_hz": chirp_frequency_hz,
            "chirps_per_frame": pipeline.CHIRPS_PER_FRAME,
            "samples_per_chirp": pipeline.ADC_SAMPLES,
            "num_complete_frames": int(frame_times.size),
            "num_acquired_chirps": int(chirp_times.size),
        },
        "ray_tracing": {
            "sampling": "triangle",
            "visibility_mode": args.visibility_mode,
            "triangles_tested_per_frame": int(faces.shape[0]),
            "visible_triangles_min": int(visible_per_frame.min()),
            "visible_triangles_mean": float(visible_per_frame.mean()),
            "visible_triangles_max": int(visible_per_frame.max()),
        },
        "stft": {
            "range_window_m": pipeline.RANGE_WINDOW_M,
            "range_bin_mode": args.range_bin_mode,
            "fixed_range_bins": fixed_range_bins.tolist() if fixed_range_bins is not None else None,
            "window_chirps": pipeline.STFT_WINDOW_CHIRPS,
            "fft_size": pipeline.STFT_FFT_SIZE,
            "hop_chirps": pipeline.STFT_HOP_CHIRPS,
            "doppler_transform": args.doppler_transform,
            "num_columns": int(spectrum.shape[1]),
        },
        "range_resolution_m": derived["range_resolution_m"],
        "strong_range_mean_m": float(np.mean(strong_ranges)),
        "result_npz": str(result_path),
        "image_png": str(image_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2), flush=True)
    return result_path, image_path, summary_path


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.sequence_dir = args.sequence_dir.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.fit_npz is not None:
        args.fit_npz = args.fit_npz.expanduser().resolve()
        with np.load(args.fit_npz, allow_pickle=False) as data:
            vertices = data["vertices"].astype(np.float32)
            faces = data["faces"].astype(np.int32)
            args.source_fps = float(data["source_fps"])
            if "source_sequence_dir" in data.files:
                args.sequence_dir = Path(str(data["source_sequence_dir"].item())).resolve()
        label = args.label or args.fit_npz.name.removesuffix("_smplfit.npz")
        if args.skip_doppler:
            print(f"[fit] loaded {args.fit_npz}; [doppler] skipped by --skip-doppler", flush=True)
            return 0
        run_micro_doppler(label, args, vertices, faces)
        return 0

    label = args.label or sequence_label(args.sequence_dir, args.start_frame, args.num_frames)

    fit_path = args.out_dir / f"{label}_smplfit.npz"
    metrics_path = args.out_dir / f"{label}_smplfit.json"
    result_path = args.out_dir / f"{label}_micro_doppler.npz"
    output_guards = (fit_path, metrics_path) if args.skip_doppler else (fit_path, metrics_path, result_path)
    if any(path.exists() for path in output_guards) and not args.overwrite:
        raise FileExistsError(f"Output exists for {label}; pass --overwrite to recompute")

    source26, source17 = load_mmradpose(args.sequence_dir, args.start_frame, args.num_frames)
    print(
        f"[load] {args.sequence_dir} frames={source26.shape[0]} "
        f"duration={(source26.shape[0] - 1) / args.source_fps:.3f}s",
        flush=True,
    )
    smpl_device = torch.device(args.smpl_device or args.device)
    fit = fit_smplx(source26, source17, args.model_dir, args.gender, smpl_device, args.iters_scale)
    _, _, _, metrics = save_fit_outputs(args.out_dir, label, args, source26, source17, fit)
    print(
        f"[fit] MPJPE26={metrics['mpjpe26_m'] * 1000.0:.2f} mm; "
        f"overlay17={metrics['mpjpe17_overlay_m'] * 1000.0:.2f} mm",
        flush=True,
    )
    if args.skip_doppler:
        print("[doppler] skipped by --skip-doppler", flush=True)
    else:
        run_micro_doppler(label, args, fit["vertices"], fit["faces"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
