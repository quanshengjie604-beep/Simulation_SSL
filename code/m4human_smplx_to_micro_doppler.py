#!/usr/bin/env python3
"""Generate micro-Doppler from an M4Human SMPL-X parameter sequence."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import smplx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smpl_mesh_to_micro_doppler as pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M4HUMAN = REPO_ROOT / "datasets" / "M4Human" / "MR-Mesh" / "rf3dpose_all"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "m4human_smplx_micro_doppler"
M4HUMAN_FRAME_RATE_HZ = 12.0
GENDER_BY_SUBJECT = {
    1: "female",
    3: "female",
    4: "female",
    7: "female",
    8: "female",
    10: "female",
    13: "female",
    15: "female",
    2: "male",
    5: "male",
    6: "male",
    9: "male",
    11: "male",
    12: "male",
    14: "male",
    16: "male",
    17: "male",
    18: "male",
    19: "male",
    20: "male",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4human-dir", default=str(DEFAULT_M4HUMAN))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "smpl_models"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--action", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--num-native-frames",
        type=int,
        default=0,
        help="Number of M4Human SMPL-X frames to load; 0 loads to the end of the sequence.",
    )
    parser.add_argument("--sequence", default="", help="Output label; default is m4human_P{subject}_A{action}_F{start}_{stop}.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smpl-device", default="", help="Device for SMPL-X mesh generation; default matches --device.")
    parser.add_argument("--backend", choices=("dirichlet", "pytorch", "slang"), default="dirichlet")
    parser.add_argument("--radar-profile", choices=tuple(pipeline.RADAR_PROFILES), default=pipeline.RADAR_PROFILE)
    parser.add_argument("--range-bin-mode", choices=("fixed", "tracked"), default="tracked")
    parser.add_argument(
        "--visibility-mode",
        choices=("linear", "hold"),
        default="linear",
        help="Interpolate binary visibility between radar-frame traces or hold it for each frame.",
    )
    parser.add_argument("--doppler-transform", choices=("fft", "nudft"), default="nudft")
    parser.add_argument("--smpl-batch-size", type=int, default=32)
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument("--max-radar-frames", type=int, default=0)
    parser.add_argument(
        "--reverse-face-winding",
        action="store_true",
        help="Reverse mesh triangle winding after the M4Human radar-to-WiTwin transform.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_numpy(obj):
    if isinstance(obj, dict) and obj.get("__nd__") is True:
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(tuple(obj["shape"]))
    if isinstance(obj, dict):
        return {key: decode_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode_numpy(value) for value in obj]
    return obj


def unpack_dict_np(value: bytes | None) -> dict[str, object]:
    if value is None:
        raise KeyError("Missing LMDB value")
    return decode_numpy(msgpack.unpackb(value, raw=False))


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def inv_rodrigues(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    theta = float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(theta))
    return axis * theta


def calibrate_param_to_radar(param: dict[str, object], calib: dict[str, object]) -> dict[str, object]:
    """Match M4Human's official Vicon/camera SMPL-X parameter to radar-space conversion."""
    converted = copy.deepcopy(param)
    root_rotation = rodrigues(np.asarray(param["root_orient"]))
    vicon_to_cam = np.asarray(calib["vicon_to_cam_rotmatrix"], dtype=np.float64)
    radar_to_cam = np.asarray(calib["radar_to_cam_rotmatrix"], dtype=np.float64)
    vicon_to_cam_t = np.asarray(calib["vicon_to_cam_tvec"], dtype=np.float64) / 1000.0
    radar_to_cam_t = np.asarray(calib["radar_to_cam_tvec"], dtype=np.float64)
    joints = np.asarray(param["joints"], dtype=np.float64)
    trans = np.asarray(param["trans"], dtype=np.float64)

    root_cam = vicon_to_cam @ root_rotation
    root_radar = np.linalg.inv(radar_to_cam) @ root_cam
    pelvis_cam = vicon_to_cam @ joints[0] + vicon_to_cam_t
    pelvis_radar = np.linalg.inv(radar_to_cam) @ (pelvis_cam - radar_to_cam_t)
    converted["root_orient"] = inv_rodrigues(root_radar)
    converted["trans"] = pelvis_radar + (trans - joints[0])
    return converted


def m4human_radar_to_witwin(vertices: np.ndarray) -> np.ndarray:
    """Map M4Human radar coordinates to the WiTwin world used by the AMASS pipeline.

    M4Human radar coordinates are approximately x=lateral, y=range-forward, z=height.
    WiTwin examples use x=lateral, y=height, and negative z as forward range.
    """
    world = np.empty_like(vertices, dtype=np.float32)
    world[..., 0] = vertices[..., 0]
    world[..., 1] = vertices[..., 2]
    world[..., 2] = -vertices[..., 1]
    return world


def open_lmdb(path: Path) -> lmdb.Environment:
    return lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, max_readers=1)


def available_frames(params_env: lmdb.Environment, subject: int, action: int) -> list[int]:
    prefix = f"[{subject}, {action}, "
    frames = []
    with params_env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        if cursor.set_range(prefix.encode()):
            for key, _ in cursor:
                text = bytes(key).decode()
                if not text.startswith(prefix):
                    break
                frames.append(int(text[len(prefix) : -1]))
    return sorted(frames)


def load_m4human_motion(
    m4human_dir: Path,
    subject: int,
    action: int,
    start_frame: int,
    num_native_frames: int,
) -> tuple[dict[str, np.ndarray | str | float], np.ndarray]:
    params_env = open_lmdb(m4human_dir / "params.lmdb")
    calib_env = open_lmdb(m4human_dir / "calib.lmdb")
    try:
        frames = [frame for frame in available_frames(params_env, subject, action) if frame >= start_frame]
        if num_native_frames:
            frames = frames[:num_native_frames]
        if len(frames) < 2:
            raise ValueError(f"Need at least two frames for subject={subject}, action={action}; got {len(frames)}")
        expected = np.arange(frames[0], frames[0] + len(frames), dtype=np.int64)
        if not np.array_equal(expected, np.asarray(frames, dtype=np.int64)):
            raise ValueError(
                "Selected M4Human frames are not contiguous; choose a different start/length "
                f"(first missing near {frames[0]}..{frames[-1]})."
            )
        params = []
        with params_env.begin(buffers=True) as txn_param, calib_env.begin(buffers=True) as txn_calib:
            for frame in frames:
                key = str([subject, action, int(frame)]).encode()
                param = unpack_dict_np(bytes(txn_param.get(key)))
                calib = unpack_dict_np(bytes(txn_calib.get(key)))
                param["gender"] = 1 if GENDER_BY_SUBJECT.get(subject, "neutral") == "male" else 0
                params.append(calibrate_param_to_radar(param, calib))
    finally:
        params_env.close()
        calib_env.close()

    gender = GENDER_BY_SUBJECT.get(subject, "neutral")
    betas = np.asarray(params[0]["betas"], dtype=np.float32)
    return (
        {
            "gender": gender,
            "mocap_frame_rate": float(M4HUMAN_FRAME_RATE_HZ),
            "root_orient": np.stack([np.asarray(p["root_orient"], dtype=np.float32) for p in params]),
            "pose_body": np.stack([np.asarray(p["pose_body"], dtype=np.float32) for p in params]),
            "pose_hand": np.zeros((len(params), 90), dtype=np.float32),
            "pose_jaw": np.zeros((len(params), 3), dtype=np.float32),
            "pose_eye": np.zeros((len(params), 6), dtype=np.float32),
            "trans": np.stack([np.asarray(p["trans"], dtype=np.float32) for p in params]),
            "betas": betas,
        },
        np.asarray(frames, dtype=np.int64),
    )


def generate_m4human_meshes(
    motion: dict[str, np.ndarray | str | float],
    model_dir: Path,
    smpl_device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = int(motion["trans"].shape[0])  # type: ignore[union-attr]
    gender = str(motion["gender"])
    model = smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=gender if gender in {"male", "female", "neutral"} else "neutral",
        use_pca=False,
        num_betas=16,
        batch_size=batch_size,
    ).to(smpl_device)
    model.eval()
    faces = model.faces.astype(np.int32)
    betas = np.asarray(motion["betas"], dtype=np.float32)[:16]
    if betas.size < 16:
        betas = np.pad(betas, (0, 16 - betas.size))
    vertices_by_batch = []
    for start in range(0, num_frames, batch_size):
        stop = min(start + batch_size, num_frames)
        ids, valid = pipeline.padded_frame_ids(start, stop, batch_size)
        inputs = {
            "global_orient": pipeline.tensor(motion["root_orient"][ids], smpl_device),  # type: ignore[index]
            "body_pose": pipeline.tensor(motion["pose_body"][ids], smpl_device),  # type: ignore[index]
            "left_hand_pose": torch.zeros(batch_size, 45, dtype=torch.float32, device=smpl_device),
            "right_hand_pose": torch.zeros(batch_size, 45, dtype=torch.float32, device=smpl_device),
            "jaw_pose": torch.zeros(batch_size, 3, dtype=torch.float32, device=smpl_device),
            "leye_pose": torch.zeros(batch_size, 3, dtype=torch.float32, device=smpl_device),
            "reye_pose": torch.zeros(batch_size, 3, dtype=torch.float32, device=smpl_device),
            "betas": pipeline.tensor(np.repeat(betas[None], batch_size, axis=0), smpl_device),
            "expression": torch.zeros(batch_size, 10, dtype=torch.float32, device=smpl_device),
            "transl": pipeline.tensor(motion["trans"][ids], smpl_device),  # type: ignore[index]
        }
        with torch.no_grad():
            output = model(**inputs)
            vertices = output.vertices[:valid].detach().cpu().numpy().astype(np.float32)
        vertices_by_batch.append(vertices)
        print(f"[smplx {stop:04d}/{num_frames:04d}] generated M4Human radar-space meshes", flush=True)
    del model
    gc.collect()
    if smpl_device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(vertices_by_batch, axis=0), faces


def default_label(subject: int, action: int, frames: np.ndarray) -> str:
    return f"m4human_P{subject:02d}_A{action:02d}_F{int(frames[0])}_{int(frames[-1])}"


def validate_args(args: argparse.Namespace) -> None:
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative")
    if args.num_native_frames < 0:
        raise ValueError("--num-native-frames cannot be negative")
    if args.max_radar_frames < 0:
        raise ValueError("--max-radar-frames cannot be negative")
    if args.smpl_batch_size < 1:
        raise ValueError("--smpl-batch-size must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for WiTwin radar simulation")


def main() -> int:
    args = parse_args()
    validate_args(args)
    m4human_dir = Path(args.m4human_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    smpl_device = torch.device(args.smpl_device or args.device)
    radar_device = torch.device(args.device)

    motion, selected_frames = load_m4human_motion(
        m4human_dir,
        args.subject,
        args.action,
        args.start_frame,
        args.num_native_frames,
    )
    label = args.sequence or default_label(args.subject, args.action, selected_frames)
    result_path = out_dir / f"{label}_micro_doppler.npz"
    image_path = out_dir / f"{label}_micro_doppler.png"
    summary_path = out_dir / f"{label}_micro_doppler.json"
    existing = [path for path in (result_path, image_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}; pass --overwrite to recompute")

    native_vertices_np, faces = generate_m4human_meshes(motion, model_dir, smpl_device, args.smpl_batch_size)
    world_vertices_np = m4human_radar_to_witwin(native_vertices_np)
    if args.reverse_face_winding:
        faces = faces[:, [0, 2, 1]]
    native_vertices = torch.as_tensor(world_vertices_np, dtype=torch.float32, device=radar_device)
    del native_vertices_np, world_vertices_np

    mocap_rate = float(motion["mocap_frame_rate"])
    duration_s = (native_vertices.shape[0] - 1) / mocap_rate
    chirp_frequency_hz = pipeline.RADAR_PROFILES[args.radar_profile]
    frame_times, chirp_times_by_frame = pipeline.actual_radar_timing(duration_s, chirp_frequency_hz)
    if args.max_radar_frames:
        frame_times = frame_times[: args.max_radar_frames]
        chirp_times_by_frame = chirp_times_by_frame[: args.max_radar_frames]
    if frame_times.size == 0:
        raise ValueError("Selected M4Human segment is too short for one complete radar frame")
    print(
        f"[timing] m4human={native_vertices.shape[0]} frames at {mocap_rate:g} Hz; "
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
        name=f"{args.radar_profile}_m4human_smplx",
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

    out_dir.mkdir(parents=True, exist_ok=True)
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
        m4human_frames=selected_frames,
        subject=np.asarray(args.subject, dtype=np.int32),
        action=np.asarray(args.action, dtype=np.int32),
    )
    pipeline.save_plot(image_path, spectrum, stft_times, velocities, label, args.db_floor, chirp_frequency_hz)
    summary = {
        "processing": (
            "M4Human SMPL-X params -> official radar-space calibration -> WiTwin coordinate transform -> "
            "linear mesh interpolation -> frame-level visibility -> per-chirp geometry/intensity -> CUDA MIMO -> STFT"
        ),
        "source_m4human_dir": str(m4human_dir),
        "sequence": label,
        "subject": int(args.subject),
        "action": int(args.action),
        "gender": str(motion["gender"]),
        "m4human_frame_ids": [int(v) for v in selected_frames.tolist()],
        "native_motion_frames": int(native_vertices.shape[0]),
        "native_motion_rate_hz": mocap_rate,
        "native_motion_duration_s": duration_s,
        "coordinate_transform": "M4Human radar (x=lateral,y=range,z=height) -> WiTwin (x=lateral,y=height,z=-range)",
        "radar": {
            "profile": args.radar_profile,
            "frequency_start_ghz": 77.1,
            "frequency_stop_ghz": 78.1,
            "frame_rate_hz": pipeline.FRAME_RATE_HZ,
            "chirp_frequency_hz": chirp_frequency_hz,
            "chirps_per_frame": pipeline.CHIRPS_PER_FRAME,
            "samples_per_chirp": pipeline.ADC_SAMPLES,
            "tx": 1,
            "rx": 1,
            "num_complete_frames": int(frame_times.size),
            "num_acquired_chirps": int(chirp_times.size),
        },
        "ray_tracing": {
            "sampling": "triangle",
            "visibility_mode": args.visibility_mode,
            "triangles_tested_per_frame": int(faces.shape[0]),
            "face_winding_reversed": bool(args.reverse_face_winding),
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
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
