#!/usr/bin/env python3
"""Profile AMASS generation with RT-Pose-style chirp-time SMPL forwarding.

The baseline generator precomputes native AMASS meshes and linearly interpolates
vertices inside each WiTwin chirp callback. RT-Pose Sim2 instead interpolates
SMPL pose parameters at chirp times, batch-forwards SMPL for those chirps before
each radar frame, and lets the WiTwin callback return cached scatterers.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("smpl_mesh_to_micro_doppler_rtpose_profiled", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Timer:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + seconds

    def report(self, frames: int, native_frames: int, chirps: int, chirp_forwards: int) -> None:
        total = sum(self.values.values())
        print("\n=== RT-Pose Chirp-Forward Profile Summary ===")
        print(f"profiled_radar_frames: {frames}")
        print(f"native_motion_frames_loaded: {native_frames}")
        print(f"profiled_chirps: {chirps}")
        print(f"unique_smpl_query_forwards: {chirp_forwards}")
        print(f"timed_total_s: {total:.6f}")
        print()
        print("stage,total_s,ms_per_radar_frame,ms_per_chirp,pct_timed")
        for name, seconds in sorted(self.values.items(), key=lambda item: item[1], reverse=True):
            per_frame_ms = seconds / max(frames, 1) * 1000.0
            per_chirp_ms = seconds / max(chirps, 1) * 1000.0
            pct = seconds / total * 100.0 if total else 0.0
            print(f"{name},{seconds:.6f},{per_frame_ms:.3f},{per_chirp_ms:.3f},{pct:.2f}")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def as_padded_betas(raw: np.ndarray, count: int = 16) -> np.ndarray:
    betas = np.asarray(raw, dtype=np.float32).reshape(-1)[:count]
    if betas.size < count:
        betas = np.pad(betas, (0, count - betas.size))
    return betas.astype(np.float32)


def build_rot_slerps(frame_times_s: np.ndarray, rotations: np.ndarray) -> list[Slerp]:
    return [
        Slerp(frame_times_s, Rotation.from_rotvec(rotations[:, joint_idx]))
        for joint_idx in range(rotations.shape[1])
    ]


class RtPoseChirpSmplxMotion:
    def __init__(
        self,
        gen,
        motion: dict[str, np.ndarray | str | float],
        model_dir: Path,
        device: torch.device,
        subject_range: float,
        subject_lateral: float,
        smpl_batch_size: int,
    ) -> None:
        import smplx

        self.gen = gen
        self.motion = motion
        self.device = device
        self.num_frames = int(np.asarray(motion["trans"]).shape[0])
        self.mocap_rate = float(motion["mocap_frame_rate"])
        self.duration_s = (self.num_frames - 1) / self.mocap_rate
        self.frame_times_s = np.arange(self.num_frames, dtype=np.float64) / self.mocap_rate
        self.smpl_batch_size = max(int(smpl_batch_size), 1)
        self.query_forward_count = 0
        self.faces_np: np.ndarray
        self.faces_t: torch.Tensor
        self._vertex_cache: dict[float, torch.Tensor] = {}

        gender = str(motion["gender"])
        if gender not in {"neutral", "male", "female"}:
            gender = "neutral"
        self.model = smplx.create(
            str(model_dir),
            model_type="smplx",
            gender=gender,
            use_pca=False,
            num_betas=16,
            batch_size=self.smpl_batch_size,
        ).to(device)
        self.model.eval()
        self.faces_np = gen.faces_for_witwin_world(self.model.faces.astype(np.int32))
        self.faces_t = torch.as_tensor(self.faces_np, dtype=torch.long, device=device)

        root = np.asarray(motion["root_orient"], dtype=np.float32).reshape(self.num_frames, 1, 3)
        body = np.asarray(motion["pose_body"], dtype=np.float32).reshape(self.num_frames, -1, 3)
        hand = np.asarray(motion["pose_hand"], dtype=np.float32).reshape(self.num_frames, -1, 3)
        self.rotation_slerps = build_rot_slerps(self.frame_times_s, np.concatenate([root, body, hand], axis=1))

        self.trans_spline = self._make_spline(np.asarray(motion["trans"], dtype=np.float32))
        pose_jaw = motion["pose_jaw"]
        pose_eye = motion["pose_eye"]
        self.jaw_spline = self._make_spline(
            np.asarray(pose_jaw, dtype=np.float32) if pose_jaw is not None else np.zeros((self.num_frames, 3), np.float32)
        )
        self.eye_spline = self._make_spline(
            np.asarray(pose_eye, dtype=np.float32) if pose_eye is not None else np.zeros((self.num_frames, 6), np.float32)
        )
        self.betas = torch.as_tensor(as_padded_betas(np.asarray(motion["betas"], dtype=np.float32)), dtype=torch.float32, device=device)

        first_vertices = self.raw_vertices_for_queries(np.asarray([0.0], dtype=np.float64))[0]
        first_world = self.amass_to_witwin_unplaced(first_vertices)
        first_center = 0.5 * (first_world.min(dim=0).values + first_world.max(dim=0).values)
        self.offset = torch.empty(3, dtype=torch.float32, device=device)
        self.offset[0] = float(subject_lateral) - first_center[0]
        self.offset[1] = -first_world[:, 1].min()
        self.offset[2] = -float(subject_range) - first_center[2]
        self._vertex_cache.clear()
        self.query_forward_count = 0

    def _make_spline(self, values: np.ndarray) -> CubicSpline:
        bc_type = "not-a-knot" if self.num_frames >= 4 else "natural"
        return CubicSpline(self.frame_times_s, values, axis=0, bc_type=bc_type, extrapolate=False)

    def clip_time(self, query_s: float) -> float:
        return float(np.clip(float(query_s), 0.0, self.duration_s))

    def cache_key(self, query_s: float) -> float:
        return round(self.clip_time(query_s), 9)

    def rotvecs_for_queries(self, queries_s: np.ndarray) -> np.ndarray:
        clipped = np.asarray([self.clip_time(float(q)) for q in queries_s], dtype=np.float64)
        return np.stack([slerp(clipped).as_rotvec() for slerp in self.rotation_slerps], axis=1).astype(np.float32)

    def cache_vertices_for_queries(self, keys: list[float], queries_s: np.ndarray) -> None:
        if not keys:
            return
        rotvecs = self.rotvecs_for_queries(queries_s)
        trans = self.trans_spline(queries_s).astype(np.float32)
        jaw = self.jaw_spline(queries_s).astype(np.float32)
        eye = self.eye_spline(queries_s).astype(np.float32)
        out_vertices: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(keys), self.smpl_batch_size):
                stop = min(start + self.smpl_batch_size, len(keys))
                n_batch = stop - start
                batch_rot = rotvecs[start:stop]
                batch_trans = trans[start:stop]
                batch_jaw = jaw[start:stop]
                batch_eye = eye[start:stop]
                if n_batch < self.smpl_batch_size:
                    pad = self.smpl_batch_size - n_batch
                    batch_rot = np.pad(batch_rot, ((0, pad), (0, 0), (0, 0)), mode="edge")
                    batch_trans = np.pad(batch_trans, ((0, pad), (0, 0)), mode="edge")
                    batch_jaw = np.pad(batch_jaw, ((0, pad), (0, 0)), mode="edge")
                    batch_eye = np.pad(batch_eye, ((0, pad), (0, 0)), mode="edge")
                inputs = {
                    "global_orient": torch.as_tensor(batch_rot[:, 0, :], dtype=torch.float32, device=self.device),
                    "body_pose": torch.as_tensor(batch_rot[:, 1:22, :].reshape(self.smpl_batch_size, -1), dtype=torch.float32, device=self.device),
                    "left_hand_pose": torch.as_tensor(batch_rot[:, 22:37, :].reshape(self.smpl_batch_size, -1), dtype=torch.float32, device=self.device),
                    "right_hand_pose": torch.as_tensor(batch_rot[:, 37:52, :].reshape(self.smpl_batch_size, -1), dtype=torch.float32, device=self.device),
                    "jaw_pose": torch.as_tensor(batch_jaw, dtype=torch.float32, device=self.device),
                    "leye_pose": torch.as_tensor(batch_eye[:, :3], dtype=torch.float32, device=self.device),
                    "reye_pose": torch.as_tensor(batch_eye[:, 3:6], dtype=torch.float32, device=self.device),
                    "betas": self.betas.expand(self.smpl_batch_size, -1).contiguous(),
                    "expression": torch.zeros(self.smpl_batch_size, 10, dtype=torch.float32, device=self.device),
                    "transl": torch.as_tensor(batch_trans, dtype=torch.float32, device=self.device),
                }
                out = self.model(**inputs)
                out_vertices.append(out.vertices[:n_batch].detach().to(dtype=torch.float32))
        for key, vertices in zip(keys, torch.cat(out_vertices, dim=0), strict=True):
            self._vertex_cache[key] = vertices
        self.query_forward_count += len(keys)

    def raw_vertices_for_queries(self, queries_s: np.ndarray) -> torch.Tensor:
        clipped = np.asarray([self.clip_time(float(q)) for q in queries_s], dtype=np.float64)
        keys = [self.cache_key(float(q)) for q in clipped]
        missing_keys: list[float] = []
        missing_queries: list[float] = []
        seen: set[float] = set()
        for key, query in zip(keys, clipped, strict=True):
            if key in self._vertex_cache or key in seen:
                continue
            missing_keys.append(key)
            missing_queries.append(float(query))
            seen.add(key)
        if missing_keys:
            self.cache_vertices_for_queries(missing_keys, np.asarray(missing_queries, dtype=np.float64))
        return torch.stack([self._vertex_cache[key] for key in keys], dim=0)

    def vertices_for_queries(self, queries_s: np.ndarray) -> torch.Tensor:
        return self.amass_to_witwin(self.raw_vertices_for_queries(queries_s))

    def amass_to_witwin_unplaced(self, vertices: torch.Tensor) -> torch.Tensor:
        world = torch.empty_like(vertices)
        world[..., 0] = vertices[..., 1]
        world[..., 1] = vertices[..., 2]
        world[..., 2] = -vertices[..., 0]
        return world

    def amass_to_witwin(self, vertices: torch.Tensor) -> torch.Tensor:
        return self.amass_to_witwin_unplaced(vertices) + self.offset


class RtPosePreparedScattererInterpolator:
    def __init__(self, gen, radar, scene, tracer, motion: RtPoseChirpSmplxMotion) -> None:
        from witwin.radar.trace import TraceResult

        self.gen = gen
        self.radar = radar
        self.scene = scene
        self.tracer = tracer
        self.motion = motion
        self.trace_result_type = TraceResult
        self.tx_position = radar.tx_pos[0].to(device=radar.device, dtype=torch.float32)
        self.rx_position = radar.rx_pos[0].to(device=radar.device, dtype=torch.float32)
        self.visible_ids = torch.empty(0, dtype=torch.long, device=motion.device)
        self.prepared_samples: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.frame_visible_counts: list[int] = []
        self.chirp_calls = 0

    def trace_visible_ids(self, frame_time_s: float) -> torch.Tensor:
        vertices = self.motion.vertices_for_queries(np.asarray([frame_time_s], dtype=np.float64))[0]
        self.gen.update_dynamic_mesh_vertices(self.scene, "human", vertices)
        with torch.no_grad():
            trace = self.tracer.trace()
        if trace._tri_indices is None:
            raise RuntimeError("WiTwin frame-level triangle trace did not return triangle indices")
        return torch.unique(trace._tri_indices.to(dtype=torch.long), sorted=True)

    def prepare_visibility(self, frame_time_s: float) -> int:
        self.visible_ids = self.trace_visible_ids(float(frame_time_s))
        count = int(self.visible_ids.numel())
        self.frame_visible_counts.append(count)
        return count

    def prepare_chirp_samples(self, chirp_times_s: np.ndarray) -> None:
        self.prepared_samples.clear()
        self.chirp_calls = 0
        vertices = self.motion.vertices_for_queries(chirp_times_s)
        triangles = vertices[:, self.motion.faces_t[self.visible_ids]]
        points = triangles.mean(dim=2)
        cross = torch.linalg.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0], dim=2)
        twice_area = torch.linalg.norm(cross, dim=2)
        normals = cross / torch.clamp(twice_area[:, :, None], min=1e-10)
        intensities = 0.5 * twice_area * self.gen.gaussian_specular_weight(
            points.reshape(-1, 3),
            normals.reshape(-1, 3),
            self.tx_position,
            self.rx_position,
        ).reshape(points.shape[0], points.shape[1])
        for chirp_time_s, point_row, intensity_row, normal_row in zip(chirp_times_s, points, intensities, normals, strict=True):
            self.prepared_samples[self.motion.cache_key(float(chirp_time_s))] = (
                intensity_row.to(dtype=torch.float32).contiguous(),
                point_row.to(dtype=torch.float32).contiguous(),
                normal_row.to(dtype=torch.float32).contiguous(),
            )

    def __call__(self, time_s: float):
        self.chirp_calls += 1
        key = self.motion.cache_key(float(time_s))
        sample = self.prepared_samples.get(key)
        if sample is None:
            self.prepare_chirp_samples(np.asarray([float(time_s)], dtype=np.float64))
            sample = self.prepared_samples[key]
        intensities, points, normals = sample
        return self.trace_result_type(points, intensities, self.visible_ids, normals=normals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="/bigdata/users/quansj/datasets/Doppler/wilddoppler/code/smpl_mesh_to_micro_doppler.py")
    parser.add_argument("--amass-npz", required=True)
    parser.add_argument("--model-dir", default="/bigdata/users/quansj/datasets/Doppler/wilddoppler/smpl_models")
    parser.add_argument("--sequence", default="profile_rtpose_chirp_sequence")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", default="pytorch")
    parser.add_argument("--doppler-transform", default="fft", choices=("fft", "nudft"))
    parser.add_argument("--subject-range", type=float, default=4.0)
    parser.add_argument("--subject-lateral", type=float, default=0.0)
    parser.add_argument("--radar-height", type=float, default=1.0)
    parser.add_argument("--tx-rx-lateral-separation-m", type=float, default=0.0)
    parser.add_argument("--smpl-batch-size", type=int, default=128)
    parser.add_argument("--max-radar-frames", type=int, default=10)
    args = parser.parse_args()

    gen = load_generator(Path(args.generator))
    gen.apply_radar_profile(gen.RADAR_PROFILE)
    timer = Timer()
    device = torch.device(args.device)

    t0 = time.perf_counter()
    motion = gen.load_amass(Path(args.amass_npz))
    timer.add("io_load_amass_npz", time.perf_counter() - t0)

    t0 = time.perf_counter()
    rt_motion = RtPoseChirpSmplxMotion(
        gen,
        motion,
        Path(args.model_dir),
        device,
        args.subject_range,
        args.subject_lateral,
        args.smpl_batch_size,
    )
    sync(device)
    timer.add("init_smplx_model_and_rtpose_interpolators", time.perf_counter() - t0)

    frame_times, chirp_times_by_frame = gen.actual_radar_timing(rt_motion.duration_s, gen.RADAR_PROFILES[gen.RADAR_PROFILE])
    if args.max_radar_frames:
        frame_times = frame_times[: args.max_radar_frames]
        chirp_times_by_frame = chirp_times_by_frame[: args.max_radar_frames]

    t0 = time.perf_counter()
    config = gen.radar_config(gen.RADAR_PROFILES[gen.RADAR_PROFILE], args.tx_rx_lateral_separation_m)
    derived = gen.derived_radar_parameters(config)
    range_bin_count = max(1, int(round(gen.RANGE_WINDOW_M / derived["range_resolution_m"])))
    fixed_center_range_m = gen.fixed_range_center_m(
        args.subject_range,
        args.subject_lateral,
        args.tx_rx_lateral_separation_m,
    )
    fixed_center_bin = int(round(fixed_center_range_m / derived["range_resolution_m"]))
    fixed_range_bins = gen.centered_range_indices(fixed_center_bin, gen.ADC_SAMPLES // 2, range_bin_count)
    Radar, RadarConfig, Scene, Tracer = gen.bootstrap_witwin_modules()
    gen.apply_radar_equation_patch()
    pose = gen.RadarPose(position=(0.0, args.radar_height, 0.0), target=(0.0, args.radar_height, -1.0))
    radar = Radar(
        RadarConfig.from_dict(config),
        backend=args.backend,
        device=args.device,
        position=pose.position,
        target=pose.target,
        up=pose.up,
        name=gen.RADAR_PROFILE,
    )
    scene = Scene(device=args.device)
    first_vertices = rt_motion.vertices_for_queries(np.asarray([0.0], dtype=np.float64))[0]
    scene.add_mesh(name="human", vertices=first_vertices.clone(), faces=rt_motion.faces_np, dynamic=True)
    tracer = Tracer(scene, radar, resolution=1, sampling="triangle", multipath=False, max_reflections=0)
    interpolator = RtPosePreparedScattererInterpolator(gen, radar, scene, tracer, rt_motion)
    sync(device)
    timer.add("init_radar_scene_tracer", time.perf_counter() - t0)

    signal_blocks = []
    strong_ranges = []
    selected_bins = []
    for frame_time, chirp_times in zip(frame_times, chirp_times_by_frame, strict=True):
        t0 = time.perf_counter()
        visible_count = interpolator.prepare_visibility(float(frame_time))
        sync(device)
        timer.add("gpu_visibility_trace_and_mesh_forward", time.perf_counter() - t0)

        t0 = time.perf_counter()
        interpolator.prepare_chirp_samples(chirp_times)
        sync(device)
        timer.add("gpu_rtpose_chirp_smpl_forward_and_scatterer_precompute", time.perf_counter() - t0)

        t0 = time.perf_counter()
        with torch.no_grad():
            mimo = radar.mimo(interpolator, t0=float(frame_time))
        sync(device)
        timer.add("gpu_mimo_echo_from_cached_scatterers", time.perf_counter() - t0)

        if interpolator.chirp_calls != gen.CHIRPS_PER_FRAME:
            raise RuntimeError(f"WiTwin evaluated {interpolator.chirp_calls} chirps, expected {gen.CHIRPS_PER_FRAME}")

        t0 = time.perf_counter()
        frame = mimo.detach().cpu().numpy()
        timer.add("d2h_mimo_frame_copy", time.perf_counter() - t0)

        t0 = time.perf_counter()
        adc = np.transpose(frame, (3, 2, 1, 0)).astype(np.complex64, copy=False)
        transformed = gen.range_fft(adc)
        points, strong_range, bins = gen.strong_reflection_points(
            transformed,
            derived["range_resolution_m"],
            fixed_range_bins,
        )
        signal_blocks.append(points)
        strong_ranges.append(strong_range)
        selected_bins.append([int(value) for value in bins])
        timer.add("cpu_range_fft_select_bins", time.perf_counter() - t0)
        print(
            f"[radar {len(signal_blocks):04d}/{len(frame_times):04d}] "
            f"t={frame_time:.3f}s visible={visible_count} peak_range={strong_range:.3f}m",
            flush=True,
        )

    t0 = time.perf_counter()
    slow_time_points = np.concatenate(signal_blocks, axis=0)
    chirp_times = chirp_times_by_frame.reshape(-1)
    timer.add("cpu_concat_slow_time", time.perf_counter() - t0)

    t0 = time.perf_counter()
    spectrum, stft_times, velocities = gen.mean_power_stft(
        slow_time_points,
        chirp_times,
        derived["wavelength_m"],
        gen.RADAR_PROFILES[gen.RADAR_PROFILE],
        args.doppler_transform,
    )
    timer.add("cpu_doppler_stft", time.perf_counter() - t0)

    visible_per_frame = np.asarray(interpolator.frame_visible_counts, dtype=np.int32)
    visible_per_chirp = np.repeat(visible_per_frame, gen.CHIRPS_PER_FRAME)
    strong_ranges_np = np.asarray(strong_ranges, dtype=np.float32)
    selected_bins_np = np.asarray(selected_bins, dtype=np.int16)

    t0 = time.perf_counter()
    handle = io.BytesIO()
    np.savez_compressed(
        handle,
        spectrum=spectrum,
        stft_time_s=stft_times,
        velocity_mps=velocities,
        strong_range_m=strong_ranges_np,
        radar_frame_time_s=frame_times,
        chirp_time_s=chirp_times,
        visible_triangles_per_frame=visible_per_frame,
        visible_triangles_per_chirp=visible_per_chirp,
        selected_range_bins_by_frame=selected_bins_np,
    )
    timer.add("io_npz_compress_serialize_no_disk", time.perf_counter() - t0)

    with tempfile.TemporaryDirectory(prefix="amass_rtpose_chirp_profile_") as tmp_name:
        tmp = Path(tmp_name)
        t0 = time.perf_counter()
        np.savez_compressed(
            tmp / "profile_micro_doppler.npz",
            spectrum=spectrum,
            stft_time_s=stft_times,
            velocity_mps=velocities,
            strong_range_m=strong_ranges_np,
            radar_frame_time_s=frame_times,
            chirp_time_s=chirp_times,
            visible_triangles_per_frame=visible_per_frame,
            visible_triangles_per_chirp=visible_per_chirp,
            selected_range_bins_by_frame=selected_bins_np,
        )
        timer.add("io_npz_compress_write_disk", time.perf_counter() - t0)

        t0 = time.perf_counter()
        gen.save_plot(
            tmp / "profile_micro_doppler.png",
            spectrum,
            stft_times,
            velocities,
            args.sequence,
            -45.0,
            gen.RADAR_PROFILES[gen.RADAR_PROFILE],
        )
        timer.add("cpu_io_png_plot_render_write", time.perf_counter() - t0)

        t0 = time.perf_counter()
        (tmp / "profile_micro_doppler.json").write_text(
            json.dumps({"sequence": args.sequence, "frames": len(frame_times)}, indent=2),
            encoding="utf-8",
        )
        timer.add("io_json_write", time.perf_counter() - t0)

    timer.report(
        len(frame_times),
        int(rt_motion.num_frames),
        int(chirp_times_by_frame.size),
        int(rt_motion.query_forward_count),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
