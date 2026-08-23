#!/usr/bin/env python3
"""Profile one AMASS micro-Doppler generation run with the current simulator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("smpl_mesh_to_micro_doppler_profiled", path)
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

    def report(self, frames: int, native_frames: int, chirps: int) -> None:
        total = sum(self.values.values())
        print("\n=== Profile Summary ===")
        print(f"profiled_radar_frames: {frames}")
        print(f"native_motion_frames_loaded: {native_frames}")
        print(f"profiled_chirps: {chirps}")
        print(f"timed_total_s: {total:.6f}")
        print()
        print("stage,total_s,ms_per_radar_frame,pct_timed")
        for name, seconds in sorted(self.values.items(), key=lambda item: item[1], reverse=True):
            per_frame_ms = seconds / max(frames, 1) * 1000.0
            pct = seconds / total * 100.0 if total else 0.0
            print(f"{name},{seconds:.6f},{per_frame_ms:.3f},{pct:.2f}")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def apply_topk_visible_triangles(gen, interpolator, frame_time: float, top_k: int) -> int:
    if top_k <= 0 or int(interpolator.visible_ids.numel()) <= top_k:
        return int(interpolator.visible_ids.numel())

    vertices = interpolator.vertices_at(float(frame_time))
    triangles = vertices[interpolator.faces[interpolator.visible_ids]]
    points = triangles.mean(dim=1)
    cross = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = torch.linalg.norm(cross, dim=1)
    normals = cross / torch.clamp(twice_area[:, None], min=1e-10)
    intensities = 0.5 * twice_area * gen.gaussian_specular_weight(
        points,
        normals,
        interpolator.tx_position,
        interpolator.rx_position,
    )
    selected = torch.topk(intensities, k=int(top_k), largest=True, sorted=False).indices
    interpolator.visible_ids = interpolator.visible_ids[selected].contiguous()
    return int(interpolator.visible_ids.numel())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="/bigdata/users/quansj/datasets/Doppler/wilddoppler/code/smpl_mesh_to_micro_doppler.py")
    parser.add_argument("--amass-npz", required=True)
    parser.add_argument("--model-dir", default="/bigdata/users/quansj/datasets/Doppler/wilddoppler/smpl_models")
    parser.add_argument("--sequence", default="profile_sequence")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", default="pytorch")
    parser.add_argument("--visibility-mode", default="hold", choices=("hold", "linear"))
    parser.add_argument("--doppler-transform", default="fft", choices=("fft", "nudft"))
    parser.add_argument("--subject-range", type=float, default=4.0)
    parser.add_argument("--subject-lateral", type=float, default=0.0)
    parser.add_argument("--radar-height", type=float, default=1.0)
    parser.add_argument("--tx-rx-lateral-separation-m", type=float, default=0.0)
    parser.add_argument("--smpl-batch-size", type=int, default=16)
    parser.add_argument("--mesh-cache-dir", default="")
    parser.add_argument("--max-radar-frames", type=int, default=40)
    parser.add_argument("--top-k-visible", type=int, default=0)
    parser.add_argument("--save-profile-dir", default="")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    gen = load_generator(Path(args.generator))
    gen.apply_radar_profile(gen.RADAR_PROFILE)
    timer = Timer()
    device = torch.device(args.device)

    t0 = time.perf_counter()
    motion = gen.load_amass(Path(args.amass_npz))
    timer.add("io_load_amass_npz", time.perf_counter() - t0)

    t0 = time.perf_counter()
    model_dir = Path(args.model_dir)
    mesh_cache = None
    if args.mesh_cache_dir:
        mesh_cache_dir = Path(args.mesh_cache_dir).expanduser().resolve()
        mesh_cache_dir.mkdir(parents=True, exist_ok=True)
        mesh_cache = gen.mesh_cache_path(Path(args.amass_npz), model_dir, mesh_cache_dir)
    if mesh_cache is not None and mesh_cache.exists():
        with np.load(mesh_cache) as cached:
            native_vertices_np = cached["vertices"].astype(np.float32, copy=False)
            faces = cached["faces"].astype(np.int32, copy=False)
        print(f"[smplx-cache] loaded {mesh_cache}", flush=True)
    else:
        native_vertices_np, faces = gen.generate_native_meshes(
            motion,
            model_dir,
            device,
            args.smpl_batch_size,
        )
        if mesh_cache is not None:
            tmp_cache = mesh_cache.with_suffix(mesh_cache.suffix + ".tmp")
            with tmp_cache.open("wb") as handle:
                np.savez(handle, vertices=native_vertices_np, faces=faces)
            tmp_cache.replace(mesh_cache)
            print(f"[smplx-cache] saved {mesh_cache}", flush=True)
    sync(device)
    timer.add("gpu_smplx_native_mesh_generation", time.perf_counter() - t0)

    t0 = time.perf_counter()
    native_vertices_np = gen.place_like_gif(native_vertices_np, args.subject_range, args.subject_lateral)
    faces = gen.faces_for_witwin_world(faces)
    timer.add("cpu_mesh_coordinate_transform", time.perf_counter() - t0)

    t0 = time.perf_counter()
    native_vertices = torch.as_tensor(native_vertices_np, dtype=torch.float32, device=device)
    sync(device)
    timer.add("h2d_native_vertices_upload", time.perf_counter() - t0)
    del native_vertices_np

    mocap_rate = float(motion["mocap_frame_rate"])
    duration_s = (native_vertices.shape[0] - 1) / mocap_rate
    frame_times, chirp_times_by_frame = gen.actual_radar_timing(duration_s, gen.RADAR_PROFILES[gen.RADAR_PROFILE])
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
    scene.add_mesh(name="human", vertices=native_vertices[0].clone(), faces=faces, dynamic=True)
    tracer = Tracer(scene, radar, resolution=1, sampling="triangle", multipath=False, max_reflections=0)
    interpolator = gen.FrameLevelScattererInterpolator(
        radar,
        scene,
        tracer,
        native_vertices,
        faces,
        mocap_rate,
        duration_s,
    )
    sync(device)
    timer.add("init_radar_scene_tracer", time.perf_counter() - t0)

    signal_blocks = []
    strong_ranges = []
    selected_bins = []
    used_visible_counts = []
    for frame_time, chirp_times in zip(frame_times, chirp_times_by_frame):
        t0 = time.perf_counter()
        if args.visibility_mode == "linear":
            interpolator.prepare_interpolated_frame(float(frame_time), float(frame_time) + 1.0 / gen.FRAME_RATE_HZ)
        else:
            interpolator.prepare_frame(float(frame_time))
        sync(device)
        timer.add("gpu_visibility_trace_and_mesh_update", time.perf_counter() - t0)

        if args.top_k_visible > 0:
            t0 = time.perf_counter()
            used_count = apply_topk_visible_triangles(gen, interpolator, float(frame_time), args.top_k_visible)
            sync(device)
            timer.add("gpu_topk_visible_triangle_selection", time.perf_counter() - t0)
        else:
            used_count = int(interpolator.visible_ids.numel())
        used_visible_counts.append(used_count)

        t0 = time.perf_counter()
        with torch.no_grad():
            mimo = radar.mimo(interpolator, t0=float(frame_time))
        sync(device)
        timer.add("gpu_mimo_chirp_simulation", time.perf_counter() - t0)

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
    used_visible_per_frame = np.asarray(used_visible_counts, dtype=np.int32)
    visible_per_chirp = np.repeat(visible_per_frame, gen.CHIRPS_PER_FRAME)
    used_visible_per_chirp = np.repeat(used_visible_per_frame, gen.CHIRPS_PER_FRAME)
    strong_ranges_np = np.asarray(strong_ranges, dtype=np.float32)
    selected_bins_np = np.asarray(selected_bins, dtype=np.int16)

    t0 = time.perf_counter()
    # Measure compressed-output serialization cost without disk IO.
    import io

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
        visible_triangles_used_per_frame=used_visible_per_frame,
        visible_triangles_used_per_chirp=used_visible_per_chirp,
        selected_range_bins_by_frame=selected_bins_np,
    )
    timer.add("io_npz_compress_serialize_no_disk", time.perf_counter() - t0)

    def write_outputs(tmp: Path) -> None:
        tmp.mkdir(parents=True, exist_ok=True)
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
            visible_triangles_used_per_frame=used_visible_per_frame,
            visible_triangles_used_per_chirp=used_visible_per_chirp,
            selected_range_bins_by_frame=selected_bins_np,
        )
        timer.add("io_npz_compress_write_disk", time.perf_counter() - t0)

        if not args.skip_plot:
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
            json.dumps(
                {
                    "sequence": args.sequence,
                    "frames": len(frame_times),
                    "top_k_visible": int(args.top_k_visible),
                    "visible_triangles_before_topk_mean": float(visible_per_frame.mean()),
                    "visible_triangles_used_mean": float(used_visible_per_frame.mean()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        timer.add("io_json_write", time.perf_counter() - t0)

    if args.save_profile_dir:
        write_outputs(Path(args.save_profile_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="amass_profile_") as tmp_name:
            write_outputs(Path(tmp_name))

    timer.report(len(frame_times), int(native_vertices.shape[0]), int(chirp_times_by_frame.size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
