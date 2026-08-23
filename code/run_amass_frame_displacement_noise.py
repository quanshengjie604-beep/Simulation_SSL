#!/usr/bin/env python3
"""Run AMASS micro-Doppler with frame-level global displacement noise."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smpl_mesh_to_micro_doppler as pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMASS = (
    REPO_ROOT
    / "legacy"
    / "datasets"
    / "AMASS_SMPLX_2022"
    / "BMLmovi"
    / "Subject_11_F_MoSh"
    / "Subject_11_F_5_stageii.npz"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "legacy"
    / "results"
    / "amass_smplx_micro_doppler"
    / "frame_displacement_noise"
)
DEFAULT_NOISE_STD_CM = (0.0, 0.5, 1.0, 2.0, 3.0)


class FrameDisplacementInterpolator:
    """Interpolate between noisy radar-frame mesh samples at each chirp time."""

    def __init__(
        self,
        radar,
        scene,
        tracer,
        native_vertices: torch.Tensor,
        faces: np.ndarray,
        mocap_frame_rate: float,
        duration_s: float,
        *,
        frame_sample_displacements: torch.Tensor,
        frame_rate_hz: float,
    ) -> None:
        from witwin.radar.trace import TraceResult

        self.radar = radar
        self.scene = scene
        self.tracer = tracer
        self.native_vertices = native_vertices.to(device="cpu", dtype=torch.float32)
        self.device = torch.device(radar.device)
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        self.mocap_frame_rate = float(mocap_frame_rate)
        self.duration_s = float(duration_s)
        self.radar_position = radar.position.to(device=radar.device, dtype=torch.float32)
        self.last_native = native_vertices.shape[0] - 1
        self.trace_result_type = TraceResult
        self.visible_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.frame_visible_counts: list[int] = []
        self.frame_next_visible_counts: list[int] = []
        self.frame_union_visible_counts: list[int] = []
        self.visibility_start: torch.Tensor | None = None
        self.visibility_end: torch.Tensor | None = None
        self.visibility_start_time_s = 0.0
        self.visibility_end_time_s = 0.0
        self._cached_trace_time_s: float | None = None
        self._cached_visible_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.chirp_calls = 0
        self.frame_rate_hz = float(frame_rate_hz)
        self.frame_sample_displacements = frame_sample_displacements.to(device=self.device, dtype=torch.float32)
        self.noisy_frame_vertices = self._build_noisy_frame_vertices()

    def native_vertices_at(self, time_s: float) -> torch.Tensor:
        position = float(np.clip(time_s, 0.0, self.duration_s)) * self.mocap_frame_rate
        low = min(int(math.floor(position)), self.last_native)
        high = min(low + 1, self.last_native)
        alpha = position - low
        return torch.lerp(self.native_vertices[low], self.native_vertices[high], alpha)

    def _build_noisy_frame_vertices(self) -> torch.Tensor:
        samples = []
        for index in range(int(self.frame_sample_displacements.shape[0])):
            sample_time_s = min(index / self.frame_rate_hz, self.duration_s)
            vertices = self.native_vertices_at(sample_time_s).to(device=self.device, dtype=torch.float32)
            samples.append(vertices + self.frame_sample_displacements[index][None, :])
        return torch.stack(samples, dim=0).contiguous()

    def vertices_at(self, time_s: float) -> torch.Tensor:
        last_sample = int(self.noisy_frame_vertices.shape[0]) - 1
        position = float(np.clip(time_s, 0.0, self.duration_s)) * self.frame_rate_hz
        low = min(int(math.floor(position)), last_sample)
        high = min(low + 1, last_sample)
        alpha = position - low
        return torch.lerp(self.noisy_frame_vertices[low], self.noisy_frame_vertices[high], alpha).contiguous()

    def trace_visible_ids(self, frame_time_s: float) -> torch.Tensor:
        if self._cached_trace_time_s is not None and math.isclose(
            frame_time_s,
            self._cached_trace_time_s,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return self._cached_visible_ids
        pipeline.update_dynamic_mesh_vertices(self.scene, "human", self.vertices_at(frame_time_s))
        with torch.no_grad():
            trace = self.tracer.trace()
        if trace._tri_indices is None:
            raise RuntimeError("WiTwin frame-level triangle trace did not return triangle indices")
        visible_ids = torch.unique(trace._tri_indices.to(dtype=torch.long), sorted=True)
        self._cached_trace_time_s = float(frame_time_s)
        self._cached_visible_ids = visible_ids
        return visible_ids

    def prepare_frame(self, frame_time_s: float) -> int:
        self.visible_ids = self.trace_visible_ids(frame_time_s)
        self.visibility_start = None
        self.visibility_end = None
        count = int(self.visible_ids.numel())
        self.frame_visible_counts.append(count)
        self.chirp_calls = 0
        return count

    def prepare_interpolated_frame(self, frame_time_s: float, next_frame_time_s: float) -> int:
        current_ids = self.trace_visible_ids(frame_time_s)
        next_ids = self.trace_visible_ids(next_frame_time_s)
        start_full = torch.zeros(self.faces.shape[0], dtype=torch.float32, device=self.device)
        end_full = torch.zeros_like(start_full)
        start_full[current_ids] = 1.0
        end_full[next_ids] = 1.0
        self.visible_ids = torch.nonzero((start_full + end_full) > 0.0, as_tuple=False).squeeze(1)
        self.visibility_start = start_full[self.visible_ids]
        self.visibility_end = end_full[self.visible_ids]
        self.visibility_start_time_s = float(frame_time_s)
        self.visibility_end_time_s = float(next_frame_time_s)
        current_count = int(current_ids.numel())
        next_count = int(next_ids.numel())
        union_count = int(self.visible_ids.numel())
        self.frame_visible_counts.append(current_count)
        self.frame_next_visible_counts.append(next_count)
        self.frame_union_visible_counts.append(union_count)
        self.chirp_calls = 0
        return union_count

    def __call__(self, time_s: float):
        self.chirp_calls += 1
        vertices = self.vertices_at(float(time_s))
        triangles = vertices[self.faces[self.visible_ids]]
        points = triangles.mean(dim=1)
        cross = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        twice_area = torch.linalg.norm(cross, dim=1)
        normals = cross / torch.clamp(twice_area[:, None], min=1e-10)
        triangle_area = 0.5 * twice_area
        intensities = triangle_area * pipeline.gaussian_normal_weight(points, normals, self.radar_position)
        if self.visibility_start is not None and self.visibility_end is not None:
            duration = max(self.visibility_end_time_s - self.visibility_start_time_s, 1e-12)
            alpha = float(np.clip((float(time_s) - self.visibility_start_time_s) / duration, 0.0, 1.0))
            visibility = torch.lerp(self.visibility_start, self.visibility_end, alpha)
            intensities = intensities * visibility
        return self.trace_result_type(points, intensities, self.visible_ids, normals=normals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-npz", default=str(DEFAULT_AMASS))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "smpl_models"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sequence", default="Subject_11_F_MoSh__Subject_11_F_5_stageii")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smpl-device", default="cpu")
    parser.add_argument("--backend", choices=("dirichlet", "pytorch", "slang"), default="dirichlet")
    parser.add_argument("--smpl-batch-size", type=int, default=32)
    parser.add_argument("--subject-range", type=float, default=4.0)
    parser.add_argument("--subject-lateral", type=float, default=0.0)
    parser.add_argument("--radar-height", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--noise-std-cm", type=float, nargs="+", default=list(DEFAULT_NOISE_STD_CM))
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument("--max-radar-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_interpolator(
    args: argparse.Namespace,
    vertices: torch.Tensor,
    faces: np.ndarray,
    mocap_rate: float,
    duration_s: float,
    frame_displacements: torch.Tensor,
):
    config = pipeline.radar_config(pipeline.RADAR_PROFILES["single77_25fps"])
    Radar, RadarConfig, Scene, Tracer = pipeline.bootstrap_witwin_modules()
    pipeline.apply_radar_equation_patch()
    pose = pipeline.RadarPose(
        position=(0.0, args.radar_height, 0.0),
        target=(0.0, args.radar_height, -1.0),
    )
    radar = Radar(
        RadarConfig.from_dict(config),
        backend=args.backend,
        device=args.device,
        position=pose.position,
        target=pose.target,
        up=pose.up,
        name="single77_25fps_frame_displacement_noise",
    )
    scene = Scene(device=args.device)
    initial_vertices = vertices[0].to(device=torch.device(args.device), dtype=torch.float32).contiguous()
    scene.add_mesh(name="human", vertices=initial_vertices.clone(), faces=faces, dynamic=True)
    tracer = Tracer(
        scene,
        radar,
        resolution=1,
        sampling="triangle",
        multipath=False,
        max_reflections=0,
    )
    interpolator = FrameDisplacementInterpolator(
        radar,
        scene,
        tracer,
        vertices,
        faces,
        mocap_rate,
        duration_s,
        frame_sample_displacements=frame_displacements,
        frame_rate_hz=pipeline.FRAME_RATE_HZ,
    )
    return radar, interpolator, config


def noise_label(std_cm: float) -> str:
    if std_cm == 0:
        return "noise_0cm"
    return f"noise_{std_cm:g}cm".replace(".", "p")


def save_shared_scale_comparison(
    path: Path,
    spectra: dict[float, np.ndarray],
    times: np.ndarray,
    velocities: np.ndarray,
    db_floor: float,
    chirp_frequency_hz: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-amass-frame-noise")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_std_cm = 0.0 if 0.0 in spectra else sorted(spectra)[0]
    reference = spectra[reference_std_cm]
    peak = max(float(np.max(reference)), np.finfo(np.float32).tiny)
    time_edges = pipeline.centers_to_edges(times, pipeline.STFT_HOP_CHIRPS / chirp_frequency_hz)
    velocity_edges = pipeline.centers_to_edges(velocities, float(np.mean(np.diff(velocities))))
    fig, axes = plt.subplots(1, len(spectra), figsize=(4.1 * len(spectra), 4.2), dpi=180, constrained_layout=True)
    if len(spectra) == 1:
        axes = [axes]
    image = None
    for ax, std_cm in zip(axes, sorted(spectra)):
        relative_db = 10.0 * np.log10(np.maximum(spectra[std_cm] / peak, 1e-12))
        relative_db = np.clip(relative_db, float(db_floor), 0.0)
        image = ax.pcolormesh(
            time_edges,
            velocity_edges,
            relative_db,
            shading="flat",
            cmap="turbo",
            vmin=db_floor,
            vmax=0.0,
        )
        title = "no noise" if std_cm == 0 else f"sigma={std_cm:g} cm"
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Radial velocity (m/s)")
        ax.set_xlim(time_edges[0], time_edges[-1])
        ax.set_ylim(velocity_edges[0], velocity_edges[-1])
    colorbar = fig.colorbar(image, ax=axes, pad=0.01)
    if reference_std_cm == 0.0:
        colorbar.set_label("Relative to no-noise peak (dB)")
    else:
        colorbar.set_label(f"Relative to sigma={reference_std_cm:g} cm peak (dB)")
    fig.savefig(path)
    plt.close(fig)


def run_condition(
    args: argparse.Namespace,
    vertices: torch.Tensor,
    faces: np.ndarray,
    mocap_rate: float,
    duration_s: float,
    frame_times: np.ndarray,
    chirp_times_by_frame: np.ndarray,
    fixed_bins: np.ndarray,
    std_cm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    label = args.sequence
    stem = f"{label}_3200hz_duty100_fixedbins_frame_global_displacement_{noise_label(std_cm)}"
    result_path = out_dir / f"{stem}_micro_doppler.npz"
    image_path = out_dir / f"{stem}_micro_doppler.png"
    summary_path = out_dir / f"{stem}_micro_doppler.json"
    if not args.overwrite:
        for path in (result_path, image_path, summary_path):
            if path.exists():
                raise FileExistsError(f"Output exists: {path}; pass --overwrite to recompute")

    std_m = float(std_cm) / 100.0
    displacements = rng.normal(0.0, std_m, size=(frame_times.size + 1, 3)).astype(np.float32)
    if std_m == 0.0:
        displacements.fill(0.0)
    displacement_tensor = torch.as_tensor(displacements, dtype=torch.float32, device=torch.device(args.device))
    radar, interpolator, config = build_interpolator(
        args,
        vertices,
        faces,
        mocap_rate,
        duration_s,
        displacement_tensor,
    )
    derived = pipeline.derived_radar_parameters(config)
    slow_time_points, chirp_times, strong_ranges, selected_bins = pipeline.simulate_slow_time(
        radar,
        interpolator,
        frame_times,
        chirp_times_by_frame,
        derived,
        fixed_bins,
        "hold",
    )
    spectrum, stft_times, velocities = pipeline.mean_power_stft(
        slow_time_points,
        chirp_times,
        derived["wavelength_m"],
        pipeline.RADAR_PROFILES["single77_25fps"],
        "fft",
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
        selected_range_bins_by_frame=np.asarray(selected_bins, dtype=np.int16),
        frame_sample_displacement_m=displacements,
        frame_displacement_m=displacements,
        noise_std_m=np.asarray(std_m, dtype=np.float32),
        noise_std_cm=np.asarray(std_cm, dtype=np.float32),
    )
    plot_title = f"{label}, frame displacement sigma={std_cm:g} cm" if std_cm else f"{label}, no noise"
    pipeline.save_plot(
        image_path,
        spectrum,
        stft_times,
        velocities,
        plot_title,
        args.db_floor,
        pipeline.RADAR_PROFILES["single77_25fps"],
    )
    summary = {
        "source_amass_npz": str(Path(args.amass_npz).expanduser().resolve()),
        "sequence": label,
        "noise_model": (
            "one zero-mean Gaussian 3D displacement vector per radar-frame sample/keyframe, "
            "applied to all SMPL-X vertices at that sample; chirp-time vertices are linearly "
            "interpolated between adjacent noisy frame samples"
        ),
        "noise_std_cm_per_axis": float(std_cm),
        "noise_std_m_per_axis": std_m,
        "frame_sample_count": int(displacements.shape[0]),
        "seed": int(args.seed),
        "radar": {
            "profile": "single77_25fps",
            "chirp_frequency_hz": pipeline.RADAR_PROFILES["single77_25fps"],
            "frame_rate_hz": pipeline.FRAME_RATE_HZ,
            "chirps_per_frame": pipeline.CHIRPS_PER_FRAME,
            "frame_chirp_duty_cycle": 1.0,
            "num_frames": int(frame_times.size),
            "num_chirps": int(chirp_times.size),
        },
        "processing": {
            "visibility_mode": "hold",
            "range_bin_mode": "fixed",
            "fixed_range_bins": fixed_bins.tolist(),
            "doppler_transform": "fft",
            "db_floor": float(args.db_floor),
        },
        "displacement_norm_m": {
            "mean": float(np.linalg.norm(displacements, axis=1).mean()),
            "max": float(np.linalg.norm(displacements, axis=1).max()),
        },
        "result_npz": str(result_path),
        "image_png": str(image_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return spectrum, stft_times, velocities, image_path


def main() -> int:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the requested WiTwin radar device")
    if args.max_radar_frames < 0:
        raise ValueError("--max-radar-frames cannot be negative")

    device = torch.device(args.device)
    smpl_device = torch.device(args.smpl_device)
    motion = pipeline.load_amass(Path(args.amass_npz).expanduser().resolve())
    mocap_rate = float(motion["mocap_frame_rate"])
    native_vertices_np, faces = pipeline.generate_native_meshes(
        motion,
        Path(args.model_dir).expanduser().resolve(),
        smpl_device,
        args.smpl_batch_size,
    )
    world_vertices_np = pipeline.place_like_gif(native_vertices_np, args.subject_range, args.subject_lateral)
    faces = pipeline.faces_for_witwin_world(faces)
    vertices = torch.as_tensor(world_vertices_np, dtype=torch.float32, device="cpu")
    duration_s = (vertices.shape[0] - 1) / mocap_rate
    chirp_frequency_hz = pipeline.RADAR_PROFILES["single77_25fps"]
    frame_times, chirp_times_by_frame = pipeline.actual_radar_timing(duration_s, chirp_frequency_hz)
    if args.max_radar_frames:
        frame_times = frame_times[: args.max_radar_frames]
        chirp_times_by_frame = chirp_times_by_frame[: args.max_radar_frames]

    config = pipeline.radar_config(chirp_frequency_hz)
    derived = pipeline.derived_radar_parameters(config)
    range_bin_count = max(1, int(round(pipeline.RANGE_WINDOW_M / derived["range_resolution_m"])))
    center_bin = int(round(args.subject_range / derived["range_resolution_m"]))
    fixed_bins = pipeline.centered_range_indices(center_bin, pipeline.ADC_SAMPLES // 2, range_bin_count)
    print(
        f"[setup] native={vertices.shape[0]} frames at {mocap_rate:g} Hz; "
        f"radar={frame_times.size} frames; fixed bins={fixed_bins.tolist()}",
        flush=True,
    )

    spectra: dict[float, np.ndarray] = {}
    stft_times = velocities = None
    image_paths = []
    seed_sequence = np.random.SeedSequence(args.seed)
    child_sequences = seed_sequence.spawn(len(args.noise_std_cm))
    for std_cm, child_sequence in zip(args.noise_std_cm, child_sequences):
        print(f"[condition] frame displacement sigma={std_cm:g} cm", flush=True)
        spectrum, times, vels, image_path = run_condition(
            args,
            vertices,
            faces,
            mocap_rate,
            duration_s,
            frame_times,
            chirp_times_by_frame,
            fixed_bins,
            float(std_cm),
            np.random.default_rng(child_sequence),
        )
        spectra[float(std_cm)] = spectrum
        stft_times = times
        velocities = vels
        image_paths.append(str(image_path))

    assert stft_times is not None and velocities is not None
    comparison_path = Path(args.out_dir).expanduser().resolve() / (
        f"{args.sequence}_3200hz_duty100_fixedbins_frame_global_displacement_noise_comparison.png"
    )
    save_shared_scale_comparison(
        comparison_path,
        spectra,
        stft_times,
        velocities,
        args.db_floor,
        chirp_frequency_hz,
    )
    manifest_path = comparison_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            {
                "source_amass_npz": str(Path(args.amass_npz).expanduser().resolve()),
                "sequence": args.sequence,
                "noise_std_cm": [float(value) for value in args.noise_std_cm],
                "seed": int(args.seed),
                "image_pngs": image_paths,
                "comparison_png": str(comparison_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] comparison={comparison_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
