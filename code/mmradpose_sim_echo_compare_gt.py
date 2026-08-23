#!/usr/bin/env python3
"""Simulate mmRadPose SMPL-X echoes and compare them with GT raw-radar Doppler-time."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import mmradpose_raw_to_doppler_time as rawpp  # noqa: E402
import smpl_mesh_to_micro_doppler as witpipe  # noqa: E402


DEFAULT_FIT = (
    REPO_ROOT
    / "results/mmradpose_smplx_micro_doppler/"
    / "mmradpose_p6_angle0_ac07_r00_F0_149_26kp_fit_smplfit.npz"
)
DEFAULT_RAW = rawpp.DEFAULT_RAW
DEFAULT_SKELETON = rawpp.DEFAULT_SKELETON
DEFAULT_TARGETLIST = rawpp.DEFAULT_TARGETLIST
DEFAULT_GT_NPZ = (
    REPO_ROOT
    / "results/mmradpose_raw_doppler_time/"
    / "mmradpose_p6_an0_ac7_r0_F0_149_gt_target_aligned_tracked_smooth_range_roi.npz"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results/mmradpose_sim_gt_compare"

LIGHT_SPEED_MPS = 299_792_458.0
CENTER_FREQUENCY_HZ = 60.0e9
FRAME_RATE_HZ = 15.0
ADC_SAMPLES = 64
CHIRPS_PER_FRAME = 128
ADC_SAMPLE_RATE_HZ = 3.8e6
RANGE_RESOLUTION_M = 0.148
DOPPLER_RESOLUTION_MPS = 0.078
CHIRP_FREQUENCY_HZ = 4_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-npz", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--raw-npz", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--gt-npz", type=Path, default=DEFAULT_GT_NPZ)
    parser.add_argument("--skeleton-npy", type=Path, default=DEFAULT_SKELETON)
    parser.add_argument("--targetlist-npy", type=Path, default=DEFAULT_TARGETLIST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default="mmradpose_p6_an0_ac7_r0_F0_149_sim_vs_gt_mmradpose_radar")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=150)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("dirichlet", "pytorch", "slang"), default="dirichlet")
    parser.add_argument("--trace-resolution", type=int, default=4)
    parser.add_argument("--specular-eta", type=float, default=0.5)
    parser.add_argument(
        "--mesh-placement",
        choices=("gt-fixed", "gt-aligned", "old-fit"),
        default="gt-fixed",
        help=(
            "gt-fixed applies one sequence-level range translation; gt-aligned applies per-frame "
            "targetlist offsets; old-fit keeps the fit NPZ range used by the earlier STFT."
        ),
    )
    parser.add_argument(
        "--gt-roi-placement",
        choices=("same-as-sim", "gt-aligned", "old-fit"),
        default="same-as-sim",
        help="Range ROI used for GT raw radar. Use gt-aligned with --mesh-placement old-fit for near-sim/far-GT comparison.",
    )
    parser.add_argument("--visibility-mode", choices=("linear", "hold"), default="linear")
    parser.add_argument("--roi-mode", choices=("per-frame", "fixed-union", "tracked-smooth"), default="tracked-smooth")
    parser.add_argument("--range-margin-m", type=float, default=0.50)
    parser.add_argument("--tracked-width-m", type=float, default=2.0)
    parser.add_argument("--tracked-smooth-frames", type=int, default=9)
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse an existing *_sim_echo_cube.npz and regenerate spectra/plots without WiTwin.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mmradpose_radar_config() -> dict[str, object]:
    ramp_end_us = ADC_SAMPLES / ADC_SAMPLE_RATE_HZ * 1e6
    chirp_period_us = 1e6 / CHIRP_FREQUENCY_HZ
    slope_mhz_per_us = (
        LIGHT_SPEED_MPS * ADC_SAMPLE_RATE_HZ / (2.0 * RANGE_RESOLUTION_M * ADC_SAMPLES)
    ) / 1e12
    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": CENTER_FREQUENCY_HZ,
        "slope": slope_mhz_per_us,
        "adc_samples": ADC_SAMPLES,
        "adc_start_time": 0,
        "sample_rate": ADC_SAMPLE_RATE_HZ / 1e3,
        "idle_time": chirp_period_us - ramp_end_us,
        "ramp_end_time": ramp_end_us,
        "chirp_per_frame": CHIRPS_PER_FRAME,
        "frame_per_second": FRAME_RATE_HZ,
        "num_doppler_bins": CHIRPS_PER_FRAME,
        "num_range_bins": ADC_SAMPLES,
        "num_angle_bins": 1,
        "power": 0,
        "tx_loc": [(0.0, 0.0, 0.0)],
        "rx_loc": [(0.0, 0.0, 0.0)],
    }


def radar_to_witwin(vertices: np.ndarray) -> np.ndarray:
    world = np.empty_like(vertices, dtype=np.float32)
    world[..., 0] = vertices[..., 0]
    world[..., 1] = vertices[..., 2]
    world[..., 2] = -vertices[..., 1]
    return world


def faces_for_witwin_world(faces: np.ndarray) -> np.ndarray:
    # mmRadPose radar -> WiTwin has determinant +1, so keep the original winding.
    return np.ascontiguousarray(faces)


def load_fit_mesh(path: Path, start_frame: int, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        vertices = data["vertices"].astype(np.float32)
        faces = data["faces"].astype(np.int32)
    stop = start_frame + num_frames
    if vertices.shape[0] < stop:
        raise ValueError(f"{path} has {vertices.shape[0]} mesh frames; need {stop}")
    return vertices[start_frame:stop].copy(), faces


def aligned_mesh_vertices(
    vertices_radar: np.ndarray,
    skeleton: np.ndarray,
    targetlist: np.ndarray | None,
    start_frame: int,
    range_margin_m: float,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    num_frames = vertices_radar.shape[0]
    range_bins, range_min, range_max, range_offset = rawpp.skeleton_range_bins(
        skeleton,
        targetlist,
        start_frame,
        num_frames,
        RANGE_RESOLUTION_M,
        range_margin_m,
        ADC_SAMPLES,
    )
    aligned = vertices_radar.copy()
    aligned[..., 1] += range_offset[:, None]
    return aligned, range_bins, range_min, range_max, range_offset


def mesh_range_bins(
    vertices_radar: np.ndarray,
    range_margin_m: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    range_min = np.maximum(np.nanmin(vertices_radar[..., 1], axis=1) - range_margin_m, 0.0)
    range_max = np.maximum(np.nanmax(vertices_radar[..., 1], axis=1) + range_margin_m, range_min)
    range_bins = []
    for low, high in zip(range_min, range_max):
        lo = min(max(int(np.floor(low / RANGE_RESOLUTION_M)), 0), ADC_SAMPLES - 1)
        hi = min(max(int(np.ceil(high / RANGE_RESOLUTION_M)), lo), ADC_SAMPLES - 1)
        range_bins.append(np.arange(lo, hi + 1, dtype=np.int16))
    range_offset = np.zeros(vertices_radar.shape[0], dtype=np.float32)
    return range_bins, range_min.astype(np.float32), range_max.astype(np.float32), range_offset


def target_aligned_range_bins(
    skeleton: np.ndarray,
    targetlist: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    return rawpp.skeleton_range_bins(
        skeleton,
        targetlist,
        args.start_frame,
        args.num_frames,
        RANGE_RESOLUTION_M,
        args.range_margin_m,
        ADC_SAMPLES,
    )


def sim_placement_setup(
    placement: str,
    vertices_radar: np.ndarray,
    skeleton: np.ndarray,
    targetlist: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if placement in {"gt-fixed", "gt-aligned"}:
        _, _, _, raw_offset = target_aligned_range_bins(skeleton, targetlist, args)
        range_offset = (
            np.full_like(raw_offset, float(np.median(raw_offset)), dtype=np.float32)
            if placement == "gt-fixed"
            else raw_offset
        )
        vertices = vertices_radar.copy()
        vertices[..., 1] += range_offset[:, None]
        bins, range_min, range_max, _ = mesh_range_bins(vertices, args.range_margin_m)
        return vertices, bins, range_min, range_max, range_offset
    vertices = vertices_radar.copy()
    bins, range_min, range_max, range_offset = mesh_range_bins(vertices, args.range_margin_m)
    return vertices, bins, range_min, range_max, range_offset


def gt_roi_setup(
    placement: str,
    vertices_radar: np.ndarray,
    skeleton: np.ndarray,
    targetlist: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if placement == "gt-aligned":
        return target_aligned_range_bins(skeleton, targetlist, args)
    return mesh_range_bins(vertices_radar, args.range_margin_m)


def choose_range_bins(
    range_bins: list[np.ndarray],
    range_min: np.ndarray,
    range_max: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, object]]:
    info: dict[str, object] = {"roi_mode": args.roi_mode}
    if args.roi_mode == "fixed-union":
        bins, roi_min, roi_max = rawpp.fixed_union_range_bins(
            range_min, range_max, RANGE_RESOLUTION_M, ADC_SAMPLES
        )
        info.update({"fixed_roi_min_m": float(roi_min), "fixed_roi_max_m": float(roi_max)})
        return bins, range_min, range_max, info
    if args.roi_mode == "tracked-smooth":
        bins, smooth_min, smooth_max = rawpp.tracked_smooth_range_bins(
            range_min,
            range_max,
            RANGE_RESOLUTION_M,
            ADC_SAMPLES,
            args.tracked_width_m,
            args.tracked_smooth_frames,
        )
        info.update(
            {
                "tracked_width_m": float(args.tracked_width_m),
                "tracked_smooth_frames": int(args.tracked_smooth_frames),
            }
        )
        return bins, smooth_min, smooth_max, info
    return range_bins, range_min, range_max, info


def simulate_echo_cube(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    if not str(args.device).startswith("cuda"):
        raise ValueError(f"--device must be CUDA, got {args.device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WiTwin simulation, but torch.cuda.is_available() is False")

    if args.specular_eta <= 0.0:
        raise ValueError("--specular-eta must be positive")
    witpipe.SPECULAR_ETA = float(args.specular_eta)
    witpipe.apply_radar_equation_patch()
    Radar, RadarConfig, Scene, Tracer = witpipe.bootstrap_witwin_modules()
    config = mmradpose_radar_config()
    derived = witpipe.derived_radar_parameters(config)
    device = torch.device(args.device)
    interp_vertices_world = vertices_world
    extrapolated_last_frame = False
    if vertices_world.shape[0] >= 2:
        last_step = vertices_world[-1:] - vertices_world[-2:-1]
        interp_vertices_world = np.concatenate([vertices_world, vertices_world[-1:] + last_step], axis=0)
        extrapolated_last_frame = True
    radar = Radar(
        RadarConfig.from_dict(config),
        backend=args.backend,
        device=args.device,
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
        name="mmradpose_single60",
    )
    scene = Scene(device=args.device)
    native_vertices = torch.as_tensor(interp_vertices_world, dtype=torch.float32, device=device)
    faces_world = faces_for_witwin_world(faces)
    scene.add_mesh(name="human", vertices=native_vertices[0].clone(), faces=faces_world, dynamic=True)
    tracer = Tracer(
        scene,
        radar,
        resolution=args.trace_resolution,
        sampling="triangle",
        multipath=False,
        max_reflections=0,
    )
    duration_s = (interp_vertices_world.shape[0] - 1) / FRAME_RATE_HZ
    interpolator = witpipe.FrameLevelScattererInterpolator(
        radar,
        scene,
        tracer,
        native_vertices,
        faces_world,
        FRAME_RATE_HZ,
        duration_s,
    )

    echo_cube = np.empty((vertices_world.shape[0], 1, 1, ADC_SAMPLES, CHIRPS_PER_FRAME), dtype=np.complex64)
    frame_times = np.arange(vertices_world.shape[0], dtype=np.float64) / FRAME_RATE_HZ
    for idx, frame_time in enumerate(frame_times):
        if args.visibility_mode == "linear":
            next_frame_time = min(float(frame_time) + 1.0 / FRAME_RATE_HZ, duration_s)
            visible_count = interpolator.prepare_interpolated_frame(float(frame_time), next_frame_time)
            visibility_text = (
                f"visible_start={interpolator.frame_visible_counts[-1]} "
                f"visible_end={interpolator.frame_next_visible_counts[-1]} union={visible_count}"
            )
        else:
            visible_count = interpolator.prepare_frame(float(frame_time))
            visibility_text = f"frame_visible={visible_count}"

        with torch.no_grad():
            mimo = radar.mimo(interpolator, t0=float(frame_time))
        if interpolator.chirp_calls != CHIRPS_PER_FRAME:
            raise RuntimeError(
                f"Frame {idx} evaluated {interpolator.chirp_calls} chirps; expected {CHIRPS_PER_FRAME}"
            )
        frame = mimo.detach().cpu().numpy()
        expected = (1, 1, CHIRPS_PER_FRAME, ADC_SAMPLES)
        if frame.shape != expected:
            raise ValueError(f"WiTwin returned {frame.shape}; expected {expected}")
        adc = np.transpose(frame, (0, 1, 3, 2)).astype(np.complex64, copy=False)
        echo_cube[idx] = adc
        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == vertices_world.shape[0]:
            print(
                f"[sim {idx + 1:04d}/{vertices_world.shape[0]:04d}] "
                f"t={frame_time:.3f}s {visibility_text}",
                flush=True,
            )
    visible = np.asarray(interpolator.frame_visible_counts, dtype=np.int32)
    union_visible = np.asarray(interpolator.frame_union_visible_counts, dtype=np.int32)
    sim_info = {
        "witwin_backend": args.backend,
        "trace_resolution": int(args.trace_resolution),
        "specular_eta": float(args.specular_eta),
        "visibility_mode": args.visibility_mode,
        "extrapolated_last_mesh_frame": extrapolated_last_frame,
        "visible_triangles_min": int(visible.min()) if visible.size else None,
        "visible_triangles_mean": float(visible.mean()) if visible.size else None,
        "visible_triangles_max": int(visible.max()) if visible.size else None,
        "union_visible_triangles_min": int(union_visible.min()) if union_visible.size else None,
        "union_visible_triangles_mean": float(union_visible.mean()) if union_visible.size else None,
        "union_visible_triangles_max": int(union_visible.max()) if union_visible.size else None,
        "radar_config": config,
        "derived_radar_parameters": derived,
    }
    return echo_cube, sim_info


def range_fft_points_from_frame(
    frame: np.ndarray,
    bins: np.ndarray,
    remove_slow_time_mean: bool = False,
) -> np.ndarray:
    adc = frame.astype(np.complex64, copy=False)
    adc = adc - adc.mean(axis=-2, keepdims=True)
    n = np.arange(1, adc.shape[-2] + 1, dtype=np.float32) / (adc.shape[-2] + 1)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n)).astype(np.float32)
    range_fft = np.fft.fft(adc * window[None, None, :, None], n=adc.shape[-2], axis=-2)
    if remove_slow_time_mean:
        range_fft = range_fft - range_fft.mean(axis=-1, keepdims=True)
    selected = range_fft[..., bins, :]
    return np.moveaxis(selected, -1, 0).reshape(adc.shape[-1], -1).astype(np.complex64, copy=False)


def stft_from_cube(
    cube: np.ndarray,
    range_bins: list[np.ndarray],
    start_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = []
    times = []
    for idx, bins in enumerate(range_bins):
        points.append(range_fft_points_from_frame(cube[idx], bins))
        frame_time = (start_frame + idx) / FRAME_RATE_HZ
        times.append(frame_time + np.arange(CHIRPS_PER_FRAME, dtype=np.float64) / CHIRP_FREQUENCY_HZ)
    slow_time_points = np.concatenate(points, axis=0)
    chirp_times = np.concatenate(times, axis=0)
    derived = witpipe.derived_radar_parameters(mmradpose_radar_config())
    return witpipe.mean_power_stft(
        slow_time_points,
        chirp_times,
        derived["wavelength_m"],
        CHIRP_FREQUENCY_HZ,
        "nudft",
    )


def union_range_bins(range_bins: list[np.ndarray]) -> list[np.ndarray]:
    lo = int(min(b[0] for b in range_bins))
    hi = int(max(b[-1] for b in range_bins))
    fixed = np.arange(lo, hi + 1, dtype=np.int16)
    return [fixed for _ in range(len(range_bins))]


def spectrum_from_cube(cube: np.ndarray, range_bins: list[np.ndarray], floor_db: float) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.empty((CHIRPS_PER_FRAME, len(range_bins)), dtype=np.float32)
    for idx, bins in enumerate(range_bins):
        rd_power = rawpp.range_doppler_power(cube[idx])
        spectrum[:, idx] = rd_power[bins].mean(axis=0)
    return spectrum, rawpp.power_to_db(spectrum, floor_db)


def load_or_compute_gt(
    args: argparse.Namespace,
    range_bins: list[np.ndarray],
    selected_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if args.gt_npz.exists():
        with np.load(args.gt_npz, allow_pickle=False) as data:
            spectrum = data["spectrum"].astype(np.float32)
            gt_bins = data["selected_range_bins"].astype(np.int16) if "selected_range_bins" in data.files else None
        if (
            spectrum.shape == (CHIRPS_PER_FRAME, len(range_bins))
            and gt_bins is not None
            and gt_bins.shape == selected_bins.shape
            and np.array_equal(gt_bins, selected_bins)
        ):
            return spectrum, rawpp.power_to_db(spectrum, args.db_floor), str(args.gt_npz.resolve())
    cube = rawpp.load_cube(args.raw_npz)
    selected = cube[args.start_frame : args.start_frame + len(range_bins)]
    spectrum, spectrum_db = spectrum_from_cube(selected, range_bins, args.db_floor)
    return spectrum, spectrum_db, str(args.raw_npz.resolve())


def save_compare_plot(
    path: Path,
    gt_db: np.ndarray,
    sim_db: np.ndarray,
    times: np.ndarray,
    velocities: np.ndarray,
    floor_db: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "logs/.cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    diff = np.clip(sim_db - gt_db, -20.0, 20.0)
    extent = [times[0], times[-1] + 1.0 / FRAME_RATE_HZ, velocities[0], velocities[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True, sharey=True)
    for ax, image, title, cmap, vmin, vmax in (
        (axes[0], gt_db, "GT raw radar", "turbo", floor_db, 0.0),
        (axes[1], sim_db, "SMPL-X WiTwin simulation", "turbo", floor_db, 0.0),
        (axes[2], diff, "Sim - GT (dB)", "coolwarm", -20.0, 20.0),
    ):
        im = ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("Radial velocity (m/s)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_stft_compare_plot(
    path: Path,
    gt_spectrum: np.ndarray,
    gt_times: np.ndarray,
    gt_velocities: np.ndarray,
    sim_spectrum: np.ndarray,
    sim_times: np.ndarray,
    sim_velocities: np.ndarray,
    floor_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "logs/.cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_db = rawpp.power_to_db(gt_spectrum, floor_db)
    sim_db = rawpp.power_to_db(sim_spectrum, floor_db)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True, sharey=True)
    for ax, image, times, velocities, title in (
        (axes[0], gt_db, gt_times, gt_velocities, "GT raw radar STFT"),
        (axes[1], sim_db, sim_times, sim_velocities, "SMPL-X WiTwin STFT"),
    ):
        extent = [times[0], times[-1], velocities[0], velocities[-1]]
        im = ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="turbo",
            vmin=floor_db,
            vmax=0.0,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("Radial velocity (m/s)")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return gt_db, sim_db


def normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return 0.0 if denom == 0.0 else float(np.dot(aa, bb) / denom)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim_cube_path = args.out_dir / f"{args.label}_sim_echo_cube.npz"
    sim_npz_path = args.out_dir / f"{args.label}_sim_doppler.npz"
    compare_npz_path = args.out_dir / f"{args.label}_compare.npz"
    stft_npz_path = args.out_dir / f"{args.label}_stft_compare.npz"
    png_path = args.out_dir / f"{args.label}_compare.png"
    stft_png_path = args.out_dir / f"{args.label}_stft_compare.png"
    json_path = args.out_dir / f"{args.label}_summary.json"
    for path in (sim_npz_path, compare_npz_path, stft_npz_path, png_path, stft_png_path, json_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    if sim_cube_path.exists() and not (args.overwrite or args.postprocess_only):
        raise FileExistsError(f"{sim_cube_path} exists; pass --overwrite or --postprocess-only")
    if args.postprocess_only and not sim_cube_path.exists():
        raise FileNotFoundError(f"{sim_cube_path} does not exist; cannot use --postprocess-only")

    vertices_radar, faces = load_fit_mesh(args.fit_npz, args.start_frame, args.num_frames)
    skeleton = np.load(args.skeleton_npy, allow_pickle=False)
    targetlist = np.load(args.targetlist_npy, allow_pickle=False).astype(np.float32, copy=False)
    if args.gt_roi_placement == "same-as-sim":
        gt_roi_placement = "gt-aligned" if args.mesh_placement == "gt-fixed" else args.mesh_placement
    else:
        gt_roi_placement = args.gt_roi_placement
    vertices_aligned, sim_base_bins, sim_range_min, sim_range_max, sim_range_offset = sim_placement_setup(
        args.mesh_placement,
        vertices_radar,
        skeleton,
        targetlist,
        args,
    )
    gt_base_bins, gt_range_min, gt_range_max, gt_range_offset = gt_roi_setup(
        gt_roi_placement,
        vertices_radar,
        skeleton,
        targetlist,
        args,
    )
    sim_range_bins, sim_roi_min, sim_roi_max, sim_roi_info = choose_range_bins(
        sim_base_bins, sim_range_min, sim_range_max, args
    )
    gt_range_bins, gt_roi_min, gt_roi_max, gt_roi_info = choose_range_bins(
        gt_base_bins, gt_range_min, gt_range_max, args
    )
    vertices_world = radar_to_witwin(vertices_aligned)
    if args.postprocess_only:
        with np.load(sim_cube_path, allow_pickle=False) as data:
            echo_cube = data["echo_cube"].astype(np.complex64, copy=False)
        sim_info = {"loaded_existing_echo_cube": str(sim_cube_path.resolve())}
    else:
        echo_cube, sim_info = simulate_echo_cube(vertices_world, faces, args)
    sim_spectrum, sim_db = spectrum_from_cube(echo_cube, sim_range_bins, args.db_floor)

    times = (args.start_frame + np.arange(args.num_frames, dtype=np.float64)) / FRAME_RATE_HZ
    velocities = (np.arange(CHIRPS_PER_FRAME, dtype=np.float64) - CHIRPS_PER_FRAME // 2) * DOPPLER_RESOLUTION_MPS
    sim_selected_bins = np.full((args.num_frames, ADC_SAMPLES), -1, dtype=np.int16)
    gt_selected_bins = np.full((args.num_frames, ADC_SAMPLES), -1, dtype=np.int16)
    for idx, bins in enumerate(sim_range_bins):
        sim_selected_bins[idx, : bins.size] = bins
    for idx, bins in enumerate(gt_range_bins):
        gt_selected_bins[idx, : bins.size] = bins
    gt_spectrum, gt_db, gt_source = load_or_compute_gt(args, gt_range_bins, gt_selected_bins)
    gt_cube = rawpp.load_cube(args.raw_npz)[args.start_frame : args.start_frame + args.num_frames]
    gt_stft_range_bins = union_range_bins(gt_range_bins)
    sim_stft_range_bins = union_range_bins(sim_range_bins)
    gt_stft, gt_stft_times, gt_stft_velocities = stft_from_cube(gt_cube, gt_stft_range_bins, args.start_frame)
    sim_stft, sim_stft_times, sim_stft_velocities = stft_from_cube(echo_cube, sim_stft_range_bins, args.start_frame)
    gt_stft_db, sim_stft_db = save_stft_compare_plot(
        stft_png_path,
        gt_stft,
        gt_stft_times,
        gt_stft_velocities,
        sim_stft,
        sim_stft_times,
        sim_stft_velocities,
        args.db_floor,
    )

    if not args.postprocess_only:
        np.savez_compressed(sim_cube_path, echo_cube=echo_cube)
    np.savez_compressed(
        sim_npz_path,
        spectrum=sim_spectrum,
        spectrum_db=sim_db,
        time_s=times,
        velocity_mps=velocities,
        selected_range_bins=sim_selected_bins,
        range_min_m=sim_roi_min,
        range_max_m=sim_roi_max,
        range_offset_m=sim_range_offset,
        mesh_placement=np.asarray(args.mesh_placement),
    )
    np.savez_compressed(
        compare_npz_path,
        gt_spectrum=gt_spectrum,
        gt_spectrum_db=gt_db,
        sim_spectrum=sim_spectrum,
        sim_spectrum_db=sim_db,
        time_s=times,
        velocity_mps=velocities,
        gt_selected_range_bins=gt_selected_bins,
        sim_selected_range_bins=sim_selected_bins,
    )
    np.savez_compressed(
        stft_npz_path,
        gt_spectrum=gt_stft,
        gt_spectrum_db=gt_stft_db,
        gt_time_s=gt_stft_times,
        gt_velocity_mps=gt_stft_velocities,
        sim_spectrum=sim_stft,
        sim_spectrum_db=sim_stft_db,
        sim_time_s=sim_stft_times,
        sim_velocity_mps=sim_stft_velocities,
        gt_selected_range_bins=gt_selected_bins,
        sim_selected_range_bins=sim_selected_bins,
        gt_stft_range_bins=gt_stft_range_bins[0],
        sim_stft_range_bins=sim_stft_range_bins[0],
    )
    save_compare_plot(png_path, gt_db, sim_db, times, velocities, args.db_floor)

    metrics = {
        "db_correlation": normalized_corr(gt_db, sim_db),
        "db_mae": float(np.mean(np.abs(sim_db - gt_db))),
        "db_rmse": float(np.sqrt(np.mean((sim_db - gt_db) ** 2))),
        "stft_db_correlation": normalized_corr(gt_stft_db, sim_stft_db),
        "stft_db_mae": float(np.mean(np.abs(sim_stft_db - gt_stft_db))),
        "stft_db_rmse": float(np.sqrt(np.mean((sim_stft_db - gt_stft_db) ** 2))),
        "gt_column_energy_ratio": float(np.max(gt_spectrum.sum(axis=0)) / np.maximum(np.min(gt_spectrum.sum(axis=0)), 1e-30)),
        "sim_column_energy_ratio": float(np.max(sim_spectrum.sum(axis=0)) / np.maximum(np.min(sim_spectrum.sum(axis=0)), 1e-30)),
    }
    summary = {
        "sequence": "mmRadPose p6/angle0/action7/repetition0 frames 0-149",
        "processing": (
            "SMPL-X mesh -> WiTwin raw ADC echo cube -> same mmRadPose raw postprocess "
            "(fast-time mean removal, Hann range FFT, slow-time mean removal, Doppler FFT, "
            f"sim {args.mesh_placement} / GT {gt_roi_placement} "
            "range ROI aggregation)"
        ),
        "sources": {
            "fit_npz": str(args.fit_npz.resolve()),
            "raw_npz": str(args.raw_npz.resolve()),
            "gt_source": gt_source,
            "gt_npz": str(args.gt_npz.resolve()) if args.gt_npz.exists() else None,
            "skeleton_npy": str(args.skeleton_npy.resolve()),
            "targetlist_npy": str(args.targetlist_npy.resolve()),
        },
        "radar": {
            "device": "TI IWR6843AOPEVM + MMWAVEICBOOST + DCA1000EVM, represented as one colocated WiTwin TX/RX channel",
            "center_frequency_hz": CENTER_FREQUENCY_HZ,
            "adc_samples_per_chirp": ADC_SAMPLES,
            "chirps_per_frame": CHIRPS_PER_FRAME,
            "frame_rate_hz": FRAME_RATE_HZ,
            "adc_sampling_frequency_hz": ADC_SAMPLE_RATE_HZ,
            "chirp_frequency_hz": CHIRP_FREQUENCY_HZ,
            "bandwidth_hz_approx": float(mmradpose_radar_config()["slope"]) * float(mmradpose_radar_config()["ramp_end_time"]) * 1e6,
            "range_resolution_m": RANGE_RESOLUTION_M,
            "doppler_resolution_mps": DOPPLER_RESOLUTION_MPS,
        },
        "roi": {
            "mesh_placement": args.mesh_placement,
            "gt_roi_placement": gt_roi_placement,
            "roi_mode": args.roi_mode,
            "sim": {
                **sim_roi_info,
                "range_source": (
                    "one sequence-level median targetlist/skeleton range translation"
                    if args.mesh_placement == "gt-fixed"
                    else "per-frame targetlist/skeleton range translation"
                    if args.mesh_placement == "gt-aligned"
                    else "fit mesh y-range at old STFT position"
                ),
                "range_bins_min": int(min(b[0] for b in sim_range_bins)),
                "range_bins_max": int(max(b[-1] for b in sim_range_bins)),
                "stft_range_bins_min": int(sim_stft_range_bins[0][0]),
                "stft_range_bins_max": int(sim_stft_range_bins[0][-1]),
                "range_offset_m_mean": float(np.mean(sim_range_offset)),
            },
            "gt": {
                **gt_roi_info,
                "range_source": "targetlist-aligned skeleton" if gt_roi_placement == "gt-aligned" else "fit mesh y-range at old STFT position",
                "range_bins_min": int(min(b[0] for b in gt_range_bins)),
                "range_bins_max": int(max(b[-1] for b in gt_range_bins)),
                "stft_range_bins_min": int(gt_stft_range_bins[0][0]),
                "stft_range_bins_max": int(gt_stft_range_bins[0][-1]),
                "range_offset_m_mean": float(np.mean(gt_range_offset)),
            },
        },
        "simulation": sim_info,
        "metrics": metrics,
        "outputs": {
            "sim_echo_cube_npz": str(sim_cube_path.resolve()),
            "sim_doppler_npz": str(sim_npz_path.resolve()),
            "compare_npz": str(compare_npz_path.resolve()),
            "stft_compare_npz": str(stft_npz_path.resolve()),
            "compare_png": str(png_path.resolve()),
            "stft_compare_png": str(stft_png_path.resolve()),
            "summary_json": str(json_path.resolve()),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
