#!/usr/bin/env python3
"""Generate a GT-person ROI Doppler-time spectrum from mmRadPose raw radar cubes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "datasets/mmRadPose/mmRadPose_rawdata/radar/data_cube_parsed_p6_an0_ac7_r0.npz"
DEFAULT_SKELETON = REPO_ROOT / "datasets/mmRadPose/mmRadPose_rawdata/skeletons/skeleton_p6_an0_ac7_r0.npy"
DEFAULT_TARGETLIST = REPO_ROOT / "datasets/mmRadPose/mmRadPose_pointclouds/p6/angle0/7/0/targetlist_64.npy"
DEFAULT_OUT_DIR = REPO_ROOT / "results/mmradpose_raw_doppler_time"

FRAME_RATE_HZ = 15.0
NUM_ADC_SAMPLES = 64
NUM_CHIRPS = 128
RANGE_RESOLUTION_M = 0.148
DOPPLER_RESOLUTION_MPS = 0.078
UNAMBIGUOUS_DOPPLER_MPS = 5.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-npz", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--skeleton-npy", type=Path, default=DEFAULT_SKELETON)
    parser.add_argument("--targetlist-npy", type=Path, default=DEFAULT_TARGETLIST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default="mmradpose_p6_an0_ac7_r0_F0_149_gt_target_aligned_range_roi")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--range-margin-m", type=float, default=0.50)
    parser.add_argument(
        "--roi-mode",
        choices=("per-frame", "fixed-union", "tracked-smooth"),
        default="fixed-union",
        help="Use a per-frame GT range ROI or one fixed ROI covering the selected motion.",
    )
    parser.add_argument("--tracked-width-m", type=float, default=2.0)
    parser.add_argument("--tracked-smooth-frames", type=int, default=9)
    parser.add_argument("--range-resolution-m", type=float, default=RANGE_RESOLUTION_M)
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cube(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if data.files != ["arr_0"]:
            raise ValueError(f"Expected a single arr_0 in {path}, got {data.files}")
        cube = data["arr_0"]
    if cube.ndim != 5:
        raise ValueError(f"Expected raw cube with 5 dims, got {cube.shape}")
    if cube.shape[-2:] != (NUM_ADC_SAMPLES, NUM_CHIRPS):
        raise ValueError(f"Expected last dims (64, 128), got {cube.shape[-2:]}")
    return cube


def skeleton_range_bins(
    skeleton: np.ndarray,
    targetlist: np.ndarray | None,
    start_frame: int,
    num_frames: int,
    range_resolution_m: float,
    margin_m: float,
    num_range_bins: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    joints = skeleton.reshape(-1, 26, 3)
    selected = joints[start_frame : start_frame + num_frames]
    if selected.shape[0] != num_frames:
        raise ValueError(f"Skeleton has only {joints.shape[0]} frames; need {start_frame + num_frames}")
    # mmRadPose coordinates are x=lateral, y=range/depth, z=height.
    skeleton_min = np.nanmin(selected[:, :, 1], axis=1)
    skeleton_max = np.nanmax(selected[:, :, 1], axis=1)
    range_offset = np.zeros(num_frames, dtype=np.float32)
    if targetlist is not None:
        target_selected = targetlist[start_frame : start_frame + num_frames]
        if target_selected.shape[0] != num_frames:
            raise ValueError(f"Targetlist has only {targetlist.shape[0]} frames; need {start_frame + num_frames}")
        skeleton_center = 0.5 * (skeleton_min + skeleton_max)
        for idx, points in enumerate(target_selected):
            valid = np.any(np.abs(points[:, :3]) > 0.0, axis=1) & (points[:, 6] > 0.0)
            if not np.any(valid):
                continue
            valid_points = points[valid]
            weights = valid_points[:, 6]
            keep = max(5, int(np.ceil(0.25 * valid_points.shape[0])))
            top_ids = np.argsort(weights)[-keep:]
            # targetlist column 1 is the radar range/depth coordinate in meters.
            target_range_center = float(np.median(valid_points[top_ids, 1]))
            range_offset[idx] = target_range_center - float(skeleton_center[idx])
    range_min = skeleton_min + range_offset - margin_m
    range_max = skeleton_max + range_offset + margin_m
    range_min = np.maximum(range_min, 0.0)
    range_bins = []
    for low, high in zip(range_min, range_max):
        lo = int(np.floor(low / range_resolution_m))
        hi = int(np.ceil(high / range_resolution_m))
        lo = min(max(lo, 0), num_range_bins - 1)
        hi = min(max(hi, lo), num_range_bins - 1)
        range_bins.append(np.arange(lo, hi + 1, dtype=np.int16))
    return range_bins, range_min.astype(np.float32), range_max.astype(np.float32), range_offset.astype(np.float32)


def fixed_union_range_bins(
    range_min: np.ndarray,
    range_max: np.ndarray,
    range_resolution_m: float,
    num_range_bins: int,
) -> tuple[list[np.ndarray], float, float]:
    union_min = float(np.min(range_min))
    union_max = float(np.max(range_max))
    lo = min(max(int(np.floor(union_min / range_resolution_m)), 0), num_range_bins - 1)
    hi = min(max(int(np.ceil(union_max / range_resolution_m)), lo), num_range_bins - 1)
    fixed = np.arange(lo, hi + 1, dtype=np.int16)
    return [fixed for _ in range(range_min.shape[0])], lo * range_resolution_m, hi * range_resolution_m


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return values.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def tracked_smooth_range_bins(
    range_min: np.ndarray,
    range_max: np.ndarray,
    range_resolution_m: float,
    num_range_bins: int,
    width_m: float,
    smooth_frames: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    center = 0.5 * (range_min + range_max)
    center = smooth_1d(center, smooth_frames)
    half = 0.5 * float(width_m)
    smooth_min = np.maximum(center - half, 0.0)
    smooth_max = center + half
    range_bins = []
    for low, high in zip(smooth_min, smooth_max):
        lo = min(max(int(np.floor(low / range_resolution_m)), 0), num_range_bins - 1)
        hi = min(max(int(np.ceil(high / range_resolution_m)), lo), num_range_bins - 1)
        range_bins.append(np.arange(lo, hi + 1, dtype=np.int16))
    return range_bins, smooth_min.astype(np.float32), smooth_max.astype(np.float32)


def range_doppler_power(frame: np.ndarray) -> np.ndarray:
    """Return range x Doppler power averaged over the two antenna axes."""
    # Frame layout is (..., fast_time=64, slow_time=128). The two leading axes are
    # parsed antenna dimensions; for a scalar Doppler-time spectrum we power-average them.
    adc = frame.astype(np.complex64, copy=False)
    adc = adc - adc.mean(axis=-2, keepdims=True)
    n = np.arange(1, adc.shape[-2] + 1, dtype=np.float32) / (adc.shape[-2] + 1)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n)).astype(np.float32)
    range_fft = np.fft.fft(adc * window[None, None, :, None], n=adc.shape[-2], axis=-2)
    range_fft = range_fft - range_fft.mean(axis=-1, keepdims=True)
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, n=adc.shape[-1], axis=-1), axes=-1)
    power = np.abs(doppler_fft) ** 2
    return power.mean(axis=(0, 1)).astype(np.float32, copy=False)


def power_to_db(power: np.ndarray, floor_db: float) -> np.ndarray:
    safe = np.maximum(power, np.finfo(np.float32).tiny)
    db = 10.0 * np.log10(safe)
    db -= float(np.nanmax(db))
    return np.maximum(db, floor_db).astype(np.float32)


def save_plot(path: Path, spectrum_db: np.ndarray, times: np.ndarray, velocities: np.ndarray, floor_db: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "logs/.cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    extent = [times[0], times[-1] + 1.0 / FRAME_RATE_HZ, velocities[0], velocities[-1]]
    im = ax.imshow(
        spectrum_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=floor_db,
        vmax=0.0,
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Radial velocity (m/s)")
    ax.set_title("mmRadPose raw radar GT-person ROI Doppler-time")
    fig.colorbar(im, ax=ax, label="Power (dB, normalized)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / f"{args.label}.npz"
    png_path = args.out_dir / f"{args.label}.png"
    json_path = args.out_dir / f"{args.label}.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"{result_path} exists; pass --overwrite")

    cube = load_cube(args.raw_npz)
    skeleton = np.load(args.skeleton_npy, allow_pickle=False)
    targetlist = np.load(args.targetlist_npy, allow_pickle=False).astype(np.float32, copy=False) if args.targetlist_npy.exists() else None
    num_frames = int(np.ceil(args.duration_s * FRAME_RATE_HZ))
    num_frames = min(num_frames, cube.shape[0] - args.start_frame, skeleton.reshape(-1, 26, 3).shape[0] - args.start_frame)
    times = (args.start_frame + np.arange(num_frames, dtype=np.float64)) / FRAME_RATE_HZ
    range_bins, range_min, range_max, range_offset = skeleton_range_bins(
        skeleton,
        targetlist,
        args.start_frame,
        num_frames,
        args.range_resolution_m,
        args.range_margin_m,
        cube.shape[-2],
    )
    fixed_roi_min_m = None
    fixed_roi_max_m = None
    if args.roi_mode == "fixed-union":
        range_bins, fixed_roi_min_m, fixed_roi_max_m = fixed_union_range_bins(
            range_min,
            range_max,
            args.range_resolution_m,
            cube.shape[-2],
        )
    elif args.roi_mode == "tracked-smooth":
        range_bins, range_min, range_max = tracked_smooth_range_bins(
            range_min,
            range_max,
            args.range_resolution_m,
            cube.shape[-2],
            args.tracked_width_m,
            args.tracked_smooth_frames,
        )

    spectrum = np.empty((NUM_CHIRPS, num_frames), dtype=np.float32)
    selected_bins_padded = np.full((num_frames, cube.shape[-2]), -1, dtype=np.int16)
    for idx in range(num_frames):
        rd_power = range_doppler_power(cube[args.start_frame + idx])
        bins = range_bins[idx]
        selected_bins_padded[idx, : bins.size] = bins
        spectrum[:, idx] = rd_power[bins].mean(axis=0)
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == num_frames:
            print(
                f"[frame {idx + 1:04d}/{num_frames:04d}] "
                f"t={times[idx]:.3f}s bins={bins[0]}..{bins[-1]} "
                f"range={range_min[idx]:.2f}..{range_max[idx]:.2f}m",
                flush=True,
            )

    spectrum_db = power_to_db(spectrum, args.db_floor)
    velocities = (np.arange(NUM_CHIRPS, dtype=np.float64) - NUM_CHIRPS // 2) * DOPPLER_RESOLUTION_MPS

    np.savez_compressed(
        result_path,
        spectrum=spectrum,
        spectrum_db=spectrum_db,
        time_s=times,
        velocity_mps=velocities,
        selected_range_bins=selected_bins_padded,
        skeleton_range_min_m=range_min,
        skeleton_range_max_m=range_max,
        skeleton_range_offset_m=range_offset,
        raw_cube_shape=np.asarray(cube.shape, dtype=np.int64),
        range_resolution_m=np.asarray(args.range_resolution_m, dtype=np.float32),
    )
    save_plot(png_path, spectrum_db, times, velocities, args.db_floor)

    summary = {
        "processing": (
            "mmRadPose parsed raw radar cube -> fast-time mean removal + Hann range FFT -> "
            "slow-time mean removal + Doppler FFT -> GT skeleton range-ROI aggregation"
        ),
        "source_raw_npz": str(args.raw_npz.resolve()),
        "source_skeleton_npy": str(args.skeleton_npy.resolve()),
        "source_targetlist_npy": str(args.targetlist_npy.resolve()) if targetlist is not None else None,
        "sequence": "p6_an0_ac7_r0",
        "frames": {
            "start_frame": args.start_frame,
            "num_frames": int(num_frames),
            "frame_rate_hz": FRAME_RATE_HZ,
            "duration_s": float(num_frames / FRAME_RATE_HZ),
            "time_start_s": float(times[0]),
            "time_stop_s": float(times[-1] + 1.0 / FRAME_RATE_HZ),
        },
        "radar": {
            "device": "TI IWR6843AOPEVM + MMWAVEICBOOST + DCA1000EVM",
            "center_frequency_ghz": 60.0,
            "tx_rx": "3 TX / 4 RX reported by paper; parsed cube has two 4-element antenna axes",
            "raw_cube_shape": list(map(int, cube.shape)),
            "samples_per_chirp": NUM_ADC_SAMPLES,
            "chirps_per_frame": NUM_CHIRPS,
            "adc_sampling_frequency_mhz": 3.8,
            "bandwidth_ghz": 1.02,
            "range_resolution_m": args.range_resolution_m,
            "doppler_resolution_mps": DOPPLER_RESOLUTION_MPS,
            "unambiguous_doppler_mps": UNAMBIGUOUS_DOPPLER_MPS,
            "unambiguous_range_m": 9.49,
        },
        "roi": {
            "coordinate": "mmRadPose skeleton y-axis range/depth",
            "range_alignment": (
                "per-frame translation from skeleton range center to median range of top-power targetlist points"
                if targetlist is not None
                else "none"
            ),
            "roi_mode": args.roi_mode,
            "range_margin_m": args.range_margin_m,
            "tracked_width_m": args.tracked_width_m if args.roi_mode == "tracked-smooth" else None,
            "tracked_smooth_frames": args.tracked_smooth_frames if args.roi_mode == "tracked-smooth" else None,
            "range_bins_min": int(min(b[0] for b in range_bins)),
            "range_bins_max": int(max(b[-1] for b in range_bins)),
            "fixed_roi_min_m": fixed_roi_min_m,
            "fixed_roi_max_m": fixed_roi_max_m,
            "range_min_m_mean": float(np.mean(range_min)),
            "range_max_m_mean": float(np.mean(range_max)),
            "range_offset_m_mean": float(np.mean(range_offset)),
        },
        "outputs": {
            "npz": str(result_path.resolve()),
            "png": str(png_path.resolve()),
            "json": str(json_path.resolve()),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
