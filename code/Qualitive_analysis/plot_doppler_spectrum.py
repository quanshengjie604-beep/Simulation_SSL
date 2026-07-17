#!/usr/bin/env python3
"""Plot a max-normalized Doppler spectrum time window for qualitative analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rtpose-qual-doppler")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read <sequence>.npy from a Doppler spectrum directory and render a selected "
            "time window as max-normalized dB heatmap."
        )
    )
    parser.add_argument(
        "--spectrum-dir",
        required=True,
        help="Directory containing Doppler spectra named <sequence>.npy.",
    )
    parser.add_argument("--sequence", required=True, help="Sequence ID; reads <spectrum-dir>/<sequence>.npy.")
    parser.add_argument(
        "--start-time",
        "--time-start",
        dest="start_time",
        type=float,
        required=True,
        help="Inclusive start time in seconds.",
    )
    parser.add_argument(
        "--stop-time",
        "--time-stop",
        dest="stop_time",
        type=float,
        required=True,
        help="Inclusive stop time in seconds.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Radar frame rate used to map columns to seconds.")
    parser.add_argument(
        "--first-frame-id",
        type=int,
        required=True,
        help="Radar_frameID represented by column 0 of the spectrum.",
    )
    parser.add_argument("--db-floor", type=float, default=-45.0, help="Minimum displayed relative power in dB.")
    parser.add_argument("--db-ceil", type=float, default=0.0, help="Maximum displayed relative power in dB.")
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap.")
    parser.add_argument("--width", type=float, default=10.5, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=5.2, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=160, help="Output DPI.")
    parser.add_argument(
        "--out",
        default="",
        help="Output PNG path; defaults to results/Qualitive_analysis/doppler_spectrum_plots/.",
    )
    return parser.parse_args()


def load_spectrum(path: Path) -> np.ndarray:
    spectrum = np.load(path)
    if spectrum.ndim != 2:
        raise ValueError(f"{path} has shape {spectrum.shape}; expected 2D Doppler x Time array")
    if spectrum.shape[0] <= 0 or spectrum.shape[1] <= 0:
        raise ValueError(f"{path} is empty: shape={spectrum.shape}")
    return spectrum.astype(np.float32, copy=False)


def time_axis(num_frames: int, first_frame_id: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    if fps <= 0.0:
        raise ValueError("--fps must be positive")
    frame_ids = int(first_frame_id) + np.arange(num_frames, dtype=np.int64)
    times = (frame_ids.astype(np.float64) - 1.0) / float(fps)
    return frame_ids, times


def select_time_window(
    spectrum: np.ndarray,
    frame_ids: np.ndarray,
    times: np.ndarray,
    start_time: float,
    stop_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stop_time < start_time:
        raise ValueError("--stop-time must be greater than or equal to --start-time")
    keep = (times >= float(start_time)) & (times <= float(stop_time))
    if not np.any(keep):
        available = f"{times[0]:.3f}..{times[-1]:.3f}s"
        raise ValueError(f"No spectrum columns in requested time window; available time range is {available}")
    return spectrum[:, keep], frame_ids[keep], times[keep]


def max_normalized_db(spectrum: np.ndarray, floor_db: float, ceil_db: float) -> np.ndarray:
    finite = np.isfinite(spectrum)
    nonnegative = np.zeros_like(spectrum, dtype=np.float32)
    nonnegative[finite] = np.maximum(spectrum[finite], 0.0)
    peak = float(np.max(nonnegative)) if np.any(finite) else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.full_like(nonnegative, float(floor_db), dtype=np.float32)
    db = 10.0 * np.log10(np.clip(nonnegative / peak, 1e-12, None))
    return np.clip(db, float(floor_db), float(ceil_db)).astype(np.float32, copy=False)


def default_out_path(sequence: str, start_time: float, stop_time: float) -> Path:
    safe_start = f"{start_time:.3f}".replace(".", "p").replace("-", "m")
    safe_stop = f"{stop_time:.3f}".replace(".", "p").replace("-", "m")
    return (
        REPO_ROOT
        / "results"
        / "Qualitive_analysis"
        / "doppler_spectrum_plots"
        / f"seq{sequence}_{safe_start}s_{safe_stop}s_doppler.png"
    )


def plot_spectrum(
    db_map: np.ndarray,
    frame_ids: np.ndarray,
    times: np.ndarray,
    sequence: str,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    frame_period = 1.0 / float(args.fps)
    x0 = float(times[0] - 0.5 * frame_period)
    x1 = float(times[-1] + 0.5 * frame_period)
    if db_map.shape[1] == 1:
        x0 = float(times[0] - 0.5 * frame_period)
        x1 = float(times[0] + 0.5 * frame_period)

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi, constrained_layout=True)
    im = ax.imshow(
        db_map,
        origin="lower",
        aspect="auto",
        cmap=args.cmap,
        vmin=float(args.db_floor),
        vmax=float(args.db_ceil),
        extent=(x0, x1, -0.5, db_map.shape[0] - 0.5),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Relative Doppler power (dB, max-normalized)")
    cbar.set_ticks([float(args.db_floor), -30.0, -15.0, float(args.db_ceil)])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Doppler bin")
    ax.set_title(
        f"Sequence {sequence} ROI Doppler spectrum | "
        f"{times[0]:.3f}-{times[-1]:.3f}s | frames {int(frame_ids[0])}-{int(frame_ids[-1])}"
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.5, db_map.shape[0] - 0.5)
    ax.tick_params(labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    spectrum_path = Path(args.spectrum_dir).expanduser().resolve() / f"{args.sequence}.npy"
    spectrum = load_spectrum(spectrum_path)
    frame_ids, times = time_axis(spectrum.shape[1], args.first_frame_id, args.fps)
    window, window_frames, window_times = select_time_window(
        spectrum,
        frame_ids,
        times,
        args.start_time,
        args.stop_time,
    )
    db_map = max_normalized_db(window, args.db_floor, args.db_ceil)
    out_path = Path(args.out).expanduser().resolve() if args.out else default_out_path(
        args.sequence,
        args.start_time,
        args.stop_time,
    )
    plot_spectrum(db_map, window_frames, window_times, args.sequence, args, out_path)
    print(
        f"wrote {out_path} shape={db_map.shape} "
        f"time={window_times[0]:.3f}..{window_times[-1]:.3f}s "
        f"frames={int(window_frames[0])}..{int(window_frames[-1])}"
    )


if __name__ == "__main__":
    main()
