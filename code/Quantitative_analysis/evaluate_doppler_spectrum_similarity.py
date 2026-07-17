#!/usr/bin/env python3
"""Evaluate similarity between two ROI Doppler spectrum files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "Echo_data_processing"))

from raw_echo_to_xyz import RadarConfig  # noqa: E402


def ssim_2d(a: np.ndarray, b: np.ndarray, window_size: int = 7, data_range: float | None = None) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return float("nan")
    fill_x = float(x[finite].mean())
    fill_y = float(y[finite].mean())
    x = np.where(finite, x, fill_x)
    y = np.where(finite, y, fill_y)
    if data_range is None:
        data_range_value = max(float(x.max()), float(y.max())) - min(float(x.min()), float(y.min()))
    else:
        data_range_value = float(data_range)
    if data_range_value <= 0.0:
        return 1.0
    size = max(3, min(int(window_size), x.shape[0], x.shape[1]))
    if size % 2 == 0:
        size -= 1
    c1 = (0.01 * data_range_value) ** 2
    c2 = (0.03 * data_range_value) ** 2
    ux = uniform_filter(x, size=size, mode="reflect")
    uy = uniform_filter(y, size=size, mode="reflect")
    uxx = uniform_filter(x * x, size=size, mode="reflect")
    uyy = uniform_filter(y * y, size=size, mode="reflect")
    uxy = uniform_filter(x * y, size=size, mode="reflect")
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    num = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    valid = den != 0.0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(num[valid] / den[valid]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two Doppler spectra over a required time window. Each spectrum is "
            "max-normalized in the selected common window before SSIM and velocity metrics."
        )
    )
    parser.add_argument("--reference", required=True, help="Reference .npy spectrum path, shape Doppler x Time.")
    parser.add_argument("--candidate", required=True, help="Candidate .npy spectrum path, shape Doppler x Time.")
    parser.add_argument("--reference-label", default="reference", help="Label used in reports.")
    parser.add_argument("--candidate-label", default="candidate", help="Label used in reports.")
    parser.add_argument(
        "--reference-first-frame-id",
        type=int,
        required=True,
        help="Radar_frameID represented by column 0 of --reference.",
    )
    parser.add_argument(
        "--candidate-first-frame-id",
        type=int,
        required=True,
        help="Radar_frameID represented by column 0 of --candidate.",
    )
    parser.add_argument("--start-time", type=float, required=True, help="Inclusive start time in seconds.")
    parser.add_argument("--stop-time", type=float, required=True, help="Inclusive stop time in seconds.")
    parser.add_argument("--fps", type=float, default=10.0, help="Radar frame rate used to map frame IDs to seconds.")
    parser.add_argument(
        "--ssim-clip",
        nargs=2,
        type=float,
        required=True,
        metavar=("LOW", "HIGH"),
        help="Required dB clip range for SSIM display mapping, e.g. --ssim-clip -20 0.",
    )
    parser.add_argument(
        "--spectrum-clip",
        nargs=2,
        type=float,
        required=True,
        metavar=("LOW", "HIGH"),
        help="Required dB clip range for centroid/peak/Wasserstein weights, e.g. --spectrum-clip -45 0.",
    )
    parser.add_argument("--ssim-window", type=int, default=7, help="SSIM uniform-filter window size.")
    parser.add_argument("--out-json", default="", help="Optional JSON output path.")
    parser.add_argument("--out-csv", default="", help="Optional one-row CSV output path.")
    args = parser.parse_args()
    args.ssim_clip = validate_clip(args.ssim_clip, "--ssim-clip")
    args.spectrum_clip = validate_clip(args.spectrum_clip, "--spectrum-clip")
    return args


def validate_clip(values: list[float], name: str) -> tuple[float, float]:
    low, high = float(values[0]), float(values[1])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"{name} must be finite and increasing, got {values}")
    return low, high


def load_spectrum(path: Path) -> np.ndarray:
    spectrum = np.load(path)
    if spectrum.ndim != 2:
        raise ValueError(f"{path} has shape {spectrum.shape}; expected Doppler x Time 2D array")
    if spectrum.shape[0] <= 0 or spectrum.shape[1] <= 0:
        raise ValueError(f"{path} is empty: shape={spectrum.shape}")
    return spectrum.astype(np.float64, copy=False)


def frame_ids_for_columns(num_frames: int, first_frame_id: int) -> np.ndarray:
    return int(first_frame_id) + np.arange(int(num_frames), dtype=np.int64)


def select_window_columns(
    frame_ids: np.ndarray,
    fps: float,
    start_time: float,
    stop_time: float,
) -> np.ndarray:
    if fps <= 0.0:
        raise ValueError("--fps must be positive")
    if stop_time < start_time:
        raise ValueError("--stop-time must be greater than or equal to --start-time")
    times = (frame_ids.astype(np.float64) - 1.0) / float(fps)
    return np.flatnonzero((times >= float(start_time)) & (times <= float(stop_time)))


def align_common_time_window(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_first_frame_id: int,
    candidate_first_frame_id: int,
    fps: float,
    start_time: float,
    stop_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    common_doppler = min(reference.shape[0], candidate.shape[0])
    reference = reference[:common_doppler]
    candidate = candidate[:common_doppler]

    ref_frames_all = frame_ids_for_columns(reference.shape[1], reference_first_frame_id)
    cand_frames_all = frame_ids_for_columns(candidate.shape[1], candidate_first_frame_id)
    ref_cols = select_window_columns(ref_frames_all, fps, start_time, stop_time)
    cand_cols = select_window_columns(cand_frames_all, fps, start_time, stop_time)
    if ref_cols.size == 0:
        raise ValueError("No reference columns inside requested time window")
    if cand_cols.size == 0:
        raise ValueError("No candidate columns inside requested time window")

    ref_frames = ref_frames_all[ref_cols]
    cand_frames = cand_frames_all[cand_cols]
    common_frames = np.intersect1d(ref_frames, cand_frames, assume_unique=True)
    if common_frames.size == 0:
        raise ValueError(
            "No common Radar_frameID between spectra inside requested time window. "
            f"reference frames {int(ref_frames[0])}..{int(ref_frames[-1])}, "
            f"candidate frames {int(cand_frames[0])}..{int(cand_frames[-1])}"
        )

    ref_lookup = {int(frame): int(col) for frame, col in zip(ref_frames_all, range(ref_frames_all.size))}
    cand_lookup = {int(frame): int(col) for frame, col in zip(cand_frames_all, range(cand_frames_all.size))}
    ref_common_cols = np.asarray([ref_lookup[int(frame)] for frame in common_frames], dtype=np.int64)
    cand_common_cols = np.asarray([cand_lookup[int(frame)] for frame in common_frames], dtype=np.int64)
    times = (common_frames.astype(np.float64) - 1.0) / float(fps)
    summary = {
        "requested_start_time_s": float(start_time),
        "requested_stop_time_s": float(stop_time),
        "actual_start_time_s": float(times[0]),
        "actual_stop_time_s": float(times[-1]),
        "start_radar_frame": int(common_frames[0]),
        "stop_radar_frame": int(common_frames[-1]),
        "num_frames": int(common_frames.size),
        "fps": float(fps),
        "reference_first_frame_id": int(reference_first_frame_id),
        "candidate_first_frame_id": int(candidate_first_frame_id),
        "common_doppler_bins": int(common_doppler),
    }
    return reference[:, ref_common_cols], candidate[:, cand_common_cols], common_frames, summary


def velocity_axis_mps(num_bins: int) -> tuple[np.ndarray, dict[str, float | int | str]]:
    cfg = RadarConfig()
    light_speed = 3.0e8
    wavelength = light_speed / float(cfg.start_freq_const)
    effective_chirp_period = (
        float(cfg.chirp_idle_time) + float(cfg.chirp_ramp_end_time)
    ) * float(cfg.num_chirps_in_loop)
    resolution = wavelength / (2.0 * float(cfg.nchirp_loops) * effective_chirp_period)
    bins = np.arange(num_bins, dtype=np.float64) - num_bins // 2
    velocities = bins * resolution
    return velocities, {
        "unit": "m/s",
        "num_bins": int(num_bins),
        "lambda_m": float(wavelength),
        "effective_chirp_period_s": float(effective_chirp_period),
        "bin_resolution_mps": float(resolution),
        "min_mps": float(velocities[0]),
        "zero_bin_index": int(num_bins // 2),
        "max_mps": float(velocities[-1]),
    }


def max_normalized_power(spectrum: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(spectrum)
    power = np.zeros_like(spectrum, dtype=np.float64)
    power[finite] = np.maximum(spectrum[finite], 0.0)
    peak = float(np.max(power)) if np.any(finite) else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(power), peak
    return power / peak, peak


def normalized_db(normalized_power: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.clip(normalized_power, np.finfo(np.float64).tiny, None))


def db_to_unit_interval(normalized_power: np.ndarray, clip_range: tuple[float, float]) -> np.ndarray:
    low, high = clip_range
    db = normalized_db(normalized_power)
    return np.clip((db - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def spectrum_weights(normalized_power: np.ndarray, clip_range: tuple[float, float]) -> np.ndarray:
    low, high = clip_range
    low_linear = 10.0 ** (low / 10.0)
    high_linear = 10.0 ** (high / 10.0)
    clipped = np.minimum(normalized_power, high_linear)
    return np.where(normalized_power >= low_linear, clipped, 0.0).astype(np.float64, copy=False)


def frame_distributions(weights: np.ndarray) -> np.ndarray:
    sums = weights.sum(axis=0, keepdims=True)
    out = np.zeros_like(weights, dtype=np.float64)
    valid = sums[0] > 0.0
    out[:, valid] = weights[:, valid] / sums[:, valid]
    return out


def centroid_series(weights: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    dist = frame_distributions(weights)
    valid = dist.sum(axis=0) > 0.0
    out = np.full(weights.shape[1], np.nan, dtype=np.float64)
    out[valid] = velocities @ dist[:, valid]
    return out


def peak_velocity_series(weights: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    valid = weights.sum(axis=0) > 0.0
    out = np.full(weights.shape[1], np.nan, dtype=np.float64)
    if np.any(valid):
        out[valid] = velocities[np.argmax(weights[:, valid], axis=0)]
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    x = a[finite] - float(np.mean(a[finite]))
    y = b[finite] - float(np.mean(b[finite]))
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan"), 0
    return float(np.mean(np.abs(a[finite] - b[finite]))), int(finite.sum())


def wasserstein_per_frame(weights_a: np.ndarray, weights_b: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    dist_a = frame_distributions(weights_a)
    dist_b = frame_distributions(weights_b)
    valid = (dist_a.sum(axis=0) > 0.0) & (dist_b.sum(axis=0) > 0.0)
    out = np.full(weights_a.shape[1], np.nan, dtype=np.float64)
    if velocities.size < 2 or not np.any(valid):
        return out
    delta = float(np.mean(np.diff(velocities)))
    cdf_diff = np.abs(np.cumsum(dist_a[:, valid] - dist_b[:, valid], axis=0))
    out[valid] = cdf_diff.sum(axis=0) * abs(delta)
    return out


def evaluate(reference: np.ndarray, candidate: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    ref_norm, ref_peak_power = max_normalized_power(reference)
    cand_norm, cand_peak_power = max_normalized_power(candidate)
    ref_ssim = db_to_unit_interval(ref_norm, args.ssim_clip)
    cand_ssim = db_to_unit_interval(cand_norm, args.ssim_clip)
    ssim_value = float(ssim_2d(ref_ssim, cand_ssim, window_size=args.ssim_window, data_range=1.0))

    ref_weights = spectrum_weights(ref_norm, args.spectrum_clip)
    cand_weights = spectrum_weights(cand_norm, args.spectrum_clip)
    velocities, velocity_summary = velocity_axis_mps(reference.shape[0])

    ref_centroid = centroid_series(ref_weights, velocities)
    cand_centroid = centroid_series(cand_weights, velocities)
    centroid_mae, centroid_valid = mean_abs_error(ref_centroid, cand_centroid)

    ref_peak = peak_velocity_series(ref_weights, velocities)
    cand_peak = peak_velocity_series(cand_weights, velocities)
    peak_mae, peak_valid = mean_abs_error(ref_peak, cand_peak)

    wasserstein = wasserstein_per_frame(ref_weights, cand_weights, velocities)
    valid_wasserstein = np.isfinite(wasserstein)
    wasserstein_mean = float(np.nanmean(wasserstein)) if np.any(valid_wasserstein) else float("nan")

    return {
        "metrics": {
            "ssim": ssim_value,
            "centroid_correlation": pearson(ref_centroid, cand_centroid),
            "centroid_mae_mps": centroid_mae,
            "peak_velocity_correlation": pearson(ref_peak, cand_peak),
            "peak_velocity_mae_mps": peak_mae,
            "wasserstein_1d_mean_mps": wasserstein_mean,
        },
        "valid_frames": {
            "centroid": centroid_valid,
            "peak_velocity": peak_valid,
            "wasserstein": int(valid_wasserstein.sum()),
        },
        "peaks": {
            "reference_power_peak": float(ref_peak_power),
            "candidate_power_peak": float(cand_peak_power),
        },
        "velocity_axis": velocity_summary,
        "normalization": {
            "power": "each input is max-normalized within the aligned evaluation window",
            "ssim": (
                "db=10*log10(max_normalized_power); "
                f"clip {args.ssim_clip[0]:g}..{args.ssim_clip[1]:g} dB maps linearly to 0..1; "
                "outside values are clamped to 0 or 1"
            ),
            "spectrum_metrics": (
                "centroid, peak velocity, and 1D Wasserstein use max-normalized linear power "
                f"within {args.spectrum_clip[0]:g}..{args.spectrum_clip[1]:g} dB; "
                "values below the lower bound have zero weight"
            ),
        },
    }


def flatten_for_csv(summary: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {}

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}{key}_" if prefix else f"{key}_", child)
        elif isinstance(value, (list, tuple)):
            row[prefix[:-1]] = " ".join(str(item) for item in value)
        else:
            row[prefix[:-1]] = value

    visit("", summary)
    return row


def write_csv(path: Path, summary: dict[str, object]) -> None:
    row = flatten_for_csv(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    ref_path = Path(args.reference).expanduser().resolve()
    cand_path = Path(args.candidate).expanduser().resolve()
    reference_full = load_spectrum(ref_path)
    candidate_full = load_spectrum(cand_path)
    reference, candidate, common_frames, window = align_common_time_window(
        reference_full,
        candidate_full,
        args.reference_first_frame_id,
        args.candidate_first_frame_id,
        args.fps,
        args.start_time,
        args.stop_time,
    )
    result = evaluate(reference, candidate, args)
    summary: dict[str, object] = {
        "inputs": {
            "reference": str(ref_path),
            "candidate": str(cand_path),
            "reference_label": str(args.reference_label),
            "candidate_label": str(args.candidate_label),
        },
        "window": window,
        "clip_ranges_db": {
            "ssim_clip": [float(args.ssim_clip[0]), float(args.ssim_clip[1])],
            "spectrum_clip": [float(args.spectrum_clip[0]), float(args.spectrum_clip[1])],
        },
        "shape": {
            "reference_window": [int(x) for x in reference.shape],
            "candidate_window": [int(x) for x in candidate.shape],
        },
        "common_radar_frames": [int(common_frames[0]), int(common_frames[-1])],
        "ssim_window": int(args.ssim_window),
        **result,
    }

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.out_csv:
        write_csv(Path(args.out_csv).expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
