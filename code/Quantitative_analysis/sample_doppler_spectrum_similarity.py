#!/usr/bin/env python3
"""Compare Doppler spectra with random fixed-duration samples per sequence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "Quantitative_analysis"))

from evaluate_doppler_spectrum_similarity import (  # noqa: E402
    centroid_series,
    db_to_unit_interval,
    max_normalized_power,
    mean_abs_error,
    peak_velocity_series,
    pearson,
    spectrum_weights,
    ssim_2d,
    velocity_axis_mps,
    wasserstein_per_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-spectrum-dir", required=True)
    parser.add_argument("--sim-spectrum-dir", required=True)
    parser.add_argument("--train", default=str(REPO_ROOT / "datasets" / "Train_sp120_train_minus_val6.json"))
    parser.add_argument("--filemeta", default=str(REPO_ROOT / "datasets" / "filemeta.json"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequences", type=int, nargs="*", help="Optional sequence subset; default: all in --train")
    parser.add_argument("--samples-per-sequence", type=int, default=10)
    parser.add_argument("--window-seconds", type=float, default=2.5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--ssim-clip", type=float, nargs=2, default=(-20.0, 0.0))
    parser.add_argument("--spectrum-clip", type=float, nargs=2, default=(-45.0, 0.0))
    parser.add_argument("--ssim-window", type=int, default=7)
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sequence_ids(train: dict[str, object], selected: list[int] | None) -> list[int]:
    if selected:
        return [int(seq) for seq in selected]
    return [int(seq) for seq in sorted(train, key=lambda item: int(item))]


def load_activity_map(filemeta_path: Path) -> dict[int, str]:
    meta = load_json(filemeta_path)
    if not isinstance(meta, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in meta.items():
        if isinstance(value, dict):
            out[int(key)] = str(value.get("Activity", "UNKNOWN"))
        else:
            out[int(key)] = "UNKNOWN"
    return out


def load_start_metadata(spectrum_dir: Path) -> dict[str, dict[str, object]]:
    path = spectrum_dir / "doppler_spectrum_start_frames.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Doppler spectrum metadata: {path}")
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_spectrum_with_frames(spectrum_dir: Path, metadata: dict[str, dict[str, object]], seq: int) -> tuple[np.ndarray, np.ndarray]:
    path = spectrum_dir / f"{seq}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing spectrum file: {path}")
    spectrum = np.load(path).astype(np.float64, copy=False)
    if spectrum.ndim != 2:
        raise ValueError(f"{path} has shape {spectrum.shape}; expected Doppler x Frame")
    item = metadata.get(path.name)
    if not isinstance(item, dict):
        raise KeyError(f"{path.name} missing from {spectrum_dir / 'doppler_spectrum_start_frames.json'}")
    start = int(item["start_frame"])
    frames = start + np.arange(spectrum.shape[1], dtype=np.int64)
    return spectrum, frames


def common_contiguous_starts(common_frames: np.ndarray, window_frames: int) -> np.ndarray:
    if common_frames.size < window_frames:
        return np.empty(0, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(common_frames) != 1) + 1
    segments = np.split(common_frames, breaks)
    starts: list[np.ndarray] = []
    for segment in segments:
        if segment.size >= window_frames:
            starts.append(segment[: segment.size - window_frames + 1])
    if not starts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(starts).astype(np.int64, copy=False)


def take_window(spectrum: np.ndarray, frames: np.ndarray, start_frame: int, window_frames: int) -> np.ndarray:
    wanted = np.arange(start_frame, start_frame + window_frames, dtype=np.int64)
    lookup = {int(frame): idx for idx, frame in enumerate(frames)}
    cols = np.asarray([lookup[int(frame)] for frame in wanted], dtype=np.int64)
    return spectrum[:, cols]


def evaluate_sample(
    gt: np.ndarray,
    sim: np.ndarray,
    ssim_clip: tuple[float, float],
    spectrum_clip: tuple[float, float],
    ssim_window: int,
) -> dict[str, float | int]:
    doppler_bins = min(gt.shape[0], sim.shape[0])
    gt = gt[:doppler_bins]
    sim = sim[:doppler_bins]
    gt_norm, gt_peak_power = max_normalized_power(gt)
    sim_norm, sim_peak_power = max_normalized_power(sim)

    gt_ssim = db_to_unit_interval(gt_norm, ssim_clip)
    sim_ssim = db_to_unit_interval(sim_norm, ssim_clip)
    ssim = ssim_2d(gt_ssim, sim_ssim, window_size=ssim_window, data_range=1.0)

    gt_weights = spectrum_weights(gt_norm, spectrum_clip)
    sim_weights = spectrum_weights(sim_norm, spectrum_clip)
    velocities, _ = velocity_axis_mps(doppler_bins)

    gt_centroid = centroid_series(gt_weights, velocities)
    sim_centroid = centroid_series(sim_weights, velocities)
    centroid_mae, centroid_valid = mean_abs_error(gt_centroid, sim_centroid)

    gt_peak = peak_velocity_series(gt_weights, velocities)
    sim_peak = peak_velocity_series(sim_weights, velocities)
    peak_mae, peak_valid = mean_abs_error(gt_peak, sim_peak)

    wasserstein = wasserstein_per_frame(gt_weights, sim_weights, velocities)
    valid_wasserstein = np.isfinite(wasserstein)

    return {
        "ssim": float(ssim),
        "centroid_correlation": pearson(gt_centroid, sim_centroid),
        "centroid_mae_mps": float(centroid_mae),
        "centroid_valid_frames": int(centroid_valid),
        "peak_velocity_correlation": pearson(gt_peak, sim_peak),
        "peak_velocity_mae_mps": float(peak_mae),
        "peak_velocity_valid_frames": int(peak_valid),
        "wasserstein_1d_mean_mps": float(np.nanmean(wasserstein)) if np.any(valid_wasserstein) else float("nan"),
        "wasserstein_valid_frames": int(valid_wasserstein.sum()),
        "gt_power_peak": float(gt_peak_power),
        "sim_power_peak": float(sim_peak_power),
    }


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(arr[finite]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_activity(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in sample_rows:
        grouped.setdefault(str(row["activity"]), []).append(row)

    metric_names = [
        "ssim",
        "centroid_correlation",
        "centroid_mae_mps",
        "peak_velocity_correlation",
        "peak_velocity_mae_mps",
        "wasserstein_1d_mean_mps",
    ]
    out = []
    for activity in sorted(grouped):
        rows = grouped[activity]
        summary: dict[str, object] = {
            "activity": activity,
            "num_samples": len(rows),
            "num_sequences": len({row["sequence"] for row in rows}),
        }
        for name in metric_names:
            summary[f"mean_{name}"] = finite_mean([float(row[name]) for row in rows])
        out.append(summary)
    return out


def main() -> None:
    args = parse_args()
    if args.samples_per_sequence <= 0:
        raise ValueError("--samples-per-sequence must be positive")
    if args.window_seconds <= 0.0 or args.fps <= 0.0:
        raise ValueError("--window-seconds and --fps must be positive")
    window_frames = int(round(float(args.window_seconds) * float(args.fps)))
    if window_frames <= 0:
        raise ValueError("window length produced zero frames")

    train = load_json(Path(args.train))
    if not isinstance(train, dict):
        raise ValueError("--train must contain a JSON object")
    activities = load_activity_map(Path(args.filemeta))
    gt_dir = Path(args.gt_spectrum_dir)
    sim_dir = Path(args.sim_spectrum_dir)
    gt_meta = load_start_metadata(gt_dir)
    sim_meta = load_start_metadata(sim_dir)
    rng = np.random.default_rng(args.seed)

    sample_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for seq in sequence_ids(train, args.sequences):
        try:
            gt_spectrum, gt_frames = load_spectrum_with_frames(gt_dir, gt_meta, seq)
            sim_spectrum, sim_frames = load_spectrum_with_frames(sim_dir, sim_meta, seq)
            common_frames = np.intersect1d(gt_frames, sim_frames, assume_unique=False)
            starts = common_contiguous_starts(common_frames, window_frames)
            if starts.size == 0:
                skipped_rows.append({"sequence": seq, "reason": f"no contiguous {window_frames}-frame common window"})
                continue
            chosen = rng.choice(starts, size=args.samples_per_sequence, replace=starts.size < args.samples_per_sequence)
            for sample_index, start_frame in enumerate(chosen, start=1):
                gt_window = take_window(gt_spectrum, gt_frames, int(start_frame), window_frames)
                sim_window = take_window(sim_spectrum, sim_frames, int(start_frame), window_frames)
                metrics = evaluate_sample(
                    gt_window,
                    sim_window,
                    tuple(args.ssim_clip),
                    tuple(args.spectrum_clip),
                    int(args.ssim_window),
                )
                stop_exclusive = int(start_frame) + window_frames
                row: dict[str, object] = {
                    "sequence": int(seq),
                    "activity": activities.get(int(seq), "UNKNOWN"),
                    "sample_index": int(sample_index),
                    "start_frame": int(start_frame),
                    "stop_frame_exclusive": int(stop_exclusive),
                    "start_time_s": (float(start_frame) - 1.0) / float(args.fps),
                    "stop_time_exclusive_s": (float(stop_exclusive) - 1.0) / float(args.fps),
                    "window_seconds": float(args.window_seconds),
                    "window_frames": int(window_frames),
                    **metrics,
                }
                sample_rows.append(row)
        except Exception as exc:
            skipped_rows.append({"sequence": seq, "reason": str(exc)})

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "doppler_sample_metrics.csv", sample_rows)
    write_csv(out_dir / "doppler_activity_summary.csv", summarize_by_activity(sample_rows))
    write_csv(out_dir / "doppler_skipped_sequences.csv", skipped_rows)
    summary = {
        "gt_spectrum_dir": str(gt_dir),
        "sim_spectrum_dir": str(sim_dir),
        "train": str(Path(args.train)),
        "filemeta": str(Path(args.filemeta)),
        "samples_per_sequence": int(args.samples_per_sequence),
        "window_seconds": float(args.window_seconds),
        "window_frames": int(window_frames),
        "fps": float(args.fps),
        "seed": int(args.seed),
        "ssim_clip": [float(args.ssim_clip[0]), float(args.ssim_clip[1])],
        "spectrum_clip": [float(args.spectrum_clip[0]), float(args.spectrum_clip[1])],
        "num_samples": int(len(sample_rows)),
        "num_skipped_sequences": int(len(skipped_rows)),
    }
    (out_dir / "doppler_sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
