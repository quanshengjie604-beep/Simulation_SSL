#!/usr/bin/env python3
"""Generate Doppler-time from mmRadPose target lists inside a GT skeleton bbox."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = REPO_ROOT / "datasets/mmRadPose/mmRadPose_pointclouds/p6/angle0/7/0"
DEFAULT_OUT_DIR = REPO_ROOT / "results/mmradpose_raw_doppler_time"

FRAME_RATE_HZ = 15.0
DOPPLER_BINS = 128
DOPPLER_RESOLUTION_MPS = 0.07838449


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default="mmradpose_p6_an0_ac7_r0_F0_149_gt_point_bbox")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--bbox-margin-m", type=float, default=0.50)
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    ax.set_title("mmRadPose target-list GT-person bbox Doppler-time")
    fig.colorbar(im, ax=ax, label="Point power (dB, normalized)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    targetlist_path = args.sequence_dir / "targetlist_64.npy"
    skeleton_path = args.sequence_dir / "skeleton.npy"
    targets = np.load(targetlist_path, allow_pickle=False).astype(np.float32, copy=False)
    skeleton = np.load(skeleton_path, allow_pickle=False).reshape(-1, 26, 3).astype(np.float32, copy=False)

    num_frames = int(np.ceil(args.duration_s * FRAME_RATE_HZ))
    num_frames = min(num_frames, targets.shape[0] - args.start_frame, skeleton.shape[0] - args.start_frame)
    times = (args.start_frame + np.arange(num_frames, dtype=np.float64)) / FRAME_RATE_HZ
    velocities = (np.arange(DOPPLER_BINS, dtype=np.float64) - DOPPLER_BINS // 2) * DOPPLER_RESOLUTION_MPS

    spectrum = np.zeros((DOPPLER_BINS, num_frames), dtype=np.float32)
    num_points_in_bbox = np.zeros(num_frames, dtype=np.int16)
    bbox_min = np.empty((num_frames, 3), dtype=np.float32)
    bbox_max = np.empty((num_frames, 3), dtype=np.float32)

    range_offset = np.zeros(num_frames, dtype=np.float32)
    for idx in range(num_frames):
        skel = skeleton[args.start_frame + idx]
        low = np.nanmin(skel, axis=0) - args.bbox_margin_m
        high = np.nanmax(skel, axis=0) + args.bbox_margin_m

        pts = targets[args.start_frame + idx]
        valid = np.any(np.abs(pts[:, :3]) > 0.0, axis=1) & (pts[:, 6] > 0.0)
        if np.any(valid):
            valid_points = pts[valid]
            keep = max(5, int(np.ceil(0.25 * valid_points.shape[0])))
            top_ids = np.argsort(valid_points[:, 6])[-keep:]
            target_range_center = float(np.median(valid_points[top_ids, 1]))
            skeleton_range_center = 0.5 * (float(np.nanmin(skel[:, 1])) + float(np.nanmax(skel[:, 1])))
            range_offset[idx] = target_range_center - skeleton_range_center
            low[1] += range_offset[idx]
            high[1] += range_offset[idx]
        low[1] = max(0.0, low[1])
        bbox_min[idx] = low
        bbox_max[idx] = high

        inside = (
            valid
            & (pts[:, 0] >= low[0])
            & (pts[:, 0] <= high[0])
            & (pts[:, 1] >= low[1])
            & (pts[:, 1] <= high[1])
            & (pts[:, 2] >= low[2])
            & (pts[:, 2] <= high[2])
        )
        selected = pts[inside]
        num_points_in_bbox[idx] = selected.shape[0]
        if selected.size:
            bin_ids = np.rint(selected[:, 3] / DOPPLER_RESOLUTION_MPS).astype(np.int64) + DOPPLER_BINS // 2
            bin_ids = np.clip(bin_ids, 0, DOPPLER_BINS - 1)
            weights = np.maximum(selected[:, 6], 0.0).astype(np.float32)
            np.add.at(spectrum[:, idx], bin_ids, weights)

        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == num_frames:
            print(
                f"[frame {idx + 1:04d}/{num_frames:04d}] t={times[idx]:.3f}s "
                f"points={num_points_in_bbox[idx]} bbox_y={low[1]:.2f}..{high[1]:.2f}m",
                flush=True,
            )

    # Keep empty columns finite and visible as floor after dB conversion.
    spectrum += np.finfo(np.float32).tiny
    spectrum_db = power_to_db(spectrum, args.db_floor)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / f"{args.label}.npz"
    png_path = args.out_dir / f"{args.label}.png"
    json_path = args.out_dir / f"{args.label}.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"{result_path} exists; pass --overwrite")
    np.savez_compressed(
        result_path,
        spectrum=spectrum,
        spectrum_db=spectrum_db,
        time_s=times,
        velocity_mps=velocities,
        num_points_in_bbox=num_points_in_bbox,
        bbox_min_m=bbox_min,
        bbox_max_m=bbox_max,
        range_offset_m=range_offset,
        targetlist_columns=np.asarray(
            ["x_lateral_m", "y_range_m", "z_height_m", "radial_velocity_mps", "field4", "field5", "point_weight"],
            dtype="U32",
        ),
    )
    save_plot(png_path, spectrum_db, times, velocities, args.db_floor)
    summary = {
        "processing": "mmRadPose targetlist_64 points -> GT skeleton 3D bbox selection -> Doppler histogram over point radial velocity",
        "source_targetlist": str(targetlist_path.resolve()),
        "source_skeleton": str(skeleton_path.resolve()),
        "sequence": "p6_angle0_ac7_r0",
        "frames": {
            "start_frame": args.start_frame,
            "num_frames": int(num_frames),
            "frame_rate_hz": FRAME_RATE_HZ,
            "duration_s": float(num_frames / FRAME_RATE_HZ),
        },
        "targetlist": {
            "shape": list(map(int, targets.shape)),
            "columns_inferred": [
                "x_lateral_m",
                "y_range_m",
                "z_height_m",
                "radial_velocity_mps",
                "field4",
                "field5",
                "point_weight",
            ],
            "doppler_resolution_mps": DOPPLER_RESOLUTION_MPS,
        },
        "roi": {
            "bbox_margin_m": args.bbox_margin_m,
            "range_alignment": "per-frame translation from skeleton range center to median range of top-power targetlist points",
            "range_offset_m_mean": float(np.mean(range_offset)),
            "mean_points_per_frame": float(np.mean(num_points_in_bbox)),
            "min_points_per_frame": int(np.min(num_points_in_bbox)),
            "max_points_per_frame": int(np.max(num_points_in_bbox)),
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
