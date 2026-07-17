#!/usr/bin/env python3
"""Render radar xyz power and motion annotations for calibration inspection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rtpose-calib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]

BONES_BY_NAME = [
    ("Pelvis", "Thomx"),
    ("Thomx", "Head"),
    ("Thomx", "Right_Shoulder"),
    ("Right_Shoulder", "Right_Elbow"),
    ("Right_Elbow", "Right_Wrist"),
    ("Thomx", "Left_Shoulder"),
    ("Left_Shoulder", "Left_Elbow"),
    ("Left_Elbow", "Left_Wrist"),
    ("Pelvis", "Right_Hip"),
    ("Right_Hip", "Right_Knee"),
    ("Right_Knee", "Right_Ankle"),
    ("Pelvis", "Left_Hip"),
    ("Left_Hip", "Left_Knee"),
    ("Left_Knee", "Left_Ankle"),
]


def adaptive_palette() -> int:
    return getattr(getattr(Image, "Palette", None), "ADAPTIVE", Image.ADAPTIVE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3-view heatmap GIF overlaying sequence radar power with human pose."
    )
    parser.add_argument("--sequence", default="1", help="Sequence ID in Train.json and GT_sequences/")
    parser.add_argument("--train", default=str(REPO_ROOT / "datasets" / "Train.json"), help="Motion annotation JSON")
    parser.add_argument("--keypoints", default=str(REPO_ROOT / "datasets" / "Keypoints_meta.txt"), help="Joint name metadata")
    parser.add_argument(
        "--radar-dir",
        default="",
        help="Directory containing D-Z-Y-X complex radar tensors; defaults to datasets/GT_sequences/<sequence>/radar/npy_DZYX_complex",
    )
    parser.add_argument(
        "--compare-radar-dir",
        default="",
        help="Optional second D-Z-Y-X tensor directory rendered side-by-side against --radar-dir.",
    )
    parser.add_argument("--radar-label", default="GT", help="Display label for --radar-dir")
    parser.add_argument("--compare-label", default="Sim1", help="Display label for --compare-radar-dir")
    parser.add_argument(
        "--out",
        default="",
        help="Output GIF path",
    )
    parser.add_argument("--max-frames", type=int, default=90, help="Evenly sample to this many frames; 0 uses all")
    parser.add_argument("--duration-ms", type=int, default=100, help="GIF frame duration")
    parser.add_argument("--dpi", type=int, default=105, help="Matplotlib render DPI")
    parser.add_argument("--width", type=float, default=14.5, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=4.8, help="Figure height in inches")
    parser.add_argument("--roi-x-margin", type=float, default=0.5, help="Meters added around pose x range")
    parser.add_argument("--roi-y-margin", type=float, default=0.5, help="Meters added around pose y range")
    parser.add_argument("--roi-z-margin", type=float, default=0.5, help="Meters added around pose z range")
    parser.add_argument("--heatmap-margin", type=float, default=0.5, help="Meters added around full-sequence pose range")
    parser.add_argument(
        "--point-roi-mode",
        choices=("per-frame", "sequence"),
        default="per-frame",
        help="Deprecated; retained for CLI compatibility.",
    )
    parser.add_argument("--db-floor", type=float, default=-45.0, help="Minimum displayed relative echo level in dB")
    parser.add_argument("--db-ceil", type=float, default=0.0, help="Maximum displayed relative echo level in dB")
    parser.add_argument(
        "--projection-reduction",
        choices=("max", "mean"),
        default="max",
        help="Reduction used for XY/XZ/YZ heatmap projections.",
    )
    parser.add_argument(
        "--flip-radar-y",
        action="store_true",
        help="Interpret older tensors whose saved Y index was reversed",
    )
    parser.add_argument(
        "--flip-annotation-y",
        action="store_true",
        help="Negate the Y coordinate of motion annotations before ROI and drawing",
    )
    parser.add_argument(
        "--point-percentile",
        type=float,
        default=99.5,
        help="Deprecated; retained for CLI compatibility.",
    )
    parser.add_argument("--max-points", type=int, default=700, help="Deprecated; retained for CLI compatibility.")
    parser.add_argument("--annotation-index", type=int, default=0, help="Person index in each annotation frame")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1) // 2)),
        help="Parallel frame render workers; 1 disables multiprocessing.",
    )
    return parser.parse_args()


def load_keypoints(path: Path) -> dict[int, str]:
    joints: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        idx_text, name = line.split(",", 1)
        joints[int(idx_text)] = name.strip()
    return joints


def build_bones(joints: dict[int, str]) -> list[tuple[int, int]]:
    name_to_idx = {name: idx for idx, name in joints.items()}
    return [(name_to_idx[a], name_to_idx[b]) for a, b in BONES_BY_NAME if a in name_to_idx and b in name_to_idx]


def sorted_numeric_keys(keys: Iterable[str]) -> list[str]:
    return sorted(keys, key=lambda item: int(item) if str(item).isdigit() else str(item))


def load_sequence_annotations(
    train_path: Path, sequence: str, annotation_index: int, radar_dir: Path
) -> list[tuple[str, np.ndarray]]:
    with train_path.open("r", encoding="utf-8") as f:
        train = json.load(f)
    if sequence not in train:
        raise KeyError(f"Sequence {sequence!r} was not found in {train_path}")

    frames: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for frame_key in sorted_numeric_keys(train[sequence].keys()):
        annotations = train[sequence][frame_key]
        if not annotations or annotation_index >= len(annotations):
            continue
        ann = annotations[annotation_index]
        radar_id = ann.get("Radar_frameID")
        pose = ann.get("pose")
        if not radar_id or not pose or radar_id in seen:
            continue
        if not (radar_dir / f"{radar_id}.npy").exists():
            continue
        frames.append((radar_id, np.asarray(pose, dtype=np.float64)))
        seen.add(radar_id)
    if not frames:
        raise RuntimeError(f"No matched radar/pose frames found for sequence {sequence}")
    return frames


def sample_frames(frames: list[tuple[str, np.ndarray]], max_frames: int) -> list[tuple[str, np.ndarray]]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    indexes = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frames[i] for i in indexes]


def flip_annotation_y(frames: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    flipped: list[tuple[str, np.ndarray]] = []
    for radar_id, pose in frames:
        pose_flipped = pose.copy()
        pose_flipped[:, 1] *= -1.0
        flipped.append((radar_id, pose_flipped))
    return flipped


def xyz_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(0.0, 11.6, 11.6 / 256)
    y = np.arange(-10.05, 10.05, 20.1 / 128)
    z = np.arange(-5.8, 5.8, 11.6 / 32)
    return x, y, z


def radar_y_axis(y_axis: np.ndarray, flip_radar_y: bool) -> np.ndarray:
    return y_axis[::-1] if flip_radar_y else y_axis


def axis_slice(axis: np.ndarray, low: float, high: float) -> slice:
    hits = np.flatnonzero((axis >= low) & (axis <= high))
    if hits.size == 0:
        raise ValueError(f"ROI [{low:.3f}, {high:.3f}] does not overlap axis [{axis.min():.3f}, {axis.max():.3f}]")
    return slice(int(hits.min()), int(hits.max()) + 1)


def compute_pose_roi(
    pose: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    margins: tuple[float, float, float],
) -> tuple[slice, slice, slice, tuple[float, float, float, float, float, float]]:
    mins = np.nanmin(pose, axis=0) - np.asarray(margins)
    maxs = np.nanmax(pose, axis=0) + np.asarray(margins)
    mins[0] = max(mins[0], x_axis.min())
    maxs[0] = min(maxs[0], x_axis.max())
    mins[1] = max(mins[1], y_axis.min())
    maxs[1] = min(maxs[1], y_axis.max())
    mins[2] = max(mins[2], z_axis.min())
    maxs[2] = min(maxs[2], z_axis.max())
    x_slice = axis_slice(x_axis, mins[0], maxs[0])
    y_slice = axis_slice(y_axis, mins[1], maxs[1])
    z_slice = axis_slice(z_axis, mins[2], maxs[2])
    bounds = (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))
    return z_slice, y_slice, x_slice, bounds


def compute_sequence_roi(
    frames: list[tuple[str, np.ndarray]],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    margin: float,
) -> tuple[slice, slice, slice, tuple[float, float, float, float, float, float]]:
    poses = np.concatenate([pose for _, pose in frames], axis=0)
    return compute_pose_roi(poses, x_axis, y_axis, z_axis, margins=(margin, margin, margin))


def radar_power_region(
    path: Path,
    spatial_slices: tuple[slice, slice, slice],
) -> np.ndarray:
    tensor = np.load(path, mmap_mode="r")
    if tensor.shape != (64, 32, 128, 256):
        raise ValueError(f"{path} has unexpected shape {tensor.shape}")
    z_slice, y_slice, x_slice = spatial_slices
    power_shape = (
        z_slice.stop - z_slice.start,
        y_slice.stop - y_slice.start,
        x_slice.stop - x_slice.start,
    )
    power = np.zeros(power_shape, dtype=np.float64)
    for doppler_idx in range(tensor.shape[0]):
        plane = tensor[doppler_idx, z_slice, y_slice, x_slice]
        power += plane.real.astype(np.float64) ** 2 + plane.imag.astype(np.float64) ** 2
    return power


def slice_len(item: slice) -> int:
    return item.stop - item.start


def union_slices(*regions: tuple[slice, slice, slice]) -> tuple[slice, slice, slice]:
    return tuple(
        slice(min(region[axis].start for region in regions), max(region[axis].stop for region in regions))
        for axis in range(3)
    )


def relative_slices(
    outer: tuple[slice, slice, slice],
    inner: tuple[slice, slice, slice],
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(inner[axis].start - outer[axis].start, inner[axis].stop - outer[axis].start)
        for axis in range(3)
    )


def relative_echo_db(
    power: np.ndarray,
    floor_db: float,
    ceil_db: float,
    max_echo: float | None = None,
) -> np.ndarray:
    if max_echo is None:
        max_echo = float(np.nanmax(power))
    if not np.isfinite(max_echo) or max_echo <= 0.0:
        return np.full_like(power, floor_db, dtype=np.float32)
    ratio = np.clip(power / max_echo, 1e-12, None)
    db = 10.0 * np.log10(ratio)
    return np.clip(db, floor_db, ceil_db).astype(np.float32)


def projection_images(values: np.ndarray, reduction: str) -> dict[str, np.ndarray]:
    reducer = np.mean if reduction == "mean" else np.max
    return {
        "xy": reducer(values, axis=0),
        "xz": reducer(values, axis=1),
        "yz": reducer(values, axis=2),
    }


def draw_skeleton_3d(ax, pose: np.ndarray, bones: list[tuple[int, int]], *, color: str) -> None:
    finite = np.isfinite(pose).all(axis=1)
    ax.scatter(pose[finite, 0], pose[finite, 1], pose[finite, 2], c=color, s=24, depthshade=False)
    for a, b in bones:
        if a < len(pose) and b < len(pose) and finite[a] and finite[b]:
            ax.plot(*zip(pose[a], pose[b]), color=color, linewidth=2.0)


def draw_skeleton_2d(
    ax, pose: np.ndarray, bones: list[tuple[int, int]], dims: tuple[int, int], *, color: str
) -> None:
    finite = np.isfinite(pose).all(axis=1)
    ax.scatter(pose[finite, dims[0]], pose[finite, dims[1]], c=color, s=18, edgecolors="black", linewidths=0.35)
    for a, b in bones:
        if a < len(pose) and b < len(pose) and finite[a] and finite[b]:
            ax.plot(
                [pose[a, dims[0]], pose[b, dims[0]]],
                [pose[a, dims[1]], pose[b, dims[1]]],
                color=color,
                linewidth=1.8,
            )


def select_points(
    norm_power: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    percentile: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    threshold = float(np.percentile(norm_power, percentile))
    mask = norm_power >= threshold
    values = norm_power[mask]
    coords = np.argwhere(mask)
    if values.size > max_points:
        keep = np.argpartition(values, -max_points)[-max_points:]
        coords = coords[keep]
        values = values[keep]
    if values.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty, empty, empty
    z_idx, y_idx, x_idx = coords[:, 0], coords[:, 1], coords[:, 2]
    return x_axis[x_idx], y_axis[y_idx], z_axis[z_idx], values


def render_source_panel(
    panel_axes: tuple,
    source_label: str,
    radar_id: str,
    pose: np.ndarray,
    radar_path: Path,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    heatmap_slices: tuple[slice, slice, slice],
    bones: list[tuple[int, int]],
    args: argparse.Namespace,
) -> None:
    x_full, y_full, z_full = axes
    radar_y = radar_y_axis(y_full, args.flip_radar_y)
    ax_xy, ax_xz, ax_yz = panel_axes
    heat_z_slice, heat_y_slice, heat_x_slice = heatmap_slices

    heat_power = radar_power_region(radar_path, heatmap_slices)
    max_echo = float(np.nanmax(heat_power))
    heat_db = relative_echo_db(heat_power, args.db_floor, args.db_ceil, max_echo=max_echo)

    x_heat = x_full[heat_x_slice]
    y_heat = radar_y[heat_y_slice]
    z_heat = z_full[heat_z_slice]

    projections = projection_images(heat_db, args.projection_reduction)
    xy = projections["xy"]
    xz = projections["xz"]
    yz = projections["yz"]

    heatmap_cmap = "jet"
    im_xy = ax_xy.imshow(
        xy,
        extent=(float(x_heat[0]), float(x_heat[-1]), float(y_heat[0]), float(y_heat[-1])),
        origin="lower",
        aspect="auto",
        cmap=heatmap_cmap,
        vmin=args.db_floor,
        vmax=args.db_ceil,
    )
    draw_skeleton_2d(ax_xy, pose, bones, (0, 1), color="#34d399")
    ax_xy.set_title(f"{source_label} XY | radar {radar_id}", fontsize=11, pad=8)
    ax_xy.set_xlabel("x (m)", labelpad=3)
    ax_xy.set_ylabel("y (m)", labelpad=3)
    ax_xy.set_xlim(float(x_heat[0]), float(x_heat[-1]))
    ax_xy.set_ylim(float(y_heat[0]), float(y_heat[-1]))

    im_xz = ax_xz.imshow(
        xz,
        extent=(float(x_heat[0]), float(x_heat[-1]), float(z_heat[0]), float(z_heat[-1])),
        origin="lower",
        aspect="auto",
        cmap=heatmap_cmap,
        vmin=args.db_floor,
        vmax=args.db_ceil,
    )
    draw_skeleton_2d(ax_xz, pose, bones, (0, 2), color="#34d399")
    ax_xz.set_title(f"{source_label} XZ", fontsize=11, pad=8)
    ax_xz.set_xlabel("x (m)", labelpad=3)
    ax_xz.set_ylabel("z (m)", labelpad=3)
    ax_xz.set_xlim(float(x_heat[0]), float(x_heat[-1]))
    ax_xz.set_ylim(float(z_heat[0]), float(z_heat[-1]))

    im_yz = ax_yz.imshow(
        yz,
        extent=(float(y_heat[0]), float(y_heat[-1]), float(z_heat[0]), float(z_heat[-1])),
        origin="lower",
        aspect="auto",
        cmap=heatmap_cmap,
        vmin=args.db_floor,
        vmax=args.db_ceil,
    )
    draw_skeleton_2d(ax_yz, pose, bones, (1, 2), color="#34d399")
    ax_yz.set_title(f"{source_label} YZ", fontsize=11, pad=8)
    ax_yz.set_xlabel("y (m)", labelpad=3)
    ax_yz.set_ylabel("z (m)", labelpad=3)
    ax_yz.set_xlim(float(y_heat[0]), float(y_heat[-1]))
    ax_yz.set_ylim(float(z_heat[0]), float(z_heat[-1]))

    for ax in (ax_xy, ax_xz, ax_yz):
        ax.tick_params(labelsize=8, pad=2)
        ax.grid(color="white", linewidth=0.35, alpha=0.22)

    return None


def render_frame(
    radar_id: str,
    pose: np.ndarray,
    radar_sources: list[tuple[str, Path]],
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    heatmap_slices: tuple[slice, slice, slice],
    bones: list[tuple[int, int]],
    args: argparse.Namespace,
) -> Image.Image:
    if len(radar_sources) == 1:
        fig, row_axes = plt.subplots(1, 3, figsize=(args.width, args.height), dpi=args.dpi, constrained_layout=True)
        panel_axes = [tuple(row_axes)]
    else:
        fig, row_axes = plt.subplots(
            len(radar_sources),
            3,
            figsize=(max(args.width, 14.5), max(args.height, 4.4 * len(radar_sources))),
            dpi=args.dpi,
            constrained_layout=True,
        )
        panel_axes = [tuple(row_axes[row]) for row in range(len(radar_sources))]
    fig.patch.set_facecolor("#f7f8fa")

    heat_axes = []
    for source_axes, (source_label, radar_path) in zip(panel_axes, radar_sources):
        render_source_panel(
            source_axes,
            source_label,
            radar_id,
            pose,
            radar_path,
            axes,
            heatmap_slices,
            bones,
            args,
        )
        heat_axes.extend(source_axes)

    scalar = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=args.db_floor, vmax=args.db_ceil))
    scalar.set_array([])
    if len(radar_sources) > 1:
        fig.colorbar(scalar, ax=heat_axes, fraction=0.018, pad=0.015, label="relative echo (dB)")
    else:
        fig.colorbar(scalar, ax=heat_axes, fraction=0.026, pad=0.02, label="relative echo (dB)")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("P", palette=adaptive_palette(), colors=192)


def add_progress_text(image: Image.Image, index: int, total: int) -> Image.Image:
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    text = f"{index + 1}/{total}"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 6
    x = image.width - (bbox[2] - bbox[0]) - pad * 2 - 12
    y = 10
    draw.rounded_rectangle((x, y, image.width - 10, y + bbox[3] - bbox[1] + pad * 2), radius=5, fill=(15, 23, 42, 180))
    draw.text((x + pad, y + pad), text, font=font, fill=(255, 255, 255, 255))
    return image.convert("P", palette=adaptive_palette(), colors=192)


def render_frame_task(task: tuple) -> tuple[int, Image.Image]:
    (
        idx,
        total,
        radar_id,
        pose,
        radar_sources,
        axes,
        heatmap_slices,
        bones,
        args,
    ) = task
    image = render_frame(
        radar_id,
        pose,
        radar_sources,
        axes,
        heatmap_slices,
        bones,
        args,
    )
    return idx, add_progress_text(image, idx, total)


def main() -> None:
    args = parse_args()
    radar_dir = (
        Path(args.radar_dir)
        if args.radar_dir
        else REPO_ROOT / "datasets" / "GT_sequences" / args.sequence / "radar" / "npy_DZYX_complex"
    )
    radar_sources = [(args.radar_label, radar_dir)]
    if args.compare_radar_dir:
        radar_sources.append((args.compare_label, Path(args.compare_radar_dir)))

    frames_all = load_sequence_annotations(Path(args.train), args.sequence, args.annotation_index, radar_dir)
    if len(radar_sources) > 1:
        frames_all = [
            (radar_id, pose)
            for radar_id, pose in frames_all
            if all((source_dir / f"{radar_id}.npy").exists() for _label, source_dir in radar_sources)
        ]
        if not frames_all:
            raise RuntimeError(f"No common radar/pose frames found for sequence {args.sequence}")
    if args.flip_annotation_y:
        frames_all = flip_annotation_y(frames_all)
    frames = sample_frames(frames_all, args.max_frames)
    joints = load_keypoints(Path(args.keypoints))
    bones = build_bones(joints)
    axes = xyz_axes()
    x_axis, y_axis, z_axis = axes
    y_for_roi = radar_y_axis(y_axis, args.flip_radar_y)
    heatmap_roi = compute_sequence_roi(
        frames_all,
        x_axis,
        y_for_roi,
        z_axis,
        margin=args.heatmap_margin,
    )
    heat_z_slice, heat_y_slice, heat_x_slice, heatmap_bounds = heatmap_roi
    heatmap_slices = (heat_z_slice, heat_y_slice, heat_x_slice)
    if len(radar_sources) > 1:
        source_names = "_vs_".join(label for label, _source_dir in radar_sources)
        default_name = f"sequence{args.sequence}_{source_names}_relative_echo_db.gif"
        default_root = REPO_ROOT / "results" / "Qualitive_analysis" / "radar_pose_compare_gifs"
    else:
        default_name = f"sequence{args.sequence}_radar_pose_calibration.gif"
        default_root = REPO_ROOT / "results" / "GT" / "radar_pose_calib_gifs"
    out_path = Path(args.out) if args.out else default_root / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            idx,
            len(frames),
            radar_id,
            pose,
            [(label, source_dir / f"{radar_id}.npy") for label, source_dir in radar_sources],
            axes,
            heatmap_slices,
            bones,
            args,
        )
        for idx, (radar_id, pose) in enumerate(frames)
    ]
    images: list[Image.Image | None] = [None] * len(tasks)
    if args.workers <= 1 or len(tasks) == 1:
        for task in tasks:
            idx = task[0]
            radar_id = task[2]
            print(f"[{idx + 1:03d}/{len(tasks):03d}] rendering radar {radar_id}", flush=True)
            out_idx, image = render_frame_task(task)
            images[out_idx] = image
    else:
        print(f"rendering {len(tasks)} frames with {args.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for out_idx, image in executor.map(render_frame_task, tasks):
                images[out_idx] = image
                print(f"[{out_idx + 1:03d}/{len(tasks):03d}] rendered", flush=True)
    final_images = [image for image in images if image is not None]
    if not final_images:
        raise RuntimeError("No frames were rendered")

    final_images[0].save(
        out_path,
        save_all=True,
        append_images=final_images[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"wrote {out_path}")
    print(f"matched_frames={len(frames_all)} rendered_frames={len(frames)}")
    print(
        "heatmap_roi="
        f"x[{heatmap_bounds[0]:.3f},{heatmap_bounds[1]:.3f}] "
        f"y[{heatmap_bounds[2]:.3f},{heatmap_bounds[3]:.3f}] "
        f"z[{heatmap_bounds[4]:.3f},{heatmap_bounds[5]:.3f}]"
    )


if __name__ == "__main__":
    main()
