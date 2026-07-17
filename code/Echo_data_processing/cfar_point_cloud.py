#!/usr/bin/env python3
"""Generate xyz radar point clouds with 3D CA-CFAR.

The installed OpenRadar package provides 1D CFAR building blocks. This script
keeps the same cell-averaging idea, but applies it over RT-Pose cartesian
``Z,Y,X`` radar power volumes so detections are target voxels in xyz space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from raw_echo_to_xyz import RadarConfig, xyz_axes


PointCloud = np.ndarray
DEFAULT_ROI = (
    0.7703125,
    8.0203125,
    -5.0250000000000234,
    5.024999999999931,
    -1.0875000000000021,
    4.7125,
)


def parse_xyz_cells(text: str) -> tuple[int, int, int]:
    cleaned = text.lower().replace("x", ",").replace(" ", ",")
    parts = [item for item in cleaned.split(",") if item]
    values = tuple(int(item) for item in parts)
    if len(values) == 1:
        value = values[0]
        if value < 0:
            raise argparse.ArgumentTypeError("cell counts must be non-negative")
        return value, value, value
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected one integer or x,y,z")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("cell counts must be non-negative")
    return values


def parse_roi(text: str) -> tuple[float, float, float, float, float, float]:
    parts = [item for item in text.replace(" ", ",").split(",") if item]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected x_min,x_max,y_min,y_max,z_min,z_max")
    values = tuple(float(item) for item in parts)
    x0, x1, y0, y1, z0, z1 = values
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise argparse.ArgumentTypeError("ROI min values must be smaller than max values")
    return values


def load_pose_bounds(
    train_path: Path,
    sequence: str,
    radar_id: str,
    annotation_index: int,
    margin: float,
) -> tuple[float, float, float, float, float, float]:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    if sequence not in train:
        raise KeyError(f"sequence {sequence!r} not found in {train_path}")
    target = radar_id.zfill(6)
    for annotations in train[sequence].values():
        for ann_index, ann in enumerate(annotations):
            if ann_index != annotation_index:
                continue
            if str(ann.get("Radar_frameID", "")).zfill(6) != target:
                continue
            pose = np.asarray(ann["pose"], dtype=np.float32)
            finite = np.isfinite(pose).all(axis=1)
            if not np.any(finite):
                raise ValueError(f"pose for sequence {sequence} radar {radar_id} has no finite joints")
            mins = pose[finite].min(axis=0) - margin
            maxs = pose[finite].max(axis=0) + margin
            return float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2])
    raise KeyError(f"radar frame {radar_id!r} not found for sequence {sequence!r} annotation {annotation_index}")


def xyz_to_zyx(cells_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
    x, y, z = cells_xyz
    return z, y, x


def axis_roi_slice(axis: np.ndarray, low: float, high: float) -> slice:
    start = int(np.argmin(np.abs(axis - low)))
    stop = int(np.argmin(np.abs(axis - high)))
    if high <= axis[-1]:
        stop -= 1
    if stop < start:
        raise ValueError(f"ROI [{low:.3f}, {high:.3f}] does not overlap axis [{axis.min():.3f}, {axis.max():.3f}]")
    return slice(start, stop + 1)


def roi_to_slices(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    roi: tuple[float, float, float, float, float, float] | None,
) -> tuple[slice, slice, slice]:
    if roi is None:
        return slice(None), slice(None), slice(None)
    x_axis, y_axis, z_axis = axes
    x0, x1, y0, y1, z0, z1 = roi
    return (
        axis_roi_slice(z_axis, z0, z1),
        axis_roi_slice(y_axis, y0, y1),
        axis_roi_slice(x_axis, x0, x1),
    )


def slice_lengths(spatial_slices: tuple[slice, slice, slice], full_shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(len(range(*item.indices(full_shape_zyx[axis]))) for axis, item in enumerate(spatial_slices))


def axes_for_power_shape(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    power_shape_zyx: tuple[int, int, int],
    spatial_slices: tuple[slice, slice, slice],
    user_roi: tuple[float, float, float, float, float, float] | None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[slice, slice, slice]]:
    full_shape_zyx = (len(axes[2]), len(axes[1]), len(axes[0]))
    if power_shape_zyx == full_shape_zyx:
        return axes, spatial_slices

    if user_roi is not None:
        expected = slice_lengths(spatial_slices, full_shape_zyx)
        if power_shape_zyx == expected:
            return axes, spatial_slices
        raise ValueError(
            f"input spatial shape {power_shape_zyx} does not match --roi shape {expected}; "
            "omit --roi for pre-cropped tensors"
        )

    default_roi_slices = roi_to_slices(axes, DEFAULT_ROI)
    if power_shape_zyx == slice_lengths(default_roi_slices, full_shape_zyx):
        z_slice, y_slice, x_slice = default_roi_slices
        x_axis, y_axis, z_axis = axes
        return (x_axis[x_slice], y_axis[y_slice], z_axis[z_slice]), (slice(None), slice(None), slice(None))

    raise ValueError(
        f"cannot infer xyz axes for input spatial shape {power_shape_zyx}; "
        "expected full (32, 128, 256) or default ROI (16, 64, 160)"
    )


def power_zyx(
    path: Path,
    spatial_slices: tuple[slice, slice, slice],
    doppler_reducer: str,
    batch_size: int = 8,
) -> np.ndarray:
    tensor = np.load(path, mmap_mode="r")
    if tensor.ndim != 4 or tensor.shape[0] != 64:
        raise ValueError(f"{path} has unexpected shape {tensor.shape}; expected D-Z-Y-X with 64 Doppler bins")

    z_slice, y_slice, x_slice = spatial_slices
    region_shape = np.asarray(tensor[0, z_slice, y_slice, x_slice]).shape
    if doppler_reducer == "sum":
        power = np.zeros(region_shape, dtype=np.float32)
    elif doppler_reducer == "max":
        power = np.zeros(region_shape, dtype=np.float32)
    else:
        raise ValueError(f"unsupported doppler reducer {doppler_reducer!r}")

    for start in range(0, tensor.shape[0], batch_size):
        region = np.asarray(tensor[start : start + batch_size, z_slice, y_slice, x_slice])
        if np.iscomplexobj(region):
            batch_power = region.real * region.real + region.imag * region.imag
        else:
            batch_power = region.astype(np.float32, copy=False)
        if doppler_reducer == "sum":
            power += np.sum(batch_power, axis=0, dtype=np.float32)
        else:
            power = np.maximum(power, np.max(batch_power, axis=0).astype(np.float32, copy=False))
    return power


def training_kernel_3d(
    guard_cells_zyx: tuple[int, int, int],
    training_cells_zyx: tuple[int, int, int],
) -> np.ndarray:
    radii = tuple(guard + training for guard, training in zip(guard_cells_zyx, training_cells_zyx))
    if any(radius == 0 for radius in radii):
        raise ValueError("each dimension needs at least one guard or training cell")
    shape = tuple(2 * radius + 1 for radius in radii)
    kernel = np.ones(shape, dtype=np.float32)
    guard_slices = tuple(slice(radius - guard, radius + guard + 1) for radius, guard in zip(radii, guard_cells_zyx))
    kernel[guard_slices] = 0.0
    if float(kernel.sum()) <= 0.0:
        raise ValueError("training cell shell is empty; increase --training-cells")
    return kernel


def box_sum(array: np.ndarray, size: tuple[int, int, int], boundary_mode: str) -> np.ndarray:
    summed = uniform_filter(array, size=size, mode=boundary_mode, cval=0.0)
    return summed * float(np.prod(size))


def ca_cfar_3d(
    power: np.ndarray,
    guard_cells_zyx: tuple[int, int, int],
    training_cells_zyx: tuple[int, int, int],
    pfa: float,
    boundary_mode: str = "constant",
    min_training_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < pfa < 1.0):
        raise ValueError("--pfa must be between 0 and 1")
    if not (0.0 <= min_training_fraction <= 1.0):
        raise ValueError("--min-training-fraction must be in [0, 1]")

    outer_radii = tuple(guard + training for guard, training in zip(guard_cells_zyx, training_cells_zyx))
    if any(radius == 0 for radius in outer_radii):
        raise ValueError("each dimension needs at least one guard or training cell")
    outer_size = tuple(2 * radius + 1 for radius in outer_radii)
    guard_size = tuple(2 * guard + 1 for guard in guard_cells_zyx)
    max_training_cells = float(np.prod(outer_size) - np.prod(guard_size))
    if max_training_cells <= 0.0:
        raise ValueError("training cell shell is empty; increase --training-cells")
    power64 = np.asarray(power, dtype=np.float64)
    valid_cells = np.ones(power.shape, dtype=np.float64)
    noise_sum = box_sum(power64, outer_size, boundary_mode) - box_sum(power64, guard_size, boundary_mode)
    training_count = box_sum(valid_cells, outer_size, boundary_mode) - box_sum(valid_cells, guard_size, boundary_mode)
    min_training_cells = max(1.0, max_training_cells * min_training_fraction)

    valid = training_count >= min_training_cells
    noise = np.full(power.shape, np.nan, dtype=np.float32)
    threshold = np.full(power.shape, np.inf, dtype=np.float32)
    detections = np.zeros(power.shape, dtype=bool)
    if np.any(valid):
        local_count = training_count[valid]
        local_noise = np.maximum(noise_sum[valid] / local_count, 0.0)
        alpha = local_count * (np.power(pfa, -1.0 / local_count) - 1.0)
        local_threshold = np.maximum(local_noise * alpha, 0.0)
        noise[valid] = local_noise.astype(np.float32, copy=False)
        threshold[valid] = local_threshold.astype(np.float32, copy=False)
        detections[valid] = power64[valid] > local_threshold
    return detections, noise, threshold


def apply_local_max_filter(
    detections: np.ndarray,
    power: np.ndarray,
    radius_zyx: tuple[int, int, int],
    boundary_mode: str,
) -> np.ndarray:
    size = tuple(2 * max(1, radius) + 1 for radius in radius_zyx)
    local_max = maximum_filter(power, size=size, mode=boundary_mode, cval=0.0)
    return detections & (power >= local_max)


def xyz_bounds_mask(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    x_axis, y_axis, z_axis = axes
    x0, x1, y0, y1, z0, z1 = bounds
    return (
        (z_axis[:, np.newaxis, np.newaxis] >= z0)
        & (z_axis[:, np.newaxis, np.newaxis] <= z1)
        & (y_axis[np.newaxis, :, np.newaxis] >= y0)
        & (y_axis[np.newaxis, :, np.newaxis] <= y1)
        & (x_axis[np.newaxis, np.newaxis, :] >= x0)
        & (x_axis[np.newaxis, np.newaxis, :] <= x1)
    )


def detections_to_point_cloud(
    detections: np.ndarray,
    power: np.ndarray,
    noise: np.ndarray,
    threshold: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    spatial_slices: tuple[slice, slice, slice],
    max_points: int,
) -> PointCloud:
    coords = np.argwhere(detections)
    if coords.size == 0:
        return np.empty((0, 7), dtype=np.float32)

    values = power[detections]
    if max_points > 0 and values.size > max_points:
        keep = np.argpartition(values, -max_points)[-max_points:]
        coords = coords[keep]
        values = values[keep]

    z_slice, y_slice, x_slice = spatial_slices
    z_start = 0 if z_slice.start is None else z_slice.start
    y_start = 0 if y_slice.start is None else y_slice.start
    x_start = 0 if x_slice.start is None else x_slice.start
    z_idx = coords[:, 0] + z_start
    y_idx = coords[:, 1] + y_start
    x_idx = coords[:, 2] + x_start

    x_axis, y_axis, z_axis = axes
    point_noise = noise[coords[:, 0], coords[:, 1], coords[:, 2]]
    point_threshold = threshold[coords[:, 0], coords[:, 1], coords[:, 2]]
    snr_db = 10.0 * np.log10(np.maximum(values, 1e-30) / np.maximum(point_noise, 1e-30))
    columns = (
        x_axis[x_idx],
        y_axis[y_idx],
        z_axis[z_idx],
        values,
        point_noise,
        point_threshold,
        snr_db,
    )
    return np.column_stack(columns).astype(np.float32, copy=False)


def save_point_cloud(path: Path, points: PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        header = "x_m,y_m,z_m,power,noise,threshold,snr_db"
        np.savetxt(path, points, delimiter=",", header=header, comments="", fmt="%.8g")
    else:
        np.save(path, points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate xyz point clouds from RT-Pose DZYX tensors with 3D CA-CFAR.")
    parser.add_argument("--input", required=True, type=Path, help="Input D-Z-Y-X .npy radar tensor")
    parser.add_argument("--out", required=True, type=Path, help="Output .npy or .csv point cloud")
    parser.add_argument("--guard-cells", type=parse_xyz_cells, default=(1, 1, 1), help="Guard cells as x,y,z or one value")
    parser.add_argument(
        "--training-cells",
        type=parse_xyz_cells,
        default=(4, 4, 2),
        help="Training cells as x,y,z or one value",
    )
    parser.add_argument("--pfa", type=float, default=1e-4, help="Desired false alarm probability")
    parser.add_argument("--doppler-reducer", choices=("sum", "max"), default="sum", help="Reduce Doppler to 3D power")
    parser.add_argument(
        "--boundary-mode",
        choices=("constant", "nearest", "reflect", "mirror", "wrap"),
        default="constant",
        help="Boundary mode used by scipy.ndimage convolution",
    )
    parser.add_argument(
        "--min-training-fraction",
        type=float,
        default=0.5,
        help="Minimum available training-cell fraction near edges",
    )
    parser.add_argument("--roi", type=parse_roi, default=None, help="Optional ROI: x_min,x_max,y_min,y_max,z_min,z_max")
    parser.add_argument("--local-max", action="store_true", help="Keep only local maxima after CFAR thresholding")
    parser.add_argument(
        "--nms-cells",
        type=parse_xyz_cells,
        default=None,
        help="Local-max radius as x,y,z; defaults to --guard-cells when --local-max is set",
    )
    parser.add_argument("--max-points", type=int, default=0, help="Keep only strongest N points; 0 keeps all")
    parser.add_argument("--save-mask", type=Path, default=None, help="Optional .npy path for the boolean detection mask")
    parser.add_argument("--metadata-out", type=Path, default=None, help="Optional JSON summary output")
    parser.add_argument("--pose-train", type=Path, default=None, help="Optional Train.json used to keep pose bbox points")
    parser.add_argument("--sequence", default="", help="Sequence ID for --pose-train")
    parser.add_argument("--radar-id", default="", help="Radar frame ID for --pose-train; defaults to input stem")
    parser.add_argument("--annotation-index", type=int, default=0, help="Person annotation index for --pose-train")
    parser.add_argument("--pose-margin", type=float, default=1.0, help="Meters added around pose bbox for --pose-train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    axes = xyz_axes(RadarConfig())

    spatial_slices = roi_to_slices(axes, args.roi)
    power = power_zyx(args.input, spatial_slices, args.doppler_reducer)
    point_axes, point_slices = axes_for_power_shape(axes, power.shape, spatial_slices, args.roi)
    guard_zyx = xyz_to_zyx(args.guard_cells)
    training_zyx = xyz_to_zyx(args.training_cells)
    detections, noise, threshold = ca_cfar_3d(
        power,
        guard_zyx,
        training_zyx,
        args.pfa,
        boundary_mode=args.boundary_mode,
        min_training_fraction=args.min_training_fraction,
    )
    if args.local_max:
        nms_zyx = xyz_to_zyx(args.nms_cells) if args.nms_cells is not None else guard_zyx
        detections = apply_local_max_filter(detections, power, nms_zyx, args.boundary_mode)

    pose_bounds = None
    if args.pose_train is not None:
        if not args.sequence:
            raise ValueError("--sequence is required with --pose-train")
        radar_id = args.radar_id or args.input.stem
        pose_bounds = load_pose_bounds(
            args.pose_train,
            args.sequence,
            radar_id,
            args.annotation_index,
            args.pose_margin,
        )
        detections &= xyz_bounds_mask(point_axes, pose_bounds)

    points = detections_to_point_cloud(detections, power, noise, threshold, point_axes, point_slices, args.max_points)
    save_point_cloud(args.out, points)
    if args.save_mask is not None:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_mask, detections)

    summary = {
        "input": str(args.input),
        "output": str(args.out),
        "points": int(points.shape[0]),
        "power_shape_zyx": list(power.shape),
        "guard_cells_xyz": list(args.guard_cells),
        "training_cells_xyz": list(args.training_cells),
        "pfa": args.pfa,
        "doppler_reducer": args.doppler_reducer,
        "local_max": bool(args.local_max),
        "nms_cells_xyz": list(args.nms_cells) if args.nms_cells is not None else None,
        "pose_bounds": list(pose_bounds) if pose_bounds is not None else None,
        "columns": ["x_m", "y_m", "z_m", "power", "noise", "threshold", "snr_db"],
    }
    if args.metadata_out is not None:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
