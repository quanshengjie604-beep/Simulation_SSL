#!/usr/bin/env python3
"""Create ROI Doppler spectra directly from RT-Pose raw radar echo bins.

The output is a ``Doppler x Frame`` float32 array. For each radar frame, motion
annotations are cubic-spline interpolated onto the radar frame id, all joints
define an xyz bounding box expanded by ``--roi-margin``, and the Doppler power
is averaged over xyz voxels inside that ROI.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import fcntl
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIB = Path(__file__).resolve().parent / "calibrateResults_high.mat"
START_FRAME_JSON = "doppler_spectrum_start_frames.json"

from raw_echo_to_xyz import (
    RadarConfig,
    as_numpy,
    astype_backend,
    files_for_index,
    interpolation_axes,
    load_calibration,
    numel_backend,
    original_interp_indices,
    raw_frame_to_drae,
    read_valid_num_frames,
    resolve_backend,
    transpose_backend,
    unique_file_indices,
    xyz_axes,
)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, **_: object):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            return iter(self.iterable)

        def update(self, _: int = 1) -> None:
            return None

        def close(self) -> None:
            return None


_BACKEND_CACHE: dict[tuple[str, int | None], tuple[object, bool]] = {}
_CALIB_CACHE: dict[tuple[Path, bool, bool], tuple[np.ndarray | None, np.ndarray | None]] = {}
_FILES_CACHE: dict[tuple[Path, str], dict[str, Path]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw echo frames to an ROI Doppler spectrum.")
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "datasets" / "GT_sequences"), help="Path containing RT-Pose sequences/")
    parser.add_argument("--sequence", type=int, default=1, help="Sequence ID to process")
    parser.add_argument("--train", default=str(REPO_ROOT / "datasets" / "Train.json"), help="Motion annotation JSON")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "GT" / "doppler_spectrum_roi"), help="Output directory")
    parser.add_argument("--roi-margin", type=float, default=0.5, help="Meters added around the interpolated pose bbox")
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy", "torch"), default="auto", help="Array backend")
    parser.add_argument("--gpu-device", type=int, help="CUDA device ID for GPU backends")
    parser.add_argument("--workers", type=int, default=1, help="Parallel frame workers")
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Maximum submitted frame jobs waiting/running at once; 0 uses 2*workers.",
    )
    parser.add_argument(
        "--calib",
        default=str(DEFAULT_CALIB),
        help="Calibration .mat path, relative to cwd or absolute.",
    )
    parser.add_argument(
        "--rangemat-correction",
        choices=("on", "off"),
        required=True,
        help="Explicitly enable or disable RangeMat fast-time phase-slope correction.",
    )
    parser.add_argument(
        "--peakvalmat-correction",
        choices=("on", "off"),
        required=True,
        help="Explicitly enable or disable PeakValMat phase-only channel correction.",
    )
    parser.add_argument("--file-idx", default="", help="Radar bin file index, e.g. 0000; default uses the first one")
    parser.add_argument("--frame-start", type=int, help="First radar frame id; default is first annotated frame")
    parser.add_argument("--frame-stop", type=int, help="Exclusive stop radar frame id; default is last annotated frame + 1")
    parser.add_argument(
        "--raw-frame-start",
        type=int,
        default=1,
        help=(
            "Radar frame id represented by the first frame stored in the raw bin files. "
            "Use this for synthetic files that store a contiguous subset starting after frame 1."
        ),
    )
    parser.add_argument(
        "--nchirp-loops",
        type=int,
        default=64,
        help="Number of chirp loops per radar frame in the raw echo files.",
    )
    parser.add_argument(
        "--roi-reducer",
        choices=("mean", "max", "topk-mean"),
        required=True,
        help="How to reduce Doppler power over xyz voxels inside the ROI.",
    )
    parser.add_argument(
        "--roi-topk-fraction",
        type=float,
        default=0.05,
        help="Fraction of highest-power ROI voxels averaged when --roi-reducer=topk-mean.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy output")
    return parser.parse_args()


def cached_backend(backend: str, gpu_device: int | None):
    key = (backend, gpu_device)
    cached = _BACKEND_CACHE.get(key)
    if cached is None:
        cached = resolve_backend(backend, gpu_device=gpu_device)
        _BACKEND_CACHE[key] = cached
    return cached


def cached_calibration(
    calib_path: Path,
    use_rangemat_correction: bool,
    use_peakvalmat_correction: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    key = (calib_path, use_rangemat_correction, use_peakvalmat_correction)
    cached = _CALIB_CACHE.get(key)
    if cached is None:
        if use_rangemat_correction or use_peakvalmat_correction:
            cached = load_calibration(calib_path)
        else:
            cached = (None, None)
        _CALIB_CACHE[key] = cached
    return cached


def cached_files(sequence_dir: Path, file_idx: str) -> dict[str, Path]:
    key = (sequence_dir, file_idx)
    cached = _FILES_CACHE.get(key)
    if cached is None:
        cached = files_for_index(sequence_dir / "radar" / "bin", file_idx)
        _FILES_CACHE[key] = cached
    return cached


def process_one_spectrum_frame(job: tuple) -> tuple[int, int, np.ndarray]:
    (
        sequence_dir,
        file_idx,
        column,
        frame_id,
        raw_frame_start,
        roi,
        axes,
        calib_path,
        backend,
        gpu_device,
        use_rangemat_correction,
        use_peakvalmat_correction,
        reducer,
        topk_fraction,
        cfg,
    ) = job
    xp, _ = cached_backend(backend, gpu_device)
    files = cached_files(sequence_dir, file_idx)
    range_mat, peak_val_mat = cached_calibration(calib_path, use_rangemat_correction, use_peakvalmat_correction)
    raw_frame_idx = int(frame_id) - int(raw_frame_start) + 1
    drae = raw_frame_to_drae(
        files,
        raw_frame_idx,
        range_mat,
        peak_val_mat,
        use_rangemat_correction,
        use_peakvalmat_correction,
        xp=xp,
        cfg=cfg,
    )
    spectrum_column = roi_doppler_spectrum(
        drae,
        roi,
        axes,
        cfg,
        xp=xp,
        reducer=reducer,
        topk_fraction=topk_fraction,
    )
    return int(column), int(frame_id), spectrum_column


def run_spectrum_pool(jobs: list[tuple], workers: int, mp_context, max_in_flight: int) -> list[tuple[int, int, np.ndarray]]:
    in_flight_limit = max_in_flight if max_in_flight > 0 else max(1, workers * 2)
    job_iter = iter(jobs)
    results: list[tuple[int, int, np.ndarray]] = []
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
        pending: set[futures.Future] = set()
        for _ in range(min(in_flight_limit, len(jobs))):
            pending.add(executor.submit(process_one_spectrum_frame, next(job_iter)))
        progress = tqdm(total=len(jobs), desc="doppler frames")
        try:
            while pending:
                done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                for fut in done:
                    results.append(fut.result())
                    progress.update(1)
                    try:
                        pending.add(executor.submit(process_one_spectrum_frame, next(job_iter)))
                    except StopIteration:
                        pass
        finally:
            progress.close()
    return results


def load_sequence_poses(train_path: Path, sequence: int) -> tuple[np.ndarray, np.ndarray]:
    with train_path.open("r", encoding="utf-8") as f:
        train = json.load(f)
    seq_key = str(sequence)
    if seq_key not in train:
        raise KeyError(f"Sequence {seq_key} was not found in {train_path}")

    items: list[tuple[int, np.ndarray]] = []
    seen: set[int] = set()
    for frame_key in sorted(train[seq_key], key=lambda item: int(item) if item.isdigit() else item):
        for ann in train[seq_key][frame_key] or []:
            radar_id = ann.get("Radar_frameID")
            pose = ann.get("pose")
            if not radar_id or not pose:
                continue
            radar_frame = int(radar_id)
            if radar_frame in seen:
                continue
            items.append((radar_frame, np.asarray(pose, dtype=np.float64)))
            seen.add(radar_frame)

    if len(items) < 2:
        raise RuntimeError(f"Sequence {sequence} needs at least two annotated radar frames for interpolation")

    items.sort(key=lambda item: item[0])
    frame_ids = np.asarray([item[0] for item in items], dtype=np.float64)
    poses = np.stack([item[1] for item in items], axis=0)
    finite = np.isfinite(poses).all(axis=(1, 2))
    return frame_ids[finite], poses[finite]


def interpolate_poses(
    annotated_frame_ids: np.ndarray,
    poses: np.ndarray,
    target_frame_ids: np.ndarray,
) -> np.ndarray:
    flat = poses.reshape(poses.shape[0], -1)
    spline = CubicSpline(annotated_frame_ids, flat, axis=0, bc_type="natural", extrapolate=False)
    interpolated = spline(target_frame_ids.astype(np.float64))
    if not np.isfinite(interpolated).all():
        raise RuntimeError("Cubic spline produced NaN/Inf values; target frames must stay inside annotation range")
    return interpolated.reshape(len(target_frame_ids), poses.shape[1], poses.shape[2])


def frame_rois(
    poses: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    margin: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x_axis, y_axis, z_axis = axes
    rois = []
    axis_triplets = ((x_axis, 0), (y_axis, 1), (z_axis, 2))
    for pose in poses:
        slices = []
        for axis, coord_idx in axis_triplets:
            low = max(float(np.nanmin(pose[:, coord_idx]) - margin), float(axis[0]))
            high = min(float(np.nanmax(pose[:, coord_idx]) + margin), float(axis[-1]))
            idx = np.flatnonzero((axis >= low) & (axis <= high))
            if idx.size == 0:
                nearest = int(np.argmin(np.abs(axis - ((low + high) * 0.5))))
                idx = np.asarray([nearest], dtype=np.int64)
            slices.append(idx.astype(np.int64, copy=False))
        rois.append((slices[0], slices[1], slices[2]))
    return rois


def lin_index_3d(ri, ei, ai, elev_len: int, az_len: int):
    return (ri * elev_len + ei) * az_len + ai


def reduce_roi_power(power, reducer: str, topk_fraction: float, xp=np):
    if reducer == "mean":
        if hasattr(power, "mean"):
            return power.mean(dim=1) if hasattr(power, "numel") else power.mean(axis=1)
        return xp.mean(power, axis=1)
    if reducer == "max":
        if hasattr(power, "max"):
            return power.max(dim=1).values if hasattr(power, "numel") else power.max(axis=1)
        return xp.max(power, axis=1)
    if reducer == "topk-mean":
        n_voxels = int(power.shape[1])
        k = max(1, int(round(n_voxels * float(topk_fraction))))
        k = min(k, n_voxels)
        if hasattr(power, "numel"):
            return power.topk(k, dim=1).values.mean(dim=1)
        if xp.__name__ == "cupy":
            vals = xp.partition(power, n_voxels - k, axis=1)[:, n_voxels - k :]
            return vals.mean(axis=1)
        vals = np.partition(as_numpy(power), n_voxels - k, axis=1)[:, n_voxels - k :]
        return vals.mean(axis=1)
    raise ValueError(f"Unsupported ROI reducer: {reducer}")


def roi_doppler_spectrum(
    drae,
    roi: tuple[np.ndarray, np.ndarray, np.ndarray],
    axes,
    cfg: RadarConfig,
    xp=np,
    reducer: str = "mean",
    topk_fraction: float = 0.05,
) -> np.ndarray:
    x_axis_cpu, y_axis_cpu, z_axis_cpu = axes
    x_idx, y_idx, z_idx = roi
    if not (x_idx.size and y_idx.size and z_idx.size):
        return np.full(cfg.doppler_fft_size, np.nan, dtype=np.float32)

    arr_x = xp.asarray(x_axis_cpu[x_idx].astype(np.float32, copy=False))
    arr_y = xp.asarray(y_axis_cpu[y_idx].astype(np.float32, copy=False))
    arr_z = xp.asarray(z_axis_cpu[z_idx].astype(np.float32, copy=False))
    arr_range_cpu, arr_elevation_cpu, arr_azimuth_cpu = interpolation_axes()
    arr_range = xp.asarray(arr_range_cpu.astype(np.float32, copy=False))
    arr_elevation = xp.asarray(arr_elevation_cpu.astype(np.float32, copy=False))
    arr_azimuth = xp.asarray(arr_azimuth_cpu.astype(np.float32, copy=False))

    zz, yy, xx = xp.meshgrid(arr_z, arr_y, arr_x, indexing="ij")
    rr = xp.sqrt(xx * xx + yy * yy + zz * zz)
    aa = xp.where(yy == 0.0, xp.pi / 2.0, xp.arctan(xx / yy))
    aa = xp.where(aa < 0.0, aa + xp.pi, aa)
    horiz = xp.sqrt(xx * xx + yy * yy)
    ee = xp.where(horiz == 0.0, xp.pi / 2.0, xp.arctan(zz / horiz))

    deg = xp.pi / 180.0
    ee = ee + 30.0 * deg
    aa = 150.0 * deg - aa

    flat_r = rr.ravel()
    flat_e = ee.ravel()
    flat_a = aa.ravel()
    drea = transpose_backend(drae, (0, 1, 3, 2))
    source = drea.reshape(drea.shape[0], -1)
    _, range_len, elev_len, az_len = drea.shape

    r0, r1 = original_interp_indices(flat_r, arr_range, xp=xp)
    e0, e1 = original_interp_indices(flat_e, arr_elevation, xp=xp)
    a0, a1 = original_interp_indices(flat_a, arr_azimuth, xp=xp)
    valid = (
        (flat_r >= 0.0)
        & (flat_r <= 11.6)
        & (flat_e >= 0.0)
        & (flat_e <= 60.0 * deg)
        & (flat_a >= 0.0)
        & (flat_a <= 120.0 * deg)
        & (r0 >= 0)
        & (e0 >= 0)
        & (a0 >= 0)
        & (r1 < range_len - 1)
        & (e1 < elev_len - 1)
        & (a1 < az_len - 1)
    )

    if not bool(as_numpy(xp.any(valid))):
        return np.full(cfg.doppler_fft_size, np.nan, dtype=np.float32)

    rv = flat_r[valid]
    ev = flat_e[valid]
    av = flat_a[valid]
    r0v, r1v = r0[valid], r1[valid]
    e0v, e1v = e0[valid], e1[valid]
    a0v, a1v = a0[valid], a1[valid]

    dr = arr_range[r1v] - arr_range[r0v]
    de = arr_elevation[e1v] - arr_elevation[e0v]
    da = arr_azimuth[a1v] - arr_azimuth[a0v]
    inv = 1.0 / (dr * de * da)

    wr0 = arr_range[r1v] - rv
    wr1 = rv - arr_range[r0v]
    we0 = arr_elevation[e1v] - ev
    we1 = ev - arr_elevation[e0v]
    wa0 = arr_azimuth[a1v] - av
    wa1 = av - arr_azimuth[a0v]

    interp = (
        source[:, lin_index_3d(r0v, e0v, a0v, elev_len, az_len)] * (wr0 * we0 * wa0)
        + source[:, lin_index_3d(r0v, e0v, a1v, elev_len, az_len)] * (wr0 * we0 * wa1)
        + source[:, lin_index_3d(r0v, e1v, a0v, elev_len, az_len)] * (wr0 * we1 * wa0)
        + source[:, lin_index_3d(r0v, e1v, a1v, elev_len, az_len)] * (wr0 * we1 * wa1)
        + source[:, lin_index_3d(r1v, e0v, a0v, elev_len, az_len)] * (wr1 * we0 * wa0)
        + source[:, lin_index_3d(r1v, e0v, a1v, elev_len, az_len)] * (wr1 * we0 * wa1)
        + source[:, lin_index_3d(r1v, e1v, a0v, elev_len, az_len)] * (wr1 * we1 * wa0)
        + source[:, lin_index_3d(r1v, e1v, a1v, elev_len, az_len)] * (wr1 * we1 * wa1)
    )
    interp = astype_backend(interp * inv, xp.complex64, copy=False)
    power = xp.abs(interp) ** 2
    spectrum = reduce_roi_power(power, reducer, topk_fraction, xp=xp)
    return as_numpy(spectrum).astype(np.float32, copy=False)


def update_start_frame_json(out_dir: Path, npy_path: Path, frame_ids: np.ndarray, spectrum: np.ndarray) -> Path:
    json_path = out_dir / START_FRAME_JSON
    lock_path = json_path.with_name(json_path.name + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not isinstance(metadata, dict):
                raise ValueError(f"{json_path} must contain a JSON object")
        else:
            metadata = {}

        metadata[npy_path.name] = {
            "start_frame": int(frame_ids[0]),
            "stop_frame_exclusive": int(frame_ids[-1] + 1),
            "num_frames": int(frame_ids.size),
            "shape": [int(dim) for dim in spectrum.shape],
        }
        tmp_path = json_path.with_name(f"{json_path.name}.{npy_path.stem}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(json_path)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return json_path


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    sequence_dir = dataset_dir / str(args.sequence)
    bin_dir = sequence_dir / "radar" / "bin"
    if not bin_dir.exists():
        raise FileNotFoundError(f"Missing radar bin directory: {bin_dir}")

    file_idx = args.file_idx or unique_file_indices(bin_dir)[0]
    files = files_for_index(bin_dir, file_idx)
    valid_frames = read_valid_num_frames(files["master_idx"])

    annotated_ids, annotated_poses = load_sequence_poses(Path(args.train), args.sequence)
    raw_frame_start = int(args.raw_frame_start)
    cfg = RadarConfig(nchirp_loops=int(args.nchirp_loops))
    raw_frame_stop = raw_frame_start + valid_frames
    start = int(args.frame_start if args.frame_start is not None else annotated_ids[0])
    stop = int(args.frame_stop if args.frame_stop is not None else annotated_ids[-1] + 1)
    start = max(start, 2, raw_frame_start)
    stop = min(stop, raw_frame_stop, int(annotated_ids[-1] + 1))
    frame_ids = np.arange(start, stop, dtype=np.int64)
    if frame_ids.size == 0:
        raise RuntimeError("No target frames after intersecting annotation and radar ranges")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{args.sequence}.npy"
    if npy_path.exists() and not args.overwrite:
        raise FileExistsError(f"{npy_path} already exists; pass --overwrite to recompute")

    poses = interpolate_poses(annotated_ids, annotated_poses, frame_ids)
    axes = xyz_axes(cfg)
    rois = frame_rois(poses, axes, args.roi_margin)
    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        calib_path = (Path.cwd() / calib_path).resolve()
    use_rangemat_correction = args.rangemat_correction == "on"
    use_peakvalmat_correction = args.peakvalmat_correction == "on"
    xp, is_gpu = resolve_backend(args.backend, gpu_device=args.gpu_device)
    if args.backend == "auto":
        print(f"Using {'GPU' if is_gpu else 'CPU'} backend")

    spectrum = np.empty((cfg.doppler_fft_size, len(frame_ids)), dtype=np.float32)
    if args.workers <= 1:
        range_mat, peak_val_mat = cached_calibration(
            calib_path,
            use_rangemat_correction,
            use_peakvalmat_correction,
        )
        for col, (frame_id, roi) in enumerate(tqdm(list(zip(frame_ids, rois)), desc=f"sequence {args.sequence}")):
            raw_frame_idx = int(frame_id) - raw_frame_start + 1
            drae = raw_frame_to_drae(
                files,
                raw_frame_idx,
                range_mat,
                peak_val_mat,
                use_rangemat_correction,
                use_peakvalmat_correction,
                xp=xp,
                cfg=cfg,
            )
            spectrum[:, col] = roi_doppler_spectrum(
                drae,
                roi,
                axes,
                cfg,
                xp=xp,
                reducer=args.roi_reducer,
                topk_fraction=args.roi_topk_fraction,
            )
    else:
        jobs = [
            (
                sequence_dir,
                file_idx,
                col,
                int(frame_id),
                raw_frame_start,
                roi,
                axes,
                calib_path,
                args.backend,
                args.gpu_device,
                use_rangemat_correction,
                use_peakvalmat_correction,
                args.roi_reducer,
                args.roi_topk_fraction,
                cfg,
            )
            for col, (frame_id, roi) in enumerate(zip(frame_ids, rois))
        ]
        mp_context = mp.get_context("spawn") if is_gpu else None
        for col, _frame_id, spectrum_column in run_spectrum_pool(
            jobs,
            args.workers,
            mp_context,
            args.max_in_flight,
        ):
            spectrum[:, col] = spectrum_column

    np.save(npy_path, spectrum)
    json_path = update_start_frame_json(out_dir, npy_path, frame_ids, spectrum)
    print(f"wrote {npy_path} shape={spectrum.shape} frames={frame_ids[0]}..{frame_ids[-1]}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
