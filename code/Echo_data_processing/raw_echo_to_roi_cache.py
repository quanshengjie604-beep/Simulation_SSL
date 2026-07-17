"""Convert RT-Pose raw radar/bin echo directly to normalized ROI cache tensors.

This is the direct equivalent of:

  raw_echo_to_xyz.py -> cache_radar_roi_f16.py

but it only interpolates the ROI grid used by the HR3D training config and
writes ``radar/npy_DZYX_mag_roi_f16_norm`` directly. The output shape is
``Doppler(64) x Z(16) x Y(64) x X(160)`` with dtype ``float16``.
Magnitude normalization is per-frame percentile scaling: p85 maps to 0 and
p99.9 maps to 1, with values above p99.9 left as linear values greater than 1.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIB = Path(__file__).resolve().parent / "calibrateResults_high.mat"

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from raw_echo_to_xyz import (  # noqa: E402
    RadarConfig,
    as_numpy,
    astype_backend,
    files_for_index,
    flip_backend,
    interpolation_axes,
    load_calibration,
    numel_backend,
    original_interp_indices,
    raw_frame_to_drae,
    read_valid_num_frames,
    resolve_backend,
    sequence_ids,
    transpose_backend,
    unique_file_indices,
    xyz_axes,
)


ROI = {
    "z": [-1.0875000000000021, 4.7125],
    "y": [-5.0250000000000234, 5.024999999999931],
    "x": [0.7703125, 8.0203125],
}
ROI_NORM_LOW_PERCENTILE = 85.0
ROI_NORM_HIGH_PERCENTILE = 99.9
_BACKEND_CACHE: dict[tuple[str, int | None], tuple[object, bool]] = {}
_CALIB_CACHE: dict[tuple[Path, bool, bool], tuple[np.ndarray | None, np.ndarray | None]] = {}
_FILES_CACHE: dict[tuple[Path, str], dict[str, Path]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HR3D ROI radar cache tensors directly from RT-Pose radar/bin echo."
    )
    parser.add_argument("--dataset_dir", required=True, help="Path containing RT-Pose sequences/")
    parser.add_argument("--out-dataset-dir", default="", help="Output dataset root; defaults to --dataset_dir")
    parser.add_argument("--sequence", type=int, nargs="*", help="Sequence IDs to process; default: labels decide")
    parser.add_argument("--frames", type=int, nargs="*", help="One-based radar frame IDs to process")
    parser.add_argument(
        "--label-files",
        nargs="+",
        default=[
            str(REPO_ROOT / "datasets" / "Train_sp120_train_minus_val6.json"),
            str(REPO_ROOT / "datasets" / "Test_sp120_by_motion6.json"),
        ],
        help="Label files used to collect sequence/frame jobs",
    )
    parser.add_argument(
        "--output-dir",
        default="npy_DZYX_mag_roi_f16_norm",
        help="Directory under sequences/<id>/radar/ for ROI cache files",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel frame workers")
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Maximum submitted frame jobs waiting/running at once; 0 uses 2*workers.",
    )
    parser.add_argument("--x-chunk", type=int, default=64, help="Number of ROI X bins interpolated per chunk")
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy", "torch"), default="auto")
    parser.add_argument("--gpu-device", type=int, help="CUDA device ID for GPU backends")
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
    parser.add_argument(
        "--calib",
        default=str(DEFAULT_CALIB),
        help="Calibration .mat path, relative to cwd or absolute.",
    )
    parser.add_argument(
        "--raw-frame-start",
        type=int,
        required=True,
        help=(
            "Radar frame id represented by the first frame stored in the raw bin files. "
            "Required; raw bin frame 1 maps to this Radar_frameID."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing outputs")
    parser.add_argument("--limit", type=int, help="Debug limit after collecting jobs")
    parser.add_argument("--dry-run", action="store_true", help="Only collect and report jobs")
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


def run_frame_pool(jobs: list[tuple], workers: int, mp_context, max_in_flight: int) -> None:
    in_flight_limit = max_in_flight if max_in_flight > 0 else max(1, workers * 2)
    job_iter = iter(jobs)
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
        pending: set[futures.Future] = set()
        for _ in range(min(in_flight_limit, len(jobs))):
            pending.add(executor.submit(process_one_frame, next(job_iter)))
        progress = tqdm(total=len(jobs), desc="roi frames")
        try:
            while pending:
                done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                for fut in done:
                    fut.result()
                    progress.update(1)
                    try:
                        pending.add(executor.submit(process_one_frame, next(job_iter)))
                    except StopIteration:
                        pass
        finally:
            progress.close()


def axis_roi_indices(axis: np.ndarray, bounds: list[float]) -> tuple[int, int]:
    low, high = bounds
    start = int(np.argmin(np.abs(axis - low)))
    stop = int(np.argmin(np.abs(axis - high)))
    if high <= axis[-1]:
        stop -= 1
    if stop < start:
        raise ValueError(f"Invalid ROI bounds {bounds} for axis [{axis[0]}, {axis[-1]}]")
    return start, stop


def roi_slices(cfg: RadarConfig) -> tuple[slice, slice, slice]:
    arr_x, arr_y, arr_z = xyz_axes(cfg)
    z0, z1 = axis_roi_indices(arr_z, ROI["z"])
    y0, y1 = axis_roi_indices(arr_y, ROI["y"])
    x0, x1 = axis_roi_indices(arr_x, ROI["x"])
    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)


def normalize_roi_mag(roi_complex: np.ndarray) -> np.ndarray:
    roi = np.abs(roi_complex).astype(np.float32, copy=False)
    finite_roi = roi[np.isfinite(roi)]
    if finite_roi.size == 0:
        raise FloatingPointError("ROI magnitude contains no finite values")
    norm_start, norm_one = np.percentile(
        finite_roi,
        [ROI_NORM_LOW_PERCENTILE, ROI_NORM_HIGH_PERCENTILE],
    )
    denom = float(norm_one - norm_start)
    if denom <= 0.0:
        return np.zeros_like(roi, dtype=np.float16)
    roi = (roi - float(norm_start)) / denom
    roi[roi < 0.0] = 0.0
    return roi.astype(np.float16, copy=False)


def drae_to_roi_cache(drae, output_path: Path, x_chunk: int, xp=np) -> None:
    cfg = RadarConfig()
    z_slice, y_slice, x_slice = roi_slices(cfg)
    arr_x_cpu, arr_y_cpu, arr_z_cpu = xyz_axes(cfg)
    arr_x_cpu = arr_x_cpu[x_slice]
    arr_y_cpu = arr_y_cpu[y_slice]
    arr_z_cpu = arr_z_cpu[z_slice]
    arr_range_cpu, arr_elevation_cpu, arr_azimuth_cpu = interpolation_axes()

    arr_x = xp.asarray(arr_x_cpu)
    arr_y = xp.asarray(arr_y_cpu)
    arr_z = xp.asarray(arr_z_cpu)
    arr_range = xp.asarray(arr_range_cpu)
    arr_elevation = xp.asarray(arr_elevation_cpu)
    arr_azimuth = xp.asarray(arr_azimuth_cpu)
    deg = xp.pi / 180.0

    drea = transpose_backend(drae, (0, 1, 3, 2))
    source = drea.reshape(drea.shape[0], -1)
    _, range_len, elev_len, az_len = drea.shape

    roi_complex = np.empty(
        (cfg.doppler_fft_size, len(arr_z_cpu), len(arr_y_cpu), len(arr_x_cpu)),
        dtype=np.complex64,
    )

    zz, yy = xp.meshgrid(arr_z, arr_y, indexing="ij")
    zz = zz[:, :, None]
    yy = yy[:, :, None]

    for x0 in range(0, len(arr_x_cpu), x_chunk):
        x1 = min(x0 + x_chunk, len(arr_x_cpu))
        xx = arr_x[x0:x1][None, None, :]

        rr = xp.sqrt(xx * xx + yy * yy + zz * zz)
        aa = xp.where(yy == 0.0, xp.pi / 2.0, xp.arctan(xx / yy))
        aa = xp.where(aa < 0.0, aa + xp.pi, aa)
        horiz = xp.sqrt(xx * xx + yy * yy)
        ee = xp.where(horiz == 0.0, xp.pi / 2.0, xp.arctan(zz / horiz))

        ee = ee + 30.0 * deg
        aa = 150.0 * deg - aa

        flat_r = rr.ravel()
        flat_e = ee.ravel()
        flat_a = aa.ravel()
        valid = (
            (flat_r >= 0.0)
            & (flat_r <= 11.6)
            & (flat_e >= 0.0)
            & (flat_e <= 60.0 * deg)
            & (flat_a >= 0.0)
            & (flat_a <= 120.0 * deg)
        )

        r0, r1 = original_interp_indices(flat_r, arr_range, xp=xp)
        e0, e1 = original_interp_indices(flat_e, arr_elevation, xp=xp)
        a0, a1 = original_interp_indices(flat_a, arr_azimuth, xp=xp)
        valid &= (
            (r0 >= 0)
            & (e0 >= 0)
            & (a0 >= 0)
            & (r1 < range_len - 1)
            & (e1 < elev_len - 1)
            & (a1 < az_len - 1)
        )

        chunk = xp.ones((cfg.doppler_fft_size, numel_backend(flat_r)), dtype=xp.complex64)
        if bool(as_numpy(xp.any(valid))):
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

            def lin_index(ri, ei, ai):
                return (ri * elev_len + ei) * az_len + ai

            interp = (
                source[:, lin_index(r0v, e0v, a0v)] * (wr0 * we0 * wa0)
                + source[:, lin_index(r0v, e0v, a1v)] * (wr0 * we0 * wa1)
                + source[:, lin_index(r0v, e1v, a0v)] * (wr0 * we1 * wa0)
                + source[:, lin_index(r0v, e1v, a1v)] * (wr0 * we1 * wa1)
                + source[:, lin_index(r1v, e0v, a0v)] * (wr1 * we0 * wa0)
                + source[:, lin_index(r1v, e0v, a1v)] * (wr1 * we0 * wa1)
                + source[:, lin_index(r1v, e1v, a0v)] * (wr1 * we1 * wa0)
                + source[:, lin_index(r1v, e1v, a1v)] * (wr1 * we1 * wa1)
            )
            chunk[:, valid] = astype_backend(interp * inv, xp.complex64, copy=False)

        chunk = chunk.reshape(cfg.doppler_fft_size, len(arr_z_cpu), len(arr_y_cpu), x1 - x0)
        chunk = flip_backend(chunk, axis=2)
        roi_complex[:, :, :, x0:x1] = as_numpy(chunk)

    roi = normalize_roi_mag(roi_complex)
    with output_path.open("wb") as handle:
        np.save(handle, roi)


def output_name(frame_idx: int) -> str:
    return f"{frame_idx:06d}.npy"


def collect_label_frames(root: Path, label_files: list[str]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for label_file in label_files:
        with (root / label_file).open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
        for seq_text, frames in labels.items():
            seq = int(seq_text)
            seq_frames = out.setdefault(seq, set())
            for frame_objs in frames.values():
                for obj in frame_objs:
                    seq_frames.add(int(obj["Radar_frameID"]))
    return out


def selected_sequences(dataset_dir: Path, args: argparse.Namespace) -> list[int]:
    if args.sequence:
        return [int(x) for x in args.sequence]
    return sorted(collect_label_frames(Path.cwd(), args.label_files))


def build_jobs(dataset_dir: Path, out_dataset_dir: Path, args: argparse.Namespace, calib_path: Path) -> list[tuple]:
    label_frames = collect_label_frames(Path.cwd(), args.label_files)
    jobs = []
    for seq in sequence_ids(dataset_dir, selected_sequences(dataset_dir, args)):
        sequence_dir = dataset_dir / str(seq)
        bin_dir = sequence_dir / "radar" / "bin"
        if not bin_dir.exists():
            print(f"Skip sequence {seq}: missing {bin_dir}")
            continue
        wanted_frames = set(args.frames or label_frames.get(seq, []))
        if not wanted_frames:
            print(f"Skip sequence {seq}: no selected frames")
            continue

        for file_idx in unique_file_indices(bin_dir):
            files = files_for_index(bin_dir, file_idx)
            valid_frames = read_valid_num_frames(files["master_idx"])
            raw_frame_start = int(args.raw_frame_start)
            raw_frame_stop = raw_frame_start + valid_frames
            for frame_idx in sorted(wanted_frames):
                if frame_idx < max(2, raw_frame_start) or frame_idx >= raw_frame_stop:
                    print(
                        f"Skip sequence {seq} frame {frame_idx}: "
                        f"valid range is {max(2, raw_frame_start)}..{raw_frame_stop - 1}"
                    )
                    continue
                jobs.append(
                    (
                        sequence_dir,
                        out_dataset_dir / str(seq),
                        file_idx,
                        int(frame_idx),
                        int(raw_frame_start),
                        calib_path,
                        args.output_dir,
                        args.overwrite,
                        args.x_chunk,
                        args.backend,
                        args.gpu_device,
                        args.rangemat_correction == "on",
                        args.peakvalmat_correction == "on",
                    )
                )
    if args.limit is not None:
        jobs = jobs[: args.limit]
    return jobs


def process_one_frame(job: tuple) -> Path:
    (
        sequence_dir,
        out_sequence_dir,
        file_idx,
        frame_idx,
        raw_frame_start,
        calib_path,
        output_dir,
        overwrite,
        x_chunk,
        backend,
        gpu_device,
        use_rangemat_correction,
        use_peakvalmat_correction,
    ) = job

    cfg = RadarConfig()
    xp, _ = cached_backend(backend, gpu_device)
    out_path = out_sequence_dir / "radar" / output_dir / output_name(frame_idx)
    if out_path.exists() and not overwrite:
        return out_path

    range_mat, peak_val_mat = cached_calibration(calib_path, use_rangemat_correction, use_peakvalmat_correction)
    files = cached_files(sequence_dir, file_idx)
    raw_frame_idx = int(frame_idx) - int(raw_frame_start) + 1
    drae = raw_frame_to_drae(
        files,
        raw_frame_idx,
        range_mat,
        peak_val_mat,
        use_rangemat_correction,
        use_peakvalmat_correction,
        xp=xp,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp.npy")
    drae_to_roi_cache(drae, tmp_path, x_chunk=x_chunk, xp=xp)
    tmp_path.replace(out_path)

    # Keep cfg referenced to make the expected output shape explicit for future edits.
    expected_shape = (cfg.doppler_fft_size, 16, 64, 160)
    arr = np.load(out_path, mmap_mode="r")
    if arr.shape != expected_shape or arr.dtype != np.float16:
        raise ValueError(f"{out_path} produced {arr.shape} {arr.dtype}, expected {expected_shape} float16")
    return out_path


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dataset_dir = Path(args.out_dataset_dir).expanduser().resolve() if args.out_dataset_dir else dataset_dir
    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        calib_path = Path.cwd() / calib_path

    jobs = build_jobs(dataset_dir, out_dataset_dir, args, calib_path)
    if not jobs:
        print("No frames to process.")
        return

    print(
        f"jobs={len(jobs)} sequences={len(set(job[0].name for job in jobs))} "
        f"workers={args.workers} backend={args.backend} gpu={args.gpu_device} "
        f"normalization=p{ROI_NORM_LOW_PERCENTILE:g}-p{ROI_NORM_HIGH_PERCENTILE:g} "
        f"rangemat_correction={args.rangemat_correction} "
        f"peakvalmat_correction={args.peakvalmat_correction} "
        f"out={out_dataset_dir} clutter_removal=True",
        flush=True,
    )
    if args.dry_run:
        return
    if args.workers <= 1:
        for job in tqdm(jobs, desc="roi frames"):
            process_one_frame(job)
    else:
        _, is_gpu = resolve_backend(args.backend, gpu_device=args.gpu_device)
        mp_context = mp.get_context("spawn") if is_gpu else None
        run_frame_pool(jobs, args.workers, mp_context, args.max_in_flight)


if __name__ == "__main__":
    main()
