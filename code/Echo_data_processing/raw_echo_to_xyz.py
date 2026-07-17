"""Convert RT-Pose raw cascade radar echo bins directly to D-Z-Y-X tensors.

This script ports the Matlab path used by ``mmWave-Matlab/process_cas.m`` and
``Process_4DRT.m``, enables Doppler clutter removal before Doppler FFT, then
applies the same polar-to-cartesian interpolation and final saved Y-axis
reversal as ``4Dradar2xyz.py``.
The output ``.npy`` files have shape
``Doppler(64) x Z(32) x Y(128) x X(256)`` and dtype ``complex64``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat, savemat

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


BROKEN_SEQUENCES = {44, 68, 107, 155}
DEVICE_NAMES = ("master", "slave1", "slave2", "slave3")
_BACKEND_CACHE: dict[tuple[str, int | None], tuple[object, bool]] = {}
_CALIB_CACHE: dict[tuple[Path, bool, bool], tuple[np.ndarray | None, np.ndarray | None]] = {}
_FILES_CACHE: dict[tuple[Path, str], dict[str, Path]] = {}


class TorchFFT:
    def __init__(self, torch_module):
        self.torch = torch_module

    def fft(self, array, n=None, axis=-1):
        return self.torch.fft.fft(array, n=n, dim=axis)

    def fftshift(self, array, axes=None):
        return self.torch.fft.fftshift(array, dim=axes)


class TorchBackend:
    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.fft = TorchFFT(torch)
        self.pi = torch.pi
        self.float32 = torch.float32
        self.complex64 = torch.complex64

    def asarray(self, array):
        return self.torch.as_tensor(array, device=self.device)

    def empty_like(self, array):
        return self.torch.empty_like(array)

    def empty(self, shape, dtype):
        return self.torch.empty(shape, dtype=dtype, device=self.device)

    def zeros(self, shape, dtype):
        return self.torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype):
        return self.torch.ones(shape, dtype=dtype, device=self.device)

    def arange(self, *args, **kwargs):
        return self.torch.arange(*args, device=self.device, **kwargs)

    def exp(self, array):
        return self.torch.exp(array)

    def abs(self, array):
        return self.torch.abs(array)

    def searchsorted(self, sorted_sequence, values, side="left"):
        return self.torch.searchsorted(sorted_sequence, values, right=(side == "right"))

    def meshgrid(self, *arrays, indexing="ij"):
        return self.torch.meshgrid(*arrays, indexing=indexing)

    def sqrt(self, array):
        return self.torch.sqrt(array)

    def where(self, condition, x, y):
        return self.torch.where(condition, x, y)

    def arctan(self, array):
        return self.torch.atan(array)

    def any(self, array):
        return self.torch.any(array)


def astype_backend(array, dtype, copy=False):
    if hasattr(array, "astype"):
        return array.astype(dtype, copy=copy)
    return array.to(dtype=dtype, copy=copy)


def transpose_backend(array, axes):
    if hasattr(array, "permute"):
        return array.permute(*axes)
    return array.transpose(axes)


def flip_backend(array, axis: int):
    if hasattr(array, "flip"):
        return array.flip(axis)
    return np.flip(array, axis=axis)


def numel_backend(array) -> int:
    if hasattr(array, "numel"):
        return int(array.numel())
    return int(array.size)


def resolve_backend(name: str, gpu_device: int | None = None):
    if name == "numpy":
        return np, False
    if name == "torch":
        import torch

        if gpu_device is not None and not torch.cuda.is_available():
            raise RuntimeError("Torch was requested with --gpu-device, but CUDA is not available.")
        device = f"cuda:{gpu_device or 0}" if torch.cuda.is_available() else "cpu"
        return TorchBackend(device), device.startswith("cuda")

    try:
        import cupy as cp
    except ModuleNotFoundError:
        if name == "cupy":
            raise RuntimeError(
                "CuPy is not installed. Install the CUDA-matched package, "
                "for example cupy-cuda12x or cupy-cuda11x, then rerun with --backend cupy."
            ) from None
        try:
            import torch
        except ModuleNotFoundError:
            return np, False
        if torch.cuda.is_available():
            device = f"cuda:{gpu_device or 0}" if gpu_device is not None else "cuda:0"
            return TorchBackend(device), True
        return np, False

    if gpu_device is not None:
        cp.cuda.Device(gpu_device).use()
    return cp, True


def as_numpy(array):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy as cp

        return cp.asnumpy(array)
    if module == "torch":
        return array.detach().cpu().numpy()
    return array


@dataclass(frozen=True)
class RadarConfig:
    num_adc_sample: int = 256
    adc_sample_rate: float = 5.0e6
    start_freq_const: float = 77.0e9
    chirp_slope: float = 6.4985e13
    chirp_idle_time: float = 5.0e-6
    adc_start_time_const: float = 5.0e-6
    chirp_ramp_end_time: float = 6.0e-5
    nchirp_loops: int = 64
    num_chirps_in_loop: int = 12
    num_rx_per_device: int = 4
    num_devices: int = 4
    slope_calib: float = 78_986_000_000_000.0
    fs_calib: float = 8_000_000.0
    calibration_interp: int = 5

    # Matlab hardware_param.m values, converted to zero-based indices where
    # used for indexing.
    tx_to_enable: tuple[int, ...] = (12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    rx_for_mimo_process: tuple[int, ...] = (
        13,
        14,
        15,
        16,
        1,
        2,
        3,
        4,
        9,
        10,
        11,
        12,
        5,
        6,
        7,
        8,
    )
    tx_position_azimuth: tuple[int, ...] = (11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0)
    tx_position_elevation: tuple[int, ...] = (6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    rx_position_azimuth: tuple[int, ...] = (
        11,
        12,
        13,
        14,
        50,
        51,
        52,
        53,
        46,
        47,
        48,
        49,
        0,
        1,
        2,
        3,
    )

    angle_fft_size: int = 128
    elevation_fft_size: int = 32

    @property
    def num_chirps_per_frame(self) -> int:
        return self.nchirp_loops * self.num_chirps_in_loop

    @property
    def range_fft_size(self) -> int:
        return 1 << int(np.ceil(np.log2(self.num_adc_sample)))

    @property
    def doppler_fft_size(self) -> int:
        return 1 << int(np.ceil(np.log2(self.nchirp_loops)))

    @property
    def chirp_ramp_time(self) -> float:
        return self.num_adc_sample / self.adc_sample_rate

    @property
    def range_max(self) -> float:
        return 11.6

    @property
    def virtual_array(self) -> np.ndarray:
        tx_az = np.array(self.tx_position_azimuth, dtype=np.int64)[np.array(self.tx_to_enable) - 1]
        tx_el = np.array(self.tx_position_elevation, dtype=np.int64)[np.array(self.tx_to_enable) - 1]
        rx_ids = np.array(self.rx_for_mimo_process, dtype=np.int64) - 1
        rx_az = np.array(self.rx_position_azimuth, dtype=np.int64)[rx_ids]
        rx_el = np.zeros(len(rx_ids), dtype=np.int64)

        out = []
        for az, el in zip(tx_az, tx_el):
            out.extend(zip(rx_az + az, rx_el + el))
        return np.asarray(out, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RT-Pose raw radar/bin echo directly to xyz-Doppler npy tensors."
    )
    parser.add_argument("--dataset_dir", required=True, help="Path containing RT-Pose sequences/")
    parser.add_argument("--sequence", type=int, nargs="*", help="Sequence IDs to process; default: all")
    parser.add_argument("--frames", type=int, nargs="*", help="Matlab one-based frame IDs to process")
    parser.add_argument("--workers", type=int, default=1, help="Parallel frame workers. Each worker may use >1GB RAM.")
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Maximum submitted frame jobs waiting/running at once; 0 uses 2*workers.",
    )
    parser.add_argument("--x-chunk", type=int, default=256, help="Number of X bins interpolated per chunk")
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy", "torch"), default="auto", help="Array backend")
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
    parser.add_argument("--save-drae", action="store_true", help="Also save intermediate D-R-A-E .mat tensors")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing outputs")
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


def run_frame_pool(
    jobs: list[tuple],
    workers: int,
    mp_context,
    max_in_flight: int,
) -> None:
    in_flight_limit = max_in_flight if max_in_flight > 0 else max(1, workers * 2)
    job_iter = iter(jobs)
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as pool:
        pending: set[futures.Future] = set()
        for _ in range(min(in_flight_limit, len(jobs))):
            pending.add(pool.submit(process_one_frame, *next(job_iter)))
        progress = tqdm(total=len(jobs), desc="frames")
        try:
            while pending:
                done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                for fut in done:
                    fut.result()
                    progress.update(1)
                    try:
                        pending.add(pool.submit(process_one_frame, *next(job_iter)))
                    except StopIteration:
                        pass
        finally:
            progress.close()


def read_valid_num_frames(idx_file: Path) -> int:
    header = np.fromfile(idx_file, dtype="<u4", count=6)
    if header.size < 4:
        raise ValueError(f"Invalid idx file: {idx_file}")
    return int(header[3])


def unique_file_indices(bin_dir: Path) -> list[str]:
    ids = set()
    for path in bin_dir.glob("*_data.bin"):
        match = re.search(r"_(\d+)_data\.bin$", path.name)
        if match:
            ids.add(int(match.group(1)))
    return [f"{idx:04d}" for idx in sorted(ids)]


def files_for_index(bin_dir: Path, file_idx: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in DEVICE_NAMES:
        files[name] = bin_dir / f"{name}_{file_idx}_data.bin"
        files[f"{name}_idx"] = bin_dir / f"{name}_{file_idx}_idx.bin"
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing radar files: " + ", ".join(missing))
    return files


def load_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(calib_path, squeeze_me=True, struct_as_record=False)
    calib = mat["calibResult"]
    return np.asarray(calib.RangeMat), np.asarray(calib.PeakValMat)


def read_device_frame(path: Path, frame_idx: int, cfg: RadarConfig) -> np.ndarray:
    samples_per_frame = (
        cfg.num_adc_sample
        * cfg.num_chirps_in_loop
        * cfg.nchirp_loops
        * cfg.num_rx_per_device
        * 2
    )
    raw = np.fromfile(
        path,
        dtype="<u2",
        count=samples_per_frame,
        offset=(frame_idx - 1) * samples_per_frame * np.dtype("<u2").itemsize,
    )
    if raw.size != samples_per_frame:
        raise ValueError(f"{path} ended early while reading frame {frame_idx}")

    signed = raw.astype(np.int16, copy=False)
    complex_iq = signed[0::2].astype(np.float32) + 1j * signed[1::2].astype(np.float32)
    data = complex_iq.reshape(
        (cfg.num_rx_per_device, cfg.num_adc_sample, cfg.num_chirps_in_loop, cfg.nchirp_loops),
        order="F",
    )
    return data.transpose(1, 3, 0, 2)


def read_adc_frame(files: dict[str, Path], frame_idx: int, cfg: RadarConfig) -> np.ndarray:
    adc = np.empty(
        (
            cfg.num_adc_sample,
            cfg.nchirp_loops,
            cfg.num_rx_per_device * cfg.num_devices,
            cfg.num_chirps_in_loop,
        ),
        dtype=np.complex64,
    )
    for device_idx, name in enumerate(DEVICE_NAMES):
        start = device_idx * cfg.num_rx_per_device
        adc[:, :, start : start + cfg.num_rx_per_device, :] = read_device_frame(files[name], frame_idx, cfg)
    return adc


def calibrate_adc(
    adc,
    range_mat,
    peak_val_mat,
    cfg: RadarConfig,
    use_rangemat_correction: bool,
    use_peakvalmat_correction: bool,
    xp=np,
):
    if use_rangemat_correction or use_peakvalmat_correction:
        out = xp.empty_like(adc)
        tx_ref = cfg.tx_to_enable[0] - 1
        sample_idx = None
        if use_rangemat_correction:
            sample_idx = xp.arange(cfg.num_adc_sample, dtype=xp.float32)[:, None, None]
        for tx_axis, tx_id in enumerate(cfg.tx_to_enable):
            tx = tx_id - 1
            corrected = adc[:, :, :, tx_axis]
            if use_rangemat_correction:
                freq_calib = (
                    (range_mat[tx, :] - range_mat[tx_ref, 0])
                    * cfg.fs_calib
                    / cfg.adc_sample_rate
                    * cfg.chirp_slope
                    / cfg.slope_calib
                )
                freq_calib = 2.0 * xp.pi * freq_calib / (cfg.num_adc_sample * cfg.calibration_interp)
                # Matlab uses a complex-conjugating transpose in:
                # correction_vec = (exp(1i*((0:N-1)'*freq_calib)))'
                freq_corr = xp.exp(-1j * sample_idx * freq_calib[None, None, :])
                corrected = corrected * freq_corr

            if use_peakvalmat_correction:
                phase_calib = peak_val_mat[tx_ref, 0] / peak_val_mat[tx, :]
                phase_calib = phase_calib / xp.abs(phase_calib)
                corrected = corrected * phase_calib[None, None, :]

            out[:, :, :, tx_axis] = corrected
    else:
        out = adc

    rx_order = xp.asarray(np.array(cfg.rx_for_mimo_process, dtype=np.int64) - 1)
    return out[:, :, rx_order, :]


def range_fft(adc_one_tx, cfg: RadarConfig, xp=np):
    input_mat = adc_one_tx - adc_one_tx.mean(axis=0, keepdims=True)
    n = np.arange(1, cfg.num_adc_sample + 1, dtype=np.float32) / (cfg.num_adc_sample + 1)
    window = xp.asarray((0.5 - 0.5 * np.cos(2.0 * np.pi * n)).astype(np.float32))
    input_mat = input_mat * window[:, None, None]
    return xp.fft.fft(input_mat, n=cfg.range_fft_size, axis=0)


def doppler_fft(range_data, cfg: RadarConfig, xp=np):
    range_data = range_data - range_data.mean(axis=1, keepdims=True)
    dop = xp.fft.fft(range_data, n=cfg.doppler_fft_size, axis=1)
    return xp.fft.fftshift(dop, axes=1)


def raw_frame_to_drae(
    files: dict[str, Path],
    frame_idx: int,
    range_mat: np.ndarray,
    peak_val_mat: np.ndarray,
    use_rangemat_correction: bool,
    use_peakvalmat_correction: bool,
    xp=np,
    cfg: RadarConfig | None = None,
):
    cfg = cfg or RadarConfig()
    adc = xp.asarray(read_adc_frame(files, frame_idx, cfg))
    range_mat_xp = xp.asarray(range_mat.astype(np.float32, copy=False)) if use_rangemat_correction else None
    peak_val_mat_xp = xp.asarray(peak_val_mat.astype(np.complex64, copy=False)) if use_peakvalmat_correction else None
    adc = calibrate_adc(
        adc,
        range_mat_xp,
        peak_val_mat_xp,
        cfg,
        use_rangemat_correction,
        use_peakvalmat_correction,
        xp=xp,
    )

    doppler_by_tx = xp.empty(
        (cfg.range_fft_size, cfg.doppler_fft_size, len(cfg.rx_for_mimo_process), len(cfg.tx_to_enable)),
        dtype=xp.complex64,
    )
    for tx_axis in range(len(cfg.tx_to_enable)):
        doppler_by_tx[:, :, :, tx_axis] = doppler_fft(
            range_fft(adc[:, :, :, tx_axis], cfg, xp=xp),
            cfg,
            xp=xp,
        )

    # Match Matlab reshape: range x Doppler x (rx fastest, tx slowest).
    radar_pre_3dfft = transpose_backend(doppler_by_tx, (0, 1, 3, 2)).reshape(
        cfg.range_fft_size, cfg.doppler_fft_size, -1
    )
    return angle_fft_4d(radar_pre_3dfft, cfg, xp=xp)


def angle_fft_4d(radar_pre_3dfft, cfg: RadarConfig, xp=np):
    virt = cfg.virtual_array
    aperture_az = int(virt[:, 0].max()) + 1
    aperture_el = int(virt[:, 1].max()) + 1
    sig = xp.zeros((cfg.range_fft_size, cfg.doppler_fft_size, aperture_az, aperture_el), dtype=xp.complex64)

    used = np.zeros((aperture_az, aperture_el), dtype=bool)
    for ant_idx, (az, el) in enumerate(virt):
        if not used[az, el]:
            sig[:, :, az, el] = radar_pre_3dfft[:, :, ant_idx]
            used[az, el] = True

    sig = transpose_backend(sig, (1, 0, 2, 3))
    az_fft = xp.fft.fftshift(xp.fft.fft(sig, n=cfg.angle_fft_size, axis=2), axes=2)
    drae = xp.fft.fftshift(xp.fft.fft(az_fft, n=cfg.elevation_fft_size, axis=3), axes=3)
    return astype_backend(drae, xp.complex64, copy=False)


def xyz_axes(cfg: RadarConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr_x = np.arange(0.0, 11.6, 11.6 / 256)
    arr_y = np.arange(-10.05, 10.05, 20.1 / 128)
    arr_z = np.arange(-5.8, 5.8, 11.6 / 32)
    if (len(arr_x), len(arr_y), len(arr_z)) != (256, 128, 32):
        raise AssertionError("Unexpected xyz axis lengths")
    return arr_x, arr_y, arr_z


def interpolation_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deg = np.pi / 180.0
    arr_range = np.arange(0.0, 11.6, 11.6 / 256)
    arr_elevation = np.arange(0.0, 60.0, 60.0 / 32) * deg
    arr_azimuth = np.arange(0.0, 120.0, 120.0 / 128) * deg
    return arr_range, arr_elevation, arr_azimuth


def original_interp_indices(values, axis, xp=np):
    # Reproduce findIndexForBiInt() from 4Dradar2xyz.py, including its one-bin
    # offset relative to ordinary bracketing.
    upper = xp.searchsorted(axis, values, side="right") - 1
    lower = upper - 1
    return lower, upper


def drae_to_dzyx(drae, output_path: Path, x_chunk: int, xp=np) -> None:
    cfg = RadarConfig()
    arr_x_cpu, arr_y_cpu, arr_z_cpu = xyz_axes(cfg)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(cfg.doppler_fft_size, len(arr_z_cpu), len(arr_y_cpu), len(arr_x_cpu)),
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
        # The original findIndexForBiInt() never returns the last grid index as
        # the upper interpolation bound. Values at or beyond the last grid
        # point are treated as invalid and left at the initialized value 1+0j.
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

            def lin_index(ri: np.ndarray, ei: np.ndarray, ai: np.ndarray) -> np.ndarray:
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
        # Match 4Dradar2xyz.py: arrDZYX = np.flip(arrDZYX, axis=2) before saving.
        chunk = flip_backend(chunk, axis=2)
        out[:, :, :, x0:x1] = as_numpy(chunk)
        out.flush()


def sequence_ids(dataset_dir: Path, selected: Iterable[int] | None) -> list[int]:
    if selected is not None:
        return [int(x) for x in selected]
    return sorted(int(path.name) for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit())


def output_name(frame_idx: int) -> str:
    return f"{frame_idx:06d}.npy"


def process_one_frame(
    sequence_dir: Path,
    file_idx: str,
    frame_idx: int,
    calib_path: Path,
    save_drae: bool,
    overwrite: bool,
    x_chunk: int,
    backend: str,
    gpu_device: int | None,
    use_rangemat_correction: bool,
    use_peakvalmat_correction: bool,
) -> Path:
    cfg = RadarConfig()
    xp, _ = cached_backend(backend, gpu_device)
    out_path = sequence_dir / "radar" / "npy_DZYX_complex" / output_name(frame_idx)
    if out_path.exists() and not overwrite:
        return out_path

    range_mat, peak_val_mat = cached_calibration(calib_path, use_rangemat_correction, use_peakvalmat_correction)
    files = cached_files(sequence_dir, file_idx)
    drae = raw_frame_to_drae(
        files,
        frame_idx,
        range_mat,
        peak_val_mat,
        use_rangemat_correction,
        use_peakvalmat_correction,
        xp=xp,
    )

    if save_drae:
        mat_dir = sequence_dir / "radar" / "mat"
        mat_dir.mkdir(parents=True, exist_ok=True)
        savemat(mat_dir / f"4dTensor-Frame{frame_idx}.mat", {"matr4": as_numpy(drae)})

    drae_to_dzyx(drae, out_path, x_chunk=x_chunk, xp=xp)
    return out_path


def frame_jobs(dataset_dir: Path, args: argparse.Namespace, calib_path: Path) -> list[tuple]:
    jobs = []
    for seq in sequence_ids(dataset_dir, args.sequence):
        if seq in BROKEN_SEQUENCES:
            print(f"Skip damaged sequence {seq}")
            continue

        sequence_dir = dataset_dir / str(seq)
        bin_dir = sequence_dir / "radar" / "bin"
        if not bin_dir.exists():
            print(f"Skip sequence {seq}: missing {bin_dir}")
            continue

        for file_idx in unique_file_indices(bin_dir):
            files = files_for_index(bin_dir, file_idx)
            valid_frames = read_valid_num_frames(files["master_idx"])
            frames = args.frames if args.frames else range(2, valid_frames + 1)
            for frame_idx in frames:
                if frame_idx < 2 or frame_idx > valid_frames:
                    print(f"Skip sequence {seq} frame {frame_idx}: valid range is 2..{valid_frames}")
                    continue
                jobs.append(
                    (
                        sequence_dir,
                        file_idx,
                        int(frame_idx),
                        calib_path,
                        args.save_drae,
                        args.overwrite,
                        args.x_chunk,
                        args.backend,
                        args.gpu_device,
                        args.rangemat_correction == "on",
                        args.peakvalmat_correction == "on",
                    )
                )
    return jobs


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        calib_path = (Path.cwd() / calib_path).resolve()

    jobs = frame_jobs(dataset_dir, args, calib_path)
    if not jobs:
        print("No frames to process.")
        return

    _, is_gpu = resolve_backend(args.backend, gpu_device=args.gpu_device)
    if args.backend == "auto":
        print(f"Using {'cupy/GPU' if is_gpu else 'numpy/CPU'} backend")

    if args.workers == 1:
        for job in tqdm(jobs, desc="frames"):
            process_one_frame(*job)
        return

    mp_context = mp.get_context("spawn") if is_gpu else None
    run_frame_pool(jobs, args.workers, mp_context, args.max_in_flight)


if __name__ == "__main__":
    main()
