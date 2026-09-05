"""
Slang CUDA solver backend — high-throughput time-domain beat signal computation.

Two parallelization strategies for chirp:
- chirp_slang (default): parallel over time samples AND target chunks
- chirp_slang_per_target: each thread = one target, loops time samples (atomicAdd)

Plus frameCuda for full MIMO frames.
"""

import os
import math
import torch
import slangtorch

from . import Solver
from .common import (
    collect_interpolated_samples,
    compute_antenna_pattern_gains,
    compute_path_amplitudes,
    compute_polarization_amplitudes,
    compute_total_path_lengths,
    ensure_current_env_on_path,
    normalize_interpolated_sample,
    pytorch_chirp_reference,
    pytorch_mimo_from_samples,
    samples_require_grad,
)

# .slang files live alongside solver modules
_SOLVERS_DIR = os.path.dirname(__file__)


def init():
    """Load the radar.slang module for the solver instance."""
    ensure_current_env_on_path()
    slang_path = os.path.join(_SOLVERS_DIR, 'radar.slang')
    return slangtorch.loadModule(slang_path)


def _env_enabled(name, default="0"):
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no"}


def _frame_t_sample(radar, *, device, dtype):
    cache = getattr(radar, "_slang_frame_t_sample_cache", None)
    if cache is None:
        cache = {}
        setattr(radar, "_slang_frame_t_sample_cache", cache)
    key = (str(torch.device(device)), dtype)
    cached = cache.get(key)
    if cached is None:
        cfg = radar.config
        cached = (
            torch.arange(0, cfg.adc_samples, dtype=dtype, device=device) / (cfg.sample_rate * 1e3)
            + cfg.adc_start_time * 1e-6
        ).contiguous()
        cache[key] = cached
    return cached


def _compute_total_lengths_and_amplitudes(radar, sample, tx_pos, rx_pos):
    dist_tx = torch.cdist(sample.entry_points, tx_pos).transpose(0, 1).unsqueeze(1)
    dist_rx = torch.cdist(sample.points, rx_pos).transpose(0, 1).unsqueeze(0)
    total_lengths = dist_tx + sample.fixed_path_lengths.view(1, 1, -1) + dist_rx

    fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(dist_tx * dist_rx, min=1e-6))
    scatter_power = torch.clamp(sample.intensities, min=0.0).view(1, 1, -1)
    pattern_gains = compute_antenna_pattern_gains(radar, sample, tx_pos, rx_pos)
    if pattern_gains is not None:
        scatter_power = scatter_power * torch.clamp(pattern_gains, min=0.0)
    amplitudes = radar.gain * torch.sqrt(scatter_power) * fspl_amp
    polarization_factor = compute_polarization_amplitudes(radar, sample)
    if polarization_factor is not None:
        amplitudes = amplitudes * polarization_factor

    return total_lengths, amplitudes


def _batched_total_lengths_and_amplitudes(radar, samples, tx_pos, rx_pos):
    points = getattr(samples, "points_batched", None)
    intensities = getattr(samples, "intensities_batched", None)
    entry_points = getattr(samples, "entry_points_batched", points)
    fixed_path_lengths = getattr(samples, "fixed_path_lengths_batched", None)
    if points is None or intensities is None or entry_points is None:
        return None
    if radar.polarization is not None:
        return None

    F, N, _ = points.shape
    TX = tx_pos.shape[0]
    RX = rx_pos.shape[0]
    if N == 0:
        empty_lengths = torch.empty((F, TX, RX, 0), dtype=torch.float32, device=points.device)
        empty_amplitudes = torch.empty_like(empty_lengths)
        return empty_lengths, empty_amplitudes

    tx_batch = tx_pos.to(dtype=torch.float32).view(1, TX, 3).expand(F, TX, 3)
    rx_batch = rx_pos.to(dtype=torch.float32).view(1, RX, 3).expand(F, RX, 3)
    dist_tx = torch.cdist(entry_points, tx_batch).permute(0, 2, 1).unsqueeze(2)
    dist_rx = torch.cdist(points, rx_batch).permute(0, 2, 1).unsqueeze(1)
    if fixed_path_lengths is None:
        fixed = 0.0
    else:
        fixed = fixed_path_lengths.view(F, 1, 1, N)
    total_lengths = dist_tx + fixed + dist_rx

    fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(dist_tx * dist_rx, min=1e-6))
    scatter_power = torch.clamp(intensities, min=0.0).view(F, 1, 1, N)
    pattern_gains = None
    if radar.antenna_pattern_config is not None:
        tx_vectors = radar.local_from_world_vectors(entry_points.unsqueeze(1) - tx_pos.view(1, TX, 1, 3))
        rx_vectors = radar.local_from_world_vectors(points.unsqueeze(1) - rx_pos.view(1, RX, 1, 3))
        tx_gains = radar.evaluate_antenna_pattern_vectors(tx_vectors).unsqueeze(2)
        rx_gains = radar.evaluate_antenna_pattern_vectors(rx_vectors).unsqueeze(1)
        pattern_gains = tx_gains * rx_gains
    if pattern_gains is not None:
        scatter_power = scatter_power * torch.clamp(pattern_gains, min=0.0)
    amplitudes = radar.gain * torch.sqrt(scatter_power) * fspl_amp
    return total_lengths.contiguous(), amplitudes.contiguous()


# ------------------------------------------------------------------
# Chirp kernels
# ------------------------------------------------------------------

def chirp_slang(solver, distances, amplitudes, targets_per_chunk=256):
    """Chunked chirp: parallel over time samples AND target chunks (default)."""
    radar = solver.radar
    cfg = radar.config
    T = cfg.adc_samples
    num_targets = distances.shape[0]

    distances = distances.to(dtype=torch.float32, device=solver.device).contiguous()
    amplitudes = amplitudes.to(dtype=torch.float32, device=solver.device).contiguous()

    num_chunks = (num_targets + targets_per_chunk - 1) // targets_per_chunk

    out_real = torch.zeros((num_chunks, T), dtype=torch.float64, device=solver.device)
    out_imag = torch.zeros((num_chunks, T), dtype=torch.float64, device=solver.device)

    block_size = 256
    grid_x = (T + block_size - 1) // block_size

    solver._module.chirp_kernel_chunked(
        t_sample=radar.t_sample,
        distances=distances, amplitudes=amplitudes,
        fc=cfg.fc, slope=cfg.slope * 1e12,
        num_targets=num_targets, targets_per_chunk=targets_per_chunk,
        out_real=out_real, out_imag=out_imag,
    ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_x, num_chunks, 1))

    return out_real.sum(dim=0) + 1j * out_imag.sum(dim=0)


def chirp_slang_per_target(solver, distances, amplitudes, targets_per_chunk=256):
    """Per-target chirp: each thread = one target, loops all time samples (atomicAdd)."""
    radar = solver.radar
    cfg = radar.config
    T = cfg.adc_samples
    num_targets = distances.shape[0]

    distances = distances.to(dtype=torch.float32, device=solver.device).contiguous()
    amplitudes = amplitudes.to(dtype=torch.float32, device=solver.device).contiguous()

    num_chunks = (num_targets + targets_per_chunk - 1) // targets_per_chunk

    out_real = torch.zeros((num_chunks, T), dtype=torch.float32, device=solver.device)
    out_imag = torch.zeros((num_chunks, T), dtype=torch.float32, device=solver.device)

    solver._module.chirp_kernel_per_target(
        t_sample=radar.t_sample,
        distances=distances, amplitudes=amplitudes,
        fc=cfg.fc, slope=cfg.slope * 1e12,
        num_targets=num_targets, T=T, targets_per_chunk=targets_per_chunk,
        out_real=out_real, out_imag=out_imag,
    ).launchRaw(blockSize=(targets_per_chunk, 1, 1), gridSize=(num_chunks, 1, 1))

    return out_real.sum(dim=0).to(torch.float64) + 1j * out_imag.sum(dim=0).to(torch.float64)


# ------------------------------------------------------------------
# MIMO frame generation
# ------------------------------------------------------------------

def frameCuda(solver, samples):
    """Generate a full MIMO frame using Slang CUDA kernels from sampled chirps."""
    radar = solver.radar
    cfg = radar.config
    TX, RX, F, T = cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples
    use_fused = _env_enabled("WITWIN_SLANG_FRAME_FUSED")
    use_phase_coeff = _env_enabled("WITWIN_SLANG_FRAME_PHASE_COEFF")
    use_float32 = _env_enabled("WITWIN_SLANG_FRAME_FLOAT32")
    use_chunked = _env_enabled("WITWIN_SLANG_FRAME_CHUNKED")

    tx_pos = radar.tx_pos
    rx_pos = radar.rx_pos

    use_batched_prep = _env_enabled("WITWIN_SLANG_BATCH_PREP", default="1")
    if use_float32 and not (use_fused or use_phase_coeff or use_chunked) and use_batched_prep:
        batched = _batched_total_lengths_and_amplitudes(radar, samples, tx_pos, rx_pos)
        if batched is not None:
            total_lengths_tensor, amplitudes_tensor = batched
            max_points = int(total_lengths_tensor.shape[-1])
            if max_points == 0:
                return torch.zeros((TX, RX, F, T), dtype=torch.complex64, device=solver.device)
            block_size = 256
            grid_t = (T + block_size - 1) // block_size
            frame_real32 = torch.empty((TX, RX, F, T), dtype=torch.float32, device=solver.device)
            frame_imag32 = torch.empty_like(frame_real32)
            solver._module.frameMIMO_kernel_f32(
                t_sample=_frame_t_sample(radar, device=solver.device, dtype=torch.float32),
                total_lengths=total_lengths_tensor,
                amplitudes=amplitudes_tensor,
                fc=cfg.fc, slope=cfg.slope * 1e6 * 1e6,
                max_N=max_points,
                out_real=frame_real32,
                out_imag=frame_imag32,
            ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_t, TX * RX, F))
            return frame_real32 + 1j * frame_imag32

    max_points = 0
    all_total_lengths = []
    all_amplitudes = []
    for sample in samples:
        N = sample.points.shape[0]
        if N > max_points:
            max_points = N
        if N == 0:
            all_total_lengths.append(None)
            all_amplitudes.append(None)
            continue
        total_lengths, amplitudes = _compute_total_lengths_and_amplitudes(radar, sample, tx_pos, rx_pos)
        all_total_lengths.append(total_lengths)
        all_amplitudes.append(amplitudes)

    if max_points == 0:
        dtype = torch.complex64 if use_float32 or use_phase_coeff or use_fused else torch.complex128
        return torch.zeros((TX, RX, F, T), dtype=dtype, device=solver.device)

    if use_fused and radar.polarization is None and radar.antenna_pattern_kind == "separable":
        simple_paths = all(
            sample.entry_points.shape == sample.points.shape
            and torch.equal(sample.entry_points, sample.points)
            and torch.count_nonzero(sample.fixed_path_lengths).item() == 0
            for sample in samples
        )
        if simple_paths:
            points_tensor = torch.zeros((F, max_points, 3), dtype=torch.float32, device=solver.device)
            intensities_tensor = torch.zeros((F, max_points), dtype=torch.float32, device=solver.device)
            for i, sample in enumerate(samples):
                N = sample.points.shape[0]
                if N == 0:
                    continue
                points_tensor[i, :N] = sample.points.to(dtype=torch.float32, device=solver.device)
                intensities_tensor[i, :N] = sample.intensities.to(dtype=torch.float32, device=solver.device)
            _, world_from_local = radar._world_from_local_matrix(device=solver.device, dtype=torch.float32)
            frame_real32 = torch.empty((TX, RX, F, T), dtype=torch.float32, device=solver.device)
            frame_imag32 = torch.empty_like(frame_real32)
            solver._module.frameMIMO_fused_f32_kernel(
                t_sample=_frame_t_sample(radar, device=solver.device, dtype=torch.float32),
                points=points_tensor,
                intensities=intensities_tensor,
                tx_pos=radar.tx_pos.to(dtype=torch.float32),
                rx_pos=radar.rx_pos.to(dtype=torch.float32),
                world_from_local=world_from_local.contiguous(),
                antenna_values=radar.antenna_pattern_x_values.to(dtype=torch.float32),
                fc=cfg.fc, slope=cfg.slope * 1e6 * 1e6,
                wavelength=radar._lambda,
                gain=radar.gain,
                max_N=max_points,
                out_real=frame_real32,
                out_imag=frame_imag32,
            ).launchRaw(blockSize=(256, 1, 1), gridSize=((T + 255) // 256, TX * RX, F))
            return frame_real32 + 1j * frame_imag32

    total_lengths_tensor = torch.empty((F, TX, RX, max_points), dtype=torch.float32, device=solver.device)
    amplitudes_tensor = torch.zeros((F, TX, RX, max_points), dtype=torch.float32, device=solver.device)
    for i in range(F):
        if all_total_lengths[i] is None:
            continue
        N = all_total_lengths[i].shape[-1]
        total_lengths_tensor[i, :, :, :N] = all_total_lengths[i]
        amplitudes_tensor[i, :, :, :N] = all_amplitudes[i]

    block_size = 256
    grid_t = (T + block_size - 1) // block_size
    if use_phase_coeff:
        slope_hz = cfg.slope * 1e6 * 1e6
        tof = total_lengths_tensor / float(radar.c0)
        phase_const = (2.0 * torch.pi * (cfg.fc * tof - 0.5 * slope_hz * tof * tof)).to(dtype=torch.float32)
        phase_rate = (2.0 * torch.pi * slope_hz * tof).to(dtype=torch.float32)
        frame_real32 = torch.empty((TX, RX, F, T), dtype=torch.float32, device=solver.device)
        frame_imag32 = torch.empty_like(frame_real32)
        solver._module.frameMIMO_phase_coeff_f32_kernel(
            t_sample=_frame_t_sample(radar, device=solver.device, dtype=torch.float32),
            phase_const=phase_const,
            phase_rate=phase_rate,
            amplitudes=amplitudes_tensor,
            max_N=max_points,
            out_real=frame_real32,
            out_imag=frame_imag32,
        ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_t, TX * RX, F))
        return frame_real32 + 1j * frame_imag32

    if use_float32:
        frame_real32 = torch.empty((TX, RX, F, T), dtype=torch.float32, device=solver.device)
        frame_imag32 = torch.empty_like(frame_real32)
        solver._module.frameMIMO_kernel_f32(
            t_sample=_frame_t_sample(radar, device=solver.device, dtype=torch.float32),
            total_lengths=total_lengths_tensor,
            amplitudes=amplitudes_tensor,
            fc=cfg.fc, slope=cfg.slope * 1e6 * 1e6,
            max_N=max_points,
            out_real=frame_real32,
            out_imag=frame_imag32,
        ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_t, TX * RX, F))
        return frame_real32 + 1j * frame_imag32

    if use_chunked:
        targets_per_chunk = int(os.environ.get("WITWIN_SLANG_FRAME_TARGETS_PER_CHUNK", "256"))
        targets_per_chunk = max(targets_per_chunk, 1)
        num_chunks = (max_points + targets_per_chunk - 1) // targets_per_chunk
        partial_real = torch.zeros((num_chunks, TX, RX, F, T), dtype=torch.float64, device=solver.device)
        partial_imag = torch.zeros_like(partial_real)

        solver._module.frameMIMO_chunked_kernel(
            t_sample=_frame_t_sample(radar, device=solver.device, dtype=torch.float64),
            total_lengths=total_lengths_tensor,
            amplitudes=amplitudes_tensor,
            fc=cfg.fc, slope=cfg.slope * 1e6 * 1e6,
            max_N=max_points,
            targets_per_chunk=targets_per_chunk,
            num_chunks=num_chunks,
            partial_real=partial_real,
            partial_imag=partial_imag,
        ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_t, TX * RX, F * num_chunks))

        return partial_real.sum(dim=0) + 1j * partial_imag.sum(dim=0)

    frame_real = torch.empty((TX, RX, F, T), dtype=torch.float64, device=solver.device)
    frame_imag = torch.empty_like(frame_real)
    solver._module.frameMIMO_kernel(
        t_sample=_frame_t_sample(radar, device=solver.device, dtype=torch.float64),
        total_lengths=total_lengths_tensor,
        amplitudes=amplitudes_tensor,
        fc=cfg.fc, slope=cfg.slope * 1e6 * 1e6,
        phi=0.0, max_N=max_points,
        out_real=frame_real,
        out_imag=frame_imag,
    ).launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_t, TX * RX, F))

    return frame_real + 1j * frame_imag


# ------------------------------------------------------------------
# Solver class
# ------------------------------------------------------------------

class SlangSolver(Solver):
    """Slang CUDA kernel solver — high-throughput time-domain computation."""

    def __init__(self, radar):
        super().__init__(radar)
        self._module = init()

    def chirp(self, distances, amplitudes):
        d_rt = distances * 2  # one-way -> round-trip
        signal = chirp_slang(self, d_rt, amplitudes)
        if distances.requires_grad or amplitudes.requires_grad:
            reference = pytorch_chirp_reference(self.radar, distances, amplitudes)
            signal = signal.to(reference.dtype)
            return signal.detach() + (reference - reference.detach())
        return signal

    def frame(self, interpolator, t0=0):
        r = self.radar
        cfg = r.config
        T_chirp = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        tx0 = r.tx_pos[0:1].contiguous()
        rx0 = r.rx_pos[0:1].contiguous()

        result = []
        for chirp_id in range(cfg.chirp_per_frame):
            time_in_frame = chirp_id * T_chirp * cfg.num_tx
            sample = normalize_interpolated_sample(interpolator(t0 + time_in_frame), device=r.device)
            total_lengths = compute_total_path_lengths(sample, tx0, rx0)
            one_way = total_lengths.squeeze(0).squeeze(0) * 0.5
            amp = compute_path_amplitudes(r, sample, total_lengths, tx_pos=tx0, rx_pos=rx0).squeeze(0).squeeze(0)
            result.append(self.chirp(one_way, amp))

        return torch.stack(result)

    def mimo(self, interpolator, t0=0, **options):
        self._ensure_no_options(options)
        samples = collect_interpolated_samples(self.radar, interpolator, t0)
        signal = frameCuda(self, samples)
        if samples_require_grad(samples):
            reference = pytorch_mimo_from_samples(self.radar, samples)
            signal = signal.to(reference.dtype)
            return signal.detach() + (reference - reference.detach())
        return signal
