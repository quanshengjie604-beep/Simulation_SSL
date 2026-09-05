"""Shared solver helpers used by all radar backends."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import math
import torch


def ensure_current_env_on_path() -> None:
    """Prepend the active interpreter environment to PATH for subprocess tools."""
    env_root = os.path.dirname(sys.executable)
    candidates = [
        env_root,
        os.path.join(env_root, "Scripts"),
        os.path.join(env_root, "Library", "bin"),
    ]
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    prepend = [path for path in candidates if os.path.isdir(path) and path not in path_entries]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + path_entries)


@dataclass(frozen=True)
class PathSample:
    intensities: torch.Tensor
    points: torch.Tensor
    entry_points: torch.Tensor
    fixed_path_lengths: torch.Tensor
    depths: torch.Tensor
    normals: torch.Tensor | None


def normalize_interpolated_sample(sample, *, device: str | torch.device) -> PathSample:
    """Promote a TraceResult-like sample (or legacy ``(intensities, points)`` tuple) onto ``device``."""
    if isinstance(sample, tuple):
        intensities, points = sample
        points = points.to(dtype=torch.float32, device=device)
        intensities = intensities.to(dtype=torch.float32, device=device)
        return PathSample(
            intensities=intensities,
            points=points,
            entry_points=points,
            fixed_path_lengths=torch.zeros(points.shape[0], dtype=torch.float32, device=device),
            depths=torch.zeros(points.shape[0], dtype=torch.int32, device=device),
            normals=None,
        )

    points = sample.points.to(dtype=torch.float32, device=device)
    intensities = sample.intensities.to(dtype=torch.float32, device=device)
    entry_points = sample.entry_points.to(dtype=torch.float32, device=device)
    fixed_path_lengths = sample.fixed_path_lengths.to(dtype=torch.float32, device=device)
    depths = sample.depths.to(dtype=torch.int32, device=device)
    normals = sample.normals.to(dtype=torch.float32, device=device) if sample.normals is not None else None

    return PathSample(
        intensities=intensities,
        points=points,
        entry_points=entry_points,
        fixed_path_lengths=fixed_path_lengths,
        depths=depths,
        normals=normals,
    )


def collect_interpolated_samples(radar, interpolator, t0=0.0):
    """Evaluate the scene interpolator once per chirp and keep tensors on-graph."""
    batched = getattr(interpolator, "collect_chirp_samples", None)
    if batched is not None:
        return batched(radar, t0)

    cfg = radar.config
    chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
    samples = []
    for chirp_id in range(cfg.chirp_per_frame):
        time_in_frame = chirp_id * chirp_period * cfg.num_tx
        sample = interpolator(t0 + time_in_frame)
        samples.append(normalize_interpolated_sample(sample, device=radar.device))
    return samples


def samples_require_grad(samples) -> bool:
    return any(
        sample.intensities.requires_grad
        or sample.points.requires_grad
        or sample.entry_points.requires_grad
        or sample.fixed_path_lengths.requires_grad
        or (sample.normals is not None and sample.normals.requires_grad)
        for sample in samples
    )


def compute_total_path_lengths(sample: PathSample, tx_pos: torch.Tensor, rx_pos: torch.Tensor) -> torch.Tensor:
    """Return total path lengths with shape (TX, RX, N)."""
    dist_tx = torch.cdist(sample.entry_points, tx_pos).transpose(0, 1).unsqueeze(1)
    dist_rx = torch.cdist(sample.points, rx_pos).transpose(0, 1).unsqueeze(0)
    return dist_tx + sample.fixed_path_lengths.view(1, 1, -1) + dist_rx


def compute_antenna_pattern_gains(
    radar,
    sample: PathSample,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
) -> torch.Tensor | None:
    """Return per-path power gains from the configured TX/RX antenna pattern."""
    tx_vectors = radar.local_from_world_vectors(sample.entry_points.unsqueeze(0) - tx_pos.unsqueeze(1))
    rx_vectors = radar.local_from_world_vectors(sample.points.unsqueeze(0) - rx_pos.unsqueeze(1))
    tx_gains = radar.evaluate_antenna_pattern_vectors(tx_vectors).unsqueeze(1)
    rx_gains = radar.evaluate_antenna_pattern_vectors(rx_vectors).unsqueeze(0)
    return tx_gains * rx_gains


def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)


def compute_polarization_amplitudes(radar, sample: PathSample) -> torch.Tensor | None:
    """Return signed TX/RX polarization projection factors for each path."""
    polarization = radar.polarization
    if polarization is None:
        return None
    if sample.normals is None:
        raise ValueError("Radar polarization requires per-path surface normals in the interpolated sample.")

    normals = _normalize_vectors(sample.normals)
    tx_world = _normalize_vectors(polarization.tx_world.to(device=normals.device, dtype=normals.dtype))
    rx_world = _normalize_vectors(polarization.rx_world.to(device=normals.device, dtype=normals.dtype))

    reflected_tx = tx_world.unsqueeze(1)
    if polarization.reflection_flip:
        reflected_tx = reflected_tx - 2.0 * (reflected_tx * normals.unsqueeze(0)).sum(dim=-1, keepdim=True) * normals.unsqueeze(0)
    reflected_tx = _normalize_vectors(reflected_tx)
    return (reflected_tx.unsqueeze(1) * rx_world.view(1, rx_world.shape[0], 1, 3)).sum(dim=-1)


def compute_path_amplitudes(
    radar,
    sample: PathSample,
    total_path_lengths: torch.Tensor,
    *,
    tx_pos: torch.Tensor | None = None,
    rx_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert power-domain material coefficients to amplitude-domain weights with FSPL."""
    if tx_pos is None:
        tx_pos = radar.tx_pos
    if rx_pos is None:
        rx_pos = radar.rx_pos
    dist_tx = torch.cdist(sample.entry_points, tx_pos).transpose(0, 1).unsqueeze(1)
    dist_rx = torch.cdist(sample.points, rx_pos).transpose(0, 1).unsqueeze(0)
    fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(dist_tx * dist_rx, min=1e-6))
    scatter_power = torch.clamp(sample.intensities, min=0.0).view(1, 1, -1)
    pattern_gains = compute_antenna_pattern_gains(radar, sample, tx_pos, rx_pos)
    if pattern_gains is not None:
        scatter_power = scatter_power * torch.clamp(pattern_gains, min=0.0)
    amplitudes = radar.gain * torch.sqrt(scatter_power) * fspl_amp
    polarization_factor = compute_polarization_amplitudes(radar, sample)
    if polarization_factor is not None:
        amplitudes = amplitudes * polarization_factor
    return amplitudes


def pytorch_chirp_reference(radar, distances, amplitudes):
    d_rt = (distances * 2).unsqueeze(-1)
    toa = d_rt / radar.c0
    rx = radar.waveform(radar.t_sample - toa)
    rx_weighted = rx * amplitudes.unsqueeze(-1)
    rx_combined = rx_weighted.sum(dim=0)
    return radar.tx_waveform * torch.conj(rx_combined)


def pytorch_mimo_from_samples(radar, samples):
    cfg = radar.config
    frame = torch.zeros(
        (cfg.chirp_per_frame, cfg.num_tx, cfg.num_rx, cfg.adc_samples),
        dtype=torch.complex128,
        device=radar.device,
    )
    tx_pos = radar.tx_pos
    rx_pos = radar.rx_pos

    for chirp_id, sample in enumerate(samples):
        distances = compute_total_path_lengths(sample, tx_pos, rx_pos).unsqueeze(-1)
        toa = distances / radar.c0
        rx = radar.waveform(radar.t_sample - toa)
        amplitudes = compute_path_amplitudes(radar, sample, distances.squeeze(-1), tx_pos=tx_pos, rx_pos=rx_pos)
        rx_weighted = rx * amplitudes.view(1, 1, -1, 1)
        rx_combined = torch.sum(rx_weighted, dim=-2)
        frame[chirp_id] = radar.tx_waveform * torch.conj(rx_combined)

    return frame.permute(1, 2, 0, 3)
