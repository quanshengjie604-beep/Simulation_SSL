"""
Dirichlet kernel solver backend.

Computes the range spectrum directly in the frequency domain and keeps all
backend-specific state on the solver instance.
"""

from __future__ import annotations

import os

import slangtorch
import torch

from . import Solver
from .common import (
    collect_interpolated_samples,
    compute_path_amplitudes,
    compute_total_path_lengths,
    ensure_current_env_on_path,
    normalize_interpolated_sample,
    pytorch_chirp_reference,
    pytorch_mimo_from_samples,
    samples_require_grad,
)

_SOLVERS_DIR = os.path.dirname(__file__)


def _load_module():
    ensure_current_env_on_path()
    slang_path = os.path.join(_SOLVERS_DIR, "dirichlet.slang")
    return slangtorch.loadModule(slang_path)


def _to_f32(solver: "DirichletSolver", value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.float32, device=solver.device).contiguous()


def spectrum(
    solver: "DirichletSolver",
    distances,
    amplitudes,
    *,
    targets_per_chunk: int = 256,
    num_bins: int | None = None,
    n_fft: int | None = None,
    k0_per_meter: float | None = None,
):
    """Dirichlet spectrum with optional overrides for MIMO-mode calls."""
    cfg = solver.radar.config
    num_bins = solver.num_bins if num_bins is None else num_bins
    n_fft = solver.N_fft if n_fft is None else n_fft
    k0_per_meter = solver.k0_per_meter if k0_per_meter is None else k0_per_meter

    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)

    num_chunks = (num_targets + targets_per_chunk - 1) // targets_per_chunk
    output_re = torch.zeros((num_chunks, num_bins), dtype=torch.float32, device=solver.device)
    output_im = torch.zeros((num_chunks, num_bins), dtype=torch.float32, device=solver.device)

    solver._module.forward_chunked(
        d=d,
        a=a,
        output_re=output_re,
        output_im=output_im,
        n=solver.n,
        k0_per_meter=k0_per_meter,
        num_bins=num_bins,
        N_fft=n_fft,
        num_targets=num_targets,
        targets_per_chunk=targets_per_chunk,
        fc=cfg.fc,
        slope=cfg.slope * 1e12,
        t_start=cfg.adc_start_time * 1e-6,
    ).launchRaw(
        blockSize=(256, 1, 1),
        gridSize=((num_bins + 255) // 256, num_chunks, 1),
    )

    return torch.complex(output_re.sum(dim=0), output_im.sum(dim=0))


def backward(solver: "DirichletSolver", distances, amplitudes, grad_output_re, grad_output_im):
    """Backward pass with one thread per target."""
    cfg = solver.radar.config
    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)
    g_re = grad_output_re.to(dtype=torch.float32, device=solver.device).contiguous()
    g_im = grad_output_im.to(dtype=torch.float32, device=solver.device).contiguous()

    grad_d = torch.zeros(num_targets, dtype=torch.float32, device=solver.device)
    grad_a = torch.zeros(num_targets, dtype=torch.float32, device=solver.device)

    solver._module.backward(
        d=d,
        a=a,
        grad_output_re=g_re,
        grad_output_im=g_im,
        grad_d=grad_d,
        grad_a=grad_a,
        n=solver.n,
        k0_per_meter=solver.k0_per_meter,
        num_bins=solver.num_bins,
        N_fft=solver.N_fft,
        num_targets=num_targets,
        fc=cfg.fc,
        slope=cfg.slope * 1e12,
        t_start=cfg.adc_start_time * 1e-6,
    ).launchRaw(
        blockSize=(256, 1, 1),
        gridSize=((num_targets + 255) // 256, 1, 1),
    )

    return grad_d, grad_a


def backward_per_bin(
    solver: "DirichletSolver",
    distances,
    amplitudes,
    grad_output_re,
    grad_output_im,
    *,
    bins_per_chunk: int = 256,
):
    """Backward pass with one thread per spectrum bin."""
    cfg = solver.radar.config
    num_targets = distances.shape[0]
    d = _to_f32(solver, distances)
    a = _to_f32(solver, amplitudes)
    g_re = grad_output_re.to(dtype=torch.float32, device=solver.device).contiguous()
    g_im = grad_output_im.to(dtype=torch.float32, device=solver.device).contiguous()

    num_chunks = (solver.num_bins + bins_per_chunk - 1) // bins_per_chunk
    grad_d = torch.zeros((num_chunks, num_targets), dtype=torch.float32, device=solver.device)
    grad_a = torch.zeros((num_chunks, num_targets), dtype=torch.float32, device=solver.device)

    solver._module.backward_per_bin(
        d=d,
        a=a,
        grad_output_re=g_re,
        grad_output_im=g_im,
        grad_d=grad_d,
        grad_a=grad_a,
        n=solver.n,
        k0_per_meter=solver.k0_per_meter,
        num_bins=solver.num_bins,
        N_fft=solver.N_fft,
        num_targets=num_targets,
        bins_per_chunk=bins_per_chunk,
        fc=cfg.fc,
        slope=cfg.slope * 1e12,
        t_start=cfg.adc_start_time * 1e-6,
    ).launchRaw(
        blockSize=(bins_per_chunk, 1, 1),
        gridSize=(num_chunks, 1, 1),
    )

    return grad_d.sum(dim=0), grad_a.sum(dim=0)


class DirichletSolver(Solver):
    """Direct frequency-domain spectrum solver."""

    def __init__(self, radar, pad_factor: int = 16):
        super().__init__(radar)
        self._module = _load_module()

        cfg = radar.config
        fs = cfg.sample_rate * 1e3
        slope_hz = cfg.slope * 1e12

        self.pad_factor = int(pad_factor)
        self.N_fft = cfg.adc_samples * self.pad_factor
        self.num_bins = self.N_fft // 2
        self.n = (cfg.adc_samples - 1) / 2
        self.k0_per_meter = (slope_hz * 2 / radar.c0) * self.N_fft / fs

        self.mimo_N_fft = cfg.adc_samples
        self.mimo_num_bins = cfg.adc_samples
        self.mimo_k0_per_meter = (slope_hz * 2 / radar.c0) * self.mimo_N_fft / fs

    def chirp(self, distances, amplitudes):
        """High-resolution Dirichlet spectrum (pad_factor bins)."""
        signal = spectrum(self, distances, amplitudes)
        if distances.requires_grad or amplitudes.requires_grad:
            reference = pytorch_chirp_reference(self.radar, distances, amplitudes)
            reference = torch.fft.fft(reference, n=self.N_fft)[: self.N_fft // 2]
            signal = signal.to(reference.dtype)
            return signal.detach() + (reference - reference.detach())
        return signal

    def chirp_mimo(self, distances, amplitudes):
        """Dirichlet spectrum at adc_samples resolution for MIMO output."""
        return spectrum(
            self,
            distances,
            amplitudes,
            num_bins=self.mimo_num_bins,
            n_fft=self.mimo_N_fft,
            k0_per_meter=self.mimo_k0_per_meter,
        )

    def frame(self, interpolator, t0=0):
        r = self.radar
        cfg = r.config
        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        tx0 = r.tx_pos[0:1].contiguous()
        rx0 = r.rx_pos[0:1].contiguous()

        result = []
        for chirp_id in range(cfg.chirp_per_frame):
            time_in_frame = chirp_id * chirp_period * cfg.num_tx
            sample = normalize_interpolated_sample(interpolator(t0 + time_in_frame), device=self.device)
            total_lengths = compute_total_path_lengths(sample, tx0, rx0)
            one_way = total_lengths.squeeze(0).squeeze(0) * 0.5
            amp = compute_path_amplitudes(r, sample, total_lengths, tx_pos=tx0, rx_pos=rx0).squeeze(0).squeeze(0)
            result.append(self.chirp(one_way, amp))

        return torch.stack(result)

    def mimo(self, interpolator, t0=0, **options):
        """Generate MIMO frame with batched TX/RX kernel launches."""
        freq_domain = self._pop_bool_option(options, "freq_domain", False)
        self._ensure_no_options(options)

        r = self.radar
        cfg = r.config
        samples = collect_interpolated_samples(r, interpolator, t0)
        tx_pos = r.tx_pos
        rx_pos = r.rx_pos
        num_pairs = cfg.num_tx * cfg.num_rx

        frame = torch.zeros(
            (cfg.num_tx, cfg.num_rx, cfg.chirp_per_frame, cfg.adc_samples),
            dtype=torch.complex64,
            device=self.device,
        )

        for chirp_id, sample in enumerate(samples):
            n_targets = sample.points.shape[0]
            if n_targets == 0:
                continue

            total_lengths = compute_total_path_lengths(sample, tx_pos, rx_pos)
            one_way = total_lengths.reshape(num_pairs, n_targets) * 0.5
            all_d = one_way.reshape(-1).contiguous()
            amp = compute_path_amplitudes(r, sample, total_lengths, tx_pos=tx_pos, rx_pos=rx_pos)
            all_a = amp.reshape(-1).contiguous()

            output_re = torch.zeros((num_pairs, self.mimo_num_bins), dtype=torch.float32, device=self.device)
            output_im = torch.zeros((num_pairs, self.mimo_num_bins), dtype=torch.float32, device=self.device)

            self._module.forward_chunked(
                d=all_d,
                a=all_a,
                output_re=output_re,
                output_im=output_im,
                n=self.n,
                k0_per_meter=self.mimo_k0_per_meter,
                num_bins=self.mimo_num_bins,
                N_fft=self.mimo_N_fft,
                num_targets=num_pairs * n_targets,
                targets_per_chunk=n_targets,
                fc=cfg.fc,
                slope=cfg.slope * 1e12,
                t_start=cfg.adc_start_time * 1e-6,
            ).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=((self.mimo_num_bins + 255) // 256, num_pairs, 1),
            )

            spectra = torch.complex(output_re, output_im)
            if not freq_domain:
                spectra = torch.fft.ifft(spectra, dim=-1)
            frame[:, :, chirp_id, :] = spectra.view(cfg.num_tx, cfg.num_rx, cfg.adc_samples)

        if samples_require_grad(samples):
            reference = pytorch_mimo_from_samples(r, samples)
            if freq_domain:
                reference = torch.fft.fft(reference, dim=-1)
            frame = frame.to(reference.dtype)
            return frame.detach() + (reference - reference.detach())
        return frame

    def backward(self, distances, amplitudes, grad_output_re, grad_output_im):
        return backward(self, distances, amplitudes, grad_output_re, grad_output_im)

    def backward_per_bin(self, distances, amplitudes, grad_output_re, grad_output_im, bins_per_chunk: int = 256):
        return backward_per_bin(
            self,
            distances,
            amplitudes,
            grad_output_re,
            grad_output_im,
            bins_per_chunk=bins_per_chunk,
        )
