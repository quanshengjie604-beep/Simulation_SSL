"""
PyTorch solver backend — pure PyTorch, no CUDA kernels.

Fully differentiable via autograd. Good for debugging and small-scale simulations.
"""

import torch

from . import Solver
from .common import (
    collect_interpolated_samples,
    compute_path_amplitudes,
    compute_total_path_lengths,
    normalize_interpolated_sample,
)


class PytorchSolver(Solver):
    """Pure PyTorch solver — fully differentiable via autograd."""

    def chirp(self, distances, amplitudes):
        r = self.radar
        d_rt = (distances * 2).unsqueeze(-1)  # (N, 1) round-trip
        toa = d_rt / r.c0
        rx = r.waveform(r.t_sample - toa)  # (N, T)
        rx_weighted = rx * amplitudes.unsqueeze(-1)  # (N, T)
        rx_combined = rx_weighted.sum(dim=0)  # (T,)
        return r.tx_waveform * torch.conj(rx_combined)

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
        r = self.radar
        cfg = r.config
        self._ensure_no_options(options)
        frame = torch.zeros(
            (cfg.chirp_per_frame, cfg.num_tx, cfg.num_rx, cfg.adc_samples),
            dtype=torch.complex128,
            device=r.device,
        )
        tx_pos = r.tx_pos
        rx_pos = r.rx_pos
        samples = collect_interpolated_samples(r, interpolator, t0)

        for loop_id, sample in enumerate(samples):
            distances = compute_total_path_lengths(sample, tx_pos, rx_pos).unsqueeze(-1)
            toa = distances / r.c0
            rx = r.waveform(r.t_sample - toa)
            amp = compute_path_amplitudes(r, sample, distances.squeeze(-1), tx_pos=tx_pos, rx_pos=rx_pos)
            rx_weighted = rx * amp.unsqueeze(-1)
            rx_combined = torch.sum(rx_weighted, dim=-2)
            frame[loop_id] = r.tx_waveform * torch.conj(rx_combined)

        return frame.permute(1, 2, 0, 3)
