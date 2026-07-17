#!/usr/bin/env python3
"""Runtime patch for WiTwin radar-equation bistatic path loss."""

from __future__ import annotations

import math

import torch


def apply() -> None:
    """Patch WiTwin solvers to use amplitude loss proportional to 1/(R_tx * R_rx)."""
    from witwin.radar.solvers import common

    def compute_path_amplitudes(
        radar,
        sample: common.PathSample,
        total_path_lengths: torch.Tensor,
        *,
        tx_pos: torch.Tensor | None = None,
        rx_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tx_pos is None:
            tx_pos = radar.tx_pos
        if rx_pos is None:
            rx_pos = radar.rx_pos

        dist_tx = torch.cdist(sample.entry_points, tx_pos).transpose(0, 1).unsqueeze(1)
        dist_rx = torch.cdist(sample.points, rx_pos).transpose(0, 1).unsqueeze(0)
        fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(dist_tx * dist_rx, min=1e-6))

        scatter_power = torch.clamp(sample.intensities, min=0.0).view(1, 1, -1)
        pattern_gains = common.compute_antenna_pattern_gains(radar, sample, tx_pos, rx_pos)
        if pattern_gains is not None:
            scatter_power = scatter_power * torch.clamp(pattern_gains, min=0.0)
        amplitudes = radar.gain * torch.sqrt(scatter_power) * fspl_amp
        polarization_factor = common.compute_polarization_amplitudes(radar, sample)
        if polarization_factor is not None:
            amplitudes = amplitudes * polarization_factor
        return amplitudes

    common.compute_path_amplitudes = compute_path_amplitudes

    for module_name in (
        "witwin.radar.solvers.solver_pytorch",
        "witwin.radar.solvers.solver_dirichlet",
        "witwin.radar.solvers.solver_slang",
    ):
        try:
            module = __import__(module_name, fromlist=["compute_path_amplitudes"])
        except Exception:
            continue
        module.compute_path_amplitudes = compute_path_amplitudes
