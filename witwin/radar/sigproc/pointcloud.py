"""
Radar signal processing: Range-Doppler FFT and point cloud extraction.

Pipeline: radar frame -> range FFT -> clutter removal -> Doppler FFT -> peak
detection -> AoA -> 3D point cloud.
"""

from __future__ import annotations

import numpy as np
import torch

from ..types import DetectorType
from ..utils.tensor import real_dtype as _real_dtype


def _hamming_window(length: int, *, reference: torch.Tensor) -> torch.Tensor:
    return torch.hamming_window(
        length,
        periodic=False,
        dtype=_real_dtype(reference),
        device=reference.device,
    )


class FrameConfig:
    """Derived frame parameters from a Radar config."""

    def __init__(self, radar):
        cfg = radar.config
        self.numTxAntennas = cfg.num_tx
        self.numRxAntennas = cfg.num_rx
        self.numLoopsPerFrame = cfg.chirp_per_frame
        self.numADCSamples = cfg.adc_samples
        self.numAngleBins = cfg.num_angle_bins

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame

        self.chirpSize = self.numRxAntennas * self.numADCSamples
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame

        self.range_resolution = radar.range_resolution
        self.doppler_resolution = radar.doppler_resolution

        self.tx_loc_hw = radar.tx_loc / (radar._lambda / 2)


class PointCloudProcessConfig:
    """Configuration for point cloud extraction from radar frames."""

    def __init__(
        self,
        radar,
        static_clutter_removal: bool = False,
        energy_top_k: int = 128,
        range_cut: bool = False,
        output_velocity: bool = True,
        output_snr: bool = True,
        output_range: bool = True,
        output_in_meter: bool = True,
    ):
        self.frame_config = FrameConfig(radar)
        self.enable_static_clutter_removal = static_clutter_removal
        self.use_energy_top_k = energy_top_k > 0
        self.energy_top_k = energy_top_k
        self.range_cut = range_cut
        self.output_velocity = output_velocity
        self.output_snr = output_snr
        self.output_range = output_range
        self.output_in_meter = output_in_meter

        dim = 3
        if self.output_velocity:
            self.velocity_dim = dim
            dim += 1
        if self.output_snr:
            self.snr_dim = dim
            dim += 1
        if self.output_range:
            self.range_dim = dim
            dim += 1


# ---------------------------------------------------------------------------
# Core DSP functions (torch-only)
# ---------------------------------------------------------------------------

def frame_reshape(frame: torch.Tensor, frame_config) -> torch.Tensor:
    """Reshape a flat IQ frame into (TX, RX, chirps, ADC_samples)."""
    shape = (
        frame_config.numLoopsPerFrame,
        frame_config.numTxAntennas,
        frame_config.numRxAntennas,
        -1,
    )
    return frame.reshape(shape).permute(1, 2, 0, 3)


def range_fft(reshaped_frame: torch.Tensor, frame_config) -> torch.Tensor:
    """Apply a Hamming-windowed FFT along the fast-time axis."""
    window = _hamming_window(frame_config.numADCSamples, reference=reshaped_frame)
    return torch.fft.fft(reshaped_frame * window.view(1, 1, 1, -1), dim=-1)


def clutter_removal(input_val: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Static clutter removal by subtracting the mean along one axis."""
    return input_val - input_val.mean(dim=axis, keepdim=True)


def doppler_fft(range_result: torch.Tensor, frame_config) -> torch.Tensor:
    """Apply a Hamming-windowed FFT along slow time with fftshift."""
    window = _hamming_window(frame_config.numLoopsPerFrame, reference=range_result)
    windowed = range_result * window.view(1, 1, -1, 1)
    return torch.fft.fftshift(torch.fft.fft(windowed, dim=2), dim=2)


def naive_xyz(virtual_ant: torch.Tensor, num_tx=3, num_rx=4, fft_size=64, tx_loc_hw=None):
    """Estimate direction cosines from virtual antenna data."""
    assert num_tx > 2, "Need > 2 TX antennas for 3D AoA estimation"

    if num_tx > 4:
        return _aoa_2d_fft(virtual_ant, num_tx, num_rx, fft_size)
    return _aoa_phase_comparison(virtual_ant, num_tx, num_rx, fft_size, tx_loc_hw=tx_loc_hw)


def _aoa_phase_comparison(virtual_ant: torch.Tensor, num_tx, num_rx, fft_size, tx_loc_hw=None):
    """Phase-comparison AoA for small arrays (for example 3TX 4RX)."""
    num_detected = virtual_ant.shape[1]
    n_az = min(2 * num_rx, fft_size)
    azimuth_ant = virtual_ant[:n_az, :]

    device = virtual_ant.device
    real_dtype = _real_dtype(virtual_ant)

    azimuth_padded = torch.zeros((fft_size, num_detected), dtype=virtual_ant.dtype, device=device)
    azimuth_padded[:n_az, :] = azimuth_ant
    azimuth_fft = torch.fft.fft(azimuth_padded, dim=0)
    k_max = torch.argmax(torch.abs(azimuth_fft), dim=0).to(torch.int64)
    peak_1 = azimuth_fft[k_max, torch.arange(num_detected, device=device)]
    signed_k_max = torch.where(k_max > (fft_size // 2) - 1, k_max - fft_size, k_max)
    wx = (2 * torch.pi / fft_size) * signed_k_max.to(real_dtype)
    x_vector = wx / torch.pi

    el_start = 2 * num_rx
    n_el = min(num_rx, virtual_ant.shape[0] - el_start)
    elevation_ant = virtual_ant[el_start:el_start + n_el, :]

    elevation_padded = torch.zeros((fft_size, num_detected), dtype=virtual_ant.dtype, device=device)
    elevation_padded[:n_el, :] = elevation_ant
    elevation_fft = torch.fft.fft(elevation_padded, dim=0)
    elevation_power = torch.log2(torch.clamp(torch.abs(elevation_fft), min=1e-12))
    elevation_max = torch.argmax(elevation_power, dim=0).to(torch.int64)
    peak_2 = elevation_fft[elevation_max, torch.arange(num_detected, device=device)]

    el_tx_dx = float(tx_loc_hw[2][0] - tx_loc_hw[0][0]) if tx_loc_hw is not None else 2.0
    phase_adjust = torch.exp(1j * torch.tensor(el_tx_dx, dtype=real_dtype, device=device) * wx)
    wz = torch.angle(peak_1 * torch.conj(peak_2) * phase_adjust)
    z_vector = wz / torch.pi

    y_possible = 1 - x_vector.square() - z_vector.square()
    valid = y_possible >= 0
    x_vector = torch.where(valid, x_vector, torch.zeros_like(x_vector))
    z_vector = torch.where(valid, z_vector, torch.zeros_like(z_vector))
    y_vector = torch.sqrt(torch.clamp(y_possible, min=0.0))
    return x_vector, y_vector, z_vector


def _aoa_2d_fft(virtual_ant: torch.Tensor, num_tx, num_rx, fft_size):
    """2D FFT AoA for larger virtual UPAs."""
    num_detected = virtual_ant.shape[1]
    n_az_per_row = 2 * num_rx
    n_el_rows = num_tx // 2
    reshaped = virtual_ant.reshape(num_tx, num_rx, num_detected)

    device = virtual_ant.device
    real_dtype = _real_dtype(virtual_ant)

    grid = torch.zeros((n_el_rows, n_az_per_row, num_detected), dtype=virtual_ant.dtype, device=device)
    grid[:, :num_rx, :] = reshaped[0::2]
    grid[:, num_rx:, :] = reshaped[1::2]

    padded = torch.zeros((fft_size, fft_size, num_detected), dtype=virtual_ant.dtype, device=device)
    padded[:n_el_rows, :n_az_per_row, :] = grid
    spectrum_2d = torch.fft.fft2(padded, dim=(0, 1))
    power_flat = torch.abs(spectrum_2d).reshape(-1, num_detected)
    peak_idx = torch.argmax(power_flat, dim=0).to(torch.int64)
    k_el = peak_idx // fft_size
    k_az = peak_idx % fft_size
    k_az = torch.where(k_az > fft_size // 2 - 1, k_az - fft_size, k_az)
    k_el = torch.where(k_el > fft_size // 2 - 1, k_el - fft_size, k_el)
    x_vector = (2 * torch.pi / fft_size) * k_az.to(real_dtype) / torch.pi
    z_vector = (2 * torch.pi / fft_size) * k_el.to(real_dtype) / torch.pi
    y_possible = 1 - x_vector.square() - z_vector.square()
    valid = y_possible >= 0
    x_vector = torch.where(valid, x_vector, torch.zeros_like(x_vector))
    z_vector = torch.where(valid, z_vector, torch.zeros_like(z_vector))
    y_vector = torch.sqrt(torch.clamp(y_possible, min=0.0))
    return x_vector, y_vector, z_vector


def _compensate_tdm_phase(aoa_input: torch.Tensor, velocities: torch.Tensor, radar, fc) -> torch.Tensor:
    """Remove the velocity-dependent TDM-MIMO phase offset."""
    lam = radar._lambda
    t_chirp = (radar.config.idle_time + radar.config.ramp_end_time) * 1e-6
    num_tx = fc.numTxAntennas
    num_rx = fc.numRxAntennas

    compensated = aoa_input.clone()
    for tx_i in range(1, num_tx):
        va_start = tx_i * num_rx
        va_end = va_start + num_rx
        phase_offset = 4 * torch.pi * velocities * tx_i * t_chirp / lam
        compensation = torch.exp(-1j * phase_offset).unsqueeze(0)
        compensated[va_start:va_end, :] *= compensation

    return compensated


def _energy_topk_mask(values: torch.Tensor, top_k: int) -> torch.Tensor:
    total_bins = values.numel()
    top_k = min(top_k, total_bins)
    if top_k <= 0:
        return torch.zeros_like(values, dtype=torch.bool)
    if top_k >= total_bins:
        return torch.ones_like(values, dtype=torch.bool)
    threshold = torch.topk(values.reshape(-1), top_k).values.min()
    return values >= threshold


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def frame2pointcloud(frame: torch.Tensor, cfg, radar=None) -> torch.Tensor:
    """Convert a radar frame tensor to a point cloud (6, N)."""
    if radar is None:
        raise ValueError("frame2pointcloud requires a radar instance so TDM-MIMO compensation is always applied.")

    fc = cfg.frame_config
    range_result = range_fft(frame, fc)
    if cfg.enable_static_clutter_removal:
        range_result = clutter_removal(range_result, axis=2)
    doppler_result = doppler_fft(range_result, fc)

    doppler_sum = doppler_result.sum(dim=(0, 1))
    doppler_db = 20 * torch.log10(torch.abs(doppler_sum) + 1e-6)

    if cfg.range_cut:
        doppler_db[:, :25] = -100
        doppler_db[:, 125:] = -100

    cfar_result = (
        _energy_topk_mask(doppler_db, cfg.energy_top_k)
        if cfg.use_energy_top_k
        else torch.zeros_like(doppler_db, dtype=torch.bool)
    )
    det_peaks = torch.argwhere(cfar_result)
    if det_peaks.shape[0] == 0:
        return torch.zeros((6, 0), dtype=torch.float64, device=frame.device)

    r = det_peaks[:, 1].to(torch.float64)
    v = (det_peaks[:, 0] - fc.numDopplerBins // 2).to(torch.float64)
    if cfg.output_in_meter:
        r = r * fc.range_resolution
        v = v * fc.doppler_resolution

    energy = doppler_db[cfar_result]
    aoa_input = doppler_result[:, :, cfar_result].reshape(fc.numTxAntennas * fc.numRxAntennas, -1)
    aoa_input = _compensate_tdm_phase(aoa_input, v, radar, fc)
    x_vec, y_vec, z_vec = naive_xyz(
        aoa_input,
        num_tx=fc.numTxAntennas,
        num_rx=fc.numRxAntennas,
        tx_loc_hw=fc.tx_loc_hw,
    )

    x, y, z = x_vec * r, y_vec * r, z_vec * r
    point_cloud = torch.stack([x, y, z, v, energy.to(torch.float64), r], dim=0)
    return point_cloud[:, y_vec != 0]


def process_pc(
    radar,
    frame: torch.Tensor,
    static_clutter_removal=True,
    positive_velocity_only=True,
    detector: DetectorType = "cfar",
    guard_cells=(2, 4),
    training_cells=(4, 8),
    pfa=1e-3,
    max_points=512,
    energy_top_k=128,
) -> np.ndarray:
    """Radar frame -> filtered point cloud (N, 6) as numpy."""
    detector_kind = DetectorType(detector)

    if detector_kind == DetectorType.TOPK:
        cfg = PointCloudProcessConfig(
            radar,
            static_clutter_removal=static_clutter_removal,
            energy_top_k=energy_top_k,
        )
        pc = frame2pointcloud(frame, cfg, radar=radar).transpose(0, 1)
        if positive_velocity_only and pc.shape[0] > 0:
            pc = pc[pc[:, 3] > 0]
        return pc.detach().cpu().numpy()

    return _process_pc_cfar(
        radar,
        frame,
        static_clutter_removal=static_clutter_removal,
        positive_velocity_only=positive_velocity_only,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
        max_points=max_points,
    )


def _process_pc_cfar(
    radar,
    frame: torch.Tensor,
    static_clutter_removal=True,
    positive_velocity_only=True,
    guard_cells=(2, 4),
    training_cells=(4, 8),
    pfa=1e-3,
    max_points=512,
) -> np.ndarray:
    """Internal: point cloud extraction via CA-CFAR detection."""
    from .cfar import ca_cfar_2d_fast

    fc = FrameConfig(radar)
    range_result = range_fft(frame, fc)
    if static_clutter_removal:
        range_result = clutter_removal(range_result, axis=2)
    doppler_result = doppler_fft(range_result, fc)

    doppler_sum = doppler_result.sum(dim=(0, 1))
    doppler_mag = torch.abs(doppler_sum)

    cfar_mask, _ = ca_cfar_2d_fast(
        doppler_mag,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
    )

    det_peaks = torch.argwhere(cfar_mask)
    if det_peaks.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float64)

    if det_peaks.shape[0] > max_points:
        energies = doppler_mag[det_peaks[:, 0], det_peaks[:, 1]]
        top_idx = torch.topk(energies, max_points).indices
        det_peaks = det_peaks[top_idx]
        cfar_mask = torch.zeros_like(cfar_mask, dtype=torch.bool)
        cfar_mask[det_peaks[:, 0], det_peaks[:, 1]] = True

    r = det_peaks[:, 1].to(torch.float64) * fc.range_resolution
    v = (det_peaks[:, 0] - fc.numDopplerBins // 2).to(torch.float64) * fc.doppler_resolution
    energy = 20 * torch.log10(doppler_mag[cfar_mask] + 1e-6)

    aoa_input = doppler_result[:, :, cfar_mask].reshape(fc.numTxAntennas * fc.numRxAntennas, -1)
    aoa_input = _compensate_tdm_phase(aoa_input, v, radar, fc)
    x_vec, y_vec, z_vec = naive_xyz(
        aoa_input,
        num_tx=fc.numTxAntennas,
        num_rx=fc.numRxAntennas,
        tx_loc_hw=fc.tx_loc_hw,
    )

    x, y, z = x_vec * r, y_vec * r, z_vec * r
    pc = torch.stack([x, y, z, v, energy.to(torch.float64), r], dim=0)
    pc = pc[:, y_vec != 0].transpose(0, 1)
    if positive_velocity_only and pc.shape[0] > 0:
        pc = pc[pc[:, 3] > 0]
    return pc.detach().cpu().numpy()


def process_rd(radar, frame: torch.Tensor, tx: int = 0, rx: int = 0, *, static_clutter_removal: bool = False):
    """Compute a Range-Doppler map from a MIMO frame.

    Returns: (rd_mag, rd_map, ranges, velocities) as numpy arrays.
    """
    data = frame[tx, rx].clone()
    data = data - data.mean(dim=-1, keepdim=True)
    if static_clutter_removal:
        data = data - data.mean(dim=-2, keepdim=True)

    n_adc = data.shape[-1]
    n_chirps = data.shape[-2]
    data = data * _hamming_window(n_adc, reference=data).view(1, -1)
    data = data * _hamming_window(n_chirps, reference=data).view(-1, 1)
    range_result = torch.fft.fft(data, dim=-1)
    rd_map = torch.fft.fftshift(torch.fft.fft(range_result, dim=-2), dim=-2)
    rd_mag = 20 * torch.log10(torch.abs(rd_map) + 1e-6)
    return (
        rd_mag.detach().cpu().numpy(),
        rd_map.detach().cpu().numpy(),
        radar.ranges.detach().cpu().numpy(),
        radar.velocities.detach().cpu().numpy(),
    )



def reg_data(data, pc_size):
    """Regularize a point cloud to a fixed size by sampling or duplication."""
    pc_tmp = np.zeros((pc_size, data.shape[1]), dtype=np.float32)
    pc_no = data.shape[0]
    if pc_no == 0:
        return pc_tmp
    if pc_no < pc_size:
        fill_list = np.random.choice(pc_size, size=pc_no, replace=False)
        fill_set = set(fill_list)
        pc_tmp[fill_list] = data
        dupl_list = [x for x in range(pc_size) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no, size=len(dupl_list), replace=True)
        pc_tmp[dupl_list] = data[dupl_pc]
    else:
        pc_tmp = data[np.random.choice(pc_no, size=pc_size, replace=False)]
    return pc_tmp
