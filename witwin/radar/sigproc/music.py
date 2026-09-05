"""
MUSIC (Multiple Signal Classification) radar imaging for Uniform Planar Arrays.

Implements 2D MUSIC-based angle estimation with spatial smoothing,
producing high-resolution radar images from raw MIMO signals.

Ported from RFRT_V1.5/main/utils/renderer.py (RadarImager class) with cleanup.
"""

import numpy as np
import torch


class MUSICImager:
    """2D MUSIC-based radar imager for Uniform Planar Arrays (UPA).

    Computes a 2D angular pseudo-spectrum from radar frame data using the
    MUSIC algorithm with forward-backward spatial smoothing.

    Args:
        num_tx: number of TX antennas (array rows, M)
        num_rx: number of RX antennas (array columns, N)
        num_signals: number of signal sources to estimate (subspace split)
        spatial_smooth: spatial smoothing factor for decorrelation
        num_pixels: output image resolution (num_pixels x num_pixels)
        fov: field of view in radians, default pi/2 (±45°)
        num_chirps: number of chirps (snapshots) to use from each range bin
    """

    def __init__(self, num_tx=20, num_rx=20, num_signals=7, spatial_smooth=3,
                 num_pixels=128, fov=np.pi / 2, num_chirps=8):
        self.M = num_tx
        self.N = num_rx
        self.num_signals = num_signals
        self.spatial_smooth = spatial_smooth
        self.num_pixels = num_pixels
        self.num_chirps = num_chirps

        self.v_angle = torch.linspace(fov / 2, -fov / 2, num_pixels)
        self.h_angle = torch.linspace(fov / 2, -fov / 2, num_pixels)

        M_eff = self.M - spatial_smooth
        N_eff = self.N - spatial_smooth

        self.steering_vec = self._build_steering_vectors(
            M_eff, N_eff, self.h_angle, self.v_angle
        ).permute(1, 2, 0).to(torch.complex64)

    @staticmethod
    def _build_steering_vectors(M, N, h_angle, v_angle, spacing=0.5):
        """Build the 2D steering vector matrix for a UPA.

        Args:
            M, N: effective array dimensions after spatial smoothing
            h_angle, v_angle: angle grids (radians)
            spacing: antenna spacing in wavelengths (default 0.5 = half-lambda)

        Returns:
            A: (M*N, len(v_angle), len(h_angle)) complex steering matrix
        """
        T = len(v_angle)
        P = len(h_angle)

        ii, jj = torch.meshgrid(torch.arange(T), torch.arange(P), indexing='ij')

        ax = torch.exp(
            1j * 2 * np.pi * spacing *
            torch.sin(v_angle[ii]).unsqueeze(-1) * torch.arange(M)
        ).reshape(T, P, M)

        ay = torch.exp(
            1j * 2 * np.pi * spacing *
            torch.sin(h_angle[jj]).unsqueeze(-1) * torch.arange(N)
        ).reshape(T, P, N)

        # Kronecker product via einsum
        A = torch.einsum('ijk,ijl->ijkl', ax, ay).view(T, P, M * N).permute(2, 0, 1)
        return A

    def music_spectrum(self, angle_data):
        """Compute 2D MUSIC pseudo-spectrum with spatial smoothing.

        Args:
            angle_data: (B, M, N, T) complex tensor — B range bins, M×N array, T snapshots

        Returns:
            S: (B, num_pixels, num_pixels) real tensor — MUSIC pseudo-spectrum
        """
        B, M, N, T = angle_data.shape
        L = self.spatial_smooth
        steering_vec = self.steering_vec.to(device=angle_data.device)

        # Forward-backward spatial smoothing
        shift_indices = [(jj, kk) for jj in range(L + 1) for kk in range(L + 1)]
        X_subs = torch.stack([
            angle_data[:, jj:M - L + jj, kk:N - L + kk]
            for jj, kk in shift_indices
        ], dim=1)
        X_vecs = X_subs.view(B, (L + 1) ** 2, -1, T)
        R_sum = torch.einsum('bijk,bilk->bjlk', X_vecs, X_vecs.conj()).sum(dim=-1)

        R_est = R_sum / (T * (L + 1) ** 2)

        # Eigen-decomposition → noise subspace
        D, U = torch.linalg.eigh(R_est)
        _, indices = torch.topk(D, k=D.size(-1), largest=True, sorted=True)
        Un = torch.gather(
            U, 2,
            indices[:, self.num_signals:].unsqueeze(1).expand(-1, U.size(1), -1)
        ).to(torch.complex64)

        # MUSIC pseudo-spectrum: 1 / (a^H * Un * Un^H * a)
        UU_H = torch.matmul(Un, Un.transpose(-1, -2).conj())
        A_UU_H_A_H = torch.matmul(
            torch.einsum('ijk,akl->aijl', steering_vec, UU_H),
            steering_vec.transpose(-1, -2).conj()
        )

        S = torch.reciprocal(
            A_UU_H_A_H.diagonal(dim1=-2, dim2=-1)
        ).view(B, self.num_pixels, self.num_pixels)

        return S

    def radar_image(self, sig, range_bins=None):
        """Generate a 3D radar image from a MIMO radar signal.

        Args:
            sig: (TX, RX, chirps, ADC_samples) complex tensor — full radar frame
            range_bins: optional list/tensor of range bin indices to image.
                        If None, auto-detects the peak ±4 bins.

        Returns:
            image3D: (num_pixels, num_pixels, num_range_bins) real tensor
        """
        range_fft = torch.fft.fft(sig, dim=3)

        if range_bins is None:
            # Auto-detect peak range bin
            _, range_index = torch.max(torch.abs(range_fft[0, 0, 0, :]), dim=0)
            range_bins = torch.arange(range_index - 4, range_index + 1, device=sig.device)
        elif not isinstance(range_bins, torch.Tensor):
            range_bins = torch.as_tensor(range_bins, device=sig.device)
        else:
            range_bins = range_bins.to(device=sig.device)

        # Extract angle data for selected range bins: (num_bins, TX, RX, chirps)
        angle_data = range_fft[:, :, :self.num_chirps, range_bins].permute(3, 0, 1, 2)

        image3D = self.music_spectrum(angle_data)
        return image3D.permute(1, 2, 0)  # (H, W, num_bins)
