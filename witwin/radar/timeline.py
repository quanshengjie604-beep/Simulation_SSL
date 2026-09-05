"""
Timeline — Multi-frame dynamic scene manager.

Stores discrete keyframes (point clouds + intensities at a fixed source frame
rate) and provides continuous-time interpolation for radar frame generation.

Usage:
    from radar.core import Timeline, Radar

    # From pre-computed point cloud sequence
    timeline = Timeline(frame_rate=30)
    timeline.add_pointcloud_sequence(pointclouds)  # list of (N_i, 3) tensors

    # Generate radar MIMO frames for the full duration
    radar = Radar(config)
    frames = timeline.generate(radar)  # (num_radar_frames, TX, RX, chirps, ADC)

    # Or get the interpolator for single-frame use
    interp = timeline.get_interpolator()
    frame = radar.mimo(interp, t0=0.5)
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .utils.vector import optional_vec3_tensor, scalar_tensor, vec3_tensor


_ALLOWED_SPACES = {"local", "world"}


def _normalize_space(value: str | None) -> str:
    if value is None:
        return "world"
    space = str(value).lower()
    if space not in _ALLOWED_SPACES:
        raise ValueError("TransformMotion.space must be 'local' or 'world'.")
    return space


class TransformMotion:
    """Single rigid transform motion for one structure."""

    def __init__(
        self,
        *,
        offset=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        axis=None,
        angular_velocity=0.0,
        angle=0.0,
        origin=None,
        space: str = "world",
        t_ref=0.0,
        parent: str | None = None,
    ) -> None:
        axis_t = optional_vec3_tensor(axis, name="TransformMotion.axis")
        if axis_t is not None and torch.linalg.norm(axis_t) <= 1e-12:
            raise ValueError("TransformMotion.axis must be non-zero.")
        if parent is not None:
            parent = str(parent)
            if not parent:
                raise ValueError("TransformMotion.parent must be a non-empty string.")

        self.offset: torch.Tensor = vec3_tensor(offset, name="TransformMotion.offset")
        self.velocity: torch.Tensor = vec3_tensor(velocity, name="TransformMotion.velocity")
        self.axis: torch.Tensor | None = axis_t
        self.angular_velocity: torch.Tensor = scalar_tensor(
            angular_velocity,
            name="TransformMotion.angular_velocity",
        )
        self.angle: torch.Tensor = scalar_tensor(angle, name="TransformMotion.angle")
        self.origin: torch.Tensor | None = optional_vec3_tensor(origin, name="TransformMotion.origin")
        self.space: str = _normalize_space(space)
        self.t_ref: torch.Tensor = scalar_tensor(t_ref, name="TransformMotion.t_ref")
        self.parent = parent

        has_translation = bool(self.offset.abs().sum().item() > 0.0 or self.velocity.abs().sum().item() > 0.0)
        has_rotation = self.axis is not None
        if not has_translation and not has_rotation and self.parent is None:
            raise ValueError("TransformMotion requires translation, rotation, or parent.")


class Timeline:
    """Multi-frame dynamic scene manager.

    Stores discrete keyframes and provides continuous-time interpolation
    for radar frame generation.
    """

    def __init__(self, frame_rate: int | float = 30, *, device: str | torch.device = "cuda"):
        """
        Args:
            frame_rate: source keyframe rate in Hz (e.g. 30 for 30 FPS motion data)
            device: torch device on which keyframe tensors are stored
        """
        self.frame_rate = frame_rate
        self.device = torch.device(device)
        self._positions: list[torch.Tensor] = []
        self._intensities: list[torch.Tensor] = []
        self._num_frames = 0

    @property
    def duration(self):
        if self._num_frames == 0:
            return 0.0
        return (self._num_frames - 1) / self.frame_rate

    @property
    def num_frames(self):
        return self._num_frames

    # ── Input methods ─────────────────────────────────────────────

    def add_keyframe(self, positions: torch.Tensor, intensities: torch.Tensor | None = None):
        """Append one keyframe.

        Args:
            positions: (N, 3) torch tensor on any device
            intensities: (N,) torch tensor or None (defaults to ones)
        """
        positions = positions.to(device=self.device, dtype=torch.float32)
        n = positions.shape[0]
        if intensities is None:
            intensities = torch.ones(n, dtype=torch.float32, device=self.device)
        else:
            intensities = intensities.to(device=self.device, dtype=torch.float32)
        self._positions.append(positions)
        self._intensities.append(intensities)
        self._num_frames += 1

    def add_pointcloud_sequence(self, pointclouds, intensities=None):
        """Bulk-add from a sequence of point clouds.

        Args:
            pointclouds: list of (N_i, 3) torch tensors, or (F, N, 3) torch tensor
            intensities: matching structure or None
        """
        if isinstance(pointclouds, torch.Tensor) and pointclouds.ndim == 3:
            for i in range(pointclouds.shape[0]):
                ints_i = intensities[i] if intensities is not None else None
                self.add_keyframe(pointclouds[i], ints_i)
            return
        for i, pc in enumerate(pointclouds):
            ints_i = intensities[i] if intensities is not None else None
            self.add_keyframe(pc, ints_i)

    def from_motion(self, scene, tracer, motion_data):
        """Render SMPL motion sequence into keyframes.

        Args:
            scene: Scene with an SMPL body named 'human'
            tracer: Tracer bound to the scene
            motion_data: dict or .npz path with keys:
                'pose': (F, 72) SMPL pose parameters
                'shape': (10,) or (F, 10) shape parameters
                'root_translation': (F, 3) root translation per frame
        """
        if isinstance(motion_data, str):
            motion_data = dict(np.load(motion_data))

        poses = motion_data["pose"]
        shapes = motion_data["shape"]
        translations = motion_data.get("root_translation", None)

        num_motion_frames = len(poses)
        if shapes.ndim == 1:
            shapes = np.tile(shapes, (num_motion_frames, 1))

        for i in tqdm(range(num_motion_frames), desc="Rendering motion frames"):
            trans = translations[i] if translations is not None else None
            scene.update_structure("human", pose=poses[i], shape=shapes[i], position=trans)
            points, intensities = tracer.trace()
            self.add_keyframe(points, intensities)

    # ── Interpolation ─────────────────────────────────────────────

    def get_interpolator(self):
        """Return an interpolator function: interpolator(t) -> (intensities, positions).

        Linear interpolation between adjacent keyframes. Points with
        intensity <= 0.01 are filtered out.
        """
        if self._num_frames == 0:
            raise ValueError("No keyframes added to timeline.")

        positions = self._positions
        intensities = self._intensities
        frame_rate = self.frame_rate
        num_frames = self._num_frames
        total_time = self.duration

        def interpolator(t):
            t = max(0.0, min(t, total_time))
            idx = int(t * frame_rate)
            if idx >= num_frames - 1:
                ints = intensities[-1]
                pos = positions[-1]
                mask = ints > 0.01
                return ints[mask], pos[mask]

            frac = (t * frame_rate) - idx

            pos1, pos2 = positions[idx], positions[idx + 1]
            int1, int2 = intensities[idx], intensities[idx + 1]

            n1, n2 = pos1.shape[0], pos2.shape[0]
            n_min = min(n1, n2)

            if n1 == n2:
                pos = pos1 * (1 - frac) + pos2 * frac
                ints = int1 * (1 - frac) + int2 * frac
            elif n1 < n2:
                pos_interp = pos1 * (1 - frac) + pos2[:n_min] * frac
                pos = torch.cat([pos_interp, pos2[n_min:]], dim=0)
                ints_interp = int1 * (1 - frac) + int2[:n_min] * frac
                ints = torch.cat([ints_interp, int2[n_min:] * frac], dim=0)
            else:
                pos_interp = pos1[:n_min] * (1 - frac) + pos2 * frac
                pos = torch.cat([pos_interp, pos1[n_min:]], dim=0)
                ints_interp = int1[:n_min] * (1 - frac) + int2 * frac
                ints = torch.cat([ints_interp, int1[n_min:] * (1 - frac)], dim=0)

            mask = ints > 0.01
            return ints[mask], pos[mask]

        return interpolator

    def get_frame_interpolator(self, radar, frame_idx):
        """Return an interpolator for a single radar frame with velocity correction.

        Scales the displacement so that the interpolated velocity matches the
        real physical velocity when source keyframe spacing differs from the
        radar frame duration.
        """
        if self._num_frames < 2:
            raise ValueError("Need at least 2 keyframes for frame interpolator.")

        cfg = radar.config
        dt_source = 1.0 / self.frame_rate
        T_chirp = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        T_frame = T_chirp * cfg.num_tx * cfg.chirp_per_frame
        vel_scale = T_frame / dt_source

        t0_real = frame_idx / cfg.frame_per_second
        kf_idx = int(t0_real * self.frame_rate)
        kf_idx = min(kf_idx, self._num_frames - 2)

        p0 = self._positions[kf_idx]
        p1 = self._positions[kf_idx + 1]
        int0 = self._intensities[kf_idx]

        n = min(p0.shape[0], p1.shape[0])
        if n == 0:
            device = self.device

            def empty_interp(_t):
                return (
                    torch.zeros(0, device=device),
                    torch.zeros(0, 3, device=device),
                )
            return empty_interp, 0.0

        p0_n, p1_n = p0[:n], p1[:n]
        inten = int0[:n]
        p1_scaled = p0_n + (p1_n - p0_n) * vel_scale

        def interpolator(t):
            frac = t / T_frame if T_frame > 0 else 0.0
            pos = p0_n + (p1_scaled - p0_n) * frac
            mask = inten > 0.01
            return inten[mask], pos[mask]

        return interpolator, 0.0

    # ── Generation ────────────────────────────────────────────────

    def generate(self, radar, progress: bool = True, velocity_corrected: bool = True) -> torch.Tensor:
        """Generate all radar MIMO frames for the full timeline.

        Returns:
            torch tensor of shape (num_radar_frames, TX, RX, chirps, ADC) complex
        """
        if self._num_frames < 1:
            raise ValueError("Need at least 1 keyframe to generate frames.")

        num_radar_frames = max(1, int(self.duration * radar.config.frame_per_second))

        frames: list[torch.Tensor] = []
        it = range(num_radar_frames)
        if progress:
            it = tqdm(it, desc="Generating radar frames")

        if velocity_corrected and self._num_frames >= 2:
            for i in it:
                interp, t0 = self.get_frame_interpolator(radar, i)
                frames.append(radar.mimo(interp, t0))
        else:
            interpolator = self.get_interpolator()
            for i in it:
                t0 = i / radar.config.frame_per_second
                frames.append(radar.mimo(interpolator, t0))

        return torch.stack(frames, dim=0)

    def generate_rd(self, radar, tx: int = 0, rx: int = 0, progress: bool = True):
        """Generate Range-Doppler maps for all frames.

        Returns:
            rd_mags: (num_frames, chirps, ADC) numpy array — magnitude in dB
            ranges: (num_range_bins//2,) numpy array — range axis in meters
            velocities: (num_doppler_bins,) numpy array — velocity axis in m/s
        """
        frames = self.generate(radar, progress=progress)
        from .sigproc import process_rd

        rd_mags = []
        for f in frames:
            rd_mag, _, _, _ = process_rd(radar, f, tx=tx, rx=rx)
            rd_mags.append(rd_mag)

        return (
            np.stack(rd_mags, axis=0),
            radar.ranges.detach().cpu().numpy(),
            radar.velocities.detach().cpu().numpy(),
        )
