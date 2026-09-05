"""Unified FMCW radar with selectable compute backend."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .types import MotionSampling, SamplingMode, SolverBackend
from .utils.antenna import evaluate_antenna_pattern_vectors, evaluate_antenna_pattern_xy
from .utils.vector import vec3_tensor


@dataclass(frozen=True)
class RadarConfig:
    num_tx: int
    num_rx: int
    fc: float
    slope: float
    adc_samples: int
    adc_start_time: float
    sample_rate: float
    idle_time: float
    ramp_end_time: float
    chirp_per_frame: int
    frame_per_second: float
    num_doppler_bins: int
    num_range_bins: int
    num_angle_bins: int
    power: float
    tx_loc: tuple[tuple[float, float, float], ...]
    rx_loc: tuple[tuple[float, float, float], ...]
    antenna_pattern: dict[str, Any] | None = None
    noise_model: dict[str, Any] | None = None
    polarization: dict[str, Any] | None = None
    receiver_chain: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RadarConfig":
        from .validation import validate_radar_config

        return validate_radar_config(config)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "RadarConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _target_from_position(position: torch.Tensor) -> torch.Tensor:
    return position + torch.tensor((0.0, 0.0, -1.0), dtype=torch.float32)


def _component_dtype(signal: torch.Tensor) -> torch.dtype:
    return torch.float64 if signal.dtype == torch.complex128 else torch.float32


def _randn(shape, *, device, dtype, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.clamp(torch.linalg.norm(vectors, dim=-1, keepdim=True), min=1e-12)


def quantize_complex_signal(signal: torch.Tensor, *, bits: int, full_scale: float) -> torch.Tensor:
    levels = 2 ** bits
    step = (2.0 * full_scale) / (levels - 1)

    def _quantize(component: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(component, min=-full_scale, max=full_scale)
        code = torch.round((clipped + full_scale) / step)
        return code * step - full_scale

    real = _quantize(signal.real)
    imag = _quantize(signal.imag)
    return torch.complex(real, imag).to(dtype=signal.dtype)


def db_to_voltage_gain(gain_db: float) -> float:
    return 10.0 ** (float(gain_db) / 20.0)


class ReceiverChainRuntime:
    def __init__(self, config: dict[str, Any], *, device: str | torch.device):
        self.config = config
        self.device = device

    @classmethod
    def from_config(cls, config: dict[str, Any], *, device: str | torch.device) -> "ReceiverChainRuntime":
        return cls(config=config, device=device)

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        processed = signal
        lna = self.config.get("lna")
        agc = self.config.get("agc")
        adc = self.config.get("adc")
        if lna is not None:
            processed = processed * db_to_voltage_gain(lna["gain_db"])
        if agc is not None:
            processed = self._apply_agc(processed, agc)
        if adc is not None:
            processed = quantize_complex_signal(
                processed,
                bits=adc["bits"],
                full_scale=adc["full_scale"],
            )
        return processed

    def _apply_agc(self, signal: torch.Tensor, config: dict) -> torch.Tensor:
        real_dtype = signal.real.dtype
        magnitude_sq = signal.real.square() + signal.imag.square()

        if signal.ndim == 4 and config["mode"] == "per_rx":
            rms = torch.sqrt(torch.clamp(magnitude_sq.mean(dim=(0, 2, 3), keepdim=True), min=1e-24))
            target = torch.tensor(config["target_rms"], dtype=real_dtype, device=signal.device).view(1, 1, 1, 1)
        else:
            rms = torch.sqrt(torch.clamp(magnitude_sq.mean(), min=1e-24))
            target = torch.tensor(config["target_rms"], dtype=real_dtype, device=signal.device)

        gain = target / rms
        min_gain = db_to_voltage_gain(config["min_gain_db"])
        max_gain = db_to_voltage_gain(config["max_gain_db"])
        gain = torch.clamp(gain, min=min_gain, max=max_gain)
        return signal * gain.to(dtype=signal.dtype)


class NoiseModelRuntime:
    def __init__(self, config: dict[str, Any], *, device: str | torch.device):
        self.config = config
        self.device = device

    @classmethod
    def from_config(cls, config: dict[str, Any], *, device: str | torch.device) -> "NoiseModelRuntime":
        return cls(config=config, device=device)

    def apply(self, signal: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        noisy = signal
        phase = self.config.get("phase")
        thermal = self.config.get("thermal")
        quantization = self.config.get("quantization")
        if phase is not None and phase["std"] > 0.0:
            noisy = self._apply_phase_noise(noisy, std=phase["std"], generator=generator)
        if thermal is not None and thermal["std"] > 0.0:
            noisy = self._apply_thermal_noise(noisy, std=thermal["std"], generator=generator)
        if quantization is not None:
            noisy = quantize_complex_signal(
                noisy,
                bits=quantization["bits"],
                full_scale=quantization["full_scale"],
            )
        return noisy

    def _apply_phase_noise(
        self,
        signal: torch.Tensor,
        *,
        std: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        real = _component_dtype(signal)
        if signal.ndim == 4:
            phase_shape = signal.shape[-2:]
            broadcast_shape = (1, 1, *phase_shape)
        elif signal.ndim in (1, 2):
            phase_shape = signal.shape
            broadcast_shape = phase_shape
        else:
            raise ValueError("Phase noise currently supports chirp (T,), frame (F, T), or mimo (TX, RX, F, T) tensors.")

        innovations = _randn(phase_shape, device=signal.device, dtype=real, generator=generator) * std
        phase = torch.cumsum(innovations.reshape(-1), dim=0).reshape(phase_shape).reshape(broadcast_shape)
        phase_factor = torch.polar(torch.ones_like(phase, dtype=real), phase)
        return signal * phase_factor.to(dtype=signal.dtype)

    def _apply_thermal_noise(
        self,
        signal: torch.Tensor,
        *,
        std: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        real = _component_dtype(signal)
        real_part = _randn(signal.shape, device=signal.device, dtype=real, generator=generator) * std
        imag_part = _randn(signal.shape, device=signal.device, dtype=real, generator=generator) * std
        noise = torch.complex(real_part, imag_part).to(dtype=signal.dtype)
        return signal + noise


class PolarizationRuntime:
    def __init__(
        self,
        *,
        tx_world: torch.Tensor,
        rx_world: torch.Tensor,
        reflection_flip: bool = True,
    ) -> None:
        self.tx_world = tx_world
        self.rx_world = rx_world
        self.reflection_flip = bool(reflection_flip)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        device: str | torch.device,
        radar,
    ) -> "PolarizationRuntime":
        tx_local = _normalize_rows(torch.tensor(config["tx"], dtype=torch.float32, device=device))
        rx_local = _normalize_rows(torch.tensor(config["rx"], dtype=torch.float32, device=device))
        tx_world = _normalize_rows(radar.world_from_local_vectors(tx_local))
        rx_world = _normalize_rows(radar.world_from_local_vectors(rx_local))
        return cls(
            tx_world=tx_world.contiguous(),
            rx_world=rx_world.contiguous(),
            reflection_flip=bool(config.get("reflection_flip", True)),
        )


class Radar:
    def __init__(
        self,
        config: RadarConfig | Mapping[str, Any],
        backend: SolverBackend = "dirichlet",
        pad_factor: int = 16,
        device: str | torch.device = "cuda",
        *,
        position=(0.0, 0.0, 0.0),
        target=None,
        up=(0.0, 1.0, 0.0),
        fov: float = 60.0,
        name: str | None = None,
    ):
        """
        Args:
            config: ``RadarConfig`` or a raw mapping accepted by ``RadarConfig.from_dict``.
            backend: "dirichlet" | "slang" | "pytorch"
            pad_factor: FFT zero-padding factor for the Dirichlet backend
            device: compute device for public tensors and PyTorch execution
            position: radar origin in world coordinates
            target: look-at target in world coordinates. Defaults to one meter along -Z from position.
            up: world-space up vector
            fov: perspective field of view in degrees for ray tracing
            name: optional identifier used by ``Radar.simulate_group``
        """
        self.c0 = 299792458
        self.backend = SolverBackend(backend)
        self.device: torch.device = self._resolve_device(
            device=torch.device(device), backend=self.backend
        )
        self.name = None if name is None else str(name)
        self._set_pose_fields(position=position, target=target, up=up, fov=fov)

        self.config: RadarConfig = config if isinstance(config, RadarConfig) else RadarConfig.from_dict(config)
        cfg = self.config

        if (
            cfg.receiver_chain is not None
            and cfg.receiver_chain.get("adc") is not None
            and cfg.noise_model is not None
            and cfg.noise_model.get("quantization") is not None
        ):
            raise ValueError(
                "Radar receiver_chain.adc and noise_model.quantization cannot both be enabled; use one quantizer."
            )

        antenna_spacing = self.c0 / cfg.fc / 2
        tx_loc = torch.tensor(cfg.tx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        rx_loc = torch.tensor(cfg.rx_loc, dtype=torch.float32, device=self.device) * antenna_spacing
        self.tx_loc = tx_loc
        self.rx_loc = rx_loc
        self._refresh_pose_dependent_state()

        self.t_sample = (
            torch.arange(0, cfg.adc_samples, dtype=torch.float64, device=self.device)
            / (cfg.sample_rate * 1e3)
            + cfg.adc_start_time * 1e-6
        )
        self.tx_waveform = self.waveform(self.t_sample)
        self._lambda = self.c0 / cfg.fc

        self.transmit_power_watts = 1e-3 * (10.0 ** (cfg.power / 10.0))
        reference_impedance = (
            cfg.receiver_chain["reference_impedance_ohm"] if cfg.receiver_chain is not None else 50.0
        )
        self.tx_voltage_rms = math.sqrt(self.transmit_power_watts * reference_impedance)
        self.gain = self.tx_voltage_rms if cfg.receiver_chain is not None else 1.0

        from .validation import default_dipole_antenna_pattern

        self.antenna_pattern_config = cfg.antenna_pattern or default_dipole_antenna_pattern()
        self._build_antenna_pattern_runtime(self.antenna_pattern_config)
        self.noise_model_config = cfg.noise_model
        self.noise_model = (
            NoiseModelRuntime.from_config(cfg.noise_model, device=self.device)
            if cfg.noise_model is not None
            else None
        )
        self.polarization_config = cfg.polarization
        self.polarization = (
            PolarizationRuntime.from_config(cfg.polarization, device=self.device, radar=self)
            if cfg.polarization is not None
            else None
        )
        self.receiver_chain_config = cfg.receiver_chain
        self.receiver_chain = (
            ReceiverChainRuntime.from_config(cfg.receiver_chain, device=self.device)
            if cfg.receiver_chain is not None
            else None
        )
        self._noise_generator = self._make_noise_generator()

        fs = cfg.sample_rate * 1e3
        slope_hz = cfg.slope * 1e12

        self.range_resolution = self.c0 * fs / (2 * slope_hz * cfg.adc_samples)
        self.max_range = self.c0 * fs / (2 * slope_hz)
        self.ranges = (
            torch.arange(0, cfg.num_range_bins // 2, dtype=torch.float64, device=self.device)
            * self.range_resolution
        )

        chirp_period = (cfg.idle_time + cfg.ramp_end_time) * 1e-6
        effective_period = chirp_period * cfg.num_tx
        self.doppler_resolution = self._lambda / (2 * cfg.num_doppler_bins * effective_period)
        self.max_doppler = self._lambda / (4 * chirp_period * cfg.num_tx)
        self.velocities = (
            torch.arange(
                -cfg.num_doppler_bins // 2,
                cfg.num_doppler_bins // 2,
                dtype=torch.float64,
                device=self.device,
            )
            * self.doppler_resolution
        )

        if self.backend == SolverBackend.PYTORCH:
            from .solvers.solver_pytorch import PytorchSolver

            self.solver = PytorchSolver(self)
        elif self.backend == SolverBackend.SLANG:
            from .solvers.solver_slang import SlangSolver

            self.solver = SlangSolver(self)
        else:
            from .solvers.solver_dirichlet import DirichletSolver

            self.solver = DirichletSolver(self, pad_factor)

    @staticmethod
    def _resolve_device(*, device: torch.device, backend: SolverBackend) -> torch.device:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Radar defaults to CUDA, but torch.cuda.is_available() is False. "
                "Pass device='cpu' only when using the PyTorch backend without CUDA."
            )
        if device.type != "cuda" and backend in {SolverBackend.SLANG, SolverBackend.DIRICHLET}:
            raise ValueError(f"Radar backend '{backend}' requires device='cuda'.")
        return device

    def _set_pose_fields(self, *, position, target, up, fov) -> None:
        position_t = vec3_tensor(position, name="Radar.position")
        target_t = _target_from_position(position_t) if target is None else vec3_tensor(target, name="Radar.target")
        up_t = vec3_tensor(up, name="Radar.up")
        forward = target_t - position_t
        if torch.linalg.norm(forward) <= 1e-12:
            raise ValueError("Radar.target must differ from Radar.position.")
        if torch.linalg.norm(up_t) <= 1e-12:
            raise ValueError("Radar.up must be non-zero.")
        if torch.linalg.norm(torch.cross(forward, up_t, dim=0)) <= 1e-12:
            raise ValueError("Radar.up must not be collinear with the viewing direction.")
        self.position = position_t
        self.target = target_t
        self.up = up_t
        self.fov = float(fov)

    def _refresh_pose_dependent_state(self) -> None:
        self.tx_pos = self.world_from_local_points(self.tx_loc).contiguous()
        self.rx_pos = self.world_from_local_points(self.rx_loc).contiguous()
        self.origin = self.position

    def _build_antenna_pattern_runtime(self, config: dict[str, Any]) -> None:
        self.antenna_pattern_kind = config["kind"]
        self.antenna_pattern_x_angles_deg = torch.tensor(config["x_angles_deg"], dtype=torch.float32, device=self.device)
        self.antenna_pattern_y_angles_deg = torch.tensor(config["y_angles_deg"], dtype=torch.float32, device=self.device)
        self.antenna_pattern_x_values = None
        self.antenna_pattern_y_values = None
        self.antenna_pattern_values = None
        if config["kind"] == "separable":
            self.antenna_pattern_x_values = torch.tensor(config["x_values"], dtype=torch.float32, device=self.device)
            self.antenna_pattern_y_values = torch.tensor(config["y_values"], dtype=torch.float32, device=self.device)
        else:
            self.antenna_pattern_values = torch.tensor(config["values"], dtype=torch.float32, device=self.device)

    def evaluate_antenna_pattern_xy(self, x_angles_deg: torch.Tensor, y_angles_deg: torch.Tensor) -> torch.Tensor:
        return evaluate_antenna_pattern_xy(
            self.antenna_pattern_kind,
            self.antenna_pattern_x_angles_deg,
            self.antenna_pattern_y_angles_deg,
            self.antenna_pattern_x_values,
            self.antenna_pattern_y_values,
            self.antenna_pattern_values,
            x_angles_deg,
            y_angles_deg,
        )

    def evaluate_antenna_pattern_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        return evaluate_antenna_pattern_vectors(
            self.antenna_pattern_kind,
            self.antenna_pattern_x_angles_deg,
            self.antenna_pattern_y_angles_deg,
            self.antenna_pattern_x_values,
            self.antenna_pattern_y_values,
            self.antenna_pattern_values,
            vectors,
        )

    def set_pose(self, *, position=None, target=None, up=None, fov=None) -> "Radar":
        """Mutate radar pose and refresh pose-dependent antenna state."""
        new_position = self.position if position is None else vec3_tensor(position, name="Radar.position")
        if target is None:
            target_t = self.target if position is None else new_position + (self.target - self.position)
        else:
            target_t = vec3_tensor(target, name="Radar.target")
        up_t = self.up if up is None else vec3_tensor(up, name="Radar.up")
        fov_value = self.fov if fov is None else float(fov)
        self._set_pose_fields(position=new_position, target=target_t, up=up_t, fov=fov_value)
        self._refresh_pose_dependent_state()
        if self.polarization_config is not None:
            self.polarization = PolarizationRuntime.from_config(
                self.polarization_config,
                device=self.device,
                radar=self,
            )
        return self

    def _world_from_local_matrix(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        position = self.position.to(device=device, dtype=dtype)
        target = self.target.to(device=device, dtype=dtype)
        up = self.up.to(device=device, dtype=dtype)

        forward = target - position
        forward = forward / torch.linalg.norm(forward)
        right = torch.cross(forward, up, dim=0)
        right = right / torch.linalg.norm(right)
        true_up = torch.cross(right, forward, dim=0)
        true_up = true_up / torch.linalg.norm(true_up)
        back = -forward
        world_from_local = torch.stack((right, true_up, back), dim=1)
        return position, world_from_local

    def world_from_local_points(self, points: torch.Tensor) -> torch.Tensor:
        position, world_from_local = self._world_from_local_matrix(device=points.device, dtype=points.dtype)
        return points @ world_from_local.transpose(0, 1) + position

    def world_from_local_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        _, world_from_local = self._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
        return vectors @ world_from_local.transpose(0, 1)

    def local_from_world_points(self, points: torch.Tensor) -> torch.Tensor:
        position, world_from_local = self._world_from_local_matrix(device=points.device, dtype=points.dtype)
        return (points - position) @ world_from_local

    def local_from_world_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        _, world_from_local = self._world_from_local_matrix(device=vectors.device, dtype=vectors.dtype)
        return vectors @ world_from_local

    def _make_noise_generator(self) -> torch.Generator | None:
        if self.config.noise_model is None or self.config.noise_model.get("seed") is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.noise_model["seed"])
        return generator

    def waveform(self, t, phi=0):
        """FMCW chirp waveform: exp(j * 2pi * (fc*t + 0.5*slope*t^2))."""
        phase = self.config.fc * t + 0.5 * (self.config.slope * 1e12) * t * t
        return torch.exp(1j * (2 * torch.pi * phase + phi))

    def apply_noise(self, signal: torch.Tensor) -> torch.Tensor:
        if self.noise_model is None:
            return signal
        return self.noise_model.apply(signal, generator=self._noise_generator)

    def apply_receiver_chain(self, signal: torch.Tensor) -> torch.Tensor:
        if self.receiver_chain is None:
            return signal
        return self.receiver_chain.apply(signal)

    def apply_signal_models(self, signal: torch.Tensor) -> torch.Tensor:
        modeled = self.apply_noise(signal)
        modeled = self.apply_receiver_chain(modeled)
        return modeled

    def chirp(self, distances, amplitudes):
        """Compute one chirp. distances/amplitudes: (N,) one-way range."""
        signal = self.solver.chirp(distances, amplitudes)
        return self.apply_signal_models(signal)

    def frame(self, interpolator, t0=0):
        """Single TX-RX frame. Returns: (chirps, adc_samples) complex."""
        signal = self.solver.frame(interpolator, t0)
        return self.apply_signal_models(signal)

    def mimo(self, interpolator, t0=0, **options):
        """Full MIMO data cube. Returns: (TX, RX, chirps, adc_samples) complex."""
        if bool(options.get("freq_domain", False)) and (
            self.noise_model is not None or self.receiver_chain is not None
        ):
            raise ValueError(
                "Radar noise_model and receiver_chain only support time-domain mimo output; "
                "omit freq_domain=True."
            )
        signal = self.solver.mimo(interpolator, t0, **options)
        return self.apply_signal_models(signal)

    def simulate(
        self,
        scene,
        *,
        resolution: int = 128,
        epsilon_r: float = 5.0,
        sampling: SamplingMode = "triangle",
        multipath: bool = False,
        max_reflections: int = 0,
        ray_batch_size: int = 65536,
        t0: float = 0.0,
        motion_sampling: MotionSampling = "per_chirp",
    ):
        """Run ray tracing plus MIMO signal generation for one scene."""
        from .trace import Tracer
        from .scene import Scene, SceneModule

        if isinstance(scene, SceneModule):
            scene = scene.to_scene()
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a radar.Scene or radar.SceneModule.")

        sampling = SamplingMode(sampling)
        motion_sampling = MotionSampling(motion_sampling)
        if max_reflections < 0:
            raise ValueError("max_reflections must be >= 0.")
        if ray_batch_size <= 0:
            raise ValueError("ray_batch_size must be > 0.")
        if multipath and sampling != SamplingMode.PIXEL:
            raise ValueError("multipath=True requires sampling='pixel'.")

        tracer = Tracer(
            scene,
            self,
            resolution=resolution,
            epsilon_r=epsilon_r,
            sampling=sampling,
            multipath=multipath,
            max_reflections=max_reflections,
            ray_batch_size=ray_batch_size,
        )
        t0 = float(t0)
        trace = tracer.trace(time=t0) if scene.has_motion else tracer.trace()

        if scene.has_motion and motion_sampling == MotionSampling.PER_CHIRP:
            def interpolator(t):
                return tracer.trace(time=t)
        else:
            def interpolator(t):
                del t
                return trace

        signal = self.mimo(interpolator, t0)
        self.last_scene = scene
        self.last_tracer = tracer
        self.last_trace = trace
        return signal

    @classmethod
    def simulate_group(
        cls,
        scene,
        *,
        radars: Mapping[str, "Radar"] | Sequence["Radar"],
        resolution: int = 128,
        epsilon_r: float = 5.0,
        sampling: SamplingMode = "triangle",
        multipath: bool = False,
        max_reflections: int = 0,
        ray_batch_size: int = 65536,
        t0: float = 0.0,
        motion_sampling: MotionSampling = "per_chirp",
    ) -> dict[str, torch.Tensor]:
        """Run the same scene for multiple named Radar instances."""
        if isinstance(radars, Mapping):
            items = tuple((str(name), radar) for name, radar in radars.items())
        else:
            items = tuple((radar.name, radar) for radar in radars)
            missing = [index for index, (name, _) in enumerate(items) if not name]
            if missing:
                raise ValueError(
                    "Radar.simulate_group requires names for sequence entries; "
                    "pass a mapping or set Radar(name=...)."
                )

        if not items:
            raise ValueError("Radar.simulate_group requires at least one radar.")
        names = [name for name, _ in items]
        if len(names) != len(set(names)):
            raise ValueError("Radar.simulate_group radar names must be unique.")
        for name, radar in items:
            if not isinstance(radar, cls):
                raise TypeError(f"radars['{name}'] must be a Radar instance.")

        return {
            name: radar.simulate(
                scene,
                resolution=resolution,
                epsilon_r=epsilon_r,
                sampling=sampling,
                multipath=multipath,
                max_reflections=max_reflections,
                ray_batch_size=ray_batch_size,
                t0=t0,
                motion_sampling=motion_sampling,
            )
            for name, radar in items
        }
