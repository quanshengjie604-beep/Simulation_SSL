#!/usr/bin/env python3
"""Generate micro-Doppler directly from an AMASS SMPL-X motion sequence."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import smplx
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMASS = (
    REPO_ROOT
    / "datasets"
    / "AMASS_SMPLX_2022"
    / "BMLmovi"
    / "Subject_11_F_MoSh"
    / "Subject_11_F_5_stageii.npz"
)
LIGHT_SPEED_MPS = 299_792_458.0
RADAR_PROFILE = "single77_25fps_3344hz"
CHIRP_FREQUENCY_HZ = 3344.0
RADAR_PROFILES = {
    "single77_25fps": 3200.0,
    "single77_25fps_3344hz": CHIRP_FREQUENCY_HZ,
    "mmradpose_single60_15fps": 4000.0,
}
FRAME_RATE_HZ = 25.0
CHIRPS_PER_FRAME = 128
ADC_SAMPLES = 128
RANGE_WINDOW_M = 1.5
STFT_WINDOW_CHIRPS = 128
STFT_FFT_SIZE = 128
STFT_OVERLAP = 0.875
STFT_HOP_CHIRPS = 16
SPECULAR_ETA = 0.5
PROFILE_FREQUENCY_START_HZ = 77.1e9
PROFILE_FREQUENCY_STOP_HZ = 78.1e9
PROFILE_ADC_DUTY_CYCLE = 0.5
PROFILE_RANGE_RESOLUTION_M: float | None = None


@dataclass(frozen=True)
class RadarPose:
    position: tuple[float, float, float] = (0.0, 1.0, 0.0)
    target: tuple[float, float, float] = (0.0, 1.0, -1.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-npz", default=str(DEFAULT_AMASS))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "smpl_models"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "amass_smplx_micro_doppler"))
    parser.add_argument("--sequence", default="", help="Output label; default derives it from the AMASS path.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, for example cuda:0.")
    parser.add_argument("--backend", choices=("dirichlet", "pytorch", "slang"), default="dirichlet")
    parser.add_argument("--radar-profile", choices=tuple(RADAR_PROFILES), default=RADAR_PROFILE)
    parser.add_argument(
        "--tx-rx-lateral-separation-m",
        type=float,
        default=0.0,
        help="SISO TX/RX lateral baseline in meters; TX=-baseline/2, RX=+baseline/2.",
    )
    parser.add_argument(
        "--save-frame-doppler-time",
        action="store_true",
        help="Also save a per-radar-frame Doppler-time spectrum, matching mmRadPose-style processing.",
    )
    parser.add_argument("--range-bin-mode", choices=("fixed", "tracked"), default="fixed")
    parser.add_argument(
        "--visibility-mode",
        choices=("linear", "hold"),
        default="linear",
        help="Interpolate binary visibility between radar-frame traces or hold it for each frame.",
    )
    parser.add_argument(
        "--doppler-transform",
        choices=("fft", "nudft"),
        default="fft",
        help="Use actual chirp timestamps (nudft) or assume uniform slow-time sampling (fft).",
    )
    parser.add_argument("--smpl-batch-size", type=int, default=32)
    parser.add_argument("--subject-range", type=float, default=4.0)
    parser.add_argument("--subject-lateral", type=float, default=0.0)
    parser.add_argument("--radar-height", type=float, default=1.0)
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument(
        "--max-radar-frames",
        type=int,
        default=0,
        help="Limit complete radar frames for diagnostics; 0 processes the full motion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sequence_label(path: Path) -> str:
    return f"{path.parent.name}__{path.stem}"


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        raise ValueError(f"--device must be a CUDA device, got {args.device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False")
    if args.smpl_batch_size < 1:
        raise ValueError("--smpl-batch-size must be positive")
    if args.max_radar_frames < 0:
        raise ValueError("--max-radar-frames cannot be negative")
    if args.tx_rx_lateral_separation_m < 0.0:
        raise ValueError("--tx-rx-lateral-separation-m cannot be negative")


def apply_radar_profile(profile: str) -> None:
    global CHIRP_FREQUENCY_HZ
    global FRAME_RATE_HZ
    global CHIRPS_PER_FRAME
    global ADC_SAMPLES
    global RANGE_WINDOW_M
    global STFT_WINDOW_CHIRPS
    global STFT_FFT_SIZE
    global STFT_HOP_CHIRPS
    global STFT_OVERLAP
    global PROFILE_FREQUENCY_START_HZ
    global PROFILE_FREQUENCY_STOP_HZ
    global PROFILE_ADC_DUTY_CYCLE
    global PROFILE_RANGE_RESOLUTION_M

    CHIRP_FREQUENCY_HZ = RADAR_PROFILES[profile]
    CHIRPS_PER_FRAME = 128
    STFT_WINDOW_CHIRPS = 128
    STFT_FFT_SIZE = 128
    STFT_HOP_CHIRPS = 16
    STFT_OVERLAP = 0.875

    if profile == "mmradpose_single60_15fps":
        FRAME_RATE_HZ = 15.0
        ADC_SAMPLES = 64
        RANGE_WINDOW_M = 2.0
        PROFILE_FREQUENCY_START_HZ = 60.0e9
        PROFILE_FREQUENCY_STOP_HZ = 61.02e9
        PROFILE_ADC_DUTY_CYCLE = (64 / 3.8e6) * CHIRP_FREQUENCY_HZ
        PROFILE_RANGE_RESOLUTION_M = 0.148
        return

    FRAME_RATE_HZ = 25.0
    ADC_SAMPLES = 128
    RANGE_WINDOW_M = 1.5
    PROFILE_FREQUENCY_START_HZ = 77.1e9
    PROFILE_FREQUENCY_STOP_HZ = 78.1e9
    PROFILE_ADC_DUTY_CYCLE = 0.5
    PROFILE_RANGE_RESOLUTION_M = None


def resolve_witwin_radar_dir() -> Path:
    candidates = []
    if os.environ.get("WITWIN_RADAR_DIR"):
        candidates.append(Path(os.environ["WITWIN_RADAR_DIR"]))
    candidates.extend(
        [
            Path(sys.prefix) / "lib/python3.11/site-packages/witwin/radar",
            Path("/bigdata/users/quansj/miniforge3/envs/witwin/lib/python3.11/site-packages/witwin/radar"),
            Path("/home/quansj/miniforge3/envs/witwin/lib/python3.11/site-packages/witwin/radar"),
        ]
    )
    for candidate in candidates:
        if (candidate / "radar.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate the installed WiTwin radar package")


def bootstrap_witwin_modules():
    cache = Path(os.environ.get("WITWIN_CACHE_DIR", "/tmp/witwin-cache"))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("DRJIT_CACHE_DIR", str(cache))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))

    package = types.ModuleType("witwin.radar")
    package.__path__ = [str(resolve_witwin_radar_dir())]
    sys.modules.setdefault("witwin.radar", package)
    from witwin.radar.radar import Radar, RadarConfig
    from witwin.radar.scene import Scene
    from witwin.radar.trace import Tracer

    return Radar, RadarConfig, Scene, Tracer


def apply_radar_equation_patch() -> None:
    """Use bistatic amplitude loss proportional to 1/(R_tx * R_rx)."""
    from witwin.radar.solvers import common

    def compute_path_amplitudes(
        radar,
        sample: common.PathSample,
        total_path_lengths: torch.Tensor,
        *,
        tx_pos: torch.Tensor | None = None,
        rx_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del total_path_lengths
        tx_pos = radar.tx_pos if tx_pos is None else tx_pos
        rx_pos = radar.rx_pos if rx_pos is None else rx_pos
        dist_tx = torch.cdist(sample.entry_points, tx_pos).transpose(0, 1).unsqueeze(1)
        dist_rx = torch.cdist(sample.points, rx_pos).transpose(0, 1).unsqueeze(0)
        fspl_amp = radar._lambda / (4.0 * math.pi * torch.clamp(dist_tx * dist_rx, min=1e-6))
        scatter_power = torch.clamp(sample.intensities, min=0.0).view(1, 1, -1)
        pattern_gains = common.compute_antenna_pattern_gains(radar, sample, tx_pos, rx_pos)
        if pattern_gains is not None:
            scatter_power = scatter_power * torch.clamp(pattern_gains, min=0.0)
        amplitudes = radar.gain * torch.sqrt(scatter_power) * fspl_amp
        polarization = common.compute_polarization_amplitudes(radar, sample)
        return amplitudes if polarization is None else amplitudes * polarization

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


def radar_config(
    chirp_frequency_hz: float = CHIRP_FREQUENCY_HZ,
    tx_rx_lateral_separation_m: float = 0.0,
) -> dict[str, object]:
    chirp_period_us = 1e6 / chirp_frequency_hz
    antenna_spacing_m = LIGHT_SPEED_MPS / PROFILE_FREQUENCY_START_HZ / 2.0
    half_baseline_hw = 0.5 * float(tx_rx_lateral_separation_m) / antenna_spacing_m
    tx_loc = [(-half_baseline_hw, 0.0, 0.0)]
    rx_loc = [(half_baseline_hw, 0.0, 0.0)]
    if PROFILE_RANGE_RESOLUTION_M is not None:
        ramp_end_us = ADC_SAMPLES / 3.8e6 * 1e6
        slope_mhz_per_us = (
            LIGHT_SPEED_MPS * 3.8e6 / (2.0 * PROFILE_RANGE_RESOLUTION_M * ADC_SAMPLES)
        ) / 1e12
        return {
            "num_tx": 1,
            "num_rx": 1,
            "fc": PROFILE_FREQUENCY_START_HZ,
            "slope": slope_mhz_per_us,
            "adc_samples": ADC_SAMPLES,
            "adc_start_time": 0,
            "sample_rate": 3.8e6 / 1e3,
            "idle_time": chirp_period_us - ramp_end_us,
            "ramp_end_time": ramp_end_us,
            "chirp_per_frame": CHIRPS_PER_FRAME,
            "frame_per_second": FRAME_RATE_HZ,
            "num_doppler_bins": STFT_FFT_SIZE,
            "num_range_bins": ADC_SAMPLES,
            "num_angle_bins": 1,
            "power": 0,
            "tx_loc": tx_loc,
            "rx_loc": rx_loc,
        }

    return {
        "num_tx": 1,
        "num_rx": 1,
        "fc": PROFILE_FREQUENCY_START_HZ,
        "slope": 1e3 / chirp_period_us,
        "adc_samples": ADC_SAMPLES,
        "adc_start_time": 0,
        "sample_rate": ADC_SAMPLES * 1e3 / (PROFILE_ADC_DUTY_CYCLE * chirp_period_us),
        "idle_time": 0,
        "ramp_end_time": chirp_period_us,
        "chirp_per_frame": CHIRPS_PER_FRAME,
        "frame_per_second": FRAME_RATE_HZ,
        "num_doppler_bins": STFT_FFT_SIZE,
        "num_range_bins": ADC_SAMPLES,
        "num_angle_bins": 1,
        "power": 0,
        "tx_loc": tx_loc,
        "rx_loc": rx_loc,
    }


def derived_radar_parameters(config: dict[str, object]) -> dict[str, float]:
    chirp_period_us = float(config["idle_time"]) + float(config["ramp_end_time"])
    sample_rate_hz = float(config["sample_rate"]) * 1e3
    slope_hz_per_s = float(config["slope"]) * 1e12
    bandwidth_hz = float(config["slope"]) * float(config["ramp_end_time"]) * 1e6
    center_frequency_hz = float(config["fc"]) + 0.5 * bandwidth_hz
    range_resolution_m = LIGHT_SPEED_MPS * sample_rate_hz / (
        2.0 * slope_hz_per_s * int(config["adc_samples"])
    )
    doppler_resolution_mps = LIGHT_SPEED_MPS / center_frequency_hz / (
        2.0 * int(config["num_doppler_bins"]) * chirp_period_us * 1e-6 * int(config["num_tx"])
    )
    return {
        "chirp_frequency_hz": 1e6 / chirp_period_us,
        "frame_rate_hz": float(config["frame_per_second"]),
        "range_resolution_m": range_resolution_m,
        "doppler_resolution_mps": doppler_resolution_mps,
        "center_frequency_hz": center_frequency_hz,
        "wavelength_m": LIGHT_SPEED_MPS / center_frequency_hz,
    }


def load_amass(path: Path) -> dict[str, np.ndarray | str | float]:
    with np.load(path, allow_pickle=True) as data:
        required = {"root_orient", "pose_body", "pose_hand", "trans", "betas", "mocap_frame_rate"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} is missing AMASS arrays: {sorted(missing)}")
        return {
            "gender": str(data["gender"].item()).lower() if "gender" in data.files else "neutral",
            "mocap_frame_rate": float(data["mocap_frame_rate"]),
            "root_orient": data["root_orient"].astype(np.float32),
            "pose_body": data["pose_body"].astype(np.float32),
            "pose_hand": data["pose_hand"].astype(np.float32),
            "pose_jaw": data["pose_jaw"].astype(np.float32) if "pose_jaw" in data.files else None,
            "pose_eye": data["pose_eye"].astype(np.float32) if "pose_eye" in data.files else None,
            "trans": data["trans"].astype(np.float32),
            "betas": data["betas"].astype(np.float32),
        }


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def padded_frame_ids(start: int, stop: int, batch_size: int) -> tuple[np.ndarray, int]:
    frame_ids = np.arange(start, stop, dtype=np.int64)
    valid = frame_ids.size
    if valid < batch_size:
        frame_ids = np.pad(frame_ids, (0, batch_size - valid), constant_values=int(frame_ids[-1]))
    return frame_ids, valid


def generate_native_meshes(
    motion: dict[str, np.ndarray | str | float],
    model_dir: Path,
    device: torch.device,
    batch_size: int,
    return_joints: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = int(motion["trans"].shape[0])  # type: ignore[union-attr]
    gender = str(motion["gender"])
    if gender not in {"neutral", "male", "female"}:
        gender = "neutral"
    model = smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=gender,
        use_pca=False,
        num_betas=16,
        batch_size=batch_size,
    ).to(device)
    model.eval()
    faces = model.faces.astype(np.int32)
    batches = []
    joint_batches = []
    betas = np.asarray(motion["betas"], dtype=np.float32)[:16]
    if betas.size < 16:
        betas = np.pad(betas, (0, 16 - betas.size))

    for start in range(0, num_frames, batch_size):
        stop = min(start + batch_size, num_frames)
        ids, valid = padded_frame_ids(start, stop, batch_size)
        pose_hand = motion["pose_hand"][ids]  # type: ignore[index]
        pose_jaw = motion["pose_jaw"]
        pose_eye = motion["pose_eye"]
        inputs = {
            "global_orient": tensor(motion["root_orient"][ids], device),  # type: ignore[index]
            "body_pose": tensor(motion["pose_body"][ids], device),  # type: ignore[index]
            "left_hand_pose": tensor(pose_hand[:, :45], device),
            "right_hand_pose": tensor(pose_hand[:, 45:90], device),
            "jaw_pose": tensor(
                pose_jaw[ids] if pose_jaw is not None else np.zeros((batch_size, 3), np.float32),  # type: ignore[index]
                device,
            ),
            "leye_pose": tensor(
                pose_eye[ids, :3] if pose_eye is not None else np.zeros((batch_size, 3), np.float32),  # type: ignore[index]
                device,
            ),
            "reye_pose": tensor(
                pose_eye[ids, 3:6] if pose_eye is not None else np.zeros((batch_size, 3), np.float32),  # type: ignore[index]
                device,
            ),
            "betas": tensor(np.repeat(betas[None], batch_size, axis=0), device),
            "expression": torch.zeros(batch_size, 10, dtype=torch.float32, device=device),
            "transl": tensor(motion["trans"][ids], device),  # type: ignore[index]
        }
        with torch.no_grad():
            output = model(**inputs)
            vertices = output.vertices[:valid].detach().cpu().numpy().astype(np.float32)
        batches.append(vertices)
        if return_joints:
            joint_batches.append(output.joints[:valid, :22].detach().cpu().numpy().astype(np.float32))
        print(f"[smplx {stop:04d}/{num_frames:04d}] generated native meshes", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    vertices = np.concatenate(batches, axis=0)
    if return_joints:
        return vertices, faces, np.concatenate(joint_batches, axis=0)
    return vertices, faces


def place_like_gif(
    vertices: np.ndarray,
    subject_range: float,
    subject_lateral: float,
) -> np.ndarray:
    """Place AMASS z-up vertices in WiTwin world coordinates used by the GIF."""
    world = np.empty_like(vertices)
    world[..., 0] = vertices[..., 1]
    world[..., 1] = vertices[..., 2]
    world[..., 2] = -vertices[..., 0]
    first = world[0]
    first_center = 0.5 * (first.min(axis=0) + first.max(axis=0))
    first_ground_y = float(first[:, 1].min())
    world[..., 0] += float(subject_lateral) - float(first_center[0])
    world[..., 1] -= first_ground_y
    world[..., 2] += -float(subject_range) - float(first_center[2])
    return world


def faces_for_witwin_world(faces: np.ndarray) -> np.ndarray:
    """Reverse winding after the handedness-changing AMASS-to-WiTwin transform."""
    return np.ascontiguousarray(faces[:, [0, 2, 1]])


def actual_radar_timing(
    duration_s: float,
    chirp_frequency_hz: float = CHIRP_FREQUENCY_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    last_chirp_offset = (CHIRPS_PER_FRAME - 1) / chirp_frequency_hz
    if duration_s < last_chirp_offset:
        raise ValueError("Motion is shorter than one complete radar frame")
    num_frames = int(math.floor((duration_s - last_chirp_offset) * FRAME_RATE_HZ + 1e-9)) + 1
    frame_times = np.arange(num_frames, dtype=np.float64) / FRAME_RATE_HZ
    chirp_times = frame_times[:, None] + np.arange(CHIRPS_PER_FRAME, dtype=np.float64)[None, :] / chirp_frequency_hz
    return frame_times, chirp_times


def gaussian_specular_weight(
    points: torch.Tensor,
    normals: torch.Tensor,
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
) -> torch.Tensor:
    incident = points - tx_position[None, :]
    incident = incident / torch.clamp(torch.linalg.norm(incident, dim=1, keepdim=True), min=1e-6)
    outgoing = rx_position[None, :] - points
    outgoing = outgoing / torch.clamp(torch.linalg.norm(outgoing, dim=1, keepdim=True), min=1e-6)
    reflected = incident - 2.0 * torch.sum(incident * normals, dim=1, keepdim=True) * normals
    reflected = reflected / torch.clamp(torch.linalg.norm(reflected, dim=1, keepdim=True), min=1e-6)
    angle_error = torch.acos(torch.clamp(torch.sum(reflected * outgoing, dim=1), min=0.0, max=1.0))
    return torch.exp(-(angle_error**2) / (2.0 * SPECULAR_ETA**2))


def gaussian_normal_weight(points: torch.Tensor, normals: torch.Tensor, radar_position: torch.Tensor) -> torch.Tensor:
    return gaussian_specular_weight(points, normals, radar_position, radar_position)


def update_dynamic_mesh_vertices(scene, name: str, vertices: torch.Tensor) -> None:
    """Update fixed-topology mesh vertices without rebuilding Mesh topology statistics."""
    for structure in scene.structures:
        if structure.name != name:
            continue
        geometry = structure.geometry
        vertex_buffer = getattr(geometry, "_vertices_tensor", None)
        if vertex_buffer is None:
            raise TypeError(f"Structure {name!r} is not an explicit WiTwin Mesh")
        if vertex_buffer.shape != vertices.shape:
            raise ValueError(f"Vertex shape changed from {vertex_buffer.shape} to {vertices.shape}")
        vertex_buffer.copy_(vertices)
        scene._set_dirty(scene.DIRTY_VERTICES)
        return
    raise KeyError(f"Structure {name!r} not found")


class FrameLevelScattererInterpolator:
    """Interpolate mesh motion per chirp and visibility between radar-frame traces."""

    def __init__(
        self,
        radar,
        scene,
        tracer,
        native_vertices: torch.Tensor,
        faces: np.ndarray,
        mocap_frame_rate: float,
        duration_s: float,
    ) -> None:
        from witwin.radar.trace import TraceResult

        self.radar = radar
        self.scene = scene
        self.tracer = tracer
        self.native_vertices = native_vertices
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=native_vertices.device)
        self.mocap_frame_rate = float(mocap_frame_rate)
        self.duration_s = float(duration_s)
        self.radar_position = radar.position.to(device=radar.device, dtype=torch.float32)
        self.tx_position = radar.tx_pos[0].to(device=radar.device, dtype=torch.float32)
        self.rx_position = radar.rx_pos[0].to(device=radar.device, dtype=torch.float32)
        self.last_native = native_vertices.shape[0] - 1
        self.trace_result_type = TraceResult
        self.visible_ids = torch.empty(0, dtype=torch.long, device=native_vertices.device)
        self.frame_visible_counts: list[int] = []
        self.frame_next_visible_counts: list[int] = []
        self.frame_union_visible_counts: list[int] = []
        self.visibility_start: torch.Tensor | None = None
        self.visibility_end: torch.Tensor | None = None
        self.visibility_start_time_s = 0.0
        self.visibility_end_time_s = 0.0
        self._cached_trace_time_s: float | None = None
        self._cached_visible_ids = torch.empty(0, dtype=torch.long, device=native_vertices.device)
        self.chirp_calls = 0

    def vertices_at(self, time_s: float) -> torch.Tensor:
        position = float(np.clip(time_s, 0.0, self.duration_s)) * self.mocap_frame_rate
        low = min(int(math.floor(position)), self.last_native)
        high = min(low + 1, self.last_native)
        alpha = position - low
        return torch.lerp(self.native_vertices[low], self.native_vertices[high], alpha).contiguous()

    def trace_visible_ids(self, frame_time_s: float) -> torch.Tensor:
        if self._cached_trace_time_s is not None and math.isclose(
            frame_time_s,
            self._cached_trace_time_s,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return self._cached_visible_ids
        update_dynamic_mesh_vertices(self.scene, "human", self.vertices_at(frame_time_s))
        with torch.no_grad():
            trace = self.tracer.trace()
        if trace._tri_indices is None:
            raise RuntimeError("WiTwin frame-level triangle trace did not return triangle indices")
        visible_ids = torch.unique(trace._tri_indices.to(dtype=torch.long), sorted=True)
        self._cached_trace_time_s = float(frame_time_s)
        self._cached_visible_ids = visible_ids
        return visible_ids

    def prepare_frame(self, frame_time_s: float) -> int:
        self.visible_ids = self.trace_visible_ids(frame_time_s)
        self.visibility_start = None
        self.visibility_end = None
        count = int(self.visible_ids.numel())
        self.frame_visible_counts.append(count)
        self.chirp_calls = 0
        return count

    def prepare_interpolated_frame(self, frame_time_s: float, next_frame_time_s: float) -> int:
        current_ids = self.trace_visible_ids(frame_time_s)
        next_ids = self.trace_visible_ids(next_frame_time_s)
        start_full = torch.zeros(self.faces.shape[0], dtype=torch.float32, device=self.native_vertices.device)
        end_full = torch.zeros_like(start_full)
        start_full[current_ids] = 1.0
        end_full[next_ids] = 1.0
        self.visible_ids = torch.nonzero((start_full + end_full) > 0.0, as_tuple=False).squeeze(1)
        self.visibility_start = start_full[self.visible_ids]
        self.visibility_end = end_full[self.visible_ids]
        self.visibility_start_time_s = float(frame_time_s)
        self.visibility_end_time_s = float(next_frame_time_s)
        current_count = int(current_ids.numel())
        next_count = int(next_ids.numel())
        union_count = int(self.visible_ids.numel())
        self.frame_visible_counts.append(current_count)
        self.frame_next_visible_counts.append(next_count)
        self.frame_union_visible_counts.append(union_count)
        self.chirp_calls = 0
        return union_count

    def __call__(self, time_s: float):
        self.chirp_calls += 1
        vertices = self.vertices_at(float(time_s))
        triangles = vertices[self.faces[self.visible_ids]]
        points = triangles.mean(dim=1)
        cross = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        twice_area = torch.linalg.norm(cross, dim=1)
        normals = cross / torch.clamp(twice_area[:, None], min=1e-10)
        triangle_area = 0.5 * twice_area
        intensities = triangle_area * gaussian_specular_weight(points, normals, self.tx_position, self.rx_position)
        if self.visibility_start is not None and self.visibility_end is not None:
            duration = max(self.visibility_end_time_s - self.visibility_start_time_s, 1e-12)
            alpha = float(np.clip((float(time_s) - self.visibility_start_time_s) / duration, 0.0, 1.0))
            visibility = torch.lerp(self.visibility_start, self.visibility_end, alpha)
            intensities = intensities * visibility
        return self.trace_result_type(points, intensities, self.visible_ids, normals=normals)


def range_fft(adc: np.ndarray) -> np.ndarray:
    fast_time = adc - adc.mean(axis=0, keepdims=True)
    window = np.hanning(adc.shape[0]).astype(np.float32)
    transformed = np.fft.fft(fast_time * window[:, None, None, None], axis=0)
    return transformed[: adc.shape[0] // 2].astype(np.complex64, copy=False)


def centered_range_indices(center: int, num_ranges: int, count: int) -> np.ndarray:
    count = min(max(1, int(count)), num_ranges - 1)
    start = min(max(1, int(center) - count // 2), num_ranges - count)
    return np.arange(start, start + count, dtype=np.int64)


def strong_reflection_points(
    transformed: np.ndarray,
    range_resolution_m: float,
    fixed_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    range_power = np.mean(np.abs(transformed) ** 2, axis=(1, 2, 3))
    center = 1 + int(np.argmax(range_power[1:]))
    count = max(1, int(round(RANGE_WINDOW_M / range_resolution_m)))
    indices = (
        centered_range_indices(center, transformed.shape[0], count)
        if fixed_indices is None
        else fixed_indices
    )
    points = transformed[indices].transpose(1, 0, 2, 3).reshape(transformed.shape[1], -1)
    return points.astype(np.complex64, copy=False), center * range_resolution_m, indices


def simulate_slow_time(
    radar,
    interpolator: FrameLevelScattererInterpolator,
    frame_times: np.ndarray,
    chirp_times_by_frame: np.ndarray,
    derived: dict[str, float],
    fixed_range_bins: np.ndarray | None = None,
    visibility_mode: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    signal_blocks = []
    strong_ranges = []
    selected_bins = []
    for index, frame_time in enumerate(frame_times, start=1):
        if visibility_mode == "linear":
            next_frame_time = float(frame_time) + 1.0 / FRAME_RATE_HZ
            visible_count = interpolator.prepare_interpolated_frame(float(frame_time), next_frame_time)
            visibility_text = (
                f"visible_start={interpolator.frame_visible_counts[-1]} "
                f"visible_end={interpolator.frame_next_visible_counts[-1]} union={visible_count}"
            )
        elif visibility_mode == "hold":
            visible_count = interpolator.prepare_frame(float(frame_time))
            visibility_text = f"frame_visible={visible_count}"
        else:
            raise ValueError(f"Unsupported visibility mode: {visibility_mode}")
        with torch.no_grad():
            mimo = radar.mimo(interpolator, t0=float(frame_time))
        if interpolator.chirp_calls != CHIRPS_PER_FRAME:
            raise RuntimeError(
                f"Frame {index - 1} evaluated {interpolator.chirp_calls} chirps, expected {CHIRPS_PER_FRAME}"
            )
        frame = mimo.detach().cpu().numpy()
        expected = (1, 1, CHIRPS_PER_FRAME, ADC_SAMPLES)
        if frame.shape != expected:
            raise ValueError(f"WiTwin returned {frame.shape}; expected {expected}")
        if not np.isfinite(frame.real).all() or not np.isfinite(frame.imag).all():
            raise FloatingPointError(f"Non-finite MIMO output at radar frame {index - 1}")
        adc = np.transpose(frame, (3, 2, 1, 0)).astype(np.complex64, copy=False)
        points, strong_range, bins = strong_reflection_points(
            range_fft(adc),
            derived["range_resolution_m"],
            fixed_range_bins,
        )
        signal_blocks.append(points)
        strong_ranges.append(strong_range)
        selected_bins.append([int(value) for value in bins])
        print(
            f"[radar {index:04d}/{len(frame_times):04d}] t={frame_time:.3f}s "
            f"{visibility_text} peak_range={strong_range:.3f}m bins={bins[0]}..{bins[-1]}",
            flush=True,
        )
    return (
        np.concatenate(signal_blocks, axis=0),
        chirp_times_by_frame.reshape(-1),
        np.asarray(strong_ranges, dtype=np.float32),
        selected_bins,
    )


def mean_power_stft(
    slow_time_points: np.ndarray,
    chirp_times: np.ndarray,
    wavelength_m: float,
    chirp_frequency_hz: float = CHIRP_FREQUENCY_HZ,
    transform: str = "nudft",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_windows = 1 + (slow_time_points.shape[0] - STFT_WINDOW_CHIRPS) // STFT_HOP_CHIRPS
    spectrum = np.empty((STFT_FFT_SIZE, num_windows), dtype=np.float32)
    center_times = np.empty(num_windows, dtype=np.float64)
    window = np.hamming(STFT_WINDOW_CHIRPS).astype(np.float32)
    window_energy = float(np.sum(window * window))
    frequencies = np.fft.fftshift(np.fft.fftfreq(STFT_FFT_SIZE, d=1.0 / chirp_frequency_hz))
    nudft_kernels: dict[int, np.ndarray] = {}
    for column in range(num_windows):
        start = column * STFT_HOP_CHIRPS
        stop = start + STFT_WINDOW_CHIRPS
        windowed = slow_time_points[start:stop] * window[:, None]
        if transform == "nudft":
            pattern = start % CHIRPS_PER_FRAME
            kernel = nudft_kernels.get(pattern)
            if kernel is None:
                times = chirp_times[start:stop]
                relative_times = times - float(np.mean(times))
                kernel = np.exp(-2j * np.pi * frequencies[:, None] * relative_times[None, :]).astype(
                    np.complex64
                )
                nudft_kernels[pattern] = kernel
            transformed = kernel @ windowed
        elif transform == "fft":
            transformed = np.fft.fft(windowed, n=STFT_FFT_SIZE, axis=0)
            transformed = np.fft.fftshift(transformed, axes=0)
        else:
            raise ValueError(f"Unsupported Doppler transform: {transform}")
        spectrum[:, column] = np.mean(np.abs(transformed) ** 2, axis=1) / window_energy
        center_times[column] = float(np.mean(chirp_times[start:stop]))
    velocities = frequencies * wavelength_m / 2.0
    return spectrum, center_times, velocities.astype(np.float64)


def mean_power_frame_doppler_time(
    slow_time_points: np.ndarray,
    frame_times: np.ndarray,
    wavelength_m: float,
    chirp_frequency_hz: float = CHIRP_FREQUENCY_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = slow_time_points.reshape(frame_times.shape[0], CHIRPS_PER_FRAME, -1)
    blocks = blocks - blocks.mean(axis=1, keepdims=True)
    transformed = np.fft.fftshift(np.fft.fft(blocks, n=CHIRPS_PER_FRAME, axis=1), axes=1)
    spectrum = np.mean(np.abs(transformed) ** 2, axis=2).T.astype(np.float32, copy=False)
    frequencies = np.fft.fftshift(np.fft.fftfreq(CHIRPS_PER_FRAME, d=1.0 / chirp_frequency_hz))
    velocities = frequencies * wavelength_m / 2.0
    return spectrum, frame_times.astype(np.float64), velocities.astype(np.float64)


def centers_to_edges(values: np.ndarray, single_width: float) -> np.ndarray:
    if values.size == 1:
        return np.asarray([values[0] - single_width / 2.0, values[0] + single_width / 2.0])
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        ([values[0] - (midpoints[0] - values[0])], midpoints, [values[-1] + (values[-1] - midpoints[-1])])
    )


def fixed_range_center_m(subject_range: float, subject_lateral: float, tx_rx_lateral_separation_m: float) -> float:
    half_baseline = 0.5 * float(tx_rx_lateral_separation_m)
    tx_range = math.hypot(float(subject_range), float(subject_lateral) + half_baseline)
    rx_range = math.hypot(float(subject_range), float(subject_lateral) - half_baseline)
    return 0.5 * (tx_range + rx_range)


def save_plot(
    path: Path,
    spectrum: np.ndarray,
    times: np.ndarray,
    velocities: np.ndarray,
    sequence: str,
    db_floor: float,
    chirp_frequency_hz: float = CHIRP_FREQUENCY_HZ,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smpl-micro-doppler")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak = max(float(np.max(spectrum)), np.finfo(np.float32).tiny)
    relative_db = 10.0 * np.log10(np.maximum(spectrum / peak, 1e-12))
    relative_db = np.clip(relative_db, float(db_floor), 0.0)
    time_edges = centers_to_edges(times, STFT_HOP_CHIRPS / chirp_frequency_hz)
    velocity_edges = centers_to_edges(velocities, float(np.mean(np.diff(velocities))))
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=170, constrained_layout=True)
    image = ax.pcolormesh(time_edges, velocity_edges, relative_db, shading="flat", cmap="turbo", vmin=db_floor, vmax=0.0)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Relative mean reflection power (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Radial velocity (m/s)")
    ax.set_title(f"{sequence} micro-Doppler")
    ax.set_xlim(time_edges[0], time_edges[-1])
    ax.set_ylim(velocity_edges[0], velocity_edges[-1])
    fig.savefig(path)
    plt.close(fig)


def save_doppler_time_plot(
    path: Path,
    spectrum: np.ndarray,
    times: np.ndarray,
    velocities: np.ndarray,
    sequence: str,
    db_floor: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smpl-micro-doppler")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak = max(float(np.max(spectrum)), np.finfo(np.float32).tiny)
    relative_db = 10.0 * np.log10(np.maximum(spectrum / peak, 1e-12))
    relative_db = np.clip(relative_db, float(db_floor), 0.0)
    time_edges = centers_to_edges(times, 1.0 / FRAME_RATE_HZ)
    velocity_edges = centers_to_edges(velocities, float(np.mean(np.diff(velocities))))
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=170, constrained_layout=True)
    image = ax.pcolormesh(time_edges, velocity_edges, relative_db, shading="flat", cmap="turbo", vmin=db_floor, vmax=0.0)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Relative mean reflection power (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Radial velocity (m/s)")
    ax.set_title(f"{sequence} Doppler-time")
    ax.set_xlim(time_edges[0], time_edges[-1])
    ax.set_ylim(velocity_edges[0], velocity_edges[-1])
    fig.savefig(path)
    plt.close(fig)


def save_percentile_plot(
    path: Path,
    spectrum: np.ndarray,
    times: np.ndarray,
    velocities: np.ndarray,
    sequence: str,
    low_percentile: float = 10.0,
    high_percentile: float = 100.0,
    chirp_frequency_hz: float = CHIRP_FREQUENCY_HZ,
) -> tuple[float, float]:
    """Render log-power after clipping and scaling between two percentiles."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smpl-micro-doppler")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    power_db = 10.0 * np.log10(np.maximum(spectrum, np.finfo(np.float32).tiny))
    low_db, high_db = np.percentile(power_db, [low_percentile, high_percentile])
    if not high_db > low_db:
        raise ValueError(f"Invalid percentile span: {low_db} to {high_db} dB")
    normalized = np.clip((power_db - low_db) / (high_db - low_db), 0.0, 1.0)
    time_edges = centers_to_edges(times, STFT_HOP_CHIRPS / chirp_frequency_hz)
    velocity_edges = centers_to_edges(velocities, float(np.mean(np.diff(velocities))))
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=170, constrained_layout=True)
    image = ax.pcolormesh(
        time_edges,
        velocity_edges,
        normalized,
        shading="flat",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(f"Normalized log power (P{low_percentile:g}-P{high_percentile:g})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Radial velocity (m/s)")
    ax.set_title(f"{sequence} micro-Doppler, P{low_percentile:g}-P{high_percentile:g} normalized")
    ax.set_xlim(time_edges[0], time_edges[-1])
    ax.set_ylim(velocity_edges[0], velocity_edges[-1])
    fig.savefig(path)
    plt.close(fig)
    return float(low_db), float(high_db)


def main() -> int:
    args = parse_args()
    validate_args(args)
    apply_radar_profile(args.radar_profile)
    amass_path = Path(args.amass_npz).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    label = args.sequence or sequence_label(amass_path)
    result_path = out_dir / f"{label}_micro_doppler.npz"
    image_path = out_dir / f"{label}_micro_doppler.png"
    summary_path = out_dir / f"{label}_micro_doppler.json"
    doppler_time_path = out_dir / f"{label}_doppler_time.npz"
    doppler_time_image_path = out_dir / f"{label}_doppler_time.png"
    maybe_doppler_paths = (doppler_time_path, doppler_time_image_path) if args.save_frame_doppler_time else ()
    existing = [path for path in (result_path, image_path, summary_path, *maybe_doppler_paths) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}; pass --overwrite to recompute")

    device = torch.device(args.device)
    chirp_frequency_hz = RADAR_PROFILES[args.radar_profile]
    motion = load_amass(amass_path)
    mocap_rate = float(motion["mocap_frame_rate"])
    native_vertices_np, faces = generate_native_meshes(
        motion,
        model_dir,
        device,
        args.smpl_batch_size,
    )
    native_vertices_np = place_like_gif(native_vertices_np, args.subject_range, args.subject_lateral)
    faces = faces_for_witwin_world(faces)
    native_vertices = torch.as_tensor(native_vertices_np, dtype=torch.float32, device=device)
    del native_vertices_np
    duration_s = (native_vertices.shape[0] - 1) / mocap_rate
    frame_times, chirp_times_by_frame = actual_radar_timing(duration_s, chirp_frequency_hz)
    if args.max_radar_frames:
        frame_times = frame_times[: args.max_radar_frames]
        chirp_times_by_frame = chirp_times_by_frame[: args.max_radar_frames]
    print(
        f"[timing] native={native_vertices.shape[0]} frames at {mocap_rate:g} Hz; "
        f"radar={len(frame_times)} complete frames, {chirp_times_by_frame.size} acquired chirps",
        flush=True,
    )

    config = radar_config(chirp_frequency_hz, args.tx_rx_lateral_separation_m)
    derived = derived_radar_parameters(config)
    range_bin_count = max(1, int(round(RANGE_WINDOW_M / derived["range_resolution_m"])))
    fixed_range_bins = None
    fixed_center_range_m = None
    if args.range_bin_mode == "fixed":
        fixed_center_range_m = fixed_range_center_m(
            args.subject_range,
            args.subject_lateral,
            args.tx_rx_lateral_separation_m,
        )
        fixed_center_bin = int(round(fixed_center_range_m / derived["range_resolution_m"]))
        fixed_range_bins = centered_range_indices(fixed_center_bin, ADC_SAMPLES // 2, range_bin_count)
        print(
            f"[range] fixed bins={fixed_range_bins[0]}..{fixed_range_bins[-1]} "
            f"({fixed_range_bins[0] * derived['range_resolution_m']:.3f}.."
            f"{fixed_range_bins[-1] * derived['range_resolution_m']:.3f} m)",
            flush=True,
        )
    Radar, RadarConfig, Scene, Tracer = bootstrap_witwin_modules()
    apply_radar_equation_patch()
    pose = RadarPose(position=(0.0, args.radar_height, 0.0), target=(0.0, args.radar_height, -1.0))
    radar = Radar(
        RadarConfig.from_dict(config),
        backend=args.backend,
        device=args.device,
        position=pose.position,
        target=pose.target,
        up=pose.up,
        name=args.radar_profile,
    )
    tx_pos_np = radar.tx_pos.detach().cpu().numpy().astype(float)
    rx_pos_np = radar.rx_pos.detach().cpu().numpy().astype(float)
    scene = Scene(device=args.device)
    # Keep WiTwin's mutable dynamic buffer separate from interpolation keyframes.
    scene.add_mesh(name="human", vertices=native_vertices[0].clone(), faces=faces, dynamic=True)
    tracer = Tracer(
        scene,
        radar,
        resolution=1,
        sampling="triangle",
        multipath=False,
        max_reflections=0,
    )
    interpolator = FrameLevelScattererInterpolator(
        radar,
        scene,
        tracer,
        native_vertices,
        faces,
        mocap_rate,
        duration_s,
    )
    slow_time_points, chirp_times, strong_ranges, selected_bins = simulate_slow_time(
        radar,
        interpolator,
        frame_times,
        chirp_times_by_frame,
        derived,
        fixed_range_bins,
        args.visibility_mode,
    )
    spectrum, stft_times, velocities = mean_power_stft(
        slow_time_points,
        chirp_times,
        derived["wavelength_m"],
        chirp_frequency_hz,
        args.doppler_transform,
    )
    frame_doppler = None
    frame_doppler_times = None
    frame_doppler_velocities = None
    if args.save_frame_doppler_time:
        frame_doppler, frame_doppler_times, frame_doppler_velocities = mean_power_frame_doppler_time(
            slow_time_points,
            frame_times,
            derived["wavelength_m"],
            chirp_frequency_hz,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    visible_per_frame = np.asarray(interpolator.frame_visible_counts, dtype=np.int32)
    visible_per_chirp = np.repeat(visible_per_frame, CHIRPS_PER_FRAME)
    np.savez_compressed(
        result_path,
        spectrum=spectrum,
        stft_time_s=stft_times,
        velocity_mps=velocities,
        strong_range_m=strong_ranges,
        radar_frame_time_s=frame_times,
        chirp_time_s=chirp_times,
        visible_triangles_per_frame=visible_per_frame,
        visible_triangles_per_chirp=visible_per_chirp,
        next_visible_triangles_per_frame=np.asarray(interpolator.frame_next_visible_counts, dtype=np.int32),
        union_visible_triangles_per_frame=np.asarray(interpolator.frame_union_visible_counts, dtype=np.int32),
        selected_range_bins_by_frame=np.asarray(selected_bins, dtype=np.int16),
    )
    save_plot(image_path, spectrum, stft_times, velocities, label, args.db_floor, chirp_frequency_hz)
    if frame_doppler is not None and frame_doppler_times is not None and frame_doppler_velocities is not None:
        np.savez_compressed(
            doppler_time_path,
            spectrum=frame_doppler,
            frame_time_s=frame_doppler_times,
            velocity_mps=frame_doppler_velocities,
            strong_range_m=strong_ranges,
            radar_frame_time_s=frame_times,
            selected_range_bins_by_frame=np.asarray(selected_bins, dtype=np.int16),
        )
        save_doppler_time_plot(
            doppler_time_image_path,
            frame_doppler,
            frame_doppler_times,
            frame_doppler_velocities,
            label,
            args.db_floor,
        )
    visible_array = visible_per_frame
    summary = {
        "processing": "AMASS SMPL-X -> linear mesh interpolation -> frame-level visibility -> per-chirp geometry/intensity -> CUDA MIMO -> STFT",
        "source_amass_npz": str(amass_path),
        "sequence": label,
        "gender": str(motion["gender"]),
        "native_motion_frames": int(native_vertices.shape[0]),
        "native_motion_rate_hz": mocap_rate,
        "native_motion_duration_s": duration_s,
        "motion_interpolation": "linear per-vertex interpolation at each acquired chirp time",
        "placement": {
            "initial_subject_range_m": float(args.subject_range),
            "initial_subject_lateral_m": float(args.subject_lateral),
            "ground_aligned": True,
            "radar_height_m": float(args.radar_height),
        },
        "radar": {
            "profile": args.radar_profile,
            "frequency_start_ghz": PROFILE_FREQUENCY_START_HZ / 1e9,
            "frequency_stop_ghz": PROFILE_FREQUENCY_STOP_HZ / 1e9,
            "frame_rate_hz": FRAME_RATE_HZ,
            "chirp_frequency_hz": chirp_frequency_hz,
            "chirps_per_frame": CHIRPS_PER_FRAME,
            "samples_per_chirp": ADC_SAMPLES,
            "adc_duty_cycle": PROFILE_ADC_DUTY_CYCLE,
            "frame_chirp_duty_cycle": CHIRPS_PER_FRAME * FRAME_RATE_HZ / chirp_frequency_hz,
            "tx": 1,
            "rx": 1,
            "tx_rx_lateral_separation_m": float(args.tx_rx_lateral_separation_m),
            "tx_world_m": tx_pos_np.tolist(),
            "rx_world_m": rx_pos_np.tolist(),
            "waveform": "continuous sawtooth up-chirp",
            "num_complete_frames": int(frame_times.size),
            "num_acquired_chirps": int(chirp_times.size),
            "last_acquired_chirp_time_s": float(chirp_times[-1]),
            "frame_gap_s": 1.0 / FRAME_RATE_HZ - CHIRPS_PER_FRAME / chirp_frequency_hz,
            "stft_time_order": "acquired chirps; physical frame gaps retained in chirp timestamps",
        },
        "ray_tracing": {
            "sampling": "triangle",
            "frequency": (
                f"at adjacent {FRAME_RATE_HZ:g} Hz radar-frame boundaries with endpoint reuse"
                if args.visibility_mode == "linear"
                else f"once at the start of each {FRAME_RATE_HZ:g} Hz radar frame"
            ),
            "triangles_tested_per_frame": int(faces.shape[0]),
            "visibility_mode": args.visibility_mode,
            "frame_visibility_policy": (
                "0/1 face visibility linearly interpolated over chirp time between adjacent frame traces"
                if args.visibility_mode == "linear"
                else "visible triangle IDs held constant for all 128 chirps in the frame"
            ),
            "chirp_position_update": "triangle centroids from linearly interpolated SMPL-X vertices",
            "chirp_intensity_update": "normal, area, and Gaussian normal weight recomputed at every chirp",
            "self_occlusion": True,
            "front_face_test": True,
            "multipath": False,
            "face_winding": "reversed after handedness-changing AMASS-to-WiTwin coordinate transform",
            "gaussian_normal_eta": SPECULAR_ETA,
            "intensity_model": (
                "triangle area * Gaussian specular weight based on angle between the "
                "surface-normal reflection of TX->point and point->RX directions"
            ),
            "visible_triangles_min": int(visible_array.min()),
            "visible_triangles_mean": float(visible_array.mean()),
            "visible_triangles_max": int(visible_array.max()),
        },
        "stft": {
            "range_window_m": RANGE_WINDOW_M,
            "range_bin_mode": args.range_bin_mode,
            "fixed_range_center_m": fixed_center_range_m,
            "fixed_range_bins": fixed_range_bins.tolist() if fixed_range_bins is not None else None,
            "range_fft_window": "hann",
            "window_chirps": STFT_WINDOW_CHIRPS,
            "fft_size": STFT_FFT_SIZE,
            "overlap_fraction": STFT_OVERLAP,
            "hop_chirps": STFT_HOP_CHIRPS,
            "window": "hamming",
            "doppler_transform": args.doppler_transform,
            "doppler_time_basis": (
                "actual acquired chirp timestamps" if args.doppler_transform == "nudft" else "uniform chirp interval"
            ),
            "reflection_reduction": "mean power after STFT",
            "num_columns": int(spectrum.shape[1]),
        },
        "range_resolution_m": derived["range_resolution_m"],
        "strong_range_mean_m": float(np.mean(strong_ranges)),
        "selected_range_bins_by_frame": selected_bins,
        "result_npz": str(result_path),
        "image_png": str(image_path),
        "doppler_time_npz": str(doppler_time_path) if frame_doppler is not None else None,
        "doppler_time_png": str(doppler_time_image_path) if frame_doppler is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
