#!/usr/bin/env python3
"""Simulate RT-Pose raw radar/bin files from Train.json poses with WiTwin.

The script uses the WiTwin point-scatterer MIMO solver without importing the
ray-tracing entry points, because the packaged rayd/drjit pair can be fragile
on machines where only point targets are needed.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "logs" / ".cache" / "witwin"


def resolve_witwin_radar_dir() -> Path:
    candidates: list[Path] = []
    if os.environ.get("WITWIN_RADAR_DIR"):
        candidates.append(Path(os.environ["WITWIN_RADAR_DIR"]))
    candidates.append(Path(sys.prefix) / "lib/python3.11/site-packages/witwin/radar")
    candidates.append(Path("/bigdata/users/quansj/miniforge3/envs/witwin/lib/python3.11/site-packages/witwin/radar"))
    candidates.append(Path("/home/quansj/miniforge3/envs/witwin/lib/python3.11/site-packages/witwin/radar"))
    for candidate in candidates:
        if (candidate / "radar.py").exists():
            return candidate
    return candidates[0]


WITWIN_RADAR_DIR = resolve_witwin_radar_dir()
WITWIN_ENV = WITWIN_RADAR_DIR.parents[4]

DEVICE_NAMES = ("master", "slave1", "slave2", "slave3")

TX_Y_HALF = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 9, 10, 11], dtype=np.float32)
TX_Z_HALF = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 6], dtype=np.float32)

# Attachment RX acquisition order.
RX_Y_HALF_ATTACHMENT = np.array(
    [50, 51, 52, 53, 0, 1, 2, 3, 46, 47, 48, 49, 11, 12, 13, 14],
    dtype=np.float32,
)
RX_Z_HALF = np.zeros(16, dtype=np.float32)

# RT-Pose Matlab/Python raw ADC order from raw_echo_to_xyz.RadarConfig.
RX_Y_HALF_RTPOSE_RAW = np.array(
    [11, 12, 13, 14, 50, 51, 52, 53, 46, 47, 48, 49, 0, 1, 2, 3],
    dtype=np.float32,
)

# Approximate per-keypoint projected areas/RCS in square meters. The 15 RT-Pose
# joints are treated as point scatterers, with larger torso/head weights and
# smaller distal limb weights.
KEYPOINT_RCS_SQM = np.array(
    [
        0.18,  # pelvis / root
        0.10,  # right upper leg / hip
        0.07,  # right lower leg / knee
        0.05,  # right foot / ankle
        0.10,  # left upper leg / hip
        0.07,  # left lower leg / knee
        0.05,  # left foot / ankle
        0.30,  # torso / neck
        0.12,  # head
        0.08,  # right shoulder / upper arm
        0.06,  # right forearm / elbow
        0.025,  # right hand / wrist
        0.08,  # left shoulder / upper arm
        0.06,  # left forearm / elbow
        0.025,  # left hand / wrist
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RadarShape:
    samples: int = 256
    loops: int = 64
    tx: int = 12
    rx: int = 16
    rx_per_device: int = 4

    @property
    def uint16_per_device_frame(self) -> int:
        return self.samples * self.tx * self.loops * self.rx_per_device * 2

    @property
    def bytes_per_device_frame(self) -> int:
        return self.uint16_per_device_frame * np.dtype("<u2").itemsize


def bootstrap_witwin_radar():
    """Import WiTwin radar, falling back to internals on broken installs."""
    os.environ.setdefault("HOME", str(CACHE_ROOT))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
    os.environ.setdefault("DRJIT_CACHE_DIR", str(CACHE_ROOT))
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        from witwin.radar import Radar, RadarConfig

        return Radar, RadarConfig
    except Exception:
        pass

    pkg = types.ModuleType("witwin.radar")
    pkg.__path__ = [str(WITWIN_RADAR_DIR)]
    sys.modules.setdefault("witwin.radar", pkg)
    from witwin.radar.radar import Radar, RadarConfig

    return Radar, RadarConfig


def rtpose_xyz_to_witwin(points_xyz: np.ndarray) -> np.ndarray:
    """Map RT-Pose [range_x, lateral_y, height_z] to WiTwin [right, up, back]."""
    out = np.empty_like(points_xyz, dtype=np.float32)
    out[:, 0] = points_xyz[:, 1]
    out[:, 1] = points_xyz[:, 2]
    out[:, 2] = -points_xyz[:, 0]
    return out


def antenna_half_to_witwin(y_half: np.ndarray, z_half: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(y), float(z), 0.0) for y, z in zip(y_half, z_half)]


def build_radar_config(rx_order: str) -> dict:
    rx_y = RX_Y_HALF_ATTACHMENT if rx_order == "attachment" else RX_Y_HALF_RTPOSE_RAW
    return {
        "num_tx": 12,
        "num_rx": 16,
        "fc": 77e9,
        "slope": 64.985,
        "adc_samples": 256,
        "adc_start_time": 5,
        "sample_rate": 5000,
        "idle_time": 5,
        "ramp_end_time": 60,
        "chirp_per_frame": 64,
        "frame_per_second": 10,
        "num_doppler_bins": 64,
        "num_range_bins": 256,
        "num_angle_bins": 128,
        "power": 0,
        "tx_loc": antenna_half_to_witwin(TX_Y_HALF, TX_Z_HALF),
        "rx_loc": antenna_half_to_witwin(rx_y, RX_Z_HALF),
    }


def load_sequence_poses(train_json: Path, sequence: str) -> tuple[np.ndarray, np.ndarray]:
    with train_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if sequence not in data:
        raise KeyError(f"Sequence {sequence} not found in {train_json}")

    by_radar: dict[int, np.ndarray] = {}
    for frame_objs in data[sequence].values():
        for obj in frame_objs:
            radar_id = int(obj["Radar_frameID"])
            by_radar[radar_id] = np.asarray(obj["pose"], dtype=np.float32)

    ids = np.asarray(sorted(by_radar), dtype=np.float32)
    poses = np.stack([by_radar[int(i)] for i in ids], axis=0)
    if poses.shape[1:] != (15, 3):
        raise ValueError(f"Expected poses with shape (F, 15, 3), got {poses.shape}")
    return ids, poses


def interpolate_pose_at(radar_time: float, ids: np.ndarray, poses: np.ndarray) -> np.ndarray:
    t = float(radar_time)
    if t <= ids[0]:
        return poses[0].copy()
    if t >= ids[-1]:
        return poses[-1].copy()
    hi = int(np.searchsorted(ids, t, side="right"))
    lo = hi - 1
    alpha = (t - ids[lo]) / (ids[hi] - ids[lo])
    return poses[lo] * (1.0 - alpha) + poses[hi] * alpha


def interpolate_pose(radar_id: int, ids: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return interpolate_pose_at(float(radar_id), ids, poses)


def choose_frame_ids(args: argparse.Namespace, ids: np.ndarray) -> list[int]:
    if args.frame_ids:
        out = [int(x) for x in args.frame_ids]
    else:
        out = [int(x) for x in ids]
    if args.max_frames is not None:
        out = out[: args.max_frames]
    if args.contiguous:
        return list(range(1, max(out) + 1))
    return out


def simulate_frame(
    radar,
    radar_id: int,
    ids: np.ndarray,
    poses: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    intensities = torch.as_tensor(KEYPOINT_RCS_SQM, dtype=torch.float32, device=device)
    frame_rate = float(radar.config.frame_per_second)
    t0 = (float(radar_id) - 1.0) / frame_rate

    def interpolator(t):
        radar_time = 1.0 + float(t) * frame_rate
        pose_xyz = interpolate_pose_at(radar_time, ids, poses)
        positions_np = rtpose_xyz_to_witwin(pose_xyz)
        positions = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
        return intensities, positions

    with torch.no_grad():
        frame = radar.mimo(interpolator, t0=t0)
    return frame.detach().cpu().numpy()


def quantize_complex(
    frame_tx_rx_loop_sample: np.ndarray,
    iq_scale: str,
) -> np.ndarray:
    # WiTwin returns (TX, RX, loop, sample). RT-Pose raw writer wants
    # (sample, loop, RX, TX).
    adc = np.transpose(frame_tx_rx_loop_sample, (3, 2, 1, 0)).astype(np.complex64, copy=False)
    if not np.isfinite(adc.real).all() or not np.isfinite(adc.imag).all():
        raise FloatingPointError("non-finite ADC values before quantization")
    if iq_scale == "auto":
        peak = max(float(np.max(np.abs(adc.real))), float(np.max(np.abs(adc.imag))), 1e-12)
        scale = 30000.0 / peak
    else:
        scale = float(iq_scale)
    real = np.clip(np.rint(adc.real * scale), -32768, 32767).astype("<i2", copy=False)
    imag = np.clip(np.rint(adc.imag * scale), -32768, 32767).astype("<i2", copy=False)
    interleaved = np.empty(real.shape + (2,), dtype="<i2")
    interleaved[..., 0] = real
    interleaved[..., 1] = imag
    return interleaved


def append_device_frame(handles: dict[str, object], iq: np.ndarray, shape: RadarShape) -> None:
    for device_idx, name in enumerate(DEVICE_NAMES):
        start = device_idx * shape.rx_per_device
        stop = start + shape.rx_per_device
        dev = iq[:, :, start:stop, :]
        # Reverse read_device_frame(): (sample, loop, rx4, tx) ->
        # (rx4, sample, tx, loop), Fortran flatten, interleaved IQ uint16.
        real_vec = np.transpose(dev[..., 0], (2, 0, 3, 1)).reshape(-1, order="F")
        imag_vec = np.transpose(dev[..., 1], (2, 0, 3, 1)).reshape(-1, order="F")
        raw_iq = np.empty(real_vec.size * 2, dtype="<i2")
        raw_iq[0::2] = real_vec
        raw_iq[1::2] = imag_vec
        handles[name].write(raw_iq.view("<u2").tobytes())


def write_idx_files(bin_dir: Path, file_idx: str, num_frames: int, shape: RadarShape) -> None:
    data_size = num_frames * shape.bytes_per_device_frame
    header = struct.pack("<IIIIQ", 1953194850, 1, 0, num_frames, data_size)
    records = []
    for i in range(num_frames):
        records.append(
            struct.pack(
                "<HHIHHIIIIIQQ",
                0,
                1,
                0,
                shape.samples,
                shape.loops * shape.tx,
                shape.bytes_per_device_frame,
                0,
                0,
                0,
                shape.bytes_per_device_frame,
                i,
                i * shape.bytes_per_device_frame,
            )
        )
    payload = header + b"".join(records)
    for name in DEVICE_NAMES:
        (bin_dir / f"{name}_{file_idx}_idx.bin").write_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", default=str(REPO_ROOT / "datasets" / "Train.json"))
    parser.add_argument("--sequence", default="185")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "datasets" / "Sim1_sequences"))
    parser.add_argument("--file-idx", default="0000")
    parser.add_argument("--frame-ids", type=int, nargs="*", help="Radar frame IDs to generate")
    parser.add_argument("--max-frames", type=int, help="Limit selected frames before contiguous expansion")
    parser.add_argument("--no-contiguous", dest="contiguous", action="store_false")
    parser.set_defaults(contiguous=True)
    parser.add_argument("--backend", default="pytorch", choices=("pytorch", "dirichlet", "slang"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iq-scale", default="auto", help="'auto' or a fixed multiplier")
    parser.add_argument("--rx-order", choices=("attachment", "rtpose_raw"), default="attachment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = RadarShape()
    Radar, RadarConfig = bootstrap_witwin_radar()

    ids, poses = load_sequence_poses(Path(args.train_json), str(args.sequence))
    frame_ids = choose_frame_ids(args, ids)

    device = torch.device(args.device)
    radar = Radar(
        RadarConfig.from_dict(build_radar_config(args.rx_order)),
        backend=args.backend,
        device=device,
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
    )

    bin_dir = Path(args.output_root) / str(args.sequence) / "radar" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {name: bin_dir / f"{name}_{args.file_idx}_data.bin" for name in DEVICE_NAMES}

    handles = {name: path.open("wb") for name, path in data_paths.items()}
    try:
        for index, radar_id in enumerate(frame_ids, start=1):
            frame = simulate_frame(
                radar,
                radar_id,
                ids,
                poses,
                device=device,
            )
            iq = quantize_complex(frame, args.iq_scale)
            append_device_frame(handles, iq, shape)
            print(f"[{index}/{len(frame_ids)}] wrote radar frame {radar_id:06d}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    write_idx_files(bin_dir, args.file_idx, len(frame_ids), shape)
    print(f"Wrote RT-Pose radar/bin files to {bin_dir}")


if __name__ == "__main__":
    main()
