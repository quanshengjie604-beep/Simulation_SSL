#!/usr/bin/env python3
"""Trace selected Sim2 frames through scatterers, antenna gains, MIMO, and quantization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate_smpl_mesh_witwin_echo as sim

point_sim = sim.point_sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-npz", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("pytorch", "dirichlet", "slang"), default="pytorch")
    parser.add_argument("--rx-order", choices=("attachment", "rtpose_raw"), default="attachment")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--epsilon-r", type=float, default=5.0)
    parser.add_argument("--interp", choices=("linear", "cubic"), default="linear")
    parser.add_argument("--intensity-threshold", type=float, default=1e-14)
    parser.add_argument("--foot-mask-radius-m", type=float, default=0.30)
    parser.add_argument("--foot-mask-height-m", type=float, default=0.08)
    parser.add_argument("--foot-ankle-area-scale", type=float, default=1.0)
    parser.add_argument("--max-triangle-scatterers", type=int, default=512)
    parser.add_argument("--iq-scale", default="auto")
    parser.set_defaults(replace_foot_mesh_with_ankle_scatterers=True)
    return parser.parse_args()


def stats(name: str, values: torch.Tensor | np.ndarray) -> str:
    arr = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    finite = np.isfinite(arr)
    if not finite.any():
        return f"{name}: shape={arr.shape} finite=0/{arr.size} nan={np.isnan(arr).sum()} inf={np.isinf(arr).sum()}"
    vals = arr[finite]
    return (
        f"{name}: shape={arr.shape} finite={finite.sum()}/{arr.size} "
        f"nan={np.isnan(arr).sum()} inf={np.isinf(arr).sum()} "
        f"min={vals.min():.6g} p50={np.percentile(vals, 50):.6g} "
        f"p99={np.percentile(vals, 99):.6g} max={vals.max():.6g}"
    )


def angle_stats(radar, sample, tx_pos, rx_pos) -> tuple[str, str]:
    tx_vectors = radar.local_from_world_vectors(sample.entry_points.unsqueeze(0) - tx_pos.unsqueeze(1))
    rx_vectors = radar.local_from_world_vectors(sample.points.unsqueeze(0) - rx_pos.unsqueeze(1))

    def one(prefix: str, vectors: torch.Tensor) -> str:
        forward = -vectors[..., 2]
        x_angles = torch.rad2deg(torch.atan2(vectors[..., 0], forward))
        y_angles = torch.rad2deg(torch.atan2(vectors[..., 1], forward))
        inside = (
            (x_angles >= radar.antenna_pattern_x_angles_deg[0])
            & (x_angles <= radar.antenna_pattern_x_angles_deg[-1])
            & (y_angles >= radar.antenna_pattern_y_angles_deg[0])
            & (y_angles <= radar.antenna_pattern_y_angles_deg[-1])
        )
        return (
            f"{prefix}: inside={int(inside.sum().item())}/{inside.numel()} "
            f"x=[{float(x_angles.min()):.3f},{float(x_angles.max()):.3f}] "
            f"y=[{float(y_angles.min()):.3f},{float(y_angles.max()):.3f}]"
        )

    return one("tx_angles", tx_vectors), one("rx_angles", rx_vectors)


def main() -> int:
    args = parse_args()
    fit = np.load(args.fit_npz)
    all_radar_frames = fit["radar_frames"].astype(int)
    all_vertices = fit["vertices"].astype(np.float32)
    rtpose_joints = fit["rtpose_world_keypoints"].astype(np.float32)
    smpl_joints = (
        fit["smpl_joints"].astype(np.float32)
        if "smpl_joints" in fit.files and fit["smpl_joints"].shape == fit["rtpose_world_keypoints"].shape
        else rtpose_joints
    )
    joint_fallback_mask = ~np.isfinite(rtpose_joints).all(axis=2)
    all_joints_raw = rtpose_joints.copy()
    all_joints_raw[joint_fallback_mask] = smpl_joints[joint_fallback_mask]
    all_joints = all_joints_raw
    faces = fit["faces"].astype(np.int32)

    selected_frames: list[int] = []
    for frame in args.frames:
        for neighbor in range(frame - args.context, frame + args.context + 2):
            if neighbor in all_radar_frames:
                selected_frames.append(neighbor)
    selected_frames = sorted(set(selected_frames))
    selected_indices = np.asarray([int(np.where(all_radar_frames == frame)[0][0]) for frame in selected_frames])
    radar_frames = all_radar_frames[selected_indices]
    vertices = all_vertices[selected_indices]
    joints = all_joints[selected_indices]
    print(f"selected radar frames: {radar_frames.tolist()}", flush=True)

    scatter = sim.trace_dense_keyframes(args, vertices, faces, joints)
    Radar, RadarConfig, _, _ = sim.mesh_scatter.bootstrap_witwin_mesh_modules()
    radar = Radar(
        RadarConfig.from_dict(point_sim.build_radar_config(args.rx_order)),
        backend=args.backend,
        device=args.device,
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
    )
    interpolator = sim.make_interpolator(
        radar,
        radar_frames,
        scatter["centroids"],
        scatter["normals"],
        scatter["areas"],
        scatter["visible"],
        args.intensity_threshold,
        args.interp,
        args.max_triangle_scatterers,
    )

    from witwin.radar.solvers.common import (
        compute_antenna_pattern_gains,
        compute_path_amplitudes,
        compute_total_path_lengths,
        normalize_interpolated_sample,
    )

    frame_rate = float(radar.config.frame_per_second)
    chirp_period = (float(radar.config.idle_time) + float(radar.config.ramp_end_time)) * 1e-6
    for frame_id in args.frames:
        print(f"\n=== radar frame {frame_id:06d} ===", flush=True)
        if frame_id in radar_frames:
            local_idx = int(np.where(radar_frames == frame_id)[0][0])
            intens = scatter["intensity"][local_idx]
            print(
                f"keyframe trace_count={int(scatter['trace_counts'][local_idx])} "
                f"positive={int((intens > 0).sum())} above_threshold={int((intens > args.intensity_threshold).sum())}",
                flush=True,
            )
            print(stats("keyframe_intensity_positive", intens[intens > 0]), flush=True)

        t0 = (float(frame_id) - 1.0) / frame_rate
        chirp_rows = []
        for chirp_id in range(int(radar.config.chirp_per_frame)):
            t = t0 + chirp_id * chirp_period * float(radar.config.num_tx)
            sample = normalize_interpolated_sample(interpolator(t), device=radar.device)
            total_lengths = compute_total_path_lengths(sample, radar.tx_pos, radar.rx_pos)
            gains = compute_antenna_pattern_gains(radar, sample, radar.tx_pos, radar.rx_pos)
            amps = compute_path_amplitudes(radar, sample, total_lengths, tx_pos=radar.tx_pos, rx_pos=radar.rx_pos)
            chirp_rows.append(
                (
                    int(sample.intensities.numel()),
                    float(sample.intensities.sum().detach().cpu()) if sample.intensities.numel() else 0.0,
                    int((sample.intensities > args.intensity_threshold).sum().detach().cpu())
                    if sample.intensities.numel()
                    else 0,
                    int((gains > 0).sum().detach().cpu()) if gains is not None else -1,
                    float(gains.max().detach().cpu()) if gains is not None and gains.numel() else 0.0,
                    int((torch.abs(amps) > 0).sum().detach().cpu()),
                    float(torch.abs(amps).max().detach().cpu()) if amps.numel() else 0.0,
                    bool(torch.isfinite(amps).all().detach().cpu()),
                )
            )
            if chirp_id in (0, int(radar.config.chirp_per_frame) // 2, int(radar.config.chirp_per_frame) - 1):
                print(f"chirp {chirp_id:02d}", flush=True)
                print(stats("sample_intensities", sample.intensities), flush=True)
                print(stats("sample_points", sample.points), flush=True)
                print(stats("pattern_gains", gains), flush=True)
                print(stats("amplitudes_abs", torch.abs(amps)), flush=True)
                tx_s, rx_s = angle_stats(radar, sample, radar.tx_pos, radar.rx_pos)
                print(tx_s, flush=True)
                print(rx_s, flush=True)

        counts = np.asarray(chirp_rows, dtype=object)
        print(
            "chirp_summary "
            f"scatterers={int(np.min(counts[:,0]))}/{int(np.max(counts[:,0]))} "
            f"intensity_sum={float(np.min(counts[:,1])):.6g}/{float(np.max(counts[:,1])):.6g} "
            f"above_thr={int(np.min(counts[:,2]))}/{int(np.max(counts[:,2]))} "
            f"gain_pos={int(np.min(counts[:,3]))}/{int(np.max(counts[:,3]))} "
            f"gain_max={float(np.min(counts[:,4])):.6g}/{float(np.max(counts[:,4])):.6g} "
            f"amp_pos={int(np.min(counts[:,5]))}/{int(np.max(counts[:,5]))} "
            f"amp_max={float(np.min(counts[:,6])):.6g}/{float(np.max(counts[:,6])):.6g} "
            f"amp_finite_all={all(bool(x) for x in counts[:,7])}",
            flush=True,
        )

        with torch.no_grad():
            mimo = radar.mimo(interpolator, t0=t0)
        mimo_np = mimo.detach().cpu().numpy()
        print(stats("mimo_real", mimo_np.real), flush=True)
        print(stats("mimo_imag", mimo_np.imag), flush=True)
        iq = point_sim.quantize_complex(mimo_np, iq_scale=args.iq_scale)
        iq_signed = iq.view("<i2")
        print(
            f"quantized nonzero={int(np.count_nonzero(iq_signed))}/{iq_signed.size} "
            f"absmax={int(np.max(np.abs(iq_signed.astype(np.int32))))}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
