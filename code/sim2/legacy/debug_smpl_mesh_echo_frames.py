#!/usr/bin/env python3
"""Debug selected SMPL-mesh Sim2 echo frames without writing raw bins."""

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
    parser.add_argument("--context", type=int, default=3)
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
    parser.set_defaults(replace_foot_mesh_with_ankle_scatterers=True)
    return parser.parse_args()


def describe_array(name: str, array: np.ndarray) -> None:
    finite = np.isfinite(array)
    print(
        f"{name}: shape={array.shape} finite={int(finite.sum())}/{array.size} "
        f"nan={int(np.isnan(array).sum())} inf={int(np.isinf(array).sum())}",
        flush=True,
    )
    if finite.any():
        vals = array[finite]
        print(
            f"{name}: min={float(vals.min()):.6g} p50={float(np.percentile(vals, 50)):.6g} "
            f"p99={float(np.percentile(vals, 99)):.6g} max={float(vals.max()):.6g}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    fit = np.load(args.fit_npz)
    all_radar_frames = fit["radar_frames"].astype(int)
    all_vertices = fit["vertices"].astype(np.float32)
    all_joints = fit["smpl_joints"].astype(np.float32)
    faces = fit["faces"]

    selected_frames: list[int] = []
    for frame in args.frames:
        for neighbor in range(frame - args.context, frame + args.context + 1):
            if neighbor in all_radar_frames:
                selected_frames.append(neighbor)
    selected_frames = sorted(set(selected_frames))
    selected_indices = np.asarray([int(np.where(all_radar_frames == frame)[0][0]) for frame in selected_frames])
    radar_frames = all_radar_frames[selected_indices]
    vertices = all_vertices[selected_indices]
    joints = all_joints[selected_indices]

    print(f"selected radar frames: {radar_frames.tolist()}", flush=True)
    right_foot_faces, left_foot_faces = sim.build_foot_face_masks(
        all_vertices,
        faces,
        all_joints,
        args.foot_mask_radius_m,
        args.foot_mask_height_m,
    )
    body_faces = faces[~(right_foot_faces | left_foot_faces)]
    print(
        f"body_faces={len(body_faces)} removed_feet="
        f"({int(right_foot_faces.sum())}, {int(left_foot_faces.sum())})",
        flush=True,
    )

    scatter = sim.mesh_scatter.trace_keyframes(args, vertices, body_faces)
    scatter = sim.add_ankle_foot_scatterers(
        args,
        scatter,
        vertices,
        faces,
        joints,
        right_foot_faces,
        left_foot_faces,
    )
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

    frame_rate = float(radar.config.frame_per_second)
    chirp_period = (float(radar.config.idle_time) + float(radar.config.ramp_end_time)) * 1e-6
    for frame in args.frames:
        print(f"\n=== radar frame {frame:06d} ===", flush=True)
        if frame in radar_frames:
            local_idx = int(np.where(radar_frames == frame)[0][0])
            intens = scatter["intensity"][local_idx]
            positive = intens[intens > 0.0]
            above = intens[intens > args.intensity_threshold]
            print(
                f"keyframe local_idx={local_idx} trace_count={int(scatter['trace_counts'][local_idx])} "
                f"positive={positive.size} above_threshold={above.size}",
                flush=True,
            )
            describe_array("keyframe_positive_intensity", positive)
        t0 = (float(frame) - 1.0) / frame_rate
        sample_counts = []
        sample_intensity_sum = []
        sample_finite = []
        for chirp_id in range(int(radar.config.chirp_per_frame)):
            sample = interpolator(t0 + chirp_id * chirp_period * float(radar.config.num_tx))
            intens_t, points_t = sample
            sample_counts.append(int(intens_t.numel()))
            sample_intensity_sum.append(float(intens_t.detach().sum().cpu()) if intens_t.numel() else 0.0)
            sample_finite.append(bool(torch.isfinite(intens_t).all() and torch.isfinite(points_t).all()))
        print(
            f"chirp scatterer count min/max={min(sample_counts)}/{max(sample_counts)} "
            f"sum_intensity min/max={min(sample_intensity_sum):.6g}/{max(sample_intensity_sum):.6g} "
            f"finite_all={all(sample_finite)}",
            flush=True,
        )
        with torch.no_grad():
            mimo = radar.mimo(interpolator, t0=t0)
        mimo_np = mimo.detach().cpu().numpy()
        describe_array("mimo_real", mimo_np.real)
        describe_array("mimo_imag", mimo_np.imag)
        iq = point_sim.quantize_complex(mimo_np, iq_scale="auto")
        iq_signed = iq.view("<i2")
        print(
            f"quantized_iq nonzero={int(np.count_nonzero(iq_signed))}/{iq_signed.size} "
            f"absmax={int(np.max(np.abs(iq_signed.astype(np.int32))))}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
