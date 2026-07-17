#!/usr/bin/env python3
"""Generate RT-Pose raw echo from SMPL mesh triangle scatterers with WiTwin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "sim1_raw_echo_generation"))

import simulate_rtpose_witwin as point_sim
import smpl_mesh_witwin_scatterers as mesh_scatter
import witwin_radar_equation_patch


DEFAULT_FIT = (
    REPO_ROOT
    / "results"
    / "SMPL_fit"
    / "smpl_sequence_fit_v9_six_motion_gifs"
    / "seq185_joint_labels_temporal_v9"
    / "seq185_joint_labels_temporal_v9_fit.npz"
)

STANDARD_SPECULAR_ETA = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-npz", default=str(DEFAULT_FIT))
    parser.add_argument("--sequence", default="185")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "datasets" / "Sim2_sequences"))
    parser.add_argument("--file-idx", default="0000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="dirichlet", choices=("pytorch", "dirichlet", "slang"))
    parser.add_argument("--rx-order", choices=("attachment", "rtpose_raw"), default="attachment")
    parser.add_argument(
        "--chirp-per-frame",
        type=int,
        default=64,
        help="Radar chirps per frame passed to WiTwin RadarConfig.",
    )
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--epsilon-r", type=float, default=5.0)
    parser.add_argument("--max-radar-frame", type=int, default=20)
    parser.add_argument("--start-radar-frame", type=int, default=1)
    parser.add_argument(
        "--interp",
        choices=("linear", "cubic"),
        default="cubic",
        help="Interpolate scatterer centroid positions between radar-frame SMPL meshes.",
    )
    parser.add_argument("--intensity-threshold", type=float, default=1e-14)
    parser.add_argument(
        "--iq-scale",
        default="auto",
        help=(
            "ADC scale passed to the raw-bin writer. Use a numeric value for fixed scaling, "
            "'auto' for per-frame peak scaling, or 'auto-global' to use one peak-derived "
            "scale for the whole generated sequence."
        ),
    )
    parser.add_argument("--save-scatterers", action="store_true")
    return parser.parse_args()


def load_fit_subset(fit_path: Path, start_radar_frame: int, max_radar_frame: int) -> dict[str, np.ndarray]:
    fit = np.load(fit_path)
    radar_frames = np.asarray([int(x) for x in fit["radar_frames"]], dtype=np.int32)
    target = (radar_frames >= int(start_radar_frame)) & (radar_frames <= int(max_radar_frame))
    if not np.any(target):
        first = int(radar_frames.min())
        last = int(radar_frames.max())
        raise ValueError(
            f"No SMPL keyframes in requested radar range {start_radar_frame}..{max_radar_frame}; "
            f"available range is {first}..{last}"
        )
    keep = target.copy()
    # Include one neighbor on each side when available so interpolation at the
    # requested window boundary is not forced to use a spline endpoint.
    kept_indices = np.flatnonzero(target)
    first_keep = int(kept_indices[0])
    last_keep = int(kept_indices[-1])
    if first_keep > 0:
        keep[first_keep - 1] = True
    if last_keep + 1 < len(radar_frames):
        keep[last_keep + 1] = True
    rtpose_joints = fit["rtpose_world_keypoints"][keep].astype(np.float32)
    smpl_joints = (
        fit["smpl_joints"][keep].astype(np.float32)
        if "smpl_joints" in fit.files and fit["smpl_joints"].shape == fit["rtpose_world_keypoints"].shape
        else rtpose_joints
    )
    joint_fallback_mask = ~np.isfinite(rtpose_joints).all(axis=2)
    joints = rtpose_joints.copy()
    joints[joint_fallback_mask] = smpl_joints[joint_fallback_mask]
    return {
        "vertices": fit["vertices"][keep].astype(np.float32),
        "faces": fit["faces"].astype(np.int32),
        "frames": fit["frames"][keep],
        "radar_frames": radar_frames[keep],
        "joints": joints,
        "rtpose_joints": rtpose_joints,
        "smpl_joints": smpl_joints,
        "joint_fallback_mask": joint_fallback_mask,
    }


def trace_dense_keyframes(
    args: argparse.Namespace,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, np.ndarray]:
    trace_args = argparse.Namespace(
        device=args.device,
        backend=args.backend,
        rx_order=args.rx_order,
        resolution=args.resolution,
        epsilon_r=args.epsilon_r,
    )
    scatter = mesh_scatter.trace_keyframes(trace_args, vertices, faces)
    scatter["tracing_skipped"] = np.asarray(False)
    scatter["traced_faces"] = faces.astype(np.int32, copy=False)
    return scatter


def build_interpolators(
    radar_frame_ids: np.ndarray,
    centroids: np.ndarray,
    interp: str,
):
    x = radar_frame_ids.astype(np.float64)
    # CubicSpline requires strictly increasing x. The SMPL fit follows Train.json
    # radar frame order, but make this explicit for safer diagnostics.
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"radar frame ids must be strictly increasing, got {radar_frame_ids.tolist()}")
    if interp == "cubic":
        bc_type = "not-a-knot" if len(x) >= 4 else "natural"
        return (
            CubicSpline(x, centroids, axis=0, bc_type=bc_type, extrapolate=False),
            float(x[0]),
            float(x[-1]),
        )
    return (None, float(x[0]), float(x[-1]))


def linear_at(x: np.ndarray, values: np.ndarray, query: float) -> np.ndarray:
    if query <= float(x[0]):
        return values[0]
    if query >= float(x[-1]):
        return values[-1]
    hi = int(np.searchsorted(x, query, side="right"))
    lo = hi - 1
    alpha = np.float32((query - float(x[lo])) / float(x[hi] - x[lo]))
    return values[lo] * (np.float32(1.0) - alpha) + values[hi] * alpha


def hold_frame_index(frame_times: np.ndarray, query: float) -> int:
    idx = int(np.searchsorted(frame_times, query, side="right") - 1)
    return int(np.clip(idx, 0, len(frame_times) - 1))


def frame_visible_angle_intensity(
    points: np.ndarray,
    frame_normals: np.ndarray,
    frame_areas: np.ndarray,
    frame_visible: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Visible area times a Gaussian lobe in incidence angle, evaluated per chirp."""
    view_dirs = -points.astype(np.float32, copy=False)
    view_dirs /= np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), np.float32(1e-6))
    normals = frame_normals.astype(np.float32, copy=False)
    cos_theta = np.sum(view_dirs * normals, axis=1)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    theta = np.arccos(cos_theta).astype(np.float32)
    eta_safe = max(float(eta), 1e-6)
    specular = np.exp(-((np.float32(2.0) * theta) ** 2) / np.float32(2.0 * eta_safe * eta_safe))
    return (
        frame_visible.astype(np.float32, copy=False)
        * np.maximum(frame_areas.astype(np.float32, copy=False), np.float32(0.0))
        * specular.astype(np.float32, copy=False)
    ).astype(np.float32)


def make_interpolator(
    radar,
    radar_frame_ids: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    visible: np.ndarray,
    threshold: float,
    interp: str,
):
    pos_spline, t_min, t_max = build_interpolators(radar_frame_ids, centroids, interp)
    radar_frame_times = radar_frame_ids.astype(np.float64)
    frame_rate = float(radar.config.frame_per_second)
    device = radar.device

    def interpolator(t_seconds: float):
        radar_time = 1.0 + float(t_seconds) * frame_rate
        query = float(np.clip(radar_time, t_min, t_max))
        if interp == "cubic":
            query_array = np.asarray([query], dtype=np.float64)
            points_np = pos_spline(query_array)[0].astype(np.float32)
        else:
            points_np = linear_at(radar_frame_times, centroids, query).astype(np.float32)

        frame_idx = hold_frame_index(radar_frame_times, query)
        intens_np = frame_visible_angle_intensity(
            points_np,
            normals[frame_idx],
            areas[frame_idx],
            visible[frame_idx],
            STANDARD_SPECULAR_ETA,
        )
        mask = intens_np > float(threshold)
        tri_points = points_np[mask]
        tri_intens = intens_np[mask]

        if tri_intens.size == 0:
            return (
                torch.zeros(0, dtype=torch.float32, device=device),
                torch.zeros((0, 3), dtype=torch.float32, device=device),
            )
        return (
            torch.as_tensor(tri_intens, dtype=torch.float32, device=device),
            torch.as_tensor(tri_points, dtype=torch.float32, device=device),
        )

    return interpolator


def simulate_echo(
    args: argparse.Namespace,
    fit_subset: dict[str, np.ndarray],
    scatter: dict[str, np.ndarray],
) -> dict:
    Radar, RadarConfig, _, _ = mesh_scatter.bootstrap_witwin_mesh_modules()
    witwin_radar_equation_patch.apply()
    radar_config = point_sim.build_radar_config(args.rx_order)
    radar_config["chirp_per_frame"] = int(args.chirp_per_frame)
    radar = Radar(
        RadarConfig.from_dict(radar_config),
        backend=args.backend,
        device=args.device,
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
    )
    interpolator = make_interpolator(
        radar,
        fit_subset["radar_frames"],
        scatter["centroids"],
        scatter["normals"],
        scatter["areas"],
        scatter["visible"],
        args.intensity_threshold,
        args.interp,
    )

    shape = point_sim.RadarShape(loops=int(args.chirp_per_frame))
    bin_dir = Path(args.output_root) / str(args.sequence) / "radar" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {name: bin_dir / f"{name}_{args.file_idx}_data.bin" for name in point_sim.DEVICE_NAMES}
    handles = {name: path.open("wb") for name, path in data_paths.items()}
    frame_ids = list(range(int(args.start_radar_frame), int(args.max_radar_frame) + 1))
    fixed_iq_scale = args.iq_scale
    global_iq_peak = None
    if str(args.iq_scale).lower() == "auto-global":
        peaks: list[float] = []
        for index, radar_id in enumerate(frame_ids, start=1):
            t0 = (float(radar_id) - 1.0) / float(radar.config.frame_per_second)
            with torch.no_grad():
                frame = radar.mimo(interpolator, t0=t0)
            frame_np = frame.detach().cpu().numpy()
            if not np.isfinite(frame_np.real).all() or not np.isfinite(frame_np.imag).all():
                raise FloatingPointError(
                    f"non-finite MIMO output while estimating IQ scale for sequence {args.sequence} "
                    f"radar frame {radar_id}"
                )
            peak = max(float(np.max(np.abs(frame_np.real))), float(np.max(np.abs(frame_np.imag))))
            peaks.append(peak)
            print(f"[scale {index:03d}/{len(frame_ids):03d}] radar frame {radar_id:06d} peak={peak:.6e}", flush=True)
        global_iq_peak = max(max(peaks), 1e-12)
        fixed_iq_scale = str(30000.0 / global_iq_peak)
        print(
            f"[scale] auto-global peak={global_iq_peak:.6e} fixed iq_scale={float(fixed_iq_scale):.6e}",
            flush=True,
        )
    written = []
    try:
        for index, radar_id in enumerate(frame_ids, start=1):
            t0 = (float(radar_id) - 1.0) / float(radar.config.frame_per_second)
            with torch.no_grad():
                frame = radar.mimo(interpolator, t0=t0)
            frame_np = frame.detach().cpu().numpy()
            if not np.isfinite(frame_np.real).all() or not np.isfinite(frame_np.imag).all():
                raise FloatingPointError(
                    f"non-finite MIMO output for sequence {args.sequence} radar frame {radar_id}; "
                    "check interpolated scatterer positions/intensities"
                )
            iq = point_sim.quantize_complex(frame_np, iq_scale=fixed_iq_scale)
            point_sim.append_device_frame(handles, iq, shape)
            written.append(int(radar_id))
            print(f"[echo {index:03d}/{len(frame_ids):03d}] wrote radar frame {radar_id:06d}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    point_sim.write_idx_files(bin_dir, args.file_idx, len(frame_ids), shape)
    return {
        "bin_dir": str(bin_dir),
        "written_radar_frames": written,
        "valid_num_frames": len(frame_ids),
        "iq_scale_requested": str(args.iq_scale),
        "iq_scale_effective": str(fixed_iq_scale),
        "iq_auto_global_peak": None if global_iq_peak is None else float(global_iq_peak),
    }


def main() -> int:
    args = parse_args()
    fit_path = Path(args.fit_npz)
    out_root = Path(args.output_root)
    fit_subset = load_fit_subset(fit_path, args.start_radar_frame, args.max_radar_frame)
    print(
        f"Loaded {len(fit_subset['radar_frames'])} SMPL keyframes "
        f"for radar frames {fit_subset['radar_frames'].tolist()}",
        flush=True,
    )
    scatter = trace_dense_keyframes(args, fit_subset["vertices"], fit_subset["faces"])
    result = simulate_echo(args, fit_subset, scatter)

    sequence_dir = out_root / str(args.sequence)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    if args.save_scatterers:
        np.savez_compressed(
            sequence_dir / f"seq{args.sequence}_mesh_echo_scatterers_used.npz",
            fit_npz=str(fit_path),
            frames=fit_subset["frames"],
            radar_frames=fit_subset["radar_frames"],
            faces=fit_subset["faces"],
            keyframe_centroids=scatter["centroids"],
            keyframe_trace_intensity=scatter["intensity"],
            keyframe_normals=scatter["normals"],
            keyframe_areas=scatter["areas"],
            keyframe_joints=fit_subset["joints"],
            traced_faces=scatter.get("traced_faces", fit_subset["faces"]).astype(np.int32, copy=False),
            keyframe_visible=scatter["visible"],
            keyframe_trace_counts=scatter["trace_counts"],
            coordinate_system="witwin world [right, up, back] = Train.json [y, z, -x]",
        )

    summary = {
        "fit_npz": str(fit_path),
        "sequence": str(args.sequence),
        "output_root": str(out_root),
        "smpl_keyframe_radar_frames": [int(x) for x in fit_subset["radar_frames"]],
        "visible_triangles_per_keyframe": [int(x) for x in scatter["trace_counts"]],
        "positive_trace_intensity_triangles_per_keyframe": [int(x) for x in (scatter["intensity"] > 0).sum(axis=1)],
        "mesh_scatterer_policy": "all traced SMPL mesh triangles; no foot replacement, no mesh-block merge, no top-N selection",
        "chirp_intensity_mode": "frame-visible-angle",
        "specular_eta": float(STANDARD_SPECULAR_ETA),
        "specular_angle_model": "exp(-((2 * incidence_angle_rad)^2) / (2 * eta^2)); incidence uses dot(normal, point_to_radar)",
        "intensity_threshold": float(args.intensity_threshold),
        "interp": args.interp,
        "tracing_skipped": bool(scatter.get("tracing_skipped", np.asarray(False))),
        "joint_fallback_count": int(fit_subset.get("joint_fallback_mask", np.empty((0, 0), dtype=bool)).sum()),
        "backend": args.backend,
        "chirp_per_frame": int(args.chirp_per_frame),
        "resolution": int(args.resolution),
        "epsilon_r": float(args.epsilon_r),
        "start_radar_frame": int(args.start_radar_frame),
        "max_radar_frame": int(args.max_radar_frame),
        "coordinate_system": "witwin world [right, up, back] = Train.json [y, z, -x]",
        **result,
    }
    summary_path = sequence_dir / f"seq{args.sequence}_mesh_echo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
