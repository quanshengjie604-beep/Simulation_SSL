#!/usr/bin/env python3
"""Render a motion GIF from a dense SMPL NPZ."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", nargs="*", type=int, help="Radar frame ids to render. Default renders all.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride when --frames is omitted.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--vertex-stride", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--fixed-view",
        action="store_true",
        help="Keep world-coordinate bounds fixed so body translation remains visible.",
    )
    parser.add_argument(
        "--diagnostics-prefix",
        help="Write per-frame motion CSV and diagnostic PNG using this path prefix.",
    )
    parser.add_argument("--radar-fps", type=float, default=10.0)
    return parser.parse_args()


RT_BONES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (7, 9),
    (9, 10),
    (10, 11),
    (7, 12),
    (12, 13),
    (13, 14),
)


def selected_indices(radar_frames: np.ndarray, requested: list[int] | None, stride: int) -> np.ndarray:
    if requested:
        lookup = {int(frame): idx for idx, frame in enumerate(radar_frames)}
        missing = [frame for frame in requested if frame not in lookup]
        if missing:
            raise KeyError(f"Requested frames missing from dense NPZ: {missing}")
        return np.asarray([lookup[int(frame)] for frame in requested], dtype=np.int64)
    return np.arange(0, len(radar_frames), max(1, int(stride)), dtype=np.int64)


def fill_keypoints(keypoints: np.ndarray, smpl_joints: np.ndarray) -> np.ndarray:
    out = keypoints.copy()
    bad = ~np.isfinite(out).all(axis=2)
    out[bad] = smpl_joints[bad]
    return out


def radial_velocity(points: np.ndarray, frame_ids: np.ndarray, fps: float) -> np.ndarray:
    """Return line-of-sight velocity for each point; row zero has no predecessor."""
    ranges = np.linalg.norm(points.astype(np.float64), axis=2)
    dt = np.diff(frame_ids.astype(np.float64)) / float(fps)
    velocity = np.full_like(ranges, np.nan)
    velocity[1:] = np.diff(ranges, axis=0) / dt[:, None]
    return velocity


def abs_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def finite_extreme(values: np.ndarray, fn) -> float:
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else float("nan")


def write_motion_diagnostics(
    prefix: Path,
    frame_ids: np.ndarray,
    vertices: np.ndarray,
    smpl_joints: np.ndarray,
    raw_keypoints: np.ndarray,
    is_keyframe: np.ndarray,
    radar_fps: float,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    mesh_rv = radial_velocity(vertices, frame_ids, radar_fps)
    smpl_rv = radial_velocity(smpl_joints, frame_ids, radar_fps)
    keypoint_rv = radial_velocity(raw_keypoints, frame_ids, radar_fps)
    root_range = np.linalg.norm(smpl_joints[:, 0].astype(np.float64), axis=1)
    root_rv = np.full(len(frame_ids), np.nan, dtype=np.float64)
    root_rv[1:] = np.diff(root_range) / (np.diff(frame_ids) / float(radar_fps))
    mpjpe = np.linalg.norm(smpl_joints - raw_keypoints, axis=2)

    rows = []
    for idx, frame_id in enumerate(frame_ids):
        rows.append(
            {
                "radar_frame": int(frame_id),
                "time_s": (float(frame_id) - 1.0) / float(radar_fps),
                "is_keyframe": int(is_keyframe[idx]),
                "root_x_m": float(smpl_joints[idx, 0, 0]),
                "root_y_m": float(smpl_joints[idx, 0, 1]),
                "root_z_m": float(smpl_joints[idx, 0, 2]),
                "root_range_m": float(root_range[idx]),
                "root_radial_velocity_mps": float(root_rv[idx]),
                "mesh_radial_min_mps": finite_extreme(mesh_rv[idx], np.min),
                "mesh_radial_max_mps": finite_extreme(mesh_rv[idx], np.max),
                "mesh_radial_abs_p95_mps": abs_percentile(mesh_rv[idx], 95.0),
                "mesh_radial_abs_p99_mps": abs_percentile(mesh_rv[idx], 99.0),
                "smpl_joint_radial_abs_p95_mps": abs_percentile(smpl_rv[idx], 95.0),
                "smpl_joint_radial_abs_max_mps": finite_extreme(np.abs(smpl_rv[idx]), np.max),
                "keypoint_radial_abs_p95_mps": abs_percentile(keypoint_rv[idx], 95.0),
                "keypoint_radial_abs_max_mps": finite_extreme(np.abs(keypoint_rv[idx]), np.max),
                "smpl_keypoint_mpjpe_m": finite_extreme(mpjpe[idx], np.mean),
                "smpl_keypoint_max_error_m": finite_extreme(mpjpe[idx], np.max),
            }
        )

    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    time_s = np.asarray([row["time_s"] for row in rows])
    mesh_min = np.asarray([row["mesh_radial_min_mps"] for row in rows])
    mesh_max = np.asarray([row["mesh_radial_max_mps"] for row in rows])
    mesh_p95 = np.asarray([row["mesh_radial_abs_p95_mps"] for row in rows])
    mesh_p99 = np.asarray([row["mesh_radial_abs_p99_mps"] for row in rows])
    smpl_p95 = np.asarray([row["smpl_joint_radial_abs_p95_mps"] for row in rows])
    keypoint_p95 = np.asarray([row["keypoint_radial_abs_p95_mps"] for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].fill_between(time_s, mesh_min, mesh_max, color="#94a3b8", alpha=0.35, label="mesh min/max")
    axes[0].plot(time_s, mesh_p95, color="#2563eb", label="mesh |radial velocity| p95")
    axes[0].plot(time_s, mesh_p99, color="#0f172a", label="mesh |radial velocity| p99")
    axes[0].axhline(1.248, color="#dc2626", linestyle="--", linewidth=1.0, label="Doppler unambiguous limit")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].legend(loc="upper left", ncols=2, fontsize=8)

    axes[1].plot(time_s, smpl_p95, color="#2563eb", marker="o", markersize=3, label="SMPL joints p95")
    axes[1].plot(time_s, keypoint_p95, color="#dc2626", marker="o", markersize=3, label="RT-Pose keypoints p95")
    axes[1].plot(time_s, np.abs(root_rv), color="#16a34a", label="root |radial velocity|")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].plot(time_s, np.asarray([row["smpl_keypoint_mpjpe_m"] for row in rows]) * 100.0, color="#7c3aed", label="MPJPE")
    axes[2].plot(time_s, np.asarray([row["smpl_keypoint_max_error_m"] for row in rows]) * 100.0, color="#ea580c", label="max joint error")
    axes[2].set_ylabel("Fit error (cm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper left", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        for time, keyframe in zip(time_s, is_keyframe):
            if keyframe:
                ax.axvline(time, color="#64748b", alpha=0.18, linewidth=0.8)
    fig.suptitle("SMPL mesh and keypoint motion diagnostics")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, png_path


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    dense = np.load(args.dense_npz)
    radar_frames = dense["radar_frames"].astype(int)
    indices = selected_indices(radar_frames, args.frames, args.stride)
    vertices = dense["vertices"][indices].astype(np.float32)
    faces = dense["faces"].astype(np.int32)
    smpl_joints = dense["smpl_joints"][indices].astype(np.float32)
    raw_keypoints = dense["rtpose_world_keypoints"][indices].astype(np.float32)
    keypoints = fill_keypoints(raw_keypoints, smpl_joints)
    frame_ids = radar_frames[indices]
    is_keyframe = dense["is_keyframe"][indices].astype(bool)

    if args.diagnostics_prefix:
        csv_path, png_path = write_motion_diagnostics(
            Path(args.diagnostics_prefix),
            frame_ids,
            vertices,
            smpl_joints,
            raw_keypoints,
            is_keyframe,
            args.radar_fps,
        )
        print(f"wrote {csv_path}")
        print(f"wrote {png_path}")

    point_stride = max(1, int(args.vertex_stride))
    body_points = np.concatenate([smpl_joints, vertices[:, ::point_stride]], axis=1)
    body_span = np.max(body_points.max(axis=1) - body_points.min(axis=1), axis=1)
    radius = max(1.15, 0.62 * float(np.percentile(body_span, 90)) + 0.20)
    fixed_points = body_points.reshape(-1, 3)
    fixed_points = fixed_points[np.isfinite(fixed_points).all(axis=1)]
    fixed_center = (fixed_points.min(axis=0) + fixed_points.max(axis=0)) * 0.5
    fixed_radius = max(1.15, 0.56 * float(np.ptp(fixed_points, axis=0).max()) + 0.12)
    face_stride = max(1, int(np.ceil(len(faces) / 2800)))
    shown_faces = faces[::face_stride]

    out_gif = Path(args.out)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out_gif.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for out_idx, frame_idx in enumerate(range(len(indices))):
        fig = plt.figure(figsize=(5.2, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        vp = vertices[frame_idx][:, [0, 2, 1]]
        mesh = Poly3DCollection(
            vp[shown_faces],
            facecolors="#94a3b8",
            edgecolors="#475569",
            linewidths=0.03,
            alpha=0.62,
        )
        ax.add_collection3d(mesh)

        sj = smpl_joints[frame_idx]
        kp = keypoints[frame_idx]
        ax.scatter(sj[:, 0], sj[:, 2], sj[:, 1], c="#2563eb", s=22, depthshade=False)
        if np.isfinite(raw_keypoints[frame_idx]).all():
            ax.scatter(kp[:, 0], kp[:, 2], kp[:, 1], c="#dc2626", s=30, depthshade=False)
        for a, b in RT_BONES:
            ax.plot([sj[a, 0], sj[b, 0]], [sj[a, 2], sj[b, 2]], [sj[a, 1], sj[b, 1]], c="#2563eb", linewidth=1.7)
            if np.isfinite(raw_keypoints[frame_idx]).all():
                ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 2], kp[b, 2]], [kp[a, 1], kp[b, 1]], c="#ef4444", linewidth=1.7)

        center = fixed_center if args.fixed_view else smpl_joints[frame_idx, 0]
        shown_radius = fixed_radius if args.fixed_view else radius
        ax.set_xlim(center[0] - shown_radius, center[0] + shown_radius)
        ax.set_ylim(center[2] - shown_radius, center[2] + shown_radius)
        ax.set_zlim(center[1] - shown_radius, center[1] + shown_radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=18, azim=-62)
        label = args.title or Path(args.dense_npz).stem
        time_s = (float(frame_ids[frame_idx]) - 1.0) / float(args.radar_fps)
        frame_kind = "observed" if is_keyframe[frame_idx] else "interpolated"
        ax.set_title(
            f"{label}  frame {int(frame_ids[frame_idx]):06d}  t={time_s:.1f}s\n{frame_kind}",
            fontsize=10,
        )
        ax.text2D(0.02, 0.05, "blue: SMPL joints   red: RT-Pose keypoints", transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        path = frame_dir / f"{out_idx:04d}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        image_paths.append(path)

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in image_paths]
    duration_ms = int(round(1000.0 / max(float(args.fps), 0.1)))
    images[0].save(out_gif, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=False)
    print(f"wrote {out_gif} frames={len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
