#!/usr/bin/env python3
"""Render a motion GIF from a dense SMPL NPZ."""

from __future__ import annotations

import argparse
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


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    dense = np.load(args.dense_npz)
    radar_frames = dense["radar_frames"].astype(int)
    indices = selected_indices(radar_frames, args.frames, args.stride)
    vertices = dense["vertices"][indices].astype(np.float32)
    smpl_joints = dense["smpl_joints"][indices].astype(np.float32)
    raw_keypoints = dense["rtpose_world_keypoints"][indices].astype(np.float32)
    keypoints = fill_keypoints(raw_keypoints, smpl_joints)
    frame_ids = radar_frames[indices]

    point_stride = max(1, int(args.vertex_stride))
    body_points = np.concatenate([smpl_joints, vertices[:, ::point_stride]], axis=1)
    body_span = np.max(body_points.max(axis=1) - body_points.min(axis=1), axis=1)
    radius = max(1.15, 0.62 * float(np.percentile(body_span, 90)) + 0.20)

    out_gif = Path(args.out)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out_gif.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for out_idx, frame_idx in enumerate(range(len(indices))):
        fig = plt.figure(figsize=(5.2, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        vp = vertices[frame_idx][:, [0, 2, 1]]
        cloud = vp[::point_stride]
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c="#475569", s=2.8, alpha=0.55, depthshade=False)

        sj = smpl_joints[frame_idx]
        kp = keypoints[frame_idx]
        ax.scatter(sj[:, 0], sj[:, 2], sj[:, 1], c="#2563eb", s=22, depthshade=False)
        if np.isfinite(raw_keypoints[frame_idx]).all():
            ax.scatter(kp[:, 0], kp[:, 2], kp[:, 1], c="#dc2626", s=30, depthshade=False)
        for a, b in RT_BONES:
            ax.plot([sj[a, 0], sj[b, 0]], [sj[a, 2], sj[b, 2]], [sj[a, 1], sj[b, 1]], c="#2563eb", linewidth=1.7)
            if np.isfinite(raw_keypoints[frame_idx]).all():
                ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 2], kp[b, 2]], [kp[a, 1], kp[b, 1]], c="#ef4444", linewidth=1.7)

        center = smpl_joints[frame_idx, 0]
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=18, azim=-62)
        label = args.title or Path(args.dense_npz).stem
        ax.set_title(f"{label}  radar {int(frame_ids[frame_idx]):06d}", fontsize=10)
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
