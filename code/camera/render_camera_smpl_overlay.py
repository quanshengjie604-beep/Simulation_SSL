#!/usr/bin/env python3
"""Render left-camera videos with RT-Pose keypoints and fitted SMPL mesh overlays."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(REPO_ROOT / "datasets" / "Train_sp120_train_minus_val6.json"))
    parser.add_argument("--gt-root", default=str(REPO_ROOT / "datasets" / "GT_sequences"))
    parser.add_argument(
        "--smpl-root",
        default=str(REPO_ROOT / "results" / "sim2" / "smpl_v10_train_minus_val6"),
    )
    parser.add_argument("--calib", default=str(REPO_ROOT / "datasets" / "calib" / "camera" / "left.json"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "results" / "camera"))
    parser.add_argument("--sequences", type=int, nargs="*", help="Optional sequence IDs; default uses all fitted train sequences.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mesh-edge-stride", type=int, default=8, help="Draw every Nth unique SMPL mesh edge.")
    parser.add_argument("--mesh-alpha", type=int, default=105, help="Mesh RGBA alpha in 0..255.")
    parser.add_argument(
        "--x-offset-px",
        type=float,
        default=0.0,
        help="Horizontal pixel offset added after camera projection; negative values move overlays left.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap per sequence; 0 renders all matched frames.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-preview", action="store_true", help="Also save the first rendered frame as PNG.")
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    calib = load_json(path)
    if not isinstance(calib, dict):
        raise ValueError(f"{path} must contain a JSON object")
    intrinsic = np.asarray(calib["intrinsic"], dtype=np.float64).reshape(3, 4)
    extrinsic = np.asarray(calib["extrinsic"], dtype=np.float64).reshape(4, 4)
    return intrinsic, extrinsic


def smpl_world_to_train(points: np.ndarray) -> np.ndarray:
    """Convert WiTwin/radar world [right, up, back] to Train/LiDAR [x, y, z]."""
    pts = np.asarray(points, dtype=np.float64)
    return np.stack((-pts[..., 2], pts[..., 0], pts[..., 1]), axis=-1)


def project_points(points_train: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_train, dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((*pts.shape[:-1], 1), dtype=np.float64)], axis=-1)
    flat = pts_h.reshape(-1, 4).T
    cam = extrinsic @ flat
    proj = intrinsic @ cam
    z = cam[2].reshape(pts.shape[:-1])
    uv = (proj[:2] / np.maximum(proj[2:3], 1e-9)).T.reshape((*pts.shape[:-1], 2))
    return uv, z


def mesh_edges(faces: np.ndarray, stride: int) -> np.ndarray:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges.astype(np.int32, copy=False), axis=1)
    edges = np.unique(edges, axis=0)
    stride = max(int(stride), 1)
    return edges[::stride]


def pose_by_frame(sequence_block: dict[str, object], annotation_index: int = 0) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    for frame_key, annotations in sequence_block.items():
        if not annotations or annotation_index >= len(annotations):
            continue
        pose = annotations[annotation_index].get("pose")
        if pose is None:
            continue
        poses[str(frame_key).zfill(6)] = np.asarray(pose, dtype=np.float64)
    return poses


def selected_sequence_ids(train: dict[str, object], smpl_root: Path, requested: list[int] | None) -> list[int]:
    fitted = {
        int(path.name.split("_", 1)[0][3:])
        for path in smpl_root.glob("seq*_joint_labels_temporal_v10_temporal_fit.npz")
    }
    if requested:
        seqs = [int(seq) for seq in requested]
    else:
        seqs = [int(seq) for seq in train.keys()]
    return [seq for seq in sorted(seqs) if seq in fitted and str(seq) in train]


def ffmpeg_process(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def draw_mesh(draw: ImageDraw.ImageDraw, uv: np.ndarray, z: np.ndarray, edges: np.ndarray, size: tuple[int, int], alpha: int) -> int:
    width, height = size
    margin = 32.0
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 0.05)
        & (uv[:, 0] >= -margin)
        & (uv[:, 0] <= width + margin)
        & (uv[:, 1] >= -margin)
        & (uv[:, 1] <= height + margin)
    )
    edge_valid = valid[edges[:, 0]] & valid[edges[:, 1]]
    color = (18, 210, 190, int(np.clip(alpha, 0, 255)))
    count = 0
    for a, b in edges[edge_valid]:
        draw.line((float(uv[a, 0]), float(uv[a, 1]), float(uv[b, 0]), float(uv[b, 1])), fill=color, width=1)
        count += 1
    return count


def draw_keypoints(draw: ImageDraw.ImageDraw, uv: np.ndarray, z: np.ndarray, size: tuple[int, int]) -> int:
    width, height = size
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 0.05)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    for a, b in RT_BONES:
        if valid[a] and valid[b]:
            draw.line((float(uv[a, 0]), float(uv[a, 1]), float(uv[b, 0]), float(uv[b, 1])), fill=(255, 210, 40), width=4)
            draw.line((float(uv[a, 0]), float(uv[a, 1]), float(uv[b, 0]), float(uv[b, 1])), fill=(180, 30, 30), width=2)
    for idx, ok in enumerate(valid):
        if not ok:
            continue
        x, y = float(uv[idx, 0]), float(uv[idx, 1])
        r = 5 if idx in (0, 7, 8) else 4
        draw.ellipse((x - r - 1, y - r - 1, x + r + 1, y + r + 1), fill=(255, 255, 255))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(235, 35, 35))
    return int(valid.sum())


def render_frame(
    image_path: Path,
    vertices_train: np.ndarray,
    pose_train: np.ndarray,
    edges: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    mesh_alpha: int,
    x_offset_px: float,
) -> tuple[Image.Image, int, int]:
    image = Image.open(image_path).convert("RGB")
    size = image.size
    mesh_uv, mesh_z = project_points(vertices_train, intrinsic, extrinsic)
    pose_uv, pose_z = project_points(pose_train, intrinsic, extrinsic)
    mesh_uv[..., 0] += float(x_offset_px)
    pose_uv[..., 0] += float(x_offset_px)

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    mesh_edges_drawn = draw_mesh(overlay_draw, mesh_uv, mesh_z, edges, size, mesh_alpha)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(image)
    keypoints_drawn = draw_keypoints(draw, pose_uv, pose_z, size)
    return image, mesh_edges_drawn, keypoints_drawn


def render_sequence(task: dict[str, object]) -> dict[str, object]:
    seq = int(task["sequence"])
    train_block = task["train_block"]
    gt_root = Path(task["gt_root"])
    smpl_root = Path(task["smpl_root"])
    out_root = Path(task["out_root"])
    intrinsic = np.asarray(task["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(task["extrinsic"], dtype=np.float64)
    fps = float(task["fps"])
    overwrite = bool(task["overwrite"])
    max_frames = int(task["max_frames"])
    mesh_alpha = int(task["mesh_alpha"])
    edge_stride = int(task["mesh_edge_stride"])
    x_offset_px = float(task["x_offset_px"])
    write_preview = bool(task["write_preview"])

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"sequence{seq:03d}_left_keypoints_smpl_10fps.mp4"
    preview_path = out_root / f"sequence{seq:03d}_left_keypoints_smpl_preview.png"
    if out_path.exists() and not overwrite:
        return {"sequence": seq, "status": "skipped_existing", "out": str(out_path), "frames": 0}

    smpl_path = smpl_root / f"seq{seq}_joint_labels_temporal_v10_temporal_fit.npz"
    if not smpl_path.exists():
        return {"sequence": seq, "status": "missing_smpl", "out": str(out_path), "frames": 0}
    camera_dir = gt_root / str(seq) / "camera" / "left"
    if not camera_dir.exists():
        return {"sequence": seq, "status": "missing_camera", "out": str(out_path), "frames": 0}

    fit = np.load(smpl_path, allow_pickle=False)
    radar_to_idx = {str(frame).zfill(6): idx for idx, frame in enumerate(fit["radar_frames"].astype(str))}
    poses = pose_by_frame(train_block)
    edges = mesh_edges(fit["faces"].astype(np.int32), edge_stride)

    matched: list[tuple[str, int, np.ndarray, Path]] = []
    for camera_frame, radar_frame in zip(fit["keyframe_frames"].astype(str), fit["keyframe_radar_frames"].astype(str)):
        camera_id = str(camera_frame).zfill(6)
        radar_id = str(radar_frame).zfill(6)
        image_path = camera_dir / f"{camera_id}.png"
        pose = poses.get(camera_id)
        idx = radar_to_idx.get(radar_id)
        if idx is None or pose is None or not image_path.exists():
            continue
        matched.append((camera_id, idx, pose, image_path))
    if max_frames > 0:
        matched = matched[:max_frames]
    if not matched:
        return {"sequence": seq, "status": "no_matched_frames", "out": str(out_path), "frames": 0}

    first_image = Image.open(matched[0][3])
    width, height = first_image.size
    proc = ffmpeg_process(out_path, width, height, fps)
    if proc.stdin is None:
        raise RuntimeError("ffmpeg stdin was not opened")

    total_mesh_edges = 0
    total_keypoints = 0
    try:
        for frame_index, (_camera_id, smpl_idx, pose, image_path) in enumerate(matched):
            vertices_train = smpl_world_to_train(fit["vertices"][smpl_idx])
            frame, mesh_count, keypoint_count = render_frame(
                image_path,
                vertices_train,
                pose,
                edges,
                intrinsic,
                extrinsic,
                mesh_alpha,
                x_offset_px,
            )
            if frame_index == 0 and write_preview:
                frame.save(preview_path)
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
            total_mesh_edges += mesh_count
            total_keypoints += keypoint_count
    finally:
        proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed for sequence {seq} with exit code {ret}")

    return {
        "sequence": seq,
        "status": "completed",
        "out": str(out_path),
        "frames": len(matched),
        "fps": fps,
        "duration_s": len(matched) / fps,
        "mesh_edges_per_frame_mean": total_mesh_edges / max(len(matched), 1),
        "keypoints_per_frame_mean": total_keypoints / max(len(matched), 1),
        "x_offset_px": x_offset_px,
        "preview": str(preview_path) if write_preview else "",
    }


def write_summary(out_root: Path, rows: list[dict[str, object]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "camera_overlay_summary.json"
    csv_path = out_root / "camera_overlay_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "sequence",
        "status",
        "frames",
        "fps",
        "duration_s",
        "mesh_edges_per_frame_mean",
        "keypoints_per_frame_mean",
        "x_offset_px",
        "out",
        "preview",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["sequence"])):
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    train = load_json(Path(args.train))
    if not isinstance(train, dict):
        raise ValueError("--train must contain a JSON object")
    intrinsic, extrinsic = load_calibration(Path(args.calib))
    smpl_root = Path(args.smpl_root)
    out_root = Path(args.out_root)
    seqs = selected_sequence_ids(train, smpl_root, args.sequences)
    if not seqs:
        raise RuntimeError("No sequences selected")

    tasks = [
        {
            "sequence": seq,
            "train_block": train[str(seq)],
            "gt_root": args.gt_root,
            "smpl_root": args.smpl_root,
            "out_root": args.out_root,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "fps": args.fps,
            "overwrite": args.overwrite,
            "max_frames": args.max_frames,
            "mesh_alpha": args.mesh_alpha,
            "mesh_edge_stride": args.mesh_edge_stride,
            "write_preview": args.write_preview,
            "x_offset_px": args.x_offset_px,
        }
        for seq in seqs
    ]

    rows: list[dict[str, object]] = []
    workers = max(int(args.workers), 1)
    if workers == 1:
        for task in tasks:
            row = render_sequence(task)
            rows.append(row)
            write_summary(out_root, rows)
            print(json.dumps(row, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_seq = {pool.submit(render_sequence, task): int(task["sequence"]) for task in tasks}
            for future in as_completed(future_to_seq):
                seq = future_to_seq[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"sequence": seq, "status": "failed", "frames": 0, "out": "", "error": str(exc)}
                rows.append(row)
                write_summary(out_root, rows)
                print(json.dumps(row, sort_keys=True), flush=True)

    failed = [row for row in rows if row.get("status") not in {"completed", "skipped_existing"}]
    if failed:
        print(f"{len(failed)} sequences failed or missing; see {out_root / 'camera_overlay_summary.json'}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
