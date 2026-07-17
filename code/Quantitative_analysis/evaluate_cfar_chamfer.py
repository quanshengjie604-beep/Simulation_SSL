#!/usr/bin/env python3
"""Evaluate GT-vs-generated ROI cache point clouds with CFAR and Chamfer distance."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "Echo_data_processing"))

from cfar_point_cloud import (  # noqa: E402
    axes_for_power_shape,
    ca_cfar_3d,
    detections_to_point_cloud,
    power_zyx,
    xyz_bounds_mask,
    xyz_to_zyx,
)
from raw_echo_to_xyz import RadarConfig, xyz_axes  # noqa: E402


POSE_BBOX_MARGIN_M = 0.25


def parse_xyz_ints(text: str) -> tuple[int, int, int]:
    parts = [part for part in text.replace("x", ",").replace(" ", ",").split(",") if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    values = tuple(int(part) for part in parts)
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("cell counts must be non-negative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-id", required=True, help="Radar frame id, e.g. 000002 or 2.")
    parser.add_argument("--gt", default="", help="GT ROI cache .npy; defaults from --sequence/--frame-id.")
    parser.add_argument("--candidate", default="", help="Generated ROI cache .npy; defaults to Sim1.")
    parser.add_argument("--candidate-label", default="Sim1")
    parser.add_argument("--train", default=str(REPO_ROOT / "datasets" / "Train.json"))
    parser.add_argument("--annotation-index", type=int, default=0)
    parser.add_argument("--guard-cells", type=parse_xyz_ints, default=(3, 3, 2))
    parser.add_argument("--training-cells", type=parse_xyz_ints, default=(10, 10, 3))
    parser.add_argument("--pfa", type=float, default=3e-3)
    parser.add_argument("--boundary-mode", choices=("constant", "nearest", "reflect", "mirror", "wrap"), default="constant")
    parser.add_argument("--min-training-fraction", type=float, default=0.5)
    parser.add_argument("--doppler-reducer", choices=("sum", "max"), default="sum")
    parser.add_argument("--local-max", action="store_true", help="Optional NMS/local-max filter; default is off.")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    return parser.parse_args()


def default_cache_path(root_name: str, sequence: str, frame_id: str) -> Path:
    return (
        REPO_ROOT
        / "datasets"
        / root_name
        / str(int(sequence))
        / "radar"
        / "npy_DZYX_mag_roi_f16_norm"
        / f"{int(frame_id):06d}.npy"
    )


def load_pose(train_path: Path, sequence: str, frame_id: str, annotation_index: int) -> np.ndarray:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    seq_key = str(int(sequence))
    target = f"{int(frame_id):06d}"
    if seq_key not in train:
        raise KeyError(f"sequence {seq_key} not found in {train_path}")
    for annotations in train[seq_key].values():
        for index, ann in enumerate(annotations or []):
            if index != annotation_index:
                continue
            if str(ann.get("Radar_frameID", "")).zfill(6) == target:
                pose = np.asarray(ann["pose"], dtype=np.float32)
                if pose.shape != (15, 3):
                    raise ValueError(f"pose has shape {pose.shape}, expected (15, 3)")
                return pose
    raise KeyError(f"Radar_frameID {target} not found in sequence {seq_key}")


def pose_bounds(
    pose: np.ndarray,
    margin: float = POSE_BBOX_MARGIN_M,
) -> tuple[
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
]:
    finite = np.isfinite(pose).all(axis=1)
    if not np.any(finite):
        raise ValueError("pose has no finite joints")
    pose_mins = pose[finite].min(axis=0)
    pose_maxs = pose[finite].max(axis=0)
    mins = pose_mins - float(margin)
    maxs = pose_maxs + float(margin)
    pose_bbox = (
        float(pose_mins[0]),
        float(pose_maxs[0]),
        float(pose_mins[1]),
        float(pose_maxs[1]),
        float(pose_mins[2]),
        float(pose_maxs[2]),
    )
    margin_bounds = (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )
    return pose_bbox, margin_bounds


def cfar_points(path: Path, bounds: tuple[float, float, float, float, float, float], args: argparse.Namespace) -> np.ndarray:
    axes = xyz_axes(RadarConfig())
    spatial_slices = (slice(None), slice(None), slice(None))
    power = power_zyx(path, spatial_slices, args.doppler_reducer)
    point_axes, point_slices = axes_for_power_shape(axes, power.shape, spatial_slices, None)
    detections, noise, threshold = ca_cfar_3d(
        power,
        xyz_to_zyx(args.guard_cells),
        xyz_to_zyx(args.training_cells),
        args.pfa,
        boundary_mode=args.boundary_mode,
        min_training_fraction=args.min_training_fraction,
    )
    if args.local_max:
        from cfar_point_cloud import apply_local_max_filter  # noqa: E402

        detections = apply_local_max_filter(detections, power, xyz_to_zyx(args.guard_cells), args.boundary_mode)
    detections &= xyz_bounds_mask(point_axes, bounds)
    points = detections_to_point_cloud(detections, power, noise, threshold, point_axes, point_slices, max_points=0)
    return points[:, :3].astype(np.float32, copy=False)


def directed_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 and b.shape[0] == 0:
        return 0.0
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    tree = cKDTree(b.astype(np.float64, copy=False))
    dist, _ = tree.query(a.astype(np.float64, copy=False), k=1)
    return float(np.mean(dist))


def flatten(summary: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {}

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}{key}_" if prefix else f"{key}_", child)
        elif isinstance(value, (list, tuple)):
            row[prefix[:-1]] = " ".join(str(item) for item in value)
        else:
            row[prefix[:-1]] = value

    visit("", summary)
    return row


def write_csv(path: Path, summary: dict[str, object]) -> None:
    row = flatten(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    frame_id = f"{int(args.frame_id):06d}"
    gt_path = Path(args.gt) if args.gt else default_cache_path("GT_sequences", args.sequence, frame_id)
    candidate_path = Path(args.candidate) if args.candidate else default_cache_path("Sim1_sequences", args.sequence, frame_id)
    pose = load_pose(Path(args.train), args.sequence, frame_id, args.annotation_index)
    pose_bbox, bounds = pose_bounds(pose)
    gt_points = cfar_points(gt_path, bounds, args)
    candidate_points = cfar_points(candidate_path, bounds, args)

    gt_to_candidate = directed_chamfer(gt_points, candidate_points)
    candidate_to_gt = directed_chamfer(candidate_points, gt_points)
    chamfer_mean = float((gt_to_candidate + candidate_to_gt) * 0.5)
    chamfer_sum = float(gt_to_candidate + candidate_to_gt)
    summary: dict[str, object] = {
        "sequence": str(int(args.sequence)),
        "frame_id": frame_id,
        "inputs": {
            "gt": str(gt_path),
            "candidate": str(candidate_path),
            "candidate_label": args.candidate_label,
            "train": str(Path(args.train)),
        },
        "tensor": {"shape": "D,Z,Y,X = 64,16,64,160", "type": "ROI normalized float16 cache"},
        "cfar": {
            "type": "3D CA-CFAR",
            "guard_cells_xyz": list(args.guard_cells),
            "training_cells_xyz": list(args.training_cells),
            "pfa": float(args.pfa),
            "boundary_mode": args.boundary_mode,
            "min_training_fraction": float(args.min_training_fraction),
            "doppler_reducer": args.doppler_reducer,
            "nms": bool(args.local_max),
        },
        "pose": {
            "bbox_margin_m": POSE_BBOX_MARGIN_M,
            "pose_bbox": {
                "x": [pose_bbox[0], pose_bbox[1]],
                "y": [pose_bbox[2], pose_bbox[3]],
                "z": [pose_bbox[4], pose_bbox[5]],
            },
            "bbox_after_margin": {
                "x": [bounds[0], bounds[1]],
                "y": [bounds[2], bounds[3]],
                "z": [bounds[4], bounds[5]],
            },
        },
        "points": {"gt": int(gt_points.shape[0]), "candidate": int(candidate_points.shape[0])},
        "chamfer": {
            "gt_to_candidate_mean_m": gt_to_candidate,
            "candidate_to_gt_mean_m": candidate_to_gt,
            "mean_m": chamfer_mean,
            "sum_m": chamfer_sum,
        },
    }
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        write_csv(Path(args.out_csv), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
