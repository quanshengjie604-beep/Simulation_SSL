#!/usr/bin/env python3
"""Evaluate RT-Pose sequence-level pose metrics from prediction JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MRPE, MPJPE, and Abs-MPJPE for one RT-Pose sequence."
    )
    parser.add_argument("--train", default=str(REPO_ROOT / "datasets" / "Train.json"), help="Ground-truth annotation JSON")
    parser.add_argument("--sequence", default="185", help="Sequence id to evaluate")
    parser.add_argument(
        "--prediction",
        default="",
        help=(
            "Prediction JSON. Supports raw detections keyed by seq/frame/radar_frame, "
            "or tools/test.py saved style keyed by sequence then frame_radar."
        ),
    )
    parser.add_argument(
        "--oracle-gt",
        action="store_true",
        help="Use ground truth as prediction; only for metric sanity checking.",
    )
    parser.add_argument("--out", default="", help="Optional output JSON path")
    return parser.parse_args()


def sorted_numeric_keys(keys) -> list[str]:
    return sorted(keys, key=lambda item: int(item) if str(item).isdigit() else str(item))


def load_gt(train_path: Path, sequence: str) -> dict[tuple[str, str], np.ndarray]:
    with train_path.open("r", encoding="utf-8") as f:
        train = json.load(f)
    if sequence not in train:
        raise KeyError(f"Sequence {sequence!r} was not found in {train_path}")

    gt: dict[tuple[str, str], np.ndarray] = {}
    for frame in sorted_numeric_keys(train[sequence].keys()):
        annotations = train[sequence][frame]
        if not annotations:
            continue
        obj = annotations[0]
        radar_frame = obj.get("Radar_frameID")
        pose = obj.get("pose")
        if radar_frame is None or pose is None:
            continue
        pose_arr = np.asarray(pose, dtype=np.float64)
        if pose_arr.shape != (15, 3):
            raise ValueError(f"GT {sequence}/{frame}/{radar_frame} has unexpected pose shape {pose_arr.shape}")
        gt[(frame, radar_frame)] = pose_arr
    if not gt:
        raise RuntimeError(f"No valid GT poses found for sequence {sequence}")
    return gt


def keypoints_to_pose(keypoints: list[Any]) -> np.ndarray | None:
    if len(keypoints) < 15:
        return None
    pose = np.full((15, 3), np.nan, dtype=np.float64)
    for item in keypoints:
        if isinstance(item, dict):
            label = int(item.get("label", item.get("id", item.get("keypoint_id", -1))))
            xyz = [item.get("x"), item.get("y"), item.get("z")]
        else:
            if len(item) < 4:
                continue
            label = int(item[0])
            xyz = item[1:4]
        if 0 <= label < 15:
            pose[label] = np.asarray(xyz, dtype=np.float64)
    if not np.isfinite(pose).all():
        return None
    return pose


def load_prediction(pred_path: Path, sequence: str) -> dict[tuple[str, str], np.ndarray]:
    with pred_path.open("r", encoding="utf-8") as f:
        pred = json.load(f)

    out: dict[tuple[str, str], np.ndarray] = {}

    # Raw detections style: {"seq/frame/radar": {"keypoints": ...}}
    for key, val in pred.items():
        if isinstance(key, str) and key.count("/") == 2 and isinstance(val, dict):
            seq, frame, radar_frame = key.split("/")
            if seq == sequence and "keypoints" in val:
                pose = keypoints_to_pose(val["keypoints"])
                if pose is not None:
                    out[(frame, radar_frame)] = pose

    if out:
        return out

    # Saved style from tools/test.py: {"seq_or_name": {"frame_radar": {"keypoints": ...}}}
    candidate_blocks = []
    if sequence in pred and isinstance(pred[sequence], dict):
        candidate_blocks.append(pred[sequence])
    for maybe_block in pred.values():
        if isinstance(maybe_block, dict):
            candidate_blocks.append(maybe_block)

    for block in candidate_blocks:
        for frame_key, val in block.items():
            if not isinstance(frame_key, str) or "_" not in frame_key or not isinstance(val, dict):
                continue
            if "keypoints" not in val:
                continue
            frame, radar_frame = frame_key.split("_", 1)
            pose = keypoints_to_pose(val["keypoints"])
            if pose is not None:
                out[(frame, radar_frame)] = pose
    return out


def compute_metrics(gt: dict[tuple[str, str], np.ndarray], pred: dict[tuple[str, str], np.ndarray]) -> dict[str, Any]:
    common = sorted(set(gt) & set(pred), key=lambda item: (int(item[0]), int(item[1])))
    if not common:
        raise RuntimeError("No overlapping frame/radar_frame keys between GT and predictions")

    root_errors = []
    rel_joint_errors = []
    abs_joint_errors = []
    for key in common:
        gt_pose = gt[key].copy()
        pred_pose = pred[key].copy()
        root_errors.append(np.linalg.norm(pred_pose[0] - gt_pose[0]))
        rel_joint_errors.append(np.linalg.norm((pred_pose - pred_pose[:1]) - (gt_pose - gt_pose[:1]), axis=1))
        abs_joint_errors.append(np.linalg.norm(pred_pose - gt_pose, axis=1))

    root_errors = np.asarray(root_errors)
    rel_joint_errors = np.asarray(rel_joint_errors)
    abs_joint_errors = np.asarray(abs_joint_errors)
    return {
        "frames_evaluated": int(len(common)),
        "MRPE_mm": float(root_errors.mean() * 1000.0),
        "MPJPE_mm": float(rel_joint_errors.mean() * 1000.0),
        "Abs-MPJPE_mm": float(abs_joint_errors.mean() * 1000.0),
        "PJPE_per_joint_mm": (rel_joint_errors.mean(axis=0) * 1000.0).tolist(),
        "Abs-PJPE_per_joint_mm": (abs_joint_errors.mean(axis=0) * 1000.0).tolist(),
    }


def main() -> None:
    args = parse_args()
    gt = load_gt(Path(args.train), args.sequence)
    if args.oracle_gt:
        pred = {key: pose.copy() for key, pose in gt.items()}
    else:
        if not args.prediction:
            raise SystemExit("Provide --prediction, or use --oracle-gt for a formula sanity check.")
        pred = load_prediction(Path(args.prediction), args.sequence)

    result = compute_metrics(gt, pred)
    result["sequence"] = args.sequence
    result["gt_frames"] = len(gt)
    result["prediction_frames_loaded"] = len(pred)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
