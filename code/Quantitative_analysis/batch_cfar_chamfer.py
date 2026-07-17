#!/usr/bin/env python3
"""Batch CFAR point-cloud Chamfer evaluation over RT-Pose label frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate_cfar_chamfer import (  # noqa: E402
    POSE_BBOX_MARGIN_M,
    cfar_points,
    directed_chamfer,
    parse_xyz_ints,
    pose_bounds,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


FRAME_METRIC_FIELDS = [
    "sequence",
    "activity",
    "frame_id",
    "gt_path",
    "candidate_path",
    "gt_points",
    "candidate_points",
    "gt_to_candidate_mean_m",
    "candidate_to_gt_mean_m",
    "chamfer_mean_m",
    "chamfer_sum_m",
    "bbox_margin_m",
    "bbox_x_min",
    "bbox_x_max",
    "bbox_y_min",
    "bbox_y_max",
    "bbox_z_min",
    "bbox_z_max",
    "pose_bbox_x_min",
    "pose_bbox_x_max",
    "pose_bbox_y_min",
    "pose_bbox_y_max",
    "pose_bbox_z_min",
    "pose_bbox_z_max",
]


SKIPPED_FIELDS = [
    "sequence",
    "frame_id",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(REPO_ROOT / "datasets" / "Train_sp120_train_minus_val6.json"))
    parser.add_argument("--filemeta", default=str(REPO_ROOT / "datasets" / "filemeta.json"))
    parser.add_argument("--gt-root", default=str(REPO_ROOT / "datasets" / "GT_sequences"))
    parser.add_argument("--candidate-root", default=str(REPO_ROOT / "datasets" / "Sim1_sequences"))
    parser.add_argument("--candidate-label", default="Sim1")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "Quantitative_analysis" / "sim1vsgt" / "RAE"))
    parser.add_argument("--sequences", type=int, nargs="*", help="Optional sequence subset; default: all in --train.")
    parser.add_argument("--annotation-index", type=int, default=0)
    parser.add_argument("--guard-cells", type=parse_xyz_ints, default=(3, 3, 2))
    parser.add_argument("--training-cells", type=parse_xyz_ints, default=(10, 10, 3))
    parser.add_argument("--pfa", type=float, default=3e-3)
    parser.add_argument("--boundary-mode", choices=("constant", "nearest", "reflect", "mirror", "wrap"), default="constant")
    parser.add_argument("--min-training-fraction", type=float, default=0.5)
    parser.add_argument("--doppler-reducer", choices=("sum", "max"), default="sum")
    parser.add_argument("--local-max", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def activity_map(path: Path) -> dict[int, str]:
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    out: dict[int, str] = {}
    for seq, meta in data.items():
        out[int(seq)] = str(meta.get("Activity", "UNKNOWN")) if isinstance(meta, dict) else "UNKNOWN"
    return out


def sequence_ids(train: dict[str, object], selected: list[int] | None) -> list[int]:
    if selected:
        return [int(seq) for seq in selected]
    return [int(seq) for seq in sorted(train, key=lambda item: int(item))]


def cache_path(root: Path, seq: int, frame_id: str) -> Path:
    return root / str(int(seq)) / "radar" / "npy_DZYX_mag_roi_f16_norm" / f"{int(frame_id):06d}.npy"


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    return float(np.mean(arr[finite])) if np.any(finite) else float("nan")


def finite_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    return float(np.std(arr[finite])) if np.any(finite) else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class IncrementalCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()
        self.handle.flush()

    def write(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "IncrementalCsvWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def summarize_activity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["activity"]), []).append(row)
    metrics = [
        "gt_points",
        "candidate_points",
        "gt_to_candidate_mean_m",
        "candidate_to_gt_mean_m",
        "chamfer_mean_m",
        "chamfer_sum_m",
    ]
    out: list[dict[str, object]] = []
    for activity in sorted(grouped):
        activity_rows = grouped[activity]
        summary: dict[str, object] = {
            "activity": activity,
            "num_frames": len(activity_rows),
            "num_sequences": len({row["sequence"] for row in activity_rows}),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in activity_rows]
            summary[f"mean_{metric}"] = finite_mean(values)
            summary[f"std_{metric}"] = finite_std(values)
        out.append(summary)
    return out


def build_summary(
    args: argparse.Namespace,
    gt_root: Path,
    candidate_root: Path,
    out_dir: Path,
    frame_rows: list[dict[str, object]],
    skipped_rows: list[dict[str, object]],
    status: str,
    last_sequence: int | None,
) -> dict[str, object]:
    return {
        "train": str(Path(args.train)),
        "filemeta": str(Path(args.filemeta)),
        "gt_root": str(gt_root),
        "candidate_root": str(candidate_root),
        "candidate_label": str(args.candidate_label),
        "output_dir": str(out_dir),
        "status": status,
        "last_sequence": last_sequence,
        "num_frames": len(frame_rows),
        "num_skipped": len(skipped_rows),
        "cfar": {
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
        },
    }


def refresh_summary_files(
    out_dir: Path,
    frame_rows: list[dict[str, object]],
    skipped_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    write_csv(out_dir / "activity_summary.csv", summarize_activity(frame_rows))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train = load_json(Path(args.train))
    if not isinstance(train, dict):
        raise ValueError("--train must contain a JSON object")

    gt_root = Path(args.gt_root)
    candidate_root = Path(args.candidate_root)
    activities = activity_map(Path(args.filemeta))
    cfar_args = SimpleNamespace(
        guard_cells=args.guard_cells,
        training_cells=args.training_cells,
        pfa=args.pfa,
        boundary_mode=args.boundary_mode,
        min_training_fraction=args.min_training_fraction,
        doppler_reducer=args.doppler_reducer,
        local_max=args.local_max,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    last_sequence: int | None = None
    summary = build_summary(args, gt_root, candidate_root, out_dir, frame_rows, skipped_rows, "running", last_sequence)
    refresh_summary_files(out_dir, frame_rows, skipped_rows, summary)

    with (
        IncrementalCsvWriter(out_dir / "frame_metrics.csv", FRAME_METRIC_FIELDS) as frame_writer,
        IncrementalCsvWriter(out_dir / "skipped_frames.csv", SKIPPED_FIELDS) as skipped_writer,
    ):
        for seq in sequence_ids(train, args.sequences):
            seq_key = str(int(seq))
            seq_block = train.get(seq_key)
            if not isinstance(seq_block, dict):
                row = {"sequence": seq, "frame_id": "", "reason": "sequence missing from train json"}
                skipped_rows.append(row)
                skipped_writer.write(row)
                continue
            seen_frames: set[str] = set()
            for frame_key in sorted(seq_block, key=lambda item: int(item) if str(item).isdigit() else str(item)):
                annotations = seq_block[frame_key] or []
                if args.annotation_index >= len(annotations):
                    row = {"sequence": seq, "frame_id": frame_key, "reason": "annotation index missing"}
                    skipped_rows.append(row)
                    skipped_writer.write(row)
                    continue
                ann = annotations[args.annotation_index]
                frame_id = str(ann.get("Radar_frameID", "")).zfill(6)
                if not frame_id or frame_id in seen_frames:
                    continue
                seen_frames.add(frame_id)
                try:
                    pose = np.asarray(ann["pose"], dtype=np.float32)
                    if pose.shape != (15, 3):
                        raise ValueError(f"pose shape {pose.shape}, expected (15, 3)")
                    gt_path = cache_path(gt_root, seq, frame_id)
                    candidate_path = cache_path(candidate_root, seq, frame_id)
                    if not gt_path.exists():
                        raise FileNotFoundError(f"missing GT cache {gt_path}")
                    if not candidate_path.exists():
                        raise FileNotFoundError(f"missing candidate cache {candidate_path}")
                    pose_bbox, bounds = pose_bounds(pose, POSE_BBOX_MARGIN_M)
                    gt_points = cfar_points(gt_path, bounds, cfar_args)
                    candidate_points = cfar_points(candidate_path, bounds, cfar_args)
                    gt_to_candidate = directed_chamfer(gt_points, candidate_points)
                    candidate_to_gt = directed_chamfer(candidate_points, gt_points)
                    row = {
                        "sequence": int(seq),
                        "activity": activities.get(int(seq), "UNKNOWN"),
                        "frame_id": frame_id,
                        "gt_path": str(gt_path),
                        "candidate_path": str(candidate_path),
                        "gt_points": int(gt_points.shape[0]),
                        "candidate_points": int(candidate_points.shape[0]),
                        "gt_to_candidate_mean_m": gt_to_candidate,
                        "candidate_to_gt_mean_m": candidate_to_gt,
                        "chamfer_mean_m": float((gt_to_candidate + candidate_to_gt) * 0.5),
                        "chamfer_sum_m": float(gt_to_candidate + candidate_to_gt),
                        "bbox_margin_m": float(POSE_BBOX_MARGIN_M),
                        "bbox_x_min": bounds[0],
                        "bbox_x_max": bounds[1],
                        "bbox_y_min": bounds[2],
                        "bbox_y_max": bounds[3],
                        "bbox_z_min": bounds[4],
                        "bbox_z_max": bounds[5],
                        "pose_bbox_x_min": pose_bbox[0],
                        "pose_bbox_x_max": pose_bbox[1],
                        "pose_bbox_y_min": pose_bbox[2],
                        "pose_bbox_y_max": pose_bbox[3],
                        "pose_bbox_z_min": pose_bbox[4],
                        "pose_bbox_z_max": pose_bbox[5],
                    }
                    frame_rows.append(row)
                    frame_writer.write(row)
                except Exception as exc:
                    row = {"sequence": seq, "frame_id": frame_id, "reason": str(exc)}
                    skipped_rows.append(row)
                    skipped_writer.write(row)

            last_sequence = int(seq)
            summary = build_summary(args, gt_root, candidate_root, out_dir, frame_rows, skipped_rows, "running", last_sequence)
            refresh_summary_files(out_dir, frame_rows, skipped_rows, summary)

    summary = build_summary(args, gt_root, candidate_root, out_dir, frame_rows, skipped_rows, "completed", last_sequence)
    refresh_summary_files(out_dir, frame_rows, skipped_rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
