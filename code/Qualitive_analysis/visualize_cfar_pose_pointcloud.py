#!/usr/bin/env python3
"""Plot human pose, GT CFAR points, and generated CFAR points for one frame."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rtpose-cfar-pose")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CODE_ROOT / "Quantitative_analysis"))
sys.path.insert(0, str(CODE_ROOT / "Echo_data_processing"))

from evaluate_cfar_chamfer import (  # noqa: E402
    POSE_BBOX_MARGIN_M,
    cfar_points,
    default_cache_path,
    load_pose,
    parse_xyz_ints,
)
from visualize_radar_pose_calibration import build_bones, load_keypoints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-id", required=True, help="Radar frame id, e.g. 000002 or 2.")
    parser.add_argument("--gt", default="", help="GT ROI cache .npy; defaults from --sequence/--frame-id.")
    parser.add_argument("--candidate", default="", help="Generated ROI cache .npy; defaults to Sim1.")
    parser.add_argument("--candidate-label", default="Sim1")
    parser.add_argument("--train", default=str(REPO_ROOT / "datasets" / "Train.json"))
    parser.add_argument("--keypoints", default=str(REPO_ROOT / "datasets" / "Keypoints_meta.txt"))
    parser.add_argument("--annotation-index", type=int, default=0)
    parser.add_argument("--x-range", type=float, nargs=2, default=(1.2, 6.2), metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-2.0, 2.0), metavar=("MIN", "MAX"))
    parser.add_argument("--z-range", type=float, nargs=2, default=(-1.5, 1.5), metavar=("MIN", "MAX"))
    parser.add_argument("--roi-z-margin", type=float, default=1.0, help="Meters added around the pose z range for CFAR ROI.")
    parser.add_argument("--guard-cells", type=parse_xyz_ints, default=(3, 3, 2))
    parser.add_argument("--training-cells", type=parse_xyz_ints, default=(10, 10, 3))
    parser.add_argument("--pfa", type=float, default=3e-3)
    parser.add_argument("--boundary-mode", choices=("constant", "nearest", "reflect", "mirror", "wrap"), default="constant")
    parser.add_argument("--min-training-fraction", type=float, default=0.5)
    parser.add_argument("--doppler-reducer", choices=("sum", "max"), default="sum")
    parser.add_argument("--local-max", action="store_true", help="Optional NMS/local-max filter; default is off.")
    parser.add_argument("--out", default="", help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def validate_range(name: str, values: tuple[float, float] | list[float]) -> tuple[float, float]:
    low, high = float(values[0]), float(values[1])
    if not low < high:
        raise ValueError(f"{name} requires MIN < MAX")
    return low, high


def set_axes(ax: plt.Axes, args: argparse.Namespace) -> None:
    x0, x1 = validate_range("--x-range", args.x_range)
    y0, y1 = validate_range("--y-range", args.y_range)
    z0, z1 = validate_range("--z-range", args.z_range)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=18, azim=-62)
    ax.grid(True, linewidth=0.4, alpha=0.35)


def cfar_roi_bounds(
    pose: np.ndarray,
    xy_margin: float,
    z_margin: float,
) -> tuple[float, float, float, float, float, float]:
    if z_margin < 0:
        raise ValueError("--roi-z-margin must be non-negative")
    finite = np.isfinite(pose).all(axis=1)
    if not np.any(finite):
        raise ValueError("pose has no finite joints")
    pose_mins = pose[finite].min(axis=0)
    pose_maxs = pose[finite].max(axis=0)
    return (
        float(pose_mins[0] - xy_margin),
        float(pose_maxs[0] + xy_margin),
        float(pose_mins[1] - xy_margin),
        float(pose_maxs[1] + xy_margin),
        float(pose_mins[2] - z_margin),
        float(pose_maxs[2] + z_margin),
    )


def draw_pose(ax: plt.Axes, pose: np.ndarray, bones: list[tuple[int, int]]) -> None:
    finite = np.isfinite(pose).all(axis=1)
    for a, b in bones:
        if a >= pose.shape[0] or b >= pose.shape[0] or not (finite[a] and finite[b]):
            continue
        segment = pose[[a, b]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#111827", linewidth=2.0)
    if np.any(finite):
        ax.scatter(pose[finite, 0], pose[finite, 1], pose[finite, 2], s=18, c="#ef4444", depthshade=False)


def draw_points(
    ax: plt.Axes,
    points: np.ndarray,
    title: str,
    color: str,
) -> None:
    if points.shape[0] > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=12, c=color, alpha=0.88, depthshade=False)
    else:
        ax.text2D(0.5, 0.5, "0 CFAR points", transform=ax.transAxes, ha="center", va="center", color="#6b7280")
    ax.set_title(f"{title} ({points.shape[0]} pts)")


def main() -> None:
    args = parse_args()
    frame_id = f"{int(args.frame_id):06d}"
    gt_path = Path(args.gt) if args.gt else default_cache_path("GT_sequences", args.sequence, frame_id)
    candidate_path = Path(args.candidate) if args.candidate else default_cache_path("Sim1_sequences", args.sequence, frame_id)
    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT
        / "results"
        / "Qualitive_analysis"
        / "cfar_pose_pointcloud"
        / f"sequence{int(args.sequence)}_frame{frame_id}_{args.candidate_label}_vs_GT.png"
    )

    pose = load_pose(Path(args.train), args.sequence, frame_id, args.annotation_index)
    bounds = cfar_roi_bounds(pose, POSE_BBOX_MARGIN_M, args.roi_z_margin)
    gt_points = cfar_points(gt_path, bounds, args)
    candidate_points = cfar_points(candidate_path, bounds, args)
    bones = build_bones(load_keypoints(Path(args.keypoints)))

    fig = plt.figure(figsize=(15.2, 5.2))
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    draw_pose(axes[0], pose, bones)
    axes[0].set_title("Pose")
    draw_points(axes[1], gt_points, "GT CFAR", "#2563eb")
    draw_points(axes[2], candidate_points, f"{args.candidate_label} CFAR", "#dc2626")
    for ax in axes:
        set_axes(ax, args)
    fig.suptitle(
        f"Seq {int(args.sequence)} Frame {frame_id} | ROI xy margin {POSE_BBOX_MARGIN_M:.2f} m, "
        f"z margin {args.roi_z_margin:.2f} m | CFAR guard {args.guard_cells}, "
        f"training {args.training_cells}, PFA {args.pfa:g}, NMS {'on' if args.local_max else 'off'}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"GT points: {gt_points.shape[0]}")
    print(f"{args.candidate_label} points: {candidate_points.shape[0]}")


if __name__ == "__main__":
    main()
