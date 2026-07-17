#!/usr/bin/env python3
"""Diagnose frame-to-frame jumps in Sim2 SMPL-mesh scatterers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate_smpl_mesh_witwin_echo as sim


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-npz", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("pytorch", "dirichlet", "slang"), default="pytorch")
    parser.add_argument("--rx-order", choices=("attachment", "rtpose_raw"), default="attachment")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--epsilon-r", type=float, default=5.0)
    parser.add_argument("--intensity-threshold", type=float, default=1e-14)
    parser.add_argument("--foot-mask-radius-m", type=float, default=0.30)
    parser.add_argument("--foot-mask-height-m", type=float, default=0.08)
    parser.add_argument("--foot-ankle-area-scale", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=512)
    parser.add_argument("--keep-foot-mesh", dest="replace_foot_mesh_with_ankle_scatterers", action="store_false")
    parser.set_defaults(replace_foot_mesh_with_ankle_scatterers=True)
    return parser.parse_args()


def load_frame_window(fit_path: Path, frames: list[int], context: int) -> dict[str, np.ndarray]:
    fit = np.load(fit_path)
    all_radar = fit["radar_frames"].astype(int)
    selected: list[int] = []
    for frame in frames:
        for neighbor in range(frame - context, frame + context + 1):
            if neighbor in all_radar:
                selected.append(neighbor)
    selected_frames = np.asarray(sorted(set(selected)), dtype=np.int32)
    if selected_frames.size < 2:
        raise RuntimeError("Need at least two selected frames to diagnose jumps.")
    indices = np.asarray([int(np.flatnonzero(all_radar == frame)[0]) for frame in selected_frames], dtype=np.int64)

    rtpose_joints = fit["rtpose_world_keypoints"][indices].astype(np.float32)
    smpl_joints = (
        fit["smpl_joints"][indices].astype(np.float32)
        if "smpl_joints" in fit.files and fit["smpl_joints"].shape == fit["rtpose_world_keypoints"].shape
        else rtpose_joints
    )
    fallback = ~np.isfinite(rtpose_joints).all(axis=2)
    joints = rtpose_joints.copy()
    joints[fallback] = smpl_joints[fallback]

    return {
        "radar_frames": selected_frames,
        "vertices": fit["vertices"][indices].astype(np.float32),
        "faces": fit["faces"].astype(np.int32),
        "joints": joints,
        "fallback": fallback,
    }


def top_indices(intensity: np.ndarray, threshold: float, top_n: int) -> np.ndarray:
    candidates = np.flatnonzero(intensity > threshold)
    if top_n > 0 and candidates.size > top_n:
        values = intensity[candidates]
        keep = np.argpartition(values, -top_n)[-top_n:]
        candidates = candidates[keep]
    return np.sort(candidates.astype(np.int64))


def jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int, int]:
    aa = set(int(x) for x in a)
    bb = set(int(x) for x in b)
    inter = len(aa & bb)
    union = len(aa | bb)
    return (inter / union if union else 1.0), inter, len(aa - bb), len(bb - aa)


def print_frame_stats(frames: np.ndarray, scatter: dict[str, np.ndarray], threshold: float, top_n: int) -> list[np.ndarray]:
    selected_top: list[np.ndarray] = []
    print("Per-frame scatterer stats:")
    for idx, frame in enumerate(frames):
        intensity = scatter["intensity"][idx]
        candidates = np.flatnonzero(intensity > threshold)
        top = top_indices(intensity, threshold, top_n)
        selected_top.append(top)
        top_values = intensity[top] if top.size else np.asarray([], dtype=np.float32)
        visible = scatter["visible"][idx]
        print(
            f"  frame {int(frame):06d}: traced_visible={int(scatter['trace_counts'][idx])} "
            f"visible_mask={int(visible.sum())} candidates={candidates.size} top={top.size} "
            f"top_sum={float(top_values.sum()):.6g} top_p50={float(np.percentile(top_values, 50)) if top_values.size else 0:.6g} "
            f"top_p99={float(np.percentile(top_values, 99)) if top_values.size else 0:.6g} "
            f"top_max={float(top_values.max()) if top_values.size else 0:.6g}",
            flush=True,
        )
    return selected_top


def print_pair_stats(
    frames: np.ndarray,
    scatter: dict[str, np.ndarray],
    selected_top: list[np.ndarray],
) -> None:
    print("\nAdjacent-frame jump stats:")
    for idx in range(len(frames) - 1):
        f0 = int(frames[idx])
        f1 = int(frames[idx + 1])
        if f1 != f0 + 1:
            print(f"  {f0:06d}->{f1:06d}: skipped non-consecutive selected window", flush=True)
            continue
        visible0 = np.flatnonzero(scatter["visible"][idx])
        visible1 = np.flatnonzero(scatter["visible"][idx + 1])
        vj, vinter, vexit, venter = jaccard(visible0, visible1)
        tj, tinter, texit, tenter = jaccard(selected_top[idx], selected_top[idx + 1])

        common = np.intersect1d(selected_top[idx], selected_top[idx + 1], assume_unique=True)
        if common.size:
            disp = np.linalg.norm(
                scatter["centroids"][idx + 1, common] - scatter["centroids"][idx, common],
                axis=1,
            )
            ratio = scatter["intensity"][idx + 1, common] / np.maximum(scatter["intensity"][idx, common], 1e-30)
            disp_text = f"common_disp_p50={np.percentile(disp, 50):.4f}m common_disp_p95={np.percentile(disp, 95):.4f}m"
            ratio_text = f"common_int_ratio_p50={np.percentile(ratio, 50):.3g} p95={np.percentile(ratio, 95):.3g}"
        else:
            disp_text = "common_disp_p50=nan common_disp_p95=nan"
            ratio_text = "common_int_ratio_p50=nan p95=nan"

        print(
            f"  {f0:06d}->{f1:06d}: "
            f"trace_count {int(scatter['trace_counts'][idx])}->{int(scatter['trace_counts'][idx + 1])} "
            f"|diff|={abs(int(scatter['trace_counts'][idx + 1]) - int(scatter['trace_counts'][idx]))}; "
            f"visible_jaccard={vj:.3f} inter={vinter} exit={vexit} enter={venter}; "
            f"top_jaccard={tj:.3f} inter={tinter} exit={texit} enter={tenter}; "
            f"{disp_text}; {ratio_text}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    window = load_frame_window(Path(args.fit_npz), args.frames, args.context)
    print(f"selected radar frames: {window['radar_frames'].tolist()}", flush=True)
    print(f"joint fallback count in window: {int(window['fallback'].sum())}", flush=True)

    scatter = sim.trace_dense_keyframes(args, window["vertices"], window["faces"], window["joints"])
    selected_top = print_frame_stats(window["radar_frames"], scatter, args.intensity_threshold, args.top_n)
    print_pair_stats(window["radar_frames"], scatter, selected_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
