#!/usr/bin/env python3
"""Generate Sim2 Doppler spectra using only one SMPL body part at a time.

The body-part echoes use visibility from a full-body ray-tracing pass. This
keeps occlusion consistent with the full human mesh: parts are selected after
full-body tracing rather than by deleting other parts before tracing.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import simulate_smpl_mesh_witwin_echo as sim2


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIT = Path("/tmp/rtpose_loop/smpl_v10_dense/seq22_dense_smpl_v10_linear.npz")
DEFAULT_FULL_ROOT = REPO_ROOT / "results" / "Sim2" / "body_part_ablation_eta010_chirp64_20s_22p5s_sequences" / "full"
DEFAULT_FULL_DOPPLER = (
    REPO_ROOT
    / "results"
    / "Sim2"
    / "body_part_ablation_eta010_chirp64_20s_22p5s_doppler_roi_topk_mean"
    / "full"
    / "22.npy"
)
DEFAULT_RAW_ROOT = REPO_ROOT / "results" / "Sim2" / "body_part_only_eta010_chirp64_20s_22p5s_sequences"
DEFAULT_DOPPLER_ROOT = REPO_ROOT / "results" / "Sim2" / "body_part_only_eta010_chirp64_20s_22p5s_doppler_roi_topk_mean"

BODY_PART_SEGMENTS: dict[str, tuple[tuple[int, int], ...]] = {
    "torso": ((0, 7), (9, 12), (1, 4), (7, 9), (7, 12), (0, 1), (0, 4)),
    "head": ((7, 8),),
    "right_arm_hand": ((9, 10), (10, 11)),
    "left_arm_hand": ((12, 13), (13, 14)),
    "right_leg_foot": ((1, 2), (2, 3)),
    "left_leg_foot": ((4, 5), (5, 6)),
}
BODY_PARTS = (
    "torso",
    "head",
    "right_arm_hand",
    "left_arm_hand",
    "right_leg_foot",
    "left_leg_foot",
)
PANEL_LABELS = {
    "full": "Full mesh",
    "torso": "Only torso",
    "head": "Only head",
    "right_arm_hand": "Only right arm+hand",
    "left_arm_hand": "Only left arm+hand",
    "right_leg_foot": "Only right leg+foot",
    "left_leg_foot": "Only left leg+foot",
}
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
    parser.add_argument("--fit-npz", default=str(DEFAULT_FIT))
    parser.add_argument("--sequence", default="22")
    parser.add_argument("--file-idx", default="0000")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("pytorch", "dirichlet", "slang"), default="dirichlet")
    parser.add_argument("--rx-order", choices=("attachment", "rtpose_raw"), default="attachment")
    parser.add_argument("--chirp-per-frame", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--epsilon-r", type=float, default=5.0)
    parser.add_argument("--start-radar-frame", type=int, default=201)
    parser.add_argument("--max-radar-frame", type=int, default=226)
    parser.add_argument("--interp", choices=("linear", "cubic"), default="linear")
    parser.add_argument("--intensity-threshold", type=float, default=1e-14)
    parser.add_argument("--raw-output-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--doppler-output-root", default=str(DEFAULT_DOPPLER_ROOT))
    parser.add_argument("--full-root", default=str(DEFAULT_FULL_ROOT))
    parser.add_argument("--full-doppler", default=str(DEFAULT_FULL_DOPPLER))
    parser.add_argument("--iq-scale", default="", help="Fixed IQ scale. Empty reads it from --full-root summary.")
    parser.add_argument("--roi-reducer", choices=("mean", "max", "topk-mean"), default="topk-mean")
    parser.add_argument("--roi-topk-fraction", type=float, default=0.05)
    parser.add_argument("--doppler-backend", choices=("auto", "numpy", "cupy", "torch"), default="torch")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--plot-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--gif-out", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-full-scatterers", action="store_true", default=True)
    parser.add_argument("--no-reuse-full-scatterers", dest="reuse_full_scatterers", action="store_false")
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-doppler", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def full_summary_path(args: argparse.Namespace) -> Path:
    return Path(args.full_root) / str(args.sequence) / f"seq{args.sequence}_mesh_echo_summary.json"


def full_scatter_path(args: argparse.Namespace) -> Path:
    return Path(args.full_root) / str(args.sequence) / f"seq{args.sequence}_mesh_echo_scatterers_used.npz"


def resolve_iq_scale(args: argparse.Namespace) -> str:
    if args.iq_scale:
        return str(args.iq_scale)
    summary_path = full_summary_path(args)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing full-body summary for fixed IQ scale: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    value = summary.get("iq_scale_effective")
    if not value:
        raise KeyError(f"{summary_path} does not contain iq_scale_effective")
    return str(value)


def squared_distance_to_segments(points: np.ndarray, joints: np.ndarray, segments: tuple[tuple[int, int], ...]) -> np.ndarray:
    dists = []
    for start_idx, end_idx in segments:
        start = joints[start_idx].astype(np.float64, copy=False)
        end = joints[end_idx].astype(np.float64, copy=False)
        axis = end - start
        denom = max(float(np.dot(axis, axis)), 1e-12)
        t = np.clip(((points - start) * axis).sum(axis=1) / denom, 0.0, 1.0)
        closest = start[None, :] + t[:, None] * axis[None, :]
        diff = points - closest
        dists.append(np.sum(diff * diff, axis=1))
    return np.min(np.stack(dists, axis=1), axis=1)


def body_part_face_labels(
    faces: np.ndarray,
    reference_vertices: np.ndarray,
    reference_joints: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    centroids = reference_vertices[faces].mean(axis=1).astype(np.float64, copy=False)
    labels = list(BODY_PART_SEGMENTS)
    distances = [
        squared_distance_to_segments(centroids, reference_joints, BODY_PART_SEGMENTS[label])
        for label in labels
    ]
    nearest = np.argmin(np.stack(distances, axis=1), axis=1)
    face_labels = np.asarray(labels, dtype=object)[nearest]
    counts = {label: int(np.sum(face_labels == label)) for label in labels}
    return face_labels, counts


def load_full_scatter(args: argparse.Namespace, fit_subset: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    path = full_scatter_path(args)
    if not args.reuse_full_scatterers or not path.exists():
        return None
    data = np.load(path)
    required = {
        "keyframe_centroids",
        "keyframe_normals",
        "keyframe_areas",
        "keyframe_visible",
        "keyframe_trace_intensity",
        "keyframe_trace_counts",
        "traced_faces",
    }
    if not required.issubset(data.files):
        return None
    traced_faces = data["traced_faces"].astype(np.int32)
    if traced_faces.shape != fit_subset["faces"].shape or not np.array_equal(traced_faces, fit_subset["faces"]):
        return None
    return {
        "centroids": data["keyframe_centroids"].astype(np.float32),
        "normals": data["keyframe_normals"].astype(np.float32),
        "areas": data["keyframe_areas"].astype(np.float32),
        "visible": data["keyframe_visible"].astype(bool),
        "intensity": data["keyframe_trace_intensity"].astype(np.float32),
        "trace_counts": data["keyframe_trace_counts"].astype(np.int32),
        "traced_faces": traced_faces,
        "tracing_skipped": np.asarray(False),
    }


def select_scatter_part(
    full_scatter: dict[str, np.ndarray],
    faces: np.ndarray,
    face_labels: np.ndarray,
    part: str,
) -> dict[str, np.ndarray]:
    mask = face_labels == part
    if not bool(np.any(mask)):
        raise ValueError(f"No faces assigned to {part}")
    out = dict(full_scatter)
    for key in ("centroids", "normals"):
        out[key] = full_scatter[key][:, mask, :]
    for key in ("areas", "visible", "intensity"):
        out[key] = full_scatter[key][:, mask]
    out["trace_counts"] = out["visible"].sum(axis=1).astype(np.int32)
    out["traced_faces"] = faces[mask].astype(np.int32, copy=False)
    return out


def echo_args(args: argparse.Namespace, part: str, iq_scale: str) -> argparse.Namespace:
    cloned = copy.copy(args)
    cloned.output_root = str(Path(args.raw_output_root) / part)
    cloned.iq_scale = str(iq_scale)
    cloned.save_scatterers = False
    return cloned


def write_part_summary(
    args: argparse.Namespace,
    part: str,
    fit_subset: dict[str, np.ndarray],
    scatter: dict[str, np.ndarray],
    result: dict[str, object],
    face_counts: dict[str, int],
    iq_scale: str,
) -> None:
    sequence_dir = Path(args.raw_output_root) / part / str(args.sequence)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fit_npz": str(Path(args.fit_npz)),
        "sequence": str(args.sequence),
        "output_root": str(Path(args.raw_output_root) / part),
        "smpl_keyframe_radar_frames": [int(x) for x in fit_subset["radar_frames"]],
        "body_part_only": part,
        "body_part_face_labels": face_counts,
        "selected_face_count": int(scatter["centroids"].shape[1]),
        "visibility_policy": "full-body ray tracing visibility, then select body-part scatterers",
        "visible_triangles_per_keyframe": [int(x) for x in scatter["trace_counts"]],
        "positive_trace_intensity_triangles_per_keyframe": [int(x) for x in (scatter["intensity"] > 0).sum(axis=1)],
        "chirp_intensity_mode": "frame-visible-angle",
        "specular_eta": float(sim2.STANDARD_SPECULAR_ETA),
        "intensity_threshold": float(args.intensity_threshold),
        "interp": args.interp,
        "backend": args.backend,
        "chirp_per_frame": int(args.chirp_per_frame),
        "resolution": int(args.resolution),
        "epsilon_r": float(args.epsilon_r),
        "start_radar_frame": int(args.start_radar_frame),
        "max_radar_frame": int(args.max_radar_frame),
        "iq_scale_reused_from_full": str(iq_scale),
        "coordinate_system": "witwin world [right, up, back] = Train.json [y, z, -x]",
        **result,
    }
    path = sequence_dir / f"seq{args.sequence}_body_part_only_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_doppler(args: argparse.Namespace, part: str) -> None:
    script = REPO_ROOT / "code" / "Echo_data_processing" / "raw_echo_to_doppler_spectrum.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-dir",
        str(Path(args.raw_output_root) / part),
        "--sequence",
        str(args.sequence),
        "--out-dir",
        str(Path(args.doppler_output_root) / part),
        "--file-idx",
        str(args.file_idx),
        "--frame-start",
        str(args.start_radar_frame),
        "--frame-stop",
        str(int(args.max_radar_frame) + 1),
        "--raw-frame-start",
        str(args.start_radar_frame),
        "--nchirp-loops",
        str(args.chirp_per_frame),
        "--roi-reducer",
        str(args.roi_reducer),
        "--roi-topk-fraction",
        str(args.roi_topk_fraction),
        "--backend",
        str(args.doppler_backend),
        "--gpu-device",
        str(args.gpu_device),
        "--rangemat-correction",
        "off",
        "--peakvalmat-correction",
        "off",
        "--overwrite",
    ]
    env = os.environ.copy()
    env.setdefault("HOME", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rtpose-doppler")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def normalized_panel(spectrum: np.ndarray, low: float, high: float) -> np.ndarray:
    shown = np.zeros_like(spectrum, dtype=np.float32)
    mask = np.isfinite(spectrum)
    logp = np.log10(np.maximum(spectrum[mask].astype(np.float64), 0.0) + 1.0)
    shown[mask] = ((logp - low) / (high - low)).astype(np.float32)
    return np.clip(shown, 0.0, 1.0)


def plot_comparison(args: argparse.Namespace, face_counts: dict[str, int]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_path = Path(args.full_doppler)
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full-body Doppler baseline: {full_path}")
    panels = [("full", full_path)] + [
        (part, Path(args.doppler_output_root) / part / f"{args.sequence}.npy")
        for part in BODY_PARTS
    ]
    spectra = []
    for part, path in panels:
        arr = np.load(path).astype(np.float32, copy=False)
        spectra.append((part, path, arr))

    baseline = spectra[0][2]
    finite = np.isfinite(baseline)
    base_log = np.log10(np.maximum(baseline[finite].astype(np.float64), 0.0) + 1.0)
    low, high = np.percentile(base_log, (10.0, 100.0))
    if not np.isfinite(high) or high <= low:
        raise RuntimeError(f"Invalid baseline normalization: low={low}, high={high}")

    out = Path(args.plot_out) if args.plot_out else Path(args.doppler_output_root) / (
        f"seq{args.sequence}_body_part_only_eta010_chirp{args.chirp_per_frame}_"
        f"{args.start_radar_frame}_{args.max_radar_frame}_baseline_norm_7panel.png"
    )
    summary_out = Path(args.summary_out) if args.summary_out else out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8.8), dpi=180, sharex=True, sharey=True)
    axes_flat = axes.ravel()
    extent = [
        (float(args.start_radar_frame) - 1.0) / 10.0,
        (float(args.max_radar_frame) - 1.0) / 10.0,
        -0.5,
        float(baseline.shape[0]) - 0.5,
    ]
    image = None
    panel_summary = []
    for ax, (part, path, arr) in zip(axes_flat, spectra):
        shown = normalized_panel(arr, low, high)
        image = ax.imshow(shown, origin="lower", aspect="auto", cmap="jet", vmin=0.0, vmax=1.0, extent=extent)
        title = PANEL_LABELS[part]
        if part != "full":
            title += f" ({face_counts[part]} faces)"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Doppler bin")
        panel_summary.append(
            {
                "part": part,
                "label": PANEL_LABELS[part],
                "path": str(path),
                "shape": [int(x) for x in arr.shape],
                "face_count": None if part == "full" else int(face_counts[part]),
                "power_max": float(np.nanmax(arr)),
            }
        )
    axes_flat[-1].axis("off")
    fig.suptitle(
        f"Seq{args.sequence} Sim2 Body-Part-Only Echo, full-body visibility, "
        f"eta 0.1, chirp_per_frame {args.chirp_per_frame}",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.08, top=0.90, wspace=0.20, hspace=0.32)
    cbar = fig.colorbar(image, ax=axes_flat[:-1].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Full-baseline-normalized log power")
    fig.savefig(out)
    plt.close(fig)

    summary = {
        "normalization": "log10(power + 1), p10-p100 from full-body baseline only, applied to all panels",
        "baseline_log_low_p10": float(low),
        "baseline_log_high_p100": float(high),
        "visibility_policy": "full-body ray tracing visibility, then select body-part scatterers",
        "body_part_face_labels": face_counts,
        "panels": panel_summary,
        "output_png": str(out),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def load_spectra_for_panels(args: argparse.Namespace) -> list[tuple[str, Path, np.ndarray]]:
    panels = [("full", Path(args.full_doppler))] + [
        (part, Path(args.doppler_output_root) / part / f"{args.sequence}.npy")
        for part in BODY_PARTS
    ]
    spectra = []
    for part, path in panels:
        if not path.exists():
            raise FileNotFoundError(f"Missing Doppler spectrum for {part}: {path}")
        spectra.append((part, path, np.load(path).astype(np.float32, copy=False)))
    return spectra


def baseline_log_norm(spectrum: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(spectrum)
    base_log = np.log10(np.maximum(spectrum[finite].astype(np.float64), 0.0) + 1.0)
    low, high = np.percentile(base_log, (10.0, 100.0))
    if not np.isfinite(high) or high <= low:
        raise RuntimeError(f"Invalid baseline normalization: low={low}, high={high}")
    return float(low), float(high)


def make_mesh_doppler_gif(args: argparse.Namespace, face_counts: dict[str, int]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    spectra = load_spectra_for_panels(args)
    low, high = baseline_log_norm(spectra[0][2])
    fit_subset = sim2.load_fit_subset(Path(args.fit_npz), args.start_radar_frame, args.max_radar_frame)
    target_mask = (fit_subset["radar_frames"] >= args.start_radar_frame) & (
        fit_subset["radar_frames"] <= args.max_radar_frame
    )
    vertices = fit_subset["vertices"][target_mask]
    rtpose_joints = fit_subset["rtpose_joints"][target_mask]
    smpl_joints = fit_subset["smpl_joints"][target_mask]
    radar_frames = fit_subset["radar_frames"][target_mask]
    if vertices.shape[0] == 0:
        raise RuntimeError("No SMPL frames available for GIF target window")

    faces = fit_subset["faces"]
    face_labels, _ = body_part_face_labels(faces, fit_subset["vertices"][0], fit_subset["joints"][0])
    stride = max(1, int(np.ceil(len(faces) / 2200)))
    mesh_faces = faces[::stride]
    mesh_labels = face_labels[::stride]
    label_colors = {
        "torso": "#7f7f7f",
        "head": "#e377c2",
        "right_arm_hand": "#1f77b4",
        "left_arm_hand": "#17becf",
        "right_leg_foot": "#ff7f0e",
        "left_leg_foot": "#2ca02c",
    }
    mesh_facecolors = [label_colors[str(label)] for label in mesh_labels]

    grid = [
        ("mesh", None),
        ("torso", Path(args.doppler_output_root) / "torso" / f"{args.sequence}.npy"),
        ("head", Path(args.doppler_output_root) / "head" / f"{args.sequence}.npy"),
        ("right_arm_hand", Path(args.doppler_output_root) / "right_arm_hand" / f"{args.sequence}.npy"),
        ("full", Path(args.full_doppler)),
        ("left_arm_hand", Path(args.doppler_output_root) / "left_arm_hand" / f"{args.sequence}.npy"),
        ("right_leg_foot", Path(args.doppler_output_root) / "right_leg_foot" / f"{args.sequence}.npy"),
        ("left_leg_foot", Path(args.doppler_output_root) / "left_leg_foot" / f"{args.sequence}.npy"),
    ]
    spectrum_by_part = {part: arr for part, _path, arr in spectra}
    time_values = (radar_frames.astype(np.float64) - 1.0) / 10.0
    extent = [
        float(time_values[0]),
        float(time_values[-1]),
        -0.5,
        float(spectra[0][2].shape[0]) - 0.5,
    ]

    fig = plt.figure(figsize=(18, 9.2), dpi=120)
    axes = []
    for idx, (part, _path) in enumerate(grid):
        if part == "mesh":
            ax = fig.add_subplot(2, 4, idx + 1, projection="3d")
        else:
            ax = fig.add_subplot(2, 4, idx + 1)
        axes.append(ax)

    coords_all = vertices[:, :, [0, 2, 1]]
    smpl_joint_coords = smpl_joints[:, :, [0, 2, 1]]
    rtpose_joint_coords = rtpose_joints[:, :, [0, 2, 1]]
    bounds_points = np.concatenate(
        [
            coords_all.reshape(-1, 3),
            smpl_joint_coords.reshape(-1, 3),
            rtpose_joint_coords.reshape(-1, 3),
        ],
        axis=0,
    )
    bounds_points = bounds_points[np.isfinite(bounds_points).all(axis=1)]
    center = bounds_points.mean(axis=0)
    span = np.ptp(bounds_points, axis=0)
    radius = max(0.8, float(span.max()) * 0.62)
    mesh_ax = axes[0]
    first_tris = coords_all[0][mesh_faces]
    mesh = Poly3DCollection(
        first_tris,
        facecolors=mesh_facecolors,
        edgecolors="#222222",
        linewidths=0.02,
        alpha=0.92,
    )
    mesh_ax.add_collection3d(mesh)
    mesh_ax.set_xlim(center[0] - radius, center[0] + radius)
    mesh_ax.set_ylim(center[1] - radius, center[1] + radius)
    mesh_ax.set_zlim(center[2] - radius, center[2] + radius)
    mesh_ax.view_init(elev=16, azim=-70)
    mesh_ax.set_axis_off()
    smpl_scatter = mesh_ax.scatter([], [], [], c="#22c55e", s=18, depthshade=False)
    rtpose_scatter = mesh_ax.scatter([], [], [], c="#ef4444", s=24, depthshade=False)
    smpl_lines = [mesh_ax.plot([], [], [], c="#22c55e", linewidth=1.5)[0] for _ in RT_BONES]
    rtpose_lines = [mesh_ax.plot([], [], [], c="#ef4444", linewidth=1.8)[0] for _ in RT_BONES]
    mesh_ax.text2D(0.02, 0.96, "green: SMPL joints", transform=mesh_ax.transAxes, color="#22c55e", fontsize=8)
    mesh_ax.text2D(0.02, 0.90, "red: RT-Pose keypoints", transform=mesh_ax.transAxes, color="#ef4444", fontsize=8)

    def update_skeleton(scatter, lines, coords: np.ndarray) -> None:
        valid = np.isfinite(coords).all(axis=1)
        shown = coords[valid]
        scatter._offsets3d = (shown[:, 0], shown[:, 1], shown[:, 2])
        for line, (start, end) in zip(lines, RT_BONES):
            if valid[start] and valid[end]:
                line.set_data([coords[start, 0], coords[end, 0]], [coords[start, 1], coords[end, 1]])
                line.set_3d_properties([coords[start, 2], coords[end, 2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

    cursor_lines = []
    image = None
    for ax, (part, _path) in zip(axes[1:], grid[1:]):
        arr = spectrum_by_part[part]
        shown = normalized_panel(arr, low, high)
        image = ax.imshow(shown, origin="lower", aspect="auto", cmap="jet", vmin=0.0, vmax=1.0, extent=extent)
        title = PANEL_LABELS[part]
        if part != "full":
            title += f" ({face_counts[part]} faces)"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Doppler bin", fontsize=8)
        ax.tick_params(labelsize=7)
        cursor_lines.append(ax.axvline(float(time_values[0]), color="white", linewidth=1.2))

    fig.suptitle(
        f"Seq{args.sequence} SMPL Motion and Body-Part-Only Doppler, full-body visibility",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.04, right=0.91, bottom=0.07, top=0.90, wspace=0.28, hspace=0.36)
    cbar = fig.colorbar(image, ax=axes[1:], fraction=0.022, pad=0.02)
    cbar.set_label("Full-baseline-normalized log power", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    def update(frame_idx: int):
        mesh.set_verts(coords_all[frame_idx][mesh_faces])
        update_skeleton(smpl_scatter, smpl_lines, smpl_joint_coords[frame_idx])
        update_skeleton(rtpose_scatter, rtpose_lines, rtpose_joint_coords[frame_idx])
        current_time = float(time_values[frame_idx])
        mesh_ax.set_title(f"SMPL mesh + keypoints  t={current_time:.1f}s", fontsize=10)
        for line in cursor_lines:
            line.set_xdata([current_time, current_time])
        return [mesh, smpl_scatter, rtpose_scatter, *smpl_lines, *rtpose_lines, *cursor_lines]

    out = Path(args.gif_out) if args.gif_out else Path(args.doppler_output_root) / (
        f"seq{args.sequence}_smpl_motion_body_part_only_eta010_chirp{args.chirp_per_frame}_"
        f"{args.start_radar_frame}_{args.max_radar_frame}_2x4.gif"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(vertices), interval=125, blit=False)
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    iq_scale = resolve_iq_scale(args)
    fit_subset = sim2.load_fit_subset(Path(args.fit_npz), args.start_radar_frame, args.max_radar_frame)
    print(f"Loaded {len(fit_subset['radar_frames'])} SMPL frames for full-body visibility")

    full_scatter = load_full_scatter(args, fit_subset)
    if full_scatter is None:
        print("Tracing full mesh for visibility")
        full_scatter = sim2.trace_dense_keyframes(args, fit_subset["vertices"], fit_subset["faces"])
    else:
        print(f"Reusing full-body scatterers from {full_scatter_path(args)}")

    face_labels, face_counts = body_part_face_labels(
        fit_subset["faces"],
        fit_subset["vertices"][0],
        fit_subset["joints"][0],
    )
    print(f"Body-part face counts: {face_counts}")

    if not args.skip_raw:
        for part in BODY_PARTS:
            print(f"===== raw echo: only {part} =====")
            part_scatter = select_scatter_part(full_scatter, fit_subset["faces"], face_labels, part)
            result = sim2.simulate_echo(echo_args(args, part, iq_scale), fit_subset, part_scatter)
            write_part_summary(args, part, fit_subset, part_scatter, result, face_counts, iq_scale)

    if not args.skip_doppler:
        for part in BODY_PARTS:
            print(f"===== doppler: only {part} =====")
            run_doppler(args, part)

    if not args.skip_plot:
        out = plot_comparison(args, face_counts)
        print(f"wrote {out}")

    if not args.skip_gif:
        out = make_mesh_doppler_gif(args, face_counts)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
