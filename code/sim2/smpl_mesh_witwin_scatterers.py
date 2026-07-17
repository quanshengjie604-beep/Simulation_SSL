#!/usr/bin/env python3
"""Prototype SMPL-mesh WiTwin triangle scatterers for one sequence.

This script is intentionally a scatterer-stage prototype. It does not write raw
ADC/bin files yet. The coordinate convention is the same as the existing
WiTwin simulator: Train.json raw [range_x, lateral_y, height_z] is converted to
WiTwin/radar world [right, up, back] as [y, z, -x]. The SMPL fit NPZ already
stores vertices in that WiTwin/radar world coordinate system, so vertices are
passed to WiTwin without another coordinate transform and with recenter=False.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "sim1_raw_echo_generation"))

import simulate_rtpose_witwin as point_sim


WITWIN_RADAR_DIR = point_sim.WITWIN_RADAR_DIR
DEFAULT_FIT = (
    REPO_ROOT
    / "results"
    / "SMPL_fit"
    / "smpl_sequence_fit_v9_six_motion_gifs"
    / "seq185_joint_labels_temporal_v9"
    / "seq185_joint_labels_temporal_v9_fit.npz"
)


def bootstrap_witwin_mesh_modules():
    """Import WiTwin submodules without triggering witwin.radar.__init__ first."""
    cache = REPO_ROOT / "logs" / ".cache" / "witwin"
    os.environ.setdefault("HOME", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("DRJIT_CACHE_DIR", str(cache))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    cache.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    pkg = types.ModuleType("witwin.radar")
    pkg.__path__ = [str(WITWIN_RADAR_DIR)]
    sys.modules.setdefault("witwin.radar", pkg)

    from witwin.radar.radar import Radar, RadarConfig
    from witwin.radar.scene import Scene
    from witwin.radar.trace import Tracer

    return Radar, RadarConfig, Scene, Tracer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-npz", default=str(DEFAULT_FIT))
    parser.add_argument("--sequence", default="185")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "sim2" / "smpl_mesh_witwin_scatterers" / "seq185"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="pytorch", choices=("pytorch", "dirichlet", "slang"))
    parser.add_argument("--rx-order", choices=("attachment", "rtpose_raw"), default="attachment")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--epsilon-r", type=float, default=5.0)
    parser.add_argument("--max-keyframes", type=int, default=5, help="Uniformly sample this many SMPL frames. 0 uses all.")
    parser.add_argument("--num-chirps", type=int, default=64, help="Chirp samples between first and last selected frame.")
    parser.add_argument("--intensity-eps", type=float, default=1e-8)
    return parser.parse_args()


def selected_indices(n_frames: int, max_keyframes: int) -> np.ndarray:
    if max_keyframes <= 0 or max_keyframes >= n_frames:
        return np.arange(n_frames, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, n_frames - 1, max_keyframes)).astype(np.int64))


def triangle_centroids_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tris = vertices[faces]
    centroids = tris.mean(axis=1)
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice_area[:, None], 1e-12)
    areas = 0.5 * twice_area
    return centroids.astype(np.float32), normals.astype(np.float32), areas.astype(np.float32)


def make_radar(args: argparse.Namespace, Radar, RadarConfig):
    config = RadarConfig.from_dict(point_sim.build_radar_config(args.rx_order))
    return Radar(
        config,
        backend=args.backend,
        device=args.device,
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, -1.0),
        up=(0.0, 1.0, 0.0),
    )


def trace_keyframes(args: argparse.Namespace, vertices: np.ndarray, faces: np.ndarray):
    Radar, RadarConfig, Scene, Tracer = bootstrap_witwin_mesh_modules()
    device = torch.device(args.device)
    radar = make_radar(args, Radar, RadarConfig)
    scene = Scene(device=args.device)
    scene.add_mesh(
        name="human",
        vertices=torch.as_tensor(vertices[0], dtype=torch.float32, device=device),
        faces=faces,
        dynamic=True,
    )
    tracer = Tracer(
        scene,
        radar,
        resolution=args.resolution,
        epsilon_r=args.epsilon_r,
        sampling="triangle",
        multipath=False,
        max_reflections=0,
    )

    n_keyframes = vertices.shape[0]
    n_faces = faces.shape[0]
    centroids = np.empty((n_keyframes, n_faces, 3), dtype=np.float32)
    normals = np.empty((n_keyframes, n_faces, 3), dtype=np.float32)
    areas = np.empty((n_keyframes, n_faces), dtype=np.float32)
    intensity = np.zeros((n_keyframes, n_faces), dtype=np.float32)
    visible = np.zeros((n_keyframes, n_faces), dtype=bool)

    trace_counts = []
    for frame_idx in range(n_keyframes):
        v = vertices[frame_idx].astype(np.float32, copy=False)
        c, n, a = triangle_centroids_normals(v, faces)
        centroids[frame_idx] = c
        normals[frame_idx] = n
        areas[frame_idx] = a

        if frame_idx > 0:
            scene.update_structure(
                "human",
                vertices=torch.as_tensor(v, dtype=torch.float32, device=device),
            )
        trace = tracer.trace()
        tri_idx = trace._tri_indices.detach().cpu().numpy().astype(np.int64)
        ints = trace.intensities.detach().cpu().numpy().astype(np.float32)
        intensity[frame_idx, tri_idx] = ints
        visible[frame_idx, tri_idx] = True
        trace_counts.append(int(tri_idx.size))
        print(f"[trace {frame_idx + 1:03d}/{n_keyframes:03d}] visible triangles={tri_idx.size}", flush=True)

    return {
        "centroids": centroids,
        "normals": normals,
        "areas": areas,
        "intensity": intensity,
        "visible": visible,
        "trace_counts": np.asarray(trace_counts, dtype=np.int32),
    }


def cubic_interpolate_scatterers(
    keyframe_times: np.ndarray,
    chirp_times: np.ndarray,
    centroids: np.ndarray,
    intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bc_type = "not-a-knot" if len(keyframe_times) >= 4 else "natural"
    position_spline = CubicSpline(keyframe_times, centroids, axis=0, bc_type=bc_type, extrapolate=False)
    intensity_spline = CubicSpline(keyframe_times, intensity, axis=0, bc_type=bc_type, extrapolate=False)
    chirp_centroids = position_spline(chirp_times).astype(np.float32)
    chirp_intensity = np.clip(intensity_spline(chirp_times), 0.0, None).astype(np.float32)
    return chirp_centroids, chirp_intensity


def choose_preview_triangle(intensity: np.ndarray) -> int:
    score = intensity.max(axis=0)
    return int(np.argmax(score))


def save_visualizations(
    out_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    frames: np.ndarray,
    radar_frames: np.ndarray,
    scatter: dict[str, np.ndarray],
    chirp_times: np.ndarray,
    chirp_centroids: np.ndarray,
    chirp_intensity: np.ndarray,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_png = out_dir / "seq185_mesh_triangle_scatterers_preview.png"
    interp_png = out_dir / "seq185_cubic_scatterer_interpolation_preview.png"

    pick = np.unique(np.rint(np.linspace(0, len(frames) - 1, min(3, len(frames)))).astype(np.int64))
    fig = plt.figure(figsize=(6 * len(pick), 6))
    mesh_faces = faces[:: max(1, len(faces) // 1500)]
    for panel_idx, frame_idx in enumerate(pick, start=1):
        ax = fig.add_subplot(1, len(pick), panel_idx, projection="3d")
        vp = vertices[frame_idx][:, [0, 2, 1]]
        tri = mtri.Triangulation(vp[:, 0], vp[:, 1], triangles=mesh_faces)
        ax.plot_trisurf(tri, vp[:, 2], color="#b8c0cc", alpha=0.18, linewidth=0.02, edgecolor="#94a3b8")

        visible_mask = scatter["visible"][frame_idx]
        mask = scatter["intensity"][frame_idx] > 0
        pts = scatter["centroids"][frame_idx, mask]
        vals = scatter["intensity"][frame_idx, mask]
        if pts.size:
            order = np.argsort(vals)
            pts = pts[order]
            vals = vals[order]
            ax.scatter(
                pts[:, 0],
                pts[:, 2],
                pts[:, 1],
                c=np.log10(vals + 1e-10),
                cmap="viridis",
                s=5,
                alpha=0.85,
                depthshade=False,
            )
        center = vertices[frame_idx].mean(axis=0)
        span = vertices[frame_idx].max(axis=0) - vertices[frame_idx].min(axis=0)
        radius = max(1.1, float(np.max(span)) * 0.65)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(center[1] - radius, center[1] + radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=18, azim=-62)
        ax.set_axis_off()
        ax.set_title(
            f"frame {frames[frame_idx]} radar {radar_frames[frame_idx]}\n"
            f"visible={int(visible_mask.sum())}/{len(faces)}, rcs>0={int(mask.sum())}"
        )
    fig.tight_layout()
    fig.savefig(preview_png, dpi=160)
    plt.close(fig)

    tri_id = choose_preview_triangle(scatter["intensity"])
    key_t = np.arange(scatter["centroids"].shape[0], dtype=np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes[0].plot(key_t, scatter["centroids"][:, tri_id, 0], "o", label="x keyframe")
    axes[0].plot(key_t, scatter["centroids"][:, tri_id, 1], "o", label="y keyframe")
    axes[0].plot(key_t, scatter["centroids"][:, tri_id, 2], "o", label="z keyframe")
    axes[0].plot(chirp_times, chirp_centroids[:, tri_id, 0], "-", label="x cubic")
    axes[0].plot(chirp_times, chirp_centroids[:, tri_id, 1], "-", label="y cubic")
    axes[0].plot(chirp_times, chirp_centroids[:, tri_id, 2], "-", label="z cubic")
    axes[0].set_ylabel("centroid coordinate (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(key_t, scatter["intensity"][:, tri_id], "o", label="keyframe intensity")
    axes[1].plot(chirp_times, chirp_intensity[:, tri_id], "-", label="cubic + clamp")
    axes[1].set_xlabel("selected keyframe index time")
    axes[1].set_ylabel("intensity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"triangle id {tri_id}: cubic chirp interpolation")
    fig.tight_layout()
    fig.savefig(interp_png, dpi=160)
    plt.close(fig)
    return preview_png, interp_png


def main() -> int:
    args = parse_args()
    fit_path = Path(args.fit_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit = np.load(fit_path)
    all_vertices = fit["vertices"].astype(np.float32)
    faces = fit["faces"].astype(np.int32)
    all_frames = fit["frames"]
    all_radar_frames = fit["radar_frames"]
    indices = selected_indices(len(all_frames), args.max_keyframes)
    vertices = all_vertices[indices]
    frames = all_frames[indices]
    radar_frames = all_radar_frames[indices]

    scatter = trace_keyframes(args, vertices, faces)
    keyframe_times = np.arange(len(indices), dtype=np.float32)
    chirp_times = np.linspace(keyframe_times[0], keyframe_times[-1], args.num_chirps, dtype=np.float32)
    chirp_centroids, chirp_intensity = cubic_interpolate_scatterers(
        keyframe_times,
        chirp_times,
        scatter["centroids"],
        scatter["intensity"],
    )

    out_npz = out_dir / "seq185_smpl_mesh_triangle_scatterers.npz"
    np.savez_compressed(
        out_npz,
        fit_npz=str(fit_path),
        sequence=str(args.sequence),
        selected_indices=indices,
        frames=frames,
        radar_frames=radar_frames,
        faces=faces,
        keyframe_centroids=scatter["centroids"],
        keyframe_normals=scatter["normals"],
        keyframe_areas=scatter["areas"],
        keyframe_intensity=scatter["intensity"],
        keyframe_visible=scatter["visible"],
        keyframe_trace_counts=scatter["trace_counts"],
        chirp_times=chirp_times,
        chirp_centroids=chirp_centroids,
        chirp_intensity=chirp_intensity,
        coordinate_system="witwin_world_from_train_json_[y,z,-x]; vertices already transformed by SMPL fitter",
        radar_position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        radar_target=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
    )

    preview_png, interp_png = save_visualizations(
        out_dir,
        vertices,
        faces,
        frames,
        radar_frames,
        scatter,
        chirp_times,
        chirp_centroids,
        chirp_intensity,
    )

    summary = {
        "fit_npz": str(fit_path),
        "out_npz": str(out_npz),
        "preview_png": str(preview_png),
        "interpolation_png": str(interp_png),
        "n_keyframes": int(len(indices)),
        "n_faces": int(faces.shape[0]),
        "visible_triangles_per_keyframe": [int(x) for x in scatter["trace_counts"]],
        "positive_intensity_triangles_per_keyframe": [
            int(x) for x in (scatter["intensity"] > 0).sum(axis=1)
        ],
        "chirp_samples": int(args.num_chirps),
        "coordinate_system": "witwin world [right, up, back] = Train.json [y, z, -x]; no extra transform applied to fit vertices",
        "intensity_interpolation": "CubicSpline, clamped at >= 0",
        "scatterer_identity": "SMPL face/triangle id",
    }
    out_summary = out_dir / "seq185_smpl_mesh_triangle_scatterers_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
