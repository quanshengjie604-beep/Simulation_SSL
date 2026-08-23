#!/usr/bin/env python3
"""Run AMASS micro-Doppler generation from a manifest shard."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


def resolve_workspace_path(workspace_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else workspace_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--workspace-root", default="..")
    parser.add_argument("--witwin-python", default="/bigdata/users/quansj/miniforge3/envs/witwin/bin/python")
    parser.add_argument("--generator", default="code/smpl_mesh_to_micro_doppler.py")
    parser.add_argument("--model-dir", default="smpl_models")
    parser.add_argument("--out-dir", default="results/amass_smplx_micro_doppler/equal_target_raw")
    parser.add_argument("--log-dir", default="results/amass_smplx_micro_doppler/equal_target_logs")
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--backend", default="pytorch", choices=("pytorch", "dirichlet", "slang"))
    parser.add_argument("--visibility-mode", default="hold", choices=("hold", "linear"))
    parser.add_argument("--smpl-batch-size", type=int, default=16)
    parser.add_argument("--mesh-cache-dir", default="")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--enable-slang-fast-f32", action="store_true")
    parser.add_argument("--torch-cuda-arch-list", default="")
    parser.add_argument("--slang-cuda-arch", default="")
    parser.add_argument("--witwin-radar-dir", default="")
    parser.add_argument(
        "--rtx-blackwell-ptx",
        action="store_true",
        help="Enable the RTX Blackwell Slang path via compute_90 PTX JIT.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = (repo_root / args.workspace_root).resolve()
    manifest_csv = (repo_root / args.manifest_csv).resolve()
    generator = resolve_workspace_path(workspace_root, args.generator).resolve()
    model_dir = resolve_workspace_path(workspace_root, args.model_dir).resolve()
    out_dir = resolve_workspace_path(workspace_root, args.out_dir).resolve()
    log_dir = resolve_workspace_path(workspace_root, args.log_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.rtx_blackwell_ptx:
        args.backend = "slang"
        args.enable_slang_fast_f32 = True
        args.torch_cuda_arch_list = args.torch_cuda_arch_list or "9.0+PTX"
        args.slang_cuda_arch = args.slang_cuda_arch or args.torch_cuda_arch_list
    if args.enable_slang_fast_f32 and args.backend != "slang":
        raise ValueError("--enable-slang-fast-f32 requires --backend slang")

    with manifest_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    cache_base = workspace_root / "results" / "amass_smplx_micro_doppler" / "cache"
    for cache_dir in (
        cache_base,
        cache_base / "witwin",
        cache_base / "matplotlib",
        cache_base / "xdg",
        cache_base / "xdg" / "torch" / "kernels",
        cache_base / "drjit",
    ):
        cache_dir.mkdir(parents=True, exist_ok=True)
    python_bin_dir = Path(args.witwin_python).expanduser().resolve().parent
    env["PATH"] = str(python_bin_dir) + os.pathsep + env.get("PATH", "")
    env.setdefault("WITWIN_CACHE_DIR", str(cache_base / "witwin"))
    env.setdefault("MPLCONFIGDIR", str(cache_base / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(cache_base / "xdg"))
    env.setdefault("DRJIT_CACHE_DIR", str(cache_base / "drjit"))
    default_radar_dir = workspace_root / "witwin" / "radar"
    if args.witwin_radar_dir:
        env["WITWIN_RADAR_DIR"] = str(resolve_workspace_path(workspace_root, args.witwin_radar_dir).resolve())
    elif default_radar_dir.exists():
        env["WITWIN_RADAR_DIR"] = str(default_radar_dir)
    if args.torch_cuda_arch_list:
        env["TORCH_CUDA_ARCH_LIST"] = args.torch_cuda_arch_list
    if args.slang_cuda_arch:
        env["WITWIN_SLANG_CUDA_ARCH"] = args.slang_cuda_arch
    elif args.torch_cuda_arch_list:
        env["WITWIN_SLANG_CUDA_ARCH"] = args.torch_cuda_arch_list
    if args.enable_slang_fast_f32:
        env["WITWIN_SLANG_FRAME_FLOAT32"] = "1"
        env["WITWIN_SLANG_BATCH_PREP"] = "1"
        env["WITWIN_SLANG_FRAME_PHASE_COEFF"] = "0"
        env["WITWIN_SLANG_FRAME_FUSED"] = "0"
        env["WITWIN_SLANG_FRAME_CHUNKED"] = "0"

    completed = 0
    skipped = 0
    failed = 0
    t_start = time.time()
    for position, row in enumerate(rows, start=1):
        sequence_id = row["sequence_id"]
        result_path = out_dir / f"{sequence_id}_micro_doppler.npz"
        item_log = log_dir / f"{sequence_id}.log"
        if result_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{position:04d}/{len(rows):04d}] skip existing {sequence_id}", flush=True)
            continue

        cmd = [
            args.witwin_python,
            str(generator),
            "--amass-npz",
            row["amass_npz"],
            "--model-dir",
            str(model_dir),
            "--out-dir",
            str(out_dir),
            "--sequence",
            sequence_id,
            "--device",
            "cuda:0",
            "--backend",
            args.backend,
            "--range-bin-mode",
            "fixed",
            "--visibility-mode",
            args.visibility_mode,
            "--doppler-transform",
            "fft",
            "--smpl-batch-size",
            str(args.smpl_batch_size),
            "--subject-range",
            row["subject_range"],
            "--subject-lateral",
            row["subject_lateral"],
            "--radar-height",
            row["radar_height"],
            "--overwrite",
        ]
        if args.mesh_cache_dir:
            mesh_cache_dir = resolve_workspace_path(workspace_root, args.mesh_cache_dir).resolve()
            cmd.extend(["--mesh-cache-dir", str(mesh_cache_dir)])
        if args.skip_plot:
            cmd.append("--skip-plot")
        print(f"[{position:04d}/{len(rows):04d}] start {sequence_id}", flush=True)
        item_start = time.time()
        with item_log.open("w", encoding="utf-8") as log_handle:
            log_handle.write(" ".join(cmd) + "\n")
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "TORCH_CUDA_ARCH_LIST",
                "WITWIN_SLANG_CUDA_ARCH",
                "WITWIN_SLANG_FRAME_FLOAT32",
                "WITWIN_SLANG_BATCH_PREP",
                "WITWIN_RADAR_DIR",
                "WITWIN_CACHE_DIR",
            ):
                if env.get(key):
                    log_handle.write(f"{key}={env[key]}\n")
            log_handle.flush()
            proc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - item_start
        if proc.returncode == 0:
            completed += 1
            print(f"[{position:04d}/{len(rows):04d}] done {sequence_id} {elapsed/60:.1f} min", flush=True)
        else:
            failed += 1
            print(f"[{position:04d}/{len(rows):04d}] FAILED {sequence_id}; see {item_log}", flush=True)
            return proc.returncode

    elapsed_total = time.time() - t_start
    print(
        f"finished shard {args.shard_index}/{args.num_shards}: "
        f"completed={completed} skipped={skipped} failed={failed} elapsed={elapsed_total/3600:.2f} h",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
