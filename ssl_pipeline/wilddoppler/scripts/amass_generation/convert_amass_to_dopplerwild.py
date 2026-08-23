#!/usr/bin/env python3
"""Convert generated AMASS micro-Doppler spectra into DopplerWild SSL format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _window_count(length: int, crop_bins: int = 90, stride: int = 45) -> int:
    if length < crop_bins:
        return 0
    return (length - crop_bins) // stride + 1


def _convert_spectrum(spectrum: np.ndarray, output_length: int, db_low_pct: float, db_high_pct: float) -> np.ndarray:
    power_db = 10.0 * np.log10(np.maximum(spectrum.astype(np.float32, copy=False), np.finfo(np.float32).tiny))
    low_db, high_db = np.percentile(power_db, [db_low_pct, db_high_pct])
    if not high_db > low_db:
        high_db = low_db + 1.0
    normalized = np.clip((power_db - low_db) / (high_db - low_db), 0.0, 1.0) * 45.0
    tensor = torch.from_numpy(normalized[None, None].astype(np.float32, copy=False))
    resized = F.interpolate(tensor, size=(256, int(output_length)), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", default="scripts/amass_generation/amass_equal_target_manifest.csv")
    parser.add_argument("--raw-dir", default="../results/amass_smplx_micro_doppler/equal_target_raw")
    parser.add_argument("--amass-out-dir", default="data/amass_equal_target_tracks_Doppler")
    parser.add_argument("--mixed-out-dir", default="data/a5_mixed_unlabeled_tracks_Doppler")
    parser.add_argument("--dw-tracklist", default="data/fold_splits/DopplerWild_unlabeled_tracklist.csv")
    parser.add_argument("--dw-data-dir", default="data/unlabeled_tracks_Doppler")
    parser.add_argument("--amass-tracklist-out", default="data/fold_splits/AMASS_equal_target_tracklist.csv")
    parser.add_argument("--mixed-tracklist-out", default="data/fold_splits/DopplerWild_AMASS_equal_mixed_tracklist.csv")
    parser.add_argument("--summary-json", default="scripts/amass_generation/amass_equal_target_conversion_summary.json")
    parser.add_argument("--bins-per-second", type=int, default=90)
    parser.add_argument("--target-val-windows", type=int, default=1360)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--db-low-percentile", type=float, default=1.0)
    parser.add_argument("--db-high-percentile", type=float, default=99.0)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_csv = (repo_root / args.manifest_csv).resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    amass_out_dir = (repo_root / args.amass_out_dir).resolve()
    mixed_out_dir = (repo_root / args.mixed_out_dir).resolve()
    dw_tracklist = (repo_root / args.dw_tracklist).resolve()
    dw_data_dir = (repo_root / args.dw_data_dir).resolve()
    amass_tracklist_out = (repo_root / args.amass_tracklist_out).resolve()
    mixed_tracklist_out = (repo_root / args.mixed_tracklist_out).resolve()
    summary_json = (repo_root / args.summary_json).resolve()

    manifest = pd.read_csv(manifest_csv)
    rng = np.random.default_rng(int(args.seed))
    order = list(range(len(manifest)))
    rng.shuffle(order)
    val_indices: set[int] = set()
    val_windows = 0
    for idx in order:
        if val_windows >= int(args.target_val_windows):
            break
        val_indices.add(idx)
        val_windows += _window_count(int(round(float(manifest.loc[idx, "duration_s"]) * args.bins_per_second)))

    amass_rows = []
    amass_out_dir.mkdir(parents=True, exist_ok=True)
    mixed_out_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in manifest.iterrows():
        sequence_id = str(row["sequence_id"])
        raw_path = raw_dir / f"{sequence_id}_micro_doppler.npz"
        out_path = amass_out_dir / f"{sequence_id}_track_0.npz"
        mixed_path = mixed_out_dir / out_path.name
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        duration = float(row["duration_s"])
        output_length = max(1, int(round(duration * args.bins_per_second)))
        if args.overwrite or not out_path.exists():
            with np.load(raw_path) as data:
                u_d = _convert_spectrum(
                    data["spectrum"],
                    output_length=output_length,
                    db_low_pct=float(args.db_low_percentile),
                    db_high_pct=float(args.db_high_percentile),
                )
            np.savez_compressed(out_path, uD=u_d)
        _link_or_copy(out_path, mixed_path, args.link_mode)
        amass_rows.append(
            {
                "file_name": sequence_id,
                "track_id": 0,
                "split": "val" if idx in val_indices else "train",
                "uD_length": output_length,
                "duration": duration,
            }
        )

    dw_df = pd.read_csv(dw_tracklist)
    for _, row in dw_df.iterrows():
        src = dw_data_dir / f"{row['file_name']}_track_{int(row['track_id'])}.npz"
        dst = mixed_out_dir / src.name
        _link_or_copy(src, dst, args.link_mode)

    amass_df = pd.DataFrame(amass_rows)
    mixed_df = pd.concat([dw_df, amass_df], ignore_index=True)
    amass_tracklist_out.parent.mkdir(parents=True, exist_ok=True)
    amass_df.to_csv(amass_tracklist_out, index=False)
    mixed_df.to_csv(mixed_tracklist_out, index=False)

    train_windows = {
        "dw": int(sum(_window_count(int(v)) for v in dw_df.loc[dw_df["split"] == "train", "uD_length"])),
        "amass": int(sum(_window_count(int(v)) for v in amass_df.loc[amass_df["split"] == "train", "uD_length"])),
    }
    val_windows_summary = {
        "dw": int(sum(max(1, int(np.ceil(int(v) / 90))) for v in dw_df.loc[dw_df["split"] == "val", "uD_length"])),
        "amass": int(sum(max(1, int(np.ceil(int(v) / 90))) for v in amass_df.loc[amass_df["split"] == "val", "uD_length"])),
    }
    sample_stats = []
    for out_path in sorted(amass_out_dir.glob("*.npz"))[:200]:
        with np.load(out_path) as data:
            arr = data["uD"]
        sample_stats.append([float(arr.mean()), float(arr.std()), float(np.percentile(arr, 1)), float(np.percentile(arr, 99))])
    stats_arr = np.asarray(sample_stats, dtype=np.float64) if sample_stats else np.empty((0, 4))
    summary = {
        "amass_rows": int(len(amass_df)),
        "mixed_rows": int(len(mixed_df)),
        "amass_split_counts": {str(k): int(v) for k, v in amass_df["split"].value_counts().to_dict().items()},
        "mixed_split_counts": {str(k): int(v) for k, v in mixed_df["split"].value_counts().to_dict().items()},
        "train_windows": train_windows,
        "val_windows": val_windows_summary,
        "amass_sample_stats_mean_std_p1_p99": stats_arr.mean(axis=0).tolist() if len(stats_arr) else None,
        "amass_out_dir": str(amass_out_dir),
        "mixed_out_dir": str(mixed_out_dir),
        "amass_tracklist": str(amass_tracklist_out),
        "mixed_tracklist": str(mixed_tracklist_out),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
