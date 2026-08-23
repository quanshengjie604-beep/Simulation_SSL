#!/usr/bin/env python3
"""Build an AMASS generation manifest sized to DopplerWild SSL pretraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


VIEWS = {
    "view0": {"subject_range": 4.0, "subject_lateral": 0.0, "radar_height": 1.0},
    "view1": {"subject_range": 6.0, "subject_lateral": 1.0, "radar_height": 1.0},
}


def window_count(duration_s: float, bins_per_second: int, crop_seconds: float, overlap: float) -> int:
    length = int(round(duration_s * bins_per_second))
    crop = int(round(crop_seconds * bins_per_second))
    stride = max(1, int(round(crop * (1.0 - overlap))))
    if length < crop:
        return 0
    return (length - crop) // stride + 1


def safe_id(path: Path, view_id: str, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return f"{view_id}__" + "__".join(rel.parts).replace("_stageii", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-root", default="../legacy/datasets/AMASS_SMPLX_2022")
    parser.add_argument("--out-csv", default="scripts/amass_generation/amass_equal_target_manifest.csv")
    parser.add_argument("--summary-json", default="scripts/amass_generation/amass_equal_target_manifest_summary.json")
    parser.add_argument("--target-windows", type=int, default=23613)
    parser.add_argument("--bins-per-second", type=int, default=90)
    parser.add_argument("--crop-seconds", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    amass_root = (repo_root / args.amass_root).resolve()
    out_csv = (repo_root / args.out_csv).resolve()
    summary_json = (repo_root / args.summary_json).resolve()

    motions: list[dict[str, object]] = []
    skipped = 0
    for path in sorted(amass_root.rglob("*stageii.npz")):
        try:
            data = np.load(path, allow_pickle=False)
            frames = int(data["poses"].shape[0])
            rate = float(data["mocap_frame_rate"])
        except Exception:
            skipped += 1
            continue
        duration_s = (frames - 1) / rate
        windows = window_count(duration_s, args.bins_per_second, args.crop_seconds, args.overlap)
        if windows <= 0:
            skipped += 1
            continue
        motions.append(
            {
                "amass_npz": str(path),
                "native_frames": frames,
                "mocap_rate_hz": rate,
                "duration_s": duration_s,
                "estimated_windows": windows,
            }
        )

    rows: list[dict[str, object]] = []
    total_windows = 0
    for motion in motions:
        view = VIEWS["view0"]
        row = {
            **motion,
            "sequence_id": safe_id(Path(str(motion["amass_npz"])), "view0", amass_root),
            "view_id": "view0",
            **view,
        }
        rows.append(row)
        total_windows += int(motion["estimated_windows"])

    added_view1 = 0
    for motion in sorted(motions, key=lambda item: int(item["estimated_windows"]), reverse=True):
        if total_windows >= args.target_windows:
            break
        view = VIEWS["view1"]
        row = {
            **motion,
            "sequence_id": safe_id(Path(str(motion["amass_npz"])), "view1", amass_root),
            "view_id": "view1",
            **view,
        }
        rows.append(row)
        total_windows += int(motion["estimated_windows"])
        added_view1 += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "view_id",
        "amass_npz",
        "subject_range",
        "subject_lateral",
        "radar_height",
        "native_frames",
        "mocap_rate_hz",
        "duration_s",
        "estimated_windows",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "amass_root": str(amass_root),
        "manifest_csv": str(out_csv),
        "target_windows": args.target_windows,
        "selected_rows": len(rows),
        "valid_single_view_motions": len(motions),
        "added_view1_motions": added_view1,
        "skipped_files": skipped,
        "estimated_windows": total_windows,
        "estimated_hours_single_view": sum(float(m["duration_s"]) for m in motions) / 3600.0,
        "estimated_hours_selected": sum(float(r["duration_s"]) for r in rows) / 3600.0,
        "bins_per_second": args.bins_per_second,
        "crop_seconds": args.crop_seconds,
        "overlap": args.overlap,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
