#!/usr/bin/env python3
"""Convert Sim2 raw echo to ROI Doppler spectra and compare them with GT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
RAW_TO_DOPPLER = CODE_ROOT / "Echo_data_processing" / "raw_echo_to_doppler_spectrum.py"
EVALUATE_DOPPLER = CODE_ROOT / "Quantitative_analysis" / "evaluate_doppler_spectrum_similarity.py"
DEFAULT_CALIB = CODE_ROOT / "Echo_data_processing" / "calibrateResults_high.mat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-convert Sim2 raw radar echo bins into Doppler-time ROI spectra, "
            "then run the Quantitative Doppler similarity evaluation against GT spectra."
        )
    )
    parser.add_argument(
        "--sim2-dataset-dir",
        default=str(REPO_ROOT / "datasets" / "Sim2_sequences"),
        help="Directory containing Sim2 sequences, each with radar/bin raw echo files.",
    )
    parser.add_argument(
        "--gt-spectrum-dir",
        default=str(REPO_ROOT / "results" / "GT" / "doppler_spectrum_roi"),
        help="Existing GT Doppler spectrum directory.",
    )
    parser.add_argument(
        "--sim2-spectrum-dir",
        default=str(REPO_ROOT / "results" / "Sim2" / "doppler_spectrum_roi"),
        help="Where Sim2 Doppler spectra (.npy) will be written.",
    )
    parser.add_argument(
        "--comparison-out-dir",
        default=str(REPO_ROOT / "results" / "Quantitative_analysis" / "doppler_spectrum_similarity_sim2"),
        help="Where per-sequence Doppler similarity JSON/CSV reports will be written.",
    )
    parser.add_argument(
        "--train",
        default=str(REPO_ROOT / "datasets" / "Train.json"),
        help="Motion annotation JSON used by raw_echo_to_doppler_spectrum.py for ROI interpolation.",
    )
    parser.add_argument(
        "--sequence",
        "--sequences",
        dest="sequences",
        nargs="+",
        help="Sequence IDs to process. Default: auto-discover Sim2 sequences with radar/bin.",
    )
    parser.add_argument(
        "--sequence-file",
        help=(
            "Optional JSON or text file listing sequence IDs. JSON dict keys, JSON list items, "
            "or whitespace-separated text entries are accepted."
        ),
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child scripts.")
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy", "torch"), default="auto")
    parser.add_argument("--gpu-device", type=int, help="CUDA device ID passed to the raw-to-Doppler converter.")
    parser.add_argument("--calib", default=str(DEFAULT_CALIB), help="Calibration .mat file for raw echo processing.")
    parser.add_argument(
        "--rangemat-correction",
        choices=("on", "off"),
        required=True,
        help="Explicitly enable or disable RangeMat correction in raw-to-Doppler conversion.",
    )
    parser.add_argument(
        "--peakvalmat-correction",
        choices=("on", "off"),
        required=True,
        help="Explicitly enable or disable PeakValMat phase-only correction in raw-to-Doppler conversion.",
    )
    parser.add_argument("--file-idx", default="", help="Radar bin file index, e.g. 0000. Default uses the first index.")
    parser.add_argument("--roi-margin", type=float, default=0.5, help="Meters added around the pose bbox ROI.")
    parser.add_argument(
        "--roi-reducer",
        choices=("mean", "max", "topk-mean"),
        required=True,
        help="How raw-to-Doppler reduces Doppler power over xyz voxels inside the ROI.",
    )
    parser.add_argument(
        "--roi-topk-fraction",
        type=float,
        default=0.05,
        help="Fraction of highest-power ROI voxels averaged when --roi-reducer=topk-mean.",
    )
    parser.add_argument("--frame-start", type=int, help="First radar frame id for every sequence.")
    parser.add_argument("--frame-stop", type=int, help="Exclusive stop radar frame id for every sequence.")
    parser.add_argument("--gt-first-frame-id", type=int, required=True)
    parser.add_argument("--sim2-first-frame-id", type=int, required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--stop-time", type=float, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--ssim-clip", nargs=2, type=float, required=True, metavar=("LOW", "HIGH"))
    parser.add_argument("--spectrum-clip", nargs=2, type=float, required=True, metavar=("LOW", "HIGH"))
    parser.add_argument("--ssim-window", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing Sim2 Doppler spectra.")
    parser.add_argument("--skip-convert", action="store_true", help="Skip raw echo conversion and only run similarity evaluation.")
    parser.add_argument("--skip-compare", action="store_true", help="Only generate Sim2 Doppler spectra.")
    parser.add_argument("--keep-going", action="store_true", help="Continue converting later sequences if one fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print child commands without running them.")
    return parser.parse_args()


def sequence_sort_key(seq: str) -> tuple[int, int | str]:
    return (0, int(seq)) if seq.isdigit() else (1, seq)


def normalize_sequences(items: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    sequences: list[str] = []
    for item in items:
        seq = str(item).strip()
        if not seq or seq in seen:
            continue
        seen.add(seq)
        sequences.append(seq)
    return sorted(sequences, key=sequence_sort_key)


def load_sequences_from_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return normalize_sequences(text.split())

    if isinstance(data, dict):
        return normalize_sequences(data.keys())
    if isinstance(data, list):
        return normalize_sequences(data)
    raise ValueError(f"{path} must contain a JSON dict, JSON list, or whitespace-separated sequence IDs")


def discover_sequences(sim2_dataset_dir: Path) -> list[str]:
    if not sim2_dataset_dir.exists():
        raise FileNotFoundError(f"Missing Sim2 dataset dir: {sim2_dataset_dir}")
    sequences = []
    for child in sim2_dataset_dir.iterdir():
        if child.is_dir() and (child / "radar" / "bin").is_dir():
            sequences.append(child.name)
    return normalize_sequences(sequences)


def require_raw_dirs(sim2_dataset_dir: Path, sequences: list[str]) -> None:
    missing = [seq for seq in sequences if not (sim2_dataset_dir / seq / "radar" / "bin").is_dir()]
    if missing:
        missing_text = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise FileNotFoundError(f"Missing Sim2 radar/bin for sequences: {missing_text}{suffix}")


def print_command(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)


def run_command(cmd: list[str], dry_run: bool) -> None:
    print_command(cmd)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_sim2_frame_range(sim2_dataset_dir: Path, sequence: str) -> tuple[int, int] | None:
    summary_path = sim2_dataset_dir / sequence / f"seq{sequence}_mesh_echo_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return int(summary["start_radar_frame"]), int(summary["max_radar_frame"])


def conversion_command(args: argparse.Namespace, sequence: str) -> list[str]:
    sim2_dataset_dir = Path(args.sim2_dataset_dir)
    frame_range = load_sim2_frame_range(sim2_dataset_dir, sequence)
    cmd = [
        args.python,
        str(RAW_TO_DOPPLER),
        "--dataset-dir",
        str(sim2_dataset_dir),
        "--sequence",
        sequence,
        "--train",
        str(Path(args.train)),
        "--out-dir",
        str(Path(args.sim2_spectrum_dir)),
        "--roi-margin",
        str(args.roi_margin),
        "--backend",
        args.backend,
        "--calib",
        str(Path(args.calib)),
        "--rangemat-correction",
        args.rangemat_correction,
        "--peakvalmat-correction",
        args.peakvalmat_correction,
        "--roi-reducer",
        args.roi_reducer,
        "--roi-topk-fraction",
        str(args.roi_topk_fraction),
    ]
    if args.gpu_device is not None:
        cmd.extend(["--gpu-device", str(args.gpu_device)])
    if args.file_idx:
        cmd.extend(["--file-idx", args.file_idx])
    if frame_range is not None:
        raw_frame_start, _raw_frame_stop_inclusive = frame_range
        cmd.extend(["--raw-frame-start", str(raw_frame_start)])
    if args.frame_start is not None:
        cmd.extend(["--frame-start", str(args.frame_start)])
    if args.frame_stop is not None:
        cmd.extend(["--frame-stop", str(args.frame_stop)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def convert_sequences(args: argparse.Namespace, sequences: list[str]) -> list[str]:
    out_dir = Path(args.sim2_spectrum_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    failed: list[tuple[str, BaseException]] = []

    for index, sequence in enumerate(sequences, start=1):
        npy_path = out_dir / f"{sequence}.npy"
        if npy_path.exists() and not args.overwrite:
            print(f"[{index}/{len(sequences)}] skip existing {npy_path}", flush=True)
            completed.append(sequence)
            continue

        print(f"[{index}/{len(sequences)}] convert Sim2 sequence {sequence}", flush=True)
        try:
            run_command(conversion_command(args, sequence), args.dry_run)
        except subprocess.CalledProcessError as exc:
            if not args.keep_going:
                raise
            failed.append((sequence, exc))
            print(f"[{sequence}] conversion failed: {exc}", flush=True)
            continue
        completed.append(sequence)

    if failed:
        failed_text = ", ".join(seq for seq, _ in failed)
        print(f"conversion failures ({len(failed)}): {failed_text}", flush=True)
    return completed


def compare_command(args: argparse.Namespace, sequence: str) -> list[str]:
    out_dir = Path(args.comparison_out_dir)
    return [
        args.python,
        str(EVALUATE_DOPPLER),
        "--reference",
        str(Path(args.gt_spectrum_dir) / f"{sequence}.npy"),
        "--candidate",
        str(Path(args.sim2_spectrum_dir) / f"{sequence}.npy"),
        "--reference-label",
        "GT",
        "--candidate-label",
        "Sim2",
        "--reference-first-frame-id",
        str(args.gt_first_frame_id),
        "--candidate-first-frame-id",
        str(args.sim2_first_frame_id),
        "--start-time",
        str(args.start_time),
        "--stop-time",
        str(args.stop_time),
        "--fps",
        str(args.fps),
        "--ssim-clip",
        str(args.ssim_clip[0]),
        str(args.ssim_clip[1]),
        "--spectrum-clip",
        str(args.spectrum_clip[0]),
        str(args.spectrum_clip[1]),
        "--ssim-window",
        str(args.ssim_window),
        "--out-json",
        str(out_dir / f"{sequence}_doppler_similarity.json"),
        "--out-csv",
        str(out_dir / f"{sequence}_doppler_similarity.csv"),
    ]


def compare_sequences(args: argparse.Namespace, sequences: list[str]) -> None:
    out_dir = Path(args.comparison_out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, BaseException]] = []
    for index, sequence in enumerate(sequences, start=1):
        print(f"[{index}/{len(sequences)}] compare Doppler sequence {sequence}", flush=True)
        try:
            run_command(compare_command(args, sequence), args.dry_run)
        except subprocess.CalledProcessError as exc:
            if not args.keep_going:
                raise
            failed.append((sequence, exc))
            print(f"[{sequence}] comparison failed: {exc}", flush=True)
    if failed:
        failed_text = ", ".join(seq for seq, _ in failed)
        print(f"comparison failures ({len(failed)}): {failed_text}", flush=True)


def collect_sequences(args: argparse.Namespace) -> list[str]:
    explicit: list[str] = []
    if args.sequences:
        explicit.extend(args.sequences)
    if args.sequence_file:
        explicit.extend(load_sequences_from_file(Path(args.sequence_file)))
    if explicit:
        return normalize_sequences(explicit)
    return discover_sequences(Path(args.sim2_dataset_dir))


def main() -> int:
    args = parse_args()
    sim2_dataset_dir = Path(args.sim2_dataset_dir)
    sequences = collect_sequences(args)
    if not sequences:
        raise RuntimeError(f"No Sim2 sequences found under {sim2_dataset_dir}")
    require_raw_dirs(sim2_dataset_dir, sequences)

    print(f"selected sequences: {len(sequences)}", flush=True)
    converted_or_existing = sequences
    if not args.skip_convert:
        converted_or_existing = convert_sequences(args, sequences)
    if args.skip_compare:
        return 0
    if not converted_or_existing:
        raise RuntimeError("No Sim2 Doppler spectra are available for comparison")

    compare_sequences(args, converted_or_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
