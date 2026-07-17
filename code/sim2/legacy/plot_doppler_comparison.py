#!/usr/bin/env python3
"""Plot multiple Doppler spectra with one normalization per panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rtpose-sim2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a multi-panel Doppler comparison using per-panel log-power normalization."
    )
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Panel label and .npy Doppler spectrum path. Repeat once per panel.",
    )
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument("--title", default="Doppler Comparison")
    parser.add_argument("--low-percentile", type=float, default=10.0)
    parser.add_argument("--high-percentile", type=float, default=100.0)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--x-axis",
        choices=("column", "radar-frame", "time"),
        default="column",
        help="Horizontal axis convention. The default is the Doppler array column index.",
    )
    parser.add_argument("--raw-frame-start", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def parse_item(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"--item must be LABEL=PATH, got: {text}")
    label, path = text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Missing panel label in --item {text!r}")
    return label, Path(path).expanduser()


def normalize_panel(
    spectrum: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.isfinite(spectrum)
    shown = np.zeros_like(spectrum, dtype=np.float32)
    stats: dict[str, float] = {
        "log_low": float("nan"),
        "log_high": float("nan"),
        "power_min": float("nan"),
        "power_max": float("nan"),
    }
    if not finite.any():
        return shown, stats

    valid_power = np.maximum(spectrum[finite].astype(np.float64, copy=False), 0.0)
    log_power = np.log10(valid_power + 1.0)
    low, high = np.percentile(log_power, (low_percentile, high_percentile))
    stats.update(
        {
            "log_low": float(low),
            "log_high": float(high),
            "power_min": float(np.nanmin(valid_power)),
            "power_max": float(np.nanmax(valid_power)),
        }
    )
    if not np.isfinite(high) or high <= low:
        shown[finite] = 1.0
        return shown, stats

    shown[finite] = (log_power - low) / (high - low)
    return np.clip(shown, 0.0, 1.0), stats


def main() -> None:
    args = parse_args()
    items = [parse_item(text) for text in args.item]
    spectra = [(label, path, np.load(path).astype(np.float32, copy=False)) for label, path in items]
    common_doppler = min(arr.shape[0] for _, _, arr in spectra)
    common_frames = min(arr.shape[1] for _, _, arr in spectra)

    panels = []
    if args.x_axis == "column":
        extent = None
        x_label = "Frame column index"
    else:
        frame_ids = int(args.raw_frame_start) + np.arange(common_frames, dtype=np.float64)
        if args.x_axis == "radar-frame":
            x_values = frame_ids
            x_label = "Radar frame ID"
        else:
            if args.fps <= 0:
                raise ValueError("--fps must be positive")
            x_values = (frame_ids - 1.0) / float(args.fps)
            x_label = "Time (s)"
        extent = [float(x_values[0]), float(x_values[-1]), -0.5, float(common_doppler) - 0.5]

    summary: dict[str, object] = {
        "normalization": (
            "per-panel log10(power + 1), "
            f"p{args.low_percentile:g}-p{args.high_percentile:g} computed independently for each panel"
        ),
        "cropped_shape": [int(common_doppler), int(common_frames)],
        "x_axis": {
            "mode": args.x_axis,
            "raw_frame_start": int(args.raw_frame_start),
            "fps": float(args.fps),
            "label": x_label,
        },
        "panels": [],
    }
    for label, path, spectrum in spectra:
        if spectrum.ndim != 2:
            raise ValueError(f"Expected 2D Doppler spectrum for {path}, got {spectrum.shape}")
        cropped = spectrum[:common_doppler, :common_frames]
        shown, stats = normalize_panel(cropped, args.low_percentile, args.high_percentile)
        panels.append((label, shown))
        summary["panels"].append(
            {
                "label": label,
                "path": str(path),
                "original_shape": [int(spectrum.shape[0]), int(spectrum.shape[1])],
                **stats,
            }
        )

    fig_width = max(4.8 * len(panels), 7.0)
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, 5.3), dpi=args.dpi, sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for idx, (ax, (label, shown)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            shown,
            origin="lower",
            aspect="auto",
            cmap=args.cmap,
            vmin=0.0,
            vmax=1.0,
            extent=extent,
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(x_label)
        if idx == 0:
            ax.set_ylabel("Doppler bin")
        else:
            ax.tick_params(labelleft=False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(args.title, fontsize=14)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.12, top=0.86, wspace=0.28)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
