from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DopplerTrackDataset(Dataset):
    """Windowed .npz micro-Doppler tracks listed by file_name, track_id, split."""

    def __init__(
        self,
        tracklist_csv: str,
        data_dir: str,
        split: str = "train",
        crop_seconds: float = 1.0,
        bins_per_second: int = 90,
        train_overlap_ratio: float = 0.5,
        resize_doppler: Optional[int] = 256,
        mean: float = 15.589631,
        std: float = 8.797207,
        cache_mode: str = "none",
    ) -> None:
        self.df = pd.read_csv(tracklist_csv)
        if "split" not in self.df.columns:
            raise ValueError(f"{tracklist_csv} must contain a split column.")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows with split={split!r} found in {tracklist_csv}.")

        self.data_dir = Path(data_dir)
        self.split = split
        self.crop_bins = int(round(float(crop_seconds) * int(bins_per_second)))
        self.bins_per_second = int(bins_per_second)
        self.train_overlap_ratio = max(0.0, min(float(train_overlap_ratio), 0.99))
        self.resize_doppler = int(resize_doppler) if resize_doppler else None
        self.mean = float(mean)
        self.std = max(1e-6, float(std))
        self.cache_mode = str(cache_mode or "none").lower()
        if self.cache_mode not in {"none", "lazy", "preload"}:
            raise ValueError("cache_mode must be one of: none, lazy, preload.")
        self._cache: dict[str, np.ndarray] = {}

        length_col = "uD_length" if "uD_length" in self.df.columns else "track_length"
        self.indices: list[tuple[int, int]] = []
        self._paths: dict[int, Path] = {}
        for row_idx, row in self.df.iterrows():
            path = self._build_path(row)
            if not path.exists():
                raise FileNotFoundError(path)
            self._paths[int(row_idx)] = path
            if length_col in self.df.columns and pd.notna(row.get(length_col)):
                total_bins = int(row[length_col])
            else:
                total_bins = int(self._read_track(path).shape[1])
            stride = self._train_stride_bins() if split == "train" else self.crop_bins
            if split == "train":
                count = max(1, (max(total_bins, self.crop_bins) - self.crop_bins) // stride + 1)
            else:
                count = max(1, math.ceil(total_bins / self.crop_bins))
            for win_idx in range(count):
                start = win_idx * stride
                if start + self.crop_bins > total_bins:
                    start = max(0, total_bins - self.crop_bins)
                self.indices.append((int(row_idx), int(start)))

        print(f"[{split}] tracks={len(self.df)} windows={len(self.indices)} crop_bins={self.crop_bins}")
        if self.cache_mode == "preload":
            for path in self._paths.values():
                self._load_track(path)
            gb = sum(arr.nbytes for arr in self._cache.values()) / (1024**3)
            print(f"[{split}] preloaded {len(self._cache)} tracks ({gb:.2f} GiB)")

    def _build_path(self, row) -> Path:
        return self.data_dir / f"{row['file_name']}_track_{int(row['track_id'])}.npz"

    def _read_track(self, path: Path) -> np.ndarray:
        with np.load(path) as data:
            return data["uD"].astype(np.float32, copy=False)

    def _load_track(self, path: Path) -> np.ndarray:
        key = str(path)
        if self.cache_mode != "none" and key in self._cache:
            return self._cache[key]
        arr = self._read_track(path)
        if self.cache_mode != "none":
            self._cache[key] = arr
        return arr

    def _train_stride_bins(self) -> int:
        return max(1, int(round(self.crop_bins * (1.0 - self.train_overlap_ratio))))

    def _crop_fixed(self, arr: np.ndarray, start: int) -> np.ndarray:
        crop = arr[:, start : start + self.crop_bins]
        if crop.shape[1] < self.crop_bins:
            pad = self.crop_bins - crop.shape[1]
            crop = np.pad(crop, ((0, 0), (0, pad)), mode="constant")
        return crop

    def _resize_freq(self, arr: np.ndarray) -> np.ndarray:
        if self.resize_doppler is None or arr.shape[0] == self.resize_doppler:
            return arr
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(self.resize_doppler, arr.shape[1]), mode="bilinear", align_corners=False)
        return x.squeeze(0).squeeze(0).numpy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        row_idx, start = self.indices[idx]
        row = self.df.iloc[row_idx]
        path = self._paths[row_idx]
        arr = self._resize_freq(self._crop_fixed(self._load_track(path), start))
        x = torch.from_numpy(arr[None].astype(np.float32, copy=False))
        x = (x - self.mean) / self.std
        meta = {
            "file_name": str(row["file_name"]),
            "track_id": int(row["track_id"]),
            "filename_id": f"{row['file_name']}_track_{int(row['track_id'])}",
            "start_bin": int(start),
            "path": str(path),
        }
        return x, meta


def _visible_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def make_loaders(args):
    common = dict(
        tracklist_csv=args.tracklist_csv,
        data_dir=args.data_dir,
        crop_seconds=args.crop_seconds,
        bins_per_second=args.bins_per_second,
        train_overlap_ratio=args.train_overlap_ratio,
        resize_doppler=args.resize_doppler,
        mean=args.uD_mean,
        std=args.uD_std,
        cache_mode=args.cache_mode,
    )
    train_ds = DopplerTrackDataset(split="train", **common)
    val_ds = None
    try:
        val_ds = DopplerTrackDataset(split="val", **common)
    except ValueError:
        pass

    workers = min(max(0, int(args.num_workers)), _visible_cpu_count())
    seed = int(getattr(args, "seed", 1))
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_ds is not None and bool(args.if_valid_set):
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader
