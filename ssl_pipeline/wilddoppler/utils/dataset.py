"""Dataset utilities for supervised load classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RadarSupervisedDataset(Dataset):
    _cache: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        csv_df: pd.DataFrame,
        data_dir: str | Path,
        split: str | None = "train",
        transform: Optional[transforms.Compose] = None,
        sample_length: float = 1.0,
        bins_per_second: int = 90,
        class_mapping: Optional[Dict[str, int]] = None,
        label_column: str = "activity_load",
        test_fold: Optional[int] = None,
        file_template: str = "{file_name}_track_{global_id}.npz",
        data_key: str = "uD",
        time_column: str = "t_start_rel",
        use_cache: bool = True,
        test_location: Optional[str] = None,
        is_regression: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sample_length = float(sample_length)
        self.bins_per_second = int(bins_per_second)
        self.class_mapping = class_mapping or {}
        self.label_column = str(label_column)
        self.file_template = str(file_template)
        self.data_key = str(data_key)
        self.time_column = str(time_column)
        self.use_cache = bool(use_cache)
        self.split = split
        self.test_fold = test_fold
        self.test_location = test_location
        self.is_regression = bool(is_regression)

        if test_location is not None and "location" in csv_df.columns:
            if split == "test":
                df = csv_df[csv_df["location"] == test_location]
            elif split == "train":
                df = csv_df[csv_df["location"] != test_location]
            else:
                df = csv_df.iloc[0:0].copy()
        elif "fold" in csv_df.columns and test_fold is not None:
            if split == "test":
                df = csv_df[csv_df["fold"] == test_fold]
            elif split == "train":
                df = csv_df[csv_df["fold"] != test_fold]
            else:
                df = csv_df.iloc[0:0].copy()
        elif split is not None and "split" in csv_df.columns:
            df = csv_df[csv_df["split"] == split]
        else:
            df = csv_df
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _build_path(self, row: pd.Series, data_dir: Optional[Path] = None) -> Path:
        if data_dir is None:
            data_dir = self.data_dir

        file_name = str(row.get("file_name", ""))
        global_id = row.get("global_id", None)
        segment_id = row.get("segment_id", None)
        try:
            filename = self.file_template.format(
                file_name=file_name,
                global_id=int(global_id) if global_id is not None else "",
                segment_id=int(segment_id) if segment_id is not None else "",
            )
        except Exception:
            filename = f"{file_name}_track_{int(global_id)}.npz"
        return data_dir / filename

    def _load_array(self, path: Path) -> np.ndarray:
        if not self.use_cache:
            data_obj = np.load(path)
            if self.data_key not in data_obj:
                raise KeyError(f"Expected key '{self.data_key}' in {path}.")
            return data_obj[self.data_key]
        key = str(path)
        if key not in self._cache:
            data_obj = np.load(path)
            if self.data_key not in data_obj:
                raise KeyError(f"Expected key '{self.data_key}' in {path}.")
            self._cache[key] = data_obj[self.data_key]
        return self._cache[key]

    def warmup_cache(self) -> None:
        """Load all unique .npz files into _cache before workers are forked."""
        if not self.use_cache:
            return
        seen: set = set()
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            path = self._build_path(row)
            key = str(path)
            if key not in seen:
                seen.add(key)
                self._load_array(path)
        print(f"Cache warmed: {len(seen)} files loaded ({self.split})")

    def _crop_time(self, md: np.ndarray, t_start: float, bins_per_second: int) -> np.ndarray:
        if md.ndim != 2:
            raise ValueError(f"Expected 2D [F, T] array, got shape {md.shape}.")

        t_total = md.shape[1]
        start_idx = int(round(t_start * bins_per_second))
        crop_bins = int(round(self.sample_length * bins_per_second))
        end_idx = min(start_idx + crop_bins, t_total)
        if start_idx < 0 or start_idx >= t_total:
            raise ValueError(f"Crop start_idx {start_idx} out of range for total {t_total}.")
        if end_idx <= start_idx:
            raise ValueError(f"Crop end_idx {end_idx} invalid for total {t_total}.")

        return md[:, start_idx:end_idx], start_idx, end_idx

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._build_path(row)
        md = self._load_array(path)

        t_start = float(row.get(self.time_column, 0.0))
        md_crop, start_idx, end_idx = self._crop_time(md, t_start, self.bins_per_second)

        x = torch.from_numpy(md_crop[None, :, :].astype(np.float32))

        if self.is_regression:
            # Predict radar-frame velocity components [v_radial, v_lateral].
            # The unit_polar loss derives speed and direction internally.
            vr = float(row["v_radial"]) if "v_radial" in row and pd.notna(row["v_radial"]) else float("nan")
            vl = float(row["v_lateral"]) if "v_lateral" in row and pd.notna(row["v_lateral"]) else float("nan")
            y = torch.tensor([vr, vl], dtype=torch.float32)
            activity = ""
        else:
            activity_value = row.get(self.label_column, None)
            activity = str(activity_value)
            if activity not in self.class_mapping:
                raise KeyError(f"Label '{activity}' not found in class_mapping.")
            y = torch.tensor(self.class_mapping[activity], dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        meta: Dict[str, Any] = {
            "file_name": str(row.get("file_name", "")),
            "global_id": int(row.get("global_id", -1)) if pd.notna(row.get("global_id", None)) else -1,
            "segment_id": int(row.get("segment_id", -1)) if pd.notna(row.get("segment_id", None)) else -1,
            "label": activity,
            "t_start": t_start,
            "duration": float(row.get("duration", 0.0)) if pd.notna(row.get("duration", None)) else 0.0,
            "split": str(row.get("split", "")),
            "path": str(path),
            "start_idx": start_idx,
            "end_idx": end_idx,
        }

        for col in ("activity_load", "activity_atomic", "x_mean", "y_mean", "vx_mean", "vy_mean"):
            if col in row:
                meta[col] = row[col]
        for col in ("short_track", "occlusion_frames", "close_person_frames"):
            if col in row and pd.notna(row[col]):
                meta[col] = int(row[col])
        for col in (
            "occlusion_avg",
            "occlusion_seconds",
            "close_person_avg_count",
            "close_person_avg_dist",
            "close_person_seconds",
            "x",
            "y",
            "vx",
            "vy",
            "distance_min",
            "distance_max",
            "range_min",
            "range_max",
        ):
            if col in row and pd.notna(row[col]):
                meta[col] = float(row[col])

        return x, y, meta
