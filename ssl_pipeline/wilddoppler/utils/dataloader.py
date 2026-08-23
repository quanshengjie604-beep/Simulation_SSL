"""DataLoader construction and sampling utilities."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms

from .constants import is_regression_task
from .dataset import RadarSupervisedDataset


class ClassBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int,
        replacement: bool = True,
        max_batches: Optional[int] = None,
    ) -> None:
        if labels.numel() == 0:
            raise ValueError("No labels available for balanced batching.")
        self.labels = labels.long()
        self.num_classes = int(self.labels.max().item()) + 1
        if batch_size % self.num_classes != 0:
            raise ValueError("batch_size must be divisible by num_classes for balanced batching.")
        self.samples_per_class = batch_size // self.num_classes
        self.replacement = bool(replacement)

        self.indices_by_class = []
        for cls_idx in range(self.num_classes):
            cls_indices = torch.nonzero(self.labels == cls_idx, as_tuple=False).view(-1)
            if cls_indices.numel() == 0:
                raise ValueError(f"Class {cls_idx} has no samples for balanced batching.")
            self.indices_by_class.append(cls_indices)

        effective_len = int(self.labels.numel() // batch_size)
        self.max_batches = max(1, int(max_batches)) if max_batches else max(1, effective_len)

    def __len__(self) -> int:
        return self.max_batches

    def __iter__(self):
        for _ in range(self.max_batches):
            batch = []
            for cls_indices in self.indices_by_class:
                if self.replacement or cls_indices.numel() < self.samples_per_class:
                    choice = torch.randint(0, cls_indices.numel(), (self.samples_per_class,))
                else:
                    choice = torch.randperm(cls_indices.numel())[: self.samples_per_class]
                batch.extend(cls_indices[choice].tolist())
            random.shuffle(batch)
            yield batch


def _normalize_mapping(raw_mapping) -> Dict[str, int]:
    if raw_mapping is None:
        return {}
    if OmegaConf.is_config(raw_mapping):
        raw_mapping = OmegaConf.to_container(raw_mapping, resolve=True)
    if isinstance(raw_mapping, Mapping):
        return {str(k): int(v) for k, v in raw_mapping.items()}
    return dict(raw_mapping)


def _auto_class_mapping(df: pd.DataFrame, label_column: str) -> Dict[str, int]:
    unique_vals = sorted(df[label_column].astype(str).unique())
    return {val: idx for idx, val in enumerate(unique_vals)}


def _compute_stats(dataset: RadarSupervisedDataset) -> Tuple[float, float]:
    means = []
    stds = []
    for x, _, _ in dataset:
        means.append(torch.mean(x).item())
        stds.append(torch.std(x).item())
    if not means:
        raise RuntimeError("No samples found when computing stats.")
    return float(np.mean(means)), float(np.mean(stds))


def _resolve_subject_column(cfg: DictConfig, df: pd.DataFrame) -> Optional[str]:
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is not None:
        candidate = getattr(data_cfg, "subject_column", None)
        if candidate and candidate in df.columns:
            return str(candidate)
    for candidate in ("subject", "subject_id", "person_id", "pid", "participant_id"):
        if candidate in df.columns:
            return candidate
    return None


def _print_subject_stats(
    df: pd.DataFrame,
    label_column: str,
    subject_column: Optional[str],
    split_name: str,
) -> None:
    if df is None or df.empty:
        print(f"Subject stats ({split_name}): no rows.")
        return
    if subject_column is None:
        print(f"Subject stats ({split_name}): no subject column found; set data.subject_column to enable.")
        return
    if label_column not in df.columns:
        print(f"Subject stats ({split_name}): label column '{label_column}' not found.")
        return
    counts = df.groupby(label_column)[subject_column].nunique().sort_index()
    print(f"Subject stats ({split_name}) by class:")
    for label, count in counts.items():
        print(f"  {label}: {int(count)}")


def _apply_test_distance_filter(
    test_ds: RadarSupervisedDataset,
    min_dist_m: float = 2.0,
) -> RadarSupervisedDataset:
    """Remove test samples with mean distance < min_dist_m. Training set is left unchanged."""
    df = test_ds.df
    if "distance_mean" in df.columns:
        dist = pd.to_numeric(df["distance_mean"], errors="coerce")
    elif "x_mean" in df.columns and "y_mean" in df.columns:
        x = pd.to_numeric(df["x_mean"], errors="coerce")
        y = pd.to_numeric(df["y_mean"], errors="coerce")
        dist = np.sqrt(x**2 + y**2)
    else:
        print("[Test distance filter] No distance/position columns found; skipping filter.")
        return test_ds

    mask_remove = dist < min_dist_m
    removed_df = df[mask_remove].copy()
    removed_dist = dist[mask_remove]
    if not removed_df.empty:
        print(f"[Test distance filter] Removing {len(removed_df)} test samples (mean dist < {min_dist_m}m):")
        for idx, row in removed_df.iterrows():
            print(
                f"  file={row.get('file_name', '?')}, global_id={row.get('global_id', '?')},"
                f" dist={removed_dist.loc[idx]:.3f}m"
            )
    else:
        print(f"[Test distance filter] No test samples removed (all mean dist >= {min_dist_m}m).")

    test_ds.df = df[~mask_remove].reset_index(drop=True)
    return test_ds


def _apply_test_sample_length_filter(
    test_ds: RadarSupervisedDataset,
    min_length_s: float,
) -> RadarSupervisedDataset:
    """Remove test samples with duration < min_length_s. Training set is left unchanged."""
    df = test_ds.df
    if "duration" not in df.columns:
        print("[Test sample length filter] No duration column found; skipping filter.")
        return test_ds
    dur = pd.to_numeric(df["duration"], errors="coerce")
    mask_remove = dur < min_length_s
    n_removed = int(mask_remove.sum())
    if n_removed > 0:
        print(f"[Test sample length filter] Removing {n_removed} test samples (duration < {min_length_s}s).")
    else:
        print(f"[Test sample length filter] No test samples removed (all duration >= {min_length_s}s).")
    test_ds.df = df[~mask_remove].reset_index(drop=True)
    return test_ds


def _apply_train_percent_filter(
    train_ds: RadarSupervisedDataset,
    train_percent: Optional[float],
) -> None:
    if train_percent is None:
        return
    train_percent = float(train_percent)
    if train_percent <= 0 or train_percent > 100:
        raise ValueError(f"train_percent must be in (0, 100], got {train_percent}.")
    if train_percent >= 100:
        return
    if "data_percent" not in train_ds.df.columns:
        print("Train percent requested but 'data_percent' column not found; using full train set.")
        return
    percent_series = pd.to_numeric(train_ds.df["data_percent"], errors="coerce")
    if percent_series.isna().any():
        raise ValueError("Found non-numeric or missing values in data_percent; cannot filter train percent.")
    before_rows = len(train_ds.df)
    before_subjects = train_ds.df["global_id"].nunique() if "global_id" in train_ds.df.columns else None
    train_ds.df = train_ds.df[percent_series <= train_percent].reset_index(drop=True)
    after_rows = len(train_ds.df)
    after_subjects = train_ds.df["global_id"].nunique() if "global_id" in train_ds.df.columns else None
    msg = f"Train percent filter: <= {train_percent}% -> {after_rows}/{before_rows} rows"
    if before_subjects is not None and after_subjects is not None:
        msg += f", {after_subjects}/{before_subjects} subjects"
    print(msg + ".")


def _build_sampling(train_labels: torch.Tensor, class_counts: torch.Tensor, cfg) -> Dict[str, Optional[Sampler]]:
    setup = {"sampler": None, "batch_sampler": None, "shuffle": False}
    sampling_cfg = getattr(cfg.train, "sampling", None)
    strategy = "weighted"
    if sampling_cfg is not None and hasattr(sampling_cfg, "strategy"):
        strategy = str(getattr(sampling_cfg, "strategy", "weighted")).lower()

    if strategy == "none":
        setup["shuffle"] = True
        return setup

    if strategy == "weighted":
        power = float(getattr(sampling_cfg, "weight_power", 0.5)) if sampling_cfg else 0.5
        safe_counts = torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts))
        class_weights = torch.pow(safe_counts, -max(power, 1e-6))
        sample_weights = class_weights[train_labels]
        replacement = bool(getattr(sampling_cfg, "replacement", True)) if sampling_cfg else True
        setup["sampler"] = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=replacement,
        )
        return setup

    if strategy == "undersample":
        target_per_class = getattr(sampling_cfg, "target_per_class", None) if sampling_cfg else None
        if target_per_class is None:
            valid_counts = class_counts[class_counts > 0]
            target_per_class = int(valid_counts.min().item()) if valid_counts.numel() else 0
        target_per_class = int(target_per_class)
        selected = []
        for cls_idx in range(int(class_counts.numel())):
            cls_indices = torch.nonzero(train_labels == cls_idx, as_tuple=False).view(-1)
            if cls_indices.numel() == 0:
                continue
            take = min(cls_indices.numel(), target_per_class)
            perm = torch.randperm(cls_indices.numel())[:take]
            selected.append(cls_indices[perm])
        if not selected:
            raise ValueError("No samples selected during undersampling.")
        setup["sampler"] = SubsetRandomSampler(torch.cat(selected).tolist())
        return setup

    if strategy == "balanced_batch":
        replacement = bool(getattr(sampling_cfg, "balanced_replacement", True)) if sampling_cfg else True
        max_batches = getattr(sampling_cfg, "max_batches", None) if sampling_cfg else None
        setup["batch_sampler"] = ClassBalancedBatchSampler(
            train_labels,
            batch_size=int(cfg.train.batch_size),
            replacement=replacement,
            max_batches=max_batches,
        )
        return setup

    raise ValueError(f"Unsupported sampling strategy '{strategy}'.")


def build_dataloaders(cfg: DictConfig, test_fold: Optional[int] = None):
    df = pd.read_csv(cfg.file_list)

    cv_cfg = getattr(cfg, "cross_validation", None)
    test_location = str(getattr(cv_cfg, "test_location", None) or "") or None if cv_cfg is not None else None
    use_location_split = test_location is not None
    use_cv = bool(cv_cfg and getattr(cv_cfg, "enabled", False) and (test_fold is not None or use_location_split))
    train_percent = getattr(cfg, "train_percent", None)

    is_regression = is_regression_task(getattr(cfg, "task_name", ""))

    label_column = str(getattr(cfg.train, "label_column", "activity_load"))
    subject_column = _resolve_subject_column(cfg, df)
    if use_cv and use_location_split and "location" in df.columns:
        train_df = df[df["location"] != test_location]
        test_df = df[df["location"] == test_location]
        print(f"Location split: test={test_location!r} ({len(test_df)} rows), train={sorted(train_df['location'].unique().tolist())} ({len(train_df)} rows)")
        if not is_regression:
            _print_subject_stats(train_df, label_column, subject_column, "train (pre-filter)")
            _print_subject_stats(test_df, label_column, subject_column, "test")
        else:
            print(f"Regression splits — train: {len(train_df)}, test: {len(test_df)}")
    elif use_cv and "fold" in df.columns:
        test_df = df[df["fold"] == test_fold]
        train_df = df[df["fold"] != test_fold]
        if not is_regression:
            _print_subject_stats(train_df, label_column, subject_column, "train (pre-filter)")
            _print_subject_stats(test_df, label_column, subject_column, "test")
        else:
            print(f"Regression splits — train: {len(train_df)}, test: {len(test_df)}")
    elif "split" in df.columns:
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]
        if not is_regression:
            _print_subject_stats(train_df, label_column, subject_column, "train (pre-filter)")
            _print_subject_stats(test_df, label_column, subject_column, "test")
    else:
        if not is_regression:
            _print_subject_stats(df, label_column, subject_column, "all")

    if is_regression:
        class_mapping: Dict[str, int] = {}
        num_outputs = int(cfg.train.num_outputs)
        # num_classes is reused as the head output size for regression.
        cfg.train.num_classes = num_outputs
        print(f"Regression mode (unit_polar): num_outputs={num_outputs}")
    else:
        class_mapping = _normalize_mapping(getattr(cfg.train, "class_mapping", {}))
        if not class_mapping:
            class_mapping = _auto_class_mapping(df, label_column)
        cfg.train.class_mapping = class_mapping
        cfg.train.num_classes = len(class_mapping)
        label_names = list(getattr(cfg.train, "label_names", []))
        if not label_names:
            cfg.train.label_names = list(class_mapping.keys())

    use_cache = bool(getattr(cfg.data, "use_cache", True))

    resize = None
    win_size = cfg.transforms.win_size
    doppler_size = cfg.transforms.resize_doppler
    if getattr(cfg.transforms, "resize_doppler", None):
        resize = transforms.Resize((doppler_size, win_size))

    def _make_transform(mean_val, std_val):
        ops = [transforms.Normalize(mean=[mean_val], std=[std_val])]
        if resize is not None:
            ops.append(resize)
        return transforms.Compose(ops)

    if bool(getattr(cfg.transforms, "auto_calculate_stats", False)):
        stats_ds = RadarSupervisedDataset(
            df,
            cfg.data_dir,
            split="train",
            transform=None,
            sample_length=cfg.transforms.sample_length,
            bins_per_second=cfg.transforms.bins_per_second,
            class_mapping=class_mapping,
            label_column=label_column,
            test_fold=test_fold if use_cv else None,
            file_template=getattr(cfg.data, "file_template", "{file_name}_track_{global_id}.npz"),
            data_key=getattr(cfg.data, "data_key", "uD"),
            time_column=getattr(cfg.data, "time_column", "t_start_rel"),
            use_cache=use_cache,
            is_regression=is_regression,
        )
        mean, std = _compute_stats(stats_ds)
        print(f"Auto-calculated stats: mean={mean:.6f}, std={std:.6f}")
        with open_dict(cfg.transforms):
            cfg.transforms.uD_mean = float(mean)
            cfg.transforms.uD_std = float(std)
            cfg.transforms.auto_calculate_stats = False
    else:
        mean = float(cfg.transforms.uD_mean)
        std = float(cfg.transforms.uD_std)

    transform = _make_transform(mean, std)

    def _make_dataset(split: str | None):
        return RadarSupervisedDataset(
            df,
            cfg.data_dir,
            split=split,
            transform=transform,
            sample_length=cfg.transforms.sample_length,
            bins_per_second=cfg.transforms.bins_per_second,
            class_mapping=class_mapping,
            label_column=label_column,
            test_fold=test_fold if (use_cv and not use_location_split) else None,
            file_template=getattr(cfg.data, "file_template", "{file_name}_track_{global_id}.npz"),
            data_key=getattr(cfg.data, "data_key", "uD"),
            time_column=getattr(cfg.data, "time_column", "t_start_rel"),
            use_cache=use_cache,
            test_location=test_location if use_location_split else None,
            is_regression=is_regression,
        )

    train_ds = _make_dataset("train")
    _apply_train_percent_filter(train_ds, train_percent)
    if not is_regression:
        _print_subject_stats(train_ds.df, label_column, subject_column, "train (post-filter)")

    if is_regression:
        train_vr = pd.to_numeric(train_ds.df["v_radial"], errors="coerce").dropna()
        train_vl = pd.to_numeric(train_ds.df["v_lateral"], errors="coerce").dropna()
        train_speed = np.sqrt(train_vr.values ** 2 + train_vl.values ** 2)
        cfg.train.label_mean = float(train_speed.mean())
        std_val = float(train_speed.std())
        cfg.train.label_std = std_val if std_val != 0.0 else 1.0
        print(
            f"Label stats (train, unit_polar): "
            f"speed mean={cfg.train.label_mean:.4f} std={cfg.train.label_std:.4f}"
        )

    test_ds = _make_dataset("test")
    test_ds = _apply_test_distance_filter(test_ds)
    test_ds = _apply_test_sample_length_filter(test_ds, float(cfg.transforms.sample_length))

    train_ds.warmup_cache()
    test_ds.warmup_cache()

    if is_regression:
        # No class-based sampling for regression — always shuffle.
        sampling = {"sampler": None, "batch_sampler": None, "shuffle": True}
    else:
        train_labels = []
        for _, y, _ in train_ds:
            train_labels.append(y.view(-1))
        train_labels = torch.cat(train_labels) if train_labels else torch.empty(0, dtype=torch.long)
        if train_labels.numel() == 0:
            raise RuntimeError("No training labels found; check split/filter settings.")
        class_counts = torch.bincount(train_labels, minlength=int(cfg.train.num_classes)).float()
        sampling = _build_sampling(train_labels, class_counts, cfg)

    _timeout = int(getattr(cfg.train, "dataloader_timeout", 120))

    if sampling["batch_sampler"] is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampling["batch_sampler"],
            num_workers=cfg.train.num_workers,
            timeout=_timeout,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            sampler=sampling["sampler"],
            shuffle=sampling["shuffle"],
            num_workers=cfg.train.num_workers,
            drop_last=True,
            timeout=_timeout,
        )

    print(f"Train dataset size: {len(train_ds)}")

    if use_location_split:
        print(f"Test location: {test_location}")
    else:
        print(f"Test fold: {test_fold}")
    print(f"Test dataset size: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        timeout=_timeout,
    )

    return train_loader, test_loader


def build_eval_test_loader(
    cfg: DictConfig,
    test_fold: Optional[int] = None,
    test_location: Optional[str] = None,
):
    """Build a single test DataLoader for eval-only flows.

    Defaults to using every row in cfg.file_list. Pass test_fold to restrict to
    one fold's test split, or test_location to restrict to one location. The
    two are mutually exclusive.
    """
    if test_fold is not None and test_location is not None:
        raise ValueError("Specify at most one of test_fold or test_location.")

    df = pd.read_csv(cfg.file_list)

    location_column = "location"

    if test_fold is not None:
        if "fold" not in df.columns:
            raise ValueError("eval.test_fold set but CSV has no 'fold' column.")
        eval_df = df[df["fold"] == int(test_fold)].reset_index(drop=True)
        split_label = f"fold {test_fold}"
    elif test_location is not None:
        if location_column not in df.columns:
            raise ValueError(
                f"eval.test_location set but CSV has no '{location_column}' column."
            )
        eval_df = df[df[location_column] == str(test_location)].reset_index(drop=True)
        split_label = f"location {test_location!r}"
    else:
        eval_df = df.reset_index(drop=True)
        split_label = "all rows"

    if len(eval_df) == 0:
        raise ValueError(f"Eval split '{split_label}' is empty after filtering.")

    is_regression = is_regression_task(getattr(cfg, "task_name", ""))
    label_column = str(getattr(cfg.train, "label_column", "activity_load"))
    subject_column = _resolve_subject_column(cfg, eval_df)

    if not is_regression:
        _print_subject_stats(eval_df, label_column, subject_column, f"eval ({split_label})")
    else:
        print(f"Eval ({split_label}): {len(eval_df)} rows.")

    if is_regression:
        class_mapping: Dict[str, int] = {}
        num_outputs = int(cfg.train.num_outputs)
        cfg.train.num_classes = num_outputs

        if test_fold is not None:
            label_train_df = df[df["fold"] != int(test_fold)].reset_index(drop=True)
            label_split_label = f"fold != {test_fold}"
        elif test_location is not None:
            label_train_df = df[df[location_column] != str(test_location)].reset_index(drop=True)
            label_split_label = f"location != {test_location!r}"
        else:
            label_train_df = df.reset_index(drop=True)
            label_split_label = "all rows"
        if len(label_train_df) == 0:
            raise ValueError(
                f"Train portion empty when recomputing label stats ({label_split_label})."
            )
        train_vr = pd.to_numeric(label_train_df["v_radial"], errors="coerce").dropna()
        train_vl = pd.to_numeric(label_train_df["v_lateral"], errors="coerce").dropna()
        train_speed = np.sqrt(train_vr.values ** 2 + train_vl.values ** 2)
        cfg.train.label_mean = float(train_speed.mean())
        std_val = float(train_speed.std())
        cfg.train.label_std = std_val if std_val != 0.0 else 1.0
        print(
            f"Label stats (train portion {label_split_label}, unit_polar): "
            f"speed mean={cfg.train.label_mean:.4f} std={cfg.train.label_std:.4f}"
        )
    else:
        class_mapping = _normalize_mapping(getattr(cfg.train, "class_mapping", {}))
        if not class_mapping:
            class_mapping = _auto_class_mapping(df, label_column)
        cfg.train.class_mapping = class_mapping
        cfg.train.num_classes = len(class_mapping)
        if not list(getattr(cfg.train, "label_names", [])):
            cfg.train.label_names = list(class_mapping.keys())

    saved_mean = getattr(cfg.transforms, "uD_mean", None)
    saved_std = getattr(cfg.transforms, "uD_std", None)
    if saved_mean is not None and saved_std is not None:
        mean = float(saved_mean)
        std = float(saved_std)
    elif bool(getattr(cfg.transforms, "auto_calculate_stats", False)):
        if test_fold is not None:
            train_df = df[df["fold"] != int(test_fold)].reset_index(drop=True)
            stats_split_label = f"fold != {test_fold}"
        elif test_location is not None:
            task_name = str(getattr(cfg, "task_name", "") or "")
            if task_name == "MotionState":
                train_df = df.reset_index(drop=True)
                stats_split_label = f"all rows (cross-loc MotionState, test_location={test_location!r})"
            else:
                train_df = df[df[location_column] != str(test_location)].reset_index(drop=True)
                stats_split_label = f"location != {test_location!r}"
        else:
            train_df = df.reset_index(drop=True)
            stats_split_label = "all rows"
        if len(train_df) == 0:
            raise ValueError(
                f"Train portion empty when recomputing eval stats ({stats_split_label})."
            )
        print(f"Recomputing uD_mean/uD_std over train portion ({stats_split_label}, {len(train_df)} rows)")
        stats_ds = RadarSupervisedDataset(
            train_df,
            cfg.data_dir,
            split=None,
            transform=None,
            sample_length=cfg.transforms.sample_length,
            bins_per_second=cfg.transforms.bins_per_second,
            class_mapping=class_mapping,
            label_column=label_column,
            test_fold=None,
            file_template=getattr(cfg.data, "file_template", "{file_name}_track_{global_id}.npz"),
            data_key=getattr(cfg.data, "data_key", "uD"),
            time_column=getattr(cfg.data, "time_column", "t_start_rel"),
            use_cache=bool(getattr(cfg.data, "use_cache", True)),
            is_regression=is_regression,
        )
        mean, std = _compute_stats(stats_ds)
        print(f"Auto-calculated stats: mean={mean:.6f}, std={std:.6f}")
    else:
        raise ValueError(
            "transforms.uD_mean / transforms.uD_std missing from saved config and "
            "auto_calculate_stats is disabled. Set them via override.transforms.uD_mean=... "
            "and override.transforms.uD_std=..."
        )

    ops = [transforms.Normalize(mean=[mean], std=[std])]
    win_size = cfg.transforms.win_size
    doppler_size = cfg.transforms.resize_doppler
    if getattr(cfg.transforms, "resize_doppler", None):
        ops.append(transforms.Resize((doppler_size, win_size)))
    transform = transforms.Compose(ops)

    test_ds = RadarSupervisedDataset(
        eval_df,
        cfg.data_dir,
        split=None,
        transform=transform,
        sample_length=cfg.transforms.sample_length,
        bins_per_second=cfg.transforms.bins_per_second,
        class_mapping=class_mapping,
        label_column=label_column,
        test_fold=None,
        file_template=getattr(cfg.data, "file_template", "{file_name}_track_{global_id}.npz"),
        data_key=getattr(cfg.data, "data_key", "uD"),
        time_column=getattr(cfg.data, "time_column", "t_start_rel"),
        use_cache=bool(getattr(cfg.data, "use_cache", True)),
        test_location=None,
        is_regression=is_regression,
    )
    test_ds = _apply_test_distance_filter(test_ds)
    test_ds = _apply_test_sample_length_filter(test_ds, float(cfg.transforms.sample_length))
    test_ds.warmup_cache()

    _timeout = int(getattr(cfg.train, "dataloader_timeout", 120))
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        timeout=_timeout,
    )
    print(f"Eval test dataset size: {len(test_ds)} ({split_label})")
    return test_loader
