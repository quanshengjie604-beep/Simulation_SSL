"""Training helper utilities shared across training scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: DictConfig) -> torch.device:
    device_cfg = getattr(cfg, "device", None)
    return torch.device(str(device_cfg)) if device_cfg else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_output_dir(cfg: DictConfig, exp_name: str) -> Path:
    output_dir = Path(cfg.paths.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def stats_to_dict(test_fold: Optional[int], stats) -> dict:
    nan = float("nan")
    return {
        "test_fold": test_fold,
        "loss": stats.loss, "accuracy": stats.acc,
        "macro_f1": stats.macro_f1 if stats.macro_f1 is not None else nan,
        "balanced_acc": stats.balanced_acc if stats.balanced_acc is not None else nan,
        "macro_precision": stats.macro_precision if stats.macro_precision is not None else nan,
        "macro_recall": stats.macro_recall if stats.macro_recall is not None else nan,
        "auroc": stats.auroc if stats.auroc is not None else nan,
        "auprc": stats.auprc if stats.auprc is not None else nan,
        "mae_speed": getattr(stats, "mae_speed", None) if getattr(stats, "mae_speed", None) is not None else nan,
        "mae_angle": getattr(stats, "mae_angle", None) if getattr(stats, "mae_angle", None) is not None else nan,
        "mae_radial": getattr(stats, "mae_radial", None) if getattr(stats, "mae_radial", None) is not None else nan,
        "mae_lateral": getattr(stats, "mae_lateral", None) if getattr(stats, "mae_lateral", None) is not None else nan,
    }
