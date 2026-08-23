"""Train/evaluate a supervised load classification model from Hydra config."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
import math
import pandas as pd

from utils import (
    apply_supervised_constants,
    build_dataloaders, build_supervised_model, is_regression_task, SupervisedTrainer,
    set_seed, get_device, setup_output_dir, stats_to_dict,
)
import matplotlib
matplotlib.use('Agg')

_set_seed = set_seed
_get_device = get_device
_setup_output_dir = setup_output_dir
_stats_to_dict = stats_to_dict

def train(
    cfg: DictConfig,
    test_fold: Optional[int],
    output_dir: Optional[Path] = None,
    all_test_stats: List[dict] = [],
    fold_name: Optional[str] = None,
) -> List[dict]:
    seed = int(getattr(cfg, "seed", 1))
    _set_seed(seed)

    device = _get_device(cfg)
    print(f"Using device: {device}")

    train_percent = getattr(cfg, "train_percent", None)
    if train_percent is not None:
        print(f"Train percent: {train_percent}% (train folds only).")

    train_loader, test_loader = build_dataloaders(cfg, test_fold=test_fold)
    model = build_supervised_model(cfg).to(device)

    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        test_fold=test_fold,
        fold_name=fold_name,
    )
    test_stats = trainer.train()
    if test_stats is not None:
        all_test_stats.append(_stats_to_dict(test_fold, test_stats))

    return all_test_stats

@hydra.main(version_base="1.2", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.method_name = "supervised"
    cfg = apply_supervised_constants(cfg)

    all_test_stats: List[dict] = []

    n_folds = cfg.cross_validation.n_folds
    cross_location_cv = bool(getattr(cfg, "cross_location", False))
    is_regression = is_regression_task(getattr(cfg, "task_name", ""))
    avg_metric_keys = (
        ("mae_speed", "mae_angle", "mae_radial", "mae_lateral")
        if is_regression
        else ("balanced_acc",)
    )

    def _nanmean(values: list) -> float:
        vals = [v for v in values if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def _nanstd(values: list) -> float:
        vals = [v for v in values if not math.isnan(v)]
        if len(vals) < 2:
            return float("nan")
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))

    base_exp_name = str(cfg.paths.exp_name)
    output_dir = _setup_output_dir(cfg, base_exp_name)

    def _write_avg_summary(stats: List[dict]) -> None:
        if not stats:
            print("No fold stats collected; skipping avg summary.")
            return
        avg_log = {}
        for metric in avg_metric_keys:
            vals = [s[metric] for s in stats]
            avg_log[f"avg/{metric}"] = _nanmean(vals)
            avg_log[f"std/{metric}"] = _nanstd(vals)

        header = f"=== Avg over {len(stats)} fold(s): {base_exp_name} ==="
        body_lines = [
            f"  {metric}: {avg_log[f'avg/{metric}']:.4f} ± {avg_log[f'std/{metric}']:.4f}"
            for metric in avg_metric_keys
        ]
        print(f"\n{header}")
        for line in body_lines:
            print(line)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics_summary.txt").write_text(
            "\n".join([header, *body_lines]) + "\n"
        )

    if cross_location_cv:
        df = pd.read_csv(cfg.file_list)
        locations = sorted(df["location"].unique().tolist())
        print(f"Cross-location CV: {len(locations)} locations = {locations}")

        for loc in locations:
            OmegaConf.update(cfg, "cross_validation.test_location", str(loc))
            fold_name = f"fold_{loc}"

            all_test_stats = train(
                cfg=cfg, test_fold=None,
                output_dir=output_dir,
                all_test_stats=all_test_stats,
                fold_name=fold_name,
            )

        _write_avg_summary(all_test_stats)
    else:
        for test_fold in range(n_folds):
            fold_name = f"fold{test_fold}"

            all_test_stats = train(
                cfg=cfg, test_fold=test_fold,
                output_dir=output_dir,
                all_test_stats=all_test_stats,
                fold_name=fold_name,
            )

        _write_avg_summary(all_test_stats)

if __name__ == "__main__":
    main()
