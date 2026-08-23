"""Evaluate supervised, MAE-, or contrastive-fine-tuned checkpoints from a run directory.

Given a directory containing one `{fold_name}.pt` checkpoint per fold (e.g.
`fold0.pt`, `fold1.pt`, `fold2.pt`, or `fold_A.pt`), this script:

1. Discovers every `{fold_name}.pt` file in `eval_checkpoint_dir`.
2. Parses the fold name to recover `test_fold` (cross-subject) or
   `test_location` (cross-location).
3. Rebuilds the model from constants/config, loads each fold checkpoint,
   runs that fold's held-out split, and writes `predictions_{fold_name}.csv`
   to `output_dir` (default: `<eval_checkpoint_dir>/eval/`).
4. Prints per-fold metrics plus mean ± std and pooled summaries.

The flow used to build the model is selected via `method_name`:
  - `supervised` (default): rebuild via `build_supervised_model`.
  - `mae` / `contrastive`: rebuild the SSL backbone from the pretrained
    SSL ckpt (auto-derived from `ssl_ckpts_dir + method_name + model_name`,
    or set explicitly via `paths.ckpt`), then attach a linear-probe head
    matching the architecture saved in the fine-tuned fold checkpoint.

Usage:
    # Supervised:
    python eval.py eval_checkpoint_dir=/abs/path/to/run_dir/

    # MAE / contrastive fine-tuned:
    python eval.py task_name=MotionState method_name=reconstruction \\
        eval_checkpoint_dir=/abs/path/to/finetuned_dir/

    # Override the data CSV via Hydra at runtime:
    python eval.py eval_checkpoint_dir=/abs/path/to/run_dir/ \\
        file_list=/path/to/all_folds.csv
"""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")

from utils import (
    apply_ssl_eval_constants,
    apply_supervised_constants,
    build_eval_test_loader,
    build_supervised_model,
    get_device,
    is_regression_task,
    resolve_default_checkpoint_dir,
    set_seed,
    SupervisedTrainer,
)
from utils.metrics import (
    classification_metrics,
    compute_auprc,
    compute_auroc,
    regression_metrics,
    save_predictions_csv,
    save_regression_csv,
)
from ssl_eval.utils.contrastive_utils import (
    load_contrastive_checkpoint,
    resolve_contrastive_embedding_dim,
)
from ssl_eval.utils.mae_utils import (
    MAEEmbeddingWrapper,
    load_mae_checkpoint,
    resolve_mae_embedding_dim,
)


_SSL_EMBEDDING_KEY = {
    "reconstruction": "embedding",
    "contrastive": "pooled",
}


def _resolve_ssl_embedding_key(method: str, task_name: str) -> str:
    # Contrastive velocity-regression fine-tunes were trained with the
    # projection-head embedding ("embedding"), not the pooled backbone
    # feature, so eval must match to load the head shapes.
    if method == "contrastive" and is_regression_task(task_name):
        return "embedding"
    return _SSL_EMBEDDING_KEY.get(method, "embedding")


class _SSLLinearProbeClassifier(nn.Module):
    """Backbone + MLP head, mirroring MAELinearProbeClassifier and
    ContrastiveLinearProbeClassifier from the tune scripts so saved
    state_dicts load with matching key paths."""

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_outputs: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        embedding_key: str = "embedding",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_key = embedding_key

        dims = [int(embed_dim)] + [int(d) for d in (hidden_dims or [])] + [int(num_outputs)]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(p=float(dropout)))
        self.head = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.backbone(x, cond=cond)
        embedding = feats.get(self.embedding_key, feats.get("embedding", feats.get("pooled")))
        if embedding is None:
            raise KeyError("SSL backbone outputs must include 'embedding' or 'pooled'.")
        return self.head(embedding)


def _nanmean(vals: List[float]) -> float:
    arr = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return float(np.mean(arr)) if arr else float("nan")


def _nanstd(vals: List[float]) -> float:
    arr = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _regression_target_names() -> List[str]:
    return ["v_radial", "v_lateral"]


_LABEL_STATS_SENTINEL = object()


def _denormalize_preds(train_cfg, preds: np.ndarray) -> np.ndarray:
    if preds.size == 0:
        return preds
    label_mean = getattr(train_cfg.train, "label_mean", _LABEL_STATS_SENTINEL)
    label_std = getattr(train_cfg.train, "label_std", _LABEL_STATS_SENTINEL)
    if label_mean is _LABEL_STATS_SENTINEL or label_std is _LABEL_STATS_SENTINEL:
        raise ValueError(
            "train.label_mean / train.label_std are required for regression eval; "
            "build_eval_test_loader should populate them from the train portion."
        )
    out = preds.copy()
    out[:, 2] = out[:, 2] * float(label_std) + float(label_mean)
    return out


def _unit_polar_to_velocity(preds_eval: np.ndarray) -> np.ndarray:
    eps = 1e-8
    a, b, speed = preds_eval[:, 0], preds_eval[:, 1], preds_eval[:, 2]
    denom = np.sqrt(a ** 2 + b ** 2 + eps)
    return np.stack([speed * a / denom, speed * b / denom], axis=1)


def _parse_fold_name(fold_name: str) -> Tuple[Optional[int], Optional[str]]:
    """Return (test_fold, test_location) inferred from a `{fold_name}.pt` stem."""
    m = re.fullmatch(r"fold(\d+)", fold_name)
    if m:
        return int(m.group(1)), None
    m = re.fullmatch(r"fold_(.+)", fold_name)
    if m:
        return None, m.group(1)
    raise ValueError(
        f"Cannot parse fold name {fold_name!r}; expected 'fold<N>' or 'fold_<location>'."
    )


def _discover_checkpoints(ckpt_dir: Path) -> List[Path]:
    """Return all `*.pt` files in ckpt_dir whose stem parses as a fold name."""
    candidates = sorted(ckpt_dir.glob("*.pt"))
    folds: List[Tuple[Tuple[int, str], Path]] = []
    for path in candidates:
        try:
            test_fold, test_location = _parse_fold_name(path.stem)
        except ValueError:
            continue
        # Sort cross-subject folds by index, cross-location alphabetically.
        sort_key = (0, f"{test_fold:09d}") if test_fold is not None else (1, str(test_location))
        folds.append((sort_key, path))
    folds.sort(key=lambda item: item[0])
    return [path for _, path in folds]


def _save_fold_predictions(
    train_cfg: DictConfig,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    meta_df: Optional[pd.DataFrame],
    out_dir: Path,
    fold_name: str,
    test_fold: Optional[int],
) -> dict:
    """Persist `predictions_<fold_name>.csv` and return metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_fields: dict = {}
    exp_name = getattr(getattr(train_cfg, "paths", None), "exp_name", None)
    if exp_name:
        extra_fields["exp_name"] = exp_name
    if test_fold is not None:
        extra_fields["test_fold"] = int(test_fold)

    pred_path = out_dir / f"predictions_{fold_name}.csv"

    is_regression = is_regression_task(getattr(train_cfg, "task_name", ""))
    if is_regression:
        target_names = _regression_target_names()
        preds_eval = _denormalize_preds(train_cfg, preds)
        metrics = regression_metrics(labels, preds_eval, target_names)
        preds_for_csv = (
            _unit_polar_to_velocity(preds_eval) if preds_eval.size > 0 else preds_eval
        )
        save_regression_csv(
            labels,
            preds_for_csv,
            target_names,
            meta_df,
            pred_path,
            extra_fields=extra_fields or None,
        )
    else:
        save_predictions_csv(
            labels,
            preds,
            meta_df,
            pred_path,
            extra_fields=extra_fields or None,
            probs=probs if probs.size > 0 else None,
        )
        metrics = classification_metrics(labels, preds)
        if probs.size > 0 and probs.ndim == 2:
            metrics["auroc"] = compute_auroc(labels, probs)
            metrics["auprc"] = compute_auprc(labels, probs)
    return metrics


def _infer_head_hidden_dims(state_dict: dict) -> List[int]:
    """Recover hidden-layer sizes of an SSL probe head from its saved state_dict.

    The tune scripts build `head` either as `nn.Linear` (single layer; keys
    `head.weight` / `head.bias`) or `nn.Sequential` of Linear/ReLU/(Dropout)
    blocks (keys `head.0.weight`, `head.2.weight`, ...). Only Linear layers
    carry weights, so listing the `.weight` keys in order recovers the layer
    output dims; the last is the model's output dim, the rest are hidden dims.
    """
    if "head.weight" in state_dict:
        return []
    weight_keys = [
        k for k in state_dict
        if k.startswith("head.") and k.endswith(".weight") and k.count(".") == 2
    ]
    if not weight_keys:
        raise ValueError(
            "Checkpoint state_dict has no 'head.*.weight' or 'head.weight' entries; "
            "cannot infer linear-probe head architecture."
        )
    weight_keys.sort(key=lambda k: int(k.split(".")[1]))
    out_dims = [int(state_dict[k].shape[0]) for k in weight_keys]
    return out_dims[:-1]


def _resolve_num_outputs(train_cfg: DictConfig) -> int:
    is_regression = is_regression_task(getattr(train_cfg, "task_name", ""))
    if is_regression:
        n = int(getattr(train_cfg.train, "num_outputs", 0))
        if n <= 0:
            raise ValueError("train.num_outputs must be set for regression eval.")
        return n
    n = int(getattr(train_cfg.train, "num_classes", 0))
    if n <= 0:
        raise ValueError("train.num_classes must be set for classification eval.")
    return n


def _load_ssl_backbone_template(
    method: str,
    train_cfg: DictConfig,
    device: torch.device,
    embedding_key: str,
) -> Tuple[nn.Module, int]:
    """Load the pretrained SSL backbone (used purely for architecture) and
    return a wrapper compatible with the probe classifier built by the tune
    scripts, plus the embedding dim.

    `paths.ckpt` is auto-derived by `apply_ssl_eval_constants` from
    `ssl_ckpts_dir + method_name + model_name` when not set explicitly.
    """
    pretrained_raw = getattr(getattr(train_cfg, "paths", None), "ckpt", None)
    if pretrained_raw in (None, ""):
        raise ValueError(
            "paths.ckpt must point to the pretrained SSL backbone "
            f"({method}); set ssl_ckpts_dir or paths.ckpt explicitly."
        )
    pretrained_path = Path(str(pretrained_raw)).resolve()
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained {method} ckpt not found: {pretrained_path}")

    if method == "reconstruction":
        backbone, _, _ = load_mae_checkpoint(pretrained_path, device)
        wrapper = MAEEmbeddingWrapper(backbone, embedding_key=embedding_key).to(device)
        embed_dim = resolve_mae_embedding_dim(backbone)
        return wrapper, int(embed_dim)
    if method == "contrastive":
        backbone, _, _ = load_contrastive_checkpoint(pretrained_path, device)
        if embedding_key == "pooled":
            embed_dim = int(getattr(getattr(backbone, "backbone", None), "embed_dim", 0))
            if embed_dim <= 0:
                raise AttributeError("Contrastive backbone is missing 'backbone.embed_dim'.")
        else:
            embed_dim = resolve_contrastive_embedding_dim(backbone)
        return backbone, int(embed_dim)
    raise ValueError(f"Unsupported SSL method: {method!r}")


def _evaluate_one_fold_ssl(
    train_cfg: DictConfig,
    method: str,
    backbone_template: nn.Module,
    embed_dim: int,
    embedding_key: str,
    ckpt_path: Path,
    out_dir: Path,
    fold_name: str,
    test_fold: Optional[int],
    test_location: Optional[str],
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    set_seed(int(getattr(train_cfg, "seed", 1)))
    device = get_device(train_cfg)
    split_label = (
        f"fold {test_fold}" if test_fold is not None
        else f"location {test_location!r}"
    )
    print(f"\n=== Evaluating {split_label} ({fold_name}, method={method}) ===")
    print(f"  device:     {device}")
    print(f"  checkpoint: {ckpt_path}")

    test_loader = build_eval_test_loader(
        train_cfg, test_fold=test_fold, test_location=test_location
    )

    finetuned = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = finetuned.get("model_state", finetuned)
    hidden_dims = _infer_head_hidden_dims(state_dict)
    dropout = float(getattr(getattr(train_cfg, "linear_probe", {}), "dropout", 0.0))
    num_outputs = _resolve_num_outputs(train_cfg)

    model = _SSLLinearProbeClassifier(
        backbone=copy.deepcopy(backbone_template),
        embed_dim=embed_dim,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        dropout=dropout,
        embedding_key=embedding_key,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    trainer = SupervisedTrainer(
        model=model,
        train_loader=None,
        test_loader=test_loader,
        cfg=train_cfg,
        device=device,
        output_dir=out_dir,
        test_fold=test_fold,
        fold_name=fold_name,
    )
    labels, preds, probs, meta_df = trainer._collect_predictions(test_loader)
    metrics = _save_fold_predictions(
        train_cfg, labels, preds, probs, meta_df, out_dir, fold_name, test_fold
    )
    print("  metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics, labels, preds, probs, meta_df


def _evaluate_one_fold(
    train_cfg: DictConfig,
    ckpt_path: Path,
    out_dir: Path,
    fold_name: str,
    test_fold: Optional[int],
    test_location: Optional[str],
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    set_seed(int(getattr(train_cfg, "seed", 1)))
    device = get_device(train_cfg)
    split_label = (
        f"fold {test_fold}" if test_fold is not None
        else f"location {test_location!r}"
    )
    print(f"\n=== Evaluating {split_label} ({fold_name}) ===")
    print(f"  device:     {device}")
    print(f"  checkpoint: {ckpt_path}")

    test_loader = build_eval_test_loader(
        train_cfg, test_fold=test_fold, test_location=test_location
    )
    model = build_supervised_model(train_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    trainer = SupervisedTrainer(
        model=model,
        train_loader=None,
        test_loader=test_loader,
        cfg=train_cfg,
        device=device,
        output_dir=out_dir,
        test_fold=test_fold,
        fold_name=fold_name,
    )
    labels, preds, probs, meta_df = trainer._collect_predictions(test_loader)
    metrics = _save_fold_predictions(
        train_cfg, labels, preds, probs, meta_df, out_dir, fold_name, test_fold
    )
    print("  metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics, labels, preds, probs, meta_df


def _print_summary(
    per_fold: List[dict],
    pooled_metrics: Optional[dict],
    out_dir: Path,
) -> None:
    df = pd.DataFrame(per_fold)
    metric_keys = [k for k in df.columns if k != "fold"]
    if "balanced_acc" in metric_keys:
        metric_keys = ["balanced_acc"] + [k for k in metric_keys if k != "balanced_acc"]
        df = df[["fold", *metric_keys]]

    header = "=== Per-fold + averaged metrics ==="

    def _fmt(v: float) -> str:
        return "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"

    means = {k: _nanmean(df[k].tolist()) for k in metric_keys}
    stds = {k: _nanstd(df[k].tolist()) for k in metric_keys}
    top_rows = [
        {"fold": "mean", **{k: _fmt(means[k]) for k in metric_keys}},
        {"fold": "std",  **{k: _fmt(stds[k])  for k in metric_keys}},
    ]
    if pooled_metrics is not None:
        top_rows.append({"fold": "pooled", **{k: _fmt(pooled_metrics.get(k, float("nan"))) for k in metric_keys}})
    per_fold_str = df.copy()
    for k in metric_keys:
        per_fold_str[k] = per_fold_str[k].map(_fmt)
    out = pd.concat([pd.DataFrame(top_rows), per_fold_str], ignore_index=True)
    body = out.to_string(index=False)

    print(f"\n{header}")
    print(body)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_summary.txt").write_text(f"{header}\n{body}\n")


def _pool_metrics(
    per_fold: List[dict],
    ref_cfg: DictConfig,
) -> Optional[dict]:
    is_regression = is_regression_task(getattr(ref_cfg, "task_name", ""))
    labels_all = np.concatenate([f["labels"] for f in per_fold]) if per_fold else np.array([])
    preds_all = np.concatenate([f["preds"] for f in per_fold]) if per_fold else np.array([])
    probs_lists = [f["probs"] for f in per_fold if f["probs"].size > 0]
    probs_all = np.concatenate(probs_lists) if probs_lists else np.array([])
    if labels_all.size == 0:
        return None

    if is_regression:
        target_names = _regression_target_names()
        preds_eval_per_fold = [
            _denormalize_preds(ref_cfg, f["preds"]) for f in per_fold
        ]
        preds_eval_all = np.concatenate(preds_eval_per_fold) if preds_eval_per_fold else np.array([])
        return regression_metrics(labels_all, preds_eval_all, target_names)

    metrics = classification_metrics(labels_all, preds_all)
    if probs_all.size > 0 and probs_all.ndim == 2:
        metrics["auroc"] = compute_auroc(labels_all, probs_all)
        metrics["auprc"] = compute_auprc(labels_all, probs_all)
    return metrics


@hydra.main(version_base="1.2", config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    method = str(getattr(cfg, "method_name", "supervised") or "supervised").lower()
    if method == "supervised":
        train_cfg = apply_supervised_constants(cfg)
    elif method in ("reconstruction", "contrastive"):
        train_cfg = apply_ssl_eval_constants(cfg)
    else:
        raise ValueError(
            f"Unsupported method_name {method!r}; expected one of supervised|reconstruction|contrastive."
        )

    ckpt_dir_raw = OmegaConf.select(train_cfg, "eval_checkpoint_dir", default=None)
    if ckpt_dir_raw in (None, ""):
        ckpt_dir_raw = resolve_default_checkpoint_dir(train_cfg)
        print(f"eval_checkpoint_dir unset; using DEFAULT_CKPTS entry: {ckpt_dir_raw}")
    ckpt_dir = Path(str(ckpt_dir_raw)).resolve()
    if not ckpt_dir.is_dir():
        raise NotADirectoryError(f"eval_checkpoint_dir is not a directory: {ckpt_dir}")

    out_dir_raw = train_cfg.get("output_dir")
    out_dir_base = (
        Path(str(out_dir_raw)).resolve() if out_dir_raw not in (None, "") else ckpt_dir / "eval"
    )

    base_ckpts_dir_raw = OmegaConf.select(train_cfg, "ckpts_dir", default=None)
    exp_subfolder: Optional[Path] = None
    if base_ckpts_dir_raw not in (None, ""):
        try:
            exp_subfolder = ckpt_dir.relative_to(Path(str(base_ckpts_dir_raw)).resolve())
        except ValueError:
            exp_subfolder = None
    if exp_subfolder is None:
        exp_subfolder = Path(ckpt_dir.name)
    timestamp = datetime.now().strftime("%b%d_%H%M")
    parts = exp_subfolder.parts
    exp_subfolder = Path(f"{parts[0]}_{timestamp}", *parts[1:])
    out_dir = out_dir_base / exp_subfolder

    ckpt_paths = _discover_checkpoints(ckpt_dir)
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No fold checkpoints found in {ckpt_dir} (expected files like fold0.pt, fold_<location>.pt)."
        )
    print(f"Discovered {len(ckpt_paths)} fold checkpoint(s) in {ckpt_dir} (method={method})")
    print(f"Writing predictions to {out_dir}")

    backbone_template = None
    embed_dim = 0
    embedding_key = _resolve_ssl_embedding_key(
        method, str(getattr(train_cfg, "task_name", "") or "")
    )
    if method in ("reconstruction", "contrastive"):
        device = get_device(train_cfg)
        backbone_template, embed_dim = _load_ssl_backbone_template(
            method, train_cfg, device, embedding_key
        )

    per_fold: List[dict] = []
    for ckpt_path in ckpt_paths:
        fold_name = ckpt_path.stem
        test_fold, test_location = _parse_fold_name(fold_name)
        if method == "supervised":
            metrics, labels, preds, probs, meta_df = _evaluate_one_fold(
                train_cfg,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                fold_name=fold_name,
                test_fold=test_fold,
                test_location=test_location,
            )
        else:
            metrics, labels, preds, probs, meta_df = _evaluate_one_fold_ssl(
                train_cfg,
                method=method,
                backbone_template=backbone_template,
                embed_dim=embed_dim,
                embedding_key=embedding_key,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                fold_name=fold_name,
                test_fold=test_fold,
                test_location=test_location,
            )
        per_fold.append(
            {
                "fold": fold_name,
                "metrics": metrics,
                "labels": labels,
                "preds": preds,
                "probs": probs,
                "meta": meta_df,
            }
        )

    pooled_metrics = _pool_metrics(per_fold, ref_cfg=train_cfg)
    summary_rows = [{"fold": f["fold"], **f["metrics"]} for f in per_fold]
    _print_summary(summary_rows, pooled_metrics, out_dir)


if __name__ == "__main__":
    main()
