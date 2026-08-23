#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional
import sys
import re
import copy
import math

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from DopplerWild.utils import (
    apply_ssl_eval_constants,
    apply_supervised_constants,
    build_dataloaders,
    is_regression_task,
    set_seed,
    SupervisedTrainer,
)

from DopplerWild.ssl_eval.utils import (
    KNNTask,
    build_eval_loader,
    run_feature_visualizations,
    run_knn_probes,
    run_knn_dual_exclusion,
    load_contrastive_checkpoint,
    resolve_contrastive_embedding_dim,
)
from DopplerWild.ssl_eval.utils import knn as knn_utils
from DopplerWild.ssl_eval.utils.feature_utils import FeaturePack, unpack_meta


class ContrastiveEmbeddingWrapper(nn.Module):
    def __init__(self, model: nn.Module, embedding_key: str = "embedding"):
        super().__init__()
        self.model = model
        self.embedding_key = embedding_key

    @property
    def embedding_keys(self) -> tuple[str, ...]:
        return (self.embedding_key,)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict:
        feats = self.model(x, cond=cond)
        embedding = feats.get(self.embedding_key, feats.get("embedding", feats.get("pooled")))
        if embedding is None:
            raise KeyError("Contrastive model outputs must include 'embedding' or 'pooled'.")
        return {self.embedding_key: embedding}


class ContrastiveLinearProbeClassifier(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embed_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        train_backbone: bool = False,
        embedding_key: str = "embedding",
    ):
        super().__init__()
        self.backbone = model
        self.train_backbone = bool(train_backbone)
        self.embedding_key = embedding_key

        dims = [int(embed_dim)] + [int(dim) for dim in (hidden_dims or [])] + [int(num_classes)]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            in_dim, out_dim = dims[idx], dims[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(p=float(dropout)))
        self.head = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

        for param in self.backbone.parameters():
            param.requires_grad = self.train_backbone

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.backbone(x, cond=cond)
        embedding = feats.get(self.embedding_key, feats.get("embedding", feats.get("pooled")))
        if embedding is None:
            raise KeyError("Contrastive model outputs must include 'embedding' or 'pooled'.")
        return self.head(embedding)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self


def extract_features(
    model: nn.Module,
    loader,
    device: torch.device,
    embedding_keys: Sequence[str],
    split_name: str,
) -> FeaturePack:
    banks = {key: [] for key in embedding_keys}
    meta_entries = []
    with torch.no_grad():
        for x, _, meta in loader:
            x = x.to(device, dtype=torch.float32)
            outputs = model(x)
            for key in embedding_keys:
                if key not in outputs:
                    raise KeyError(f"Embedding key '{key}' missing from model outputs.")
                banks[key].append(outputs[key].detach().cpu().numpy())
            entries = unpack_meta(meta)
            for entry in entries:
                entry.setdefault("split", split_name)
            meta_entries.extend(entries)
    embeddings = {key: np.concatenate(parts, axis=0) for key, parts in banks.items()}
    return FeaturePack(embeddings=embeddings, meta=meta_entries)


def _resolve_path(path_value, base_dir: Path) -> Path:
    raw_path = Path(str(path_value)).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates = [
        base_dir / raw_path,
        Path(__file__).resolve().parent / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _sanitize_for_filename(value: str) -> str:
    """Return a filesystem-friendly string (letters, numbers, underscore, dash, dot)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "unnamed"


def _resolve_run_tag(folder_name, exp_name, ckpt_path: Path) -> str:
    if folder_name not in (None, ""):
        return str(folder_name)
    if exp_name not in (None, ""):
        return str(exp_name)
    return ckpt_path.stem


def _normalize_ckpt_specs(paths_cfg, base_dir: Path) -> list[dict]:
    specs: list[dict] = []

    def _add_entry(name, path_value, folder_override):
        if path_value in (None, ""):
            return
        entry = {"path": _resolve_path(path_value, base_dir)}
        if name not in (None, ""):
            entry["name"] = str(name)
        if folder_override not in (None, ""):
            entry["folder_name"] = str(folder_override)
        specs.append(entry)

    ckpts_cfg = getattr(paths_cfg, "ckpts", None)
    if isinstance(ckpts_cfg, Mapping):
        for raw_name, raw_value in ckpts_cfg.items():
            if isinstance(raw_value, Mapping):
                name = raw_value.get("name", raw_name)
                folder_override = raw_value.get("folder_name")
                path_value = raw_value.get("path")
            else:
                name = raw_name
                folder_override = None
                path_value = raw_value
            _add_entry(name, path_value, folder_override)
    elif isinstance(ckpts_cfg, Sequence) and not isinstance(ckpts_cfg, (str, bytes, bytearray)):
        for idx, item in enumerate(ckpts_cfg, start=1):
            if isinstance(item, Mapping):
                name = item.get("name", f"ckpt{idx}")
                folder_override = item.get("folder_name")
                path_value = item.get("path")
            else:
                name = f"ckpt{idx}"
                folder_override = None
                path_value = item
            _add_entry(name, path_value, folder_override)

    if not specs:
        _add_entry(None, getattr(paths_cfg, "ckpt", None), getattr(paths_cfg, "folder_name", None))

    return specs


def _coerce_label_list(raw_value, default) -> tuple[str, ...]:
    if raw_value in (None, "", []):
        return tuple(default)
    if isinstance(raw_value, (list, tuple, ListConfig)):
        return tuple(str(item) for item in raw_value)
    return (str(raw_value),)


def _resolve_linear_probe_specs(lp_cfg) -> list[dict]:
    if not lp_cfg or not bool(getattr(lp_cfg, "enabled", False)):
        return []

    default_hidden_dims = [int(v) for v in (getattr(lp_cfg, "hidden_dims", []) or [])]
    default_dropout = float(getattr(lp_cfg, "dropout", 0.0))
    default_train_backbone = bool(getattr(lp_cfg, "train_backbone", False))
    variants_cfg = getattr(lp_cfg, "variants", None)

    def _build_spec(name: str, spec_cfg) -> dict | None:
        if spec_cfg is not None and not bool(getattr(spec_cfg, "enabled", True)):
            return None
        hidden_dims = default_hidden_dims
        dropout = default_dropout
        train_backbone = default_train_backbone
        if spec_cfg is not None:
            raw_hidden_dims = getattr(spec_cfg, "hidden_dims", None)
            if raw_hidden_dims not in (None, ""):
                hidden_dims = [int(v) for v in (raw_hidden_dims or [])]
            raw_dropout = getattr(spec_cfg, "dropout", None)
            if raw_dropout not in (None, ""):
                dropout = float(raw_dropout)
            raw_train_backbone = getattr(spec_cfg, "train_backbone", None)
            if raw_train_backbone is not None:
                train_backbone = bool(raw_train_backbone)
        return {
            "name": str(name),
            "slug": _sanitize_for_filename(str(name)),
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "train_backbone": train_backbone,
        }

    specs: list[dict] = []
    if isinstance(variants_cfg, Mapping):
        for raw_name, raw_cfg in variants_cfg.items():
            spec = _build_spec(str(raw_name), raw_cfg)
            if spec is not None:
                specs.append(spec)
    elif isinstance(variants_cfg, Sequence) and not isinstance(variants_cfg, (str, bytes, bytearray)):
        for idx, raw_cfg in enumerate(variants_cfg, start=1):
            raw_name = getattr(raw_cfg, "name", f"probe_{idx}") if raw_cfg is not None else f"probe_{idx}"
            spec = _build_spec(str(raw_name), raw_cfg)
            if spec is not None:
                specs.append(spec)

    if specs:
        return specs

    default_spec = _build_spec("default", lp_cfg)
    return [] if default_spec is None else [default_spec]


def _coerce_log_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _nanmean(values: list) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _nanstd(values: list) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if len(vals) < 2:
        return float("nan")
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _epoch_stats_to_dict(stats) -> dict:
    """Convert an EpochStats into a flat {metric_name: value} dict, dropping None fields.

    Regression-only fields (mae_speed, mae_angle) and classification-only fields
    (acc, macro_f1, ...) are emitted only when the trainer populated them; the
    other set is left as None and filtered out here.
    """
    fields = [
        "loss",
        "acc", "macro_f1", "balanced_acc", "macro_precision", "macro_recall", "auroc", "auprc",
        "mae_speed", "mae_angle", "mae_radial", "mae_lateral",
    ]
    return {f: getattr(stats, f) for f in fields if getattr(stats, f) is not None}


def _extract_fold_summary(metrics_rows: list[dict]) -> dict:
    """Extract scalar metrics from metrics_rows for cross-fold averaging."""
    result = {}
    for row in metrics_rows:
        key = row.get("metric", "")
        val = row.get("value")
        if isinstance(val, (int, float)):
            result[key] = float(val)
    return result




def _summarize_folds(
    all_fold_summaries: list[dict],
    tag: str | None = None,
    keys_to_print: Sequence[str] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Compute avg/std across folds, print the human-readable summary, and return the
    flat {"avg/<key>": ..., "std/<key>": ...} dict for the caller to log wherever it
    wants. Returns {} when no fold summaries were collected.

    `tag` appears after "fold(s)" in the header, matching the supervised script's format.
    `keys_to_print` overrides the default sorted/filtered key set; pass it (e.g. for
    regression) to fix both the printed metrics and their order.
    `output_dir`, when given, also writes the same printed text to
    `<output_dir>/metrics_summary.txt`.
    """
    if not all_fold_summaries:
        print("No fold summaries collected; skipping avg summary.")
        return {}
    all_keys: set[str] = set()
    for summary in all_fold_summaries:
        all_keys.update(summary.keys())
    avg_log: dict = {}
    for key in all_keys:
        vals = [s[key] for s in all_fold_summaries if key in s]
        avg_log[f"avg/{key}"] = _nanmean(vals)
        avg_log[f"std/{key}"] = _nanstd(vals)

    header_suffix = f": {tag}" if tag else ""
    header = f"=== Avg over {len(all_fold_summaries)} fold(s){header_suffix} ==="
    if keys_to_print is not None:
        ordered = [k for k in keys_to_print if k in all_keys]
    else:
        ordered = sorted(all_keys)
        bal_keys = [k for k in ordered if k == "balanced_acc" or k.endswith("/balanced_acc")]
        if bal_keys:
            ordered = bal_keys + [k for k in ordered if k not in bal_keys]
    body_lines = [
        f"  {key}: {avg_log[f'avg/{key}']:.4f} ± {avg_log[f'std/{key}']:.4f}"
        for key in ordered
    ]
    print(f"\n{header}")
    for line in body_lines:
        print(line)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics_summary.txt").write_text(
            "\n".join([header, *body_lines]) + "\n"
        )

    return avg_log


def _save_activity_predictions(
    train_pack,
    test_pack,
    task: KNNTask,
    k: int,
    normalize: bool,
    whiten_cfg,
    run_dir: Path,
    fold_name: str,
    model_name: str | None = None,
    classification_name: str | None = None,
    test_fold: int | None = None,
):
    """Save per-sample KNN predictions for the activity task (include-same-track)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    train_labels = knn_utils.gather_labels(train_pack.meta, task.label_key, task.label_fallbacks)
    test_labels = knn_utils.gather_labels(test_pack.meta, task.label_key, task.label_fallbacks)
    train_y, test_y, label_names = knn_utils._encode_labels(train_labels, test_labels)
    train_feats = train_pack.embeddings[task.embedding_key]
    test_feats = test_pack.embeddings[task.embedding_key]

    if normalize:
        train_feats = knn_utils._l2_normalize(train_feats)
        test_feats = knn_utils._l2_normalize(test_feats)

    whiten_enabled = False
    whiten_method = "pca"
    whiten_eps = 1e-5
    if whiten_cfg is not None:
        whiten_enabled = bool(getattr(whiten_cfg, "enabled", False))
        whiten_method = str(getattr(whiten_cfg, "method", "pca")).lower()
        whiten_eps = float(getattr(whiten_cfg, "eps", 1e-5))
    if whiten_enabled:
        train_feats, test_feats = knn_utils._pca_whiten(train_feats, test_feats, method=whiten_method, eps=whiten_eps)
        train_feats = knn_utils._l2_normalize(train_feats)
        test_feats = knn_utils._l2_normalize(test_feats)

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(train_feats, train_y)
    preds = knn.predict(test_feats)
    id_to_label = {i: lbl for i, lbl in enumerate(label_names)}
    true_labels_str = [id_to_label.get(int(y), str(y)) for y in test_y]
    pred_labels_str = [id_to_label.get(int(y), str(y)) for y in preds]

    meta_df = pd.DataFrame(test_pack.meta)
    if classification_name:
        meta_df.insert(0, "multilabel_classification_name", classification_name)
    if model_name:
        meta_df.insert(0, "model_name", model_name)
    if "fold" not in meta_df.columns:
        meta_df["fold"] = test_fold if test_fold is not None else -1
    meta_df["true_label"] = true_labels_str
    meta_df["pred_label"] = pred_labels_str
    out_path = run_dir / f"predictions_{fold_name}.csv"
    meta_df.to_csv(out_path, index=False)
    print(f"Saved activity predictions CSV to {out_path}")


def _sync_lr_scheduler_t_max(train_cfg) -> None:
    """Sync lr_scheduler.t_max to train.epochs."""
    lr_sched = getattr(train_cfg, "lr_scheduler", None)
    if lr_sched is None:
        return
    if str(getattr(lr_sched, "name", "none")).lower() in {"none", "null", "off"}:
        return
    epochs = getattr(train_cfg, "epochs", None)
    if epochs is not None:
        OmegaConf.update(train_cfg, "lr_scheduler.t_max", int(epochs))


def _run_supervised_linear_probe(
    contrastive_model: nn.Module,
    cls_cfg: DictConfig,
    data_train,
    data_test,
    device: torch.device,
    embed_dim: int,
    probe_output_dir: Path,
    fold_name: str,
    probe_name: str,
    train_backbone: bool,
    embedding_key: str,
    hidden_dims: Sequence[int],
    dropout: float,
    test_fold: int | None = None,
) -> None:
    import copy
    num_classes = int(getattr(cls_cfg.train, "num_classes", 0))
    if num_classes <= 0:
        raise ValueError("train.num_classes must be set before running linear probe.")

    # Deep-copy backbone so each probe variant starts from the original SSL weights,
    # not weights mutated by a previous full-finetune variant.
    backbone = copy.deepcopy(contrastive_model)
    model = ContrastiveLinearProbeClassifier(
        backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        train_backbone=train_backbone,
        embedding_key=embedding_key,
    ).to(device)
    model.train_backbone = train_backbone
    trainer = SupervisedTrainer(
        model=model,
        train_loader=data_train,
        test_loader=data_test,
        cfg=cls_cfg,
        device=device,
        output_dir=probe_output_dir,
        test_fold=test_fold,
        fold_name=fold_name,
    )
    return trainer.train()


def _run_contrastive_tune_for_ckpt(
    cfg: DictConfig,
    ckpt_path: Path,
    output_dir: Path,
    run_tag: str,
    device: torch.device,
    seed: int,
    n_folds: int = 3,
) -> list[dict]:
    # --- Fold-independent setup ---
    # The global cfg already carries everything the classification flow needs;
    # apply_supervised_constants layers the supervised-only sections (transforms,
    # train block defaults, model defaults, etc.) on top.
    cls_cfg_template = copy.deepcopy(cfg)
    cls_cfg_template = apply_supervised_constants(cls_cfg_template)
    OmegaConf.set_struct(cls_cfg_template, False)
    OmegaConf.resolve(cls_cfg_template)
    _sync_lr_scheduler_t_max(cls_cfg_template.train)
    is_regression = is_regression_task(getattr(cfg, "task_name", ""))
    if is_regression:
        print(
            "Regression task detected; skipping viz, KNN probes, and "
            "frozen-backbone heads. Only full-finetune linear-probe variants will run."
        )
    eval_cv_cfg = getattr(cfg, "cross_validation", None)
    if not getattr(cls_cfg_template, "paths", None):
        cls_cfg_template.paths = OmegaConf.create({})
    cls_cfg_template.paths.encoder_type = "contrastive"
    cls_cfg_template.paths.backbone_loadpath = str(ckpt_path)
    cls_cfg_template.paths.network_name = "SSLBackbone"
    classification_name_base = "supervised"
    model_name = run_tag

    labels_cfg = getattr(cfg, "labels", {})
    classification_label_key = str(
        getattr(labels_cfg, "activity_classification", getattr(cls_cfg_template.train, "label_column", "activity_load"))
    )
    cls_cfg_template.train.label_column = classification_label_key
    classification_fallbacks = _coerce_label_list(
        getattr(labels_cfg, "activity_classification_fallbacks", ("activity_label", "label_name")),
        ("activity_label", "label_name"),
    )
    viz_cfg = getattr(cfg, "viz", {})
    viz_label_key = str(
        getattr(
            viz_cfg,
            "activity_label_key",
            getattr(labels_cfg, "activity_visualization", "activity_atomic"),
        )
    )
    viz_label_fallbacks = _coerce_label_list(
        getattr(labels_cfg, "activity_visualization_fallbacks", getattr(viz_cfg, "activity_label_fallbacks", None)),
        classification_fallbacks,
    )

    embedding_cfg = getattr(cfg, "embeddings", {})
    embedding_key = str(getattr(embedding_cfg, "key", "embedding"))

    print(f"Using device: {device}")
    cfg_path_override = getattr(getattr(cfg, "paths", {}), "config", None)
    contrastive_model, contrastive_cfg, ckpt = load_contrastive_checkpoint(
        ckpt_path,
        device,
        config_path=cfg_path_override,
    )
    embed_wrapper = ContrastiveEmbeddingWrapper(contrastive_model, embedding_key=embedding_key).to(device)
    embed_wrapper.eval()

    embed_dim = resolve_contrastive_embedding_dim(contrastive_model)
    if embedding_key == "pooled":
        embed_dim = int(getattr(getattr(contrastive_model, "backbone", None), "embed_dim", embed_dim))

    epoch = ckpt.get("epoch", "unknown")
    print(f"Contrastive checkpoint epoch: {epoch}")

    # --- Fold loop ---
    all_fold_summaries: list[dict] = []

    cross_location_cv = bool(getattr(cfg, "cross_location", False))
    if cross_location_cv:
        _loc_col = str(getattr(eval_cv_cfg, "location_column", "location") if eval_cv_cfg else "location")
        _csv_df = pd.read_csv(cls_cfg_template.file_list)
        _fold_locations = sorted(_csv_df[_loc_col].unique().tolist())
        _n_iters = len(_fold_locations)
        print(f"Cross-location CV: {_n_iters} locations = {_fold_locations}")
    else:
        _fold_locations = None
        _n_iters = n_folds

    run_dir = output_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    for _fold_idx in range(_n_iters):
        set_seed(seed)
        if cross_location_cv:
            test_location = str(_fold_locations[_fold_idx])
            test_fold = None
            fold_label = f"fold_{test_location}"
        else:
            test_fold = _fold_idx
            test_location = None
            fold_label = f"fold{test_fold}"

        print(f"\n--- {fold_label} ({_fold_idx + 1}/{_n_iters}) ---")
        cls_cfg = copy.deepcopy(cls_cfg_template)
        if not getattr(cls_cfg, "cross_validation", None):
            cls_cfg.cross_validation = OmegaConf.create({})
        cls_cfg.cross_validation.enabled = True
        if cross_location_cv:
            cls_cfg.cross_validation.test_location = test_location
        else:
            cls_cfg.cross_validation.test_fold = test_fold

        classification_name = classification_name_base
        if fold_label not in classification_name_base:
            classification_name = f"{classification_name_base}-{fold_label}"

        data_train, data_test = build_dataloaders(cls_cfg, test_fold=test_fold)
        train_loader = build_eval_loader(data_train.dataset, cls_cfg.train.batch_size, cls_cfg.train.num_workers)
        test_loader = build_eval_loader(data_test.dataset, cls_cfg.train.batch_size, cls_cfg.train.num_workers)
        eval_loaders = {"train": train_loader, "test": test_loader}

        feature_packs = {}
        for split_name, loader in eval_loaders.items():
            print(f"Extracting contrastive embeddings for '{split_name}' split...")
            feature_packs[split_name] = extract_features(
                embed_wrapper,
                loader,
                device,
                (embedding_key,),
                split_name,
            )

        train_pack = feature_packs["train"]
        test_pack = feature_packs["test"]

        metrics_rows: list[dict] = []

        if not is_regression and bool(getattr(viz_cfg, "enabled", True)):
            max_samples = getattr(viz_cfg, "max_samples", None)
            if max_samples is not None:
                max_samples = int(max_samples)
            run_feature_visualizations(
                train_pack,
                test_pack,
                embedding_key=embedding_key,
                activity_label_key=viz_label_key,
                activity_label_fallbacks=viz_label_fallbacks,
                method=str(getattr(viz_cfg, "method", "tsne")),
                perplexity=float(getattr(viz_cfg, "perplexity", 30.0)),
                max_samples=max_samples,
                seed=seed,
                output_dir=run_dir,
                run_tag=fold_label,
                id_embedding_key=embedding_key,
                activity_embedding_key=embedding_key,
            )
        else:
            print("Skipping visualization step.")

        knn_cfg = getattr(cfg, "knn", {})
        if not is_regression and bool(getattr(knn_cfg, "enabled", True)):
            whiten_cfg = getattr(knn_cfg, "whiten", None)
            track_key = str(getattr(knn_cfg, "track_key", "global_id"))
            knn_k = int(getattr(knn_cfg, "k", 20))
            knn_normalize = bool(getattr(knn_cfg, "normalize", True))

            motion_filter_cfg = getattr(knn_cfg, "motion_filter", {})
            exclude_similar_motion = bool(getattr(knn_cfg, "exclude_similar_motion", False))
            speed_tolerance = float(getattr(motion_filter_cfg, "speed_tolerance", 1.5))
            direction_min_cosine = float(getattr(motion_filter_cfg, "direction_min_cosine", 0.866))
            speed_feature_names = list(getattr(motion_filter_cfg, "speed_features", ["v_x", "v_y"]))
            zero_neighbors_fallback = str(getattr(motion_filter_cfg, "zero_neighbors_fallback", "use_all"))
            if exclude_similar_motion:
                print(
                    f"Similar-motion exclusion enabled for activity KNN "
                    f"(speed_tolerance: {speed_tolerance}, direction_min_cosine: {direction_min_cosine}, "
                    f"features: {speed_feature_names}, fallback: {zero_neighbors_fallback})"
                )

            activity_task = KNNTask(
                name="activity",
                embedding_key=embedding_key,
                label_key=classification_label_key,
                label_fallbacks=classification_fallbacks,
            )
            activity_knn_results = run_knn_dual_exclusion(
                train_pack,
                test_pack,
                task=activity_task,
                k=knn_k,
                normalize=knn_normalize,
                whiten=whiten_cfg,
                track_key=track_key,
                exclude_similar_motion=exclude_similar_motion,
                speed_tolerance=speed_tolerance,
                direction_min_cosine=direction_min_cosine,
                speed_feature_names=speed_feature_names,
                zero_neighbors_fallback=zero_neighbors_fallback,
            )
            include_metrics = activity_knn_results.get("include_same_track", {})
            bal_acc = include_metrics.get("balanced_acc")
            if bal_acc is not None:
                metrics_rows.append({"metric": "knn/test/balanced_acc", "value": _coerce_log_value(bal_acc)})
            if bool(getattr(cfg, "save_predictions", True)):
                _save_activity_predictions(
                    train_pack,
                    test_pack,
                    activity_task,
                    k=knn_k,
                    normalize=knn_normalize,
                    whiten_cfg=whiten_cfg,
                    run_dir=run_dir / "knn",
                    fold_name=fold_label,
                    model_name=model_name,
                    classification_name=classification_name,
                    test_fold=test_fold,
                )

        else:
            print("Skipping activity KNN.")

        lp_cfg = getattr(cfg, "linear_probe", {})
        lp_specs = _resolve_linear_probe_specs(lp_cfg)
        if is_regression:
            lp_specs = [s for s in lp_specs if s["train_backbone"]]
        print(f"\n[Embedding dim] {embed_dim}\n")
        if lp_specs:
            multi_variant = len(lp_specs) > 1
            for lp_spec in lp_specs:
                print(
                    f"{lp_spec['name']}: train_backbone={lp_spec['train_backbone']}, "
                    f"hidden_dims={lp_spec['hidden_dims']}, dropout={lp_spec['dropout']}"
                )
                probe_output_dir = run_dir / lp_spec["slug"] if multi_variant else run_dir
                probe_test_stats = _run_supervised_linear_probe(
                    contrastive_model,
                    cls_cfg,
                    data_train,
                    data_test,
                    device,
                    int(embed_dim),
                    probe_output_dir=probe_output_dir,
                    fold_name=fold_label,
                    probe_name=lp_spec["slug"],
                    train_backbone=lp_spec["train_backbone"],
                    embedding_key=embedding_key,
                    hidden_dims=lp_spec["hidden_dims"],
                    dropout=lp_spec["dropout"],
                    test_fold=test_fold,
                )
                slug = lp_spec["slug"]
                if probe_test_stats is not None:
                    if is_regression:
                        for k, v in _epoch_stats_to_dict(probe_test_stats).items():
                            metrics_rows.append({"metric": k, "value": _coerce_log_value(v)})
                    else:
                        bal_acc = getattr(probe_test_stats, "balanced_acc", None)
                        if bal_acc is not None:
                            metrics_rows.append({
                                "metric": f"{slug}/test/balanced_acc",
                                "value": _coerce_log_value(bal_acc),
                            })
        else:
            print("Skipping linear probe training.")

        print(f"Outputs saved under: {run_dir}")

        all_fold_summaries.append(_extract_fold_summary(metrics_rows))

    if is_regression:
        summary_keys = ("mae_speed", "mae_angle", "mae_radial", "mae_lateral")
    else:
        summary_keys = (
            "full_finetune_nonlinear_head/test/balanced_acc",
            "frozen_backbone_nonlinear_head/test/balanced_acc",
            "knn/test/balanced_acc",
            "linear_probe/test/balanced_acc",
        )
    _summarize_folds(
        all_fold_summaries,
        tag=run_tag,
        keys_to_print=summary_keys,
        output_dir=run_dir,
    )

    return all_fold_summaries


@hydra.main(version_base="1.2", config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg.method_name = "contrastive"
    cfg = apply_ssl_eval_constants(cfg)
    orig_cwd = Path(get_original_cwd())
    device = _resolve_device(getattr(cfg, "device", None))
    seed = int(getattr(cfg, "seed", 0))
    n_folds = int(getattr(getattr(cfg, "cross_validation", {}), "n_folds", 3))

    paths_cfg = getattr(cfg, "paths", {})
    output_dir = _resolve_path(paths_cfg.output_dir, orig_cwd)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_specs = _normalize_ckpt_specs(paths_cfg, orig_cwd)
    if not ckpt_specs:
        raise RuntimeError("No contrastive checkpoints configured under 'paths'.")

    print(f"Using device: {device}")

    exp_name = getattr(paths_cfg, "exp_name", None)

    def _iter_ckpt_specs():
        total = len(ckpt_specs)
        for idx, spec in enumerate(ckpt_specs, start=1):
            ckpt_path = spec["path"]
            folder_override = spec.get("folder_name")
            base_folder = folder_override if folder_override not in (None, "") else getattr(paths_cfg, "folder_name", None)
            run_tag = _resolve_run_tag(base_folder, exp_name, ckpt_path)
            spec_name = spec.get("name")
            if spec_name not in (None, "") and folder_override in (None, ""):
                run_tag = f"{run_tag}-{spec_name}"
            print(f"\n=== [{idx}/{total}] Tuning contrastive checkpoint: {ckpt_path} ===")
            yield ckpt_path, run_tag

    for ckpt_path, run_tag in _iter_ckpt_specs():
        _run_contrastive_tune_for_ckpt(
            cfg, ckpt_path, output_dir, run_tag,
            device, seed, n_folds=n_folds,
        )


if __name__ == "__main__":
    main()
