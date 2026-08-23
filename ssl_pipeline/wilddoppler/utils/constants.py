"""Hard-coded defaults for fields that were stripped from the Hydra YAML configs.

The single global YAML at `conf/train.yaml` keeps only the small set of fields
users actually tune per run (task_name, model_name, learning rates,
epochs, paths, etc.). Everything else lives here and is injected into `cfg` at
runtime via the `apply_*` helpers below, so the rest of the codebase keeps
reading `cfg.train.batch_size`, `cfg.paths.exp_name`, etc. exactly as before.

To run a new task, add an entry to `TASK_DEFAULTS` keyed by `task_name`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from omegaconf import DictConfig, OmegaConf


SEED = 1


# ---------------------------------------------------------------------------
# Per-task values: file paths, label/target columns. Looked up by `task_name`.
# ---------------------------------------------------------------------------
TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "SingleHand": {
        "file_list": "data/fold_splits/singlehand_constrained_1s_fold3.csv",
        "train": {
            "label_column": "activity_load",
        },
    },
    "MotionState": {
        "file_list": "data/fold_splits/motionstate_1s_fold3.csv",
        "train": {
            "label_column": "activity_load",
        },
    },
    "VelocityRegression": {
        "file_list": "data/fold_splits/velocity_regression_1s_folds3.csv",
        "train": {
            "batch_size": 64,
        },
    },
}


# ---------------------------------------------------------------------------
# Regression tasks. All other entries in TASK_DEFAULTS are classification.
# Regression always uses the unit_polar formulation (3-dim head: [a, b, s]).
# ---------------------------------------------------------------------------
REGRESSION_TASKS: set[str] = {"VelocityRegression"}
UNIT_POLAR_NUM_OUTPUTS: int = 3


def is_regression_task(task_name: str) -> bool:
    return str(task_name) in REGRESSION_TASKS


# ---------------------------------------------------------------------------
# Classification head sizes per task.
# ---------------------------------------------------------------------------
TASK_NUM_CLASSES: Dict[str, int] = {
    "MotionState": 3,
    "SingleHand":  2,
}


# ---------------------------------------------------------------------------
# Per-(task, method, cross_location_cv) defaults for `train.learning_rate`
# and `train.epochs`. User-provided YAML/CLI values always win; null values
# in the YAML are dropped so these defaults take effect (see _drop_null_keys).
#
# Tasks not listed (e.g. SingleHand, or any model variant of VelocityRegression)
# fall back to the cross-subject (cross_location_cv=False) entry of the same
# method within the same category — see HYPERPARAM_FALLBACK_TASK and
# get_hyperparam_defaults below.
# ---------------------------------------------------------------------------
_DEFAULT_LR_EPOCHS: Dict[str, Any] = {"learning_rate": 1.0e-4, "epochs": 3}
HYPERPARAM_DEFAULTS: Dict[Tuple[str, str, bool], Dict[str, Any]] = {
    # Classification — MotionState
    ("MotionState",         "supervised",  False): {"learning_rate": 1.0e-3, "epochs": 100},
    ("MotionState",         "reconstruction", False): {"learning_rate": 5.0e-6, "epochs": 100},
    ("MotionState",         "contrastive", False): {"learning_rate": 5.0e-5, "epochs":  30},
    ("MotionState",         "supervised",  True):  {"learning_rate": 5.0e-3, "epochs": 100},
    ("MotionState",         "reconstruction", True):  {"learning_rate": 9.0e-5, "epochs": 100},
    ("MotionState",         "contrastive", True):  {"learning_rate": 9.0e-5, "epochs": 100},
    # Regression — VelocityRegression
    ("VelocityRegression",  "supervised",  False): {"learning_rate": 8.0e-4, "epochs": 150},
    ("VelocityRegression",  "reconstruction", False): {"learning_rate": 9.0e-4, "epochs": 150},
    ("VelocityRegression",  "contrastive", False): {"learning_rate": 9.0e-4, "epochs": 150},
    ("VelocityRegression",  "supervised",  True):  {"learning_rate": 5.0e-4, "epochs": 100},
    ("VelocityRegression",  "reconstruction", True):  {"learning_rate": 1.0e-3, "epochs":  50},
    ("VelocityRegression",  "contrastive", True):  {"learning_rate": 6.0e-4, "epochs": 150},
}

# Per-category fallback task. Unspecified (task, method, split) combos resolve
# to the cross-subject (False) entry of this task for the same method.
HYPERPARAM_FALLBACK_TASK: Dict[str, str] = {
    "classification": "MotionState",
    "regression":     "VelocityRegression",
}


def get_hyperparam_defaults(task_name: str, method_name: str, cross_location_cv: bool) -> Dict[str, Any]:
    """Resolve (learning_rate, epochs) for a (task, method, split) combo.

    Falls back to the cross-subject entry of HYPERPARAM_FALLBACK_TASK[category]
    for the same method when the exact combo isn't specified.
    """
    key = (str(task_name), str(method_name), bool(cross_location_cv))
    if key in HYPERPARAM_DEFAULTS:
        return dict(HYPERPARAM_DEFAULTS[key])
    category = "regression" if is_regression_task(task_name) else "classification"
    fallback_task = HYPERPARAM_FALLBACK_TASK.get(category)
    if fallback_task is not None:
        fb = HYPERPARAM_DEFAULTS.get((fallback_task, str(method_name), False))
        if fb is not None:
            return dict(fb)
    return dict(_DEFAULT_LR_EPOCHS)


# ---------------------------------------------------------------------------
# Cross-validation derivation rules.
# ---------------------------------------------------------------------------
def n_folds_for(cross_location_cv: bool) -> int:
    """4 folds when iterating over locations, 3 otherwise."""
    return 4 if bool(cross_location_cv) else 3


# ---------------------------------------------------------------------------
# SSL checkpoint filename derivation. The `ssl_ckpts_dir` directory holds:
#   contrastive_mobilenet_v2_epoch300.pt, contrastive_resnet18_epoch300.pt,
#   reconstruction_mobilenet_v2_epoch300.pt, reconstruction_resnet18_epoch300.pt
# ---------------------------------------------------------------------------
def ssl_ckpt_filename(method_name: str, model_name: str) -> str:
    return f"{method_name}_pretraining_{model_name}_epoch300.pt"


# ---------------------------------------------------------------------------
# Supervised constants — applied to both supervised configs and the
# classification_config that SSL-eval scripts merge into.
# ---------------------------------------------------------------------------
SUPERVISED_CONSTANTS: Dict[str, Any] = {
    "transforms": {
        "sample_length": 1,
        "bins_per_second": 90,
        "win_size": 90,
        "resize_doppler": 256,
        "auto_calculate_stats": True,
    },
    "train": {
        # Placeholders for eval-only flows; training YAMLs override these.
        "epochs": 1,
        "learning_rate": 1.0e-4,
        "batch_size": 60,
        "num_workers": 8,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "use_amp": True,
        "save_every": 10,
        "dataloader_timeout": 120,
        "lr_scheduler": {
            "name": "cosine",
            "min_lr": 0.0,
            "t_0": 30,
            "t_mult": 2,
        },
        "sampling": {
            "strategy": "balanced_batch",
            "weight_power": 0.5,
            "replacement": True,
        },
        "unit_polar_eps": 1.0e-8,
        "unit_polar_lambda_s": 1.0,
        "unit_polar_lambda_d": 1.0,
        "label_names": [],
        "class_mapping": {},
    },
    "model": {
        "embed_dim": 256,
        "head_hidden_dims": [1024],
        "head_dropout": 0.0,
        "pretrained": False,
    },
    "data": {
        "file_template": "{file_name}_track_{global_id}.npz",
        "subject_column": "global_id",
        "data_key": "uD",
        "time_column": "t_start_rel",
    },
    "cross_validation": {
        "enabled": True,
        "test_location": None,
        "test_fold": None,
        "location_column": "location",
    },
}


# ---------------------------------------------------------------------------
# SSL eval constants — sections that only exist in the SSL eval flow.
# Linear probe, KNN, visualization, label-key fallbacks, conditioning, etc.
# ---------------------------------------------------------------------------
SSL_EVAL_CONSTANTS: Dict[str, Any] = {
    "train": {
        # Eval-only training extras layered on top of the supervised train block.
        "backbone_learning_rate": None,
        "head_learning_rate": None,
        "lr_scheduler": {
            "name": "cosine",
            "t_max": 30,  # synced to train.epochs at runtime
            "min_lr": 0.0,
        },
    },
    "embeddings": {
        "key": "embedding",
    },
    "labels": {
        "activity_classification": "activity_load",
        "activity_classification_fallbacks": ["activity_load", "activity_label", "label_name"],
        "activity_visualization": "activity_load",
        "activity_visualization_fallbacks": ["activity_atomic", "activity_label", "activity_load"],
        "identity": "global_id",
        "identity_fallbacks": ["global_id"],
    },
    "conditioning": {
        "enabled": False,
        "feature_columns": ["x_mean", "y_mean", "vx_mean", "vy_mean"],
        "fusion": "film",
        "fusion_hidden_dim": 512,
        "fusion_dropout": 0.5,
    },
    "viz": {
        "enabled": False,
        "method": "umap",
        "perplexity": 30.0,
        "max_samples": 1500,
    },
    "knn": {
        "enabled": True,
        "k": 20,
        "normalize": True,
        "run_identity": False,
        "exclude_similar_motion": False,
        "motion_filter": {
            "speed_tolerance": 1.5,
            "direction_min_cosine": 0.866,
            "speed_features": ["v_x", "v_y"],
            "zero_neighbors_fallback": "use_all",
        },
        "test_center_by_id": {
            "enabled": False,
            "id_key": "global_id",
            "id_fallbacks": ["global_id"],
            "embedding_key": None,
        },
        "whiten": {
            "enabled": False,
            "method": "pca",
            "eps": 1.0e-5,
        },
    },
    "linear_probe": {
        "enabled": True,
        "train_backbone": True,
        "hidden_dims": [1024],
        "dropout": 0.0,
        "variants": {
            "full_finetune_nonlinear_head": {
                "enabled": True,
            },
            "frozen_backbone_nonlinear_head": {
                "enabled": True,
                "train_backbone": False,
            },
            "linear_probe": {
                "enabled": True,
                "train_backbone": False,
                "hidden_dims": [],
            },
        },
    },
    "save_predictions": True,
    "cross_validation": {
        "test_location": None,
        "location_column": "location",
    },
}


def _to_omega(d: Dict[str, Any]) -> DictConfig:
    return OmegaConf.create(d)


_DROP_IF_NULL_KEYS = ("file_list", "data_dir")
_DROP_IF_NULL_TRAIN_KEYS = ("learning_rate", "epochs")


def _drop_null_keys(cfg: DictConfig) -> None:
    """Pop keys whose value is None/'' so a constant default can win.

    The YAMLs set `file_list: null` (or leave `train.learning_rate` blank) to
    mean "use the keyed default from TASK_DEFAULTS / HYPERPARAM_DEFAULTS".
    Without this, OmegaConf.merge would let the null override the constant.
    """
    for key in _DROP_IF_NULL_KEYS:
        if key in cfg and cfg[key] in (None, ""):
            del cfg[key]
    train_cfg = getattr(cfg, "train", None)
    if train_cfg is not None:
        for key in _DROP_IF_NULL_TRAIN_KEYS:
            if key in train_cfg and train_cfg[key] in (None, ""):
                del train_cfg[key]


def _resolved_method_name(cfg: DictConfig) -> str:
    return str(getattr(cfg, "method_name", "") or "supervised")


def _resolved_model_name(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "model_name", default="mobilenet_v2") or "mobilenet_v2")


def _inject_paths_block(cfg: DictConfig, method_name: str) -> None:
    """Populate cfg.paths.{output_dir, exp_name} from top-level cfg fields and
    method_name. Doesn't overwrite values the user has already set explicitly."""
    OmegaConf.set_struct(cfg, False)
    if getattr(cfg, "paths", None) is None:
        cfg.paths = OmegaConf.create({})

    output_dir = getattr(cfg, "output_dir", None)
    if output_dir not in (None, "") and getattr(cfg.paths, "output_dir", None) in (None, ""):
        cfg.paths.output_dir = str(output_dir)

    task_name = str(getattr(cfg, "task_name", "") or "")
    model_name = _resolved_model_name(cfg)
    cross_location_cv = bool(getattr(cfg, "cross_location", False))
    timestamp = datetime.now().strftime("%b%d_%H%M")
    exp_name = build_exp_name(task_name, method_name, model_name, timestamp, cross_location_cv)

    if getattr(cfg.paths, "exp_name", None) in (None, ""):
        cfg.paths.exp_name = exp_name


def _inject_ssl_ckpt(cfg: DictConfig, method_name: str) -> None:
    """When cfg.paths.ckpt is unset, derive it from ssl_ckpts_dir +
    method_name + model_name. A user-provided cfg.paths.ckpt wins."""
    OmegaConf.set_struct(cfg, False)
    if getattr(cfg, "paths", None) is None:
        cfg.paths = OmegaConf.create({})
    existing = getattr(cfg.paths, "ckpt", None)
    if existing not in (None, ""):
        return
    ssl_ckpts_dir = getattr(cfg, "ssl_ckpts_dir", None)
    if ssl_ckpts_dir in (None, ""):
        return
    model_name = _resolved_model_name(cfg)
    cfg.paths.ckpt = str(Path(str(ssl_ckpts_dir)) / ssl_ckpt_filename(method_name, model_name))


def _merge_hyperparam_defaults(
    base: DictConfig,
    task_name: str,
    method_name: str,
    cross_location_cv: bool,
) -> DictConfig:
    overrides = get_hyperparam_defaults(task_name, method_name, cross_location_cv)
    if not overrides:
        return base
    return OmegaConf.merge(base, _to_omega({"train": dict(overrides)}))


def _merge_num_classes(base: DictConfig, task_name: str, is_regression: bool) -> DictConfig:
    if is_regression:
        return base
    num_classes = TASK_NUM_CLASSES.get(task_name)
    if num_classes is None:
        return base
    return OmegaConf.merge(base, _to_omega({"train": {"num_classes": int(num_classes)}}))


def apply_supervised_constants(cfg: DictConfig) -> DictConfig:
    """Inject supervised constants into a stripped supervised cfg.

    Order (later wins): SUPERVISED_CONSTANTS -> TASK_DEFAULTS[task_name] ->
    HYPERPARAM_DEFAULTS[(task_name, method_name, cross_location_cv)] ->
    TASK_NUM_CLASSES[task_name] -> cfg. Also derives cross_validation.n_folds
    from cross_location_cv, and (for regression tasks) sets train.num_outputs
    to the unit_polar head size.
    """
    OmegaConf.set_struct(cfg, False)
    _drop_null_keys(cfg)

    task_name = str(getattr(cfg, "task_name", "") or "")
    method_name = _resolved_method_name(cfg)
    is_regression = is_regression_task(task_name)

    cross_location_cv = bool(getattr(cfg, "cross_location", False))

    base = OmegaConf.create({"seed": SEED})
    base = OmegaConf.merge(base, _to_omega(SUPERVISED_CONSTANTS))

    task_specific = TASK_DEFAULTS.get(task_name, {})
    if task_specific:
        base = OmegaConf.merge(base, _to_omega(task_specific))

    base = _merge_hyperparam_defaults(base, task_name, method_name, cross_location_cv)
    base = _merge_num_classes(base, task_name, is_regression)

    base.cross_validation.n_folds = n_folds_for(cross_location_cv)

    if is_regression:
        base.train.num_outputs = UNIT_POLAR_NUM_OUTPUTS

    merged = OmegaConf.merge(base, cfg)
    OmegaConf.set_struct(merged, False)

    _inject_paths_block(merged, method_name)
    return merged


def apply_ssl_eval_constants(cfg: DictConfig) -> DictConfig:
    """Inject SSL-eval-specific constants (linear_probe, knn, viz, ...) and
    the supervised constants the SSL-eval scripts also depend on (transforms,
    train block, model defaults, data, cross_validation).

    Also derives `paths.ckpt` from `ssl_ckpts_dir` + method_name + model_name
    when `cfg.paths.ckpt` is unset.
    """
    OmegaConf.set_struct(cfg, False)
    _drop_null_keys(cfg)

    task_name = str(getattr(cfg, "task_name", "") or "")
    method_name = _resolved_method_name(cfg)
    is_regression = is_regression_task(task_name)

    cross_location_cv = bool(getattr(cfg, "cross_location", False))

    base = OmegaConf.create({"seed": SEED})
    # Layer in supervised constants too — SSL eval needs transforms/model/data
    # blocks that used to come from the separate classification_config.
    base = OmegaConf.merge(base, _to_omega(SUPERVISED_CONSTANTS))
    base = OmegaConf.merge(base, _to_omega(SSL_EVAL_CONSTANTS))

    task_specific = TASK_DEFAULTS.get(task_name, {})
    if task_specific:
        base = OmegaConf.merge(base, _to_omega(task_specific))

    base = _merge_hyperparam_defaults(base, task_name, method_name, cross_location_cv)
    base = _merge_num_classes(base, task_name, is_regression)

    base.cross_validation.n_folds = n_folds_for(cross_location_cv)

    if is_regression:
        base.train.num_outputs = UNIT_POLAR_NUM_OUTPUTS

    merged = OmegaConf.merge(base, cfg)
    OmegaConf.set_struct(merged, False)

    _inject_paths_block(merged, method_name)
    _inject_ssl_ckpt(merged, method_name)
    return merged


def build_exp_name(
    task_name: str,
    mode: str,
    model_name: str,
    timestamp: str,
    cross_location_cv: bool = False,
) -> str:
    """Consistent experiment-name format used by all entry points."""
    cv_tag = "_cross-loc" if cross_location_cv else ""
    return f"{task_name}_{mode}_{model_name}{cv_tag}_{timestamp}"


# ---------------------------------------------------------------------------
# Default per-fold checkpoint layout for eval.py.
# When `eval_checkpoint_dir` is left blank, eval.py falls back to this lookup
# using (method_name, task_name, cross_location, model_name, and — for
# SSL methods — linear_probe.train_backbone). Paths are resolved relative to
# `cfg.ckpts_dir` (default: `data/checkpoints`). SSL methods include an extra train_backbone level
# (full_finetune | frozen_backbone); supervised has no such inner level.
# ---------------------------------------------------------------------------
DEFAULT_CKPTS: Dict[str, Any] = {
    "supervised": {
        "MotionState": {
            "cross-subj": {
                "mobilenet_v2": "supervised_cross-subj_motionstate",
                "resnet18":     "supervised_cross-subj_motionstate_resnet18",
            },
            "cross-loc": {
                "mobilenet_v2": "supervised_cross-loc_motionstate",
            },
        },
        "SingleHand": {
            "cross-subj": {
                "mobilenet_v2": "supervised_cross-subj_singlehand",
            },
        },
        "VelocityRegression": {
            "cross-subj": {
                "mobilenet_v2": "supervised_cross-subj_velocityregression",
                "resnet18":     "supervised_cross-subj_velocityregression_resnet18",
            },
            "cross-loc": {
                "mobilenet_v2": "supervised_cross-loc_velocityregression",
            },
        },
    },
    "reconstruction": {
        "MotionState": {
            "cross-subj": {
                "mobilenet_v2": {
                    "full_finetune":   "reconstruction_cross-subj_motionstate/full_finetune",
                    "frozen_backbone": "reconstruction_cross-subj_motionstate/frozen_backbone",
                },
                "resnet18": {
                    "full_finetune":   "reconstruction_cross-subj_motionstate_resnet18/full_finetune",
                    "frozen_backbone": "reconstruction_cross-subj_motionstate_resnet18/frozen_backbone",
                },
            },
            "cross-loc": {
                "mobilenet_v2": {
                    "full_finetune":   "reconstruction_cross-loc_motionstate/full_finetune",
                    "frozen_backbone": "reconstruction_cross-loc_motionstate/frozen_backbone",
                },
            },
        },
        "SingleHand": {
            "cross-subj": {
                "mobilenet_v2": {
                    "full_finetune":   "reconstruction_cross-subj_singlehand/full_finetune",
                    "frozen_backbone": "reconstruction_cross-subj_singlehand/frozen_backbone",
                },
            },
        },
        "VelocityRegression": {
            "cross-subj": {
                "mobilenet_v2": {
                    "full_finetune":   "reconstruction_cross-subj_velocityregression/full_finetune",
                },
                "resnet18": {
                    "full_finetune":   "reconstruction_cross-subj_velocityregression_resnet18/full_finetune",
                },
            },
            "cross-loc": {
                "mobilenet_v2": {
                    "full_finetune":   "reconstruction_cross-loc_velocityregression/full_finetune",
                },
            },
        },
    },
    "contrastive": {
        "MotionState": {
            "cross-subj": {
                "mobilenet_v2": {
                    "full_finetune":   "contrastive_cross-subj_motionstate/full_finetune",
                    "frozen_backbone": "contrastive_cross-subj_motionstate/frozen_backbone",
                },
                "resnet18": {
                    "full_finetune":   "contrastive_cross-subj_motionstate_resnet18/full_finetune",
                    "frozen_backbone": "contrastive_cross-subj_motionstate_resnet18/frozen_backbone",
                },
            },
            "cross-loc": {
                "mobilenet_v2": {
                    "full_finetune":   "contrastive_cross-loc_motionstate/full_finetune",
                    "frozen_backbone": "contrastive_cross-loc_motionstate/frozen_backbone",
                },
            },
        },
        "SingleHand": {
            "cross-subj": {
                "mobilenet_v2": {
                    "full_finetune":   "contrastive_cross-subj_singlehand/full_finetune",
                    "frozen_backbone": "contrastive_cross-subj_singlehand/frozen_backbone",
                },
            },
        },
        "VelocityRegression": {
            "cross-subj": {
                "mobilenet_v2": {
                    "full_finetune":   "contrastive_cross-subj_velocityregression/full_finetune",
                },
                "resnet18": {
                    "full_finetune":   "contrastive_cross-subj_velocityregression_resnet18/full_finetune",
                },
            },
            "cross-loc": {
                "mobilenet_v2": {
                    "full_finetune":   "contrastive_cross-loc_velocityregression/full_finetune",
                },
            },
        },
    },
}


def resolve_default_checkpoint_dir(cfg: DictConfig) -> str:
    base_dir = OmegaConf.select(cfg, "ckpts_dir", default="data/checkpoints")
    if base_dir in (None, ""):
        base_dir = "data/checkpoints"

    method = str(OmegaConf.select(cfg, "method_name", default="") or "supervised").lower()
    task = str(OmegaConf.select(cfg, "task_name", default="") or "")
    model_name = _resolved_model_name(cfg)
    cross_location_cv = bool(
        OmegaConf.select(cfg, "cross_location", default=False)
    )
    cv_key = "cross-loc" if cross_location_cv else "cross-subj"

    parts = [method, task, cv_key, model_name]
    if method != "supervised":
        train_backbone = bool(
            OmegaConf.select(cfg, "linear_probe.train_backbone", default=True)
        )
        parts.append("full_finetune" if train_backbone else "frozen_backbone")

    combo = (
        f"method={method}, task={task}, split={cv_key}, model={model_name}"
        + (f", variant={parts[-1]}" if method != "supervised" else "")
    )

    node: Any = DEFAULT_CKPTS
    for key in parts:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(
                f"This configuration ({combo}) is not supported in this release. "
                f"No checkpoint is shipped under data/checkpoints for it."
            )
        node = node[key]
    if not isinstance(node, str):
        raise ValueError(
            f"This configuration ({combo}) is not supported in this release. "
            f"No checkpoint is shipped under data/checkpoints for it."
        )
    return str(Path(str(base_dir)) / node)
