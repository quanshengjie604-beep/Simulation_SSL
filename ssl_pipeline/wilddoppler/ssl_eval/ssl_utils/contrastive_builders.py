from __future__ import annotations

import torch.nn as nn
from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf

from .model import MobileNetBackbone, MobileViTBackbone, ResNetBackbone
from .single_embedding import SingleEmbeddingModel


def _model_value(args: DictConfig, attr: str, default):
    model_cfg = getattr(args, "model", None)
    if model_cfg is not None:
        if isinstance(model_cfg, DictConfig):
            if attr in model_cfg:
                return model_cfg[attr]
        elif isinstance(model_cfg, Mapping):
            if attr in model_cfg:
                return model_cfg[attr]
        elif hasattr(model_cfg, attr):
            return getattr(model_cfg, attr)
    if isinstance(args, DictConfig):
        if attr in args:
            return args[attr]
    elif isinstance(args, Mapping):
        if attr in args:
            return args[attr]
    elif hasattr(args, attr):
        return getattr(args, attr)
    return default


def _conditioning_cfg(args: DictConfig):
    cond_cfg = getattr(args, "conditioning", None)
    if cond_cfg is None:
        return None
    if OmegaConf.is_config(cond_cfg):
        return OmegaConf.to_container(cond_cfg, resolve=True)
    if isinstance(cond_cfg, Mapping):
        return dict(cond_cfg)
    if hasattr(cond_cfg, "__dict__"):
        return dict(vars(cond_cfg))
    try:
        return dict(cond_cfg)
    except Exception:
        return None


def _build_backbone(args: DictConfig) -> nn.Module:
    model_name = _model_value(args, "model_name", "mobilevit_s")
    stem_channels = _model_value(args, "stem_channels", 32)
    embed_dim = int(_model_value(args, "embed_dim", 512))
    output_stride = int(_model_value(args, "output_stride", 16))
    use_radar_stem = bool(_model_value(args, "use_radar_stem", False))
    conditioning_cfg = _conditioning_cfg(args)

    if model_name.startswith("mobilevit"):
        return MobileViTBackbone(
            model_name=model_name,
            embed_dim=embed_dim,
            stem_channels=stem_channels,
            output_stride=output_stride,
            use_radar_stem=use_radar_stem,
            conditioning_cfg=conditioning_cfg,
        )
    if model_name.startswith("mobilenet"):
        return MobileNetBackbone(
            model_name=model_name,
            embed_dim=embed_dim,
            stem_channels=int(stem_channels),
            use_radar_stem=use_radar_stem,
            conditioning_cfg=conditioning_cfg,
        )
    if model_name.startswith("resnet"):
        return ResNetBackbone(
            model_name=model_name,
            embed_dim=embed_dim,
            stem_channels=int(stem_channels),
            use_radar_stem=use_radar_stem,
            conditioning_cfg=conditioning_cfg,
        )
    raise ValueError(f"Unsupported backbone '{model_name}'.")


def build_backbone(args: DictConfig) -> nn.Module:
    return _build_backbone(args)


def _ensure_config(node):
    if OmegaConf.is_config(node):
        return node
    return OmegaConf.create(node)


def build_single_embedding_model(args: DictConfig) -> SingleEmbeddingModel:
    backbone = _build_backbone(args)
    contrastive_cfg = getattr(args, "contrastive", {})
    if not OmegaConf.is_config(contrastive_cfg):
        contrastive_cfg = OmegaConf.create(contrastive_cfg)

    # Keep a compatibility fallback for older checkpoints/configs that stored
    # the embedding-head settings under `ranking`.
    legacy_embedding_cfg = getattr(args, "ranking", {})
    if not OmegaConf.is_config(legacy_embedding_cfg):
        legacy_embedding_cfg = OmegaConf.create(legacy_embedding_cfg)

    head_raw = getattr(
        contrastive_cfg,
        "embedding_head",
        getattr(legacy_embedding_cfg, "head", getattr(legacy_embedding_cfg, "embedding_head", {})),
    )
    head_cfg = OmegaConf.to_container(_ensure_config(head_raw), resolve=True)
    if not head_cfg:
        proj_dim = _model_value(args, "proj_dim", None)
        proj_hidden_dim = _model_value(args, "proj_hidden_dim", None)
        if proj_dim is not None or proj_hidden_dim is not None:
            head_cfg = {}
            if proj_dim is not None:
                head_cfg["out_dim"] = int(proj_dim)
            if proj_hidden_dim is not None:
                head_cfg["hidden_dim"] = int(proj_hidden_dim)
    pool_heads = int(getattr(contrastive_cfg, "pool_num_heads", getattr(legacy_embedding_cfg, "pool_num_heads", 4)))
    pool_dropout = float(
        getattr(contrastive_cfg, "pool_dropout", getattr(legacy_embedding_cfg, "pool_dropout", 0.0))
    )
    default_pool_type = "mean" if _model_value(args, "proj_dim", None) is not None else "attention_cls"
    pool_type = str(getattr(contrastive_cfg, "pool_type", default_pool_type))
    return SingleEmbeddingModel(
        backbone=backbone,
        pool_num_heads=pool_heads,
        pool_dropout=pool_dropout,
        head_cfg=head_cfg,
        pool_type=pool_type,
    )


__all__ = ["build_backbone", "build_single_embedding_model"]
