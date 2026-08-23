from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def _coerce_config(cfg: DictConfig | dict) -> DictConfig:
    if OmegaConf.is_config(cfg):
        return cfg  # type: ignore[return-value]
    return OmegaConf.create(cfg)


def _projection_output_dim(head: nn.Module | None) -> int:
    if head is None:
        raise AttributeError("Contrastive model head is missing; cannot infer embedding dimension.")
    if hasattr(head, "net"):
        for module in reversed(list(head.net.children())):  # type: ignore[attr-defined]
            if isinstance(module, nn.Linear):
                return int(module.out_features)
    raise AttributeError("Contrastive projection head lacks a terminal Linear layer.")


def resolve_contrastive_embedding_dim(model: nn.Module) -> int:
    """
    Infer the output embedding dimension for a single-embedding contrastive model.
    """

    head = getattr(model, "head", None)
    try:
        return _projection_output_dim(head)
    except AttributeError:
        pass

    backbone = getattr(model, "backbone", None)
    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        raise AttributeError("Contrastive model is missing 'embed_dim' metadata.")
    return int(embed_dim)


def load_contrastive_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    config_path: Path | str | None = None,
):
    """
    Load a single-embedding contrastive checkpoint and rebuild the model/config metadata.
    """

    def _resolve_fallback_config_path(path_value: Path | str) -> Path:
        raw = Path(path_value).expanduser()
        if raw.is_absolute():
            return raw.resolve()

        repo_root = Path(__file__).resolve().parents[4]
        candidates = [
            Path.cwd() / raw,
            repo_root / raw,
            ckpt_path.parent / raw,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    # Lazy import to avoid circular imports during module initialization
    from ..ssl_utils.contrastive_builders import build_single_embedding_model

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args_cfg = ckpt.get("args")
    if args_cfg is None:
        if config_path in (None, ""):
            raise RuntimeError(f"Contrastive checkpoint '{ckpt_path}' is missing serialized args.")
        resolved_cfg_path = _resolve_fallback_config_path(config_path)
        if not resolved_cfg_path.exists():
            raise RuntimeError(
                f"Contrastive checkpoint '{ckpt_path}' is missing serialized args and fallback config "
                f"'{resolved_cfg_path}' does not exist."
            )
        print(
            f"[INFO] Contrastive checkpoint '{ckpt_path}' missing serialized args; "
            f"loading configuration from '{resolved_cfg_path}'."
        )
        args_cfg = OmegaConf.load(str(resolved_cfg_path))
    try:
        contrastive_cfg = _coerce_config(args_cfg)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise RuntimeError(f"Failed to reconstruct contrastive config from '{ckpt_path}': {exc}") from exc

    model = build_single_embedding_model(contrastive_cfg)
    state_dict = ckpt.get("model_state")
    if state_dict is None:
        if isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError(f"Contrastive checkpoint '{ckpt_path}' is missing model weights.")
    state_dict = _remap_legacy_pretrain_keys(state_dict, model.state_dict())
    missing = model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys:
        print(f"[WARN] Missing contrastive weights: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"[WARN] Unexpected contrastive weights: {missing.unexpected_keys}")
    model.to(device)
    model.eval()
    return model, contrastive_cfg, ckpt


def _remap_legacy_pretrain_keys(state_dict: dict, target_state: dict) -> dict:
    """Map self_supervised_pretrain backbone keys to ssl_eval model keys."""

    if not any(str(key).startswith("backbone.features.") for key in state_dict):
        return state_dict
    remapped = {}
    target_keys = set(target_state.keys())
    for key, value in state_dict.items():
        new_key = str(key)
        if new_key.startswith("backbone.features."):
            candidate = "backbone.backbone." + new_key[len("backbone.features.") :]
            if candidate in target_keys:
                new_key = candidate
        remapped[new_key] = value
    return remapped


__all__ = ["load_contrastive_checkpoint", "resolve_contrastive_embedding_dim"]
