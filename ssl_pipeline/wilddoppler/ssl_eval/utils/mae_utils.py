from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from ..ssl_utils.mae import BackboneMAENetwork, RadarMAENetwork


@dataclass
class MAEInputTransform:
    swapped_axes: bool
    resized: bool
    pre_resize_size: Tuple[int, int]


def _coerce_config(node) -> DictConfig:
    if OmegaConf.is_config(node):
        return node  # type: ignore[return-value]
    return OmegaConf.create(node)


def _build_mae_model(args: DictConfig | dict) -> nn.Module:
    args_cfg = _coerce_config(args)
    mae_cfg = _coerce_config(getattr(args_cfg, "mae", {}))
    model_cfg = _coerce_config(getattr(args_cfg, "model", {}))
    use_backbone = bool(getattr(mae_cfg, "use_backbone", False))
    if use_backbone:
        return BackboneMAENetwork(args_cfg, model_cfg, mae_cfg)
    return RadarMAENetwork(model_cfg, mae_cfg)


def load_mae_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    config_path: Path | str | None = None,
):
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

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args_cfg = ckpt.get("args")
    if args_cfg is None:
        if config_path in (None, ""):
            raise RuntimeError(f"MAE checkpoint '{ckpt_path}' is missing serialized args.")
        resolved_cfg_path = _resolve_fallback_config_path(config_path)
        if not resolved_cfg_path.exists():
            raise RuntimeError(
                f"MAE checkpoint '{ckpt_path}' is missing serialized args and fallback config "
                f"'{resolved_cfg_path}' does not exist."
            )
        print(
            f"[INFO] MAE checkpoint '{ckpt_path}' missing serialized args; "
            f"loading configuration from '{resolved_cfg_path}'."
        )
        args_cfg = OmegaConf.load(str(resolved_cfg_path))
    try:
        mae_cfg = OmegaConf.create(args_cfg)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to reconstruct MAE config from checkpoint '{ckpt_path}': {exc}") from exc
    model = _build_mae_model(mae_cfg)
    state_dict = ckpt.get("model_state")
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, mae_cfg, ckpt


def resolve_mae_embedding_dim(model: nn.Module) -> int:
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        patch_proj = getattr(encoder, "patch_proj", None)
        if patch_proj is not None and hasattr(patch_proj, "out_features"):
            return int(patch_proj.out_features)
        embed_dim = getattr(encoder, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        embed_dim = getattr(backbone, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)
    model_dim = getattr(model, "embed_dim", None)
    if model_dim is not None:
        return int(model_dim)
    raise AttributeError("Unable to resolve MAE embedding dimension from model.")


class MAEEmbeddingWrapper(nn.Module):
    """
    Thin wrapper that exposes MAE encoder/backbone embeddings for downstream evaluation.
    """

    def __init__(self, mae_model: nn.Module, embedding_key: str = "embedding"):
        super().__init__()
        self.mae = mae_model
        self.encoder = getattr(mae_model, "encoder", None)
        self.embedding_key = embedding_key
        self._warned_transpose = False
        self._warned_resize = False

    @property
    def embedding_keys(self) -> tuple[str, ...]:
        return (self.embedding_key,)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict:
        proc_inputs, _ = self._prepare_inputs(x)
        if self.encoder is not None:
            tokens = self.encoder.embed_tokens(proc_inputs)
            encoded = self.encoder.encode_tokens(tokens)
            pooled = encoded.mean(dim=1)
        else:
            pooled = self._encode_with_backbone(proc_inputs)
        return {self.embedding_key: pooled}

    def _encode_with_backbone(self, inputs: torch.Tensor) -> torch.Tensor:
        backbone = getattr(self.mae, "backbone", None)
        if backbone is None:
            raise AttributeError("MAE model has no encoder or backbone for embedding extraction.")
        outputs = backbone(inputs)
        if isinstance(outputs, tuple):
            tokens = outputs[0]
        elif isinstance(outputs, dict) and "tokens" in outputs:
            tokens = outputs["tokens"]
        else:
            tokens = outputs
        if tokens.dim() != 3:
            raise ValueError("Backbone output does not appear to be token embeddings.")
        return tokens.mean(dim=1)

    def _prepare_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, MAEInputTransform]:
        target_size = self._get_target_size()
        if target_size is None:
            return inputs, MAEInputTransform(swapped_axes=False, resized=False, pre_resize_size=tuple(inputs.shape[-2:]))
        orig_size = tuple(inputs.shape[-2:])
        swapped_axes = tuple(orig_size[::-1]) == target_size
        proc_inputs = inputs.transpose(-1, -2) if swapped_axes else inputs
        if swapped_axes and not self._warned_transpose:
            print(
                f"[MAE Eval] Input spectrogram size {orig_size} appears transposed relative to MAE "
                f"training size {target_size}; transposing axes for evaluation."
            )
            self._warned_transpose = True
        pre_resize_size = tuple(proc_inputs.shape[-2:])
        need_resize = pre_resize_size != target_size
        if need_resize:
            if not self._warned_resize:
                print(
                    f"[MAE Eval] Input spectrogram size {pre_resize_size} differs from MAE training "
                    f"size {target_size}; resizing before feature extraction."
                )
                self._warned_resize = True
            proc_inputs = F.interpolate(proc_inputs, size=target_size, mode="bilinear", align_corners=False)
        return proc_inputs, MAEInputTransform(swapped_axes=swapped_axes, resized=need_resize, pre_resize_size=pre_resize_size)

    def _get_target_size(self) -> tuple[int, int] | None:
        if self.encoder is not None:
            return (int(self.encoder.input_freq), int(self.encoder.input_time))
        input_freq = getattr(self.mae, "input_freq", None)
        input_time = getattr(self.mae, "input_time", None)
        if input_freq is not None and input_time is not None:
            return (int(input_freq), int(input_time))
        return None


__all__ = ["MAEEmbeddingWrapper", "load_mae_checkpoint", "resolve_mae_embedding_dim"]
