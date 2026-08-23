"""Model construction for supervised load classification."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)
from omegaconf import DictConfig


def _model_value(cfg: DictConfig, attr: str, default):
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is not None:
        if isinstance(model_cfg, DictConfig):
            if attr in model_cfg:
                return model_cfg[attr]
        elif isinstance(model_cfg, Mapping):
            if attr in model_cfg:
                return model_cfg[attr]
        elif hasattr(model_cfg, attr):
            return getattr(model_cfg, attr)
    if isinstance(cfg, DictConfig):
        if attr in cfg:
            return cfg[attr]
    elif isinstance(cfg, Mapping):
        if attr in cfg:
            return cfg[attr]
    elif hasattr(cfg, attr):
        return getattr(cfg, attr)
    return default


def _replace_conv_input_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        # Preserve the pretrained filter scale by averaging over the original RGB channels,
        # then replicating that average into the requested number of radar channels.
        base_weight = conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(base_weight.repeat(1, in_channels, 1, 1))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _infer_feature_channels(module: nn.Module) -> int:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return int(child.out_channels)
    raise ValueError(f"Unable to infer output channels from module type '{type(module).__name__}'.")


class MeanTokenPooling(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, N, D) but received shape {tokens.shape}.")
        return tokens.mean(dim=1)


class MobileNetBackbone(nn.Module):
    def __init__(
        self,
        model_name="mobilenet_v2",
        embed_dim=256,
        pretrained: bool = True,
        input_channels: int = 1,
    ):
        super().__init__()
        builders = {
            "mobilenet_v2": (mobilenet_v2, MobileNet_V2_Weights.DEFAULT),
        }
        if model_name not in builders:
            supported = "', '".join(sorted(builders))
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Expected one of '{supported}'."
            )

        model_builder, default_weights = builders[model_name]
        weights = default_weights if pretrained else None
        mobilenet = model_builder(weights=weights)

        first_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = _replace_conv_input_channels(first_conv, int(input_channels))

        self.backbone = mobilenet.features
        feature_channels = _infer_feature_channels(self.backbone)
        self.proj = nn.Conv2d(feature_channels, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        _, _, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, (H, W)


class ResNetBackbone(nn.Module):
    """ResNet backbone (resnet18/34/50) for single-channel microDoppler."""

    def __init__(
        self,
        model_name="resnet18",
        embed_dim=256,
        pretrained: bool = True,
        input_channels: int = 1,
    ):
        super().__init__()
        builders = {
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        }
        if model_name not in builders:
            supported = "', '".join(sorted(builders))
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Expected one of '{supported}'."
            )
        model_builder, default_weights = builders[model_name]
        weights = default_weights if pretrained else None
        resnet = model_builder(weights=weights)

        resnet.conv1 = _replace_conv_input_channels(resnet.conv1, int(input_channels))

        # Drop avgpool and fc — keep spatial feature maps for token pooling
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        feature_channels = _infer_feature_channels(self.backbone)
        self.proj = nn.Conv2d(feature_channels, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        _, _, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, (H, W)


class SupervisedRadarClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int,
        head_hidden_dims=None,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = MeanTokenPooling()
        hidden_dims = [int(dim) for dim in (head_hidden_dims or [])]
        dims = [int(embed_dim)] + hidden_dims + [int(num_classes)]

        layers = []
        for idx in range(len(dims) - 1):
            in_dim, out_dim = dims[idx], dims[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if head_dropout and head_dropout > 0:
                    layers.append(nn.Dropout(p=float(head_dropout)))
        self.head = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.backbone(x)
        pooled = self.pool(tokens)
        return self.head(pooled)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.backbone(x)
        return self.pool(tokens)


def _build_backbone(cfg: DictConfig) -> nn.Module:
    model_name = str(_model_value(cfg, "model_name", "mobilenet_v2"))
    embed_dim = int(_model_value(cfg, "embed_dim", 512))
    pretrained = bool(_model_value(cfg, "pretrained", False))
    input_channels = int(_model_value(cfg, "input_channels", 1))

    if model_name.startswith("mobilenet"):
        return MobileNetBackbone(
            model_name=model_name,
            embed_dim=embed_dim,
            pretrained=pretrained,
            input_channels=input_channels,
        )
    if model_name.startswith("resnet"):
        return ResNetBackbone(
            model_name=model_name,
            embed_dim=embed_dim,
            pretrained=pretrained,
            input_channels=input_channels,
        )
    raise ValueError(f"Unsupported model_name '{model_name}'.")


def build_supervised_model(cfg: DictConfig) -> SupervisedRadarClassifier:
    backbone = _build_backbone(cfg)
    embed_dim = int(_model_value(cfg, "embed_dim", 512))
    head_hidden = _model_value(cfg, "head_hidden_dims", [])
    head_dropout = float(_model_value(cfg, "head_dropout", 0.0))
    num_classes = int(getattr(cfg.train, "num_classes", 0))
    if num_classes <= 0:
        raise ValueError("train.num_classes must be set before building the model.")
    return SupervisedRadarClassifier(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        head_hidden_dims=head_hidden,
        head_dropout=head_dropout,
    )
