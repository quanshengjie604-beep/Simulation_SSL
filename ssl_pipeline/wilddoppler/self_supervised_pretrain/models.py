from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_torchvision():
    try:
        from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights, mobilenet_v2, resnet18
    except ImportError as exc:  # pragma: no cover - depends on active environment
        raise ImportError(
            "torchvision is required for mobilenet_v2/resnet18 pretraining. "
            "Install the repo requirements or run from an environment with torchvision."
        ) from exc
    return MobileNet_V2_Weights, ResNet18_Weights, mobilenet_v2, resnet18


class RadarConvStem(nn.Module):
    def __init__(self, out_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileNetV2Backbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        stem_channels: int = 32,
        use_radar_stem: bool = False,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        MobileNet_V2_Weights, _, mobilenet_v2, _ = _require_torchvision()
        net = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        with torch.no_grad():
            net.eval()
            last_channels = int(net.features(torch.zeros(1, 3, 64, 64)).shape[1])
            net.train()
        self.use_radar_stem = bool(use_radar_stem)
        if self.use_radar_stem:
            # Radar stem + 1×1 projection lets early conv layers specialize before MobileNet's RGB layers.
            self.stem = RadarConvStem(out_ch=stem_channels)
            self.to_rgb = nn.Conv2d(stem_channels, 3, kernel_size=1)
        else:
            # Simpler path: patch the first conv in-place to accept 1-channel input.
            first = net.features[0][0]
            net.features[0][0] = nn.Conv2d(
                1,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=first.bias is not None,
            )
            self.stem = None
            self.to_rgb = None
        self.features = net.features
        self.proj = nn.Conv2d(last_channels, embed_dim, kernel_size=1)
        self.embed_dim = int(embed_dim)

    def forward(self, x: torch.Tensor):
        if self.use_radar_stem:
            x = self.to_rgb(self.stem(x))
        feat = self.proj(self.features(x))
        b, d, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2), (h, w)


class ResNet18Backbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        stem_channels: int = 32,
        use_radar_stem: bool = False,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        _, ResNet18_Weights, _, resnet18 = _require_torchvision()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.use_radar_stem = bool(use_radar_stem)
        if self.use_radar_stem:
            self.stem = RadarConvStem(out_ch=stem_channels)
            self.to_rgb = nn.Conv2d(stem_channels, 3, kernel_size=1)
        else:
            first = net.conv1
            net.conv1 = nn.Conv2d(
                1,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=first.bias is not None,
            )
            self.stem = None
            self.to_rgb = None
        self.features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.embed_dim = int(embed_dim)

    def forward(self, x: torch.Tensor):
        if self.use_radar_stem:
            x = self.to_rgb(self.stem(x))
        feat = self.proj(self.features(x))
        b, d, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2), (h, w)


def build_backbone(
    model_name: str = "mobilenet_v2",
    embed_dim: int = 256,
    stem_channels: int = 32,
    use_radar_stem: bool = False,
    pretrained_imagenet: bool = False,
) -> nn.Module:
    if model_name == "mobilenet_v2":
        return MobileNetV2Backbone(embed_dim, stem_channels, use_radar_stem, pretrained_imagenet)
    if model_name == "resnet18":
        return ResNet18Backbone(embed_dim, stem_channels, use_radar_stem, pretrained_imagenet)
    raise ValueError("Only mobilenet_v2 and resnet18 are supported.")


class MeanPooling(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512, hidden_dim: int = 1024, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(cur, hidden_dim), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1, eps=1e-6)  # unit sphere required by contrastive loss


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        out_dim: int = 512,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        embed_dim = int(backbone.embed_dim)
        self.pool = MeanPooling()
        self.head = ProjectionHead(embed_dim, out_dim=out_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.backbone(x)
        return self.head(self.pool(tokens))


def build_contrastive_model(args) -> ContrastiveModel:
    backbone = build_backbone(
        model_name=args.model_name,
        embed_dim=args.embed_dim,
        stem_channels=args.stem_channels,
        use_radar_stem=args.use_radar_stem,
        pretrained_imagenet=args.pretrained_imagenet,
    )
    return ContrastiveModel(
        backbone=backbone,
        out_dim=args.proj_dim,
        hidden_dim=args.proj_hidden_dim,
    )
