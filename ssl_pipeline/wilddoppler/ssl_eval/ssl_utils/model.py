import copy
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights,
)
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# -------------------------------------------------
# Token-level conditioning utilities
# -------------------------------------------------
class TokenConditioning(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        embed_dim: int,
        fusion: str = "film",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.embed_dim = int(embed_dim)
        self.fusion = str(fusion or "film").lower()
        self.drop_prob = max(0.0, min(1.0, float(drop_prob)))
        if self.fusion not in {"film", "token", "concat_proj"}:
            raise ValueError(f"Unsupported conditioning fusion '{fusion}'.")
        hidden_dim = hidden_dim or max(self.embed_dim, self.cond_dim)
        layers = [
            nn.Linear(self.cond_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        out_dim = self.embed_dim * 2 if self.fusion == "film" else self.embed_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        # concat_proj: project [tokens || cond_proj] back to embed_dim
        if self.fusion == "concat_proj":
            self.proj_out = nn.Linear(self.embed_dim * 2, self.embed_dim)

    def forward(self, tokens: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            return tokens
        if self.drop_prob > 0 and self.training:
            if torch.rand(1, device=tokens.device).item() < self.drop_prob:
                return tokens
        cond = cond.float()
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = self.net(cond)
        if self.fusion == "film":
            gamma, beta = cond.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            return tokens * (1 + gamma) + beta
        if self.fusion == "concat_proj":
            # cond: [B, D] → expand over tokens → concat → project back
            cond_proj = cond.unsqueeze(1).expand(-1, tokens.shape[1], -1)  # [B, N, D]
            return self.proj_out(torch.cat([tokens, cond_proj], dim=-1))   # [B, N, D]
        # token fusion: prepend a conditioning token
        cond_token = cond.unsqueeze(1)
        return torch.cat([cond_token, tokens], dim=1)


def _build_token_conditioner(cfg, embed_dim: int) -> Optional[TokenConditioning]:
    if not cfg:
        return None
    enabled = bool(cfg.get("enabled", False))
    feature_cols = cfg.get("feature_columns", [])
    default_dim = len(feature_cols) if isinstance(feature_cols, (list, tuple)) else 0
    cond_dim = int(cfg.get("condition_dim", default_dim))
    if not enabled or cond_dim <= 0:
        return None
    fusion = cfg.get("fusion", "film")
    hidden = int(cfg.get("fusion_hidden_dim", max(cond_dim, embed_dim)))
    dropout = float(cfg.get("fusion_dropout", 0.0))
    return TokenConditioning(
        cond_dim=cond_dim,
        embed_dim=embed_dim,
        fusion=fusion,
        hidden_dim=hidden,
        dropout=dropout,
    )


# -------------------------------------------------
# Radar-specific conv stem (CNN part)
# -------------------------------------------------
class RadarConvStem(nn.Module):
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        # Small, relatively cheap conv stack
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            # Preserve higher Doppler resolution by avoiding early downsampling
            nn.Conv2d(16, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, 1, F, T]
        return self.net(x)  # [B, out_ch, F', T']


# -------------------------------------------------
# MobileViT backbone for microDoppler (CNN + MobileViT)
# -------------------------------------------------
class MobileViTBackbone(nn.Module):
    def __init__(
        self,
        model_name="mobilevit_s",
        embed_dim=256,
        stem_channels=32,
        output_stride=16,
        use_radar_stem: bool = True,
        conditioning_cfg=None,
    ):
        super().__init__()

        self.use_radar_stem = bool(use_radar_stem)
        if self.use_radar_stem:
            # 1) Radar CNN stem (radar-specific, trained from scratch)
            self.stem = RadarConvStem(in_ch=1, out_ch=stem_channels)
            # 2) Map stem output → 3 channels for MobileViT
            self.to_rgb = nn.Conv2d(stem_channels, 3, kernel_size=1)
            in_chans = 3
        else:
            self.stem = None
            self.to_rgb = None
            in_chans = 1

        # 3) MobileViT feature extractor (pretrained on ImageNet)
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=[-1],   # last feature map
            in_chans=in_chans,
            output_stride=int(output_stride), # reduce spatial size by configurable stride
        )

        # 4) Get output channels of last feature map
        C_out = self.backbone.feature_info.channels()[-1]

        # 5) Project to JEPA embedding dim
        self.proj = nn.Linear(C_out, embed_dim)
        self.embed_dim = embed_dim
        self.conditioner = _build_token_conditioner(conditioning_cfg, embed_dim)

    def forward(self, x, cond: Optional[torch.Tensor] = None):
        """
        x: [B, 1, F, T]
        cond: [B, C] optional conditioning vector
        Returns:
          tokens: [B, N, D]
          (H, W): spatial size after backbone
        """
        if self.use_radar_stem:
            # Radar CNN stem + 1x1 conv to RGB for MobileViT.
            x = self.stem(x)            # [B, stem_channels, F', T']
            x = self.to_rgb(x)          # [B, 3, F', T']

        # MobileViT feature maps
        feat = self.backbone(x)[0]  # [B, C_out, H, W]
        B, C_out, H, W = feat.shape

        # Flatten spatial → tokens
        feat = feat.flatten(2).transpose(1, 2)  # [B, N, C_out], N = H*W

        # Project to JEPA embedding dim
        tokens = self.proj(feat)                # [B, N, D]
        if self.conditioner is not None:
            tokens = self.conditioner(tokens, cond)
        return tokens, (H, W)


class MobileNetBackbone(nn.Module):
    """
    MobileNetV2 backbone adapted for single-channel microDoppler inputs.
    Optionally prepends a radar-specific CNN stem that maps the input to 3
    channels, which lets us keep the pretrained MobileNet weights intact.
    """

    def __init__(
        self,
        model_name="mobilenet_v2",
        embed_dim=256,
        stem_channels=32,
        use_radar_stem: bool = False,
        pretrained: bool = True,
        conditioning_cfg=None,
    ):
        super().__init__()
        if model_name != "mobilenet_v2":
            raise ValueError(f"Unsupported model_name '{model_name}'. Expected 'mobilenet_v2'.")

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        mobilenet = mobilenet_v2(weights=weights)
        self.use_radar_stem = bool(use_radar_stem)
        self.stem = None
        self.to_rgb = None

        if self.use_radar_stem:
            # Learn a lightweight radar-specific adapter → 3 channels,
            # keeping the pretrained MobileNet weights untouched.
            self.stem = RadarConvStem(in_ch=1, out_ch=stem_channels)
            self.to_rgb = nn.Conv2d(stem_channels, 3, kernel_size=1)
        else:
            # Directly adapt the first MobileNet conv to single-channel input.
            first_conv = mobilenet.features[0][0]
            mobilenet.features[0][0] = nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

        self.backbone = mobilenet.features
        self.proj = nn.Conv2d(mobilenet.last_channel, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim
        self.conditioner = _build_token_conditioner(conditioning_cfg, embed_dim)

    def forward(self, x, cond: Optional[torch.Tensor] = None):
        if self.use_radar_stem:
            x = self.stem(x)
            x = self.to_rgb(x)
        feat = self.backbone(x)
        feat = self.proj(feat)
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        if self.conditioner is not None:
            tokens = self.conditioner(tokens, cond)
        return tokens, (H, W)


class ResNetBackbone(nn.Module):
    """
    ResNet backbone (resnet18/34/50/101/152) for single-channel microDoppler.
    """

    _REGISTRY = {
        "resnet18":  (resnet18,  ResNet18_Weights,  512),
        "resnet34":  (resnet34,  ResNet34_Weights,  512),
        "resnet50":  (resnet50,  ResNet50_Weights,  2048),
        "resnet101": (resnet101, ResNet101_Weights, 2048),
        "resnet152": (resnet152, ResNet152_Weights, 2048),
    }

    def __init__(
        self,
        model_name: str = "resnet18",
        embed_dim: int = 256,
        stem_channels: int = 32,
        use_radar_stem: bool = False,
        pretrained: bool = True,
        conditioning_cfg=None,
    ):
        super().__init__()
        if model_name not in self._REGISTRY:
            raise ValueError(
                f"Unsupported ResNet variant '{model_name}'. "
                f"Choose from: {list(self._REGISTRY.keys())}"
            )
        factory, weights_cls, last_channels = self._REGISTRY[model_name]
        weights = weights_cls.DEFAULT if pretrained else None
        net = factory(weights=weights)

        self.use_radar_stem = bool(use_radar_stem)
        self.stem = None
        self.to_rgb = None

        if self.use_radar_stem:
            self.stem = RadarConvStem(in_ch=1, out_ch=stem_channels)
            self.to_rgb = nn.Conv2d(stem_channels, 3, kernel_size=1)
        else:
            first = net.conv1
            net.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=first.bias is not None,
            )

        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.proj = nn.Conv2d(last_channels, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim
        self.conditioner = _build_token_conditioner(conditioning_cfg, embed_dim)

    def forward(self, x, cond: Optional[torch.Tensor] = None):
        if self.use_radar_stem:
            x = self.stem(x)
            x = self.to_rgb(x)
        feat = self.backbone(x)          # [B, C, H, W]
        feat = self.proj(feat)           # [B, D, H, W]
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # [B, N, D]
        if self.conditioner is not None:
            tokens = self.conditioner(tokens, cond)
        return tokens, (H, W)


# -------------------------------------------------
# Simple random token mask
# -------------------------------------------------
def sample_token_mask(B, N, mask_ratio, device):
    """
    Returns boolean mask [B, N], True for masked tokens.
    """
    num_mask = int(mask_ratio * N)
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(N, device=device)[:num_mask]
        mask[b, idx] = True
    return mask


# def sample_block_mask(
#     B,
#     H,
#     W,
#     mask_ratio,
#     device,
#     min_block: int = 4,
#     max_block: int | None = None,
# ):
#     """
#     Block masking in (H, W) space.

#     Returns:
#         mask_flat: [B, H*W] boolean, True = masked token.

#     Strategy:
#       - Keep sampling random rectangles until ~mask_ratio of tokens are masked.
#       - Rectangles are axis-aligned blocks in the HxW grid.
#     """
#     if max_block is None:
#         # don't let blocks be bigger than half the side by default
#         max_block = max(4, min(H, W) // 2)

#     total_tokens = H * W
#     target_mask = int(mask_ratio * total_tokens)

#     mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

#     for b in range(B):
#         num_masked = 0
#         # keep adding blocks until we hit target
#         while num_masked < target_mask:
#             bh = torch.randint(min_block, max_block + 1, (1,), device=device).item()
#             bw = torch.randint(min_block, max_block + 1, (1,), device=device).item()

#             top = torch.randint(0, max(1, H - bh + 1), (1,), device=device).item()
#             left = torch.randint(0, max(1, W - bw + 1), (1,), device=device).item()

#             block = mask[b, top : top + bh, left : left + bw]
#             newly = (~block).sum().item()
#             if newly == 0:
#                 # this block is already fully masked, try another
#                 continue

#             block[:] = True
#             num_masked += newly

#     # [B, H, W] -> [B, H*W]
#     return mask.view(B, -1)


def sample_block_mask(
    B,
    H,
    W,
    mask_ratio,
    device,
    min_block: int = 4,
    max_block: int | None = None,
):
    """
    Block masking in (H, W) space without any frequency bias.
    """
    if max_block is None:
        max_block = max(4, min(H, W) // 2)

    total_tokens = H * W
    target_mask = int(mask_ratio * total_tokens)

    mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

    for b in range(B):
        num_masked = 0
        while num_masked < target_mask:
            bh = torch.randint(min_block, max_block + 1, (1,), device=device).item()
            bw = torch.randint(min_block, max_block + 1, (1,), device=device).item()

            # clamp block size if larger than grid
            bh = min(bh, H)
            bw = min(bw, W)

            top = torch.randint(0, max(1, H - bh + 1), (1,), device=device).item()
            left = torch.randint(0, max(1, W - bw + 1), (1,), device=device).item()

            block = mask[b, top : top + bh, left : left + bw]
            newly = (~block).sum().item()
            if newly == 0:
                continue

            block[:] = True
            num_masked += newly

    return mask.view(B, -1)

def sample_raw_block_mask(
    x,
    mask_ratio,
    min_block_f: int = 8,
    max_block_f: int | None = None,
    min_block_t: int = 8,
    max_block_t: int | None = None,
    mid_band_ratio=(0.3, 0.8),
    mid_focus_prob: float = 0.7,
):
    """
    Block masking directly in raw spectrogram space (F x T),
    with a bias toward masking middle frequency rows (F dimension).

    Args:
        x: [B, 1, F, T] input spectrogram
        mask_ratio: fraction of *pixels* (approx.) to mask
        min_block_f, max_block_f: vertical block size range in freq bins
        min_block_t, max_block_t: horizontal block size range in time bins
        mid_band_ratio: (lo, hi) fractional range in F to define
                        the "mid-frequency" band, e.g. (0.3, 0.8)
        mid_focus_prob: probability that a sampled block is biased
                        to start in the mid-frequency band.

    Returns:
        x_masked: [B, 1, F, T] with blocks zeroed
        mask:     [B, 1, F, T] boolean, True = masked pixel
    """
    B, C, F, T = x.shape
    device = x.device
    assert C == 1, "Assuming single-channel microDoppler input."

    if max_block_f is None:
        max_block_f = max(min_block_f, F // 2)
    if max_block_t is None:
        max_block_t = max(min_block_t, T // 2)

    total_pixels = F * T
    target_mask = int(mask_ratio * total_pixels)

    mask = torch.zeros(B, 1, F, T, dtype=torch.bool, device=device)

    # define mid-band indices in F (freq axis)
    f_lo = int(F * mid_band_ratio[0])
    f_hi = int(F * mid_band_ratio[1]) - 1
    f_lo = max(0, min(f_lo, F - 1))
    f_hi = max(f_lo, min(f_hi, F - 1))

    for b in range(B):
        num_masked = 0
        while num_masked < target_mask:
            bf = torch.randint(min_block_f, max_block_f + 1, (1,), device=device).item()
            bt = torch.randint(min_block_t, max_block_t + 1, (1,), device=device).item()

            bf = min(bf, F)
            bt = min(bt, T)

            use_mid = torch.rand(1, device=device).item() < mid_focus_prob

            if use_mid and F > bf:
                top_lo = f_lo
                top_hi = min(f_hi, F - bf)
                if top_lo > top_hi:
                    top_lo, top_hi = 0, F - bf
            else:
                top_lo, top_hi = 0, max(0, F - bf)

            top = torch.randint(top_lo, top_hi + 1, (1,), device=device).item()
            left = torch.randint(0, max(1, T - bt + 1), (1,), device=device).item()

            block = mask[b, 0, top : top + bf, left : left + bt]
            newly = (~block).sum().item()
            if newly == 0:
                continue

            block[:] = True
            num_masked += newly

    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def nested_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dims: list[int] | tuple[int, ...],
    weights: list[float] | tuple[float, ...],
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute coarse-to-fine nested MSE where progressively larger feature
    slices receive different weights.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must share shape for nested loss")

    if pred.ndim < 1:
        raise ValueError("nested loss expects at least 1D tensors")

    embed_dim = pred.size(-1)
    dims = [int(d) for d in (dims or []) if int(d) > 0]
    dims = sorted(set(min(embed_dim, d) for d in dims if d < embed_dim))
    num_levels = len(dims) + 1

    if not weights:
        weights = [1.0] * num_levels
    else:
        weights = [float(w) for w in weights]
        if len(weights) < num_levels:
            pad_value = weights[-1] if weights else 1.0
            weights = weights + [pad_value] * (num_levels - len(weights))
        elif len(weights) > num_levels:
            weights = weights[:num_levels]

    total_weight = float(sum(weights)) if normalize else 1.0
    if normalize and total_weight <= 0.0:
        total_weight = 1.0

    loss = pred.new_zeros(())
    for idx, dim in enumerate(dims):
        partial_loss = F.mse_loss(pred[..., :dim], target[..., :dim])
        loss = loss + weights[idx] * partial_loss

    full_loss = F.mse_loss(pred, target)
    loss = loss + weights[-1] * full_loss

    if normalize:
        loss = loss / total_weight
    return loss


# -------------------------------------------------
# RJEPA-style wrapper
# -------------------------------------------------
class RJEPA(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim=256,
        pred_dim=512,
        ema_decay=0.996,
        use_block_mask=True,
        args=None,
        use_raw_mask: bool = True,
        use_predictor: bool = True,
    ):
        super().__init__()
        self.context_encoder = backbone
        self.target_encoder = copy.deepcopy(backbone)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if use_predictor:
            self.predictor = nn.Sequential(
                nn.Linear(embed_dim, pred_dim),
                nn.GELU(),
                nn.Linear(pred_dim, embed_dim),
            )
        else:
            self.predictor = nn.Identity()

        self.ema_decay = ema_decay
        self.embed_dim = embed_dim
        self.use_block_mask = use_block_mask
        self.args = args
        self.use_raw_mask = use_raw_mask   # NEW
        self.use_predictor = use_predictor
        loss_cfg = getattr(args, "loss", None) if args is not None else None
        nested_cfg = getattr(loss_cfg, "nested_loss", None) if loss_cfg is not None else None
        self.use_nested_loss = bool(getattr(nested_cfg, "enabled", False)) if nested_cfg is not None else False
        if self.use_nested_loss:
            dims = getattr(nested_cfg, "dims", None) or []
            weights = getattr(nested_cfg, "weights", None) or []
            self.nested_dims = [int(v) for v in dims]
            self.nested_weights = [float(w) for w in weights]
            self.nested_normalize = bool(getattr(nested_cfg, "normalize", False))
        else:
            self.nested_dims = []
            self.nested_weights = []
            self.nested_normalize = False

    @torch.no_grad()
    def update_ema(self):
        for p_t, p_s in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_s.data, alpha=1.0 - self.ema_decay)

    def forward(
        self,
        x,
        mask_ratio=0.5,
        use_raw_mask: bool | None = None,
        student_x: torch.Tensor | None = None,
        return_debug: bool = False,
    ):
        """
        x: [B, 1, F, T]
        Returns:
          loss: scalar
          aux:  dict with H, W, mask_ratio
        """
        B = x.size(0)
        device = x.device

        debug_payload = None

        # select which input drives the student/context branch
        context_input = student_x if student_x is not None else x

        # --------- RAW PATCH MASKING (augmentation) ----------
        use_raw = self.use_raw_mask if use_raw_mask is None else use_raw_mask

        if use_raw:
            # you can keep mask_ratio here or add another hyper-param
            x_masked, raw_mask = sample_raw_block_mask(
                context_input,
                mask_ratio=mask_ratio,
                min_block_f=self.args.model.min_block_f,
                max_block_f=self.args.model.max_block_f,
                min_block_t=self.args.model.min_block_t,
                max_block_t=self.args.model.max_block_t,
                mid_band_ratio=self.args.model.mid_band_ratio,
                mid_focus_prob=self.args.model.mid_focus_prob,
            )
        else:
            x_masked, raw_mask = context_input, None

        # --------- ENCODERS (same masked input to both) ----------
        z_s, (H, W) = self.context_encoder(x_masked)   # [B, N, D]
        with torch.no_grad():
            z_t, _ = self.target_encoder(x)     # [B, N, D]

        # Global representations for Barlow Twins loss (mean-pool tokens)
        student_global = z_s.mean(dim=1)        # [B, D]
        teacher_global = z_t.mean(dim=1)        # [B, D]
        # N = H * W
        # --------- TOKEN-SPACE MASK (for JEPA loss) ----------
        if self.use_block_mask:
            mask = sample_block_mask(
                B=B,
                H=H,
                W=W,
                mask_ratio=mask_ratio,
                device=device,
                min_block=self.args.model.min_block,
                max_block=self.args.model.max_block,
            )   # [B, H*W]
        else:
            mask = sample_token_mask(B, H * W, mask_ratio, device)  # [B, N]

        pred = self.predictor(z_s)         # [B, N, D]

        mask_float = mask.float()
        tokens_per_sample = H * W
        mask_ratio_actual = float(mask_float.mean().item())
        masked_tokens_per_sample = mask_ratio_actual * tokens_per_sample
        mask_counts = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        mask_float_exp = mask_float.unsqueeze(-1)

        pred_global = (pred * mask_float_exp).sum(dim=1) / mask_counts
        target_global = (z_t.detach() * mask_float_exp).sum(dim=1) / mask_counts
        global_loss = self._compute_reconstruction_loss(pred_global, target_global)

        # Local/token reconstruction
        pred_masked   = pred[mask]         # [Nm, D]
        target_masked = z_t[mask].detach() # [Nm, D]
        local_loss = self._compute_reconstruction_loss(pred_masked, target_masked)

        if return_debug:
            debug_payload = {
                "target": x.detach().cpu(),
                "context": x_masked.detach().cpu(),
                "raw_mask": raw_mask.detach().cpu() if raw_mask is not None else None,
            }

        aux = {
            "H": H,
            "W": W,
            "mask_ratio": mask_ratio,
            "raw_masked_fraction": float(raw_mask.float().mean().item()) if raw_mask is not None else 0.0,
            "student_global": student_global,
            "teacher_global": teacher_global,
            "local_loss": local_loss,
            "tokens_per_sample": float(tokens_per_sample),
            "masked_tokens_per_sample": float(masked_tokens_per_sample),
            "mask_ratio_actual": mask_ratio_actual,
        }
        if debug_payload is not None:
            aux["debug"] = debug_payload
        return global_loss, aux

    def _compute_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.use_nested_loss:
            return F.mse_loss(pred, target)
        return nested_mse_loss(
            pred,
            target,
            dims=self.nested_dims,
            weights=self.nested_weights,
            normalize=self.nested_normalize,
        )
