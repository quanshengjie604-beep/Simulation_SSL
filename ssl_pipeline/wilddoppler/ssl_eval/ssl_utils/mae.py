import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from .contrastive_builders import build_backbone
except ModuleNotFoundError:  # pragma: no cover - support legacy import paths
    from .contrastive_builders import build_backbone

def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return tensor.normal_(mean=0.0, std=std).clamp_(-2 * std, 2 * std)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=max(1, int(num_heads)),
            dropout=max(0.0, float(dropout)),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_out = self.attn(attn_input, attn_input, attn_input, need_weights=False)[0]
        x = x + attn_out
        mlp_input = self.norm2(x)
        x = x + self.mlp(mlp_input)
        return x


class GaussianBlur2d(nn.Module):
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Gaussian kernel size must be odd.")
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        weight = self.kernel.expand(channels, 1, -1, -1)
        padding = self.kernel.size(-1) // 2
        return F.conv2d(x, weight, padding=padding, groups=channels)


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
    freq_coords: Optional[torch.Tensor] = None,
    mid_band_ratio: Optional[Tuple[float, float]] = None,
    mid_focus_prob: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, D = x.shape
    len_keep = max(1, int(round(N * (1.0 - mask_ratio))))
    freq_bias = freq_coords is not None and mid_focus_prob > 0.0
    if not freq_bias:
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        batch_idx = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_masked = torch.gather(x, dim=1, index=batch_idx)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    freq_coords = freq_coords.to(x.device)
    if freq_coords.dim() == 1:
        freq_coords = freq_coords.unsqueeze(0)
    if freq_coords.size(0) == 1 and B > 1:
        freq_coords = freq_coords.expand(B, -1)
    if freq_coords.shape[0] != B or freq_coords.shape[1] != N:
        raise ValueError("freq_coords must have shape [B, N] or broadcastable to it.")

    try:
        mid_lo, mid_hi = mid_band_ratio if mid_band_ratio is not None else (0.25, 0.75)
    except Exception:
        mid_lo, mid_hi = 0.25, 0.75
    mid_lo = float(max(0.0, min(1.0, mid_lo)))
    mid_hi = float(max(0.0, min(1.0, mid_hi)))
    if mid_lo > mid_hi:
        mid_lo, mid_hi = mid_hi, mid_lo

    len_mask = N - len_keep
    mask = torch.ones(B, N, device=x.device)
    ids_keep = torch.empty(B, len_keep, dtype=torch.long, device=x.device)
    ids_restore = torch.empty(B, N, dtype=torch.long, device=x.device)
    mid_focus_prob = float(max(0.0, min(1.0, mid_focus_prob)))

    for b in range(B):
        freq_vec = freq_coords[b]
        mid_mask = (freq_vec >= mid_lo) & (freq_vec <= mid_hi)
        mid_indices = torch.nonzero(mid_mask, as_tuple=False).squeeze(1)
        target_mid = int(round(len_mask * mid_focus_prob))
        target_mid = min(len_mask, target_mid)
        if mid_indices.numel() == 0:
            target_mid = 0
        else:
            target_mid = min(target_mid, mid_indices.numel())

        mask_selected = torch.zeros(N, dtype=torch.bool, device=x.device)
        if target_mid > 0:
            perm = torch.randperm(mid_indices.numel(), device=x.device)
            chosen_mid = mid_indices[perm[:target_mid]]
            mask_selected[chosen_mid] = True

        remaining = len_mask - int(mask_selected.sum().item())
        if remaining > 0:
            candidates = (~mask_selected).nonzero(as_tuple=False).squeeze(1)
            if candidates.numel() <= remaining:
                mask_selected[candidates] = True
            else:
                perm = torch.randperm(candidates.numel(), device=x.device)
                extra = candidates[perm[:remaining]]
                mask_selected[extra] = True

        keep_indices = (~mask_selected).nonzero(as_tuple=False).squeeze(1)
        mask_indices = mask_selected.nonzero(as_tuple=False).squeeze(1)
        keep_perm = torch.randperm(keep_indices.numel(), device=x.device)
        mask_perm = torch.randperm(mask_indices.numel(), device=x.device)
        keep_shuffled = keep_indices[keep_perm]
        mask_shuffled = mask_indices[mask_perm]
        ids_keep[b] = keep_shuffled
        mask[b, keep_indices] = 0.0
        ids_shuffle = torch.cat([keep_shuffled, mask_shuffled], dim=0)
        ids_restore_b = torch.empty(N, dtype=torch.long, device=x.device)
        ids_restore_b[ids_shuffle] = torch.arange(N, device=x.device)
        ids_restore[b] = ids_restore_b

    batch_idx = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_masked = torch.gather(x, dim=1, index=batch_idx)
    return x_masked, mask, ids_restore


class RadarMAEEncoder(nn.Module):
    def __init__(
        self,
        input_freq: int,
        input_time: int,
        stem_channels: int,
        embed_dim: int,
        num_blocks: int,
        mlp_ratio: float,
        num_heads: int,
        stem_stride_freq: int,
        stem_stride_time: int,
        patch_freq: int,
        patch_time: int,
    ) -> None:
        super().__init__()
        if input_freq % stem_stride_freq != 0 or input_time % stem_stride_time != 0:
            raise ValueError("Input size must be divisible by stem stride.")
        feature_freq = input_freq // stem_stride_freq
        feature_time = input_time // stem_stride_time
        if feature_freq % patch_freq != 0 or feature_time % patch_time != 0:
            raise ValueError("Feature map must be divisible by patch size.")
        self.input_freq = input_freq
        self.input_time = input_time
        self.feature_freq = feature_freq
        self.feature_time = feature_time
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        self.grid_freq = feature_freq // patch_freq
        self.grid_time = feature_time // patch_time
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, kernel_size=3, stride=(stem_stride_freq, stem_stride_time), padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )
        self.patch_unfold = nn.Unfold(kernel_size=(patch_freq, patch_time), stride=(patch_freq, patch_time))
        self.patch_dim = stem_channels * patch_freq * patch_time
        self.patch_norm = nn.LayerNorm(self.patch_dim)
        self.patch_proj = nn.Linear(self.patch_dim, embed_dim)
        self.num_tokens = self.grid_freq * self.grid_time
        freq_coords = torch.arange(self.grid_freq, dtype=torch.float32).unsqueeze(1)
        freq_coords = freq_coords.repeat(1, self.grid_time).view(-1)
        if self.grid_freq > 1:
            freq_coords = freq_coords / (self.grid_freq - 1)
        else:
            freq_coords = freq_coords * 0.0
        self.register_buffer("token_freq_coords", freq_coords, persistent=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        trunc_normal_(self.pos_embed)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        mid_channels = max(1, stem_channels // 2)
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(input_freq, input_time), mode="bilinear", align_corners=False),
            nn.Conv2d(stem_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
        )
        self.patch_fold = nn.Fold(
            output_size=(feature_freq, feature_time),
            kernel_size=(patch_freq, patch_time),
            stride=(patch_freq, patch_time),
        )

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        patches = self.patch_unfold(feat).transpose(1, 2)
        tokens = self.patch_proj(self.patch_norm(patches))
        return tokens + self.pos_embed

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def unpatchify(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        patches = patch_tokens.transpose(1, 2)
        return self.patch_fold(patches)

    def reconstruct(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        feature_map = self.unpatchify(patch_tokens)
        return self.reconstruction_head(feature_map)


class RadarMAEDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        decoder_dim: int,
        decoder_layers: int,
        decoder_heads: int,
        patch_dim: int,
        decoder_mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        trunc_normal_(self.mask_token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, decoder_dim))
        trunc_normal_(self.pos_embed)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_dim,
                    decoder_heads,
                    mlp_ratio=decoder_mlp_ratio,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_dim)

    def forward(self, encoded: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(encoded)
        B, N_vis, _ = x.shape
        total_tokens = self.pos_embed.shape[1]
        if total_tokens < N_vis:
            raise ValueError("Decoder positional embeddings smaller than visible tokens.")
        if total_tokens != ids_restore.shape[1]:
            raise ValueError("ids_restore must match total token count.")
        mask_tokens = self.mask_token.expand(B, total_tokens - N_vis, -1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, total_tokens, x_.shape[-1]))
        x_ = x_ + self.pos_embed
        for block in self.blocks:
            x_ = block(x_)
        x_ = self.norm(x_)
        return self.pred(x_)


class RadarMAENetwork(nn.Module):
    def __init__(self, model_cfg, mae_cfg) -> None:
        super().__init__()
        mask_ratio = float(getattr(mae_cfg, "mask_ratio", 0.7))
        self.mask_ratio = min(0.99, max(0.0, mask_ratio))
        mid_focus = float(getattr(mae_cfg, "mid_focus_prob", 0.0))
        self.mid_focus_prob = max(0.0, min(1.0, mid_focus))
        mid_band = getattr(mae_cfg, "mid_band_ratio", (0.25, 0.75))
        try:
            self.mid_band_ratio = (float(mid_band[0]), float(mid_band[1]))
        except Exception:
            self.mid_band_ratio = (0.25, 0.75)
        self.encoder = RadarMAEEncoder(
            input_freq=int(getattr(model_cfg, "input_freq", 256)),
            input_time=int(getattr(model_cfg, "input_time", 90)),
            stem_channels=int(getattr(model_cfg, "stem_channels", 32)),
            embed_dim=int(getattr(model_cfg, "embed_dim", 256)),
            num_blocks=int(getattr(model_cfg, "num_blocks", 6)),
            mlp_ratio=float(getattr(model_cfg, "mlp_ratio", 2.0)),
            num_heads=int(getattr(model_cfg, "num_heads", 4)),
            stem_stride_freq=int(getattr(model_cfg, "stem_stride_freq", 4)),
            stem_stride_time=int(getattr(model_cfg, "stem_stride_time", 2)),
            patch_freq=int(getattr(model_cfg, "patch_freq", 4)),
            patch_time=int(getattr(model_cfg, "patch_time", 3)),
        )
        decoder_dim = int(getattr(mae_cfg, "decoder_dim", 256))
        decoder_layers = int(getattr(mae_cfg, "decoder_layers", 3))
        decoder_heads = int(getattr(mae_cfg, "decoder_heads", 4))
        decoder_mlp_ratio = float(getattr(mae_cfg, "decoder_mlp_ratio", 2.0))
        self.decoder = RadarMAEDecoder(
            num_tokens=self.encoder.num_tokens,
            embed_dim=int(getattr(model_cfg, "embed_dim", 256)),
            decoder_dim=decoder_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            patch_dim=self.encoder.patch_dim,
            decoder_mlp_ratio=decoder_mlp_ratio,
        )

    def mask_tokens_to_input(self, mask: torch.Tensor) -> torch.Tensor:
        """Upsample the token-level mask to the input spectrogram resolution."""
        if mask.dim() != 2:
            mask = mask.view(mask.size(0), -1)
        B, N = mask.shape
        grid_tokens = self.encoder.grid_freq * self.encoder.grid_time
        if N != grid_tokens:
            raise ValueError(
                f"Mask has {N} tokens but encoder grid expects {grid_tokens}."
            )
        mask_map = mask.view(B, 1, self.encoder.grid_freq, self.encoder.grid_time)
        mask_map = mask_map.repeat_interleave(self.encoder.patch_freq, dim=2)
        mask_map = mask_map.repeat_interleave(self.encoder.patch_time, dim=3)
        mask_map = F.interpolate(
            mask_map.float(),
            size=(self.encoder.input_freq, self.encoder.input_time),
            mode="nearest",
        )
        return mask_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder.embed_tokens(x)
        freq_coords = self.encoder.token_freq_coords if self.mid_focus_prob > 0 else None
        masked_tokens, mask, ids_restore = random_masking(
            tokens,
            self.mask_ratio,
            freq_coords=freq_coords,
            mid_band_ratio=self.mid_band_ratio,
            mid_focus_prob=self.mid_focus_prob,
        )
        encoded = self.encoder.encode_tokens(masked_tokens)
        decoded_patches = self.decoder(encoded, ids_restore)
        recon = self.encoder.reconstruct(decoded_patches)
        return recon, mask


class BackboneMAENetwork(nn.Module):
    def __init__(self, args: DictConfig, model_cfg, mae_cfg) -> None:
        super().__init__()
        mask_ratio = float(getattr(mae_cfg, "mask_ratio", 0.7))
        self.mask_ratio = min(0.99, max(0.0, mask_ratio))
        mid_focus = float(getattr(mae_cfg, "mid_focus_prob", 0.0))
        self.mid_focus_prob = max(0.0, min(1.0, mid_focus))
        mid_band = getattr(mae_cfg, "mid_band_ratio", (0.25, 0.75))
        try:
            self.mid_band_ratio = (float(mid_band[0]), float(mid_band[1]))
        except Exception:
            self.mid_band_ratio = (0.25, 0.75)
        self.input_freq = int(getattr(model_cfg, "input_freq", 256))
        self.input_time = int(getattr(model_cfg, "input_time", 90))

        self.backbone = build_backbone(args)
        self.embed_dim, self.grid_freq, self.grid_time = self._infer_backbone_grid()
        self.num_tokens = int(self.grid_freq * self.grid_time)

        freq_coords = torch.linspace(0.0, 1.0, steps=self.grid_freq)
        freq_coords = freq_coords.repeat_interleave(self.grid_time)
        self.register_buffer("token_freq_coords", freq_coords, persistent=False)

        decoder_dim = int(getattr(mae_cfg, "decoder_dim", 256))
        decoder_layers = int(getattr(mae_cfg, "decoder_layers", 3))
        decoder_heads = int(getattr(mae_cfg, "decoder_heads", 4))
        decoder_mlp_ratio = float(getattr(mae_cfg, "decoder_mlp_ratio", 2.0))
        self.decoder = RadarMAEDecoder(
            num_tokens=self.num_tokens,
            embed_dim=self.embed_dim,
            decoder_dim=decoder_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            patch_dim=self.embed_dim,
            decoder_mlp_ratio=decoder_mlp_ratio,
        )
        mid_channels = max(1, int(self.embed_dim // 2))
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(self.input_freq, self.input_time), mode="nearest"),
            nn.Conv2d(self.embed_dim, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
        )

    def _infer_backbone_grid(self) -> Tuple[int, int, int]:
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_freq, self.input_time)
            tokens, (grid_freq, grid_time) = self.backbone(dummy)
        embed_dim = int(tokens.shape[-1])
        grid_freq = int(grid_freq)
        grid_time = int(grid_time)
        if tokens.shape[1] != grid_freq * grid_time:
            raise ValueError(
                "Backbone token count does not match grid size: "
                f"{tokens.shape[1]} vs {grid_freq}*{grid_time}."
            )
        return embed_dim, grid_freq, grid_time

    def mask_tokens_to_input(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() != 2:
            mask = mask.view(mask.size(0), -1)
        B, N = mask.shape
        if N != self.num_tokens:
            raise ValueError(
                f"Mask has {N} tokens but backbone expects {self.num_tokens}."
            )
        mask_map = mask.view(B, 1, self.grid_freq, self.grid_time)
        mask_map = F.interpolate(
            mask_map.float(),
            size=(self.input_freq, self.input_time),
            mode="nearest",
        )
        return mask_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        dummy = x.new_zeros(B, self.num_tokens, 1)
        _, mask, ids_restore = random_masking(
            dummy,
            self.mask_ratio,
            freq_coords=self.token_freq_coords if self.mid_focus_prob > 0 else None,
            mid_band_ratio=self.mid_band_ratio,
            mid_focus_prob=self.mid_focus_prob,
        )
        mask_input = self.mask_tokens_to_input(mask)
        x_masked = x * (1.0 - mask_input)
        tokens, _ = self.backbone(x_masked)
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        len_keep = max(1, int(round(self.num_tokens * (1.0 - self.mask_ratio))))
        ids_keep = ids_shuffle[:, :len_keep]
        masked_tokens = torch.gather(
            tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        )
        decoded_tokens = self.decoder(masked_tokens, ids_restore)
        feat_map = decoded_tokens.transpose(1, 2).view(
            x.size(0), self.embed_dim, self.grid_freq, self.grid_time
        )
        recon = self.reconstruction_head(feat_map)
        return recon, mask


class MAETrainer:
    def __init__(
        self,
        model: RadarMAENetwork,
        train_loader,
        val_loader,
        args,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device

        self.epochs = int(args.train.epochs)
        self.base_lr = float(args.train.learning_rate)
        self.weight_decay = float(args.train.weight_decay)
        self.max_grad_norm = float(getattr(args.train, "max_grad_norm", 1.0))
        self.log_every = int(getattr(args.train, "log_every", 50))
        self.use_amp = bool(getattr(args.train, "use_amp", True)) and device.type == "cuda"

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        self.steps_per_epoch = max(1, len(self.train_loader))
        self.total_steps = self.steps_per_epoch * self.epochs
        self.current_step = 0

        mae_cfg = getattr(args, "mae", args)
        self.lambda_hf = max(0.0, float(getattr(mae_cfg, "hf_lambda", 3.0)))
        blur_kernel = int(getattr(mae_cfg, "blur_kernel", 5))
        blur_sigma = float(getattr(mae_cfg, "blur_sigma", 1.0))
        self.blur = GaussianBlur2d(kernel_size=blur_kernel, sigma=blur_sigma).to(self.device)

        self.save_every = max(1, int(getattr(args, "save_every", 10)))
        self.ckpt_dir = self._prepare_checkpoint_dir(getattr(args, "ckpt_dir", "checkpoints"))
        self.model_name = str(getattr(args, "model_name", None) or "mae")
        self.best_val = float("inf")
        self.serialized_args = self._serialize_args(args)

    def train(self) -> None:
        for epoch in range(self.epochs):
            train_losses = self._run_epoch(self.train_loader, epoch, training=True)
            val_losses = None
            if self.val_loader is not None and bool(getattr(self.args.result, "if_valid_set", True)):
                val_losses = self._run_epoch(self.val_loader, epoch, training=False)
            msg = (
                f"[Epoch {epoch+1:03d}] train_total={train_losses['total']:.4f} "
                f"(pix={train_losses['pixel']:.4f}, hf={train_losses['hf']:.4f})"
            )
            if val_losses is not None:
                msg += (
                    f" | val_total={val_losses['total']:.4f} "
                    f"(pix={val_losses['pixel']:.4f}, hf={val_losses['hf']:.4f})"
                )
            print(msg)
            self._handle_checkpoints(epoch + 1, val_losses["total"] if val_losses else None)
        final_path = os.path.join(self.ckpt_dir, f"mae_{self.model_name}_last.pt")
        self._save_checkpoint(self.epochs, final_path, verbose=True, reason="final model")

    def _run_epoch(self, loader, epoch_idx: int, training: bool) -> Dict[str, float]:
        metrics = {"total": [], "pixel": [], "hf": []}
        context = torch.enable_grad() if training else torch.no_grad()
        self.model.train(mode=training)
        with context:
            pbar = tqdm(loader, desc=("Train" if training else "Eval") + f" {epoch_idx:03d}", leave=False)
            for step_idx, (batch, _) in enumerate(pbar):
                batch = batch.to(self.device, non_blocking=True)
                with autocast('cuda', enabled=self.use_amp):
                    recon, mask_tokens = self.model(batch)
                    loss_mask = self.model.mask_tokens_to_input(mask_tokens)
                    pix_loss = self._masked_l1_loss(recon, batch, loss_mask)
                    hf_loss = (
                        self._high_freq_loss(recon, batch, loss_mask)
                        if self.lambda_hf > 0
                        else batch.new_zeros(())
                    )
                    loss = pix_loss + self.lambda_hf * hf_loss
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.current_step += 1
                metrics["total"].append(float(loss.item()))
                metrics["pixel"].append(float(pix_loss.item()))
                metrics["hf"].append(float(hf_loss.item()))
                if training and (step_idx + 1) % max(1, self.log_every) == 0:
                    pbar.set_postfix(total=f"{metrics['total'][-1]:.4f}", pix=f"{metrics['pixel'][-1]:.4f}")
        means = {key: float(sum(values) / max(1, len(values))) for key, values in metrics.items()}
        means["_raw_total"] = metrics["total"]
        return means

    def _high_freq_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_blur = self.blur(target)
        recon_blur = self.blur(recon)
        target_residual = target - target_blur
        recon_residual = recon - recon_blur
        if mask is not None:
            return self._masked_l1_loss(recon_residual, target_residual, mask)
        return F.l1_loss(recon_residual, target_residual)

    @staticmethod
    def _masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return F.l1_loss(pred, target)
        if mask.shape != pred.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match tensor shape {pred.shape}."
            )
        mask_float = mask.to(pred.dtype)
        denom = mask_float.sum()
        if denom <= 0:
            return F.l1_loss(pred, target)
        return (pred - target).abs().mul(mask_float).sum() / denom

    def _prepare_checkpoint_dir(self, root_dir: str) -> str:
        resolved_root = os.path.abspath(os.path.expanduser(str(root_dir)))
        run_cfg = getattr(self.args, "run", None)
        task_name = getattr(run_cfg, "task", None) if run_cfg is not None else None
        ckpt_dir = os.path.join(resolved_root, task_name) if task_name else resolved_root
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def _handle_checkpoints(self, epoch_idx: int, val_loss: Optional[float]) -> None:
        if epoch_idx % self.save_every == 0:
            periodic = os.path.join(self.ckpt_dir, f"mae_{self.model_name}_epoch{epoch_idx:03d}.pt")
            self._save_checkpoint(epoch_idx, periodic, verbose=True, reason="periodic save")
        if val_loss is not None and val_loss < self.best_val:
            self.best_val = val_loss
            best_path = os.path.join(self.ckpt_dir, f"mae_{self.model_name}_best.pt")
            self._save_checkpoint(epoch_idx, best_path, verbose=True, reason="best val")

    def _save_checkpoint(self, epoch_idx: int, path: str, verbose: bool = False, reason: Optional[str] = None) -> None:
        ckpt = {
            "epoch": epoch_idx,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "current_step": self.current_step,
            "best_val": self.best_val,
            "args": self.serialized_args,
        }
        torch.save(ckpt, path)
        if verbose:
            msg = f"  -> Saved checkpoint to {path}"
            if reason:
                msg += f" ({reason})"
            print(msg)

    @staticmethod
    def _serialize_args(args_cfg: Any) -> Any:
        try:
            return OmegaConf.to_container(args_cfg, resolve=True)
        except Exception:
            return args_cfg
