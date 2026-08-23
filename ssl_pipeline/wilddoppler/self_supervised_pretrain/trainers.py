from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .models import build_backbone


class LabelEncoder:
    def __init__(self) -> None:
        self.lookup: dict[str, int] = {}

    def encode(self, values) -> torch.Tensor:
        ids = []
        for value in values:
            key = str(value)
            if key not in self.lookup:
                self.lookup[key] = len(self.lookup)
            ids.append(self.lookup[key])
        return torch.tensor(ids, dtype=torch.long)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = max(1e-6, float(temperature))

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        logits = z @ z.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # numerical stability
        labels = labels.view(-1, 1)
        pos = (labels == labels.T).float().to(z.device)
        eye = torch.eye(pos.size(0), device=z.device)
        pos = pos * (1.0 - eye)
        exp_logits = torch.exp(logits) * (1.0 - eye)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        valid = pos.sum(dim=1) > 0
        if not torch.any(valid):
            return z.new_zeros(())
        return -((pos * log_prob).sum(dim=1) / (pos.sum(dim=1) + 1e-12))[valid].mean()


class AugmentedContrastiveForward(nn.Module):
    """Run contrastive augmentations on each DataParallel shard before encoding."""

    def __init__(self, model: nn.Module, augmenter) -> None:
        super().__init__()
        self.model = model
        self.augmenter = augmenter

    @property
    def num_views(self) -> int:
        return int(self.augmenter.num_global_views + self.augmenter.num_local_views)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        views = self.augmenter(batch)
        return torch.cat([self.model(view) for view in views], dim=0)


class ContrastiveTrainer:
    def __init__(self, model, augmenter, train_loader, val_loader, args, device: torch.device) -> None:
        self.is_data_parallel = isinstance(model, nn.DataParallel)
        if self.is_data_parallel:
            self.trainable_model = model.module
            self.model = nn.DataParallel(
                AugmentedContrastiveForward(model.module, augmenter),
                device_ids=model.device_ids,
                output_device=model.output_device,
            )
        else:
            self.trainable_model = model
            self.model = model
        self.augmenter = augmenter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.loss_fn = SupConLoss(args.temperature)
        self.encoder = LabelEncoder()
        self.use_amp = bool(args.use_amp) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        backbone_params, head_params = [], []
        for name, param in self.trainable_model.named_parameters():
            if not param.requires_grad:
                continue
            canonical_name = name.removeprefix("module.")
            (backbone_params if canonical_name.startswith("backbone.") else head_params).append(param)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.backbone_learning_rate},
                {"params": head_params, "lr": args.head_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        os.makedirs(args.ckpt_dir, exist_ok=True)
        self.best_val = float("inf")

    def train(self) -> None:
        for epoch in range(self.args.epochs):
            train_loss = self._run_epoch(self.train_loader, epoch, training=True)
            val_loss = self._run_epoch(self.val_loader, epoch, training=False) if self.val_loader is not None else None
            msg = f"[Epoch {epoch + 1:03d}] train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f}"
            print(msg)
            if (epoch + 1) % self.args.save_every == 0:
                self._save(epoch + 1, f"contrastive_{self.args.model_name}_epoch{epoch + 1:03d}.pt")
            if val_loss is not None and val_loss < self.best_val:
                self.best_val = val_loss
                self._save(epoch + 1, f"contrastive_{self.args.model_name}_best.pt")
        self._save(self.args.epochs, f"contrastive_{self.args.model_name}_last.pt")

    def _run_epoch(self, loader, epoch: int, training: bool) -> float:
        self.model.train(mode=training)
        losses = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            pbar = tqdm(loader, desc=("Train" if training else "Eval") + f" {epoch:03d}", leave=False)
            for step, (batch, meta) in enumerate(pbar):
                if training and getattr(self.args, "max_train_steps", 0) and step >= int(self.args.max_train_steps):
                    break
                batch = batch.to(self.device, non_blocking=True)
                base_labels = self.encoder.encode(meta[self.args.label_key]).to(self.device)
                if self.is_data_parallel:
                    num_views = int(self.augmenter.num_global_views + self.augmenter.num_local_views)
                    labels = torch.cat(
                        [chunk.repeat(num_views) for chunk in torch.chunk(base_labels, len(self.model.device_ids), dim=0)],
                        dim=0,
                    )
                else:
                    views = [view.to(self.device, non_blocking=True) for view in self.augmenter(batch)]
                    labels = base_labels.repeat(len(views))
                with autocast(enabled=self.use_amp):
                    if self.is_data_parallel:
                        z = self.model(batch)
                    else:
                        z = torch.cat([self.model(view) for view in views], dim=0)
                    loss = self.loss_fn(z, labels)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.args.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.trainable_model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                losses.append(float(loss.item()))
                if training and (step + 1) % self.args.log_every == 0:
                    pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        return float(np.mean(losses)) if losses else 0.0

    def _save(self, epoch: int, name: str) -> None:
        path = os.path.join(self.args.ckpt_dir, name)
        model_state = self.trainable_model.state_dict()
        torch.save(
            {
                "epoch": epoch,
                "model_state": model_state,
                "optimizer_state": self.optimizer.state_dict(),
                "args": vars(self.args),
                "label_encoder": self.encoder.lookup,
                "best_val": self.best_val,
            },
            path,
        )
        print(f"  -> Saved checkpoint to {path}")


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return tensor.normal_(0.0, std).clamp_(-2 * std, 2 * std)


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
    freq_coords: Optional[torch.Tensor] = None,
    mid_band_ratio: tuple[float, float] = (0.2, 0.8),
    mid_focus_prob: float = 0.0,
):
    b, n, d = x.shape
    keep = max(1, int(round(n * (1.0 - mask_ratio))))
    mask_count = n - keep
    # mid_focus_prob biases masking toward middle Doppler frequencies (torso returns) to avoid trivial reconstruction.
    if freq_coords is not None and mid_focus_prob > 0 and mask_count > 0:
        freq_coords = freq_coords.to(x.device)
        keep_ids = []
        masks = []
        restore_ids = []
        lo, hi = mid_band_ratio
        mid_focus_prob = max(0.0, min(1.0, float(mid_focus_prob)))
        for _ in range(b):
            mid = torch.nonzero((freq_coords >= lo) & (freq_coords <= hi), as_tuple=False).flatten()
            selected = torch.zeros(n, dtype=torch.bool, device=x.device)
            mid_count = min(mid.numel(), int(round(mask_count * mid_focus_prob)))
            if mid_count > 0:
                selected[mid[torch.randperm(mid.numel(), device=x.device)[:mid_count]]] = True
            remaining = mask_count - int(selected.sum().item())
            if remaining > 0:
                pool = torch.nonzero(~selected, as_tuple=False).flatten()
                selected[pool[torch.randperm(pool.numel(), device=x.device)[:remaining]]] = True
            keep_idx = torch.nonzero(~selected, as_tuple=False).flatten()
            mask_idx = torch.nonzero(selected, as_tuple=False).flatten()
            keep_idx = keep_idx[torch.randperm(keep_idx.numel(), device=x.device)]
            mask_idx = mask_idx[torch.randperm(mask_idx.numel(), device=x.device)]
            shuffle = torch.cat([keep_idx, mask_idx], dim=0)
            restore = torch.empty(n, dtype=torch.long, device=x.device)
            restore[shuffle] = torch.arange(n, device=x.device)
            mask = selected.float()
            keep_ids.append(keep_idx)
            masks.append(mask)
            restore_ids.append(restore)
        ids_keep = torch.stack(keep_ids, dim=0)
        mask = torch.stack(masks, dim=0)
        ids_restore = torch.stack(restore_ids, dim=0)
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        return x_masked, mask, ids_restore
    noise = torch.rand(b, n, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :keep]
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
    mask = torch.ones(b, n, device=x.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_masked, mask, ids_restore


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, max(1, heads), batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ReconstructionDecoder(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int, decoder_dim: int, decoder_layers: int, decoder_heads: int):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, decoder_dim))
        self.blocks = nn.ModuleList([TransformerBlock(decoder_dim, decoder_heads, 1.5) for _ in range(decoder_layers)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, embed_dim)
        trunc_normal_(self.mask_token)
        trunc_normal_(self.pos_embed)

    def forward(self, encoded: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(encoded)
        b, visible, d = x.shape
        n = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(b, n - visible, d)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d))
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.pred(self.norm(x))


class BackboneReconstruction(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.input_freq = int(args.resize_doppler)
        self.input_time = int(round(args.crop_seconds * args.bins_per_second))
        self.mask_ratio = min(0.99, max(0.0, float(args.mask_ratio)))
        self.mid_focus_prob = max(0.0, min(1.0, float(args.mid_focus_prob)))
        self.mid_band_ratio = (float(args.mid_band_ratio[0]), float(args.mid_band_ratio[1]))
        self.backbone = build_backbone(
            model_name=args.model_name,
            embed_dim=args.embed_dim,
            stem_channels=args.stem_channels,
            use_radar_stem=args.use_radar_stem,
            pretrained_imagenet=args.pretrained_imagenet,
        )
        with torch.no_grad():
            tokens, (grid_f, grid_t) = self.backbone(torch.zeros(1, 1, self.input_freq, self.input_time))
        self.embed_dim = int(tokens.shape[-1])
        self.grid_freq = int(grid_f)
        self.grid_time = int(grid_t)
        self.num_tokens = self.grid_freq * self.grid_time
        freq_coords = torch.linspace(0.0, 1.0, steps=self.grid_freq).repeat_interleave(self.grid_time)
        self.register_buffer("token_freq_coords", freq_coords, persistent=False)
        self.decoder = ReconstructionDecoder(
            self.num_tokens,
            self.embed_dim,
            args.decoder_dim,
            args.decoder_layers,
            args.decoder_heads,
        )
        mid = max(1, self.embed_dim // 2)
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(self.input_freq, self.input_time), mode="nearest"),
            nn.Conv2d(self.embed_dim, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1),
        )

    def mask_tokens_to_input(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.view(mask.size(0), 1, self.grid_freq, self.grid_time)
        return F.interpolate(mask.float(), size=(self.input_freq, self.input_time), mode="nearest")

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        dummy = x.new_zeros(b, self.num_tokens, 1)
        _, mask, ids_restore = random_masking(
            dummy,
            self.mask_ratio,
            freq_coords=self.token_freq_coords,
            mid_band_ratio=self.mid_band_ratio,
            mid_focus_prob=self.mid_focus_prob,
        )
        # Masking applied in pixel space (not token space) so the backbone never sees masked regions.
        x_masked = x * (1.0 - self.mask_tokens_to_input(mask))
        tokens, _ = self.backbone(x_masked)
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        keep = max(1, int(round(self.num_tokens * (1.0 - self.mask_ratio))))
        ids_keep = ids_shuffle[:, :keep]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        decoded = self.decoder(visible, ids_restore)
        feat_map = decoded.transpose(1, 2).view(b, self.embed_dim, self.grid_freq, self.grid_time)
        return self.reconstruction_head(feat_map), mask


class ReconstructionTrainer:
    def __init__(
        self,
        model: BackboneReconstruction,
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
        self.use_amp = bool(args.use_amp) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.best_val = float("inf")
        os.makedirs(args.ckpt_dir, exist_ok=True)

    def train(self) -> None:
        for epoch in range(self.args.epochs):
            train = self._run_epoch(self.train_loader, epoch, training=True)
            val = self._run_epoch(self.val_loader, epoch, training=False) if self.val_loader is not None else None
            msg = f"[Epoch {epoch + 1:03d}] train_loss={train:.4f}"
            if val is not None:
                msg += f" val_loss={val:.4f}"
            print(msg)
            if (epoch + 1) % self.args.save_every == 0:
                self._save(epoch + 1, f"reconstruction_{self.args.model_name}_epoch{epoch + 1:03d}.pt")
            if val is not None and val < self.best_val:
                self.best_val = val
                self._save(epoch + 1, f"reconstruction_{self.args.model_name}_best.pt")
        self._save(self.args.epochs, f"reconstruction_{self.args.model_name}_last.pt")

    def _run_epoch(self, loader, epoch: int, training: bool) -> float:
        self.model.train(mode=training)
        losses = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            pbar = tqdm(loader, desc=("Train" if training else "Eval") + f" {epoch:03d}", leave=False)
            for step, (batch, _) in enumerate(pbar):
                batch = batch.to(self.device, non_blocking=True)
                with autocast(enabled=self.use_amp):
                    recon, mask = self.model(batch)
                    loss_mask = self.model.mask_tokens_to_input(mask)
                    loss = self._masked_l1(recon, batch, loss_mask)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.args.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                losses.append(float(loss.item()))
                if training and (step + 1) % self.args.log_every == 0:
                    pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        return float(np.mean(losses)) if losses else 0.0

    @staticmethod
    def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None or mask.sum() <= 0:
            return F.l1_loss(pred, target)
        return ((pred - target).abs() * mask).sum() / mask.sum()

    def _save(self, epoch: int, name: str) -> None:
        path = os.path.join(self.args.ckpt_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "args": vars(self.args),
                "best_val": self.best_val,
            },
            path,
        )
        print(f"  -> Saved checkpoint to {path}")
