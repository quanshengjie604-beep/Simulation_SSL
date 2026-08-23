import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .pooling import AttentionCLSPooling


def _normalize_condition_cfg(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    try:
        return dict(cfg)
    except Exception:
        return {}


class ContrastiveProjectionHead(nn.Module):
    """
    Simple MLP projection head that outputs L2-normalized embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = in_dim if hidden_dim is None else int(hidden_dim)
        layers: List[nn.Module] = []
        dim_in = in_dim
        for layer_idx in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_raw: bool = False):
        proj = self.net(x)
        if return_raw:
            normalized = F.normalize(proj, dim=-1, eps=1e-6)
            return normalized, proj
        return F.normalize(proj, dim=-1, eps=1e-6)


class RadarContrastiveModel(nn.Module):
    """
    Backbone + pooling + two contrastive heads (identity & activity).
    """

    def __init__(
        self,
        backbone: nn.Module,
        pool_num_heads: int = 4,
        pool_dropout: float = 0.0,
        id_head_cfg: Optional[dict] = None,
        act_head_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if not hasattr(backbone, "embed_dim"):
            raise ValueError("Backbone must expose 'embed_dim' for contrastive training.")
        embed_dim = int(backbone.embed_dim)
        self.pool = AttentionCLSPooling(embed_dim=embed_dim, num_heads=pool_num_heads, dropout=pool_dropout)
        id_cfg = dict(id_head_cfg or {})
        act_cfg = dict(act_head_cfg or {})
        id_out = int(id_cfg.pop("out_dim", 256))
        act_out = int(act_cfg.pop("out_dim", 256))
        self.head_id = ContrastiveProjectionHead(
            in_dim=embed_dim,
            out_dim=id_out,
            hidden_dim=id_cfg.pop("hidden_dim", embed_dim),
            num_layers=int(id_cfg.pop("num_layers", 2)),
            dropout=float(id_cfg.pop("dropout", 0.0)),
        )
        self.head_act = ContrastiveProjectionHead(
            in_dim=embed_dim,
            out_dim=act_out,
            hidden_dim=act_cfg.pop("hidden_dim", embed_dim),
            num_layers=int(act_cfg.pop("num_layers", 2)),
            dropout=float(act_cfg.pop("dropout", 0.0)),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        tokens, _ = self.backbone(x, cond=cond)
        pooled = self.pool(tokens)
        return {
            "tokens": tokens,
            "pooled": pooled,
            "z_id": self.head_id(pooled),
            "z_act": self.head_act(pooled),
        }


class SupervisedContrastiveLoss(nn.Module):
    """
    Multi-positive InfoNCE loss using class labels to define positives.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = max(1e-6, float(temperature))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError(
                f"Expected features [N, D] and labels [N], got {features.shape} and {labels.shape}"
            )
        if features.size(0) != labels.size(0):
            raise ValueError("Feature and label count mismatch for contrastive loss.")
        device = features.device
        feats = F.normalize(features, dim=-1)
        logits = torch.matmul(feats, feats.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).to(device=device, dtype=torch.float32)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mask_sum = mask.sum(dim=1)
        valid = mask_sum > 0
        if not torch.any(valid):
            return None
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask_sum + 1e-12)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class LabelEncoder:
    """
    Maps arbitrary string/integer identifiers to contiguous integer IDs.
    """

    def __init__(self) -> None:
        self.lookup: Dict[Any, int] = {}
        self.next_id = 0

    def encode(self, values: Sequence[Any]) -> torch.Tensor:
        encoded: List[int] = []
        for value in values:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError("Label tensors must be scalar.")
                value = value.item()
            if isinstance(value, (int, float)):
                encoded.append(int(value))
                continue
            key = str(value)
            if key not in self.lookup:
                self.lookup[key] = self.next_id
                self.next_id += 1
            encoded.append(self.lookup[key])
        return torch.tensor(encoded, dtype=torch.long)


class ContrastiveTrainer:
    """
    Handles the dual-head contrastive training loop.
    """

    def __init__(
        self,
        model: RadarContrastiveModel,
        augmenter,
        train_loader,
        val_loader,
        args,
        device: torch.device,
    ) -> None:
        self.model = model
        self.augmenter = augmenter
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

        contrastive_cfg = getattr(args, "contrastive", args)
        self.lambda_act = max(0.0, float(getattr(contrastive_cfg, "lambda_act", 1.0)))
        temp_id = float(getattr(contrastive_cfg, "temperature_id", 0.1))
        temp_act = float(getattr(contrastive_cfg, "temperature_act", temp_id))
        self.id_loss_fn = SupervisedContrastiveLoss(temp_id)
        self.act_loss_fn = SupervisedContrastiveLoss(temp_act)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.steps_per_epoch = max(1, len(self.train_loader))
        self.total_steps = self.steps_per_epoch * self.epochs
        self.current_step = 0

        self.save_every = max(1, int(getattr(args, "save_every", 10)))
        self.ckpt_dir = self._prepare_checkpoint_dir(getattr(args, "ckpt_dir", "checkpoints"))
        self.model_name = str(getattr(args, "model_name", None) or "model")
        self.best_val_loss = float("inf")
        self.serialized_args = self._serialize_args(args)
        self.condition_cfg = _normalize_condition_cfg(getattr(args, "conditioning", None))
        self.conditioning_enabled = bool(self.condition_cfg.get("enabled", False))
        self.condition_feature_dim = 0
        if self.conditioning_enabled:
            feature_cols = self.condition_cfg.get("feature_columns", [])
            default_dim = len(feature_cols) if isinstance(feature_cols, (list, tuple)) else 0
            self.condition_feature_dim = int(self.condition_cfg.get("condition_dim", default_dim))
            if self.condition_feature_dim <= 0:
                raise ValueError("Conditioning enabled but no feature columns were provided.")
        if self.condition_feature_dim > 0:
            feature_cols = self.condition_cfg.get("feature_columns", [])
            if not feature_cols or len(feature_cols) != self.condition_feature_dim:
                feature_cols = [f"cond_{idx}" for idx in range(self.condition_feature_dim)]
            self.condition_feature_names = list(feature_cols)
        else:
            self.condition_feature_names = []
        self.log_condition_metrics = bool(self.condition_cfg.get("log_condition_metrics", True))

        self.track_encoder = LabelEncoder()

    def train(self) -> None:
        for epoch in range(self.epochs):
            train_total, train_id, train_act, train_cond, train_cond_ratio = self._run_epoch(
                self.train_loader, epoch, training=True
            )
            val_total = val_id = val_act = val_cond = val_cond_ratio = None
            if self.val_loader is not None:
                val_total, val_id, val_act, val_cond, val_cond_ratio = self._run_epoch(
                    self.val_loader, epoch, training=False
                )
            msg = (
                f"[Epoch {epoch+1:03d}] train_loss={train_total:.4f} "
                f"(id={train_id:.4f}"
            )
            if train_act is not None:
                msg += f", act={train_act:.4f}"
            msg += ")"
            if train_cond_ratio is not None:
                msg += f", cond_avail={train_cond_ratio:.2f}"
            if val_total is not None:
                msg += f" | val_loss={val_total:.4f} (id={val_id:.4f}"
                if val_act is not None:
                    msg += f", act={val_act:.4f}"
                msg += ")"
                if val_cond_ratio is not None:
                    msg += f", val_cond_avail={val_cond_ratio:.2f}"
            print(msg)
            self._handle_checkpoints(epoch + 1, val_total)
        final_path = os.path.join(self.ckpt_dir, "contrastive_last.pt")
        self._save_checkpoint(self.epochs, final_path, verbose=True, reason="final model")

    def _run_epoch(
        self,
        loader,
        epoch_idx: int,
        training: bool,
    ) -> Tuple[float, float, Optional[float], Optional[List[float]], Optional[float]]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_losses: List[float] = []
        id_losses: List[float] = []
        act_losses: List[float] = []
        condition_means: List[torch.Tensor] = []
        condition_presence: List[float] = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            pbar = tqdm(loader, desc=("Train" if training else "Eval") + f" {epoch_idx:03d}", leave=False)
            for step_idx, (batch, meta) in enumerate(pbar):
                batch = batch.to(self.device, non_blocking=True)
                track_labels = self._build_track_labels(meta, batch.size(0))
                window_labels = torch.arange(batch.size(0), device=self.device, dtype=torch.long)
                condition_batch = self._build_condition_batch(meta, batch.size(0))
                if condition_batch is not None:
                    condition_means.append(condition_batch.detach().mean(dim=0).cpu())
                    presence = self._condition_presence(meta, batch.size(0))
                    condition_presence.append(presence)
                views = self.augmenter(batch)
                view_tensors = [view.to(self.device, non_blocking=True) for view in views]

                with autocast(enabled=self.use_amp):
                    id_loss, act_loss = self._compute_losses(
                        view_tensors,
                        condition_batch,
                        track_labels,
                        window_labels,
                    )
                    loss = id_loss
                    if act_loss is not None and self.lambda_act > 0:
                        loss = loss + self.lambda_act * act_loss

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.current_step += 1

                total_losses.append(float(loss.item()))
                id_losses.append(float(id_loss.item()))
                if act_loss is not None:
                    act_losses.append(float(act_loss.item()))
                if training and (step_idx + 1) % max(1, self.log_every) == 0:
                    postfix = {"loss": f"{total_losses[-1]:.4f}", "id": f"{id_losses[-1]:.4f}"}
                    if act_loss is not None:
                        postfix["act"] = f"{act_losses[-1]:.4f}"
                    pbar.set_postfix(**postfix)
        avg_total = float(sum(total_losses) / max(1, len(total_losses)))
        avg_id = float(sum(id_losses) / max(1, len(id_losses)))
        avg_act = float(sum(act_losses) / len(act_losses)) if act_losses else None
        avg_cond = None
        avg_presence = None
        if condition_means:
            stacked = torch.stack(condition_means, dim=0)
            avg_cond = stacked.mean(dim=0).tolist()
            if condition_presence:
                avg_presence = float(sum(condition_presence) / len(condition_presence))
        return avg_total, avg_id, avg_act, avg_cond, avg_presence

    def _compute_losses(
        self,
        views: Sequence[torch.Tensor],
        condition_batch: Optional[torch.Tensor],
        track_labels: torch.Tensor,
        window_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        id_features: List[torch.Tensor] = []
        id_targets: List[torch.Tensor] = []
        act_features: List[torch.Tensor] = []
        act_targets: List[torch.Tensor] = []
        num_global = getattr(self.augmenter, "num_global_views", len(views))
        for view_idx, view in enumerate(views):
            outputs = self.model(view, cond=condition_batch)
            id_features.append(outputs["z_id"])
            id_targets.append(track_labels)
            if view_idx >= num_global and self.lambda_act > 0:
                act_features.append(outputs["z_act"])
                act_targets.append(window_labels)
        id_feat = torch.cat(id_features, dim=0)
        id_target = torch.cat(id_targets, dim=0)
        id_loss = self.id_loss_fn(id_feat, id_target)
        act_loss = None
        if act_features and self.lambda_act > 0:
            act_feat = torch.cat(act_features, dim=0)
            act_target = torch.cat(act_targets, dim=0)
            act_loss = self.act_loss_fn(act_feat, act_target)
        if id_loss is None:
            raise RuntimeError("Identity contrastive loss returned None — check augmentation config.")
        return id_loss, act_loss

    def _build_track_labels(self, meta: Dict[str, Any], batch_size: int) -> torch.Tensor:
        if not isinstance(meta, dict):
            raise ValueError("Expected metadata dictionary from dataloader.")
        track_values = None
        if "track_id" in meta:
            track_values = self._meta_to_list(meta["track_id"])
            file_values = None
            if "file_name" in meta:
                file_values = self._meta_to_list(meta["file_name"])
            if file_values is not None and len(file_values) == len(track_values):
                combined = [f"{fname}_track_{tid}" for fname, tid in zip(file_values, track_values)]
                encoded = self.track_encoder.encode(combined)
                return encoded.to(self.device)
            encoded = self.track_encoder.encode(track_values)
            return encoded.to(self.device)
        for candidate in ("global_id", "target_id"):
            if candidate in meta:
                values = self._meta_to_list(meta[candidate])
                encoded = self.track_encoder.encode(values)
                return encoded.to(self.device)
        if "filename_id" in meta:
            values = self._meta_to_list(meta["filename_id"])
            encoded = self.track_encoder.encode(values)
            return encoded.to(self.device)
        # Fallback: treat each sample as its own identity
        return torch.arange(batch_size, device=self.device, dtype=torch.long)

    def _build_condition_batch(self, meta: Dict[str, Any], batch_size: int) -> Optional[torch.Tensor]:
        if not self.conditioning_enabled or self.condition_feature_dim <= 0:
            return None
        if not isinstance(meta, dict):
            return None
        if "condition_vec" not in meta:
            return None
        cond_values = meta["condition_vec"]
        if isinstance(cond_values, torch.Tensor):
            cond_tensor = cond_values
        elif isinstance(cond_values, (list, tuple)):
            cond_tensor = torch.stack(
                [torch.as_tensor(v, dtype=torch.float32) for v in cond_values],
                dim=0,
            )
        else:
            cond_tensor = torch.as_tensor(cond_values, dtype=torch.float32)
        cond_tensor = cond_tensor.to(self.device, dtype=torch.float32)
        if cond_tensor.ndim == 1:
            cond_tensor = cond_tensor.unsqueeze(0).expand(batch_size, -1)
        if cond_tensor.size(0) != batch_size:
            cond_tensor = cond_tensor.view(batch_size, -1)
        return cond_tensor

    def _condition_presence(self, meta: Dict[str, Any], batch_size: int) -> float:
        if not isinstance(meta, dict):
            return 0.0
        has_cond = meta.get("has_condition")
        if has_cond is None:
            return 0.0
        values = self._meta_to_list(has_cond)
        if not values:
            return 0.0
        positives = sum(1 for v in values if bool(v))
        total = len(values)
        return positives / max(1, total)

    @staticmethod
    def _meta_to_list(value: Any) -> List[Any]:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().view(-1).tolist()
        if isinstance(value, (list, tuple)):
            result: List[Any] = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    if item.numel() == 1:
                        result.append(item.item())
                    else:
                        result.extend(item.detach().cpu().view(-1).tolist())
                else:
                    result.append(item)
            return result
        return [value]

    def _prepare_checkpoint_dir(self, root_dir: str) -> str:
        resolved_root = os.path.abspath(os.path.expanduser(str(root_dir)))
        task_dir = resolved_root
        run_cfg = getattr(self.args, "run", None)
        task_name = getattr(run_cfg, "task", None) if run_cfg is not None else None
        if task_name:
            task_dir = os.path.join(resolved_root, task_name)
        os.makedirs(task_dir, exist_ok=True)
        return task_dir

    def _handle_checkpoints(self, epoch_idx: int, val_loss: Optional[float]) -> None:
        if epoch_idx % self.save_every == 0:
            periodic_path = os.path.join(self.ckpt_dir, f"contrastive_{self.model_name}_epoch{epoch_idx:03d}.pt")
            self._save_checkpoint(epoch_idx, periodic_path, verbose=True, reason="periodic save")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.ckpt_dir, f"contrastive_{self.model_name}_best.pt")
            self._save_checkpoint(epoch_idx, best_path, verbose=True, reason="best val")

    def _save_checkpoint(self, epoch_idx: int, path: str, verbose: bool = False, reason: Optional[str] = None) -> None:
        ckpt = {
            "epoch": epoch_idx,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "current_step": self.current_step,
            "best_val_loss": self.best_val_loss,
            "args": self.serialized_args,
        }
        torch.save(ckpt, path)
        if verbose:
            msg = f"  -> Saved checkpoint to {path}"
            if reason:
                msg += f" ({reason})"
            print(msg)

    def _serialize_args(self, args_cfg: Any) -> Any:
        try:
            return OmegaConf.to_container(args_cfg, resolve=True)
        except Exception:
            return args_cfg
