"""Training loop and evaluation helpers for supervised load models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .constants import is_regression_task
from .metrics import (
    classification_metrics,
    compute_auroc,
    compute_auprc,
    regression_metrics,
    save_predictions_csv,
    save_regression_csv,
)


@dataclass
class EpochStats:
    loss: float
    acc: float | None
    macro_f1: float | None = None
    balanced_acc: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    auroc: float | None = None
    auprc: float | None = None
    mae_speed: float | None = None
    mae_angle: float | None = None
    mae_radial: float | None = None
    mae_lateral: float | None = None


class SupervisedTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        test_loader,
        cfg,
        device: torch.device,
        output_dir: Path,
        test_fold: Optional[int] = None,
        fold_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.cfg = cfg
        self.device = device

        train_cfg = cfg.train
        self.epochs = int(train_cfg.epochs)
        self.base_lr = float(train_cfg.learning_rate)
        self.weight_decay = float(getattr(train_cfg, "weight_decay", 0.0))
        self.max_grad_norm = float(getattr(train_cfg, "max_grad_norm", 1.0))
        self.use_amp = bool(getattr(train_cfg, "use_amp", True)) and device.type == "cuda"
        self.save_every = int(getattr(train_cfg, "save_every", 10))
        self.lr_scheduler_cfg = getattr(train_cfg, "lr_scheduler", None)

        self.is_regression = is_regression_task(getattr(cfg, "task_name", ""))
        self.label_mean = float(getattr(cfg.train, "label_mean", 0.0))
        self.label_std = float(getattr(cfg.train, "label_std", 1.0))
        if self.is_regression:
            self._unit_polar_eps = float(getattr(train_cfg, "unit_polar_eps", 1e-8))
            self._unit_polar_lambda_s = float(getattr(train_cfg, "unit_polar_lambda_s", 1.0))
            self._unit_polar_lambda_d = float(getattr(train_cfg, "unit_polar_lambda_d", 1.0))

        self.criterion = None if self.is_regression else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.test_fold = test_fold
        self.fold_name = fold_name

        self.exp_name = getattr(getattr(cfg, "paths", None), "exp_name", None)

    def _build_scheduler(self):
        if not self.lr_scheduler_cfg:
            return None
        name = str(getattr(self.lr_scheduler_cfg, "name", "none")).lower()
        if name in {"none", "null", "off"}:
            return None
        if name == "cosine":
            t_max = int(getattr(self.lr_scheduler_cfg, "t_max", self.epochs))
            eta_min = float(getattr(self.lr_scheduler_cfg, "min_lr", 0.0))
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        elif name == "cosine_warm_restarts":
            eta_min = float(getattr(self.lr_scheduler_cfg, "min_lr", 0.0))
            T_0 = int(getattr(self.lr_scheduler_cfg, "t_0", 25))
            T_mult = int(getattr(self.lr_scheduler_cfg, "t_mult", 1))
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                        T_0 = T_0,# Number of iterations for the first restart
                                                                        T_mult = T_mult, # A factor increases TiTi​ after a restart
                                                                        eta_min =eta_min) # Minimum learning rate
        raise ValueError(f"Unsupported lr scheduler: {name}")

    def _forward_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, _ = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32 if self.is_regression else torch.long)
        return x, y

    def _regression_target_names(self) -> list[str]:
        return ["v_radial", "v_lateral"]

    def _compute_unit_polar_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3] = [a, b, s_norm].  y: [B, 2] = [v_r, v_l].
        eps = self._unit_polar_eps
        v_r = y[:, 0]
        v_l = y[:, 1]
        s_true = torch.sqrt(v_r ** 2 + v_l ** 2)
        cos_theta = v_r / (s_true + eps)
        sin_theta = v_l / (s_true + eps)
        s_norm_true = (s_true - self.label_mean) / self.label_std

        a_pred = logits[:, 0]
        b_pred = logits[:, 1]
        s_norm_pred = logits[:, 2]

        denom = torch.sqrt(a_pred ** 2 + b_pred ** 2 + eps)
        u_r = a_pred / denom
        u_l = b_pred / denom

        loss_s = ((s_norm_pred - s_norm_true) ** 2).mean()
        loss_d = ((u_r - cos_theta) ** 2 + (u_l - sin_theta) ** 2).mean()
        return self._unit_polar_lambda_s * loss_s + self._unit_polar_lambda_d * loss_d

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.is_regression:
            return self._compute_unit_polar_loss(logits, y)
        return self.criterion(logits, y)

    def _unit_polar_preds_to_velocity(self, preds: np.ndarray) -> np.ndarray:
        # preds: [N, 3] = [a, b, speed_orig].  Returns [N, 2] = [pred_v_r, pred_v_l].
        eps = self._unit_polar_eps
        a = preds[:, 0]
        b = preds[:, 1]
        denom = np.sqrt(a ** 2 + b ** 2 + eps)
        u_r = a / denom
        u_l = b / denom
        speed = preds[:, 2]
        return np.stack([speed * u_r, speed * u_l], axis=1)

    def _run_epoch(self, loader, train: bool = True, compute_metrics: bool = False) -> EpochStats:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        pred_batches = []
        label_batches = []
        prob_batches = []

        for batch in loader:
            x, y = self._forward_batch(batch)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(x)
                    loss = self._compute_loss(logits, y)
                self.scaler.scale(loss).backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad():
                    logits = self.model(x)
                    loss = self._compute_loss(logits, y)

            batch_size = y.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            if self.is_regression:
                if compute_metrics:
                    label_batches.append(y.detach().cpu())
                    pred_batches.append(logits.detach().cpu())
            else:
                pred_batch = logits.argmax(dim=1)
                total_correct += int((pred_batch == y).sum().item())
                if compute_metrics:
                    label_batches.append(y.detach().cpu())
                    pred_batches.append(pred_batch.detach().cpu())
                    prob_batches.append(torch.softmax(logits, dim=1).detach().cpu())

        mean_loss = total_loss / max(1, total_samples)
        mean_acc = None if self.is_regression else total_correct / max(1, total_samples)
        stats = EpochStats(loss=mean_loss, acc=mean_acc)
        if compute_metrics:
            if label_batches:
                labels_arr = torch.cat(label_batches).numpy()
                preds_arr = torch.cat(pred_batches).numpy()
                probs_arr = torch.cat(prob_batches).numpy() if prob_batches else np.array([])
            else:
                labels_arr = np.array([])
                preds_arr = np.array([])
                probs_arr = np.array([])

            if self.is_regression:
                preds_eval = preds_arr
                if preds_arr.size > 0:
                    preds_eval = preds_arr.copy()
                    preds_eval[:, 2] = preds_eval[:, 2] * self.label_std + self.label_mean
                metric_values = regression_metrics(
                    labels_arr,
                    preds_eval,
                    self._regression_target_names(),
                )
                stats.mae_speed = metric_values.get("mae_speed", metric_values.get("mae", 0.0))
                stats.mae_angle = metric_values.get("mae_angle_deg", 0.0)
                stats.mae_radial = metric_values.get("mae_radial", 0.0)
                stats.mae_lateral = metric_values.get("mae_lateral", 0.0)
            else:
                metric_values = classification_metrics(labels_arr, preds_arr)
                stats.macro_f1 = metric_values["macro_f1"]
                stats.balanced_acc = metric_values["balanced_acc"]
                stats.macro_precision = metric_values["macro_precision"]
                stats.macro_recall = metric_values["macro_recall"]
                stats.auroc = compute_auroc(labels_arr, probs_arr) if probs_arr.ndim == 2 else None
                stats.auprc = compute_auprc(labels_arr, probs_arr) if probs_arr.ndim == 2 else None
        return stats

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    def train(self) -> Optional[EpochStats]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        test_stats: Optional[EpochStats] = None
        for epoch in range(1, self.epochs + 1):
            train_stats = self._run_epoch(self.train_loader, train=True)
            test_stats = None
            if self.test_loader is not None:
                test_stats = self._run_epoch(self.test_loader, train=False, compute_metrics=True)

            if self.is_regression:
                msg = f"Epoch {epoch:03d} | train loss {train_stats.loss:.4f}"
                if test_stats is not None:
                    msg += (
                        f" | test loss {test_stats.loss:.4f}"
                        f" speed mae {test_stats.mae_speed:.3f} m/s"
                        f" angle mae {test_stats.mae_angle:.2f}°"
                        f" radial mae {test_stats.mae_radial:.3f} m/s"
                        f" lateral mae {test_stats.mae_lateral:.3f} m/s"
                    )
            else:
                msg = f"Epoch {epoch:03d} | train loss {train_stats.loss:.4f} acc {train_stats.acc:.3f}"
                if test_stats is not None:
                    msg += (
                        f" | test loss {test_stats.loss:.4f} acc {test_stats.acc:.3f}"
                        f" f1 {test_stats.macro_f1:.3f} bal {test_stats.balanced_acc:.3f}"
                    )
            print(msg)

            if epoch % self.save_every == 0 or epoch == self.epochs:
                ckpt_name = f"{self.fold_name}.pt" if self.fold_name else "last.pt"
                self._save_checkpoint(ckpt_name, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

        self.evaluate_and_save()
        return test_stats

    def evaluate_and_save(self) -> None:
        labels, preds, probs, meta_df = self._collect_predictions(self.test_loader)
        if labels.size == 0:
            print("No test samples available for evaluation.")
            return

        extra_fields = {}
        if self.exp_name:
            extra_fields["exp_name"] = self.exp_name
        if self.test_fold is not None:
            extra_fields["test_fold"] = int(self.test_fold)

        pred_filename = f"predictions_{self.fold_name}.csv" if self.fold_name else "predictions.csv"

        if self.is_regression:
            preds_eval = preds
            if preds.size > 0:
                preds_eval = preds.copy()
                preds_eval[:, 2] = preds_eval[:, 2] * self.label_std + self.label_mean
            target_names = self._regression_target_names()
            metrics = regression_metrics(labels, preds_eval, target_names)
            print("Final regression metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            preds_for_csv = preds_eval
            if preds_eval.size > 0:
                preds_for_csv = self._unit_polar_preds_to_velocity(preds_eval)
            save_regression_csv(
                labels,
                preds_for_csv,
                target_names,
                meta_df,
                self.output_dir / pred_filename,
                extra_fields=extra_fields if extra_fields else None,
            )
        else:
            save_predictions_csv(
                labels,
                preds,
                meta_df,
                self.output_dir / pred_filename,
                extra_fields=extra_fields if extra_fields else None,
                probs=probs if probs.size > 0 else None,
            )

    def _collect_predictions(self, loader):
        self.model.eval()
        labels = []
        preds = []
        probs = []
        meta_frames = []
        y_dtype = torch.float32 if self.is_regression else torch.long
        with torch.no_grad():
            for batch in loader:
                x, y, meta = batch
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=y_dtype)
                logits = self.model(x)
                if self.is_regression:
                    batch_preds = logits
                else:
                    batch_preds = logits.argmax(dim=1)
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                labels.append(y.cpu().numpy())
                preds.append(batch_preds.cpu().numpy())
                if meta:
                    meta_frames.append(pd.DataFrame(meta))

        if not labels:
            return np.array([]), np.array([]), np.array([]), None
        labels_arr = np.concatenate(labels)
        preds_arr = np.concatenate(preds)
        probs_arr = np.concatenate(probs) if probs else np.array([])
        meta_df = pd.concat(meta_frames, axis=0).reset_index(drop=True) if meta_frames else None
        return labels_arr, preds_arr, probs_arr, meta_df
