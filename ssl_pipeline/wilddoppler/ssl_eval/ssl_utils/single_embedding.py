from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .contrastive import ContrastiveProjectionHead
from .pooling import AttentionCLSPooling, MeanPooling


class SingleEmbeddingModel(nn.Module):
    """Backbone + pooling + single projection head used by contrastive eval."""

    def __init__(
        self,
        backbone: nn.Module,
        pool_num_heads: int = 4,
        pool_dropout: float = 0.0,
        head_cfg: Optional[dict] = None,
        pool_type: str = "attention_cls",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if not hasattr(backbone, "embed_dim"):
            raise ValueError("Backbone must expose 'embed_dim' for single-embedding models.")
        embed_dim = int(backbone.embed_dim)
        if str(pool_type).lower() == "mean":
            self.pool = MeanPooling()
        else:
            self.pool = AttentionCLSPooling(embed_dim=embed_dim, num_heads=pool_num_heads, dropout=pool_dropout)
        cfg = dict(head_cfg or {})
        out_dim = int(cfg.pop("out_dim", embed_dim))
        self.head = ContrastiveProjectionHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            hidden_dim=cfg.pop("hidden_dim", embed_dim),
            num_layers=int(cfg.pop("num_layers", 2)),
            dropout=float(cfg.pop("dropout", 0.0)),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, return_raw: bool = False) -> dict:
        tokens, _ = self.backbone(x, cond=cond)
        pooled = self.pool(tokens)
        if return_raw:
            embedding, raw = self.head(pooled, return_raw=True)
        else:
            embedding = self.head(pooled)
            raw = None
        outputs = {
            "tokens": tokens,
            "pooled": pooled,
            "embedding": embedding,
        }
        if raw is not None:
            outputs["embedding_raw"] = raw
        return outputs


__all__ = ["SingleEmbeddingModel"]
