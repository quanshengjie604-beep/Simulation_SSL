from __future__ import annotations

import torch
import torch.nn as nn


class AttentionCLSPooling(nn.Module):
    """Learnable CLS-token pooling over a token sequence."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=max(1, num_heads),
            dropout=max(0.0, float(dropout)),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, N, D) but received shape {tokens.shape}.")
        batch_size = tokens.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_token, tokens], dim=1)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        return self.norm(attn_out[:, 0, :])


class MeanPooling(nn.Module):
    """Mean pooling over the token dimension."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, N, D) but received shape {tokens.shape}.")
        return tokens.mean(dim=1)


__all__ = ["AttentionCLSPooling", "MeanPooling"]
