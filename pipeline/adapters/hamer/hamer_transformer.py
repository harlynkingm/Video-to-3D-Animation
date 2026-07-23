"""The cross-attention transformer decoder used by HaMeR's MANO head.

Clean-room port of `hamer/models/components/pose_transformer.py`, restricted to
the single configuration HaMeR's checkpoint actually uses: `norm="layer"` (plain
LayerNorm, not the adaptive/conditional variants), no token positional-frequency
embedding, and inference only (dropout is a no-op). The dropped pieces
(`AdaptiveLayerNorm1D`, `FrequencyEmbedder`, `TransformerEncoder`, the various
token-dropout modules) are never on the code path for the released
`transformer_decoder` MANO head.

Module attribute names (`to_qkv`, `to_out`, `to_kv`, `to_q`, `net`, `norm`,
`fn`, `layers`, `to_token_embedding`, `pos_embedding`, `transformer`) match the
checkpoint's `mano_head.transformer.*` keys exactly, so the head strict-loads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    """Self-attention. `inner_dim = dim_head * heads` can differ from `dim`
    (here 512 vs 1024), so `to_out` projects back up to `dim`."""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)  # scale = 1/sqrt(dim_head), matches source
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Query tokens attend to `context` (the ViT feature tokens)."""

    def __init__(self, dim: int, context_dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Indices 0 and 3 hold the two Linears (1 = GELU, 2/4 = no-op dropout in
        # eval) -- matching the checkpoint's `net.0`/`net.3` keys.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            return self.fn(self.norm(x))
        return self.fn(self.norm(x), context=context)


class TransformerCrossAttn(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, context_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                    PreNorm(dim, CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head)),
                    PreNorm(dim, FeedForward(dim, mlp_dim)),
                ])
            )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context=context) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self, num_tokens: int, token_dim: int, dim: int, depth: int, heads: int,
        mlp_dim: int, dim_head: int, context_dim: int,
    ):
        super().__init__()
        self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.transformer = TransformerCrossAttn(dim, depth, heads, dim_head, mlp_dim, context_dim)

    def forward(self, inp: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.to_token_embedding(inp)
        x = x + self.pos_embedding[:, : x.shape[1]]
        return self.transformer(x, context=context)
