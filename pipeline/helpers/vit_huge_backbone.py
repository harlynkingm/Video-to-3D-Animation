"""The ViT-Huge backbone shared by ViTPose, HMR2, and HaMeR -- three separate
checkpoints, identical architecture, different learned weights and different
heads on top. Lives here (not inside any one adapter) because all three of
those model adapters load their own weights into an instance of this same
class; confirmed by strict-loading each checkpoint's `backbone.*` tensors into
it with zero missing/unexpected keys.

Restricted to the single configuration all three checkpoints actually use
(confirmed identical in every source: `img_size=(256,192)`, `patch_size=16`,
`embed_dim=1280`, `depth=32`, `num_heads=16`, `mlp_ratio=4`, `qkv_bias=True`) --
not a generic, arbitrarily-configurable ViT class. Also dropped, since this
project only ever runs inference: dropout/drop-path (no-ops in eval mode
regardless of rate), gradient checkpointing (a training memory optimization),
and the unused classification head/`ratio`-based patch multiplier from the
source's more general version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = (256, 192)  # (H, W) -- a single person/hand crop, not the full frame
PATCH_SIZE = 16
IN_CHANS = 3
EMBED_DIM = 1280
DEPTH = 32
NUM_HEADS = 16
MLP_RATIO = 4.0
# padding=2 is what actually makes the patch grid come out to exactly 16x12
# (192 patches, matching the checkpoint's [1, 193, 1280] pos_embed) -- confirmed
# against the real checkpoints, not a default left over from a generic formula.
PATCH_CONV_PADDING = 2
GRID_H, GRID_W = IMG_SIZE[0] // PATCH_SIZE, IMG_SIZE[1] // PATCH_SIZE  # 16, 12
NUM_PATCHES = GRID_H * GRID_W  # 192


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Fused-QKV self-attention (checkpoint's own layout, matching the SDPA
    convention used throughout this project's ports)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(IN_CHANS, EMBED_DIM, kernel_size=PATCH_SIZE,
                               stride=PATCH_SIZE, padding=PATCH_CONV_PADDING)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, GRID_H, GRID_W)
        return x.flatten(2).transpose(1, 2)  # (B, GRID_H*GRID_W, embed_dim)


class VitHugeBackbone(nn.Module):
    """Expects input already cropped/resized to exactly IMG_SIZE (H, W). Matches
    `backbone.*` in `vitpose.safetensors`, `hmr2.safetensors`, and
    `hamer.safetensors` exactly -- same architecture, loaded from different
    checkpoints by the modules that each own an instance of this class.
    """

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES + 1, EMBED_DIM))
        self.blocks = nn.ModuleList([Block(EMBED_DIM, NUM_HEADS, MLP_RATIO) for _ in range(DEPTH)])
        self.last_norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, embed_dim, GRID_H, GRID_W) spatial features."""
        B = x.shape[0]
        x = self.patch_embed(x)
        # The checkpoint's pos_embed has one extra learned row beyond the 192
        # patch positions; the source adds it to every patch as a shared bias
        # rather than treating it as a separate CLS token -- ported as-is,
        # since this is the exact behavior the checkpoints were trained under.
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for block in self.blocks:
            x = block(x)

        x = self.last_norm(x)
        return x.permute(0, 2, 1).reshape(B, EMBED_DIM, GRID_H, GRID_W)
