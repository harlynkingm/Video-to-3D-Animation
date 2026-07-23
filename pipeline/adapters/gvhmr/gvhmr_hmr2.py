"""HMR2's per-frame image-feature extractor: its own ViT backbone instance
(separate weights from ViTPose's, same architecture) feeding a small
cross-attention transformer, of which GVHMR only ever consumes the pooled
output token (`token_out`, i.e. `f_imgseq` -- GVHMR's per-frame image feature
conditioning). Ported from `comfyui-motioncapture/nodes/hmr2/model.py`.

**Only the `token_out` path is implemented.** The source's `SMPLTransformerDecoderHead`
can also directly decode its own SMPL pose/shape/camera prediction from `token_out`
(`decpose`/`decshape`/`deccam` + `init_body_pose`/`init_betas`/`init_cam` mean-pose
buffers) -- GVHMR's own pipeline never calls that path (only `feat_mode=True`, i.e.
`only_return_token_out=True` in the source), so it's provably dead code for this
port and isn't ported. The buffers/layers themselves are still declared (as
zero-initialized -- their real values come from the checkpoint at load time either
way) so `strict=True` loading still proves this module matches the checkpoint's
actual architecture, same reasoning as `sam31_vitdet_backbone.py`'s `interactive_convs`.

Module nesting below (`PreNorm` wrapping each sub-layer, `to_out`/`net` as a
`Sequential` with an inert `Dropout` placeholder shifting indices) looks more
convoluted than a hand-rolled version would -- it's ported to match the
checkpoint's actual parameter names exactly (confirmed against the real
`hmr2.safetensors` keys), not a style choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...helpers.vit_huge_backbone import VitHugeBackbone

D_MODEL = 1024
DEPTH = 6
NUM_HEADS = 8
DIM_HEAD = 64
MLP_DIM = 1024
CONTEXT_DIM = 1280  # HMR2 ViT backbone's embed_dim
NUM_BODY_JOINTS = 23
NPOSE = 6 * (NUM_BODY_JOINTS + 1)  # 144, unused buffer shape only (see module docstring)


class PoseAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, N, self.heads, -1).transpose(1, 2) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = x.shape
        Nk = context.shape[1]
        q = self.to_q(x).reshape(B, Nq, self.heads, -1).transpose(1, 2)
        k, v = (t.reshape(B, Nk, self.heads, -1).transpose(1, 2) for t in self.to_kv(context).chunk(2, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim), nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class CrossAttnTransformer(nn.Module):
    """Matches `smpl_head.transformer.transformer.*`: `depth` x
    [self-attn, cross-attn, feed-forward], each pre-normed.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(D_MODEL, PoseAttention(D_MODEL, NUM_HEADS, DIM_HEAD)),
                PreNorm(D_MODEL, CrossAttention(D_MODEL, CONTEXT_DIM, NUM_HEADS, DIM_HEAD)),
                PreNorm(D_MODEL, FeedForward(D_MODEL, MLP_DIM)),
            ])
            for _ in range(DEPTH)
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context=context) + x
            x = ff(x) + x
        return x


class TokenTransformerDecoder(nn.Module):
    """Matches `smpl_head.transformer.*`. A single learned query token
    (`num_tokens=1` in the source) cross-attends to the image feature map.
    """

    def __init__(self):
        super().__init__()
        self.to_token_embedding = nn.Linear(1, D_MODEL)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, D_MODEL))
        self.transformer = CrossAttnTransformer()

    def forward(self, token: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.to_token_embedding(token) + self.pos_embedding
        return self.transformer(x, context)


class SMPLTransformerDecoderHead(nn.Module):
    """Matches `smpl_head.*` exactly. Only `forward()`'s `token_out` path is
    implemented -- see module docstring."""

    def __init__(self):
        super().__init__()
        self.transformer = TokenTransformerDecoder()

        self.decpose = nn.Linear(D_MODEL, NPOSE)
        self.decshape = nn.Linear(D_MODEL, 10)
        self.deccam = nn.Linear(D_MODEL, 3)
        self.register_buffer("init_body_pose", torch.zeros(1, NPOSE))
        self.register_buffer("init_betas", torch.zeros(1, 10))
        self.register_buffer("init_cam", torch.zeros(1, 3))

    def forward(self, vit_feats: torch.Tensor) -> torch.Tensor:
        """vit_feats: (B, C, H, W) from `VitHugeBackbone`. Returns (B, D_MODEL) token_out."""
        B, C, H, W = vit_feats.shape
        context = vit_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        token = torch.zeros(B, 1, 1, device=vit_feats.device, dtype=vit_feats.dtype)
        return self.transformer(token, context).squeeze(1)


class GVHMRHMR2(nn.Module):
    """Matches the checkpoint's `backbone.*`/`smpl_head.*` exactly."""

    def __init__(self):
        super().__init__()
        self.backbone = VitHugeBackbone()
        self.smpl_head = SMPLTransformerDecoderHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 256, 192) -- returns (B, 1024) f_imgseq, one token per frame."""
        return self.smpl_head(self.backbone(x))
