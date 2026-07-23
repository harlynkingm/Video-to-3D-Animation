"""SAM 3.1's vision backbone: a ViTDet trunk (windowed attention + periodic global
attention + 2D RoPE) plus an FPN neck producing multi-scale features for the
detector and tracker.

Ported from `comfy/ldm/sam3/sam.py`'s `ViTDet`/`SAM3VisionBackbone`, with
`comfy.ops`/`comfy.model_management` dependencies replaced by plain PyTorch
(this project has no ComfyUI runtime dependency, to avoid the GPL-3.0/Apache-2.0
conflict that would come with vendoring ComfyUI's own code). The 2D RoPE math
(`rope`, `EmbedND`, `apply_rope`) is borrowed from ComfyUI's own Flux
implementation (`comfy/ldm/flux/math.py`, `comfy/ldm/flux/layers.py`), which
SAM3.1 itself reuses -- credited here, not vendored, since it's plain rotation
math with no SAM3-specific weights.

Deliberately NOT ported: the interactive SAM prompt-encoder/mask-decoder
classes in the same source file (`SAMAttention`, `TwoWayTransformer`,
`PositionEmbeddingRandom`, the point/box `MLP` head). Those exist only to
support click/box-prompted conditioning (`initial_masks`), which this project
never uses (all detections come from text prompts, matched to tracked objects
by `sam31_tracker.py`) -- confirmed while writing the tracker port that nothing
here needs them; `sam31_tracker.py` has its own, separate copy of similar
classes for its own interactive mask decoder, not shared with this file.
"""

from __future__ import annotations

import enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackerMode(enum.StrEnum):
    """Which of `Sam31VisionBackbone`'s two tracker-only FPN conv groups to run.
    `PROPAGATION` is per-frame mask propagation (this project's only real usage);
    `INTERACTIVE` seeds tracking state from a freshly detected mask (loaded for a
    clean `strict=True` checkpoint load, but never actually invoked -- see this
    module's docstring).
    """

    PROPAGATION = "propagation"
    INTERACTIVE = "interactive"

# These are fixed by the checkpoint's own training configuration -- not free parameters,
# and not something read from the checkpoint file itself (there's no "img_size" tensor).
# The RoPE position buffers below are precomputed once, at construction, for exactly a
# 72x72 patch grid (IMG_SIZE // PATCH_SIZE). They do NOT adapt to a differently-sized
# input: forward() would not error, it would silently crop the RoPE table to fit,
# producing spatially-wrong positional encoding for every frame. So every frame this
# pipeline processes must be resized to exactly IMG_SIZE x IMG_SIZE before it reaches
# this backbone. Verified against the real checkpoint.
IMG_SIZE = 1008
PATCH_SIZE = 14
PRETRAIN_IMG_SIZE = 336  # what the absolute pos_embed was actually trained at (24^2 + 1 cls token = 577)
EMBED_DIM = 1024
DEPTH = 32
NUM_HEADS = 16
MLP_RATIO = 4.625  # fc1 width 4736 / embed_dim 1024, confirmed against the real checkpoint
WINDOW_SIZE = 24
GLOBAL_ATT_BLOCKS = (7, 15, 23, 31)  # every 8th block (of 32) attends globally; the rest are windowed
FPN_D_MODEL = 256

# --- 2D RoPE (borrowed from ComfyUI's Flux implementation, credited above) ---


def _rope_freqs(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """Per-position rotation matrices for `dim//2` frequency pairs. Returns (..., n, dim//2, 2, 2), fp32."""
    assert dim % 2 == 0
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32), omega.to(torch.float32))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(torch.float32)


def rope_2d(end_x: int, end_y: int, dim: int, theta: float = 10000.0, scale_pos: float = 1.0) -> torch.Tensor:
    """2D axial RoPE over an end_x * end_y grid. Returns (1, 1, end_x*end_y, dim//2, 2, 2)."""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    x_ids = (t % end_x) * scale_pos
    y_ids = torch.div(t, end_x, rounding_mode="floor") * scale_pos
    axis_dim = dim // 2
    emb = torch.cat([_rope_freqs(x_ids, axis_dim, theta), _rope_freqs(y_ids, axis_dim, theta)], dim=-3)
    return emb.unsqueeze(0).unsqueeze(0)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a query/key pair by the same per-position frequencies (the usual RoPE attention use)."""
    return _rotate_pairs(xq, freqs_cis), _rotate_pairs(xk, freqs_cis)


def _rotate_pairs(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate one tensor's last dimension, taken as (dim//2) adjacent (real, imag) pairs, by
    freqs_cis's per-position 2x2 rotation matrices. The actual rotation math; `apply_rope`
    just calls this once for the query and once for the key.
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out = x_out + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)


# --- Windowed attention helpers ---


def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# --- ViTDet trunk ---


class ViTMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ViTDetAttention(nn.Module):
    """Fused-QKV attention with optional 2D RoPE."""

    def __init__(self, dim: int, num_heads: int, use_rope: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # each (B, heads, N, head_dim)
        if self.use_rope and freqs_cis is not None:
            q, k = apply_rope(q, k, freqs_cis)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ViTDetBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, window_size: int, use_rope: bool):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTDetAttention(dim, num_heads, use_rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = x.view(x.shape[0], self.window_size * self.window_size, -1)
            x = self.attn(x, freqs_cis=freqs_cis)
            x = x.view(-1, self.window_size, self.window_size, x.shape[-1])
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
            x = self.attn(x, freqs_cis=freqs_cis)
            x = x.view(B, H, W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ViTDet(nn.Module):
    """The SAM 3.1 image encoder trunk. Fixed configuration (see the module-level
    constants above) -- there's no legitimate reason for this project to construct it
    any other way, so it's not parameterized like the source's more general version.
    Expects input already resized to exactly IMG_SIZE x IMG_SIZE.
    """

    def __init__(self):
        super().__init__()
        self.global_att_blocks = set(GLOBAL_ATT_BLOCKS)

        self.patch_embed = PatchEmbed()

        num_patches = (PRETRAIN_IMG_SIZE // PATCH_SIZE) ** 2 + 1  # +1 for cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, EMBED_DIM))

        self.ln_pre = nn.LayerNorm(EMBED_DIM)

        grid_size = IMG_SIZE // PATCH_SIZE
        pretrain_grid = PRETRAIN_IMG_SIZE // PATCH_SIZE

        self.blocks = nn.ModuleList([
            ViTDetBlock(EMBED_DIM, NUM_HEADS, MLP_RATIO,
                        window_size=0 if i in self.global_att_blocks else WINDOW_SIZE, use_rope=True)
            for i in range(DEPTH)
        ])

        head_dim = EMBED_DIM // NUM_HEADS
        rope_scale = pretrain_grid / grid_size
        self.register_buffer("freqs_cis", rope_2d(grid_size, grid_size, head_dim, scale_pos=rope_scale), persistent=False)
        self.register_buffer("freqs_cis_window", rope_2d(WINDOW_SIZE, WINDOW_SIZE, head_dim), persistent=False)

    def _get_pos_embed(self, num_tokens: int) -> torch.Tensor:
        """Absolute position embedding, tiled (not interpolated) from the pretrain grid to the
        current grid size -- matches the source exactly, since that's how these particular
        weights were adapted, not a generic choice.
        """
        pos = self.pos_embed
        if pos.shape[1] == num_tokens:
            return pos
        cls_pos = pos[:, :1]
        spatial_pos = pos[:, 1:]
        old_size = int(math.sqrt(spatial_pos.shape[1]))
        new_size = int(math.sqrt(num_tokens - 1)) if num_tokens > 1 else old_size
        spatial_2d = spatial_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        tiles_h = new_size // old_size + 1
        tiles_w = new_size // old_size + 1
        tiled = spatial_2d.tile([1, 1, tiles_h, tiles_w])[:, :, :new_size, :new_size]
        tiled = tiled.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        return torch.cat([cls_pos, tiled], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)

        pos = self._get_pos_embed(Hp * Wp + 1).to(x.dtype)
        x = x + pos[:, 1:Hp * Wp + 1]

        x = x.view(B, Hp, Wp, C)
        x = self.ln_pre(x)

        freqs_cis_global = self.freqs_cis.to(x.dtype)
        freqs_cis_win = self.freqs_cis_window.to(x.dtype)

        for block in self.blocks:
            fc = freqs_cis_win if block.window_size > 0 else freqs_cis_global
            x = block(x, freqs_cis=fc)

        return x.permute(0, 3, 1, 2)


# --- FPN neck ---


class FPNScaleConv(nn.Module):
    """One FPN output scale: optionally up/down-sample the trunk's single-scale feature map,
    then two convs to project to d_model channels.

    The checkpoint's own parameter names for the 4x-upsample case are the generic
    `dconv_2x2_0`/`dconv_2x2_1` (positional, not descriptive); renamed here to
    `upsample_4x_stage1`/`upsample_4x_stage2` since that's what they actually do (the two
    sequential 2x transposed-conv steps that together reach 4x). `_load_from_state_dict`
    below remaps the checkpoint's original key names automatically, so
    `load_state_dict(..., strict=True)` still works with zero caller-side remapping.
    """

    def __init__(self, in_dim: int, out_dim: int, scale: float):
        super().__init__()
        if scale == 4.0:
            self.upsample_4x_stage1 = nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)
            self.upsample_4x_stage2 = nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2)
            proj_in = in_dim // 4
        elif scale == 2.0:
            self.dconv_2x2 = nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)
            proj_in = in_dim // 2
        elif scale == 1.0:
            proj_in = in_dim
        elif scale == 0.5:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            proj_in = in_dim
        else:
            raise ValueError(f"Unsupported FPN scale: {scale}")
        self.scale = scale
        self.conv_1x1 = nn.Conv2d(proj_in, out_dim, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        rename = {"dconv_2x2_0": "upsample_4x_stage1", "dconv_2x2_1": "upsample_4x_stage2"}
        for old, new in rename.items():
            for suffix in (".weight", ".bias"):
                old_key = prefix + old + suffix
                if old_key in state_dict:
                    state_dict[prefix + new + suffix] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 4.0:
            x = F.gelu(self.upsample_4x_stage1(x))
            x = self.upsample_4x_stage2(x)
        elif self.scale == 2.0:
            x = self.dconv_2x2(x)
        elif self.scale == 0.5:
            x = self.pool(x)
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal position encoding (DETR-style), computed in fp32 for numerical
    stability then cast to match the caller's working dtype -- not just a generic
    choice, replicating a precision detail from the source: certain positional-encoding
    math stays fp32 internally regardless of the surrounding working dtype (e.g. fp16),
    cast back only at the end, for numerical stability.
    """

    def __init__(self, num_pos_feats: int, temperature: float = 10000.0, normalize: bool = True):
        super().__init__()
        assert num_pos_feats % 2 == 0
        self.half_dim = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def _sincos(self, vals: torch.Tensor) -> torch.Tensor:
        freqs = self.temperature ** (2 * (torch.arange(self.half_dim, dtype=torch.float32, device=vals.device) // 2) / self.half_dim)
        raw = vals[..., None] * self.scale / freqs
        return torch.stack((raw[..., 0::2].sin(), raw[..., 1::2].cos()), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gy = torch.arange(H, dtype=torch.float32, device=x.device)
        gx = torch.arange(W, dtype=torch.float32, device=x.device)
        if self.normalize:
            gy, gx = gy / (H - 1 + 1e-6), gx / (W - 1 + 1e-6)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        pe = torch.cat((self._sincos(yy), self._sincos(xx)), dim=-1).permute(2, 0, 1).unsqueeze(0)
        return pe.expand(B, -1, -1, -1).to(x.dtype)


class Sam31VisionBackbone(nn.Module):
    """ViTDet trunk + FPN neck. Matches `detector.backbone.vision_backbone.*` exactly
    (prefix stripped). This checkpoint is the multiplex variant: three FPN conv groups
    (`convs`, `propagation_convs`, `interactive_convs`), not the non-multiplex
    `sam2_convs` single group.

    `interactive_convs` is loaded (needed for a clean strict=True checkpoint load) but
    this project's own code never calls `tracker_mode=TrackerMode.INTERACTIVE` -- that
    path only matters for `initial_masks`-based conditioning, which this project doesn't use.
    """

    def __init__(self):
        super().__init__()
        self.trunk = ViTDet()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=FPN_D_MODEL, normalize=True)

        scales = [4.0, 2.0, 1.0]
        self.convs = nn.ModuleList([FPNScaleConv(EMBED_DIM, FPN_D_MODEL, s) for s in scales])
        self.propagation_convs = nn.ModuleList([FPNScaleConv(EMBED_DIM, FPN_D_MODEL, s) for s in scales])
        self.interactive_convs = nn.ModuleList([FPNScaleConv(EMBED_DIM, FPN_D_MODEL, s) for s in scales])

    def forward(
        self,
        images: torch.Tensor,
        tracker_mode: TrackerMode | None = None,
        tracker_only: bool = False,
        cached_trunk: torch.Tensor | None = None,
    ):
        """Returns (features, positions, tracker_features, tracker_positions) -- any pair may be
        None depending on tracker_only/tracker_mode, matching the source's calling convention.

        `cached_trunk` lets a caller that already ran `self.trunk(images)` (e.g. the tracker's
        per-frame `backbone_fn`, which needs both detector and tracker FPN outputs from the same
        frame) skip a second, redundant ViTDet trunk forward pass -- the trunk is by far the most
        expensive part of this backbone.
        """
        backbone_out = cached_trunk if cached_trunk is not None else self.trunk(images)

        if tracker_only:
            tracker_convs = self.propagation_convs if tracker_mode == TrackerMode.PROPAGATION else self.interactive_convs
            tracker_features = [conv(backbone_out) for conv in tracker_convs]
            tracker_positions = [self.position_encoding(f) for f in tracker_features]
            return None, None, tracker_features, tracker_positions

        features = [conv(backbone_out) for conv in self.convs]
        positions = [self.position_encoding(f) for f in features]

        if tracker_mode is None:
            return features, positions, None, None

        tracker_convs = self.propagation_convs if tracker_mode == TrackerMode.PROPAGATION else self.interactive_convs
        tracker_features = [conv(backbone_out) for conv in tracker_convs]
        tracker_positions = [self.position_encoding(f) for f in tracker_features]
        return features, positions, tracker_features, tracker_positions
