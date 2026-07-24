"""SAM 3.1's multiplex video tracker: memory bank + decoupled memory attention, an
interactive (single-object) SAM decoder used to seed new tracks from a detected mask,
a multiplex (16-object) SAM decoder used for per-frame propagation, and the
detection/reconditioning/keep-alive bookkeeping that ties them together across a clip.

Ported from `comfy/ldm/sam3/tracker.py`, replacing `comfy.ops`/`optimized_attention`/
`comfy.model_management` with plain PyTorch (this project has no ComfyUI runtime
dependency -- see the module docstrings in this package's other files for why).
Mask convention: boolean, True = attend, same as the rest of this port.

**Scope deliberately narrower than the source file.** This project only ever drives
the tracker through `track_video_with_detection` with `initial_masks=None` (detections
come from `human_prompt`/`object_prompt` text, never from user clicks or boxes) on the
SAM 3.1 *multiplex* checkpoint specifically. So, dropped entirely:
  - `SAM3Tracker` and its non-decoupled `MemoryAttnLayer`/`MemoryAttnEncoder`/
    `MemoryTransformer` -- these belong to the older, non-multiplex SAM3 tracker
    (a different checkpoint's tensor layout), never loaded by this project.
  - The point/box interactive-click path: `track_step`'s `point_inputs` branch,
    `SAM3Model.forward_segment`, and `SAMPromptEncoder`'s `boxes` handling. A mask is
    always this tracker's only conditioning input here (from a detected object, not a
    click), so `boxes` is always `None` and `points` is always the same internally-
    synthesized dummy value `forward_sam_heads` builds when no real point is given --
    ported directly as that fixed behavior rather than as a conditional.
  - The CUDA-stream backbone prefetch in `track_video_with_detection` (computing frame
    N+1's backbone on a second stream while frame N is processed). This is a wall-clock
    optimization with no effect on correctness or memory footprint; dropped for a
    simpler, easier-to-verify sequential loop. Worth adding back only if profiling on
    real clips shows it's actually needed.

Everything else -- the multiplex state bookkeeping, the memory bank encode/attend
cycle, mask-based NMS, reconditioning of degraded tracks, and the keep-alive hysteresis
that prevents single-frame flicker -- is ported faithfully, since none of it is
optional for correct multi-object tracking over a real clip.
"""

from __future__ import annotations

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.progress_tracker import StageName

from .sam31_vitdet_backbone import IMG_SIZE, PATCH_SIZE, TrackerMode, _rotate_pairs, rope_2d

from ...helpers.progress_reporter import frame_progress

NO_OBJ_SCORE = -1024.0

# `detect_fn`'s expected return-dict keys -- this project's own contract between
# `sam31_adapter.py` (which implements `detect_fn`) and `track_video_with_detection`
# below (which calls it), and `track_video_with_detection`'s own returned-dict keys.
KEY_SCORES = "scores"
KEY_MASKS = "masks"
KEY_PACKED_MASKS = "packed_masks"
KEY_N_FRAMES = "n_frames"

D_MODEL = 256
NUM_MASKMEM = 7  # size of the rolling memory window (this frame's 7 most recent conditioning/tracked frames)
NUM_MULTIPLEX = 16  # objects packed per multiplex "bucket" -- see MultiplexState
NUM_MULTIMASK_OUTPUTS = 3
MAX_OBJ_PTRS_IN_ENCODER = 16
SIGMOID_SCALE_FOR_MEM_ENC = 2.0
SIGMOID_BIAS_FOR_MEM_ENC = -1.0
BACKBONE_STRIDE = PATCH_SIZE
INTERNAL_MAX_OBJECTS = 64  # hard ceiling on accumulated tracks; track_video_with_detection's
                           # max_objects=0 (or anything above this) is clamped to this.


def to_spatial(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, H*W, C) -> (B, C, H, W)."""
    return x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2)


# --- Mask utilities (NMS, connected components, hole-filling, bit-packing) ---


def _compute_mask_overlap(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """Max of IoU and IoM (intersection over minimum area) -- more robust to size
    differences than IoU alone (e.g. a small hand mask fully inside a large body mask).
    """
    a_flat = (masks_a > 0).float().flatten(1)
    b_flat = (masks_b > 0).float().flatten(1)
    intersection = a_flat @ b_flat.T
    area_a = a_flat.sum(1, keepdim=True)
    area_b = b_flat.sum(1, keepdim=True).T
    iou = intersection / (area_a + area_b - intersection).clamp(min=1)
    iom = intersection / torch.min(area_a.expand_as(iou), area_b.expand_as(iou)).clamp(min=1)
    return torch.max(iou, iom)


def _nms_masks(masks: torch.Tensor, scores: torch.Tensor, thresh: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-based NMS using the IoU/IoM overlap above. Returns (filtered_masks, filtered_scores)."""
    order = scores.argsort(descending=True)
    masks, scores = masks[order], scores[order]
    keep: list[int] = []
    for i in range(masks.shape[0]):
        if keep:
            if _compute_mask_overlap(masks[i:i + 1], masks[torch.tensor(keep, device=masks.device)]).max() >= thresh:
                continue
        keep.append(i)
    return masks[keep], scores[keep]


def _get_connected_components(mask_bin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Connected-component labels and per-pixel component areas. mask_bin: [B, 1, H, W] uint8."""
    labels_list, areas_list = [], []
    for i in range(mask_bin.shape[0]):
        m = mask_bin[i, 0].cpu().numpy()
        _, labeled, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        areas = stats[labeled, cv2.CC_STAT_AREA].astype("int32")
        labels_list.append(torch.from_numpy(labeled).to(mask_bin.device))
        areas_list.append(torch.from_numpy(areas).to(device=mask_bin.device, dtype=torch.int32))
    return torch.stack(labels_list).unsqueeze(1), torch.stack(areas_list).unsqueeze(1)


def fill_holes_in_mask_scores(mask: torch.Tensor, max_area: int = 0) -> torch.Tensor:
    """Remove small foreground sprinkles and fill small background holes via connected components."""
    if max_area <= 0:
        return mask

    mask_bg = (mask <= 0).to(torch.uint8)
    _, areas_bg = _get_connected_components(mask_bg)
    small_bg = mask_bg.bool() & (areas_bg <= max_area)
    mask = torch.where(small_bg, 0.1, mask)

    # Only remove a foreground sprinkle if it's smaller than both max_area and half the
    # total foreground area -- guards against erasing a genuinely small-but-real object.
    mask_fg = (mask > 0).to(torch.uint8)
    fg_area_thresh = mask_fg.sum(dim=(2, 3), keepdim=True, dtype=torch.int32)
    fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
    _, areas_fg = _get_connected_components(mask_fg)
    small_fg = mask_fg.bool() & (areas_fg <= fg_area_thresh)
    mask = torch.where(small_fg, -0.1, mask)

    return mask


def pack_masks(masks: torch.Tensor) -> torch.Tensor:
    """Pack binary masks [*, H, W] to bit-packed [*, H, W//8] uint8 (8 pixels/byte, LSB-first).
    W must be divisible by 8. Keeps a clip's worth of per-frame masks small enough to move
    off-GPU every frame without them dominating host RAM.
    """
    binary = masks > 0
    shifts = torch.arange(8, device=masks.device)
    return (binary.view(*masks.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def unpack_masks(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack_masks`: bit-packed [*, H, W//8] uint8 -> bool [*, H, W*8]."""
    bits = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device)
    return (packed.unsqueeze(-1) & bits).bool().view(*packed.shape[:-1], -1)


# --- Memory-attention RoPE and temporal position encodings ---


def apply_rope_memory(
    q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor, num_heads: int, num_k_exclude_rope: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same 2D axial RoPE used by the vision backbone (see
    `sam31_vitdet_backbone.rope_2d`/`_rotate_pairs`) to memory-attention queries and keys.

    Args:
        q: [B, Nq, C] projected queries (current frame features).
        k: [B, Nk, C] projected keys (memory tokens) -- the trailing `num_k_exclude_rope`
            tokens (object pointers, which carry no 2D spatial position) are left untouched.
        freqs: [1, 1, Nq, dim//2, 2, 2] rotation matrices for one frame's spatial grid.
    """
    B, Nq, C = q.shape
    head_dim = C // num_heads

    q_h = q.view(B, Nq, num_heads, head_dim).transpose(1, 2)
    q_h = _rotate_pairs(q_h, freqs)
    q = q_h.transpose(1, 2).reshape(B, Nq, C)

    Nk = k.shape[1]
    num_k_rope = Nk - num_k_exclude_rope
    if num_k_rope > 0:
        Nf = freqs.shape[2]  # spatial positions in one frame
        if num_k_rope > Nf:
            # Memory spans multiple past frames concatenated -- tile this frame's position
            # table across them (each past frame reuses the same spatial grid).
            r = (num_k_rope + Nf - 1) // Nf
            pe_k = freqs.repeat(1, 1, r, 1, 1, 1)[:, :, :num_k_rope]
        else:
            pe_k = freqs[:, :, :num_k_rope]

        k_h = k[:, :num_k_rope].view(B, num_k_rope, num_heads, head_dim).transpose(1, 2)
        k_h = _rotate_pairs(k_h, pe_k)
        k = k.clone()
        k[:, :num_k_rope] = k_h.transpose(1, 2).reshape(B, num_k_rope, C)

    return q, k


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """1D sinusoidal positional encoding for temporal (frame-distance) positions."""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


def compute_tpos_enc(
    rel_pos_list: list[int], device: torch.device, d_model: int, proj_layer: nn.Module,
    dtype: torch.dtype | None = None, max_abs_pos: int | None = None,
) -> torch.Tensor:
    """Temporal position encoding for object pointers, projected to memory-token width."""
    pos_enc = torch.tensor(rel_pos_list, dtype=torch.float32, device=device) / max((max_abs_pos or 2) - 1, 1)
    pos_enc = get_1d_sine_pe(pos_enc, dim=d_model)
    if dtype is not None:
        pos_enc = pos_enc.to(dtype)
    return proj_layer(pos_enc)


def _pad_to_buckets(tensor: torch.Tensor, target_buckets: int) -> torch.Tensor:
    """Zero-pad a [num_buckets, ...] tensor to `target_buckets` along dim 0 -- lets memory
    recorded before the multiplex state grew (new objects detected mid-clip) still be read
    back at the current, larger bucket count.
    """
    if tensor.shape[0] >= target_buckets:
        return tensor
    pad_shape = (target_buckets - tensor.shape[0],) + tensor.shape[1:]
    return torch.cat([tensor, torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)], dim=0)


def _prep_frame(images: torch.Tensor, idx, device: torch.device, dtype: torch.dtype, size: int) -> torch.Tensor:
    """Slice CPU full-res frame(s), move to GPU in the working dtype, and resize to size x size."""
    frame = images[idx].to(device=device, dtype=dtype)
    return F.interpolate(frame, size=(size, size), mode="bicubic", align_corners=False)


def _compute_backbone(backbone_fn, frame: torch.Tensor, frame_idx: int | None = None):
    """Run `backbone_fn` on one frame and reshape its multi-scale FPN outputs to token
    sequences. Returns (vision_feats, vision_pos, feat_sizes, features, trunk_out).
    """
    features, positions, trunk_out = backbone_fn(frame, frame_idx=frame_idx)
    feat_sizes = [(x.shape[-2], x.shape[-1]) for x in features]
    vision_feats = [x.flatten(2).permute(0, 2, 1) for x in features]
    vision_pos = [x.flatten(2).permute(0, 2, 1) for x in positions]
    return vision_feats, vision_pos, feat_sizes, features, trunk_out


def collect_memory_tokens(
    output_dict: dict, frame_idx: int, num_maskmem: int, maskmem_tpos_enc: torch.Tensor, device: torch.device,
    collect_image_feats: bool = False, tpos_v2: bool = False, num_buckets: int | None = None,
):
    """Gather spatial memory (+ position encodings, + optionally raw image features for the
    decoupled cross-attention) from past conditioning and tracked frames within the rolling
    `num_maskmem`-frame window.
    """
    to_cat_memory, to_cat_memory_pos = [], []
    to_cat_image_feat, to_cat_image_pos = [], []

    def _append(out: dict, tpos_idx: int) -> None:
        feats = out["maskmem_features"].to(device)
        if num_buckets is not None:
            feats = _pad_to_buckets(feats, num_buckets)
        to_cat_memory.append(feats.flatten(2).permute(0, 2, 1))
        enc = out["maskmem_pos_enc"][-1].to(device).flatten(2).permute(0, 2, 1)
        if num_buckets is not None:
            enc = _pad_to_buckets(enc, num_buckets)
        tpos = maskmem_tpos_enc[tpos_idx].to(dtype=enc.dtype)
        to_cat_memory_pos.append(enc + tpos)
        if collect_image_feats and "image_features" in out:
            to_cat_image_feat.append(out["image_features"].to(device))
            to_cat_image_pos.append(out["image_pos_enc"].to(device) + tpos)

    cond_outputs = output_dict["cond_frame_outputs"]
    for t, out in cond_outputs.items():
        if tpos_v2:
            t_pos = frame_idx - t
            tpos_idx = num_maskmem - t_pos - 1 if 0 < t_pos < num_maskmem else num_maskmem - 1
        else:
            tpos_idx = num_maskmem - 1
        _append(out, tpos_idx)

    for t_pos in range(1, num_maskmem):
        out = output_dict["non_cond_frame_outputs"].get(frame_idx - (num_maskmem - t_pos), None)
        if out is None or out.get("maskmem_features") is None:
            continue
        _append(out, num_maskmem - t_pos - 1)

    return to_cat_memory, to_cat_memory_pos, to_cat_image_feat, to_cat_image_pos, cond_outputs


# --- Interactive (single-object) SAM decoder primitives ---
#
# These back `interactive_sam_mask_decoder`/`interactive_sam_prompt_encoder`, used once per
# newly-detected object to turn its detection mask into initial tracking state (see
# `Sam31Tracker._condition_with_masks`). This project never issues real point/box clicks --
# `forward_sam_heads` below always synthesizes the same "no real point" dummy input SAM's own
# mask-decoder architecture expects when only a mask prompt is given.


class MLP(nn.Module):
    """Plain N-layer MLP with ReLU between layers. Same shape as
    `sam31_detector.SimpleMLP` -- duplicated rather than imported so this file and
    `sam31_detector.py` stay independently loadable/testable (their checkpoint weights
    are separate tensor groups; see `sam31_detector.py`'s module docstring for the same
    reasoning applied there).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class SAMAttention(nn.Module):
    """Separate q/k/v/out projections (matches the checkpoint's own parameter layout
    exactly), with an optional internal downsample for the cross-attention variants.
    """

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.num_heads = num_heads
        internal_dim = embedding_dim // downsample_rate
        self.q_proj = nn.Linear(embedding_dim, internal_dim)
        self.k_proj = nn.Linear(embedding_dim, internal_dim)
        self.v_proj = nn.Linear(embedding_dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, embedding_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        B, Lq, C = q.shape
        head_dim = C // self.num_heads
        q = q.view(B, Lq, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, k.shape[1], self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, v.shape[1], self.num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Lq, -1)
        return self.out_proj(out)


class MLPBlock(nn.Module):
    """The original SAM code's own MLP block naming (`lin1`/`lin2`) -- the raw checkpoint
    uses this directly. ComfyUI's own loader remaps these to `nn.Sequential`-style `0`/`2`
    keys before loading (see `comfy/supported_models.py`'s SAM3 key remap); since this
    project loads the raw checkpoint directly, matching its actual names here is simpler
    than porting that remap step.
    """

    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(F.relu(self.lin1(x)))


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe
        self.self_attn = SAMAttention(embedding_dim, num_heads)
        self.cross_attn_token_to_image = SAMAttention(embedding_dim, num_heads, attention_downsample_rate)
        self.cross_attn_image_to_token = SAMAttention(embedding_dim, num_heads, attention_downsample_rate)
        self.mlp = MLPBlock(embedding_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor):
        if self.skip_first_layer_pe:
            queries = self.norm1(self.self_attn(queries, queries, queries))
        else:
            q = queries + query_pe
            queries = self.norm1(queries + self.self_attn(q, q, queries))
        q, k = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q, k, keys))
        queries = self.norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(k, q, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int = 2, embedding_dim: int = D_MODEL, num_heads: int = 8,
                 mlp_dim: int = 2048, attention_downsample_rate: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(embedding_dim, num_heads, mlp_dim, attention_downsample_rate,
                                  skip_first_layer_pe=(i == 0))
            for i in range(depth)
        ])
        self.final_attn_token_to_image = SAMAttention(embedding_dim, num_heads, attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: torch.Tensor, image_pe: torch.Tensor, point_embedding: torch.Tensor):
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        q, k = queries + point_embedding, keys + image_pe
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(q, k, keys))
        return queries, keys


class PositionEmbeddingRandom(nn.Module):
    """Fourier-feature positional encoding via a fixed random Gaussian projection."""

    def __init__(self, num_pos_feats: int = 64):
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn(2, num_pos_feats))

    def _encode(self, normalized_coords: torch.Tensor) -> torch.Tensor:
        """Map normalized [0,1] coordinates to fourier features. Computed in fp32 for
        stability (see `sam31_vitdet_backbone.py`'s "fp32 island" note), cast back after.
        """
        orig_dtype = normalized_coords.dtype
        proj = self.positional_encoding_gaussian_matrix.to(dtype=torch.float32)
        projected = 2 * torch.pi * (2 * normalized_coords.float() - 1) @ proj
        return torch.cat([projected.sin(), projected.cos()], dim=-1).to(orig_dtype)

    def forward(self, size: tuple[int, int], device: torch.device | None = None) -> torch.Tensor:
        h, w = size
        dev = device if device is not None else self.positional_encoding_gaussian_matrix.device
        ones = torch.ones((h, w), device=dev, dtype=torch.float32)
        norm_xy = torch.stack([(ones.cumsum(1) - 0.5) / w, (ones.cumsum(0) - 0.5) / h], dim=-1)
        return self._encode(norm_xy).permute(2, 0, 1).unsqueeze(0)

    def forward_with_coords(self, pixel_coords: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        norm = pixel_coords.clone()
        norm[:, :, 0] /= image_size[1]
        norm[:, :, 1] /= image_size[0]
        return self._encode(norm)


class SAMMaskDecoder(nn.Module):
    """Single-object SAM mask decoder (the `interactive_sam_mask_decoder`)."""

    def __init__(self, d_model: int = D_MODEL, num_multimask_outputs: int = NUM_MULTIMASK_OUTPUTS):
        super().__init__()
        self.num_mask_tokens = num_multimask_outputs + 1

        self.transformer = TwoWayTransformer(depth=2, embedding_dim=d_model, num_heads=8, mlp_dim=2048)

        self.iou_token = nn.Embedding(1, d_model)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, d_model)
        self.obj_score_token = nn.Embedding(1, d_model)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 4, kernel_size=2, stride=2),
            LayerNorm2d(d_model // 4), nn.GELU(),
            nn.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=2, stride=2), nn.GELU(),
        )
        self.conv_s0 = nn.Conv2d(d_model, d_model // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(d_model, d_model // 4, kernel_size=1)

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(d_model, d_model, d_model // 8, 3) for _ in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(d_model, d_model, self.num_mask_tokens, 3)
        self.pred_obj_score_head = MLP(d_model, d_model, 1, 3)

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,
                high_res_features: list[torch.Tensor] | None = None):
        B = sparse_prompt_embeddings.shape[0]
        tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)
        tokens = torch.cat([tokens.unsqueeze(0).expand(B, -1, -1), sparse_prompt_embeddings], dim=1)

        src = image_embeddings
        if src.shape[0] != B:
            src = src.expand(B, -1, -1, -1)
        src = src + dense_prompt_embeddings
        pos_src = image_pe.expand(B, -1, -1, -1)

        b, c, h, w = src.shape
        hs, src_out = self.transformer(src.flatten(2).permute(0, 2, 1), pos_src.flatten(2).permute(0, 2, 1), tokens)

        obj_score_token_out = hs[:, 0, :]
        iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2:2 + self.num_mask_tokens, :]

        src_out = src_out.permute(0, 2, 1).view(b, c, h, w)
        upscaled = _upscale_masks(self.output_upscaling, self.conv_s0, self.conv_s1, src_out, high_res_features)

        hyper_in = torch.stack([
            mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.output_hypernetworks_mlps)
        ], dim=1)
        masks = (hyper_in @ upscaled.flatten(2)).view(B, self.num_mask_tokens, upscaled.shape[2], upscaled.shape[3])
        iou_pred = self.iou_prediction_head(iou_token_out)
        object_score_logits = self.pred_obj_score_head(obj_score_token_out)

        # This project only ever wants the single-mask output (index 0), never the best-of-3
        # multimask selection (that only matters for ambiguous single-point clicks).
        return masks[:, 0:1], iou_pred[:, 0:1], mask_tokens_out[:, 0:1], object_score_logits


class SAMPromptEncoder(nn.Module):
    """Single-object prompt encoder (the `interactive_sam_prompt_encoder`). Always
    conditions on a mask (this project's only prompt kind here); `points`/`boxes` are
    dropped from the signature since a click is never issued -- see this module's
    docstring. `forward_sam_heads` below always supplies the fixed "no real point" dummy
    coordinate that the source synthesizes for the same case, since that dummy token is
    still architecturally required (its embedding is checkpoint-trained weight, not a
    no-op).
    """

    def __init__(self, d_model: int = D_MODEL, image_embedding_size: tuple[int, int] = (72, 72),
                 input_image_size: tuple[int, int] = (IMG_SIZE, IMG_SIZE)):
        super().__init__()
        self.embed_dim = d_model
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, d_model) for _ in range(4)])
        self.not_a_point_embed = nn.Embedding(1, d_model)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, stride=2), LayerNorm2d(4), nn.GELU(),
            nn.Conv2d(4, 16, kernel_size=2, stride=2), LayerNorm2d(16), nn.GELU(),
            nn.Conv2d(16, d_model, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, d_model)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size)

    def forward(self, dummy_point_coords: torch.Tensor, dummy_point_labels: torch.Tensor,
                masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = dummy_point_coords.shape[0]
        # The source pads an extra "not a point" token whenever no box is given (always
        # true here) -- a fixed architectural quirk of SAM's own prompt encoder, ported
        # as-is rather than "simplified", since the two point-embedding tokens it produces
        # are what the checkpoint's `not_a_point_embed` weight was actually trained against.
        coords = torch.cat([dummy_point_coords, torch.zeros_like(dummy_point_coords)], dim=1)
        labels = torch.cat([dummy_point_labels, -torch.ones_like(dummy_point_labels)], dim=1)
        pe = self.pe_layer.forward_with_coords(coords + 0.5, self.input_image_size)
        for i in range(4):
            pe[labels == i] += self.point_embeddings[i].weight
        invalid = labels == -1
        pe[invalid] = 0.0
        pe[invalid] += self.not_a_point_embed.weight
        sparse = pe

        dense = self.mask_downscaling(masks)
        return sparse, dense


def _upscale_masks(output_upscaling, conv_s0, conv_s1, src_out, high_res_features):
    """Shared deconv + high-res feature integration for both SAM decoders below."""
    dc1, ln1, act1, dc2, act2 = output_upscaling
    if high_res_features is not None:
        upscaled = act1(ln1(dc1(src_out) + conv_s1(high_res_features[1])))
        upscaled = act2(dc2(upscaled) + conv_s0(high_res_features[0]))
    else:
        upscaled = act2(dc2(act1(ln1(dc1(src_out)))))
    return upscaled


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) tensor (ConvNeXt-style)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def forward_sam_heads(
    backbone_features: torch.Tensor, prompt_encoder: SAMPromptEncoder, mask_decoder: SAMMaskDecoder,
    obj_ptr_proj: nn.Module, no_obj_fn, mask_inputs: torch.Tensor, high_res_features: list[torch.Tensor] | None = None,
):
    """Run the interactive SAM prompt encoder + mask decoder on an already mask-conditioned
    frame. Always synthesizes the fixed "no real point" dummy prompt -- see
    `SAMPromptEncoder`'s docstring.
    """
    device = backbone_features.device
    B = mask_inputs.shape[0]
    dummy_point_coords = torch.zeros(B, 1, 2, device=device, dtype=backbone_features.dtype)
    dummy_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

    prompt_size = (prompt_encoder.image_embedding_size[0] * 4, prompt_encoder.image_embedding_size[1] * 4)
    if mask_inputs.shape[-2:] != prompt_size:
        sam_mask_prompt = F.interpolate(mask_inputs, size=prompt_size, mode="bilinear", align_corners=False, antialias=True)
    else:
        sam_mask_prompt = mask_inputs

    sparse, dense = prompt_encoder(dummy_point_coords, dummy_point_labels, sam_mask_prompt)
    image_pe = prompt_encoder.get_dense_pe().to(dtype=backbone_features.dtype)

    low_res_multimasks, ious, sam_output_tokens, object_score_logits = mask_decoder(
        image_embeddings=backbone_features, image_pe=image_pe,
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, high_res_features=high_res_features,
    )

    is_obj_appearing = object_score_logits > 0
    low_res_multimasks = torch.where(
        is_obj_appearing[:, None, None], low_res_multimasks,
        torch.tensor(NO_OBJ_SCORE, device=device, dtype=low_res_multimasks.dtype))
    high_res_multimasks = F.interpolate(low_res_multimasks, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

    obj_ptr = obj_ptr_proj(sam_output_tokens[:, 0])
    obj_ptr = no_obj_fn(obj_ptr, is_obj_appearing)

    return low_res_multimasks, high_res_multimasks, obj_ptr, object_score_logits


def use_mask_as_output(
    backbone_features: torch.Tensor, high_res_features, mask_inputs: torch.Tensor, mask_downsample: nn.Module,
    prompt_encoder: SAMPromptEncoder, mask_decoder: SAMMaskDecoder, obj_ptr_proj: nn.Module, no_obj_fn,
):
    """Turn a ground-truth-quality mask (e.g. a fresh detection) directly into tracker
    output for this frame, still running it through the SAM heads once to get a matching
    object pointer. `out_scale`/`out_bias` are the mask decoder's own logit scale, fixed
    constants distinct from the memory encoder's own sigmoid scale/bias below.
    """
    out_scale, out_bias = 20.0, -10.0
    mask_inputs_float = mask_inputs.to(dtype=backbone_features.dtype)
    high_res_masks = mask_inputs_float * out_scale + out_bias
    low_res_masks = F.interpolate(high_res_masks, size=(IMG_SIZE // BACKBONE_STRIDE * 4,) * 2,
                                   mode="bilinear", align_corners=False, antialias=True)
    _, _, obj_ptr, _ = forward_sam_heads(
        backbone_features, prompt_encoder, mask_decoder, obj_ptr_proj, no_obj_fn,
        mask_inputs=mask_downsample(mask_inputs_float), high_res_features=high_res_features,
    )
    is_obj_appearing = torch.any(mask_inputs.flatten(1) > 0.0, dim=1)[..., None]
    alpha = is_obj_appearing.to(obj_ptr.dtype)
    object_score_logits = out_scale * alpha + out_bias
    return low_res_masks, high_res_masks, obj_ptr, object_score_logits


# --- Memory bank: mask/pixel fusion (`maskmem_backbone`) ---


class CXBlock(nn.Module):
    """ConvNeXt-style block used by the memory bank's `Fuser`."""

    def __init__(self, dim: int = D_MODEL, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(F.gelu(self.pwconv1(self.norm(x))))
        x = x * self.gamma
        return residual + x.permute(0, 3, 1, 2)


class MaskDownSampler(nn.Module):
    """Downsamples the (multiplexed) mask-plus-conditioning-channel stack to the memory
    bank's working resolution, resizing to a fixed `interpol_size` first if needed.
    """

    def __init__(self, out_dim: int, in_chans: int, channels: list[int], interpol_size: tuple[int, int] = (1152, 1152)):
        super().__init__()
        self.interpol_size = list(interpol_size)
        layers = []
        prev = in_chans
        for ch in channels:
            layers += [nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1), LayerNorm2d(ch), nn.GELU()]
            prev = ch
        layers.append(nn.Conv2d(prev, out_dim, kernel_size=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if list(x.shape[-2:]) != self.interpol_size:
            x = F.interpolate(x, size=self.interpol_size, mode="bilinear", align_corners=False, antialias=True)
        return self.encoder(x)


class Fuser(nn.Module):
    def __init__(self, dim: int = D_MODEL, num_layers: int = 2):
        super().__init__()
        self.layers = nn.Sequential(*[CXBlock(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal position encoding, same math as `sam31_vitdet_backbone`'s copy --
    duplicated (not imported) since the memory bank's copy is a separate, purely
    functional (no learned weights) instance in the checkpoint's own module tree.
    """

    def __init__(self, num_pos_feats: int):
        super().__init__()
        self.half_dim = num_pos_feats // 2
        self.temperature = 10000.0

    def _sincos(self, vals: torch.Tensor) -> torch.Tensor:
        freqs = self.temperature ** (2 * (torch.arange(self.half_dim, dtype=torch.float32, device=vals.device) // 2) / self.half_dim)
        raw = vals[..., None] * (2 * torch.pi) / freqs
        return torch.stack((raw[..., 0::2].sin(), raw[..., 1::2].cos()), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gy = torch.arange(H, dtype=torch.float32, device=x.device) / (H - 1 + 1e-6)
        gx = torch.arange(W, dtype=torch.float32, device=x.device) / (W - 1 + 1e-6)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        pe = torch.cat((self._sincos(yy), self._sincos(xx)), dim=-1).permute(2, 0, 1).unsqueeze(0)
        return pe.expand(B, -1, -1, -1).to(x.dtype)


class MemoryBackbone(nn.Module):
    """`maskmem_backbone`: downsamples the (multiplexed) mask stack, fuses it with the
    frame's own pixel features, and attaches a position encoding. SAM 3.1's checkpoint
    needs no output-channel compression (`out_dim == d_model` always), unlike the
    older SAM3 tracker's version -- so that option is dropped here rather than ported
    as an unused branch.
    """

    def __init__(self, d_model: int, in_chans: int, channels: list[int]):
        super().__init__()
        self.mask_downsampler = MaskDownSampler(d_model, in_chans=in_chans, channels=channels)
        self.pix_feat_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.fuser = Fuser(d_model, num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=d_model)

    def forward(self, image_features: torch.Tensor, mux_input: torch.Tensor) -> dict:
        mask_features = self.mask_downsampler(mux_input)
        if mask_features.shape[-2:] != image_features.shape[-2:]:
            mask_features = F.interpolate(mask_features, size=image_features.shape[-2:], mode="bilinear", align_corners=False)
        features = self.pix_feat_proj(image_features) + mask_features
        features = self.fuser(features)
        pos = self.position_encoding(features).to(features.dtype)
        return {"vision_features": features, "vision_pos_enc": [pos]}


# --- Decoupled memory attention (SAM 3.1's cross-attention over both raw image features
# and mask-derived memory, kept as separate keys/values rather than one merged stream) ---


class DecoupledMemoryAttnLayer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, num_heads: int = 8, dim_ff: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn_q_proj = nn.Linear(d_model, d_model)
        self.self_attn_k_proj = nn.Linear(d_model, d_model)
        self.self_attn_v_proj = nn.Linear(d_model, d_model)
        self.self_attn_out_proj = nn.Linear(d_model, d_model)
        self.cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.cross_attn_k_proj = nn.Linear(d_model, d_model)
        self.cross_attn_v_proj = nn.Linear(d_model, d_model)
        self.cross_attn_out_proj = nn.Linear(d_model, d_model)
        self.image_cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.image_cross_attn_k_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, Lq, C = q.shape
        head_dim = C // self.num_heads
        q = q.view(B, Lq, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, k.shape[1], self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, v.shape[1], self.num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(B, Lq, -1)

    def forward(self, image: torch.Tensor, x: torch.Tensor, memory_image: torch.Tensor, memory: torch.Tensor,
                memory_image_pos: torch.Tensor | None = None, rope: torch.Tensor | None = None,
                num_k_exclude_rope: int = 0):
        normed = self.norm1(x)
        q = self.self_attn_q_proj(normed)
        k = self.self_attn_k_proj(normed)
        v = self.self_attn_v_proj(normed)
        if rope is not None:
            q, k = apply_rope_memory(q, k, rope, self.num_heads, 0)
        x = x + self.self_attn_out_proj(self._attend(q, k, v))

        # Decoupled cross-attention: current-frame features attend to raw past-frame image
        # features and mask-derived memory as separate key/value streams, fused only in q/k.
        normed = self.norm2(x)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(normed)
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if memory_image_pos is not None:
            k = k + memory_image_pos
        v = self.cross_attn_v_proj(memory)
        if rope is not None:
            q, k = apply_rope_memory(q, k, rope, self.num_heads, num_k_exclude_rope)
        x = x + self.cross_attn_out_proj(self._attend(q, k, v))

        x = x + self.linear2(F.gelu(self.linear1(self.norm3(x))))
        return image, x


class DecoupledMemoryEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL, num_heads: int = 8, dim_ff: int = 2048, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([DecoupledMemoryAttnLayer(d_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        hw = IMG_SIZE // PATCH_SIZE
        self.register_buffer("_rope", rope_2d(hw, hw, d_model // num_heads), persistent=False)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_pos: torch.Tensor | None = None,
                src_pos: torch.Tensor | None = None, num_k_exclude_rope: int = 0,
                memory_image: torch.Tensor | None = None, memory_image_pos: torch.Tensor | None = None):
        image = x  # constant residual stream for the decoupled image cross-attention
        output = x
        if src_pos is not None:
            output = output + 0.1 * src_pos

        B, _, C = x.shape
        rope = self._rope.to(device=x.device)

        if memory_image is None:
            # No raw past-frame image features recorded (shouldn't happen once any memory
            # exists, but matches the source's defensive fallback): reuse the spatial part
            # of `memory` itself as a stand-in.
            num_spatial = memory.shape[1] - num_k_exclude_rope
            memory_image = memory[:, :num_spatial]
            memory_image_pos = memory_pos[:, :num_spatial] if memory_pos is not None else None
        if memory_image.shape[1] < memory.shape[1]:
            pad_len = memory.shape[1] - memory_image.shape[1]
            pad = torch.zeros(B, pad_len, C, device=memory.device, dtype=memory.dtype)
            memory_image = torch.cat([memory_image, pad], dim=1)
            if memory_image_pos is not None:
                ptr_pos = memory_pos[:, -pad_len:] if memory_pos is not None else torch.zeros_like(pad)
                memory_image_pos = torch.cat([memory_image_pos, ptr_pos], dim=1)

        for layer in self.layers:
            image, output = layer(image, output, memory_image, memory,
                                   memory_image_pos=memory_image_pos, rope=rope,
                                   num_k_exclude_rope=num_k_exclude_rope)

        return self.norm(output)


class DecoupledMemoryTransformer(nn.Module):
    """Thin wrapper matching the checkpoint's `transformer.encoder.*` module tree shape."""

    def __init__(self, d_model: int = D_MODEL, num_heads: int = 8, dim_ff: int = 2048, num_layers: int = 4):
        super().__init__()
        self.encoder = DecoupledMemoryEncoder(d_model, num_heads, dim_ff, num_layers)


# --- Multiplex (16-object) propagation decoder ---


class MultiplexMaskDecoder(nn.Module):
    """SAM mask decoder that predicts masks for all `num_multiplex` multiplex slots in one
    pass. Always `multimask_outputs_only` (no single-mask token, unlike `SAMMaskDecoder`):
    hypernetwork MLPs are shared across slots, applied per mask-output index.
    Token order: [obj_score_token(M), iou_token(M), mask_tokens(M*T)].
    """

    def __init__(self, d_model: int = D_MODEL, num_multiplex: int = NUM_MULTIPLEX,
                 num_multimask_outputs: int = NUM_MULTIMASK_OUTPUTS):
        super().__init__()
        self.num_multiplex = num_multiplex
        self.num_mask_output_per_object = num_multimask_outputs
        total_mask_tokens = num_multiplex * self.num_mask_output_per_object

        self.transformer = TwoWayTransformer(depth=2, embedding_dim=d_model, num_heads=8, mlp_dim=2048)

        self.obj_score_token = nn.Embedding(num_multiplex, d_model)
        self.iou_token = nn.Embedding(num_multiplex, d_model)
        self.mask_tokens = nn.Embedding(total_mask_tokens, d_model)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 4, kernel_size=2, stride=2),
            LayerNorm2d(d_model // 4), nn.GELU(),
            nn.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=2, stride=2), nn.GELU(),
        )
        self.conv_s0 = nn.Conv2d(d_model, d_model // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(d_model, d_model // 4, kernel_size=1)

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(d_model, d_model, d_model // 8, 3) for _ in range(self.num_mask_output_per_object)
        ])
        self.iou_prediction_head = MLP(d_model, d_model, self.num_mask_output_per_object, 3)
        self.pred_obj_score_head = MLP(d_model, d_model, 1, 3)

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,
                high_res_features: list[torch.Tensor] | None, extra_per_object_embeddings: torch.Tensor):
        B = sparse_prompt_embeddings.shape[0]
        M, T = self.num_multiplex, self.num_mask_output_per_object

        mask_tokens = self.mask_tokens.weight.view(1, M, T, -1).expand(B, -1, -1, -1) + extra_per_object_embeddings.unsqueeze(2)
        mask_tokens = mask_tokens.flatten(1, 2)  # [B, M*T, C]
        other_tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight], dim=0).unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([other_tokens, mask_tokens, sparse_prompt_embeddings], dim=1)

        src = image_embeddings
        if src.shape[0] != B:
            src = src.expand(B, -1, -1, -1)
        src = src + dense_prompt_embeddings
        pos_src = image_pe.expand(B, -1, -1, -1)

        b, c, h, w = src.shape
        hs, src_out = self.transformer(src.flatten(2).permute(0, 2, 1), pos_src.flatten(2).permute(0, 2, 1), tokens)

        obj_score_token_out = hs[:, :M]
        iou_token_out = hs[:, M:2 * M]
        mask_tokens_out = hs[:, 2 * M:2 * M + M * T]

        src_out = src_out.permute(0, 2, 1).view(b, c, h, w)
        upscaled = _upscale_masks(self.output_upscaling, self.conv_s0, self.conv_s1, src_out, high_res_features)

        mask_tokens_2d = mask_tokens_out.view(B, M, T, -1)
        hyper_in = torch.stack([
            self.output_hypernetworks_mlps[i](mask_tokens_2d[:, :, i, :]) for i in range(T)
        ], dim=2)  # [B, M, T, C//8]

        masks = torch.bmm(hyper_in.flatten(1, 2), upscaled.flatten(2)).view(b, M, T, upscaled.shape[2], upscaled.shape[3])
        iou_pred = self.iou_prediction_head(iou_token_out).view(b, M, T)
        object_score_logits = self.pred_obj_score_head(obj_score_token_out)  # [B, M, 1]
        sam_tokens_out = mask_tokens_2d[:, :, 0:1]  # [B, M, 1, C]

        return masks, iou_pred, sam_tokens_out, object_score_logits


# --- Multiplex bucket bookkeeping ---


class MultiplexState:
    """Tracks which multiplex "slot" each tracked object occupies. SAM 3.1's decoders
    process objects in fixed-size groups of `multiplex_count` (16); with more than 16
    objects, a second "bucket" of 16 slots is added. `mux`/`demux` move between the
    flat per-object view the rest of this project's code wants and the
    (num_buckets, multiplex_count, ...) view the decoders operate on.
    """

    def __init__(self, num_objects: int, multiplex_count: int, device: torch.device, dtype: torch.dtype):
        self.multiplex_count = multiplex_count
        self.device = device
        self.dtype = dtype
        self._build(num_objects)

    def mux(self, x: torch.Tensor) -> torch.Tensor:
        """[N_obj, ...] -> [num_buckets, multiplex_count, ...]"""
        out_shape = (self.num_buckets, self.multiplex_count) + x.shape[1:]
        return (self.mux_matrix.to(device=x.device, dtype=x.dtype) @ x.reshape(self.total_valid_entries, -1)).view(out_shape)

    def demux(self, x: torch.Tensor) -> torch.Tensor:
        """[num_buckets, multiplex_count, ...] -> [N_obj, ...]"""
        out_shape = (self.total_valid_entries,) + x.shape[2:]
        flat = x.reshape(self.num_buckets * self.multiplex_count, -1)
        return (self.demux_matrix.to(device=x.device, dtype=x.dtype) @ flat).view(out_shape)

    def get_valid_object_mask(self) -> torch.Tensor:
        """[num_buckets, multiplex_count] bool, True for slots holding a real object."""
        return (self.mux_matrix.sum(dim=1) > 0).reshape(self.num_buckets, self.multiplex_count)

    def _build(self, num_objects: int) -> None:
        M = self.multiplex_count
        self.num_buckets = (num_objects + M - 1) // M
        self.total_valid_entries = num_objects
        total_slots = self.num_buckets * M
        self.mux_matrix = torch.zeros(total_slots, num_objects, device=self.device, dtype=self.dtype)
        self.demux_matrix = torch.zeros(num_objects, total_slots, device=self.device, dtype=self.dtype)
        oids = torch.arange(num_objects, device=self.device)
        slots = (oids // M) * M + (oids % M)
        self.mux_matrix[slots, oids] = 1.0
        self.demux_matrix[oids, slots] = 1.0

    def add_objects(self, n_new: int) -> None:
        """Grow to accommodate `n_new` additional objects (rebuilds the mux/demux matrices)."""
        self._build(self.total_valid_entries + n_new)


class Sam31Tracker(nn.Module):
    """SAM 3.1 multiplex tracker. Matches `tracker.model.*` exactly (prefix stripped --
    note the extra `.model` level the checkpoint uses here, unlike `detector.*`).

    Call pattern this project actually uses: `track_video_with_detection` with
    `initial_masks=None`, driven entirely by a `detect_fn` built from
    `sam31_detector.Sam31Detector.forward_from_trunk` over `human_prompt`/`object_prompt`
    text embeddings (wired up in `sam31_adapter.py`, not this file).
    """

    def __init__(self):
        super().__init__()

        self.transformer = DecoupledMemoryTransformer(D_MODEL, num_heads=8, dim_ff=2048, num_layers=4)
        self.sam_mask_decoder = MultiplexMaskDecoder(D_MODEL, NUM_MULTIPLEX, NUM_MULTIMASK_OUTPUTS)
        self.interactive_sam_mask_decoder = SAMMaskDecoder(D_MODEL, NUM_MULTIMASK_OUTPUTS)
        self.interactive_sam_prompt_encoder = SAMPromptEncoder(D_MODEL)
        # Mask-plus-conditioning-channel stack: one mask channel + one conditioning channel
        # per multiplex slot -- see _encode_new_memory.
        self.maskmem_backbone = MemoryBackbone(D_MODEL, in_chans=NUM_MULTIPLEX * 2, channels=[16, 64, 256, 1024])

        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(NUM_MASKMEM, 1, 1, D_MODEL))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(NUM_MULTIPLEX, D_MODEL))
        self.interactivity_no_mem_embed = nn.Parameter(torch.zeros(1, 1, D_MODEL))

        self.obj_ptr_proj = MLP(D_MODEL, D_MODEL, D_MODEL, 3)
        self.obj_ptr_tpos_proj = nn.Linear(D_MODEL, D_MODEL)
        self.no_obj_ptr_linear = nn.Linear(D_MODEL, D_MODEL)
        self.interactive_obj_ptr_proj = MLP(D_MODEL, D_MODEL, D_MODEL, 3)

        self.interactive_mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        self.output_valid_embed = nn.Parameter(torch.zeros(NUM_MULTIPLEX, D_MODEL))
        self.output_invalid_embed = nn.Parameter(torch.zeros(NUM_MULTIPLEX, D_MODEL))

        self.image_pe_layer = PositionEmbeddingRandom(D_MODEL // 2)

    # --- shared small helpers ---

    def _no_obj_blend(self, obj_ptr: torch.Tensor, is_obj: torch.Tensor) -> torch.Tensor:
        alpha = is_obj.to(obj_ptr.dtype)
        return torch.lerp(self.no_obj_ptr_linear(obj_ptr), obj_ptr, alpha)

    def _use_mask_as_output(self, backbone_features: torch.Tensor, high_res_features, mask_inputs: torch.Tensor):
        return use_mask_as_output(
            backbone_features, high_res_features, mask_inputs, self.interactive_mask_downsample,
            self.interactive_sam_prompt_encoder, self.interactive_sam_mask_decoder,
            self.interactive_obj_ptr_proj, self._no_obj_blend,
        )

    def _compute_backbone_frame(self, backbone_fn, frame: torch.Tensor, frame_idx: int | None = None):
        vision_feats, vision_pos, feat_sizes, features, trunk_out = _compute_backbone(backbone_fn, frame, frame_idx)
        return vision_feats, vision_pos, feat_sizes, list(features[:-1]), trunk_out

    # --- memory-conditioned feature prep (propagation path) ---

    def _prepare_memory_conditioned_features(
        self, frame_idx: int, is_init_cond_frame: bool, current_vision_feats, current_vision_pos_embeds,
        feat_sizes, output_dict: dict, num_frames: int, multiplex_state: MultiplexState,
    ) -> torch.Tensor:
        C = D_MODEL
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        num_buc = multiplex_state.num_buckets

        if is_init_cond_frame:
            pix_feat = current_vision_feats[-1] + self.interactivity_no_mem_embed
            return to_spatial(pix_feat, H, W)

        to_cat_memory, to_cat_memory_pos, to_cat_image_feat, to_cat_image_pos, cond_outputs = collect_memory_tokens(
            output_dict, frame_idx, NUM_MASKMEM, self.maskmem_tpos_enc, device,
            collect_image_feats=True, tpos_v2=True, num_buckets=num_buc)

        max_obj_ptrs = min(num_frames, MAX_OBJ_PTRS_IN_ENCODER)
        pos_and_ptrs = []
        for t, out in cond_outputs.items():
            if t <= frame_idx and "obj_ptr" in out:
                pos_and_ptrs.append(((frame_idx - t), _pad_to_buckets(out["obj_ptr"].to(device), num_buc)))
        for t_diff in range(1, max_obj_ptrs):
            t = frame_idx - t_diff
            if t < 0:
                break
            out = output_dict["non_cond_frame_outputs"].get(t, None)
            if out is not None and "obj_ptr" in out:
                pos_and_ptrs.append((t_diff, _pad_to_buckets(out["obj_ptr"].to(device), num_buc)))

        num_obj_ptr_tokens = 0
        if pos_and_ptrs:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=1)  # [num_buckets, N, M, C]
            B_ptr, N_ptrs, M = obj_ptrs.shape[0], obj_ptrs.shape[1], obj_ptrs.shape[2]
            obj_ptrs = obj_ptrs.reshape(B_ptr, N_ptrs * M, -1)
            obj_pos = compute_tpos_enc(list(pos_list), device, D_MODEL, self.obj_ptr_tpos_proj,
                                        max_abs_pos=max_obj_ptrs, dtype=current_vision_feats[-1].dtype)
            obj_pos = obj_pos.unsqueeze(0).expand(B_ptr, -1, -1)
            obj_pos = obj_pos.unsqueeze(2).expand(-1, -1, M, -1).reshape(B_ptr, N_ptrs * M, -1)
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[1]

        if not to_cat_memory:
            pix_feat = current_vision_feats[-1] + self.interactivity_no_mem_embed
            return to_spatial(pix_feat, H, W)

        memory = torch.cat(to_cat_memory, dim=1)
        memory_pos = torch.cat(to_cat_memory_pos, dim=1)

        mem_B = memory.shape[0]
        x = current_vision_feats[-1]
        x_pos = current_vision_pos_embeds[-1]
        if x.shape[0] < mem_B:
            x = x.expand(mem_B, -1, -1)
            x_pos = x_pos.expand(mem_B, -1, -1)

        memory_image = memory_image_pos = None
        if to_cat_image_feat:
            memory_image = torch.cat(to_cat_image_feat, dim=1).to(dtype=x.dtype)
            memory_image_pos = torch.cat(to_cat_image_pos, dim=1).to(dtype=x.dtype)
            if memory_image.shape[0] < mem_B:
                memory_image = memory_image.expand(mem_B, -1, -1)
                memory_image_pos = memory_image_pos.expand(mem_B, -1, -1)

        pix_feat_with_mem = self.transformer.encoder(
            x=x, memory=memory.to(dtype=x.dtype), memory_pos=memory_pos.to(dtype=x.dtype), src_pos=x_pos,
            num_k_exclude_rope=num_obj_ptr_tokens, memory_image=memory_image, memory_image_pos=memory_image_pos,
        )
        return to_spatial(pix_feat_with_mem, H, W)

    # --- memory encoding ---

    def _encode_new_memory(
        self, pix_feat: torch.Tensor, pred_masks_high_res: torch.Tensor, object_score_logits: torch.Tensor,
        multiplex_state: MultiplexState, is_mask_from_pts: bool = False, is_conditioning: bool = False,
        cond_obj_mask: torch.Tensor | None = None,
    ):
        if is_mask_from_pts:
            mask_for_mem = (pred_masks_high_res > 0).to(pix_feat.dtype)
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        mask_for_mem = mask_for_mem * SIGMOID_SCALE_FOR_MEM_ENC + SIGMOID_BIAS_FOR_MEM_ENC

        mux_masks = multiplex_state.mux(mask_for_mem[:, 0])  # [num_buckets, M, H, W]

        # Conditioning channel: 1.0 = clean detection mask (trust it), 0.0 = propagated
        # (noisier) mask -- tells the memory bank how much to trust each slot's mask this frame.
        N_obj = mask_for_mem.shape[0]
        cond_values = torch.full((N_obj,), 0.0, device=mask_for_mem.device, dtype=mask_for_mem.dtype)
        if is_conditioning:
            cond_values[:] = 1.0
        elif cond_obj_mask is not None:
            cond_values[cond_obj_mask] = 1.0
        cond_spatial = cond_values.view(-1, 1, 1, 1).expand_as(mask_for_mem[:, 0:1, :, :]).squeeze(1)
        mux_cond = multiplex_state.mux(cond_spatial)
        mux_input = torch.cat([mux_masks, mux_cond], dim=1)  # [num_buckets, 2*M, H, W]

        maskmem_out = self.maskmem_backbone(pix_feat, mux_input)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        # Blend in the learned "no object" embedding for slots whose object is occluded
        # this frame, weighted by how many of a bucket's slots are actually occluded.
        is_obj = (object_score_logits > 0).float()  # [N_obj, 1]
        mux_is_obj = multiplex_state.mux(is_obj)  # [num_buckets, M, 1]
        no_obj_spatial = self.no_obj_embed_spatial.unsqueeze(0)[..., None, None]  # [1, M, C, 1, 1]
        alpha = mux_is_obj[..., None, None]
        per_slot_no_obj = ((1 - alpha) * no_obj_spatial).sum(dim=1)  # [num_buckets, C, 1, 1]
        maskmem_features = maskmem_features + per_slot_no_obj.expand_as(maskmem_features)

        return maskmem_features, maskmem_pos_enc

    def _deferred_memory_encode(
        self, current_out: dict, N_obj: int, vision_feats, feat_sizes, mux_state: MultiplexState,
        cond_obj_mask: torch.Tensor | None = None,
    ) -> None:
        """Re-encode memory after `current_out["pred_masks"]` was modified in place
        (reconditioning, newly-added objects, or occlusion suppression) -- keeps the
        memory bank consistent with whatever mask ends up being trusted for this frame.
        """
        low_res_masks = current_out["pred_masks"]  # [N_obj, 1, H_low, W_low]

        if N_obj > 1:
            # Suppress overlapping low-confidence pixels between objects so one object's
            # mask can't "leak" into another's memory encoding.
            lr = low_res_masks.squeeze(1)
            max_obj = torch.argmax(lr, dim=0, keepdim=True)
            batch_inds = torch.arange(N_obj, device=lr.device)[:, None, None]
            pixel_nol = torch.where(max_obj == batch_inds, lr, torch.clamp(lr, max=-10.0))
            area_before = (lr > 0).sum(dim=(-1, -2)).float().clamp(min=1)
            area_after = (pixel_nol > 0).sum(dim=(-1, -2)).float()
            shrink_ok = (area_after / area_before) >= 0.3
            low_res_masks = torch.where(
                shrink_ok[:, None, None, None].expand_as(low_res_masks), low_res_masks, torch.clamp(low_res_masks, max=-10.0))

        interpol_size = self.maskmem_backbone.mask_downsampler.interpol_size
        mem_masks = F.interpolate(low_res_masks, size=interpol_size, mode="bilinear", align_corners=False)
        obj_scores = torch.where((mem_masks > 0).any(dim=(-1, -2)), 10.0, -10.0)

        pix_feat = to_spatial(vision_feats[-1], feat_sizes[-1][0], feat_sizes[-1][1])
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            pix_feat=pix_feat, pred_masks_high_res=mem_masks, object_score_logits=obj_scores,
            multiplex_state=mux_state, cond_obj_mask=cond_obj_mask)
        current_out["maskmem_features"] = maskmem_features
        current_out["maskmem_pos_enc"] = maskmem_pos_enc

    # --- propagation (multiplex decoder) ---

    def _forward_propagation(self, backbone_features: torch.Tensor, high_res_features, multiplex_state: MultiplexState):
        B = backbone_features.shape[0]
        device = backbone_features.device

        valid_mask = multiplex_state.get_valid_object_mask().unsqueeze(-1).float().to(dtype=backbone_features.dtype)
        extra_embed = valid_mask * self.output_valid_embed.unsqueeze(0) + (1 - valid_mask) * self.output_invalid_embed.unsqueeze(0)

        image_pe = self.image_pe_layer(backbone_features.shape[-2:], device=device).to(dtype=backbone_features.dtype)

        masks, iou_pred, sam_tokens_out, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features, image_pe=image_pe,
            sparse_prompt_embeddings=torch.empty(B, 0, D_MODEL, device=device, dtype=backbone_features.dtype),
            dense_prompt_embeddings=torch.zeros(B, D_MODEL, *backbone_features.shape[-2:], device=device, dtype=backbone_features.dtype),
            high_res_features=high_res_features, extra_per_object_embeddings=extra_embed.expand(B, -1, -1),
        )
        # masks: [num_buckets, M, T, H, W] -> per-object [N_obj, T, H, W]
        masks_obj = multiplex_state.demux(masks)
        iou_obj = multiplex_state.demux(iou_pred)
        score_obj = multiplex_state.demux(object_score_logits)
        tokens_obj = multiplex_state.demux(sam_tokens_out)

        best_idx = torch.argmax(iou_obj, dim=-1)
        N_obj = masks_obj.shape[0]
        obj_range = torch.arange(N_obj, device=device)
        low_res_masks = masks_obj[obj_range, best_idx].unsqueeze(1)
        is_obj = score_obj > 0
        low_res_masks = torch.where(is_obj[:, :, None, None], low_res_masks,
                                     torch.tensor(NO_OBJ_SCORE, device=device, dtype=low_res_masks.dtype))
        high_res_masks = F.interpolate(low_res_masks.float(), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        sam_token = tokens_obj[:, 0]
        obj_ptr = self.obj_ptr_proj(sam_token)
        is_obj_f = (score_obj > 0).float()
        no_obj = self.no_obj_ptr_linear(obj_ptr)
        obj_ptr = is_obj_f * obj_ptr + (1 - is_obj_f) * no_obj
        obj_ptr_muxed = multiplex_state.mux(obj_ptr)

        return low_res_masks, high_res_masks, obj_ptr_muxed, score_obj

    # --- per-frame step ---

    def track_step(
        self, frame_idx: int, is_init_cond_frame: bool, current_vision_feats, current_vision_pos_embeds,
        feat_sizes, mask_inputs: torch.Tensor | None, output_dict: dict, num_frames: int,
        interactive_high_res=None, interactive_backbone: torch.Tensor | None = None,
        propagation_high_res=None, multiplex_state: MultiplexState | None = None, run_mem_encoder: bool = True,
    ) -> dict:
        current_out: dict = {}
        H, W = feat_sizes[-1]

        if mask_inputs is not None:
            # Conditioning frame: seed tracking state from a detected mask. Prefer the
            # backbone's dedicated "interactive" FPN features when available (matches the
            # source's tracker_mode="interactive" pass); fall back to propagation features
            # if called in a context that never computed them.
            if interactive_backbone is not None:
                pix_flat = interactive_backbone.flatten(2).permute(0, 2, 1) + self.interactivity_no_mem_embed
                pix_feat = to_spatial(pix_flat, H, W)
                hi_res = interactive_high_res
            else:
                pix_feat = to_spatial(current_vision_feats[-1], H, W)
                hi_res = propagation_high_res
            sam_outputs = self._use_mask_as_output(pix_feat, hi_res, mask_inputs)
        else:
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats, current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes, output_dict=output_dict, num_frames=num_frames, multiplex_state=multiplex_state,
            )
            sam_outputs = self._forward_propagation(pix_feat_with_mem, propagation_high_res, multiplex_state=multiplex_state)

        low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs

        if multiplex_state is not None and obj_ptr.dim() == 2:
            obj_ptr = multiplex_state.mux(obj_ptr)  # interactive path returns [N_obj, C]; store muxed like propagation

        if run_mem_encoder:
            pix_feat = to_spatial(current_vision_feats[-1], H, W)
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                pix_feat=pix_feat, pred_masks_high_res=high_res_masks, object_score_logits=object_score_logits,
                is_mask_from_pts=(mask_inputs is not None), multiplex_state=multiplex_state,
                is_conditioning=(mask_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        # Raw propagation image features, kept for the next frame's decoupled cross-attention.
        current_out["image_features"] = current_vision_feats[-1]
        current_out["image_pos_enc"] = current_vision_pos_embeds[-1]
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits
        return current_out

    # --- occlusion / reconditioning bookkeeping ---

    @staticmethod
    def _suppress_recently_occluded(low_res_masks: torch.Tensor, last_occluded: torch.Tensor, frame_idx: int,
                                     threshold: float = 0.3) -> torch.Tensor:
        """Between two overlapping object masks, suppress whichever object was occluded
        more recently -- prevents a just-reappeared object's corrupted mask from bleeding
        into a currently-stable neighbor. `last_occluded` is updated in place.
        """
        N_obj = low_res_masks.shape[0]
        if N_obj <= 1:
            return low_res_masks
        binary = low_res_masks[:, 0] > 0
        iou = _compute_mask_overlap(low_res_masks[:, 0], low_res_masks[:, 0])
        overlapping = torch.triu(iou >= threshold, diagonal=1)
        last_occ_i = last_occluded.unsqueeze(1)
        last_occ_j = last_occluded.unsqueeze(0)
        suppress_i = overlapping & (last_occ_i > last_occ_j) & (last_occ_j > -1)
        suppress_j = overlapping & (last_occ_j > last_occ_i) & (last_occ_i > -1)
        to_suppress = suppress_i.any(dim=1) | suppress_j.any(dim=0)
        is_empty = ~binary.any(dim=(-1, -2))
        newly_occluded = is_empty | to_suppress
        last_occluded[newly_occluded] = frame_idx
        low_res_masks[to_suppress] = -10.0
        return low_res_masks

    def _add_detected_objects(self, new_masks: torch.Tensor, mux_state: MultiplexState, vision_feats, feat_sizes, current_out: dict) -> None:
        """Grow the multiplex state with newly detected objects, append their masks to
        `current_out`, and re-encode memory marking them as clean (conditioning) detections.
        """
        n_old = mux_state.total_valid_entries
        mux_state.add_objects(new_masks.shape[0])
        N_obj = mux_state.total_valid_entries
        for k in ("pred_masks", "pred_masks_high_res"):
            det = F.interpolate(new_masks.unsqueeze(1), size=current_out[k].shape[-2:], mode="bilinear", align_corners=False)
            current_out[k] = torch.cat([current_out[k], det], dim=0)
        cond_mask = torch.zeros(N_obj, dtype=torch.bool, device=new_masks.device)
        cond_mask[n_old:] = True
        self._deferred_memory_encode(current_out, N_obj, vision_feats, feat_sizes, mux_state, cond_obj_mask=cond_mask)

    def _condition_with_masks(
        self, masks: torch.Tensor, frame_idx: int, vision_feats, vision_pos, feat_sizes, high_res_prop,
        output_dict: dict, N: int, mux_state: MultiplexState, backbone_obj, frame: torch.Tensor,
        trunk_out: torch.Tensor, threshold: float = 0.5,
    ) -> dict:
        """Seed tracking state for one or more objects from detected masks on `frame_idx`."""
        mask_input = F.interpolate(masks if masks.dim() == 4 else masks.unsqueeze(1),
                                    size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        mask_input = (mask_input > threshold).to(masks.dtype)
        _, _, tracker_features, _ = backbone_obj(frame, tracker_mode=TrackerMode.INTERACTIVE, cached_trunk=trunk_out, tracker_only=True)
        hi_res, lo_feat = tracker_features[:-1], tracker_features[-1]
        current_out = self.track_step(
            frame_idx=frame_idx, is_init_cond_frame=True, current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=mask_input,
            output_dict=output_dict, num_frames=N, interactive_high_res=hi_res, interactive_backbone=lo_feat,
            propagation_high_res=high_res_prop, multiplex_state=mux_state, run_mem_encoder=True,
        )
        output_dict["cond_frame_outputs"][frame_idx] = current_out
        return current_out

    def _match_and_add_detections(
        self, det_masks: torch.Tensor, det_scores: torch.Tensor | None, current_out: dict, mux_state: MultiplexState,
        vision_feats, feat_sizes, device: torch.device, max_objects: int = 0, keep_alive: dict | None = None,
    ) -> list[float]:
        """Match this frame's fresh detections against currently tracked masks: refresh
        high-confidence matches (reconditioning), add unmatched detections as new objects,
        and update each object's `keep_alive` hysteresis counter (+1 matched, -1 unmatched,
        clamped to [-4, 8]) -- the counter, not a single missed frame, is what actually
        hides an object's mask (see `track_video_with_detection`), which is what prevents
        one-frame tracking glitches from flickering the object in and out.
        """
        N_obj = mux_state.total_valid_entries
        if det_masks.shape[0] == 0:
            if keep_alive is not None:
                for i in range(N_obj):
                    keep_alive[i] = max(-4, keep_alive.get(i, 0) - 1)
            return []

        trk_masks = current_out["pred_masks"][:, 0]
        det_resized = F.interpolate(det_masks.unsqueeze(1), size=trk_masks.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        overlap = _compute_mask_overlap(det_resized, trk_masks)

        matched: set[int] = set()
        if overlap.shape[1] > 0:
            matched = set((overlap >= 0.5).any(dim=0).nonzero(as_tuple=True)[0].tolist())
        if keep_alive is not None:
            for i in range(N_obj):
                keep_alive[i] = min(8, keep_alive.get(i, 0) + 1) if i in matched else max(-4, keep_alive.get(i, 0) - 1)

        reconditioned = False
        HIGH_CONF = 0.8
        if det_scores is not None and overlap.shape[1] > 0:
            for det_idx in range(overlap.shape[0]):
                if det_scores[det_idx] < HIGH_CONF:
                    continue
                best_trk = overlap[det_idx].argmax().item()
                if overlap[det_idx, best_trk] >= 0.5:
                    current_out["pred_masks"][best_trk] = det_resized[det_idx].unsqueeze(0)
                    det_hr = F.interpolate(det_masks[det_idx:det_idx + 1].unsqueeze(1),
                                            size=current_out["pred_masks_high_res"].shape[-2:], mode="bilinear", align_corners=False)
                    current_out["pred_masks_high_res"][best_trk] = det_hr[0]
                    reconditioned = True

        if reconditioned:
            self._deferred_memory_encode(current_out, N_obj, vision_feats, feat_sizes, mux_state)

        if max_objects > 0 and N_obj >= max_objects:
            return []
        max_overlap = overlap.max(dim=1)[0] if overlap.shape[1] > 0 else torch.zeros(overlap.shape[0], device=device)
        new_dets = max_overlap < 0.5
        if not new_dets.any():
            return []
        if max_objects > 0:
            slots = max_objects - N_obj
            new_dets = new_dets & (torch.cumsum(new_dets.int(), 0) <= slots)
        self._add_detected_objects(det_masks[new_dets], mux_state, vision_feats, feat_sizes, current_out)
        if keep_alive is not None:
            for i in range(N_obj, mux_state.total_valid_entries):
                keep_alive[i] = 1
        return det_scores[new_dets].tolist() if det_scores is not None else [0.0] * int(new_dets.sum().item())

    # --- top-level entry point ---

    def track_video_with_detection(
        self, backbone_fn, images: torch.Tensor, initial_masks: torch.Tensor | None, detect_fn=None,
        new_det_thresh: float = 0.5, max_objects: int = 0, detect_interval: int = 1,
        backbone_obj=None, target_device: torch.device | None = None, target_dtype: torch.dtype | None = None,
        progress_label: str = StageName.STAGE_1_MASK_AND_TRACK.title,
    ) -> dict:
        """Track a clip with per-frame text-prompted detection (this project never passes
        `initial_masks`; new objects are found purely by `detect_fn`, built in
        `sam31_adapter.py` from `human_prompt`/`object_prompt`).

        Returns {"packed_masks": [N_frames, max_N_obj, H, W//8] bit-packed uint8 (or None
        if nothing was ever tracked), "n_frames": N, "scores": per-object first-detection
        confidence}.
        """
        if max_objects <= 0 or max_objects > INTERNAL_MAX_OBJECTS:
            max_objects = INTERNAL_MAX_OBJECTS
        N = images.shape[0]
        device = target_device if target_device is not None else images.device
        dt = target_dtype if target_dtype is not None else images.dtype
        size = IMG_SIZE
        output_dict: dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        all_masks: list[torch.Tensor | None] = []
        mux_state: MultiplexState | None = None
        if initial_masks is not None:
            mux_state = MultiplexState(initial_masks.shape[0], NUM_MULTIPLEX, device, dt)
        obj_scores: list[float] = []
        keep_alive: dict[int, int] | None = {} if detect_fn is not None else None
        last_occluded = torch.empty(0, device=device, dtype=torch.long)

        for frame_idx in frame_progress(range(N), total=N, label=progress_label):
            frame = _prep_frame(images, slice(frame_idx, frame_idx + 1), device, dt, size)
            vision_feats, vision_pos, feat_sizes, high_res_prop, trunk_out = self._compute_backbone_frame(
                backbone_fn, frame, frame_idx=frame_idx)

            det_masks = torch.empty(0, device=device)
            det_scores = None
            run_det = (detect_fn is not None
                       and frame_idx % max(detect_interval, 1) == 0
                       and not (mux_state is not None and mux_state.total_valid_entries >= max_objects))
            if run_det:
                det_out = detect_fn(trunk_out)
                scores = det_out[KEY_SCORES][0].sigmoid()
                keep = scores > new_det_thresh
                det_masks, det_scores = det_out[KEY_MASKS][0][keep], scores[keep]
                if det_masks.shape[0] > 1:
                    det_masks, det_scores = _nms_masks(det_masks, det_scores)

            if frame_idx == 0 and initial_masks is not None:
                current_out = self._condition_with_masks(
                    initial_masks.to(device=device, dtype=dt), frame_idx, vision_feats, vision_pos,
                    feat_sizes, high_res_prop, output_dict, N, mux_state, backbone_obj, frame, trunk_out)
                last_occluded = torch.full((mux_state.total_valid_entries,), -1, device=device, dtype=torch.long)
                obj_scores = [1.0] * mux_state.total_valid_entries
                if keep_alive is not None:
                    for i in range(mux_state.total_valid_entries):
                        keep_alive[i] = 8
            elif mux_state is None or mux_state.total_valid_entries == 0:
                if det_masks.shape[0] > 0:
                    det_scores = det_scores[:max_objects]
                    det_masks = det_masks[:max_objects]
                    mux_state = MultiplexState(det_masks.shape[0], NUM_MULTIPLEX, device, dt)
                    current_out = self._condition_with_masks(
                        det_masks, frame_idx, vision_feats, vision_pos, feat_sizes, high_res_prop,
                        output_dict, N, mux_state, backbone_obj, frame, trunk_out, threshold=0.0)
                    last_occluded = torch.full((mux_state.total_valid_entries,), -1, device=device, dtype=torch.long)
                    obj_scores = det_scores[:mux_state.total_valid_entries].tolist()
                    if keep_alive is not None:
                        for i in range(mux_state.total_valid_entries):
                            keep_alive[i] = 1
                else:
                    # Nothing detected yet anywhere in the clip so far.
                    all_masks.append(None)
                    continue
            else:
                N_obj = mux_state.total_valid_entries
                current_out = self.track_step(
                    frame_idx=frame_idx, is_init_cond_frame=False, current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=None,
                    output_dict=output_dict, num_frames=N, propagation_high_res=high_res_prop,
                    multiplex_state=mux_state, run_mem_encoder=False)
                current_out["pred_masks"] = fill_holes_in_mask_scores(current_out["pred_masks"], max_area=16)
                if last_occluded.shape[0] == N_obj and N_obj > 1:
                    self._suppress_recently_occluded(current_out["pred_masks"], last_occluded, frame_idx)
                self._deferred_memory_encode(current_out, N_obj, vision_feats, feat_sizes, mux_state)
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out
                lookback = max(NUM_MASKMEM, MAX_OBJ_PTRS_IN_ENCODER)
                for old_idx in list(output_dict["non_cond_frame_outputs"]):
                    if old_idx < frame_idx - lookback:
                        del output_dict["non_cond_frame_outputs"][old_idx]
                n_before = mux_state.total_valid_entries
                new_obj_scores = self._match_and_add_detections(
                    det_masks, det_scores, current_out, mux_state, vision_feats, feat_sizes, device,
                    max_objects, keep_alive if run_det else None)
                n_added = mux_state.total_valid_entries - n_before
                if n_added > 0:
                    last_occluded = torch.cat([last_occluded, torch.full((n_added,), -1, device=device, dtype=torch.long)])
                    obj_scores.extend(new_obj_scores)

            masks_out = current_out["pred_masks_high_res"][:, 0]
            if keep_alive is not None:
                for i in range(masks_out.shape[0]):
                    if keep_alive.get(i, 0) <= 0:
                        masks_out[i] = NO_OBJ_SCORE
            all_masks.append(pack_masks(masks_out).to("cpu") if mux_state is not None and mux_state.total_valid_entries > 0 else None)

        if not all_masks or all(m is None for m in all_masks):
            return {KEY_PACKED_MASKS: None, KEY_N_FRAMES: N, KEY_SCORES: []}

        max_obj = max(m.shape[0] for m in all_masks if m is not None)
        sample = next(m for m in all_masks if m is not None)
        empty_packed = torch.zeros(max_obj, *sample.shape[1:], dtype=torch.uint8, device=sample.device)
        for i, m in enumerate(all_masks):
            if m is None:
                all_masks[i] = empty_packed
            elif m.shape[0] < max_obj:
                pad = torch.zeros(max_obj - m.shape[0], *m.shape[1:], dtype=torch.uint8, device=m.device)
                all_masks[i] = torch.cat([m, pad], dim=0)
        return {KEY_PACKED_MASKS: torch.stack(all_masks, dim=0), KEY_N_FRAMES: N, KEY_SCORES: obj_scores}
