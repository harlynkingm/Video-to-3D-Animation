"""SAM 3.1's text-conditioned detector: a DETR-style transformer encoder/decoder over
the vision backbone's top FPN level, a small geometry encoder (only its unconditional
cls-token path is used here -- see below), a segmentation head that upsamples decoder
queries back through the FPN to per-query masks, and a dot-product query/prompt scorer.

Ported from `comfy/ldm/sam3/detector.py`, replacing `comfy.ops`/`optimized_attention`
with plain PyTorch (this project has no ComfyUI runtime dependency, to avoid the
GPL-3.0/Apache-2.0 conflict that would come with vendoring ComfyUI's own code).
Only `forward_from_trunk` is ported (given an already-computed ViTDet trunk
output plus one already-encoded text prompt, returns boxes/scores/masks) -- the
source's other entry points (`forward`, `forward_segment`) exist for single-image and
interactive point/box-prompted use, which this project's pure text-prompt video
tracking never calls.

Mask convention throughout: boolean, True = attend (matches `sam31_clip_text.py` and
the rest of this port) -- NOT `nn.MultiheadAttention`'s inverted `key_padding_mask`
convention, which is why attention here is built from a fused `in_proj`/`out_proj`
(matching the checkpoint's own parameter layout exactly, so no weight remapping is
needed) plus a manual `F.scaled_dot_product_attention` call, rather than
`nn.MultiheadAttention` itself.

`GeometryEncoder`'s point/box-prompt layers are defined (for a clean `strict=True`
checkpoint load) but their forward-pass logic is not ported: this project never gives
point or box prompts, only text, and geometry_encoder's cls-token path runs
unconditionally whenever *any* prompt is given (confirmed by reading `_detect`) --
so cls_embed + its small encoder + norms are load-bearing even for text-only prompts,
while the point/box-specific projections are simply never exercised.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 256
NUM_HEADS = 8
DIM_FF = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_QUERIES = 200
GEO_ENCODE_LAYERS = 3
ROI_SIZE = 7  # unused (box-prompt only), kept for strict-load parameter shape matching


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def gen_sineembed_for_position(pos_tensor: torch.Tensor, num_feats: int = D_MODEL) -> torch.Tensor:
    """Per-coordinate sinusoidal embedding: (..., N) -> (..., N * num_feats)."""
    assert num_feats % 2 == 0
    hdim = num_feats // 2
    freqs = 10000.0 ** (2 * (torch.arange(hdim, dtype=torch.float32, device=pos_tensor.device) // 2) / hdim)
    embeds = []
    for c in range(pos_tensor.shape[-1]):
        raw = (pos_tensor[..., c].float() * 2 * math.pi).unsqueeze(-1) / freqs
        embeds.append(torch.stack([raw[..., 0::2].sin(), raw[..., 1::2].cos()], dim=-1).flatten(-2))
    return torch.cat(embeds, dim=-1).to(pos_tensor.dtype)


class SimpleMLP(nn.Module):
    """Plain N-layer MLP with ReLU between layers (no residual/norm) -- used for the
    various small prediction heads (reference points, boxes, presence, box-RPB, masks).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class MLPWithNorm(nn.Module):
    """MLP with a residual connection (when in/out dims match) and an output LayerNorm."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.out_norm = nn.LayerNorm(output_dim)
        self.residual = input_dim == output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        if self.residual:
            x = x + orig
        return self.out_norm(x)


class CrossAttention(nn.Module):
    """Multi-head attention with a single fused in_proj (matching the checkpoint's own
    parameter layout exactly -- same as nn.MultiheadAttention's native parameter names,
    zero remapping needed), supporting separate query/key/value inputs for
    cross-attention. Mask convention: boolean, True = attend (see module docstring).
    """

    def __init__(self, d_model: int = D_MODEL, num_heads: int = NUM_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q_input: torch.Tensor, k_input: torch.Tensor | None = None,
                v_input: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        d = q_input.shape[-1]
        w_q, w_k, w_v = self.in_proj_weight.split(d)
        b_q, b_k, b_v = self.in_proj_bias.split(d)

        q = F.linear(q_input, w_q, b_q)
        if k_input is None:
            k = F.linear(q_input, w_k, b_k)
            v = F.linear(q_input, w_v, b_v)
        else:
            k = F.linear(k_input, w_k, b_k)
            v = F.linear(v_input if v_input is not None else k_input, w_v, b_v)

        B, Lq, _ = q.shape
        Lk = k.shape[1]
        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :] if mask.ndim == 2 else mask

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Lq, -1)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = CrossAttention()
        self.cross_attn_image = CrossAttention()
        self.linear1 = nn.Linear(D_MODEL, DIM_FF)
        self.linear2 = nn.Linear(DIM_FF, D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, text_memory: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm1(x)
        q_k = normed + pos
        x = x + self.self_attn(q_k, q_k, normed)
        if text_memory is not None:
            normed = self.norm2(x)
            x = x + self.cross_attn_image(normed, text_memory, text_memory, mask=text_mask)
        normed = self.norm3(x)
        x = x + self.linear2(F.relu(self.linear1(normed)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(NUM_ENCODER_LAYERS)])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, text_memory: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos, text_memory, text_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = CrossAttention()
        self.cross_attn = CrossAttention()
        self.ca_text = CrossAttention()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)
        self.catext_norm = nn.LayerNorm(D_MODEL)
        self.linear1 = nn.Linear(D_MODEL, DIM_FF)
        self.linear2 = nn.Linear(DIM_FF, D_MODEL)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, x_pos: torch.Tensor, memory_pos: torch.Tensor,
                text_memory: torch.Tensor | None = None, text_mask: torch.Tensor | None = None,
                cross_attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        q_k = x + x_pos
        x = self.norm2(x + self.self_attn(q_k, q_k, x))
        if text_memory is not None:
            x = self.catext_norm(x + self.ca_text(x + x_pos, text_memory, text_memory, mask=text_mask))
        x = self.norm1(x + self.cross_attn(x + x_pos, memory + memory_pos, memory, mask=cross_attn_bias))
        x = self.norm3(x + self.linear2(F.relu(self.linear1(x))))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(NUM_DECODER_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.query_embed = nn.Embedding(NUM_QUERIES, D_MODEL)
        self.reference_points = nn.Embedding(NUM_QUERIES, 4)  # learned anchor boxes, cxcywh
        self.ref_point_head = SimpleMLP(D_MODEL * 2, D_MODEL, D_MODEL, 2)
        self.bbox_embed = SimpleMLP(D_MODEL, D_MODEL, 4, 3)

        self.boxRPB_embed_x = SimpleMLP(2, D_MODEL, NUM_HEADS, 2)
        self.boxRPB_embed_y = SimpleMLP(2, D_MODEL, NUM_HEADS, 2)

        self.presence_token = nn.Embedding(1, D_MODEL)
        self.presence_token_head = SimpleMLP(D_MODEL, D_MODEL, 1, 3)
        self.presence_token_out_norm = nn.LayerNorm(D_MODEL)

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1 - x + 1e-6) + 1e-6)

    def _compute_box_rpb(self, ref_points: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Box rotary position bias: (B, Q, 4) cxcywh -> (B, n_heads, Q+1, H*W) bias."""
        boxes_xyxy = box_cxcywh_to_xyxy(ref_points)
        B, _, _ = boxes_xyxy.shape
        coords_h = torch.arange(H, device=ref_points.device, dtype=torch.float32) / H
        coords_w = torch.arange(W, device=ref_points.device, dtype=torch.float32) / W
        deltas_x = coords_w.view(1, 1, -1, 1) - boxes_xyxy[:, :, None, 0:3:2]
        deltas_y = coords_h.view(1, 1, -1, 1) - boxes_xyxy[:, :, None, 1:4:2]

        log2_8 = float(math.log2(8))

        def log_scale(d):
            return torch.sign(d * 8) * torch.log2(torch.abs(d * 8) + 1.0) / log2_8

        rpb_x = self.boxRPB_embed_x(log_scale(deltas_x).to(ref_points.dtype))
        rpb_y = self.boxRPB_embed_y(log_scale(deltas_y).to(ref_points.dtype))

        bias = (rpb_y.unsqueeze(3) + rpb_x.unsqueeze(2)).flatten(2, 3).permute(0, 3, 1, 2)
        pres_bias = torch.zeros(B, bias.shape[1], 1, bias.shape[3], device=bias.device, dtype=bias.dtype)
        return torch.cat([pres_bias, bias], dim=2)

    def forward(self, memory: torch.Tensor, memory_pos: torch.Tensor, text_memory: torch.Tensor | None,
                text_mask: torch.Tensor | None, H: int, W: int) -> dict:
        B = memory.shape[0]
        tgt = self.query_embed.weight.to(memory.dtype).unsqueeze(0).expand(B, -1, -1)
        presence_out = self.presence_token.weight.to(memory.dtype)[None].expand(B, -1, -1)
        ref_points = self.reference_points.weight.to(memory.dtype).unsqueeze(0).expand(B, -1, -1).sigmoid()

        for layer_idx, layer in enumerate(self.layers):
            query_pos = self.ref_point_head(gen_sineembed_for_position(ref_points, D_MODEL))
            tgt_with_pres = torch.cat([presence_out, tgt], dim=1)
            pos_with_pres = torch.cat([torch.zeros_like(presence_out), query_pos], dim=1)
            tgt_with_pres = layer(tgt_with_pres, memory, pos_with_pres, memory_pos,
                                   text_memory, text_mask, self._compute_box_rpb(ref_points, H, W))
            presence_out, tgt = tgt_with_pres[:, :1], tgt_with_pres[:, 1:]
            if layer_idx < len(self.layers) - 1:
                ref_inv = self._inverse_sigmoid(ref_points)
                ref_points = (ref_inv + self.bbox_embed(self.norm(tgt))).sigmoid().detach()

        query_out = self.norm(tgt)
        ref_inv = self._inverse_sigmoid(ref_points)
        boxes = (ref_inv + self.bbox_embed(query_out)).sigmoid()
        presence = self.presence_token_head(self.presence_token_out_norm(presence_out)).squeeze(-1)
        return {"decoder_output": query_out, "pred_boxes": boxes, "presence": presence}


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()


class PositionEmbeddingSine2(nn.Module):
    """Duplicated from `sam31_vitdet_backbone.py` rather than imported: same math, but
    `geometry_encoder`'s copy is a separate set of (non-learned, so purely functional --
    no weights to load) instance, matching the checkpoint's own module tree shape.
    """

    def __init__(self, num_pos_feats: int = D_MODEL):
        super().__init__()
        self.half_dim = num_pos_feats // 2
        self.temperature = 10000.0
        self.scale = 2 * math.pi

    def _encode_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dim_t = self.temperature ** (2 * (torch.arange(self.half_dim, dtype=torch.float32, device=x.device) // 2) / self.half_dim)
        pos_x = x[:, None] * self.scale / dim_t
        pos_y = y[:, None] * self.scale / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y


class GeometryEncoder(nn.Module):
    """Only the unconditional cls-token path is actually run by this project (see module
    docstring) -- the point/box projection layers below are defined for a clean
    `strict=True` load but their forward-pass logic (`_encode_points`/`_encode_boxes` in
    the source) is deliberately not ported.
    """

    def __init__(self):
        super().__init__()
        self.pos_enc = PositionEmbeddingSine2(D_MODEL)
        self.points_direct_project = nn.Linear(2, D_MODEL)
        self.points_pool_project = nn.Linear(D_MODEL, D_MODEL)
        self.points_pos_enc_project = nn.Linear(D_MODEL, D_MODEL)
        self.boxes_direct_project = nn.Linear(4, D_MODEL)
        self.boxes_pool_project = nn.Conv2d(D_MODEL, D_MODEL, kernel_size=ROI_SIZE)
        self.boxes_pos_enc_project = nn.Linear(D_MODEL + 2, D_MODEL)
        self.label_embed = nn.Embedding(2, D_MODEL)
        self.cls_embed = nn.Embedding(1, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)
        self.img_pre_norm = nn.LayerNorm(D_MODEL)
        self.encode = nn.ModuleList([EncoderLayer() for _ in range(GEO_ENCODE_LAYERS)])
        self.encode_norm = nn.LayerNorm(D_MODEL)
        self.final_proj = nn.Linear(D_MODEL, D_MODEL)

    def compute_geo_cls(self, image_features: torch.Tensor, image_pos: torch.Tensor) -> torch.Tensor:
        """The cls-token path that runs whenever any prompt (including text-only) is given."""
        B = image_features.shape[0]
        geo_cls = self.norm(self.final_proj(self.cls_embed.weight.to(image_features.dtype).view(1, 1, -1).expand(B, -1, -1)))
        for layer in self.encode:
            geo_cls = geo_cls + layer.self_attn(layer.norm1(geo_cls))
            geo_cls = geo_cls + layer.cross_attn_image(layer.norm2(geo_cls), image_features + image_pos, image_features)
            geo_cls = geo_cls + layer.linear2(F.relu(layer.linear1(layer.norm3(geo_cls))))
        return self.encode_norm(geo_cls)


class PixelDecoder(nn.Module):
    """Top-down FPN pixel decoder with GroupNorm + ReLU + nearest interpolation."""

    def __init__(self, num_stages: int = 3):
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(D_MODEL, D_MODEL, kernel_size=3, padding=1) for _ in range(num_stages)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, D_MODEL) for _ in range(num_stages)])

    def forward(self, backbone_features: list[torch.Tensor]) -> torch.Tensor:
        prev = backbone_features[-1]
        for i, feat in enumerate(backbone_features[:-1][::-1]):
            prev = F.relu(self.norms[i](self.conv_layers[i](feat + F.interpolate(prev, size=feat.shape[-2:], mode="nearest"))))
        return prev


class MaskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_embed = SimpleMLP(D_MODEL, D_MODEL, D_MODEL, 3)

    def forward(self, query_embeddings: torch.Tensor, pixel_features: torch.Tensor) -> torch.Tensor:
        mask_embed = self.mask_embed(query_embeddings)
        return torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_features)


class SegmentationHead(nn.Module):
    """`semantic_seg_head` is defined (checkpoint has it) but never invoked, matching the
    source exactly -- it's an auxiliary training-time head, unused at inference.
    """

    def __init__(self):
        super().__init__()
        self.pixel_decoder = PixelDecoder(3)
        self.mask_predictor = MaskPredictor()
        self.cross_attend_prompt = CrossAttention()
        self.cross_attn_norm = nn.LayerNorm(D_MODEL)
        self.instance_seg_head = nn.Conv2d(D_MODEL, D_MODEL, kernel_size=1)
        self.semantic_seg_head = nn.Conv2d(D_MODEL, 1, kernel_size=1)

    def forward(self, query_embeddings: torch.Tensor, backbone_features: list[torch.Tensor],
                encoder_hidden_states: torch.Tensor, prompt: torch.Tensor | None = None,
                prompt_mask: torch.Tensor | None = None) -> torch.Tensor:
        if prompt is not None:
            enc_normed = self.cross_attn_norm(encoder_hidden_states)
            enc_cross = self.cross_attend_prompt(enc_normed, prompt, prompt, mask=prompt_mask)
            encoder_hidden_states = enc_cross + encoder_hidden_states

        B = encoder_hidden_states.shape[0]
        H, W = backbone_features[-1].shape[-2:]
        encoder_visual = encoder_hidden_states[:, :H * W].permute(0, 2, 1).view(B, D_MODEL, H, W)
        backbone_features = list(backbone_features)
        backbone_features[-1] = encoder_visual

        pixel_features = self.pixel_decoder(backbone_features)
        instance_features = self.instance_seg_head(pixel_features)
        return self.mask_predictor(query_embeddings, instance_features)


class DotProductScoring(nn.Module):
    def __init__(self):
        super().__init__()
        self.hs_proj = nn.Linear(D_MODEL, D_MODEL)
        self.prompt_proj = nn.Linear(D_MODEL, D_MODEL)
        self.prompt_mlp = MLPWithNorm(D_MODEL, DIM_FF, D_MODEL, 2)
        self.scale = 1.0 / (D_MODEL ** 0.5)

    def forward(self, query_embeddings: torch.Tensor, prompt_embeddings: torch.Tensor,
                prompt_mask: torch.Tensor | None = None) -> torch.Tensor:
        prompt = self.prompt_mlp(prompt_embeddings)
        if prompt_mask is not None:
            weight = prompt_mask.unsqueeze(-1).to(dtype=prompt.dtype)
            pooled = (prompt * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)
        else:
            pooled = prompt.mean(dim=1)
        hs = self.hs_proj(query_embeddings)
        pp = self.prompt_proj(pooled).unsqueeze(-1).to(hs.dtype)
        scores = torch.matmul(hs, pp)
        return (scores * self.scale).clamp(-12.0, 12.0).squeeze(-1)


class Sam31Detector(nn.Module):
    """Ties the pieces above together. Deliberately decoupled from
    `Sam31VisionBackbone`: `forward_from_trunk` takes already-computed FPN features and
    positions (the caller runs the vision backbone itself), not a raw trunk output plus
    an internal backbone reference -- this keeps the two files independently loadable
    and testable, matching how their checkpoint weights are genuinely separate tensor
    groups (only `text_resizer` bridges CLIP's 1024-dim space to this detector's 256-dim
    working space; matches `detector.backbone.language_backbone.resizer` exactly).
    """

    def __init__(self):
        super().__init__()
        self.text_resizer = nn.Linear(1024, D_MODEL)
        self.transformer = Transformer()
        self.segmentation_head = SegmentationHead()
        self.geometry_encoder = GeometryEncoder()
        self.dot_prod_scoring = DotProductScoring()

    def forward_from_trunk(self, features: list[torch.Tensor], positions: list[torch.Tensor],
                            text_embeddings: torch.Tensor, text_mask: torch.Tensor) -> dict:
        """features/positions: the 3 FPN levels (288/144/72) from Sam31VisionBackbone, already
        computed by the caller. text_embeddings: raw (1, L, 1024) from sam31_clip_text.py
        (NOT yet resized -- this method applies text_resizer itself). text_mask: (1, L)
        boolean, True = real token (same convention as sam31_clip_text.py).

        Returns {"boxes": normalized xyxy, "scores": (1, num_queries), "masks": (1, num_queries, 288, 288)}.
        """
        text_embeddings = self.text_resizer(text_embeddings)

        seg_features = features  # all 3 levels feed the segmentation head
        enc_feat, enc_pos = features[-1], positions[-1]  # 72x72 level feeds the transformer
        _, _, H, W = enc_feat.shape
        img_flat = enc_feat.flatten(2).permute(0, 2, 1)
        pos_flat = enc_pos.flatten(2).permute(0, 2, 1)

        geo_cls = self.geometry_encoder.compute_geo_cls(img_flat, pos_flat)
        combined_text = torch.cat([text_embeddings, geo_cls], dim=1)
        combined_mask = torch.cat([text_mask, torch.ones(text_mask.shape[0], 1, dtype=torch.bool, device=text_mask.device)], dim=1)

        memory = self.transformer.encoder(img_flat, pos_flat, combined_text, combined_mask)
        dec_out = self.transformer.decoder(memory, pos_flat, combined_text, combined_mask, H, W)
        query_out, pred_boxes = dec_out["decoder_output"], dec_out["pred_boxes"]

        scores = self.dot_prod_scoring(query_out, combined_text, combined_mask)
        masks = self.segmentation_head(query_out, seg_features, encoder_hidden_states=memory,
                                        prompt=combined_text, prompt_mask=combined_mask)

        return {"boxes": box_cxcywh_to_xyxy(pred_boxes), "scores": scores, "masks": masks}
