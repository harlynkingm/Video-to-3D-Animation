"""GVHMR's own contribution: a temporal RoPE transformer that fuses per-frame
2D keypoints (`obs`), per-frame image features (`f_imgseq`, from HMR2), and
camera conditioning (`f_cliffcam`/`f_cam_angvel`) into a single 151-dim pose
vector per frame (decoded by `gvhmr_endecoder.py`). Ported from
`comfyui-motioncapture/nodes/gvhmr/model.py`'s `NetworkEncoderRoPE` and its
rotary-embedding/attention building blocks, confirmed against the real
`gvhmr.safetensors` (`pipeline.denoiser3d.*`, 12 blocks, latent_dim=512).

This 1D temporal RoPE is unrelated to `sam31_vitdet_backbone.py`'s 2D axial
RoPE (different math, different checkpoint, coincidentally similar name) --
not reused, ported separately here to match this checkpoint's own weights.

`pred_cam_mean`/`pred_cam_std` are hardcoded constants below, not loaded from
the checkpoint: the source registers them `persistent=False`, confirmed absent
from the real checkpoint's key list.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 512
NUM_LAYERS = 12
NUM_HEADS = 8
MLP_RATIO = 4.0
OUTPUT_DIM = 151
MAX_LEN = 120  # sequences longer than this use a local attention window, not global attention
CLIFFCAM_DIM = 3
CAM_ANGVEL_DIM = 6
IMGSEQ_DIM = 1024
PRED_CAM_DIM = 3
STATIC_CONF_DIM = 6
NUM_KP2D_JOINTS = 17

PRED_CAM_MEAN = torch.tensor([1.0606, -0.0027, 0.2702])
PRED_CAM_STD = torch.tensor([0.1784, 0.0956, 0.0764])


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """t: (B, H, L, D). freqs: (max_len, D) -- sliced to the last L positions."""
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:].to(t)
    return t * freqs.cos() + _rotate_half(t) * freqs.sin()


def _rope_table(d_model: int, max_seq_len: int) -> torch.Tensor:
    t = torch.arange(max_seq_len).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    freqs = torch.einsum("i,j->ij", t, freqs)
    return freqs.repeat_interleave(2, dim=-1)  # (max_seq_len, d_model)


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.register_buffer("rope_table", _rope_table(self.head_dim, MAX_LEN), persistent=False)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, C). attn_mask: (L, L) bool, True=masked. key_padding_mask: (B, L) bool, True=padded."""
        B, L, _ = x.shape
        rope = self.rope_table if L <= MAX_LEN else _rope_table(self.head_dim, L).to(x.device)

        q = self.query(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        k = self.key(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        v = self.value(x).reshape(B, L, self.num_heads, -1).transpose(1, 2)
        q = _apply_rotary_emb(rope, q)
        k = _apply_rotary_emb(rope, k)

        mask = None
        if attn_mask is not None or key_padding_mask is not None:
            mask = torch.zeros(B, 1, L, L, device=x.device, dtype=q.dtype)
            if attn_mask is not None:
                mask = mask.masked_fill(attn_mask.view(1, 1, L, L), float("-inf"))
            if key_padding_mask is not None:
                mask = mask.masked_fill(key_padding_mask.view(B, 1, 1, L), float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class EncoderRoPEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.gate_msa * self.attn(self.norm1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.gate_mlp * self.mlp(self.norm2(x))
        return x


def _length_to_mask(length: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) frame counts -> (B, max_len) bool, True where the position is a real
    (non-padded) frame."""
    return torch.arange(max_len, device=length.device)[None, :] < length[:, None]


class GVHMRTemporalTransformer(nn.Module):
    """Matches `pipeline.denoiser3d.*` exactly."""

    def __init__(self):
        super().__init__()
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(NUM_KP2D_JOINTS, 32))
        self.embed_noisyobs = Mlp(NUM_KP2D_JOINTS * 32, hidden_features=LATENT_DIM * 2, out_features=LATENT_DIM)

        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(CLIFFCAM_DIM, LATENT_DIM), nn.SiLU(), nn.Dropout(0.0), nn.Linear(LATENT_DIM, LATENT_DIM))
        self.cam_angvel_embedder = nn.Sequential(
            nn.Linear(CAM_ANGVEL_DIM, LATENT_DIM), nn.SiLU(), nn.Dropout(0.0), nn.Linear(LATENT_DIM, LATENT_DIM))
        self.imgseq_embedder = nn.Sequential(nn.LayerNorm(IMGSEQ_DIM), nn.Linear(IMGSEQ_DIM, LATENT_DIM))

        self.blocks = nn.ModuleList([EncoderRoPEBlock(LATENT_DIM, NUM_HEADS, MLP_RATIO) for _ in range(NUM_LAYERS)])

        self.final_layer = Mlp(LATENT_DIM, out_features=OUTPUT_DIM)
        self.pred_cam_head = Mlp(LATENT_DIM, out_features=PRED_CAM_DIM)
        self.static_conf_head = Mlp(LATENT_DIM, out_features=STATIC_CONF_DIM)
        self.register_buffer("pred_cam_mean", PRED_CAM_MEAN.clone(), persistent=False)
        self.register_buffer("pred_cam_std", PRED_CAM_STD.clone(), persistent=False)

    def forward(self, length: torch.Tensor, obs: torch.Tensor, f_cliffcam: torch.Tensor,
                f_cam_angvel: torch.Tensor, f_imgseq: torch.Tensor) -> dict:
        """
        length: (B,) real frame count per batch item (this project always passes B=1, length=[N_frames]).
        obs: (B, L, 17, 3) [x, y, confidence] 2D keypoints, already bbox-normalized.
        f_cliffcam: (B, L, 3). f_cam_angvel: (B, L, 6). f_imgseq: (B, L, 1024).
        """
        B, L, J, C = obs.shape
        assert J == NUM_KP2D_JOINTS and C == 3

        obs = obs.clone()
        visible = obs[..., [2]] > 0.5
        obs[~visible[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs[..., :2])
        f_obs = f_obs * visible + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible
        x = self.embed_noisyobs(f_obs.reshape(B, L, -1))

        x = x + self.cliffcam_embedder(f_cliffcam)
        x = x + self.cam_angvel_embedder(f_cam_angvel)
        x = x + self.imgseq_embedder(f_imgseq)

        pad_mask = ~_length_to_mask(length, L)  # True where padded
        attn_mask = None
        if L > MAX_LEN:
            # Local attention window: each position only attends within +/- MAX_LEN//2
            # of itself (clamped to stay inside [0, L)) -- avoids O(L^2) global attention
            # on long clips while still giving every frame a wide local temporal context.
            attn_mask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - MAX_LEN // 2)
                max_ind = min(L, i + MAX_LEN // 2)
                max_ind = max(MAX_LEN, max_ind)
                min_ind = min(L - MAX_LEN, min_ind)
                attn_mask[i, min_ind:max_ind] = False

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=pad_mask)

        sample = self.final_layer(x)
        # Average the predicted betas (shape params) across all real frames --
        # a person's body shape doesn't change frame to frame, so pooling gives a
        # single more-stable estimate instead of 151-dim independent per-frame noise.
        betas = (sample[..., 126:136] * (~pad_mask[..., None])).sum(1) / length[:, None]
        betas = betas.unsqueeze(1).expand(-1, L, -1)
        sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)

        pred_cam = self.pred_cam_head(x)
        pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
        pred_cam[..., 0].clamp_min_(0.25)

        static_conf_logits = self.static_conf_head(x)

        return {"pred_x": sample, "pred_cam": pred_cam, "static_conf_logits": static_conf_logits}
