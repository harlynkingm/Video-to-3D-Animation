"""Text encoding for SAM 3.1's bundled CLIP-style text tower.

The checkpoint bundles its own text transformer under the
`detector.backbone.language_backbone.encoder.*` keys -- structured like
OpenAI's original CLIP code (a raw `positional_embedding` parameter, combined
QKV `in_proj_weight`/`in_proj_bias`, `c_fc`/`c_proj` MLP naming), not like
HuggingFace's restructured `CLIPTextModel`. Building this with `nn.MultiheadAttention`
directly means the state dict keys match the checkpoint exactly -- no remapping.

Tokenization uses the standard public CLIP BPE vocabulary (the same one every
CLIP variant reuses) -- only the transformer weights are SAM3-specific.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPTokenizer

# The CLIP BPE tokenizer is vendored into the repo (next to this file) rather
# than fetched from the HF Hub, so the pipeline runs fully offline and
# reproducibly with no runtime network dependency. This is the standard public
# CLIP vocabulary (saved from openai/clip-vit-large-patch14); only the text
# transformer's weights are SAM3-specific, and those load from the checkpoint.
CLIP_TOKENIZER_DIR = Path(__file__).resolve().parent / "clip_tokenizer"
MAX_PROMPT_TOKENS = 32  # hard limit -- baked into the checkpoint's positional_embedding size
HIDDEN_SIZE = 1024
NUM_LAYERS = 24
NUM_HEADS = 16
VOCAB_SIZE = 49408
PAD_TOKEN_ID = 0  # the checkpoint's own convention: literal pad id 0, not repeated EOS


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        ln = self.ln_1(x)
        attn_out, _ = self.attn(ln, ln, ln, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class Sam31TextTower(nn.Module):
    """Matches `detector.backbone.language_backbone.encoder.*` exactly (prefix stripped)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        max_len: int = MAX_PROMPT_TOKENS,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.empty(max_len, d_model))
        self.transformer = Transformer(d_model, num_layers, num_heads)
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/attention_mask: (1, L) -- batch size 1 only, matching how encode_prompt calls this
        (one prompt per SAM 3.1 forward_video text_prompts entry, never batched together).
        Returns last-hidden-state embeddings (1, L, d_model) -- not pooled.
        """
        assert input_ids.shape[0] == 1, "Sam31TextTower only supports batch size 1 (see docstring)"
        seq_len = input_ids.shape[1]
        x = self.token_embedding(input_ids) + self.positional_embedding[:seq_len]

        # OpenAI CLIP's text transformer is causal (trained that way; must replicate to get correct
        # outputs). Combined with padding into one additive float mask, rather than passing a separate
        # bool key_padding_mask, since PyTorch deprecated mixing the two mask types.
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
        pad_cols = attention_mask[0] == 0  # (L,) True where padded
        attn_mask = causal.masked_fill(pad_cols.unsqueeze(0), float("-inf"))

        x = self.transformer(x, attn_mask)
        return self.ln_final(x)


def load_tokenizer() -> CLIPTokenizer:
    # local_files_only=True guarantees no HF Hub call (and no "unauthenticated
    # request" warning) even when HF_HUB_OFFLINE isn't set in the environment.
    return CLIPTokenizer.from_pretrained(str(CLIP_TOKENIZER_DIR), local_files_only=True)


def encode_prompt(
    prompt: str,
    model: Sam31TextTower,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one prompt string to (embeddings, attention_mask), both batch size 1.

    embeddings: (1, L, HIDDEN_SIZE) float, attention_mask: (1, L) bool, True = real token,
    False = padding -- this is the "True = attend" convention used throughout this port
    (see sam31_detector.py's module docstring), so it can be passed directly as an SDPA
    attn_mask by downstream consumers without a dtype fixup.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    ids = ids[:MAX_PROMPT_TOKENS]
    real_len = len(ids)
    ids = ids + [PAD_TOKEN_ID] * (MAX_PROMPT_TOKENS - real_len)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.zeros((1, MAX_PROMPT_TOKENS), dtype=torch.bool, device=device)
    attention_mask[0, :real_len] = True

    with torch.inference_mode():
        embeddings = model(input_ids, attention_mask).to(dtype)

    return embeddings, attention_mask
