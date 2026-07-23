"""HaMeR's MANO regression head: cross-attends a single query token to the ViT
feature tokens, then reads out MANO hand pose + shape + a weak-perspective
camera. Clean-room port of `hamer/models/heads/mano_head.py`, restricted to the
released config (`JOINT_REP='6d'`, `TRANSFORMER_INPUT='zero'`, `IEF_ITERS=1`).

The head's initial-mean buffers (`init_hand_pose`/`init_betas`/`init_cam`) are
baked into the checkpoint, so they're registered here as correctly-shaped zeros
and filled by the strict weight load -- the separate `mano_mean_params.npz` is
only a training-time construction detail, not needed at inference.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hamer_transformer import TransformerDecoder

NUM_HAND_JOINTS = 15  # MANO's finger joints; +1 for the global (wrist) orientation
JOINT_REP_DIM = 6  # 6D rotation representation
NPOSE = JOINT_REP_DIM * (NUM_HAND_JOINTS + 1)  # 96
NUM_BETAS = 10

DIM = 1024
DEPTH = 6
HEADS = 8
DIM_HEAD = 64
MLP_DIM = 1024
CONTEXT_DIM = 1280  # ViT-H feature channels


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) 6D rotation -> (..., 3, 3) rotation matrix, via Gram-Schmidt
    (Zhou et al., CVPR 2019). Ported exactly from HaMeR: the 6 values are read
    as two columns (reshape to (2,3) then transpose), which is a *different*
    layout than pytorch3d's row-major `rotation_6d_to_matrix` -- reusing that
    would silently transpose every predicted joint rotation.
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()  # (B, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.linalg.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)  # (B, 3, 3), columns b1|b2|b3


class MANOTransformerDecoderHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = TransformerDecoder(
            num_tokens=1, token_dim=1, dim=DIM, depth=DEPTH, heads=HEADS,
            mlp_dim=MLP_DIM, dim_head=DIM_HEAD, context_dim=CONTEXT_DIM,
        )
        self.decpose = nn.Linear(DIM, NPOSE)
        self.decshape = nn.Linear(DIM, NUM_BETAS)
        self.deccam = nn.Linear(DIM, 3)
        # Filled by the strict weight load (see module docstring).
        self.register_buffer("init_hand_pose", torch.zeros(1, NPOSE))
        self.register_buffer("init_betas", torch.zeros(1, NUM_BETAS))
        self.register_buffer("init_cam", torch.zeros(1, 3))

    def forward(self, vit_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """vit_features: (B, CONTEXT_DIM, H, W) channel-first ViT output.
        Returns global_orient (B,1,3,3), hand_pose (B,15,3,3), betas (B,10),
        pred_cam (B,3) -- all in the hand crop's camera frame."""
        batch_size = vit_features.shape[0]
        context = einops.rearrange(vit_features, "b c h w -> b (h w) c")

        token = torch.zeros(batch_size, 1, 1, device=context.device, dtype=context.dtype)
        token_out = self.transformer(token, context=context).squeeze(1)  # (B, DIM)

        # Single iterative-error-feedback step: predict a residual on the mean.
        pred_hand_pose = self.decpose(token_out) + self.init_hand_pose.expand(batch_size, -1)
        pred_betas = self.decshape(token_out) + self.init_betas.expand(batch_size, -1)
        pred_cam = self.deccam(token_out) + self.init_cam.expand(batch_size, -1)

        rotmats = rot6d_to_rotmat(pred_hand_pose).view(batch_size, NUM_HAND_JOINTS + 1, 3, 3)
        return {
            "global_orient": rotmats[:, [0]],
            "hand_pose": rotmats[:, 1:],
            "betas": pred_betas,
            "pred_cam": pred_cam,
        }
