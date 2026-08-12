"""DECA's coarse-branch encoder: an image feature extractor feeding a small
MLP that regresses FLAME shape/expression/pose plus a weak-perspective camera.
Clean-room port of `decalib/models/encoders.py`'s `ResnetEncoder` and
`decalib/deca.py`'s `decompose_code`.

The feature extractor (`encoder.*` in the checkpoint) is not hand-ported: its
state dict is byte-for-byte a vanilla `torchvision.models.resnet50` with the
1000-class classifier dropped (verified by loading the real checkpoint and
diffing every key and shape against `torchvision.models.resnet50()`, 318/318
match exactly), so this reuses that directly rather than reimplementing
bottleneck blocks by hand.

`tex`/`light` (DECA's own texture-PCA and spherical-harmonics lighting
parameters, used for its photometric-rendering branch) are sliced out of the
raw encoder output but dropped from the returned dict, nothing downstream
renders with them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

FEATURE_DIM = 2048
HIDDEN_DIM = 1024

N_SHAPE = 100
N_TEX = 50
N_EXP = 50
N_POSE = 6  # [:3] global (head) rotation, [3:6] jaw rotation, see DECA's FLAME.forward
N_CAM = 3  # weak-perspective [scale, tx, ty]
N_LIGHT = 27  # 9 spherical-harmonics coefficients x 3 RGB channels, unused here
OUTPUT_DIM = N_SHAPE + N_TEX + N_EXP + N_POSE + N_CAM + N_LIGHT  # 236, fixed by the checkpoint


class DecaEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torchvision.models.resnet50(weights=None)
        self.encoder.fc = nn.Identity()  # (B, 2048) global-average-pooled features
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """image: (B, 3, 224, 224), see `deca_preprocess.crop_face` for the
        exact normalization DECA expects. Returns shape (B,100), exp (B,50),
        pose (B,6), cam (B,3)."""
        code = self.layers(self.encoder(image))
        return {
            "shape": code[:, :N_SHAPE],
            "exp": code[:, N_SHAPE + N_TEX:N_SHAPE + N_TEX + N_EXP],
            "pose": code[:, N_SHAPE + N_TEX + N_EXP:N_SHAPE + N_TEX + N_EXP + N_POSE],
            "cam": code[:, N_SHAPE + N_TEX + N_EXP + N_POSE:N_SHAPE + N_TEX + N_EXP + N_POSE + N_CAM],
        }
