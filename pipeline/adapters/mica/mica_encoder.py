"""MICA's identity encoder: an IResNet-100 face-recognition backbone (the
ArcFace architecture from the `insightface` project, MIT licensed) feeding a
5-layer MLP that regresses the full 300-dim FLAME shape space, unlike
DECA's own 100-dim shape output, MICA specializes in identity/shape alone and
uses FLAME's complete shape basis.

Clean-room port of `insightface`'s `iresnet.py` (`IBasicBlock`/`IResNet`,
restricted to the `[3, 13, 30, 3]` block configuration this checkpoint
actually uses, confirmed by loading the real checkpoint and diffing every
key/shape, including the absence of squeeze-excite keys, which rules out the
IR-SE variant) and `models/generator.py`'s `MappingNetwork` regressor
(confirmed shapes: `Linear(512,300)` then four `Linear(300,300)` layers, the
last one unnamed as `.output`, LeakyReLU(0.2) between every layer).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# insightface's own block counts per stage for this specific depth variant.
LAYER_BLOCKS = (3, 13, 30, 3)
LAYER_CHANNELS = (64, 128, 256, 512)
EMBEDDING_DIM = 512
FC_SCALE = 7 * 7  # 112x112 input, 4 stride-2 stages -> 7x7 final feature map

N_SHAPE = 300  # FLAME's full shape-identity space (vs. DECA's 100)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """Pre-activation residual block: BN -> Conv -> BN -> PReLU -> Conv -> BN,
    stride applied on the *second* conv (not the first)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-5)
        self.conv1 = conv3x3(in_planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


def _make_layer(in_planes: int, planes: int, num_blocks: int, stride: int) -> tuple[nn.Sequential, int]:
    downsample = None
    if stride != 1 or in_planes != planes:
        downsample = nn.Sequential(conv1x1(in_planes, planes, stride), nn.BatchNorm2d(planes, eps=1e-5))
    blocks = [IBasicBlock(in_planes, planes, stride, downsample)]
    for _ in range(1, num_blocks):
        blocks.append(IBasicBlock(planes, planes))
    return nn.Sequential(*blocks), planes


class IResNet100(nn.Module):
    """ArcFace backbone: a 112x112 RGB crop in, a 512-dim identity embedding out."""

    def __init__(self) -> None:
        super().__init__()
        in_planes = LAYER_CHANNELS[0]
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-5)
        self.prelu = nn.PReLU(in_planes)

        layers = []
        for i, (channels, num_blocks) in enumerate(zip(LAYER_CHANNELS, LAYER_BLOCKS)):
            layer, in_planes = _make_layer(in_planes, channels, num_blocks, stride=2)
            layers.append(layer)
        self.layer1, self.layer2, self.layer3, self.layer4 = layers

        self.bn2 = nn.BatchNorm2d(LAYER_CHANNELS[-1], eps=1e-5)
        self.fc = nn.Linear(LAYER_CHANNELS[-1] * FC_SCALE, EMBEDDING_DIM)
        self.features = nn.BatchNorm1d(EMBEDDING_DIM, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 112, 112). Returns a (B, 512) identity embedding."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.features(x)


class MicaRegressor(nn.Module):
    """512-dim ArcFace embedding -> 300-dim FLAME shape coefficients."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.ModuleList([
            nn.Linear(EMBEDDING_DIM, N_SHAPE),
            nn.Linear(N_SHAPE, N_SHAPE),
            nn.Linear(N_SHAPE, N_SHAPE),
            nn.Linear(N_SHAPE, N_SHAPE),
        ])
        self.output = nn.Linear(N_SHAPE, N_SHAPE)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = embedding
        for layer in self.network:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
        return self.output(x)
