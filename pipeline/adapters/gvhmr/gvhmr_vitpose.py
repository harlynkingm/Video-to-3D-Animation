"""ViTPose: 2D keypoint (COCO-17) estimation on a person crop.

Ported from `comfyui-motioncapture/nodes/vitpose/model.py` (backbone+head) and
`nodes/motion_utils/kp2d_utils.py`'s `keypoints_from_heatmaps` (heatmap decode),
restricted to the single configuration and code path this project actually uses:
the "ViTPose_huge_coco_256x192" config (the only one the reference itself defines),
and UDP-style decoding with `target_type="GaussianHeatmap"` (the only branch the
reference's own extractor calls) -- dropped the unbiased/megvii/CombinedTarget
decode variants and the config-name lookup table entirely.

Also dropped: flip-test (running the horizontally-flipped crop too and averaging
heatmaps) -- a real but optional test-time accuracy boost, not essential
correctness; skipped for a simpler first version, easy to add back if real output
quality ever calls for it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gvhmr_preprocess import bbox_xywh_to_xys, crop_and_normalize
from ...helpers.vit_huge_backbone import VitHugeBackbone

NUM_KEYPOINTS = 17  # COCO
DECONV_FILTERS = 256
HEATMAP_H, HEATMAP_W = 64, 48  # 16x12 patch grid, upsampled 4x by two stride-2 deconvs


class KeypointHead(nn.Module):
    """Two stride-2 deconvs (1280 -> 256 -> 256) then a 1x1 conv to 17 heatmaps."""

    def __init__(self):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1280, DECONV_FILTERS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(DECONV_FILTERS),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(DECONV_FILTERS, DECONV_FILTERS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(DECONV_FILTERS),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(DECONV_FILTERS, NUM_KEYPOINTS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.deconv_layers(x))


class GVHMRViTPoseModel(nn.Module):
    """Matches the checkpoint's `backbone.*`/`keypoint_head.*` exactly."""

    def __init__(self):
        super().__init__()
        self.backbone = VitHugeBackbone()
        self.keypoint_head = KeypointHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 256, 192) -- returns (B, 17, 64, 48) heatmaps."""
        return self.keypoint_head(self.backbone(x))


# --- UDP heatmap decode (DARK: Zhang et al. CVPR 2020), restricted to the
# GaussianHeatmap + use_udp=True path this project's model actually produces ---


def _get_max_preds(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, K, H, W) -> preds (N, K, 2) argmax location, maxvals (N, K, 1) confidence."""
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = np.argmax(flat, axis=2).reshape(N, K, 1)
    maxvals = np.amax(flat, axis=2).reshape(N, K, 1)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    return np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1), maxvals


def _post_dark_udp(coords: np.ndarray, heatmaps: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Second-order Taylor refinement of the argmax location using the local
    heatmap curvature (the actual accuracy-improving step DARK/UDP contributes
    over plain argmax).
    """
    import cv2

    B, K, H, W = heatmaps.shape
    N = coords.shape[0]
    heatmaps = heatmaps.copy()
    for frame in heatmaps:
        for hm in frame:
            cv2.GaussianBlur(hm, (kernel, kernel), 0, hm)
    np.clip(heatmaps, 0.001, 50, heatmaps)
    np.log(heatmaps, heatmaps)

    padded = np.pad(heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").flatten()
    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)

    i_ = padded[index]
    ix1 = padded[index + 1]
    iy1 = padded[index + W + 2]
    ix1y1 = padded[index + W + 3]
    ix1_y1_ = padded[index - W - 3]
    ix1_ = padded[index - 1]
    iy1_ = padded[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1).reshape(N, K, 2, 1)
    dxx, dyy = ix1 - 2 * i_ + ix1_, iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze(-1)
    return coords


def _transform_preds(coords: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Heatmap-grid coords -> original-image pixel coords, given the crop's
    center and (width, height) footprint (`scale`, in mmpose's 200px-unit convention).
    """
    scale = scale * 200.0
    scale_x = scale[0] / (HEATMAP_W - 1.0)
    scale_y = scale[1] / (HEATMAP_H - 1.0)
    out = np.ones_like(coords)
    out[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    out[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return out


def estimate_keypoints(
    model: GVHMRViTPoseModel, frame_rgb: np.ndarray, bbox_xywh: tuple[int, int, int, int],
    device: torch.device, dtype: torch.dtype,
) -> np.ndarray:
    """Full single-frame pipeline: mask bbox -> crop -> heatmaps -> UDP decode ->
    original-image keypoints. Returns (17, 3) [x, y, confidence].
    """
    bbx_xys = bbox_xywh_to_xys(bbox_xywh)
    crop = crop_and_normalize(frame_rgb, bbx_xys).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.inference_mode():
        heatmap = model(crop)

    heatmap_np = heatmap.float().cpu().numpy()
    preds, maxvals = _get_max_preds(heatmap_np)
    preds = _post_dark_udp(preds, heatmap_np)

    cx, cy, size = bbx_xys
    scale = np.array([size * (HEATMAP_W / HEATMAP_H), size], dtype=np.float32) / 200.0
    preds[0] = _transform_preds(preds[0], np.array([cx, cy], dtype=np.float32), scale)

    return np.concatenate([preds[0], maxvals[0]], axis=-1)  # (17, 3)
