"""Shared crop/normalize preprocessing that ViTPose and HMR2 both need identically:
mask-derived bbox -> square `bbx_xys` (center_x, center_y, size) -> an affine-warped
256x256 crop -> ImageNet-normalized (256, 192) model input.

Ported from `comfyui-motioncapture/nodes/motion_utils/hmr_cam.py`'s `get_bbx_xys*`
functions and `nodes/vitpose/feat_extractor.py`'s `crop_and_resize`/`get_batch`,
restricted to the single frame-at-a-time, no-augmentation, no-downscale path this
project needs (the source's `do_augment` is training-only; `img_ds` downscaling
is a speed/quality tradeoff for very high-res source video, not needed at this
project's target resolutions).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

CROP_SIZE = 256  # square crop before narrowing to the model's (256, 192) input
MODEL_SIZE = (256, 192)  # (H, W) -- both ViTPose and HMR2's ViT input shape
WIDTH_CROP_MARGIN = 32  # each side trimmed off the square crop's width: 256 - 2*32 = 192
BBOX_ASPECT_RATIO = MODEL_SIZE[1] / MODEL_SIZE[0]  # width:height = 192:256, for fitting a bbox to this model's shape
BBOX_ENLARGE = 1.2  # matches the reference's actual per-frame call (expand_bbox scale=1.2, then no further enlarge)

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])


def bbox_xywh_to_xys(bbox_xywh: tuple[int, int, int, int]) -> np.ndarray:
    """[x, y, w, h] (from a mask's bounding rect) -> [center_x, center_y, size]:
    enlarged by BBOX_ENLARGE, then fit to this model's fixed aspect ratio (matches
    `expand_bbox` + `get_bbx_xys_from_xyxy` in the reference, base_enlarge=1.0 there
    since the enlargement already happened via `expand_bbox`'s own scale=1.2).
    """
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0

    if w > BBOX_ASPECT_RATIO * h:
        h = w / BBOX_ASPECT_RATIO
    elif w < BBOX_ASPECT_RATIO * h:
        w = h * BBOX_ASPECT_RATIO

    size = max(h, w) * BBOX_ENLARGE
    return np.array([cx, cy, size], dtype=np.float32)


def crop_and_normalize(frame_rgb: np.ndarray, bbx_xys: np.ndarray) -> torch.Tensor:
    """Affine-warp `frame_rgb` (H, W, 3) uint8 to a CROP_SIZE square centered on
    `bbx_xys`, narrow to MODEL_SIZE, and ImageNet-normalize. Returns (3, 256, 192) float.
    """
    cx, cy, size = bbx_xys
    half = size / 2.0
    src = np.array([
        [cx - half, cy - half],  # left-up corner
        [cx + half, cy - half],  # right-up corner
        [cx, cy],                # center
    ], dtype=np.float32)
    dst = np.array([
        [0, 0], [CROP_SIZE - 1, 0], [CROP_SIZE / 2 - 0.5, CROP_SIZE / 2 - 0.5],
    ], dtype=np.float32)
    affine = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(frame_rgb, affine, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR)

    crop = torch.from_numpy(crop).float() / 255.0  # (256, 256, 3)
    crop = (crop - IMAGE_MEAN) / IMAGE_STD
    crop = crop.permute(2, 0, 1)  # (3, 256, 256)
    return crop[:, :, WIDTH_CROP_MARGIN:CROP_SIZE - WIDTH_CROP_MARGIN]  # (3, 256, 192)
