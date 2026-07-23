"""Hand-crop preprocessing for HaMeR: turn a frame + a hand bounding box into
the exact normalized 256x256 patch the model expects.

Ported from `hamer/datasets/vitdet_dataset.py` + `datasets/utils.py`, restricted
to inference (no augmentation). Two project-specific choices, both flagged:

  - HaMeR's demo derives hand boxes from a whole-body (133-keypoint) ViTPose.
    This project only has a COCO-17 body ViTPose, so `hand_box_from_body_kpts`
    derives the box from the wrist+elbow instead -- a heuristic, isolated here
    so a whole-body-ViTPose upgrade can replace just this one function later.
  - `BBOX_SHAPE = (192, 256)` is HaMeR's released `model_config.yaml` value
    (that file ships only inside the checkpoint tarball, not the public repo);
    verified indirectly by checking that real crops yield plausible hand poses.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = 256
IMAGE_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
IMAGE_STD = 255.0 * np.array([0.229, 0.224, 0.225])
BBOX_SHAPE = (192, 256)  # (w, h) aspect the box is expanded to before cropping
RESCALE_FACTOR = 2.5  # padding around the detected hand box

# COCO-17 keypoint indices (what gvhmr_vitpose outputs).
COCO_L_ELBOW, COCO_R_ELBOW = 7, 8
COCO_L_WRIST, COCO_R_WRIST = 9, 10
_MIN_KPT_CONF = 0.3


def expand_to_aspect_ratio(input_shape: np.ndarray, target_aspect_ratio: tuple[int, int]) -> np.ndarray:
    """Grow a (w, h) box to match a target aspect ratio (never shrinks)."""
    w, h = input_shape
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new, w_new = max(w * h_t / w_t, h), w
    else:
        h_new, w_new = h, max(h * w_t / h_t, w)
    return np.array([w_new, h_new])


def _gen_trans(c_x: float, c_y: float, src_size: float, dst_size: int) -> np.ndarray:
    """Affine transform mapping a `src_size` box centered at (c_x, c_y) to a
    `dst_size` x `dst_size` patch (no rotation/scale augmentation)."""
    src = np.array([[c_x, c_y], [c_x, c_y + src_size * 0.5], [c_x + src_size * 0.5, c_y]], dtype=np.float32)
    half = dst_size * 0.5
    dst = np.array([[half, half], [half, half + half], [half + half, half]], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def hand_box_from_body_kpts(keypoints: np.ndarray, is_right: bool) -> np.ndarray | None:
    """Estimate a hand bounding box `[x1, y1, x2, y2]` from COCO-17 body
    keypoints, by extrapolating past the wrist along the elbow->wrist forearm
    direction. Returns None if the wrist/elbow aren't confident enough.

    keypoints: (17, 3) as (x, y, confidence) in the frame's pixel coordinates.
    """
    wrist_idx = COCO_R_WRIST if is_right else COCO_L_WRIST
    elbow_idx = COCO_R_ELBOW if is_right else COCO_L_ELBOW
    wrist, elbow = keypoints[wrist_idx], keypoints[elbow_idx]
    if wrist[2] < _MIN_KPT_CONF or elbow[2] < _MIN_KPT_CONF:
        return None

    forearm = wrist[:2] - elbow[:2]
    forearm_len = float(np.linalg.norm(forearm))
    if forearm_len < 1.0:
        return None

    # Hand center sits a bit past the wrist; box side scales with forearm length.
    hand_center = wrist[:2] + 0.4 * forearm
    half = 0.6 * forearm_len
    return np.array([hand_center[0] - half, hand_center[1] - half,
                     hand_center[0] + half, hand_center[1] + half], dtype=np.float32)


def crop_hand(img_bgr: np.ndarray, box_xyxy: np.ndarray, is_right: bool) -> np.ndarray:
    """Crop and normalize a hand patch to (3, 256, 256), ready for the backbone.
    Left hands are flipped horizontally (HaMeR is trained on right hands); the
    caller must flip the predicted pose back."""
    center = (box_xyxy[2:4] + box_xyxy[0:2]) / 2.0
    scale = RESCALE_FACTOR * (box_xyxy[2:4] - box_xyxy[0:2]) / 200.0
    bbox_size = float(expand_to_aspect_ratio(scale * 200.0, BBOX_SHAPE).max())

    img = img_bgr
    c_x = center[0]
    if not is_right:  # flip left hands to look like right hands
        img = img[:, ::-1, :]
        c_x = img.shape[1] - c_x - 1

    trans = _gen_trans(c_x, center[1], bbox_size, IMAGE_SIZE)
    patch = cv2.warpAffine(img, trans, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
    patch = patch[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    patch = (patch - IMAGE_MEAN) / IMAGE_STD
    return np.transpose(patch, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
