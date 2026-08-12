"""Face-crop preprocessing for DECA: turn a frame + a face bounding box into
the exact normalized 224x224 patch the model expects.

Ported from `decalib/datasets/datasets.py`'s `TestData`, restricted to
inference (no augmentation), with one project-specific substitution: DECA's
own demo runs a dedicated face detector (FAN) to find the crop box. This
project instead derives it from the body pose's own nose/eye/ear keypoints
(`face_box_from_body_kpts`), the same idea as `hamer_preprocess.hand_box_from_body_kpts`
for hands, isolated in its own function so a dedicated face detector can
replace it later without touching the rest of this module. It's written
generically (COCO-17 keypoints in, a box out) so MICA and the landmark
detector can reuse it too, rather than each deriving their own crop.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = 224
BBOX_SCALE = 1.25  # padding around the detected face box, matching DECA's own `scale`

# COCO-17 keypoint indices (what gvhmr_vitpose outputs).
COCO_NOSE, COCO_LEYE, COCO_REYE, COCO_LEAR, COCO_REAR = 0, 1, 2, 3, 4
_MIN_KPT_CONF = 0.3
_FACE_KPT_INDICES = (COCO_NOSE, COCO_LEYE, COCO_REYE, COCO_LEAR, COCO_REAR)


def face_box_from_body_kpts(keypoints: np.ndarray) -> np.ndarray | None:
    """Estimate a face bounding box `[x1, y1, x2, y2]` from COCO-17 body
    keypoints, as the confidently-detected subset of nose/eyes/ears bracketed
    by a square sized off their own spread. Returns None if fewer than 2 of
    the 5 face keypoints are confident (a single point gives no size
    estimate).

    keypoints: (17, 3) as (x, y, confidence) in the frame's pixel coordinates.
    """
    face_kpts = keypoints[list(_FACE_KPT_INDICES)]
    confident = face_kpts[face_kpts[:, 2] >= _MIN_KPT_CONF]
    if len(confident) < 2:
        return None

    xs, ys = confident[:, 0], confident[:, 1]
    left, right, top, bottom = xs.min(), xs.max(), ys.min(), ys.max()
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    size = max(right - left, bottom - top) * BBOX_SCALE
    # Nose/eyes cluster in the upper half of a real face box (chin is
    # unobserved); DECA's own preprocessing applies the same downward
    # recentering for its landmark-derived boxes.
    center[1] += size * 0.12
    half = size / 2.0
    return np.array([center[0] - half, center[1] - half, center[0] + half, center[1] + half], dtype=np.float32)


def crop_face(img_bgr: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """Crop and normalize a face patch to (3, 224, 224), ready for the DECA
    encoder. Unlike HaMeR's ImageNet mean/std normalization, DECA's own
    preprocessing only rescales pixels to [0, 1], no mean/std subtraction."""
    x1, y1, x2, y2 = box_xyxy
    size = max(x2 - x1, y2 - y1)
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    src = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [0, IMAGE_SIZE - 1], [IMAGE_SIZE - 1, 0]], dtype=np.float32)
    trans = cv2.getAffineTransform(src, dst)

    patch = cv2.warpAffine(img_bgr, trans, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
    patch = patch[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
    return np.transpose(patch, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
