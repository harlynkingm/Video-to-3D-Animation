"""Crop derivation for MediaPipe's FaceLandmarker.

MediaPipe's face detector is tuned for faces that occupy a meaningful
fraction of the input image, confirmed directly: it misses the face
entirely on a native 3840x2160 frame where the face is a small fraction of
the pixels, but finds it reliably once cropped down to a face-sized region.
So, matching `deca_preprocess.py`'s own crop-first approach, this locates a
face box from body keypoints before ever calling the landmarker.
"""

from __future__ import annotations

import cv2
import numpy as np

CROP_SCALE = 1.6  # wider than DECA's 1.25: MediaPipe's own face mesh needs
# forehead/chin/ear margin for reliable detection, not just the inner face.


def crop_for_landmarker(frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a square, padded region around `box_xyxy` (from
    `deca_preprocess.face_box_from_body_kpts`). Returns `(crop_rgb, offset_xy)`,
    `offset_xy` is the crop's top-left corner in the source frame, needed to
    map the landmarker's output back to full-frame pixel coordinates."""
    h, w = frame_bgr.shape[:2]
    cx, cy = (box_xyxy[0] + box_xyxy[2]) / 2.0, (box_xyxy[1] + box_xyxy[3]) / 2.0
    side = max(box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]) * CROP_SCALE

    x0 = int(np.clip(cx - side / 2, 0, w - 1))
    y0 = int(np.clip(cy - side / 2, 0, h - 1))
    x1 = int(np.clip(cx + side / 2, x0 + 1, w))
    y1 = int(np.clip(cy + side / 2, y0 + 1, h))

    crop_bgr = frame_bgr[y0:y1, x0:x1]
    crop_rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    return crop_rgb, np.array([x0, y0], dtype=np.float32)


def landmarks_to_full_frame(landmarks_norm: np.ndarray, crop_shape: tuple[int, int], offset_xy: np.ndarray) -> np.ndarray:
    """landmarks_norm: (N, 2) or (N, 3), MediaPipe's [0,1]-normalized output
    (x, y relative to the crop; z left untouched, it's a relative depth, not
    a pixel coordinate). Returns full-frame pixel coordinates, same shape."""
    crop_h, crop_w = crop_shape[:2]
    out = landmarks_norm.copy()
    out[:, 0] = landmarks_norm[:, 0] * crop_w + offset_xy[0]
    out[:, 1] = landmarks_norm[:, 1] * crop_h + offset_xy[1]
    return out
