"""Face alignment for MICA's ArcFace backbone: unlike DECA's tolerant bbox
crop, ArcFace-family networks are trained on precisely aligned faces and are
materially less accurate on an imprecisely-framed crop, this warps 5 facial
landmark points onto a fixed canonical template via a similarity transform,
the standard `insightface` alignment (`utils.face_align.norm_crop`).

This module only does the geometry: it takes 5 already-detected points and
produces the aligned 112x112 crop. *Sourcing* those 5 points from a frame
(a real landmark detector) is not implemented here, MICA's own pipeline
uses RetinaFace; this project's face-capture stage supplies them from its own
landmark detector once that module exists.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

IMAGE_SIZE = 112

# insightface's own canonical 5-point template for 112x112 alignment (left
# eye, right eye, nose, left mouth corner, right mouth corner), a fixed
# public constant, not fit from this project's own data.
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def norm_crop(img_bgr: np.ndarray, landmarks_5pt: np.ndarray) -> np.ndarray:
    """Align and crop a face to (112, 112, 3) BGR via a least-squares
    similarity transform (rotation + uniform scale + translation) from
    `landmarks_5pt` (5,2 pixel coords, left eye/right eye/nose/left mouth/
    right mouth order) onto `REFERENCE_LANDMARKS`."""
    tform = SimilarityTransform.from_estimate(landmarks_5pt.astype(np.float32), REFERENCE_LANDMARKS)
    matrix = tform.params[0:2, :]
    return cv2.warpAffine(img_bgr, matrix, (IMAGE_SIZE, IMAGE_SIZE), borderValue=0.0)


def normalize_for_arcface(aligned_bgr: np.ndarray) -> np.ndarray:
    """(112,112,3) BGR uint8 -> (3,112,112) float32, ArcFace's own
    `(pixel - 127.5) / 127.5` normalization (maps [0,255] to [-1,1]), RGB."""
    rgb = aligned_bgr[:, :, ::-1].astype(np.float32)
    normalized = (rgb - 127.5) / 127.5
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)
