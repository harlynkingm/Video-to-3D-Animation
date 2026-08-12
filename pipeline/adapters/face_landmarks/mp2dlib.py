"""Correspondence between MediaPipe's 478 dense face-mesh landmarks and the
standard 68-point Dlib/iBUG scheme that FLAME's own static landmark embedding
(`flame_static_embedding.pkl`) and MICA/ArcFace's 5-point alignment are both
indexed against.

Index table ported from Peizhi Yan's `Mediapipe_2_Dlib_Landmarks`
(https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks, also vendored as
`utils/mp2dlib.py` in `flame-head-tracker`), verified against that source
directly, not re-derived. Where two Mediapipe indices map to one Dlib point,
their coordinates are averaged.
"""

from __future__ import annotations

import numpy as np

# Each entry: the Mediapipe landmark index(es) for Dlib points 1-68, in order.
_MP2DLIB = [
    # Face contour (Dlib 1-17)
    [127], [234], [93], [132, 58], [58, 172], [136], [150], [176], [152],
    [400], [379], [365], [397, 288], [361], [323], [454], [356],
    # Right brow (18-22)
    [156], [70, 63], [105], [66], [107, 55],
    # Left brow (23-27)
    [336, 285], [296], [334], [293, 300], [383],
    # Nose (28-36)
    [168, 6], [197, 195], [5], [4], [98], [97], [2], [326], [327],
    # Right eye (37-42)
    [33], [160], [158], [133], [153], [144],
    # Left eye (43-48)
    [362], [385], [387], [263], [373], [380],
    # Upper lip contour, top (49-55)
    [61], [39], [37], [0], [267], [269], [291],
    # Lower lip contour, bottom (56-60)
    [321], [314], [17], [84], [91],
    # Upper lip contour, bottom (61-65)
    [78], [82], [13], [312], [308],
    # Lower lip contour, top (66-68)
    [317], [14], [87],
]
assert len(_MP2DLIB) == 68

# Dlib's inner-51 (indices 17-67 of the 0-indexed 68 set): everything except
# the jawline/face-contour points (0-16). This is the subset FLAME's static
# landmark embedding (`use_face_contour=False`) actually regresses, the
# jawline isn't a fixed set of FLAME vertices, since it depends on head pose
# (that requires the *dynamic* contour embedding, not used here).
DLIB_INNER_51_START = 17

# Standard dlib-68 indices for the ArcFace/insightface 5-point alignment set,
# matching `flame-head-tracker`'s `get_5_landmarks_from_68` (which in turn
# matches insightface's own convention): the eye points are averaged over
# each eye's 6 contour points, not the corner points alone.
_DLIB_LEFT_EYE = slice(36, 42)
_DLIB_RIGHT_EYE = slice(42, 48)
_DLIB_NOSE_TIP = 30
_DLIB_LEFT_MOUTH = 48
_DLIB_RIGHT_MOUTH = 54


def mediapipe_to_dlib68(landmarks_478: np.ndarray) -> np.ndarray:
    """landmarks_478: (478, D), D is 2 or 3. Returns (68, D)."""
    return np.stack([landmarks_478[idx].mean(axis=0) for idx in _MP2DLIB]).astype(landmarks_478.dtype)


def dlib68_to_flame51(landmarks_68: np.ndarray) -> np.ndarray:
    """landmarks_68: (68, D). Returns (51, D), in the same point order as
    FLAME's static embedding (`smplx.FLAME`'s last 51 output joints when
    `use_face_contour=False`)."""
    return landmarks_68[DLIB_INNER_51_START:]


def dlib68_to_arcface5(landmarks_68: np.ndarray) -> np.ndarray:
    """landmarks_68: (68, D). Returns (5, D): left eye, right eye, nose,
    left mouth corner, right mouth corner, the order `mica_preprocess.norm_crop`
    (and insightface's own `arcface_dst` template) expects."""
    return np.stack([
        landmarks_68[_DLIB_LEFT_EYE].mean(axis=0),
        landmarks_68[_DLIB_RIGHT_EYE].mean(axis=0),
        landmarks_68[_DLIB_NOSE_TIP],
        landmarks_68[_DLIB_LEFT_MOUTH],
        landmarks_68[_DLIB_RIGHT_MOUTH],
    ])
