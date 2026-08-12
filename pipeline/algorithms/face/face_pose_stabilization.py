"""Per-frame head orientation from MediaPipe's dense 478-point landmarks via
a rigid-subset Kabsch fit, ported from BlendCap's stabilization step
(https://github.com/Arcomade/BlendCap/blob/main/pipeline/face_landmark_topology.py,
`RIGID_INDICES`/`similarity_fit`/`stabilize_landmarks`).

`RIGID_INDICES` is a fixed 19-point subset (eye corners, nose bridge column,
forehead arc, temples) that deliberately excludes the chin/lips/brows/
nose-wings, points that move under expression. A rotation fit against only
those points can't trade off against jaw pose the way `face_landmark_fit.
fit_clip`'s joint optimization can, since jaw motion isn't in its input at
all, a head-orientation signal independent of DECA's own per-frame
regression.
"""

from __future__ import annotations

import numpy as np

# Subject-relative semantic indices, from BlendCap's face_landmark_topology.py.
LEFT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382,
                 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155,
                  133, 173, 157, 158, 159, 160, 161, 246]
NOSE_BRIDGE = 168
CHIN_CENTER = 152
LEFT_BROW_LOWER = [285, 295, 282, 283, 276]
RIGHT_BROW_LOWER = [55, 65, 52, 53, 46]
MOUTH_CORNER_LEFT = 291
MOUTH_CORNER_RIGHT = 61
INNER_LIP_PAIRS = [(13, 14), (82, 87), (81, 178), (80, 88),
                   (312, 317), (311, 402), (310, 318)]

# Landmarks that barely move under facial expression, eye corners, nose
# bridge column, upper forehead. The rotation fit below is computed against
# only these, so expression motion (a wide-open jaw, a raised brow) can never
# leak into the orientation estimate.
RIGID_INDICES = [33, 133, 362, 263,
                 168, 6, 197, 195,
                 10, 109, 338, 67, 297,
                 151, 9, 8,
                 127, 356, 234, 454]

# How many of a clip's lowest-expression-activity frames to median-average
# into the per-clip neutral template that every frame's rigid points are
# then Kabsch-fit against.
DEFAULT_N_NEUTRAL_FRAMES = 60


def _similarity_fit(src: np.ndarray, dst: np.ndarray, z_weight: float = 1.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping src -> dst (both (P, 3)):
    dst ~ s * R @ src + t. z_weight < 1 downweights the z axis before the
    fit (BlendCap's own default, treating MediaPipe's relative-z as its
    noisiest channel), but that downweighting introduces a rotation-
    dependent bias in the recovered R whenever the true rotation mixes z
    into x/y, not just added noise tolerance. BlendCap never notices this
    because it only uses the *relative* alignment for expression
    stabilization; this module uses R as an absolute orientation signal
    (`stabilize_orientation`), where the bias would matter, so z_weight is
    kept at 1.0 by default here."""
    w = np.array([1.0, 1.0, float(z_weight)])
    src_w, dst_w = src * w, dst * w
    mu_s, mu_d = src_w.mean(axis=0), dst_w.mean(axis=0)
    a, b = src_w - mu_s, dst_w - mu_d
    cov = b.T @ a
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    var = (a ** 2).sum()
    s = (S * np.diag(D)).sum() / max(var, 1e-12)
    t = dst.mean(axis=0) - s * R @ src.mean(axis=0)
    return s, R, t


def _canonical_frame_from(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(origin, R_rows, scale) s.t. canonical = ((p - origin) @ R_rows.T) * scale.
    +X = subject left, +Y = up (chin->bridge), +Z = out of the face."""
    eye_l = landmarks[LEFT_EYE_RING].mean(axis=0)
    eye_r = landmarks[RIGHT_EYE_RING].mean(axis=0)
    origin = (eye_l + eye_r) / 2.0
    left = eye_l - eye_r
    io = np.linalg.norm(left)
    left = left / max(io, 1e-9)
    up = landmarks[NOSE_BRIDGE] - landmarks[CHIN_CENTER]
    up = up - left * (up @ left)
    up = up / max(np.linalg.norm(up), 1e-9)
    fwd = np.cross(left, up)
    return origin, np.stack([left, up, fwd], axis=0), 1.0 / max(io, 1e-9)


def _to_canonical(landmarks: np.ndarray, origin: np.ndarray, R_rows: np.ndarray, scale: float) -> np.ndarray:
    return ((landmarks - origin) @ R_rows.T) * scale


def _compute_activity(canon_landmarks: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Per-frame expression-activity score for picking neutral frames. Lower
    = more neutral; undetected frames get +inf. Mouth-gap, mouth-corner, and
    brow deviation only, the same three cheap signals BlendCap uses, since
    we only need "is this frame moving" not a full activity model."""
    n = len(canon_landmarks)
    act = np.full(n, np.inf, dtype=np.float64)
    idx = np.flatnonzero(detected)
    if len(idx) == 0:
        return act
    lm = canon_landmarks[idx]

    gap = np.zeros(len(idx))
    for u, l in INNER_LIP_PAIRS:
        gap += np.abs(lm[:, u, 1] - lm[:, l, 1])
    gap /= len(INNER_LIP_PAIRS)

    corners = lm[:, [MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT], :2]
    corner_dev = np.linalg.norm(corners - np.median(corners, axis=0), axis=2).mean(axis=1)

    brows = lm[:, LEFT_BROW_LOWER + RIGHT_BROW_LOWER, 1]
    brow_dev = np.abs(brows - np.median(brows, axis=0)).mean(axis=1)

    act[idx] = 2.0 * gap + 1.5 * corner_dev + 1.0 * brow_dev
    return act


def stabilize_orientation(
    landmarks: np.ndarray, detected: np.ndarray, n_neutral_frames: int = DEFAULT_N_NEUTRAL_FRAMES,
) -> np.ndarray:
    """Per-frame (F, 3, 3) head rotation relative to the clip's own neutral
    pose, from MediaPipe's dense landmarks (`face_landmarks_adapter.KEY_LANDMARKS`, 
    full-frame pixel x/y + relative z). Undetected frames get `np.nan`.
    Two-pass, matching BlendCap's `stabilize_landmarks`: pass one
    canonicalizes against the first detected frame (good enough to rank
    frames by expression activity), then a per-clip neutral template is
    built from the lowest-activity frames and every frame is re-fit against
    that template directly.

    landmarks: (F, 478, 3). detected: (F,) bool.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    detected = np.asarray(detected, dtype=bool)
    n = len(landmarks)
    det_idx = np.flatnonzero(detected)
    if len(det_idx) == 0:
        return np.full((n, 3, 3), np.nan)

    ref = landmarks[det_idx[0]]
    o0, R0, s0 = _canonical_frame_from(ref)
    ref_c = _to_canonical(ref, o0, R0, s0)
    rig_ref = ref_c[RIGID_INDICES]

    stab_a = np.zeros_like(landmarks)
    for i in det_idx:
        s, R, t = _similarity_fit(landmarks[i][RIGID_INDICES], rig_ref)
        stab_a[i] = (s * (R @ landmarks[i].T)).T + t

    activity = _compute_activity(stab_a, detected)
    order = np.argsort(activity)
    n_pick = int(np.clip(len(det_idx) // 10, 5, n_neutral_frames))
    n_pick = min(n_pick, len(det_idx))
    neutral_frames = sorted(int(i) for i in order[:n_pick])
    template_a = np.median(stab_a[neutral_frames], axis=0)

    ot, Rt, st = _canonical_frame_from(template_a)
    template = _to_canonical(template_a, ot, Rt, st)
    rig_t = template[RIGID_INDICES]

    rotmat = np.full((n, 3, 3), np.nan)
    for i in det_idx:
        _, R, _ = _similarity_fit(landmarks[i][RIGID_INDICES], rig_t)
        rotmat[i] = R
    return rotmat.astype(np.float32)
