"""Temporal smoothing to remove per-frame jitter from the pose estimators'
output, applied inside stage 2 (body) and stage 4 (hands) so their saved output
is already higher-quality than the raw model prediction.

Why the two stages need very different strengths: GVHMR (stage 2) runs a
temporal transformer over the whole clip, so the body arrives with frame-to-frame
consistency and only needs light polish. HaMeR (stage 4) infers every frame
independently from an isolated crop with no temporal model at all, so the hands
are far jitterier and need heavier smoothing -- plus per-frame validity handling,
since a hand that wasn't detected on some frames must not bleed its placeholder
identity pose into its detected neighbours.

Rotations are smoothed in quaternion space (Savitzky-Golay per joint, with
sign-continuity enforced) rather than directly on axis-angle: naively filtering
across an axis-angle wraparound produces spikes instead of removing them.
Translation is smoothed with a zero-phase (filtfilt) Butterworth low-pass, which
adds no lag. Ported from this project's own ComfyUI `SmoothSMPLMotion` node
(savgol-in-quaternion + butterworth-filtfilt), extended here with validity-aware
gap handling for the hands.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.spatial.transform import Rotation

POSE_AXIS_DIM = 3

# Internal defaults -- the exposed knobs are the savgol *window* (per stage) and
# the butterworth *cutoff* (body only); polynomial/filter order are left fixed at
# the values the ComfyUI node proved out, since they're rarely worth touching.
DEFAULT_POLYORDER = 3
DEFAULT_BUTTER_ORDER = 2


def _odd_window(window: int, n_frames: int) -> int | None:
    """Clamp a requested savgol window to an odd value in [3, n_frames], or None
    if the clip is too short (< 3 frames) to filter meaningfully."""
    if n_frames < 3:
        return None
    w = min(window, n_frames)
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def _fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid rows of `values` (T, C) per channel from the valid ones, so
    the filter downstream never sees the placeholder poses on undetected frames.
    `np.interp`'s own default boundary behavior gives exactly the two occlusion
    cases their right treatment for free: an *interior* run of invalid frames
    (bounded by a valid frame on both sides -- the tracked thing reappears) is
    linearly interpolated between those two real values; a *leading* or
    *trailing* run (invalid all the way to one end of the clip -- it never
    reappears, or wasn't yet detected at the start) has no second real endpoint
    to interpolate toward, so `np.interp` holds it constant at the one real
    value it does have (`left`/`right` default to `fp[0]`/`fp[-1]`) -- a freeze,
    not a fabricated guess. This is the caller's actual occlusion contract, not
    just an internal filtering detail, so downstream code depends on it."""
    frame_idx = np.arange(values.shape[0])
    valid_idx = frame_idx[valid]
    filled = values.copy()
    for channel in range(values.shape[1]):
        filled[:, channel] = np.interp(frame_idx, valid_idx, values[valid, channel])
    return filled


def smooth_rotation_sequence(
    axis_angle: np.ndarray, window: int, polyorder: int = DEFAULT_POLYORDER, valid: np.ndarray | None = None
) -> np.ndarray:
    """Smooth a per-frame axis-angle rotation sequence in quaternion space.

    Args:
        axis_angle: (T, ...) where the trailing dims flatten to a multiple of 3
            (e.g. (T, 3), (T, 63), (T, 15, 3)) -- each 3-vector a joint rotation.
        window: Savitzky-Golay window in frames (clamped to odd, <= T, >= 3).
        polyorder: polynomial order fit within each window (< window).
        valid: optional (T,) bool. Where given, invalid frames are filled before
            filtering: interpolated if bounded by valid frames on both sides
            (the tracked thing reappears), held constant at the nearest valid
            frame if not (occlusion runs to either end of the clip) -- see
            `_fill_invalid`. The filter then runs over the filled series.

    Returns the same shape/dtype, smoothed. Returned unchanged when the clip is
    too short, or when `valid` marks too few real frames to smooth meaningfully.
    """
    axis_angle = np.asarray(axis_angle)
    original_shape = axis_angle.shape
    n_frames = original_shape[0]
    flat = axis_angle.reshape(n_frames, -1)
    if flat.shape[1] % POSE_AXIS_DIM != 0:
        raise ValueError(f"rotation sequence trailing dims not divisible by 3: {original_shape}")
    n_joints = flat.shape[1] // POSE_AXIS_DIM
    joints = flat.reshape(n_frames, n_joints, POSE_AXIS_DIM)

    w = _odd_window(window, n_frames)
    if w is None:
        return axis_angle
    poly = min(polyorder, w - 1)

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if int(valid.sum()) < max(3, poly + 1):
            return axis_angle  # too few detected frames to smooth meaningfully

    out = np.zeros_like(joints)
    for j in range(n_joints):
        quats = Rotation.from_rotvec(joints[:, j, :]).as_quat()  # (T, 4) xyzw

        # Sign-continuity across *detected* frames: a unit quaternion and its
        # negation are the same rotation, so flip each so it shares a hemisphere
        # with the previous real one -- otherwise interpolation/filtering treats a
        # sign flip as a huge jump.
        last = None
        for t in range(n_frames):
            if valid is None or valid[t]:
                if last is not None and float(np.dot(quats[t], last)) < 0:
                    quats[t] = -quats[t]
                last = quats[t]

        if valid is not None:
            quats = _fill_invalid(quats, valid)

        smoothed = savgol_filter(quats, w, poly, axis=0)
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        smoothed /= norms
        out[:, j, :] = Rotation.from_quat(smoothed).as_rotvec()

    return out.reshape(original_shape).astype(axis_angle.dtype, copy=False)


def smooth_translation_sequence(transl: np.ndarray, cutoff: float, order: int = DEFAULT_BUTTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass on a (T, 3) root-translation sequence.
    `cutoff` is a fraction of Nyquist (0-0.5). Returned unchanged when the clip is
    too short for `filtfilt`'s edge padding."""
    transl = np.asarray(transl)
    b, a = butter(order, cutoff)
    padlen = 3 * max(len(a), len(b))
    if transl.shape[0] <= padlen:
        return transl
    return filtfilt(b, a, transl, axis=0).astype(transl.dtype, copy=False)
