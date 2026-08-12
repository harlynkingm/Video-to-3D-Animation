"""Geometric gaze estimation from MediaPipe's own raw iris landmarks, not
MediaPipe's built-in blendshape output (a separate, black-box model bundled
in the same `.task` file). FLAME's static landmark embedding has no
iris/pupil points, so gaze can't come from the FLAME track at all; this
measures and calibrates the signal directly instead.

Technique: for each eye, the iris center's position is expressed as a ratio
along the eye's own inner-to-outer-corner axis (horizontal), and separately
as a vertical displacement normalized by that same horizontal eye width, not
the top-to-bottom lid span (`_vertical_displacement`, lid span shrinks
toward zero near a blink, so it isn't a stable normalizer). Both are
scale-invariant, independent of the face's distance from the camera or its
pixel size. This is a coarse eye-in-head proxy, not a full 3D gaze solve,
and carries some head-pose dependence.

Sign convention is handled structurally, not by a manual flip: "inner"/
"outer" corners are defined per-eye (nearer the nose vs. farther), so
`EyeLookIn`/`EyeLookOut` fall out of the same ratio computation for both
eyes without an explicit left/right sign flip.

Per-frame pipeline (`_processed_gaze_ratios`): raw per-eye ratios
(`eye_gaze_ratios`) -> gap-fill MediaPipe detection dropout -> per-clip
percentile calibration (`_calibrate_signed_ratio`, stretching each clip's
own 5th-95th percentile range to fill [-1, 1]) -> one-euro temporal
smoothing. `GAZE_ONE_EURO_MIN_CUTOFF_HZ`/`BETA` and `ANGLE_SCALE_DEG` are
untuned placeholders.
"""

from __future__ import annotations

import numpy as np

# MediaPipe's canonical 478-point face mesh: iris landmarks (indices
# 468-477), confirmed against MediaPipe's own `FACEMESH_LEFT_IRIS`/
# `FACEMESH_RIGHT_IRIS` connection data (not assumed), right eye's iris
# ring is 469-472 (center 468), left eye's is 474-477 (center 473). "Right"/
# "left" here are subject-anatomical, matching `mp2dlib.py`'s own existing
# convention (its right-eye Dlib points map from MediaPipe indices in the
# 33/133 neighborhood, the same region used below).
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Eye-corner and lid landmarks, MediaPipe's standard canonical indices.
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 159, 145
LEFT_EYE_OUTER, LEFT_EYE_INNER = 263, 362
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 386, 374

# Untuned placeholder: maps a normalized +-1 ratio to a plausible eye-Euler
# degree range. Real human eye rotation is roughly +-30-45 deg horizontally,
# less vertically, needs real-data validation, not assumed correct.
ANGLE_SCALE_DEG = 30.0

# Percentile calibration range (mirrors face_eyelid.py's OPEN_PERCENTILE/
# CLOSED_PERCENTILE, same rationale: absolute ratio scale is subject- and
# camera-dependent, so "extreme" is defined relative to what THIS clip's
# own eyes actually did).
CALIBRATION_LOW_PERCENTILE = 5.0
CALIBRATION_HIGH_PERCENTILE = 95.0
# Below this, the clip's own observed range is too degenerate to calibrate
# against (e.g. genuinely no real gaze movement the whole clip), same
# reasoning as face_eyelid.MIN_CALIBRATION_RANGE, prevents dividing by a
# near-zero scale and amplifying noise into a spurious full-extreme reading.
MIN_CALIBRATION_SCALE = 0.05

GAZE_ONE_EURO_MIN_CUTOFF_HZ = 0.6  # real eye saccades are fast, matches expression's own cutoff, not jaw's slower one
GAZE_ONE_EURO_BETA = 1.0


def _axis_ratio(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Signed, centered [-1, 1] position of `point` projected onto the
    `start`->`end` axis (0.5 ratio, i.e. output 0, is the midpoint),
    scale-invariant: doesn't depend on the eye's pixel size. `point`/`start`/
    `end`: (..., 2)."""
    axis = end - start
    axis_len_sq = np.clip((axis ** 2).sum(axis=-1), 1e-9, None)
    ratio = ((point - start) * axis).sum(axis=-1) / axis_len_sq
    return (ratio - 0.5) * 2.0


def _vertical_displacement(iris: np.ndarray, top: np.ndarray, bottom: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Signed vertical iris displacement from the eye's own vertical
    center, normalized by the eye's own HORIZONTAL width (inner-to-outer
    distance), not the top-to-bottom lid span. That was the original
    design and a real, measured bug: the lid span shrinks toward zero as
    the eye closes, so the ratio blows up numerically right when a blink is
    approaching (measured directly: raw values reaching -5 to +7 on real
    footage, where a well-behaved ratio should stay near [-1, 1]), and
    those outliers dominated the correlation against real ground truth. Eye
    width never approaches zero the same way, so this can't blow up the
    same way, positive is toward the lower lid (`EyeLookDown`), matching
    the previous convention's own sign so nothing downstream needed to
    change meaning, only stability."""
    center_y = (top[..., 1] + bottom[..., 1]) / 2.0
    width = np.clip(np.linalg.norm(outer - inner, axis=-1), 1e-6, None)
    return (iris[..., 1] - center_y) / width


def eye_gaze_ratios(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """`landmarks`: (F, 478, 2 or 3) MediaPipe full-frame landmarks (only x/y
    used). Returns per-eye horizontal/vertical signed ratios, raw and
    uncalibrated (see module docstring for the full processing pipeline
    applied on top before these reach the CSV): `h` > 0 is toward the outer
    corner (`EyeLookOut`), `v` > 0 is toward the lower lid (`EyeLookDown`)
   , for both eyes, since "inner"/"outer" are already anatomically
    per-eye.
    """
    xy = landmarks[..., :2]
    return {
        "right_h": _axis_ratio(xy[:, RIGHT_IRIS_CENTER], xy[:, RIGHT_EYE_INNER], xy[:, RIGHT_EYE_OUTER]),
        "right_v": _vertical_displacement(
            xy[:, RIGHT_IRIS_CENTER], xy[:, RIGHT_EYE_TOP], xy[:, RIGHT_EYE_BOTTOM], xy[:, RIGHT_EYE_INNER], xy[:, RIGHT_EYE_OUTER],
        ),
        "left_h": _axis_ratio(xy[:, LEFT_IRIS_CENTER], xy[:, LEFT_EYE_INNER], xy[:, LEFT_EYE_OUTER]),
        "left_v": _vertical_displacement(
            xy[:, LEFT_IRIS_CENTER], xy[:, LEFT_EYE_TOP], xy[:, LEFT_EYE_BOTTOM], xy[:, LEFT_EYE_INNER], xy[:, LEFT_EYE_OUTER],
        ),
    }


def _calibrate_signed_ratio(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rescales `raw` so this clip's own 5th/95th percentile (over `valid`
    frames only) lands at roughly [-1, 1], the real fix for the magnitude
    undershoot found against ground truth (see module docstring). Not
    two-sided independently (a separate positive/negative scale): a single
    symmetric scale from the larger-magnitude side keeps the calibration
    simple and avoids amplifying whichever side happened to have less real
    motion in this particular clip into looking equally extreme."""
    if not valid.any():
        return np.zeros_like(raw)
    p_low = np.percentile(raw[valid], CALIBRATION_LOW_PERCENTILE)
    p_high = np.percentile(raw[valid], CALIBRATION_HIGH_PERCENTILE)
    scale = max(abs(p_low), abs(p_high), MIN_CALIBRATION_SCALE)
    return np.clip(raw / scale, -1.0, 1.0)


def _processed_gaze_ratios(landmarks: np.ndarray, valid: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Runs the full pipeline (see module docstring) once, shared by
    `direct_arkit_gaze_channels` and `eye_euler_degrees` so the two can
    never compute inconsistent values for the same underlying signal."""
    from ..motion_smoothing import fill_invalid, one_euro_filter_sequence

    raw = eye_gaze_ratios(landmarks)

    processed = {}
    for key in ("right_h", "right_v", "left_h", "left_v"):
        filled = fill_invalid(raw[key][:, None], valid)[:, 0]
        calibrated = _calibrate_signed_ratio(filled, valid)
        smoothed = one_euro_filter_sequence(
            calibrated[:, None], fps, GAZE_ONE_EURO_MIN_CUTOFF_HZ, GAZE_ONE_EURO_BETA,
        )[:, 0]
        processed[key] = smoothed
    return processed


def direct_arkit_gaze_channels(landmarks: np.ndarray, valid: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Returns the 4 horizontal ARKit `EyeLookIn/Out*` channels (Group D),
    each (F,) in [0, 1], sign-split per eye, never simultaneously nonzero
    within an eye (matches `direct_arkit_jaw_channels`'s convention).

    Horizontal only: this geometric measurement beats MediaPipe's own native
    blendshape output for horizontal gaze, but the reverse holds for
    *vertical* gaze, so `stage_9_capture_face.py`'s channel merge sources
    `EyeLookUp/Down*` from MediaPipe directly instead. `right_v`/`left_v` are
    still computed here since `eye_euler_degrees` needs them for
    `LeftEyePitch`/`RightEyePitch`, which have no MediaPipe equivalent.

    `valid`: MediaPipe's per-frame detection mask (`mp_valid`). `fps`: for
    the one-euro smoothing pass. Not blink-aware by design, gaze holds its
    own last measured position regardless of eyelid state.

    In/Out sign is measured against ground truth rather than assumed: `h`'s
    raw sign (positive = iris toward this eye's own outer corner) correlates
    positively with `EyeLookIn*` and negatively with `EyeLookOut*`, for both
    eyes, the opposite of the naive "toward-nose = In" assumption."""
    ratios = _processed_gaze_ratios(landmarks, valid, fps)
    return {
        "EyeLookInRight": np.clip(ratios["right_h"], 0.0, 1.0),
        "EyeLookOutRight": np.clip(-ratios["right_h"], 0.0, 1.0),
        "EyeLookInLeft": np.clip(ratios["left_h"], 0.0, 1.0),
        "EyeLookOutLeft": np.clip(-ratios["left_h"], 0.0, 1.0),
    }


def eye_euler_degrees(landmarks: np.ndarray, valid: np.ndarray, fps: float) -> np.ndarray:
    """The 6 `{Left,Right}Eye{Yaw,Pitch,Roll}` CSV columns, degrees. `Roll`
    is always 0, gaze direction alone can't carry eyeball roll (torsion),
    and this pipeline has no other source for it. Same calibrated, smoothed
    signal `direct_arkit_gaze_channels` uses, see that function's own
    docstring for the argument contract.

    `RightEyeYaw` is negated relative to `right_h`, `LeftEyeYaw` isn't,
    verified against a real ground-truth capture (`left_h` vs.
    `LeftEyeYaw` corr +0.97; `right_h` vs. `RightEyeYaw` corr -0.95, i.e.
    already needs the flip applied here). This asymmetry is real, not a
    residual bug: `h` is per-eye-anatomy-relative (positive = toward that
    eye's own outer corner, which is a mirrored, opposite screen-space
    direction between the two eyes, confirmed directly, the left eye's
    outer corner sits at higher image-x than its inner corner, the right
    eye's the reverse), while ARKit's own Yaw is world/screen-frame-relative
    (the same absolute direction reads as positive for both eyes), so
    only one of the two eyes' raw per-eye sign happens to already agree with
    the world-frame convention; the other structurally can't without this
    per-eye correction."""
    ratios = _processed_gaze_ratios(landmarks, valid, fps)
    n = landmarks.shape[0]
    zeros = np.zeros(n, dtype=np.float32)
    return np.stack([
        ratios["left_h"] * ANGLE_SCALE_DEG, ratios["left_v"] * ANGLE_SCALE_DEG, zeros,
        -ratios["right_h"] * ANGLE_SCALE_DEG, ratios["right_v"] * ANGLE_SCALE_DEG, zeros,
    ], axis=1).astype(np.float32)
