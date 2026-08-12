"""Geometric eyelid aperture measurement (Group E: `EyeBlink{L,R}`,
`EyeWide{L,R}`) from MediaPipe's own raw landmarks, not FLAME's tracked mesh,
FLAME's PCA expression fit undershoots real blink depth, so eyelid state
is measured directly from MediaPipe's dense eye-contour landmarks instead.

Per-clip percentile calibration: eyelid aperture (vertical lid distance
normalized by eye width) is subject- and camera-dependent, so "closed" and
"open" are defined relative to what this clip's own eyes did.

Hysteresis (seed/release, mirroring `contact_detection.py`'s own
contact-event logic) turns the noisy per-frame closure signal into clean
blink events instead of a smeared partial closure. Within a recognized
event, only frames whose value crosses `BLINK_LOCK_CLOSURE` snap to a crisp
1.0, entry/exit frames keep their own smoothed value, so a blink ramps the
way a real eyelid does instead of jumping instantly (see `_crisp_blink`).

One-euro temporal smoothing (`motion_smoothing.one_euro_filter_sequence`)
runs on the calibrated blink/wide signal before hysteresis, mirroring
`face_gaze.py`'s own fill_invalid -> calibrate -> smooth pipeline.
`EYELID_ONE_EURO_*` follow jaw's low-cutoff/high-beta pair (heavy smoothing
at rest, fast unlock once a blink's velocity ramps up) rather than gaze's,
since a blink is a discrete fast transition closer to jaw's open/close
profile than to a continuous saccade.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..contact_detection import contiguous_true_runs
from ..motion_smoothing import fill_invalid, one_euro_filter_sequence
from .face_gaze import LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP

OPEN_PERCENTILE = 95.0
CLOSED_PERCENTILE = 5.0

BLINK_SEED_CLOSURE = 0.75  # a frame must cross this to seed a blink event
BLINK_RELEASE_CLOSURE = 0.35  # a rolling window must stay above this to remain "in" the event
BLINK_WINDOW = 3  # frames, shorter than contact_detection's own CONTACT_WINDOW; a real blink is brief

# Within a recognized blink event (elevated/seed above), only frames whose
# own smoothed value already exceeds this get force-set to literal 1.0,
# everything else in the event keeps its own already one-euro-smoothed
# value, which is what produces the ramp (see `_crisp_blink` for why this
# must be a plain per-frame threshold, not a rolling-max gate). Calibrated
# against a real ground-truth-paired capture to be the highest value that
# still lets every real blink reach the crisp plateau.
BLINK_LOCK_CLOSURE = 0.8

# `core`'s plain per-frame threshold (see `_crisp_blink`) can noise-dip a
# frame just under `BLINK_LOCK_CLOSURE` in the middle of an otherwise
# fully-closed stretch. `_bridge_short_core_gaps` bridges only a short
# INTERIOR gap bounded by `core` on both sides, never at an event's own
# edges (those must stay untouched for `_crisp_blink`'s ramp-softening to
# work). Kept small so a longer, genuine partial reopening isn't papered
# over as noise.
BLINK_CORE_MAX_GAP_FRAMES = 3

EYELID_ONE_EURO_MIN_CUTOFF_HZ = 0.15  # heavy smoothing at rest, matches jaw's own open/close profile
EYELID_ONE_EURO_BETA = 4.0  # unlock fast once a real blink's own velocity ramps up

# Below this, the clip's own open<->closed range is too degenerate to
# calibrate against (e.g. an eye that genuinely never blinks the whole
# clip), without this guard, a near-zero range amplifies float noise into
# a spurious full-closure reading on every frame. Defaults to "never
# closes" rather than guessing, since blinking is the rare event, not the
# baseline state.
MIN_CALIBRATION_RANGE = 0.05


def _eye_openness_ratio(landmarks: np.ndarray, top: int, bottom: int, inner: int, outer: int) -> np.ndarray:
    """Vertical lid distance normalized by the eye's own horizontal width
    (scale-invariant, same reasoning as `face_gaze.py`'s gaze ratios)."""
    xy = landmarks[..., :2]
    vertical = np.linalg.norm(xy[:, top] - xy[:, bottom], axis=-1)
    width = np.clip(np.linalg.norm(xy[:, outer] - xy[:, inner], axis=-1), 1e-6, None)
    return vertical / width


def _calibrated_blink_and_wide(openness_ratio: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip percentile calibration -> raw (uncrisp) blink closure [0,1]
    and wide-eye [0,1] signals, before hysteresis."""
    if not valid.any():
        zeros = np.zeros_like(openness_ratio)
        return zeros, zeros

    p_open = np.percentile(openness_ratio[valid], OPEN_PERCENTILE)
    p_closed = np.percentile(openness_ratio[valid], CLOSED_PERCENTILE)
    p_mid = np.percentile(openness_ratio[valid], 50.0)

    if p_open - p_closed < MIN_CALIBRATION_RANGE:
        return np.zeros_like(openness_ratio), np.zeros_like(openness_ratio)

    openness_norm = np.clip((openness_ratio - p_closed) / (p_open - p_closed), 0.0, 1.0)
    blink_raw = 1.0 - openness_norm

    wide_range = max(p_open - p_mid, 1e-6)
    wide = np.clip((openness_ratio - p_open) / wide_range, 0.0, 1.0)

    return blink_raw, wide


def _bridge_short_core_gaps(core: np.ndarray, max_gap: int) -> np.ndarray:
    """Fills a `core` False-run with True when it's short (`<= max_gap`) AND
    bounded by True on both sides, an interior noise dip, not a real edge.
    A run touching either array boundary (index 0 or the last index) is
    never bridged: with no True neighbor on that side, there's nothing to
    say it's a dip rather than a genuine ramp start/end."""
    bridged = core.copy()
    for start, end in contiguous_true_runs(~core):
        if start == 0 or end == len(core) - 1:
            continue
        if (end - start + 1) <= max_gap and core[start - 1] and core[end + 1]:
            bridged[start:end + 1] = True
    return bridged


def _crisp_blink(blink_raw: np.ndarray) -> np.ndarray:
    """Hysteresis lock/release (same idea as `contact_detection.detect_
    contact_events`) gates which frames belong to a recognized blink event
    (`elevated`/`seed`, a rolling-max test, deliberately wide so a real
    event's own ramp-in/ramp-out edges count as part of it). Within that
    event, only frames whose own value already reaches `BLINK_LOCK_CLOSURE`
    get force-set to literal 1.0 (`core`, a plain per-frame test, not
    rolling-max, a rolling max would let a high neighbor pull an edge
    frame's lower value over the lock threshold too), the fully-closed
    plateau reads as crisp while entry/exit frames keep their own
    one-euro-smoothed value, producing a ramp instead of an instant jump.
    `_bridge_short_core_gaps` patches short interior noise gaps in `core`
   , see `BLINK_CORE_MAX_GAP_FRAMES`."""
    rolling_max = maximum_filter1d(blink_raw, size=BLINK_WINDOW, mode="nearest")
    elevated = rolling_max > BLINK_RELEASE_CLOSURE
    seed = blink_raw > BLINK_SEED_CLOSURE
    core = _bridge_short_core_gaps(blink_raw > BLINK_LOCK_CLOSURE, BLINK_CORE_MAX_GAP_FRAMES)

    crisp = blink_raw.copy()
    for start, end in contiguous_true_runs(elevated):
        if seed[start:end + 1].any():
            crisp[start:end + 1] = np.where(core[start:end + 1], 1.0, blink_raw[start:end + 1])
    return crisp


def _smooth(signal: np.ndarray, fps: float) -> np.ndarray:
    """One-euro smoothing for a single [0, 1] blink/wide signal, a convex
    blend of already-clipped values, so the result stays in [0, 1] with no
    extra clipping needed."""
    return one_euro_filter_sequence(signal[:, None], fps, EYELID_ONE_EURO_MIN_CUTOFF_HZ, EYELID_ONE_EURO_BETA)[:, 0]


def eyelid_arkit_channels(landmarks: np.ndarray, valid: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """`landmarks`: (F, 478, 2 or 3) MediaPipe full-frame landmarks. `valid`:
    (F,) bool, MediaPipe detection validity, only valid frames inform the
    per-clip percentile calibration. `fps`: for one-euro smoothing. Returns
    the 4 Group-E channels, each (F,) in [0, 1]."""
    right_ratio = _eye_openness_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
    left_ratio = _eye_openness_ratio(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER)
    right_ratio = fill_invalid(right_ratio[:, None], valid)[:, 0]
    left_ratio = fill_invalid(left_ratio[:, None], valid)[:, 0]

    right_blink_raw, right_wide = _calibrated_blink_and_wide(right_ratio, valid)
    left_blink_raw, left_wide = _calibrated_blink_and_wide(left_ratio, valid)

    return {
        "EyeBlinkRight": _crisp_blink(_smooth(right_blink_raw, fps)).astype(np.float32),
        "EyeBlinkLeft": _crisp_blink(_smooth(left_blink_raw, fps)).astype(np.float32),
        "EyeWideRight": _smooth(right_wide, fps).astype(np.float32),
        "EyeWideLeft": _smooth(left_wide, fps).astype(np.float32),
    }
