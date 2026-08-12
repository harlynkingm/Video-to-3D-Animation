"""Unit tests for `face_gaze.py`. Pure numpy, synthetic landmark arrays,
no model assets needed, always runs.

In/Out sign matches `direct_arkit_gaze_channels`' own real-data-verified
convention (see that function's docstring): iris shifted toward this eye's
own OUTER corner -> `EyeLookIn` fires, toward the INNER corner ->
`EyeLookOut` fires, the opposite of the naive "toward the nose = In"
anatomical assumption, which a real ground-truth capture contradicted
directly (strong, consistent correlation, both eyes) after that assumption
shipped and was measured against real capture data.

Raw-ratio tests (`eye_gaze_ratios`) exercise a single synthetic frame,
sign/scale-invariance are frame-local properties. The full-pipeline tests
(`direct_arkit_gaze_channels`/`eye_euler_degrees`, which additionally
gap-fill for MediaPipe detection dropout, percentile-calibrate, and
one-euro smooth) need a real multi-frame sequence to exercise meaningfully.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.algorithms.face.face_gaze import (
    LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_IRIS_CENTER, MIN_CALIBRATION_SCALE,
    RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_IRIS_CENTER, _calibrate_signed_ratio,
    direct_arkit_gaze_channels, eye_euler_degrees, eye_gaze_ratios,
)

NUM_LANDMARKS = 478


def _make_landmarks(right_iris_offset=(0.0, 0.0), left_iris_offset=(0.0, 0.0), n: int = 1) -> np.ndarray:
    """`n` identical synthetic frames: right eye corners at x=(-1, 1), y=0
    (inner at x=-1, outer at x=1, arbitrary but consistent), top/bottom at
    y=(-1, 1); left eye mirrored in x. Iris centers at the eye midpoint plus
    an offset."""
    lm = np.zeros((n, NUM_LANDMARKS, 3), dtype=np.float32)
    lm[:, RIGHT_EYE_INNER] = [-1, 0, 0]
    lm[:, RIGHT_EYE_OUTER] = [1, 0, 0]
    lm[:, RIGHT_EYE_TOP] = [0, -1, 0]
    lm[:, RIGHT_EYE_BOTTOM] = [0, 1, 0]
    lm[:, RIGHT_IRIS_CENTER] = [right_iris_offset[0], right_iris_offset[1], 0]
    lm[:, LEFT_EYE_INNER] = [1, 10, 0]
    lm[:, LEFT_EYE_OUTER] = [-1, 10, 0]
    lm[:, LEFT_EYE_TOP] = [0, 9, 0]
    lm[:, LEFT_EYE_BOTTOM] = [0, 11, 0]
    lm[:, LEFT_IRIS_CENTER] = [left_iris_offset[0], 10 + left_iris_offset[1], 0]
    return lm


# --- Raw ratio: sign convention, scale invariance --------------------------

def test_centered_iris_gives_zero_raw_ratio():
    lm = _make_landmarks()
    ratios = eye_gaze_ratios(lm)
    for name, values in ratios.items():
        assert values[0] == 0.0, f"{name} nonzero for centered iris"


def test_right_eye_iris_toward_outer_corner_is_positive_h():
    lm = _make_landmarks(right_iris_offset=(0.5, 0.0))
    ratios = eye_gaze_ratios(lm)
    assert ratios["right_h"][0] > 0.0


def test_iris_toward_bottom_lid_is_positive_v():
    lm = _make_landmarks(right_iris_offset=(0.0, 0.5))
    ratios = eye_gaze_ratios(lm)
    assert ratios["right_v"][0] > 0.0


def test_horizontal_ratio_is_scale_invariant():
    small = _make_landmarks(right_iris_offset=(0.5, 0.0))
    lm_big = small.copy()
    lm_big[0, [RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER]] *= 3.0
    assert eye_gaze_ratios(small)["right_h"][0] == pytest.approx(eye_gaze_ratios(lm_big)["right_h"][0], abs=1e-4)


def test_vertical_ratio_is_scale_invariant():
    # Regression guard for the real bug: the old top/bottom-span-normalized
    # vertical ratio was NOT scale-invariant near closure, width
    # normalization must hold regardless of how open the eye is.
    small = _make_landmarks(right_iris_offset=(0.0, 0.3))
    lm_big = small.copy()
    lm_big[0, [RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER]] *= 3.0
    assert eye_gaze_ratios(small)["right_v"][0] == pytest.approx(eye_gaze_ratios(lm_big)["right_v"][0], abs=1e-4)


def test_vertical_ratio_stays_finite_as_eyelid_approaches_closed():
    # The old design (normalize by top-bottom lid span) blew up here,
    # values reaching -5 to +7 on real footage instead of staying near
    # [-1, 1]. Width normalization must not reproduce that.
    lm = _make_landmarks(right_iris_offset=(0.0, 0.3))
    lm[0, RIGHT_EYE_TOP] = [0, -0.001, 0]  # lid nearly shut
    lm[0, RIGHT_EYE_BOTTOM] = [0, 0.001, 0]
    ratios = eye_gaze_ratios(lm)
    assert np.isfinite(ratios["right_v"][0])
    assert abs(ratios["right_v"][0]) < 2.0


# --- Percentile calibration --------------------------------------------------

def test_calibrate_signed_ratio_stretches_to_unit_range():
    raw = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])  # real range only reaches +-0.2
    valid = np.ones(5, dtype=bool)
    calibrated = _calibrate_signed_ratio(raw, valid)
    assert calibrated.max() == pytest.approx(1.0, abs=0.05)
    assert calibrated.min() == pytest.approx(-1.0, abs=0.05)


def test_calibrate_signed_ratio_degenerate_range_returns_zero_not_amplified_noise():
    raw = np.full(10, 1e-8)  # no real variation at all
    valid = np.ones(10, dtype=bool)
    calibrated = _calibrate_signed_ratio(raw, valid)
    assert np.allclose(calibrated, 0.0, atol=MIN_CALIBRATION_SCALE)


def test_calibrate_signed_ratio_zero_valid_frames_returns_zero():
    raw = np.array([0.5, -0.5, 0.3])
    valid = np.zeros(3, dtype=bool)
    assert np.array_equal(_calibrate_signed_ratio(raw, valid), np.zeros(3))


# --- Full pipeline: sign and shape ------------------------------------------

def test_right_eye_iris_toward_outer_corner_is_in_not_out():
    n = 30
    lm = _make_landmarks(right_iris_offset=(0.5, 0.0), n=n)
    valid = np.ones(n, dtype=bool)
    out = direct_arkit_gaze_channels(lm, valid, fps=30.0)
    assert out["EyeLookInRight"][-1] > 0.0
    assert out["EyeLookOutRight"][-1] == 0.0


def test_both_eyes_shifted_toward_their_own_inner_corner_both_register_out():
    # Right toward -x, left toward +x, opposite raw x-directions, but both
    # should register as "Out" under the real-data-verified convention. The
    # sign-mirror-between-eyes check the design doc warns about.
    n = 30
    lm = _make_landmarks(right_iris_offset=(-0.5, 0.0), left_iris_offset=(0.5, 0.0), n=n)
    valid = np.ones(n, dtype=bool)
    out = direct_arkit_gaze_channels(lm, valid, fps=30.0)
    assert out["EyeLookOutRight"][-1] > 0.0 and out["EyeLookInRight"][-1] == 0.0
    assert out["EyeLookOutLeft"][-1] > 0.0 and out["EyeLookInLeft"][-1] == 0.0


def test_gaze_stays_stable_through_a_near_closed_eye():
    # Regression guard for the real bug this module's own docstring covers
    # in detail: the old top/bottom-lid-span vertical normalization blew up
    # numerically as the eyelid narrowed, producing visible jitter on real
    # footage. Width normalization (this module's own fix) must stay stable
    # through the same shape of event, a steady horizontal gaze while the
    # eye narrows toward closed and back open.
    n = 40
    right_offset = np.zeros((n, 2))
    right_offset[:, 0] = 0.5  # steady rightward gaze throughout
    # Eyelid narrows toward closed over frames 15-24, reopens after.
    lid_half_height = np.ones(n)
    lid_half_height[15:25] = 0.02

    lm = np.zeros((n, NUM_LANDMARKS, 3), dtype=np.float32)
    lm[:, RIGHT_EYE_INNER] = [-1, 0, 0]
    lm[:, RIGHT_EYE_OUTER] = [1, 0, 0]
    lm[:, RIGHT_EYE_TOP, 1] = -lid_half_height
    lm[:, RIGHT_EYE_BOTTOM, 1] = lid_half_height
    lm[:, RIGHT_IRIS_CENTER, 0] = right_offset[:, 0]
    lm[:, LEFT_EYE_INNER] = [1, 10, 0]
    lm[:, LEFT_EYE_OUTER] = [-1, 10, 0]
    lm[:, LEFT_EYE_TOP] = [0, 9, 0]
    lm[:, LEFT_EYE_BOTTOM] = [0, 11, 0]
    lm[:, LEFT_IRIS_CENTER] = [0, 10, 0]

    valid = np.ones(n, dtype=bool)
    out = direct_arkit_gaze_channels(lm, valid, fps=30.0)

    # The steady rightward gaze should stay roughly steady throughout,
    # including while the eyelid is nearly closed, no blow-up, no jitter.
    assert np.abs(np.diff(out["EyeLookInRight"])).max() < 0.3


def test_eye_euler_degrees_shape_and_roll_zero():
    n = 10
    lm = _make_landmarks(right_iris_offset=(0.3, 0.0), left_iris_offset=(-0.2, 0.1), n=n)
    valid = np.ones(n, dtype=bool)
    out = eye_euler_degrees(lm, valid, fps=30.0)
    assert out.shape == (n, 6)
    assert np.all(out[:, 2] == 0.0)  # LeftEyeRoll
    assert np.all(out[:, 5] == 0.0)  # RightEyeRoll


def test_right_eye_yaw_is_negated_relative_to_left():
    # Same-magnitude outward shift for both eyes, RightEyeYaw and
    # LeftEyeYaw should come out with opposite sign (real-data-verified,
    # see eye_euler_degrees' own docstring for why this asymmetry is
    # expected: Yaw is world-frame-consistent, the underlying ratio isn't).
    n = 30
    lm = _make_landmarks(right_iris_offset=(0.5, 0.0), left_iris_offset=(-0.5, 0.0), n=n)
    valid = np.ones(n, dtype=bool)
    out = eye_euler_degrees(lm, valid, fps=30.0)
    left_yaw, right_yaw = out[-1, 0], out[-1, 3]
    assert left_yaw > 0.0
    assert right_yaw < 0.0
