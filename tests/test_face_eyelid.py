"""Unit tests for `face_eyelid.py`. Pure numpy/scipy, synthetic landmark
sequences, no model assets needed, always runs.
"""
from __future__ import annotations

import numpy as np

from pipeline.algorithms.face.face_eyelid import (
    _bridge_short_core_gaps, _calibrated_blink_and_wide, _eye_openness_ratio, eyelid_arkit_channels,
)
from pipeline.algorithms.face.face_gaze import (
    LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER, RIGHT_EYE_TOP,
)

NUM_LANDMARKS = 478


def _make_clip(right_vertical: np.ndarray) -> np.ndarray:
    """A clip where only the right eye's vertical lid distance varies (per
    frame); everything else stays at a fixed, plausible open-eye geometry."""
    n = len(right_vertical)
    lm = np.zeros((n, NUM_LANDMARKS, 3), dtype=np.float32)
    lm[:, RIGHT_EYE_INNER] = [-1, 0, 0]
    lm[:, RIGHT_EYE_OUTER] = [1, 0, 0]
    lm[:, RIGHT_EYE_TOP, 1] = -right_vertical / 2
    lm[:, RIGHT_EYE_BOTTOM, 1] = right_vertical / 2
    # Left eye: constant, fully open, never blinks in this synthetic clip.
    lm[:, LEFT_EYE_INNER] = [1, 10, 0]
    lm[:, LEFT_EYE_OUTER] = [-1, 10, 0]
    lm[:, LEFT_EYE_TOP] = [0, 9.4, 0]
    lm[:, LEFT_EYE_BOTTOM] = [0, 10.6, 0]
    return lm


def test_eye_that_never_closes_reports_no_blink():
    rng = np.random.default_rng(0)
    right_vertical = 0.6 + rng.normal(0, 0.02, size=60)  # stays open, small noise
    lm = _make_clip(right_vertical)
    valid = np.ones(60, dtype=bool)

    out = eyelid_arkit_channels(lm, valid, fps=30.0)
    assert out["EyeBlinkRight"].max() < 0.3


def test_a_real_closure_dip_is_detected_as_a_crisp_blink():
    right_vertical = np.full(60, 0.6)
    right_vertical[28:32] = 0.02  # a brief, deep closure
    lm = _make_clip(right_vertical)
    valid = np.ones(60, dtype=bool)

    out = eyelid_arkit_channels(lm, valid, fps=30.0)
    assert out["EyeBlinkRight"][30] == 1.0  # crisp: forced to full closure at the peak
    assert out["EyeBlinkRight"][0] < 0.3  # baseline stays low


def test_blink_transition_ramps_instead_of_jumping_instantly():
    # Regression guard for a real bug: the original hysteresis
    # forced the WHOLE recognized blink event (including its ramp-in/ramp-out
    # edges) to literal 1.0, so a blink's open->closed transition jumped in a
    # single frame instead of taking the several frames a real eyelid needs.
    # A gradual, realistic closing/opening ramp (unlike the other tests here,
    # which use an instant dip) exercises that ramp shape directly.
    n = 90
    right_vertical = np.full(n, 0.6)
    right_vertical[20:35] = np.linspace(0.6, 0.02, 15)  # closes smoothly
    right_vertical[35:45] = 0.02  # holds deep
    right_vertical[45:60] = np.linspace(0.02, 0.6, 15)  # reopens smoothly
    lm = _make_clip(right_vertical)
    valid = np.ones(n, dtype=bool)

    out = eyelid_arkit_channels(lm, valid, fps=30.0)
    blink = out["EyeBlinkRight"]

    assert blink[40] == 1.0  # the deep hold still reaches a crisp full closure
    assert np.abs(np.diff(blink)).max() < 0.5  # no single-frame all-or-nothing jump
    ramp_region = blink[20:35]
    assert ((ramp_region > 0.05) & (ramp_region < 0.95)).any()  # a real in-between frame exists


def test_bridge_short_core_gaps_fills_interior_dip():
    core = np.array([False, True, True, False, False, True, True, False])
    bridged = _bridge_short_core_gaps(core, max_gap=3)
    assert bridged[3] and bridged[4]  # interior gap, bounded by True both sides, bridged
    assert not bridged[0] and not bridged[7]  # boundary runs untouched


def test_bridge_short_core_gaps_leaves_long_gap_untouched():
    core = np.array([True, False, False, False, False, True])
    bridged = _bridge_short_core_gaps(core, max_gap=3)
    assert not bridged[1:5].any()  # 4-frame gap exceeds max_gap=3, a real reopening, not noise


def test_sustained_closure_survives_a_brief_sub_lock_noise_dip():
    # Regression guard for the real bug found against real mcds_test_7
    # ground truth: the eye is genuinely closed the whole stretch, but the
    # smoothed signal dips a few hundredths below BLINK_LOCK_CLOSURE for a
    # handful of frames in the middle, confirmed there was no real
    # reopening at that point. The old plain per-frame `core` check dropped
    # those frames out of the crisp plateau; the fix bridges a short
    # interior dip like this without touching a real ramp edge.
    n = 60
    right_vertical = np.full(n, 0.6)
    right_vertical[10:50] = 0.015  # sustained deep closure
    right_vertical[28:31] = 0.021  # a brief, small partial-reopening blip mid-closure, noise, not real
    lm = _make_clip(right_vertical)
    valid = np.ones(n, dtype=bool)

    out = eyelid_arkit_channels(lm, valid, fps=60.0)
    assert (out["EyeBlinkRight"][15:45] == 1.0).all()  # stays crisp all through, including the blip


def test_left_eye_unaffected_by_right_eye_blink():
    right_vertical = np.full(30, 0.6)
    right_vertical[14:17] = 0.02
    lm = _make_clip(right_vertical)
    valid = np.ones(30, dtype=bool)

    out = eyelid_arkit_channels(lm, valid, fps=30.0)
    assert out["EyeBlinkLeft"].max() < 0.3


def test_invalid_frames_excluded_from_calibration():
    right_vertical = np.full(30, 0.6)
    lm = _make_clip(right_vertical)
    valid = np.zeros(30, dtype=bool)  # nothing valid, must not crash or divide by zero

    out = eyelid_arkit_channels(lm, valid, fps=30.0)
    assert np.isfinite(out["EyeBlinkRight"]).all()
    assert np.isfinite(out["EyeWideRight"]).all()


def test_output_bounded_in_unit_range():
    rng = np.random.default_rng(1)
    right_vertical = np.clip(0.5 + rng.normal(0, 0.3, size=40), 0.01, 1.5)
    lm = _make_clip(right_vertical)
    valid = np.ones(40, dtype=bool)

    out = eyelid_arkit_channels(lm, valid, fps=30.0)
    for values in out.values():
        assert (values >= 0.0).all() and (values <= 1.0).all()


def test_one_euro_smoothing_suppresses_open_eye_jitter():
    # Regression guard for a real bug found from a rendered preview: a
    # steady-open eye's own small per-frame landmark noise showed up as
    # visible residual jitter because Blink/Wide had no temporal filter at
    # all before this fix. Compares
    # against the pre-smoothing raw calibrated signal directly rather than an
    # absolute threshold, so this stays a genuine before/after regression
    # check rather than a magic number.
    rng = np.random.default_rng(2)
    right_vertical = 0.6 + rng.normal(0, 0.05, size=90)
    lm = _make_clip(right_vertical)
    valid = np.ones(90, dtype=bool)

    ratio = _eye_openness_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
    raw_blink, raw_wide = _calibrated_blink_and_wide(ratio, valid)
    out = eyelid_arkit_channels(lm, valid, fps=30.0)

    raw_jitter = np.abs(np.diff(raw_wide)).max()
    smoothed_jitter = np.abs(np.diff(out["EyeWideRight"])).max()
    assert smoothed_jitter < raw_jitter * 0.75


def test_gap_fill_runs_before_calibration_not_just_after():
    # A dropout run inside the true blink's own location must not corrupt
    # calibration or blow up, the raw openness ratio itself is gap-filled
    # (fill_invalid) before percentile calibration, mirroring face_gaze.py's
    # own pipeline, not just patched over the final channel afterward. The
    # closure is wide enough (8 frames) that removing 2 to dropout still
    # leaves the rest comfortably above the percentile calibration's own
    # minimum-representation floor.
    right_vertical = np.full(60, 0.6)
    right_vertical[26:34] = 0.02
    lm = _make_clip(right_vertical)
    valid = np.ones(60, dtype=bool)
    valid[29:31] = False  # MediaPipe dropout during the closure itself

    out = eyelid_arkit_channels(lm, valid, fps=30.0)
    assert np.isfinite(out["EyeBlinkRight"]).all()
    assert out["EyeBlinkRight"][30] > 0.5  # closure still recovered through the gap
