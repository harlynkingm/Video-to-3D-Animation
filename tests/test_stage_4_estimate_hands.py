"""Stage 4 regression test: runs real HaMeR on the test clip and checks the
per-frame MANO hand pose looks correct, right shapes, no NaN, physically
plausible finger rotations, both hands detected on this clearly-two-handed
tennis clip, and smooth frame-to-frame motion (a broken crop/flip would produce
jumpy garbage). Needs the HaMeR + ViTPose checkpoints and a CUDA GPU, skipped
automatically otherwise (see conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.adapters.hamer.hamer_adapter import (
    KEY_LEFT_GLOBAL_ORIENT,
    KEY_LEFT_HAND_POSE,
    KEY_LEFT_VALID,
    KEY_RIGHT_GLOBAL_ORIENT,
    KEY_RIGHT_HAND_POSE,
    KEY_RIGHT_VALID,
    MIN_ROLLING_WRIST_CONFIDENCE,
    _reject_low_confidence_stretches,
)
from conftest import TEST_VIDEO_FRAME_COUNT

MAX_PLAUSIBLE_FINGER_ROTATION_RAD = 3.15  # any single axis-angle rotation maxes out at pi
MAX_PLAUSIBLE_FRAME_DELTA = 2.0  # generous bound on frame-to-frame full-hand pose change


def _load(stage_4_result):
    return np.load(stage_4_result["hand_pose"])


def test_hand_pose_has_correct_shapes_and_no_nan(stage_4_result):
    data = _load(stage_4_result)
    for pose_key in (KEY_LEFT_HAND_POSE, KEY_RIGHT_HAND_POSE):
        assert data[pose_key].shape == (TEST_VIDEO_FRAME_COUNT, 45)
        assert not np.isnan(data[pose_key]).any()
    assert data[KEY_RIGHT_GLOBAL_ORIENT].shape == (TEST_VIDEO_FRAME_COUNT, 3)
    assert data[KEY_RIGHT_VALID].dtype == bool


def test_both_hands_detected_on_the_tennis_clip(stage_4_result):
    data = _load(stage_4_result)
    # The player has both hands clearly visible throughout this clip; allow a
    # little slack but expect most frames to detect each hand.
    assert data[KEY_RIGHT_VALID].mean() > 0.5
    assert data[KEY_LEFT_VALID].mean() > 0.5


def test_finger_rotations_are_physically_plausible(stage_4_result):
    data = _load(stage_4_result)
    pose = data[KEY_RIGHT_HAND_POSE].reshape(TEST_VIDEO_FRAME_COUNT, 15, 3)
    assert np.linalg.norm(pose, axis=-1).max() < MAX_PLAUSIBLE_FINGER_ROTATION_RAD


def test_hand_motion_is_temporally_smooth(stage_4_result):
    data = _load(stage_4_result)
    pose = data[KEY_RIGHT_HAND_POSE]
    frame_deltas = np.linalg.norm(np.diff(pose, axis=0), axis=1)
    assert frame_deltas.max() < MAX_PLAUSIBLE_FRAME_DELTA


def test_confidence_gate_rejects_a_low_confidence_stretch():
    """A body-occluded wrist (real HaMeR still returns *a* pose, confidently
    wrong) is caught by dips in wrist keypoint confidence, not by the pose
    itself, see hamer_adapter's module docstring for why a kinematic check
    was tried and rejected in favor of this."""
    n = 30
    valid = np.ones(n, bool)
    conf = np.full(n, 0.9)
    conf[12:20] = 0.4  # a stretch of low confidence, as if occluded

    gated = _reject_low_confidence_stretches(valid, conf)

    assert not gated[12:20].any()
    # Well outside the rolling window's reach from the low stretch (half-window
    # of 3 frames either side of it).
    assert gated[:9].all()
    assert gated[23:].all()


def test_confidence_gate_catches_a_blip_inside_an_occluded_stretch():
    """A single frame's confidence can bounce back above threshold in the
    middle of an otherwise-occluded stretch (ViTPose momentarily picking up a
    partial cue), the rolling MIN, not the frame's own raw confidence, is
    what still rejects it, since its neighbors are still low."""
    n = 30
    valid = np.ones(n, bool)
    conf = np.full(n, 0.9)
    conf[12:20] = 0.4
    conf[16] = 0.9  # one-frame blip back up to "confident", still mid-occlusion

    gated = _reject_low_confidence_stretches(valid, conf)

    assert not gated[16]  # still rejected, neighbors are still low
    assert not gated[12:20].any()


def test_confidence_gate_never_resurrects_an_already_invalid_frame():
    """A frame already marked invalid upstream (no box at all) must stay
    invalid regardless of confidence, the gate can only narrow validity, not
    widen it."""
    n = 10
    valid = np.zeros(n, bool)
    conf = np.full(n, 0.9)  # high confidence everywhere, doesn't matter
    gated = _reject_low_confidence_stretches(valid, conf)
    assert not gated.any()


def test_confidence_gate_passes_through_uniformly_confident_sequence():
    n = 20
    valid = np.ones(n, bool)
    conf = np.full(n, MIN_ROLLING_WRIST_CONFIDENCE + 0.1)
    gated = _reject_low_confidence_stretches(valid, conf)
    assert gated.all()


def test_hands_bvh_preview_is_structurally_valid(tmp_path):
    """The optional BVH preview builds a valid two-hand animated skeleton. Needs
    only the SMPL-X model file (for the hand joint offsets), no GPU/checkpoints."""
    from pipeline.adapters.hamer.hamer_bvh_preview import render_hands_bvh
    from pipeline.helpers.smplx_bvh_preview import SMPLX_MODEL_PATH

    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (see README's Setup section)")

    n = 8
    synthetic = {
        KEY_LEFT_HAND_POSE: np.zeros((n, 45), np.float32),
        KEY_RIGHT_HAND_POSE: np.zeros((n, 45), np.float32),
        KEY_LEFT_GLOBAL_ORIENT: np.zeros((n, 3), np.float32),
        KEY_RIGHT_GLOBAL_ORIENT: np.zeros((n, 3), np.float32),
        KEY_LEFT_VALID: np.ones(n, bool),
        KEY_RIGHT_VALID: np.ones(n, bool),
    }
    out = tmp_path / "hands.bvh"
    render_hands_bvh(synthetic, fps=30.0, out_path=out)

    text = out.read_text()
    assert text.startswith("HIERARCHY")
    assert "MOTION" in text
    assert f"Frames: {n}" in text
    # root + 2 hands x (wrist + 15 fingers) = 33 joints => 32 JOINT lines + 1 ROOT
    assert text.count("JOINT ") == 32
    assert text.count("End Site") == 10  # 5 fingertips x 2 hands
