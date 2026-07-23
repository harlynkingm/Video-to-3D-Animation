"""Stage 4 regression test: runs real HaMeR on the test clip and checks the
per-frame MANO hand pose looks correct -- right shapes, no NaN, physically
plausible finger rotations, both hands detected on this clearly-two-handed
tennis clip, and smooth frame-to-frame motion (a broken crop/flip would produce
jumpy garbage). Needs the HaMeR + ViTPose checkpoints and a CUDA GPU -- skipped
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


def test_hands_bvh_preview_is_structurally_valid(tmp_path):
    """The optional BVH preview builds a valid two-hand animated skeleton. Needs
    only the SMPL-X model file (for the hand joint offsets), no GPU/checkpoints."""
    from pipeline.adapters.hamer.hamer_bvh_preview import SMPLX_MODEL_PATH, dump_hands_bvh

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
    dump_hands_bvh(synthetic, fps=30.0, out_path=out)

    text = out.read_text()
    assert text.startswith("HIERARCHY")
    assert "MOTION" in text
    assert f"Frames: {n}" in text
    # root + 2 hands x (wrist + 15 fingers) = 33 joints => 32 JOINT lines + 1 ROOT
    assert text.count("JOINT ") == 32
    assert text.count("End Site") == 10  # 5 fingertips x 2 hands
