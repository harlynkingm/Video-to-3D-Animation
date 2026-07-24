"""Stage 5 regression test.

The load-bearing math (the wrist change-of-basis) is checked with a GPU-free
synthetic round-trip: after retargeting, forward-kinematics'ing the merged body
must reproduce HaMeR's wrist global orientation on valid frames, and leave
GVHMR's own wrist untouched on invalid ones. The remaining tests run the real
stage output forward (needs the SMPL-X model file + the upstream stages' GPU
work, skipped automatically otherwise -- see conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.adapters.gvhmr.gvhmr_adapter import KEY_BODY_POSE, KEY_GLOBAL_ORIENT
from pipeline.adapters.gvhmr.gvhmr_rotation_math import axis_angle_to_matrix
from pipeline.adapters.hamer.hamer_adapter import (
    KEY_LEFT_HAND_POSE,
    KEY_RIGHT_HAND_POSE,
)
from pipeline.algorithms.hand_retarget import (
    LEFT_WRIST,
    NUM_BODY_JOINTS,
    POSE_AXIS_DIM,
    RIGHT_WRIST,
    _global_joint_rotations,
    retarget_hands,
)
from conftest import TEST_VIDEO_FRAME_COUNT

# SMPL-X body kinematic tree (first 22 joints): each entry is the joint's parent.
# Hardcoded so the pure-math tests need no model file -- the round-trip property
# holds for any consistent tree, and this is the real one for realistic indices.
SMPLX_BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def _wrist_slot(wrist_joint: int) -> slice:
    start = (wrist_joint - 1) * POSE_AXIS_DIM
    return slice(start, start + POSE_AXIS_DIM)


def test_wrist_reconciliation_round_trip():
    torch.manual_seed(0)
    n = 5
    global_orient = torch.randn(n, 3) * 0.3
    body_pose = torch.randn(n, 63) * 0.3
    left_wrist_global = torch.randn(n, 3) * 0.5
    right_wrist_global = torch.randn(n, 3) * 0.5
    valid = torch.ones(n, dtype=torch.bool)

    merged_body_pose, _, _ = retarget_hands(
        global_orient, body_pose, SMPLX_BODY_PARENTS,
        left_wrist_global, right_wrist_global,
        torch.randn(n, 45) * 0.2, torch.randn(n, 45) * 0.2, valid, valid,
    )

    local_aa = torch.cat([global_orient, merged_body_pose], dim=1).reshape(n, NUM_BODY_JOINTS, 3)
    global_rot = _global_joint_rotations(axis_angle_to_matrix(local_aa), SMPLX_BODY_PARENTS)

    for wrist, wrist_global in ((LEFT_WRIST, left_wrist_global), (RIGHT_WRIST, right_wrist_global)):
        reproduced = global_rot[:, wrist]
        target = axis_angle_to_matrix(wrist_global)
        assert torch.allclose(reproduced, target, atol=1e-4)


def test_invalid_hand_keeps_body_wrist_and_flattens_fingers():
    torch.manual_seed(1)
    n = 3
    body_pose = torch.randn(n, 63) * 0.3
    left_hand_pose = torch.randn(n, 45) * 0.2
    left_valid = torch.tensor([True, False, True])
    right_valid = torch.zeros(n, dtype=torch.bool)

    merged_body_pose, left_hand_out, right_hand_out = retarget_hands(
        torch.randn(n, 3) * 0.3, body_pose, SMPLX_BODY_PARENTS,
        torch.randn(n, 3) * 0.5, torch.randn(n, 3) * 0.5,
        left_hand_pose, torch.randn(n, 45) * 0.2, left_valid, right_valid,
    )

    # Left hand: frame 1 invalid -> body wrist untouched, fingers zeroed.
    assert torch.equal(merged_body_pose[1, _wrist_slot(LEFT_WRIST)], body_pose[1, _wrist_slot(LEFT_WRIST)])
    assert torch.equal(left_hand_out[1], torch.zeros(45))
    assert torch.equal(left_hand_out[0], left_hand_pose[0])  # frame 0 valid -> copied
    # Right hand never valid -> right wrist slots all unchanged, all fingers zero.
    assert torch.equal(merged_body_pose[:, _wrist_slot(RIGHT_WRIST)], body_pose[:, _wrist_slot(RIGHT_WRIST)])
    assert torch.equal(right_hand_out, torch.zeros(n, 45))


def test_retarget_preview_is_structurally_valid(tmp_path):
    """The optional full-body-plus-hands BVH builds a valid animated skeleton.
    Needs only the SMPL-X model file (for joint offsets), no GPU/checkpoints."""
    from pipeline.helpers.smplx_bvh_preview import SMPLX_MODEL_PATH, dump_body_hands_bvh

    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (see README's Setup section)")

    n = 6
    out = tmp_path / "retarget.bvh"
    dump_body_hands_bvh(
        np.zeros((n, 3), np.float32), np.zeros((n, 63), np.float32),
        np.zeros((n, 45), np.float32), np.zeros((n, 45), np.float32), fps=30.0, out_path=out,
    )

    text = out.read_text()
    assert text.startswith("HIERARCHY")
    assert f"Frames: {n}" in text
    # 22 body + 30 finger joints = 52 => 1 ROOT + 51 JOINT lines.
    assert text.count("JOINT ") == 51
    # Leaves: head, both feet, and 5 fingertips x 2 hands = 13 End Sites.
    assert text.count("End Site") == 13


def test_retargeted_motion_shapes_and_no_nan(stage_5_result):
    merged = torch.load(stage_5_result["retarget_motion"], weights_only=False)
    assert merged[KEY_GLOBAL_ORIENT].shape == (TEST_VIDEO_FRAME_COUNT, 3)
    assert merged[KEY_BODY_POSE].shape == (TEST_VIDEO_FRAME_COUNT, 63)
    for hand_key in (KEY_LEFT_HAND_POSE, KEY_RIGHT_HAND_POSE):
        assert merged[hand_key].shape == (TEST_VIDEO_FRAME_COUNT, 45)
    for value in merged.values():
        assert not torch.isnan(value).any()


def test_retarget_changes_wrists_and_copies_fingers(stage_5_result, stage_2_result, stage_4_result):
    merged = torch.load(stage_5_result["retarget_motion"], weights_only=False)
    body_motion = torch.load(stage_2_result["human_motion"], weights_only=False)
    from pipeline.adapters.gvhmr.gvhmr_adapter import KEY_PRED_SMPL_PARAMS_INCAM

    original_body_pose = body_motion[KEY_PRED_SMPL_PARAMS_INCAM][KEY_BODY_POSE]
    hands = np.load(stage_4_result["hand_pose"])

    # On this clearly-two-handed clip both hands are detected on most frames, so
    # the reconciled wrists must differ from GVHMR's body-only estimate somewhere.
    left_wrist_changed = not torch.allclose(
        merged[KEY_BODY_POSE][:, _wrist_slot(LEFT_WRIST)],
        torch.as_tensor(original_body_pose)[:, _wrist_slot(LEFT_WRIST)].float(),
    )
    assert left_wrist_changed

    # Fingers on valid frames are copied verbatim from HaMeR.
    left_valid = hands["left_valid"]
    if left_valid.any():
        got = merged[KEY_LEFT_HAND_POSE].numpy()[left_valid]
        assert np.allclose(got, hands[KEY_LEFT_HAND_POSE][left_valid], atol=1e-5)
