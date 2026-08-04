"""Stage 5 regression test.

The load-bearing math (the wrist change-of-basis) is checked with a GPU-free
synthetic round-trip: after retargeting, forward-kinematics'ing the merged body
must reproduce HaMeR's wrist global orientation. Also covers the two
occlusion-handling edge cases: a hand never detected anywhere in the clip keeps
GVHMR's own wrist and flat fingers, while a hand detected at least once gets
every frame reconciled (stage 4 has already gap-filled the undetected frames by
this point). The remaining tests run the real stage output forward (needs the
SMPL-X model file + the upstream stages' GPU work, skipped automatically
otherwise -- see conftest.py).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pipeline.adapters.gvhmr.gvhmr_adapter import KEY_BODY_POSE, KEY_GLOBAL_ORIENT, KEY_ROOT_MOTION_UNRELIABLE
from pipeline.adapters.gvhmr.gvhmr_rotation_math import axis_angle_to_matrix
from pipeline.adapters.hamer.hamer_adapter import (
    KEY_LEFT_HAND_POSE,
    KEY_RIGHT_HAND_POSE,
)
from pipeline.algorithms.hand_retarget import (
    LEFT_ELBOW,
    LEFT_MIDDLE1,
    LEFT_WRIST,
    NUM_BODY_JOINTS,
    POSE_AXIS_DIM,
    RIGHT_ELBOW,
    RIGHT_MIDDLE1,
    RIGHT_WRIST,
    _global_joint_rotations,
    body_joint_global_rotations,
    reject_biomechanically_implausible_wrist,
    reject_hand_swung_past_forearm,
    reject_hand_through_forearm,
    reject_wrist_velocity_spikes,
    rest_forearm_and_hand_offsets,
    rest_hand_direction,
    retarget_hands,
    wrist_relative_to_elbow,
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
        torch.randn(n, 45) * 0.2, torch.randn(n, 45) * 0.2,
        valid, valid, valid, valid,  # left/right wrist_valid, left/right finger_valid
    )

    local_aa = torch.cat([global_orient, merged_body_pose], dim=1).reshape(n, NUM_BODY_JOINTS, 3)
    global_rot = _global_joint_rotations(axis_angle_to_matrix(local_aa), SMPLX_BODY_PARENTS)

    for wrist, wrist_global in ((LEFT_WRIST, left_wrist_global), (RIGHT_WRIST, right_wrist_global)):
        reproduced = global_rot[:, wrist]
        target = axis_angle_to_matrix(wrist_global)
        assert torch.allclose(reproduced, target, atol=1e-4)


def test_never_detected_hand_keeps_body_wrist_and_flat_fingers():
    """A hand with zero real detections anywhere in the clip has nothing usable
    to reconcile -- it keeps GVHMR's own wrist and flat (zero) fingers for its
    entire duration."""
    torch.manual_seed(1)
    n = 3
    body_pose = torch.randn(n, 63) * 0.3
    left_valid = torch.ones(n, dtype=torch.bool)
    right_invalid = torch.zeros(n, dtype=torch.bool)

    merged_body_pose, _, right_hand_out = retarget_hands(
        torch.randn(n, 3) * 0.3, body_pose, SMPLX_BODY_PARENTS,
        torch.randn(n, 3) * 0.5, torch.randn(n, 3) * 0.5,
        torch.randn(n, 45) * 0.2, torch.randn(n, 45) * 0.2,
        left_valid, right_invalid, left_valid, right_invalid,
    )

    assert torch.equal(merged_body_pose[:, _wrist_slot(RIGHT_WRIST)], body_pose[:, _wrist_slot(RIGHT_WRIST)])
    assert torch.equal(right_hand_out, torch.zeros(n, 45))


def test_hand_detected_at_least_once_reconciles_every_frame():
    """A hand detected on only some frames still gets every frame reconciled and
    copied, including the ones flagged invalid -- stage 4 has already gap-filled
    (interpolated/frozen) those frames by the time this runs, so `left_valid`
    here only decides whether the hand has any real data at all, not which
    individual frames to trust."""
    torch.manual_seed(1)
    n = 3
    body_pose = torch.randn(n, 63) * 0.3
    left_hand_pose = torch.randn(n, 45) * 0.2
    left_valid = torch.tensor([True, False, True])  # some frames flagged invalid
    right_invalid = torch.zeros(n, dtype=torch.bool)

    merged_body_pose, left_hand_out, _ = retarget_hands(
        torch.randn(n, 3) * 0.3, body_pose, SMPLX_BODY_PARENTS,
        torch.randn(n, 3) * 0.5, torch.randn(n, 3) * 0.5,
        left_hand_pose, torch.randn(n, 45) * 0.2,
        left_valid, right_invalid, left_valid, right_invalid,
    )

    # Every frame reconciled/copied, not just the ones flagged valid.
    assert torch.equal(left_hand_out, left_hand_pose)
    assert not torch.equal(merged_body_pose[:, _wrist_slot(LEFT_WRIST)], body_pose[:, _wrist_slot(LEFT_WRIST)])


def test_wrist_invalid_does_not_block_finger_copy():
    """The whole point of splitting wrist/finger validity: a hand whose wrist is
    never usable but whose fingers are should still get its fingers copied --
    only the wrist falls back to GVHMR's own."""
    torch.manual_seed(4)
    n = 3
    body_pose = torch.randn(n, 63) * 0.3
    left_hand_pose = torch.randn(n, 45) * 0.2
    wrist_invalid = torch.zeros(n, dtype=torch.bool)
    finger_valid = torch.ones(n, dtype=torch.bool)
    right_invalid = torch.zeros(n, dtype=torch.bool)

    merged_body_pose, left_hand_out, _ = retarget_hands(
        torch.randn(n, 3) * 0.3, body_pose, SMPLX_BODY_PARENTS,
        torch.randn(n, 3) * 0.5, torch.randn(n, 3) * 0.5,
        left_hand_pose, torch.randn(n, 45) * 0.2,
        wrist_invalid, right_invalid, finger_valid, right_invalid,
    )

    # Wrist never reconciled -- stays GVHMR's own.
    assert torch.equal(merged_body_pose[:, _wrist_slot(LEFT_WRIST)], body_pose[:, _wrist_slot(LEFT_WRIST)])
    # Fingers copied anyway.
    assert torch.equal(left_hand_out, left_hand_pose)


def test_body_joint_global_rotations_matches_manual_fk():
    """The root joint (parent -1) has no ancestor to compose with, so its
    global rotation must equal `global_orient` verbatim -- a direct sanity
    check independent of the recursive composition the rest of the tree uses."""
    torch.manual_seed(2)
    n = 4
    global_orient = torch.randn(n, 3) * 0.4
    body_pose = torch.randn(n, 63) * 0.3

    global_rot = body_joint_global_rotations(global_orient, body_pose, SMPLX_BODY_PARENTS)

    assert torch.allclose(global_rot[:, 0], axis_angle_to_matrix(global_orient), atol=1e-5)


def test_wrist_relative_to_elbow_round_trips_through_elbow():
    """R_wrist_local = R_elbow^T @ R_wrist_global -- applying the elbow back
    (R_elbow @ R_wrist_local) must reproduce the original wrist_global."""
    torch.manual_seed(3)
    n = 6
    elbow_aa = torch.randn(n, 3) * 0.5
    wrist_global_aa = torch.randn(n, 3) * 0.5
    elbow_global = axis_angle_to_matrix(elbow_aa)

    wrist_local_aa = wrist_relative_to_elbow(elbow_global, wrist_global_aa)
    reproduced = elbow_global @ axis_angle_to_matrix(wrist_local_aa)
    assert torch.allclose(reproduced, axis_angle_to_matrix(wrist_global_aa), atol=1e-4)


GATE_MAX_DEG = 110.0
GATE_RELEASE_DEG = 55.0
GATE_MAX_EXPANSION_FRAMES = 100  # effectively unbounded at these tests' short lengths


def _wrist_aa_deg_x(degs: list[float]) -> torch.Tensor:
    """(len(degs), 3) axis-angle rotations about X, one per listed degree value."""
    return torch.stack([torch.tensor([math.radians(d), 0.0, 0.0]) for d in degs])


def test_reject_biomechanically_implausible_wrist_flags_impossible_bend():
    """A wrist rotated ~170 degrees relative to an identity elbow -- no real
    human wrist bends this far -- must be rejected."""
    n = 3
    elbow_global = torch.eye(3).expand(n, 3, 3)
    wrist_global_aa = _wrist_aa_deg_x([170.0] * n)
    valid = torch.ones(n, dtype=torch.bool)

    gated = reject_biomechanically_implausible_wrist(
        valid, wrist_global_aa, elbow_global, GATE_MAX_DEG, GATE_RELEASE_DEG, window=3, max_expansion_frames=GATE_MAX_EXPANSION_FRAMES
    )
    assert not gated.any()


def test_reject_biomechanically_implausible_wrist_keeps_plausible_bend():
    """A modest ~20 degree wrist bend relative to an identity elbow is well
    within real range and must stay valid."""
    n = 3
    elbow_global = torch.eye(3).expand(n, 3, 3)
    wrist_global_aa = _wrist_aa_deg_x([20.0] * n)
    valid = torch.ones(n, dtype=torch.bool)

    gated = reject_biomechanically_implausible_wrist(
        valid, wrist_global_aa, elbow_global, GATE_MAX_DEG, GATE_RELEASE_DEG, window=3, max_expansion_frames=GATE_MAX_EXPANSION_FRAMES
    )
    assert gated.all()


def test_reject_biomechanically_implausible_wrist_never_resurrects_invalid():
    """A frame already invalid (e.g. never detected) must stay invalid
    regardless of how plausible its (placeholder) rotation happens to be."""
    n = 3
    elbow_global = torch.eye(3).expand(n, 3, 3)
    wrist_global_aa = torch.zeros(n, 3)  # trivially "plausible" -- identity rotation
    valid = torch.zeros(n, dtype=torch.bool)

    gated = reject_biomechanically_implausible_wrist(
        valid, wrist_global_aa, elbow_global, GATE_MAX_DEG, GATE_RELEASE_DEG, window=3, max_expansion_frames=GATE_MAX_EXPANSION_FRAMES
    )
    assert not gated.any()


def test_reject_biomechanically_implausible_wrist_hysteresis_expands_around_seed():
    """A moderate, individually-plausible-looking bend immediately adjacent to
    a confirmed-implausible spike gets swept into the same rejected region --
    the whole point of hysteresis: catching the 'shoulder' frames of a bad
    excursion that a single instantaneous threshold alone would let through."""
    n = 7
    elbow_global = torch.eye(3).expand(n, 3, 3)
    # degrees:  20    20    90(elevated) 170(SEED) 90(elevated)  20    20
    wrist_global_aa = _wrist_aa_deg_x([20.0, 20.0, 90.0, 170.0, 90.0, 20.0, 20.0])
    valid = torch.ones(n, dtype=torch.bool)

    gated = reject_biomechanically_implausible_wrist(
        valid, wrist_global_aa, elbow_global, GATE_MAX_DEG, GATE_RELEASE_DEG, window=1, max_expansion_frames=GATE_MAX_EXPANSION_FRAMES
    )
    assert gated.tolist() == [True, True, False, False, False, True, True]


def test_reject_biomechanically_implausible_wrist_rolling_max_bridges_a_dip():
    """A single-frame dip back to a calm value, sandwiched between two
    seed-exceeding spikes, must not be treated as a genuine return to baseline
    -- matches the real chaotic-stretch pattern this was built for (a raw
    HaMeR estimate bouncing between clearly-implausible and momentarily-calm
    values from frame to frame, not a smooth excursion)."""
    n = 9
    elbow_global = torch.eye(3).expand(n, 3, 3)
    #                    0     1  2(SEED) 3(dip) 4(SEED)  5     6     7     8
    wrist_global_aa = _wrist_aa_deg_x([20.0, 20.0, 170.0, 20.0, 170.0, 20.0, 20.0, 20.0, 20.0])
    valid = torch.ones(n, dtype=torch.bool)

    gated = reject_biomechanically_implausible_wrist(
        valid, wrist_global_aa, elbow_global, GATE_MAX_DEG, GATE_RELEASE_DEG, window=3, max_expansion_frames=GATE_MAX_EXPANSION_FRAMES
    )
    # Frames well outside the window's reach from the spike/dip/spike stretch
    # stay valid; the dip at frame 3 (between the two seeds, well within the
    # window's reach of both) is bridged into the same rejected region, not
    # treated as a recovery.
    assert gated.tolist() == [True, False, False, False, False, False, True, True, True]


def test_reject_biomechanically_implausible_wrist_isolated_spike_does_not_engulf_busy_clip():
    """Regression guard for a real bug: a window wide enough that every frame's
    rolling max touches SOME nearby elevated value (true of any consistently
    busy/energetic clip -- confirmed on a real short sports clip whose typical
    wrist motion routinely reached 55-80 degrees) rejected the ENTIRE clip from
    a single isolated spike, even though every other frame was individually
    unremarkable and nowhere near the frame that actually seeded detection."""
    n = 20
    elbow_global = torch.eye(3).expand(n, 3, 3)
    degs = [36.4, 136.7, 30.0, 27.2, 42.4, 41.2, 34.1, 71.5, 71.8, 34.7,
            25.4, 24.4, 42.6, 46.0, 45.4, 63.1, 50.3, 73.7, 79.0, 54.4]
    wrist_global_aa = _wrist_aa_deg_x(degs)
    valid = torch.ones(n, dtype=torch.bool)

    gated = reject_biomechanically_implausible_wrist(
        valid, wrist_global_aa, elbow_global, GATE_MAX_DEG, GATE_RELEASE_DEG, window=5, max_expansion_frames=GATE_MAX_EXPANSION_FRAMES
    )
    # Only the frames immediately around the one real spike (index 1) are
    # rejected; frames 7-8 and 17-18 (individually elevated but far from the
    # spike, comparable to the clip's own natural busy range) must survive.
    assert not gated.all()
    assert gated[10:14].all()  # a clearly-calm stretch, far from the spike
    assert gated[16:19].all()  # elevated but legitimate, far from the spike


def test_reject_biomechanically_implausible_wrist_caps_expansion_on_a_long_plausible_stretch():
    """Regression guard for a real bug found on a BEHAVE backpack clip: a long
    (60+ frame) stretch smoothly varying in a genuinely plausible 60-80 degree
    range (real sustained motion, e.g. gripping a strap) stayed continuously
    above `release_deg` throughout, so a single seed-exceeding frame at one
    edge caused the entire contiguous elevated run to be rejected -- even
    though most of it was never itself close to `max_deg`. `max_expansion_
    frames` bounds the reject region to a fixed radius around each seed
    instead of the whole elevated run."""
    n = 40
    elbow_global = torch.eye(3).expand(n, 3, 3)
    degs = [70.0] * n
    degs[2] = 170.0  # one seed frame, near the start
    wrist_global_aa = _wrist_aa_deg_x(degs)
    valid = torch.ones(n, dtype=torch.bool)

    gated = reject_biomechanically_implausible_wrist(
        valid, wrist_global_aa, elbow_global, GATE_MAX_DEG, GATE_RELEASE_DEG, window=1, max_expansion_frames=5
    )
    # Rejected only within 5 frames of the seed (index 2) -- not the whole
    # 40-frame elevated run.
    assert not gated[2].item()
    assert not gated[0:8].any()  # within reach of the seed
    assert gated[8:].all()  # far from the seed, stays valid despite being elevated too


VELOCITY_MAX_DEG_PER_SEC = 2400.0
VELOCITY_FPS = 30.0


def test_reject_wrist_velocity_spikes_keeps_smooth_motion_valid():
    """Ordinary continuous motion (a few degrees per frame) must stay valid --
    well under any physically-plausible wrist speed."""
    degs = [0.0, 5.0, 12.0, 18.0, 25.0, 30.0]
    wrist_global_aa = _wrist_aa_deg_x(degs)
    valid = torch.ones(len(degs), dtype=torch.bool)

    gated = reject_wrist_velocity_spikes(valid, wrist_global_aa, VELOCITY_FPS, VELOCITY_MAX_DEG_PER_SEC)
    assert gated.all()


def test_reject_wrist_velocity_spikes_flags_isolated_flip():
    """A single frame that jumps ~165 degrees away from its neighbors and back
    -- physically impossible within one frame at 30fps -- gets flagged, along
    with the neighbor on either side of each huge transition (a spike's
    rotation looks anomalous relative to both its predecessor and successor,
    so both sides of each huge transition are marked, not an arbitrarily
    chosen one)."""
    degs = [0.0, 5.0, 170.0, 10.0, 15.0]  # index 2 is the flip
    wrist_global_aa = _wrist_aa_deg_x(degs)
    valid = torch.ones(len(degs), dtype=torch.bool)

    gated = reject_wrist_velocity_spikes(valid, wrist_global_aa, VELOCITY_FPS, VELOCITY_MAX_DEG_PER_SEC)
    assert gated.tolist() == [True, False, False, False, True]


def test_reject_wrist_velocity_spikes_never_resurrects_invalid():
    degs = [0.0, 5.0, 10.0, 15.0]
    wrist_global_aa = _wrist_aa_deg_x(degs)
    valid = torch.tensor([True, False, True, True])

    gated = reject_wrist_velocity_spikes(valid, wrist_global_aa, VELOCITY_FPS, VELOCITY_MAX_DEG_PER_SEC)
    assert not gated[1].item()


def test_reject_wrist_velocity_spikes_skips_delta_across_gap():
    """A velocity computed across a frame already invalid isn't meaningful --
    a huge jump adjacent to a gap must not itself get treated as a spike and
    propagate a rejection onto an otherwise-fine neighboring frame."""
    degs = [0.0, 5.0, 170.0, 175.0]  # the big jump spans index 2, already invalid
    wrist_global_aa = _wrist_aa_deg_x(degs)
    valid = torch.tensor([True, True, False, True])

    gated = reject_wrist_velocity_spikes(valid, wrist_global_aa, VELOCITY_FPS, VELOCITY_MAX_DEG_PER_SEC)
    assert gated.tolist() == [True, True, False, True]


REST_DIR_X = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_reject_hand_swung_past_forearm_keeps_rest_pose_valid():
    n = 2
    elbow_global = torch.eye(3).expand(n, 3, 3)
    wrist_global_aa = torch.zeros(n, 3)  # identity local rotation -- hand at rest
    valid = torch.ones(n, dtype=torch.bool)

    gated = reject_hand_swung_past_forearm(valid, wrist_global_aa, elbow_global, REST_DIR_X, max_swing_deg=90.0)
    assert gated.all()


def test_reject_hand_swung_past_forearm_flags_a_real_swing():
    """A 90-degree rotation about an axis perpendicular to the rest direction
    swings the hand's own pointing direction by the full 90 degrees."""
    elbow_global = torch.eye(3).expand(1, 3, 3)
    wrist_global_aa = torch.tensor([[0.0, 0.0, math.pi / 2]])  # rotates rest X onto Y
    valid = torch.ones(1, dtype=torch.bool)

    kept = reject_hand_swung_past_forearm(valid, wrist_global_aa, elbow_global, REST_DIR_X, max_swing_deg=91.0)
    rejected = reject_hand_swung_past_forearm(valid, wrist_global_aa, elbow_global, REST_DIR_X, max_swing_deg=89.0)
    assert kept.all()
    assert not rejected.any()


def test_reject_hand_swung_past_forearm_ignores_pure_twist():
    """A rotation entirely ABOUT the rest direction itself (pronation/
    supination-like) leaves the hand's own pointing direction unchanged --
    zero swing, regardless of how large the twist is. This is the whole point
    of measuring swing instead of the blended axis-angle magnitude, which
    would read this as a full 90-degree deviation."""
    elbow_global = torch.eye(3).expand(1, 3, 3)
    wrist_global_aa = torch.tensor([[math.pi / 2, 0.0, 0.0]])  # rotation about the rest direction itself
    valid = torch.ones(1, dtype=torch.bool)

    gated = reject_hand_swung_past_forearm(valid, wrist_global_aa, elbow_global, REST_DIR_X, max_swing_deg=1.0)
    assert gated.all()  # a 1-degree ceiling still passes -- swing is ~0 despite a 90-degree rotation


def test_reject_hand_swung_past_forearm_never_resurrects_invalid():
    elbow_global = torch.eye(3).expand(1, 3, 3)
    wrist_global_aa = torch.zeros(1, 3)
    valid = torch.zeros(1, dtype=torch.bool)

    gated = reject_hand_swung_past_forearm(valid, wrist_global_aa, elbow_global, REST_DIR_X, max_swing_deg=180.0)
    assert not gated.any()


def test_rest_hand_direction_is_unit_vector():
    """Needs the real SMPL-X model file (rest-pose joint offsets)."""
    from pipeline.helpers.smplx_bvh_preview import SMPLX_MODEL_PATH

    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (see README's Setup section)")

    for wrist, middle1 in ((LEFT_WRIST, LEFT_MIDDLE1), (RIGHT_WRIST, RIGHT_MIDDLE1)):
        d = rest_hand_direction(wrist, middle1)
        assert d.shape == (3,)
        assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5)


FOREARM_OFFSET = np.array([1.0, 0.0, 0.0], dtype=np.float32)
HAND_OFFSET = np.array([0.5, 0.0, 0.0], dtype=np.float32)


def test_reject_hand_through_forearm_keeps_rest_pose_valid():
    elbow_global = torch.eye(3).expand(1, 3, 3)
    wrist_global_aa = torch.zeros(1, 3)  # identity -- hand points straight out, same direction as forearm
    valid = torch.ones(1, dtype=torch.bool)

    gated = reject_hand_through_forearm(
        valid, wrist_global_aa, elbow_global, FOREARM_OFFSET, HAND_OFFSET,
        forearm_radius_m=0.1, forearm_interior_max_t=0.95,
    )
    assert gated.all()


def test_reject_hand_through_forearm_flags_a_real_fold_back():
    """A 180-degree rotation about an axis perpendicular to the forearm flips
    the hand back onto the forearm's own segment, directly overlapping it."""
    elbow_global = torch.eye(3).expand(1, 3, 3)
    wrist_global_aa = torch.tensor([[0.0, 0.0, math.pi]])  # flips hand_offset from +X to -X
    valid = torch.ones(1, dtype=torch.bool)

    gated = reject_hand_through_forearm(
        valid, wrist_global_aa, elbow_global, FOREARM_OFFSET, HAND_OFFSET,
        forearm_radius_m=0.1, forearm_interior_max_t=0.95,
    )
    assert not gated.any()


def test_reject_hand_through_forearm_ignores_a_hand_held_out_to_the_side():
    """Low `t` alone isn't enough -- a hand swung sideways far enough to sit
    well off the forearm's own axis, even if its projection lands inside the
    segment's length, isn't actually overlapping it."""
    elbow_global = torch.eye(3).expand(1, 3, 3)
    wrist_global_aa = torch.tensor([[0.0, 0.0, 2 * math.pi / 3]])  # 120 degrees about Z
    valid = torch.ones(1, dtype=torch.bool)

    gated = reject_hand_through_forearm(
        valid, wrist_global_aa, elbow_global, FOREARM_OFFSET, HAND_OFFSET,
        forearm_radius_m=0.1, forearm_interior_max_t=0.95,
    )
    assert gated.all()


def test_reject_hand_through_forearm_never_resurrects_invalid():
    elbow_global = torch.eye(3).expand(1, 3, 3)
    wrist_global_aa = torch.zeros(1, 3)
    valid = torch.zeros(1, dtype=torch.bool)

    gated = reject_hand_through_forearm(
        valid, wrist_global_aa, elbow_global, FOREARM_OFFSET, HAND_OFFSET,
        forearm_radius_m=0.1, forearm_interior_max_t=0.95,
    )
    assert not gated.any()


def test_rest_forearm_and_hand_offsets_are_real_metric_lengths():
    """Needs the real SMPL-X model file (rest-pose joint offsets)."""
    from pipeline.helpers.smplx_bvh_preview import SMPLX_MODEL_PATH

    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (see README's Setup section)")

    for elbow, wrist, middle1 in ((LEFT_ELBOW, LEFT_WRIST, LEFT_MIDDLE1), (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_MIDDLE1)):
        forearm, hand = rest_forearm_and_hand_offsets(elbow, wrist, middle1)
        # A real adult forearm length and hand-to-first-knuckle length, in meters.
        assert 0.15 < np.linalg.norm(forearm) < 0.35
        assert 0.05 < np.linalg.norm(hand) < 0.20


def test_retarget_preview_is_structurally_valid(tmp_path):
    """The optional full-body-plus-hands BVH builds a valid animated skeleton.
    Needs only the SMPL-X model file (for joint offsets), no GPU/checkpoints."""
    from pipeline.helpers.smplx_bvh_preview import SMPLX_MODEL_PATH, render_body_hands_bvh

    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (see README's Setup section)")

    n = 6
    out = tmp_path / "retarget.bvh"
    render_body_hands_bvh(
        np.zeros((n, 3), np.float32), np.zeros((n, 63), np.float32),
        np.zeros((n, 45), np.float32), np.zeros((n, 45), np.float32),
        np.zeros((n, 3), np.float32), fps=30.0, out_path=out,
    )

    text = out.read_text()
    assert text.startswith("HIERARCHY")
    assert f"Frames: {n}" in text
    assert "CHANNELS 6 Xposition Yposition Zposition" in text  # root has a real position channel
    # 22 body + 30 finger joints = 52 => 1 ROOT + 51 JOINT lines.
    assert text.count("JOINT ") == 51
    # Leaves: head, both feet, and 5 fingertips x 2 hands = 13 End Sites.
    assert text.count("End Site") == 13


def test_retarget_preview_writes_real_root_translation(tmp_path):
    """This is the only BVH preview in the pipeline with real root motion (the
    stage 4 hands-only preview stays root-locked on purpose) -- the root's
    position channel must carry GVHMR's own translation, reoriented into BVH
    space the same way the root's rotation is, not a static zero."""
    from pipeline.helpers.bvh_export import CAMERA_TO_BVH_ROOT_ROTATION
    from pipeline.helpers.smplx_bvh_preview import SMPLX_MODEL_PATH, render_body_hands_bvh

    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (see README's Setup section)")

    n = 3
    transl = np.zeros((n, 3), np.float32)
    transl[0] = [1.0, 2.0, 3.0]
    out = tmp_path / "retarget_root_motion.bvh"
    render_body_hands_bvh(
        np.zeros((n, 3), np.float32), np.zeros((n, 63), np.float32),
        np.zeros((n, 45), np.float32), np.zeros((n, 45), np.float32),
        transl, fps=30.0, out_path=out,
    )

    motion_lines = out.read_text().split("MOTION\n")[1].splitlines()
    frame0_values = [float(v) for v in motion_lines[2].split()[:3]]  # [0]=Frames, [1]=Frame Time
    expected = CAMERA_TO_BVH_ROOT_ROTATION @ np.array([1.0, 2.0, 3.0])
    assert np.allclose(frame0_values, expected, atol=1e-5)
    # Frame 1's translation was zero going in -- must stay at the origin, not
    # pick up frame 0's value.
    frame1_values = [float(v) for v in motion_lines[3].split()[:3]]
    assert np.allclose(frame1_values, [0.0, 0.0, 0.0], atol=1e-5)


def test_retargeted_motion_shapes_and_no_nan(stage_5_result):
    merged = torch.load(stage_5_result["retarget_motion"], weights_only=False)
    assert merged[KEY_GLOBAL_ORIENT].shape == (TEST_VIDEO_FRAME_COUNT, 3)
    assert merged[KEY_BODY_POSE].shape == (TEST_VIDEO_FRAME_COUNT, 63)
    for hand_key in (KEY_LEFT_HAND_POSE, KEY_RIGHT_HAND_POSE):
        assert merged[hand_key].shape == (TEST_VIDEO_FRAME_COUNT, 45)
    assert merged[KEY_ROOT_MOTION_UNRELIABLE].shape == (TEST_VIDEO_FRAME_COUNT,)
    assert merged[KEY_ROOT_MOTION_UNRELIABLE].dtype == torch.bool
    for key, value in merged.items():
        if key == KEY_ROOT_MOTION_UNRELIABLE:
            continue  # bool mask, not a motion field -- isnan doesn't apply to bool tensors
        assert not torch.isnan(value).any()


def test_retarget_motion_npz_matches_the_pt_file(stage_5_result):
    """A plain-numpy companion artifact for the one downstream consumer
    (export) that runs in an environment with no torch at all -- must
    carry the exact same keys and values as the .pt, just without torch."""
    merged = torch.load(stage_5_result["retarget_motion"], weights_only=False)
    with np.load(stage_5_result["retarget_motion_npz"]) as npz:
        assert set(npz.keys()) == set(merged.keys())
        for key, value in merged.items():
            assert np.allclose(npz[key], value.numpy())


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

    # Fingers are copied verbatim from stage 4's (already gap-filled) output --
    # every frame now, not just the ones flagged valid, since a hand detected at
    # least once anywhere in the clip has every frame reconciled.
    if hands["left_valid"].any():
        assert np.allclose(merged[KEY_LEFT_HAND_POSE].numpy(), hands[KEY_LEFT_HAND_POSE], atol=1e-5)
