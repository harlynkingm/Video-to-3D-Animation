"""Unit tests for `hoi_object_pose`: pure-math/synthetic-data tests, no
GPU/checkpoints/DA3 needed, always run. `object_position_fn` is supplied as
a plain Python callable per test, standing in for the real depth-based
back-projection stage 8 itself wires up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from pipeline.algorithms.contact_detection import REGION_JOINTS, attachment_joint_index
from pipeline.algorithms.hoi_object_pose import (
    MAX_SNAP_SEARCH_FRAMES,
    _INCAM_UP,
    _find_snap_measurement,
    _joint_world_transforms,
    _signed_permutation_matrices,
    _snap_axis_to_up,
    compute_object_pose_sequence,
    disambiguate_rotation,
)

SMPLX_MODEL_PATH = Path(__file__).resolve().parents[1] / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"


def test_signed_permutation_matrices_are_all_proper_rotations():
    matrices = _signed_permutation_matrices()
    assert matrices.shape == (24, 3, 3)
    for m in matrices:
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(m) - 1.0) < 1e-9


def test_disambiguate_rotation_recovers_a_permuted_and_signed_reference():
    reference = Rotation.from_euler("xyz", [12.0, -34.0, 56.0], degrees=True).as_matrix()
    # A "fresh" fit that measured the exact same physical axes, just labeled
    # differently (columns permuted + a sign flipped), the kind of ambiguity
    # a real independent PCA fit at a different frame would introduce.
    permuted = reference[:, [1, 2, 0]]
    permuted[:, 0] *= -1

    result = disambiguate_rotation(permuted, reference)

    assert np.allclose(result, reference, atol=1e-9)


def test_disambiguate_rotation_preserves_a_genuine_large_rotation():
    """A real ~180-degree flip (e.g. a mug set down and tipped over) must not
    be suppressed back toward the old reference, every one of the 24
    candidates *is* the freshly-measured rotation, just relabeled, so the
    true new orientation survives regardless of which relabeling wins."""
    reference = np.eye(3)
    flipped = Rotation.from_euler("x", 178.0, degrees=True).as_matrix()

    result = disambiguate_rotation(flipped, reference)

    # The result must be *a* relabeling of `flipped` (same rotation up to
    # signed-axis-permutation), not something pulled back toward `reference`.
    angle_to_flipped = min(
        Rotation.from_matrix(result.T @ (flipped @ p)).magnitude() for p in _signed_permutation_matrices()
    )
    assert angle_to_flipped < 1e-6


def test_snap_axis_to_up_leaves_an_already_upright_axis_unchanged():
    rotation = np.eye(3)  # column 1 is [0, 1, 0]; its negation is exactly _INCAM_UP already.
    result = _snap_axis_to_up(rotation)
    assert np.allclose(result, rotation, atol=1e-9)


def test_snap_axis_to_up_points_the_closest_column_exactly_vertical_same_sign():
    # Column 1 tilted 10 degrees off _INCAM_UP but already the right sign
    # (no negation needed), the closest axis, not exactly aligned yet.
    rotation = Rotation.from_euler("x", 170.0, degrees=True).as_matrix()
    result = _snap_axis_to_up(rotation)
    assert np.allclose(result[:, 1], _INCAM_UP, atol=1e-9)
    assert np.allclose(result @ result.T, np.eye(3), atol=1e-9)  # still a proper rotation


def test_snap_axis_to_up_accepts_a_negative_signed_axis():
    # Column 1 points close to [0, 1, 0], the *opposite* of _INCAM_UP, its
    # own negation is the actual closest candidate, but the resulting column
    # still ends up exactly vertical (up to sign), not left tilted.
    rotation = Rotation.from_euler("x", 10.0, degrees=True).as_matrix()
    result = _snap_axis_to_up(rotation)
    alignment = np.abs(result[:, 1] @ _INCAM_UP)
    assert np.isclose(alignment, 1.0, atol=1e-9)


def test_snap_axis_to_up_picks_whichever_column_is_actually_closest():
    # Rotate so column 0 (not 1) ends up closest to _INCAM_UP (up to sign).
    rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    result = _snap_axis_to_up(rotation)
    alignment = np.abs(result.T @ _INCAM_UP)
    assert np.isclose(alignment.max(), 1.0, atol=1e-9)
    assert np.argmax(alignment) == 0


class _FakeSkeleton:
    """22 joints in a trivial sequential chain (joint i's parent is i-1, the
    root has none), each offset (0, 0, 0.1) from its own parent, simple,
    hand-verifiable rest geometry, deliberately decoupled from the real
    SMPL-X skeleton (betas-independent here) so these tests don't depend on
    its specific numbers."""

    def __init__(self):
        self.parents = [-1] + list(range(21))

    def get_skeleton(self, betas: torch.Tensor) -> torch.Tensor:
        positions = np.zeros((55, 3))
        positions[:, 2] = np.arange(55) * 0.1
        return torch.from_numpy(positions).float()


def _fake_body_motion(n_frames: int) -> dict:
    """Only the root (joint 0) moves, every other joint's own local
    rotation is identity, so joint 20's world transform is fully determined
    by the root's own known motion composed with fixed rest offsets, making
    the expected result something `_joint_world_transforms` itself computes
    (used here as ground truth, not hand-derived) rather than a separate,
    error-prone manual FK re-derivation."""
    global_orient = np.zeros((n_frames, 3))
    global_orient[:, 2] = np.linspace(0.0, 1.0, n_frames)  # root yaws over time
    global_orient[:, 0] = np.linspace(0.0, 0.3, n_frames)
    body_pose = np.zeros((n_frames, 63))
    transl = np.zeros((n_frames, 3))
    transl[:, 0] = np.linspace(0.0, 2.0, n_frames)  # root also translates
    betas = np.zeros((n_frames, 10))
    return {"global_orient": global_orient, "body_pose": body_pose, "betas": betas, "transl": transl}


def test_find_snap_measurement_prefers_start_frame_then_searches_outward():
    """Tried first at `start_frame` itself (least occluded); once that's
    unavailable, alternates outward (+1, -1, +2, -2, ...) rather than
    scanning monotonically in one direction."""
    calls = []

    def object_position_fn(f):
        calls.append(f)
        return (np.array([float(f), 0.0, 0.0]), np.eye(3)) if f == 12 else None

    result = _find_snap_measurement(10, 20, object_position_fn)

    assert result is not None
    frame, center, rotation = result
    assert frame == 12
    assert np.allclose(center, [12.0, 0.0, 0.0])
    # start_frame first, then each delta step tried in ascending order
    # (start-delta before start+delta), confirms the outward-alternating
    # search order, not a plain forward scan.
    assert calls == [10, 9, 11, 8, 12]


def test_find_snap_measurement_returns_none_when_nothing_found_in_range():
    result = _find_snap_measurement(
        10, 10 + MAX_SNAP_SEARCH_FRAMES + 5, lambda f: None,
    )
    assert result is None


def test_attached_segment_propagates_through_real_joint_motion_ignoring_measured_position():
    """Rigid propagation: for every frame of the hold, the object's pose
    should exactly match `joint(frame) @ offset`, where `offset` is fixed at
    the snap frame. The measurement's own center is deliberately wrong in
    every axis here (contact-point anchoring never trusts it for position at
    all, only its rotation, see the module docstring), so a passing test
    here proves that anchoring actually happens, not just that rigid
    propagation of *some* position works.
    """
    n_frames = 10
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_world = _joint_world_transforms(body_motion, skeleton)

    joint_idx = REGION_JOINTS["left_hand"][-1]
    event = {"region": "left_hand", "start_frame": 2, "end_frame": 7}
    snap_frame = event["start_frame"]
    true_rotation = joint_world[snap_frame, joint_idx, :3, :3]
    wrong_center = np.array([123.0, -456.0, 789.0])  # should be ignored entirely

    def object_position_fn(f):
        return (wrong_center, true_rotation) if f == snap_frame else None

    result = compute_object_pose_sequence(
        n_frames=n_frames,
        attachment_events=[event],
        body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=true_rotation,
        object_position_fn=object_position_fn,
        skeleton=skeleton,
    )

    # The reference rotation is snapped upright (see _snap_axis_to_up), so the
    # fixed joint-relative rotation offset isn't identity here, derive it
    # the same way compute_object_pose_sequence itself does, rather than
    # assuming the measurement's own rotation propagates unchanged.
    rotation_offset = true_rotation.T @ _snap_axis_to_up(true_rotation)

    for f in range(event["start_frame"], event["end_frame"] + 1):
        assert np.allclose(result["translation"][f], joint_world[f, joint_idx, :3, 3], atol=1e-6)
        assert np.allclose(result["rotation"][f], joint_world[f, joint_idx, :3, :3] @ rotation_offset, atol=1e-6)
        assert not result["is_low_confidence"][f]


def test_resolved_events_carries_the_joint_anchored_center_and_raw_rotation():
    """A caller that wants the object to stay correctly attached after the
    skeleton is later retargeted onto a different rig needs a *live*
    parent/constraint relationship to the joint, not the baked per-frame
    `translation`/`rotation` above (frozen for this rig's own proportions) --
    `resolved_events` carries what's needed to build that: the event's own
    reference frame/joint, `ref_center` (the attaching joint's own position
    at the snap frame, contact-point anchoring, never the measurement's
    own center, deliberately wrong here to prove it's ignored) and
    `ref_rotation` (the object's *own* measured orientation, snapped
    upright), so a caller can re-derive the offset itself in whatever space
    it needs (e.g. stage 10 does this in Blender's own live coordinate
    space)."""
    n_frames = 10
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_world = _joint_world_transforms(body_motion, skeleton)

    joint_idx = REGION_JOINTS["right_hand"][-1]
    event = {"region": "right_hand", "start_frame": 3, "end_frame": 6}
    snap_frame = event["start_frame"]
    true_center = joint_world[snap_frame, joint_idx, :3, 3]
    true_rotation = joint_world[snap_frame, joint_idx, :3, :3]
    wrong_center = np.array([-1.0, 2.0, -3.0])  # should be ignored entirely

    def object_position_fn(f):
        return (wrong_center, true_rotation) if f == snap_frame else None

    result = compute_object_pose_sequence(
        n_frames=n_frames,
        attachment_events=[event],
        body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=np.eye(3),
        object_position_fn=object_position_fn,
        skeleton=skeleton,
    )

    assert len(result["resolved_events"]) == 1
    resolved = result["resolved_events"][0]
    assert resolved["region"] == "right_hand"
    assert resolved["joint_idx"] == joint_idx
    assert resolved["start_frame"] == event["start_frame"]
    assert resolved["end_frame"] == event["end_frame"]
    assert resolved["snap_frame"] == snap_frame
    assert np.allclose(resolved["ref_center"], true_center, atol=1e-6)
    assert np.allclose(resolved["ref_rotation"], _snap_axis_to_up(true_rotation), atol=1e-6)
    assert not resolved["is_low_confidence"]


def test_head_top_event_attaches_to_the_real_head_joint_not_its_own_mesh_vertex_index():
    """Regression test: "head_top" (see contact_detection.HEAD_TOP_JOINT_INDEX)
    is a synthetic mesh-vertex index, not a real skeletal joint, indexing
    `joint_world` with it directly would be out of bounds (`joint_world` only
    covers SmplxSkeleton's own 22 body joints; the real crash this test
    guards against: `IndexError: index 127 is out of bounds for axis 1 with
    size 22`, hit on a real testPutOnHat rerun). `attachment_joint_index`
    redirects it to the real HEAD_JOINT, the same bone the "head" region
    already attaches to, since that's the only bone actually near it in the
    exported rig."""
    n_frames = 10
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_world = _joint_world_transforms(body_motion, skeleton)

    head_joint_idx = attachment_joint_index("head_top")
    assert head_joint_idx == REGION_JOINTS["head"][-1]  # same real joint "head" itself uses

    event = {"region": "head_top", "start_frame": 2, "end_frame": 5}
    snap_frame = event["start_frame"]
    true_center = joint_world[snap_frame, head_joint_idx, :3, 3]
    true_rotation = joint_world[snap_frame, head_joint_idx, :3, :3]

    def object_position_fn(f):
        return (true_center, true_rotation) if f == snap_frame else None

    result = compute_object_pose_sequence(
        n_frames=n_frames,
        attachment_events=[event],
        body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=true_rotation,
        object_position_fn=object_position_fn,
        skeleton=skeleton,
    )

    for f in range(event["start_frame"], event["end_frame"] + 1):
        assert np.allclose(result["translation"][f], joint_world[f, head_joint_idx, :3, 3], atol=1e-6)
        assert not result["is_low_confidence"][f]
    assert result["resolved_events"][0]["joint_idx"] == head_joint_idx


def test_held_before_the_first_event_matches_its_own_reference_pose():
    """Regression test for a real bug found reviewing a real
    export: an earlier design measured a separate "early resting position"
    for the period before the first event, independent of that event's own
    reference measurement, two independent depth reads of the same
    physically-stationary object disagreed enough to produce a real, jarring
    multi-meter pop right at the first contact frame. Fixed by holding the
    *same* reference pose the first event itself resolves to, for the whole
    period beforehand, so entry is now exactly as pop-free as exit already
    was (see the module docstring)."""
    n_frames = 10
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_world = _joint_world_transforms(body_motion, skeleton)

    joint_idx = REGION_JOINTS["left_hand"][-1]
    event = {"region": "left_hand", "start_frame": 4, "end_frame": 7}
    snap_frame = event["start_frame"]
    ref_center = joint_world[snap_frame, joint_idx, :3, 3]
    ref_rotation = joint_world[snap_frame, joint_idx, :3, :3]

    def object_position_fn(f):
        return (ref_center, ref_rotation) if f == snap_frame else None

    result = compute_object_pose_sequence(
        n_frames=n_frames,
        attachment_events=[event],
        body_motion=body_motion,
        initial_center=np.array([999.0, 999.0, 999.0]), initial_rotation=np.eye(3),
        object_position_fn=object_position_fn,
        skeleton=skeleton,
    )

    for f in range(0, event["start_frame"]):
        assert np.allclose(result["translation"][f], ref_center, atol=1e-6)
        assert np.allclose(result["rotation"][f], _snap_axis_to_up(ref_rotation), atol=1e-6)
    # No pop at the entry boundary: frame start-1 and frame start are identical.
    assert np.allclose(result["translation"][snap_frame - 1], result["translation"][snap_frame], atol=1e-6)


def test_held_after_an_event_freezes_at_its_final_attached_pose():
    n_frames = 10
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_idx = REGION_JOINTS["left_hand"][-1]
    event = {"region": "left_hand", "start_frame": 1, "end_frame": 4}
    joint_world = _joint_world_transforms(body_motion, skeleton)
    ref_center = joint_world[event["start_frame"], joint_idx, :3, 3]
    ref_rotation = joint_world[event["start_frame"], joint_idx, :3, :3]

    def object_position_fn(f):
        return (ref_center, ref_rotation) if f == event["start_frame"] else None

    result = compute_object_pose_sequence(
        n_frames=n_frames,
        attachment_events=[event],
        body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=np.eye(3),
        object_position_fn=object_position_fn,
        skeleton=skeleton,
    )

    final_attached_translation = result["translation"][event["end_frame"]]
    for f in range(event["end_frame"] + 1, n_frames):
        assert np.allclose(result["translation"][f], final_attached_translation, atol=1e-6)
        assert result["is_low_confidence"][f]
    # No pop at the exit boundary either.
    assert np.allclose(
        result["translation"][event["end_frame"]], result["translation"][event["end_frame"] + 1], atol=1e-6,
    )


def test_held_between_two_events_freezes_at_the_first_events_final_pose():
    """No independent tracking happens between two holds, the object stays
    wherever the first event left it until the second event's own snap
    resolves, mirroring how a real object set down and left alone would
    behave (this project's deliberate choice over trying to independently
    depth-track the object while it isn't held, see the module docstring)."""
    n_frames = 14
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_idx_left = REGION_JOINTS["left_hand"][-1]
    joint_idx_right = REGION_JOINTS["right_hand"][-1]
    joint_world = _joint_world_transforms(body_motion, skeleton)

    event_a = {"region": "left_hand", "start_frame": 1, "end_frame": 3}
    event_b = {"region": "right_hand", "start_frame": 8, "end_frame": 11}
    ref_a_center = joint_world[event_a["start_frame"], joint_idx_left, :3, 3]
    ref_a_rotation = joint_world[event_a["start_frame"], joint_idx_left, :3, :3]
    ref_b_center = joint_world[event_b["start_frame"], joint_idx_right, :3, 3]
    ref_b_rotation = joint_world[event_b["start_frame"], joint_idx_right, :3, :3]

    def object_position_fn(f):
        if f == event_a["start_frame"]:
            return ref_a_center, ref_a_rotation
        if f == event_b["start_frame"]:
            return ref_b_center, ref_b_rotation
        return None

    result = compute_object_pose_sequence(
        n_frames=n_frames,
        attachment_events=[event_a, event_b],
        body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=np.eye(3),
        object_position_fn=object_position_fn,
        skeleton=skeleton,
    )

    held_value = result["translation"][event_a["end_frame"]]
    for f in range(event_a["end_frame"] + 1, event_b["start_frame"]):
        assert np.allclose(result["translation"][f], held_value, atol=1e-6)
        assert result["is_low_confidence"][f]
    # Event b's own attached window is real motion again, not held.
    assert not result["is_low_confidence"][event_b["start_frame"]]


def test_falls_back_to_initial_pose_when_there_are_no_attachment_events():
    n_frames = 5
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    initial_center = np.array([1.0, 2.0, 3.0])
    initial_rotation = Rotation.from_euler("y", 30.0, degrees=True).as_matrix()

    result = compute_object_pose_sequence(
        n_frames=n_frames,
        attachment_events=[],
        body_motion=body_motion,
        initial_center=initial_center, initial_rotation=initial_rotation,
        object_position_fn=lambda f: (np.zeros(3), np.eye(3)),  # never consulted, no events
        skeleton=skeleton,
    )

    assert np.allclose(result["translation"], initial_center)
    assert np.allclose(result["rotation"], initial_rotation)
    assert result["is_low_confidence"].all()


def test_attached_segment_falls_back_and_flags_low_confidence_when_unfindable():
    """The object is never measured anywhere near the one event's own search
    window, the event anchors to the initial fallback pose instead of
    crashing, and is flagged low-confidence throughout."""
    n_frames = 6
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()

    result = compute_object_pose_sequence(
        n_frames=n_frames,
        attachment_events=[{"region": "left_hand", "start_frame": 1, "end_frame": 4}],
        body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=np.eye(3),
        object_position_fn=lambda f: None,
        skeleton=skeleton,
    )

    assert result["is_low_confidence"][1:5].all()
    assert result["is_low_confidence"].all()  # before/during/after all held or unmeasured


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_joint_world_transforms_matches_real_smplx_forward_wrist_position():
    """Regression test for a real bug: `_joint_world_transforms` originally
    never incorporated `transl` (the root's own per-frame world position) at
    all, silently building every joint's world transform relative to the
    world origin instead. The bug was invisible to the propagation tests
    above, which only ever compare against `_joint_world_transforms`'s own
    output as ground truth (a self-consistency check, blind to whether that
    ground truth itself is correct), found instead by comparing against a
    real `smplx.create().forward()` call on real retargeted motion data. This
    test guards against it recurring by checking the real model's own wrist
    joint directly, for a real (non-fake) skeleton."""
    import smplx

    from pipeline.adapters.gvhmr.gvhmr_smplx_skeleton import SmplxSkeleton

    n_frames = 5
    rng = np.random.default_rng(0)
    global_orient = rng.normal(scale=0.2, size=(n_frames, 3)).astype(np.float32)
    body_pose = rng.normal(scale=0.1, size=(n_frames, 63)).astype(np.float32)
    betas = np.zeros((n_frames, 10), dtype=np.float32)
    transl = np.array([[1.0, 2.0, 3.0]] * n_frames, dtype=np.float32) + np.arange(n_frames)[:, None] * 0.1

    model = smplx.create(
        str(SMPLX_MODEL_PATH), model_type="smplx", gender="neutral", num_betas=10,
        use_pca=False, flat_hand_mean=True, batch_size=n_frames,
    )
    output = model(
        global_orient=torch.from_numpy(global_orient), body_pose=torch.from_numpy(body_pose),
        betas=torch.from_numpy(betas), transl=torch.from_numpy(transl),
    )
    real_joints = output.joints.detach().numpy()

    body_motion = {"global_orient": global_orient, "body_pose": body_pose, "betas": betas, "transl": transl}
    joint_world = _joint_world_transforms(body_motion, SmplxSkeleton())

    for joint_idx in [0, 15, 20, 21]:  # pelvis, head, left wrist, right wrist
        assert np.allclose(joint_world[:, joint_idx, :3, 3], real_joints[:, joint_idx], atol=0.02)


def test_object_radius_zero_reproduces_joint_coincident_behavior():
    """Default `object_radius=0.0` must exactly reproduce the plain contact-
    point-anchoring behavior (`ref_center` at the joint, no offset), also
    confirms the hand-direction lookup is skipped entirely when radius is
    zero, so this needs no SMPL-X model file."""
    n_frames = 10
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_world = _joint_world_transforms(body_motion, skeleton)

    joint_idx = REGION_JOINTS["left_hand"][-1]
    event = {"region": "left_hand", "start_frame": 3, "end_frame": 6}
    snap_frame = event["start_frame"]
    true_rotation = joint_world[snap_frame, joint_idx, :3, :3]

    def object_position_fn(f):
        return (np.zeros(3), true_rotation) if f == snap_frame else None

    result = compute_object_pose_sequence(
        n_frames=n_frames, attachment_events=[event], body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=true_rotation,
        object_position_fn=object_position_fn, skeleton=skeleton,
    )
    resolved = result["resolved_events"][0]
    assert np.allclose(resolved["ref_center"], joint_world[snap_frame, joint_idx, :3, 3], atol=1e-6)


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_left_hand_event_offsets_by_the_two_term_anatomical_model():
    """The offset is `radius * GRIP_NORMAL_SCALE * normal + palm_length *
    GRIP_DISTAL_SCALE * distal`, `palm_length` comes from `skeleton.get_
    skeleton(betas)` (here `_FakeSkeleton`'s own synthetic rest geometry, not
    real anatomy, the point of this test is the *formula*, not real body
    proportions), matching how `compute_object_pose_sequence` itself derives
    it from `body_motion["betas"]`."""
    from pipeline.algorithms.hand_retarget import (
        LEFT_INDEX1, LEFT_MIDDLE1, LEFT_PINKY1, LEFT_WRIST,
        palm_distal_direction, palm_normal_direction,
    )
    from pipeline.algorithms.hoi_object_pose import GRIP_DISTAL_SCALE, GRIP_NORMAL_SCALE

    n_frames = 10
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_world = _joint_world_transforms(body_motion, skeleton)

    joint_idx = REGION_JOINTS["left_hand"][-1]
    event = {"region": "left_hand", "start_frame": 3, "end_frame": 6}
    snap_frame = event["start_frame"]
    true_rotation = joint_world[snap_frame, joint_idx, :3, :3]
    object_radius = 0.15

    def object_position_fn(f):
        return (np.zeros(3), true_rotation) if f == snap_frame else None

    result = compute_object_pose_sequence(
        n_frames=n_frames, attachment_events=[event], body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=true_rotation,
        object_position_fn=object_position_fn, skeleton=skeleton, object_radius=object_radius,
    )

    joint_pos = joint_world[snap_frame, joint_idx, :3, 3]
    fake_rest = skeleton.get_skeleton(torch.zeros(10)).numpy()
    palm_length = float(np.linalg.norm(fake_rest[LEFT_MIDDLE1] - fake_rest[LEFT_WRIST]))
    normal = palm_normal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, mirror=False)
    distal = palm_distal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, LEFT_MIDDLE1)
    local_offset = object_radius * GRIP_NORMAL_SCALE * normal + palm_length * GRIP_DISTAL_SCALE * distal
    expected_center = joint_pos + joint_world[snap_frame, joint_idx, :3, :3] @ local_offset

    resolved = result["resolved_events"][0]
    assert np.allclose(resolved["ref_center"], expected_center, atol=1e-6)
    # Sanity check independent of the exact formula: must NOT still sit
    # exactly on the joint (the bug this feature fixes).
    offset = np.array(resolved["ref_center"]) - joint_pos
    assert np.linalg.norm(offset) > 1e-3


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_palm_normal_direction_is_a_unit_vector():
    from pipeline.algorithms.hand_retarget import LEFT_INDEX1, LEFT_PINKY1, LEFT_WRIST, palm_normal_direction

    d = palm_normal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, mirror=False)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5)


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_palm_normal_direction_mirror_flag_flips_the_sign():
    """A cross product's sign encodes handedness, the same joint order
    read for the un-mirrored hand must come out negated for its mirror
    image, confirming `mirror` actually corrects for that (not a no-op)."""
    from pipeline.algorithms.hand_retarget import LEFT_INDEX1, LEFT_PINKY1, LEFT_WRIST, palm_normal_direction

    unmirrored = palm_normal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, mirror=False)
    mirrored = palm_normal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, mirror=True)
    assert np.allclose(mirrored, -unmirrored, atol=1e-6)


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_palm_normal_direction_matches_real_grip_data_regression():
    """Ground-truth regression: these exact directions (mirror=False for
    left, mirror=True for right) were validated against real hand-placed
    reference points across several real grip events, both hands, ~35-39
    degrees average error vs. the real grip direction. Pins the exact
    numeric result so a future change to this formula gets caught here, not
    silently drifts away from the validated direction."""
    from pipeline.algorithms.hand_retarget import (
        LEFT_INDEX1,
        LEFT_PINKY1,
        LEFT_WRIST,
        RIGHT_INDEX1,
        RIGHT_PINKY1,
        RIGHT_WRIST,
        palm_normal_direction,
    )

    left = palm_normal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, mirror=False)
    right = palm_normal_direction(RIGHT_WRIST, RIGHT_INDEX1, RIGHT_PINKY1, mirror=True)
    assert np.allclose(left, [-0.10726821, -0.986735, 0.1218503], atol=1e-5)
    assert np.allclose(right, [0.14146563, -0.981235, 0.13101627], atol=1e-5)


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_palm_distal_direction_is_a_unit_vector_orthogonal_to_the_normal():
    from pipeline.algorithms.hand_retarget import (
        LEFT_INDEX1, LEFT_MIDDLE1, LEFT_PINKY1, LEFT_WRIST, palm_distal_direction, palm_normal_direction,
    )

    distal = palm_distal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, LEFT_MIDDLE1)
    normal = palm_normal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, mirror=False)
    assert np.isclose(np.linalg.norm(distal), 1.0, atol=1e-5)
    assert np.isclose(np.dot(distal, normal), 0.0, atol=1e-6)


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_palm_distal_direction_matches_real_grip_data_regression():
    """Pins the exact numeric result, matching `test_palm_normal_direction_
    matches_real_grip_data_regression`'s own convention, these two
    together validated against 24 real reference points across several
    grip types, cross-validated (see `hoi_object_pose.py`'s
    `GRIP_NORMAL_SCALE` comment for the numbers)."""
    from pipeline.algorithms.hand_retarget import (
        LEFT_INDEX1, LEFT_MIDDLE1, LEFT_PINKY1, LEFT_WRIST,
        RIGHT_INDEX1, RIGHT_MIDDLE1, RIGHT_PINKY1, RIGHT_WRIST,
        palm_distal_direction,
    )

    left = palm_distal_direction(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, LEFT_MIDDLE1)
    right = palm_distal_direction(RIGHT_WRIST, RIGHT_INDEX1, RIGHT_PINKY1, RIGHT_MIDDLE1)
    assert np.allclose(left, [0.99330336, -0.11165075, -0.0297072], atol=1e-5)
    assert np.allclose(right, [-0.98889714, -0.1461555, -0.02685117], atol=1e-5)


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_non_hand_region_gets_no_offset_even_with_nonzero_radius():
    """left_arm has no defined "grip direction", object_radius>0 must not
    move it off the joint the way it does for left_hand/right_hand."""
    n_frames = 10
    body_motion = _fake_body_motion(n_frames)
    skeleton = _FakeSkeleton()
    joint_world = _joint_world_transforms(body_motion, skeleton)

    joint_idx = REGION_JOINTS["left_arm"][-1]
    event = {"region": "left_arm", "start_frame": 3, "end_frame": 6}
    snap_frame = event["start_frame"]
    true_rotation = joint_world[snap_frame, joint_idx, :3, :3]

    def object_position_fn(f):
        return (np.zeros(3), true_rotation) if f == snap_frame else None

    result = compute_object_pose_sequence(
        n_frames=n_frames, attachment_events=[event], body_motion=body_motion,
        initial_center=np.zeros(3), initial_rotation=true_rotation,
        object_position_fn=object_position_fn, skeleton=skeleton, object_radius=0.15,
    )
    resolved = result["resolved_events"][0]
    assert np.allclose(resolved["ref_center"], joint_world[snap_frame, joint_idx, :3, 3], atol=1e-6)
