"""Unit tests for `contact_detection`: pure-numpy geometry + hysteresis, no
GPU/checkpoints needed -- always runs.
"""

from __future__ import annotations

import numpy as np

from pipeline.algorithms.contact_detection import (
    CONTACT_PIXEL_THRESHOLD,
    HEAD_JOINT,
    HEAD_TOP_JOINT_INDEX,
    LEFT_FINGERTIP_JOINTS,
    LEFT_WRIST_JOINT,
    REGION_JOINT_NAMES,
    REGION_JOINTS,
    REGION_NAMES,
    ContactEvent,
    _confidence_from_mask_distance,
    candidate_joint_indices,
    consolidate_overlapping_events,
    depth_gap_for_joint,
    detect_contact_events,
    frame_confidence_for_region,
    per_frame_region_confidence,
)

FPS = 30.0


def _square_mask(size=100, box=(40, 40, 60, 60)):
    mask = np.zeros((size, size), dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def test_confidence_is_full_inside_the_mask():
    mask = _square_mask()
    pixels = np.array([[50.0, 50.0]])  # dead center of the box
    confidence = _confidence_from_mask_distance(pixels, mask)
    assert confidence[0] == 1.0


def test_confidence_decays_linearly_with_distance():
    mask = _square_mask()
    # The box spans columns 40-59 inclusive (mask[:, 40:60]), so pixel 65 is
    # 6px past the nearest True column (59), not 5 -- an off-by-one against
    # the exclusive slice bound if computed naively.
    pixels = np.array([[65.0, 50.0]])
    confidence = _confidence_from_mask_distance(pixels, mask)
    expected = 1.0 - 6.0 / CONTACT_PIXEL_THRESHOLD
    assert abs(confidence[0] - expected) < 1e-6


def test_confidence_is_zero_beyond_the_threshold():
    mask = _square_mask()
    pixels = np.array([[60.0 + CONTACT_PIXEL_THRESHOLD + 10, 50.0]])
    confidence = _confidence_from_mask_distance(pixels, mask)
    assert confidence[0] == 0.0


def test_all_nine_regions_are_defined_with_matching_joint_names():
    assert set(REGION_NAMES) == {
        "left_hand", "right_hand", "head", "head_top", "chest",
        "left_arm", "right_arm", "left_leg", "right_leg",
    }
    for region in REGION_NAMES:
        assert len(candidate_joint_indices(region)) == len(REGION_JOINT_NAMES[region])


def test_head_top_joint_index_does_not_collide_with_any_real_region_joint():
    """HEAD_TOP_JOINT_INDEX is a synthetic index (the appended mesh vertex,
    see that constant's own comment) -- it must never coincide with a real
    skeletal joint index another region uses, or consolidate_overlapping_events
    would wrongly treat a head-top contact and some other region's contact as
    the same physical joint."""
    real_region_indices = {
        idx for region, indices in REGION_JOINTS.items() if region != "head_top" for idx in indices
    }
    assert HEAD_TOP_JOINT_INDEX not in real_region_indices


def test_attachment_joint_index_redirects_head_top_to_the_real_head_joint():
    """stage 8/9 rigidly attach to a real skeletal bone -- head_top's own
    REGION_JOINTS entry (a synthetic mesh-vertex index) has no such bone, so
    attachment_joint_index must redirect it to HEAD_JOINT instead (the same
    one the "head" region already uses), not return the raw vertex index."""
    assert attachment_joint_index("head_top") == HEAD_JOINT


def test_attachment_joint_index_matches_region_joints_last_entry_elsewhere():
    for region in REGION_NAMES:
        if region == "head_top":
            continue
        assert attachment_joint_index(region) == REGION_JOINTS[region][-1]


def test_candidate_joint_indices_left_vs_right_hand_are_disjoint():
    left = candidate_joint_indices("left_hand")
    right = candidate_joint_indices("right_hand")
    assert len(left) == len(REGION_JOINT_NAMES["left_hand"]) == 6
    assert set(left).isdisjoint(right)
    assert LEFT_WRIST_JOINT in left
    assert LEFT_FINGERTIP_JOINTS[0] in left


def test_frame_confidence_for_region_picks_the_closest_joint():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    mask = _square_mask()  # object mask covers pixels (40-60, 40-60)

    joint_ids = candidate_joint_indices("left_hand")
    n_joints_total = max(joint_ids) + 1
    joints_xyz = np.zeros((1, n_joints_total, 3))
    # Every candidate joint far off to the side (projects way outside the
    # mask, with distinct X/Y so they don't all collapse onto the same pixel)
    # except the wrist, which sits dead in the middle of the mask's own pixel
    # region.
    joints_xyz[0, joint_ids] = [2.0, 2.0, 1.0]
    joints_xyz[0, LEFT_WRIST_JOINT] = [0.0, 0.0, 1.0]  # projects to (50, 50) via K

    confidence, joint_idx = frame_confidence_for_region(joints_xyz, "left_hand", K, [mask])

    assert confidence[0] == 1.0
    assert REGION_JOINT_NAMES["left_hand"][joint_idx[0]] == "wrist"


def test_frame_confidence_for_region_works_for_a_body_region_too():
    """Same mechanism for a non-hand region (chest): two candidate joints
    (spine2, spine3), neither of them a hand/finger joint."""
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    mask = _square_mask()

    joint_ids = candidate_joint_indices("chest")
    joints_xyz = np.zeros((1, max(joint_ids) + 1, 3))
    joints_xyz[0, joint_ids[0]] = [2.0, 2.0, 1.0]  # off to the side
    joints_xyz[0, joint_ids[1]] = [0.0, 0.0, 1.0]  # dead center of the mask

    confidence, joint_idx = frame_confidence_for_region(joints_xyz, "chest", K, [mask])

    assert confidence[0] == 1.0
    assert REGION_JOINT_NAMES["chest"][joint_idx[0]] == "spine3"


def test_frame_confidence_for_region_is_zero_without_a_tracked_mask():
    K = np.eye(3)
    joints_xyz = np.zeros((2, 22, 3))
    confidence, joint_idx = frame_confidence_for_region(joints_xyz, "left_hand", K, [None, None])
    assert np.all(confidence == 0.0)
    assert np.all(joint_idx == -1)


def test_per_frame_region_confidence_matches_frame_confidence_for_region():
    """The batched, single-distance-transform version (used by stage 7's
    per-frame loop) should agree exactly with the simpler per-region function
    it's meant to replace at the call site, just computed all at once."""
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    mask = _square_mask()

    # Sized to cover HEAD_TOP_JOINT_INDEX (the appended mesh-vertex slot, past
    # the real 55-joint body+hands layout) -- per_frame_region_confidence
    # indexes every region's own joints, not just left_hand's. z=1.0 everywhere
    # (a realistic camera-space depth) avoids a divide-by-zero projecting the
    # other regions' still-untouched joints.
    frame_joints = np.zeros((HEAD_TOP_JOINT_INDEX + 1, 3))
    frame_joints[..., 2] = 1.0
    joint_ids = candidate_joint_indices("left_hand")
    frame_joints[joint_ids] = [2.0, 2.0, 1.0]
    frame_joints[LEFT_WRIST_JOINT] = [0.0, 0.0, 1.0]

    result = per_frame_region_confidence(frame_joints, K, mask)

    expected_confidence, expected_joint_idx = frame_confidence_for_region(frame_joints[None], "left_hand", K, [mask])
    assert result["left_hand"] == (expected_confidence[0], expected_joint_idx[0])
    assert REGION_JOINT_NAMES["left_hand"][result["left_hand"][1]] == "wrist"


def test_per_frame_region_confidence_covers_every_region():
    K = np.eye(3)
    frame_joints = np.zeros((HEAD_TOP_JOINT_INDEX + 1, 3))
    frame_joints[..., 2] = 1.0
    mask = _square_mask()

    result = per_frame_region_confidence(frame_joints, K, mask)
    assert set(result.keys()) == set(REGION_NAMES)


def test_per_frame_region_confidence_is_zero_without_a_tracked_mask():
    K = np.eye(3)
    frame_joints = np.zeros((22, 3))
    result = per_frame_region_confidence(frame_joints, K, None)
    assert all(confidence == 0.0 and joint_idx == -1 for confidence, joint_idx in result.values())


def test_detect_contact_events_finds_a_clear_contact_stretch():
    confidence = np.zeros(20)
    confidence[10:15] = 0.9
    joint_idx = np.zeros(20, dtype=int)

    events = detect_contact_events(confidence, joint_idx, "left_hand", REGION_JOINT_NAMES["left_hand"])

    # CONTACT_WINDOW=5's rolling max naturally widens the elevated region by
    # (window//2) frames on each side of the raw 10-14 spike, same as the
    # wrist-plausibility gate's own hysteresis -- 8-16, not exactly 10-14.
    assert len(events) == 1
    assert events[0].start_frame == 8
    assert events[0].end_frame == 16
    assert events[0].regions == ["left_hand"]
    assert events[0].joint == REGION_JOINT_NAMES["left_hand"][0]
    # Peak confidence (0.9) sits at raw frames 10-14; argmax picks the first,
    # frame 10, within the widened 8-16 window.
    assert events[0].peak_frame == 10


def test_detect_contact_events_ignores_a_run_that_never_seeds():
    # Elevated (above release) throughout, but never crosses the stricter seed
    # threshold -- should never be reported as a real contact.
    confidence = np.full(20, 0.5)
    joint_idx = np.zeros(20, dtype=int)
    events = detect_contact_events(confidence, joint_idx, "left_hand", REGION_JOINT_NAMES["left_hand"])
    assert events == []


def test_detect_contact_events_bridges_a_brief_dip():
    """Same hysteresis idea as the wrist-plausibility gate: a real grip dipping
    briefly (but staying above the release threshold, within the rolling
    window) shouldn't fragment into two separate events."""
    confidence = np.zeros(20)
    confidence[5:9] = 0.9
    confidence[9] = 0.4  # dip -- below seed, but still above release
    confidence[10:14] = 0.9
    joint_idx = np.zeros(20, dtype=int)

    events = detect_contact_events(confidence, joint_idx, "left_hand", REGION_JOINT_NAMES["left_hand"])

    # Same window-widening as above: raw activity spans 5-13, elevated spans 3-15.
    assert len(events) == 1
    assert events[0].start_frame == 3
    assert events[0].end_frame == 15


def test_detect_contact_events_two_separate_grips_stay_separate():
    confidence = np.zeros(30)
    confidence[2:6] = 0.9
    confidence[20:24] = 0.9
    joint_idx = np.zeros(30, dtype=int)

    events = detect_contact_events(confidence, joint_idx, "right_hand", REGION_JOINT_NAMES["right_hand"])

    # Each raw spike's elevated region widens by (window//2) on each side
    # (2-5 -> 0-7, 20-23 -> 18-25); the gap between them is still wide enough
    # that the two stay separate rather than merging into one event.
    assert len(events) == 2
    assert (events[0].start_frame, events[0].end_frame) == (0, 7)
    assert (events[1].start_frame, events[1].end_frame) == (18, 25)
    assert all(e.regions == ["right_hand"] for e in events)


def _event(regions, joint, start, end, peak, confidence):
    return ContactEvent(
        regions=list(regions), joint=joint, start_frame=start, end_frame=end,
        peak_frame=peak, mean_confidence=confidence,
    )


def test_consolidate_merges_hand_and_arm_events_sharing_the_wrist():
    hand_event = _event(["left_hand"], "wrist", 10, 20, 15, 0.95)
    arm_event = _event(["left_arm"], "wrist", 12, 18, 14, 0.6)

    merged = consolidate_overlapping_events([hand_event, arm_event])

    assert len(merged) == 1
    assert set(merged[0].regions) == {"left_hand", "left_arm"}
    assert merged[0].joint == "wrist"
    assert merged[0].start_frame == 10
    assert merged[0].end_frame == 20
    # The more confident sub-event (hand, 0.95) is the representative one.
    assert merged[0].peak_frame == 15
    assert merged[0].mean_confidence == 0.95


def test_consolidate_leaves_different_joints_separate():
    hand_event = _event(["left_hand"], "wrist", 10, 20, 15, 0.9)
    chest_event = _event(["chest"], "spine2", 10, 20, 15, 0.9)  # same frames, unrelated joint

    merged = consolidate_overlapping_events([hand_event, chest_event])

    assert len(merged) == 2


def test_consolidate_leaves_non_overlapping_same_joint_events_separate():
    """Two genuinely separate grips (same wrist joint) shouldn't merge just
    because they share a joint -- only overlapping-in-time events represent
    the same physical touch."""
    first_grip = _event(["left_hand"], "wrist", 0, 5, 2, 0.9)
    second_grip = _event(["left_arm"], "wrist", 50, 55, 52, 0.9)

    merged = consolidate_overlapping_events([first_grip, second_grip])

    assert len(merged) == 2


def test_consolidate_merges_a_chain_of_three_overlapping_events():
    a = _event(["left_hand"], "wrist", 0, 10, 5, 0.5)
    b = _event(["left_arm"], "wrist", 8, 15, 12, 0.9)
    c = _event(["left_hand"], "wrist", 14, 20, 17, 0.3)

    merged = consolidate_overlapping_events([a, b, c])

    assert len(merged) == 1
    assert merged[0].start_frame == 0
    assert merged[0].end_frame == 20
    assert merged[0].mean_confidence == 0.9
    assert merged[0].peak_frame == 12


def test_depth_gap_for_joint_is_zero_when_object_and_body_are_at_the_same_depth():
    depth = np.full((100, 100), 2.0)
    object_mask = np.zeros((100, 100), dtype=bool)
    object_mask[40:60, 40:60] = True
    human_mask = np.zeros((100, 100), dtype=bool)
    human_mask[35:65, 35:65] = True

    gap = depth_gap_for_joint(depth, object_mask, human_mask, joint_pixel=np.array([50.0, 50.0]))

    assert gap == 0.0


def test_depth_gap_for_joint_is_large_for_incidental_occlusion():
    """Object and body silhouettes overlap where the joint projects, but
    they're at genuinely different depths -- one is merely in front of the
    other in the image, not touching."""
    depth = np.zeros((100, 100))
    object_mask = np.zeros((100, 100), dtype=bool)
    object_mask[40:60, 40:60] = True
    depth[object_mask] = 3.0

    human_mask = np.zeros((100, 100), dtype=bool)
    human_mask[0:20, 0:20] = True
    depth[human_mask] = 1.0

    gap = depth_gap_for_joint(depth, object_mask, human_mask, joint_pixel=np.array([10.0, 10.0]))

    assert abs(gap - 2.0) < 1e-6


def test_depth_gap_for_joint_is_none_when_a_mask_is_empty():
    depth = np.zeros((100, 100))
    object_mask = np.zeros((100, 100), dtype=bool)  # never tracked this frame
    human_mask = np.zeros((100, 100), dtype=bool)
    human_mask[10:20, 10:20] = True

    assert depth_gap_for_joint(depth, object_mask, human_mask, np.array([15.0, 15.0])) is None
