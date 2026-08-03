"""Stage 8 tests: a real-data regression test (needs GPU/checkpoints/the
SMPL-X model file, skipped otherwise -- see conftest.py) plus pure-logic unit
tests for the attachment-event resolution pipeline (no GPU/model dependency,
always run).
"""

from __future__ import annotations

import numpy as np
import torch

from pipeline.stages.stage_8_optimize_hoi import _qualifying_attachment_events

FPS = 30.0


def _contact_event(region: str, start: int, end: int, confidence: float, is_low_confidence: bool = False) -> dict:
    return {
        "regions": [region], "start_frame": start, "end_frame": end, "mean_confidence": confidence,
        "is_low_confidence": is_low_confidence,
    }


def test_qualifying_attachment_events_re_bridges_a_gap_reopened_by_overlap_resolution():
    """Real failure mode from `testPassMugBetweenHands`: stage 7's own
    `contact_events.json` had `right_hand` (119-177), a weaker `right_arm`
    candidate spanning the hand-off (174-184), and `left_hand` (185-262) --
    the arm candidate would have bridged the two hands, but
    `_resolve_overlapping_events` gives the overlapping frames to the
    higher-confidence hand events instead, stripping the arm candidate down
    to nothing and reopening a real gap (178-184) between the two final
    events that stage 7's own bridging had no reason to expect. The second
    bridge pass in `_qualifying_attachment_events` must close it."""
    contact_events = [
        _contact_event("right_hand", 119, 177, 0.9),
        _contact_event("right_arm", 174, 184, 0.5),  # loses the overlap, ends up unclaimed
        _contact_event("left_hand", 185, 262, 0.9),
    ]

    qualifying = _qualifying_attachment_events(contact_events, FPS)

    regions = [e["region"] for e in qualifying]
    assert "right_arm" not in regions  # fully out-competed, as in the real data
    right_hand = next(e for e in qualifying if e["region"] == "right_hand")
    left_hand = next(e for e in qualifying if e["region"] == "left_hand")
    assert left_hand["start_frame"] - right_hand["end_frame"] - 1 <= 1, (
        f"gap between {right_hand} and {left_hand} was not re-bridged"
    )


def test_qualifying_attachment_events_bridges_a_direct_gap_between_two_resolved_events():
    """Simpler case, no overlap-resolution involved: two otherwise-unrelated
    candidate events left with a short gap between them after duration
    filtering should still come out bridged."""
    contact_events = [
        _contact_event("left_hand", 0, 50, 0.9),
        _contact_event("right_hand", 55, 100, 0.9),  # 4-frame gap
    ]

    qualifying = _qualifying_attachment_events(contact_events, FPS)

    assert len(qualifying) == 2
    left, right = sorted(qualifying, key=lambda e: e["start_frame"])
    assert right["start_frame"] - left["end_frame"] - 1 <= 1


def test_qualifying_attachment_events_leaves_a_genuine_release_ungap_bridged():
    """A gap well beyond BRIDGE_GAP_MAX_SECONDS is a real release (the object
    was set down), not a hand-off -- must NOT be bridged."""
    contact_events = [
        _contact_event("left_hand", 0, 50, 0.9),
        _contact_event("right_hand", 200, 250, 0.9),  # ~5s gap at 30fps
    ]

    qualifying = _qualifying_attachment_events(contact_events, FPS)

    left, right = sorted(qualifying, key=lambda e: e["start_frame"])
    assert left["end_frame"] == 50
    assert right["start_frame"] == 200


def test_qualifying_attachment_events_excludes_a_low_confidence_non_grip_region():
    """Real case from `testCoffeeMug`: a mug resting on the floor near a
    passing foot produced a 16-frame `left_leg` event (0.55 mean confidence,
    a 0.043m depth gap -- small enough to look like real contact by depth
    alone, since the object and foot really are close in space, just never
    touching) that cleared the duration bar and would have become a real
    (wrong) attachment. `is_low_confidence=True` (as stage 7 now flags this
    exact case -- see `_is_low_confidence`'s own docstring) must keep a
    non-`GRIP_CAPABLE_REGIONS` event like this one out entirely, not just
    flagged."""
    contact_events = [_contact_event("left_leg", 565, 580, 0.55, is_low_confidence=True)]

    qualifying = _qualifying_attachment_events(contact_events, FPS)

    assert qualifying == []


def test_qualifying_attachment_events_still_includes_a_low_confidence_hand_grip():
    """The exact concern the exclusion above must not break: a real hand
    grip can legitimately be flagged `is_low_confidence` (a full grip
    wrapped around/behind the object commonly depresses both the 2D and
    depth signals for reasons unrelated to whether contact is real -- see
    `GRIP_CAPABLE_REGIONS`'s own comment) and must still qualify, since
    `GRIP_CAPABLE_REGIONS` is exempt from this gate entirely."""
    contact_events = [_contact_event("right_hand", 100, 130, 0.49, is_low_confidence=True)]

    qualifying = _qualifying_attachment_events(contact_events, FPS)

    assert len(qualifying) == 1
    assert qualifying[0]["region"] == "right_hand"


def test_object_pose_output_is_plausible(stage_8_result):
    data = torch.load(stage_8_result["object_pose"], weights_only=False)

    assert set(data.keys()) == {"translation", "rotation", "is_low_confidence"}
    translation = data["translation"]
    rotation = data["rotation"]
    is_low_confidence = data["is_low_confidence"]

    n_frames = translation.shape[0]
    assert translation.shape == (n_frames, 3)
    assert rotation.shape == (n_frames, 3, 3)
    assert is_low_confidence.shape == (n_frames,)

    assert torch.isfinite(translation).all()
    assert torch.isfinite(rotation).all()

    # Every rotation should be a proper (orthonormal, det=+1) rotation matrix.
    identity = torch.eye(3, dtype=rotation.dtype).expand(n_frames, 3, 3)
    assert torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-3)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(n_frames, dtype=rotation.dtype), atol=1e-2)

    # The test clip's own racket-holding grip spans (close to) the whole clip
    # (already confirmed by stage 7's own real-data test: one continuous,
    # full-confidence contact event) -- most frames should be a real
    # attachment, not a fallback/never-tracked flag.
    assert is_low_confidence.float().mean() < 0.5


def test_object_pose_npz_matches_the_pt_file(stage_8_result):
    pt_data = torch.load(stage_8_result["object_pose"], weights_only=False)
    with np.load(stage_8_result["object_pose_npz"]) as npz_data:
        assert np.allclose(npz_data["translation"], pt_data["translation"].numpy())
        assert np.allclose(npz_data["rotation"], pt_data["rotation"].numpy())
        assert np.array_equal(npz_data["is_low_confidence"], pt_data["is_low_confidence"].numpy())
