"""Regression tests for stage 7's contact-detection hysteresis, run against
REAL per-frame confidence curves cached from real clips -- not synthetic
idealized data. `tests/assets/real_clip_contact_confidence/*.npz` each store,
per region, the real `detect_contact_events` input (confidence, joint_idx)
computed once, offline, from that clip's own raw-incam-translation joint
projections against its real object mask (see `pipeline.algorithms.
contact_detection.per_frame_region_confidence` -- the fixtures are that
function's own output, frozen). Fast and GPU-free at test time: the expensive
part (SMPL-X forward kinematics + real-mask distance-transform) already
happened once to produce them.

These exist because two real bugs both slipped past the existing synthetic
unit tests in `test_contact_detection.py`: a translation correction that
looked correct in isolation nonetheless shifted real joint positions enough
to fabricate a contact event on one clip and fragment a real one on another.
Synthetic confidence curves are idealized and don't reproduce that kind of
noise; these fixtures do, because they're real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.algorithms.contact_detection import (
    REGION_JOINT_NAMES,
    REGION_NAMES,
    ContactEvent,
    bridge_short_gaps,
    consolidate_overlapping_events,
    contiguous_true_runs,
    detect_contact_events,
)
from pipeline.stages.stage_8_optimize_hoi import _qualifying_attachment_events

FIXTURES_DIR = Path(__file__).parent / "assets" / "real_clip_contact_confidence"


def _load_fixture(name: str) -> dict:
    return dict(np.load(FIXTURES_DIR / f"{name}.npz"))


def _events_for_region(fixture: dict, region: str) -> list[ContactEvent]:
    confidence = fixture[f"{region}_confidence"]
    joint_idx = fixture[f"{region}_joint_idx"]
    return consolidate_overlapping_events(
        detect_contact_events(confidence, joint_idx, region, REGION_JOINT_NAMES[region])
    )


def _covering_event(events: list[ContactEvent], frame: int) -> ContactEvent:
    covering = [e for e in events if e.start_frame <= frame <= e.end_frame]
    assert len(covering) == 1, f"expected exactly one event covering frame {frame}, found {covering}"
    return covering[0]


def test_coffee_mug_never_produces_a_qualifying_leg_attachment():
    """Regression: the foot-lock translation correction once shifted the
    right leg into coincidental 2D proximity with the mug for 30 frames
    (186-216) -- long enough to become a real (wrong) attachment in stage 8.
    Runs the actual `_qualifying_attachment_events` (duration filter +
    cross-region overlap resolution), not just a duration check in isolation
    -- a shorter leg candidate (e.g. 316-328) can legitimately clear the
    duration bar on its own and still never become a real attachment, because
    it's fully overlapped in time by the real, higher-confidence hand grip
    and loses that resolution -- exactly what happens in the real pipeline."""
    fixture = _load_fixture("testCoffeeMug")
    fps = float(fixture["fps"])

    all_events = []
    for region in REGION_NAMES:
        if f"{region}_confidence" not in fixture:
            continue  # fixture frozen before this region existed (e.g. head_top) -- nothing to check
        all_events.extend(_events_for_region(fixture, region))
    candidates = [
        {"regions": e.regions, "start_frame": e.start_frame, "end_frame": e.end_frame, "mean_confidence": e.mean_confidence}
        for e in all_events
    ]
    qualifying = _qualifying_attachment_events(candidates, fps)

    leg_attachments = [e for e in qualifying if e["region"] in ("left_leg", "right_leg")]
    assert leg_attachments == [], f"a leg region became a real attachment: {leg_attachments}"


def test_coffee_mug_still_detects_the_real_hand_grip():
    """The regression fix above must not come at the cost of the real grip
    -- a passing check here isn't meaningful without this one too."""
    fixture = _load_fixture("testCoffeeMug")
    event = _covering_event(_events_for_region(fixture, "right_hand"), frame=380)
    assert event.start_frame <= 310 and event.end_frame >= 460


def test_pick_up_put_down_mug_all_four_grips_have_no_internal_gap():
    """Regression: the same translation correction fragmented one continuous
    grip (frames 149-254) into two events with a gap at 215-227. Checks all
    4 real pickups, not just the one that broke -- each should be exactly
    one continuous event, covering its own known real span."""
    fixture = _load_fixture("testPickUpPutDownMug")
    events = _events_for_region(fixture, "right_hand")

    known_grips = [(14, 111), (149, 254), (304, 409), (446, 543)]
    for start, end in known_grips:
        mid = (start + end) // 2
        event = _covering_event(events, frame=mid)
        assert event.start_frame <= start + 10, f"grip near {start}-{end}: starts too late ({event.start_frame})"
        assert event.end_frame >= end - 10, f"grip near {start}-{end}: ends too early ({event.end_frame})"


def test_pass_mug_between_hands_has_continuous_hand_contact_throughout():
    """The object is held throughout this clip (passed hand-to-hand 5 times,
    never set down) -- real contact should cover virtually the whole clip
    with no meaningful gap. Raw translation alone doesn't bridge every brief
    real confidence dip during a hand-off (this test failed without
    `bridge_short_gaps` -- max gap was 14 frames); `bridge_short_gaps` closes
    those gaps at the stage 7 level, which this test now exercises directly."""
    fixture = _load_fixture("testPassMugBetweenHands")
    n_frames = int(fixture["n_frames"])
    fps = float(fixture["fps"])

    events: list[ContactEvent] = []
    for region in ("left_hand", "right_hand"):
        events.extend(_events_for_region(fixture, region))
    events = bridge_short_gaps(events, fps)

    covered = np.zeros(n_frames, dtype=bool)
    for event in events:
        covered[event.start_frame:event.end_frame + 1] = True

    gaps = contiguous_true_runs(~covered)
    max_gap = max((end - start + 1 for start, end in gaps), default=0)
    assert max_gap <= 10, f"largest gap in hand-object contact is {max_gap} frames: {gaps}"
