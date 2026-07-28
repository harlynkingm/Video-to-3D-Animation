"""Stage 7 tests: a real-data regression test (needs GPU/checkpoints, skipped
otherwise -- see conftest.py) plus a synthetic unit test for the contacts
preview renderer, which has no model/GPU dependency of its own and so always
runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pipeline.adapters.depth_anything3_adapter import KEY_DEPTH
from pipeline.adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, pack_masks
from pipeline.algorithms.contact_detection import REGION_NAMES, ContactEvent
from pipeline.stages.stage_0_ingest_video import INPUT_FRAMES_DIRNAME
from pipeline.stages import stage_7_annotate_contacts
from pipeline.stages.stage_7_annotate_contacts import (
    DEPTH_GAP_OCCLUSION_THRESHOLD_M,
    LOW_CONFIDENCE_THRESHOLD,
    OUTPUT_CONTACTS_PREVIEW,
    _LazyMaskLoader,
    _event_to_dict,
    _render_contacts_preview,
    _verify_events_with_depth,
)


def test_contact_events_output_is_plausible(stage_7_result):
    events = json.loads(open(stage_7_result["contact_events"]).read())

    # The test clip tracks a tennis racket in the player's grip throughout --
    # a real, sustained contact should show up as at least one event.
    assert len(events) >= 1

    for event in events:
        assert event["regions"] and all(r in REGION_NAMES for r in event["regions"])
        assert 0 <= event["start_frame"] <= event["peak_frame"] <= event["end_frame"]
        assert 0.0 <= event["mean_confidence"] <= 1.0
        assert isinstance(event["is_low_confidence"], bool)
        # A gap over threshold means confidently-occlusion -- _verify_events_with_depth
        # drops those outright, so nothing surviving into the output should have one.
        assert event["depth_gap_m"] is None or 0.0 <= event["depth_gap_m"] <= DEPTH_GAP_OCCLUSION_THRESHOLD_M

    # conftest.py's shared `progress` fixture turns render_contacts_preview on,
    # so this real run should have written exactly one JPEG per event.
    preview_dir = Path(stage_7_result[OUTPUT_CONTACTS_PREVIEW])
    written = list(preview_dir.glob("*.jpg"))
    assert len(written) == len(events)


def test_render_contacts_preview_writes_one_labeled_jpeg_per_event(tmp_path):
    frames_dir = tmp_path / INPUT_FRAMES_DIRNAME
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "000003.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))

    K = np.array([[10.0, 0, 10.0], [0, 10.0, 10.0], [0, 0, 1.0]])
    joints = np.zeros((5, 22, 3))
    joints[3, 20] = [0.0, 0.0, 1.0]  # left wrist (joint 20) projects to (10, 10) via K

    event = ContactEvent(
        regions=["left_arm"], joint="wrist", start_frame=1, end_frame=5, peak_frame=3, mean_confidence=0.9,
    )

    out_dir = tmp_path / "preview"
    _render_contacts_preview([event], joints, K, frames_dir, out_dir)

    written = list(out_dir.iterdir())
    assert len(written) == 1
    assert written[0].name == "000003_left_arm.jpg"

    image = cv2.imread(str(written[0]))
    assert image is not None
    assert image.any()  # the circle/label actually drew onto the all-black frame


def test_render_contacts_preview_skips_an_event_with_no_source_frame(tmp_path):
    """A frame that was never extracted (shouldn't happen, but the renderer
    shouldn't crash the stage over a missing preview image)."""
    frames_dir = tmp_path / INPUT_FRAMES_DIRNAME
    frames_dir.mkdir()

    K = np.eye(3)
    joints = np.zeros((5, 22, 3))
    joints[..., 2] = 1.0  # realistic camera-space depth, avoids a divide-by-zero in projection
    event = ContactEvent(
        regions=["head"], joint="head", start_frame=0, end_frame=2, peak_frame=1, mean_confidence=0.9,
    )

    out_dir = tmp_path / "preview"
    _render_contacts_preview([event], joints, K, frames_dir, out_dir)

    assert list(out_dir.iterdir()) == []


def test_render_contacts_preview_clears_stale_images_from_a_previous_run(tmp_path):
    """A re-run can produce fewer events than a previous one (e.g. a
    detection-logic change), so a stale image left over from a now-removed
    event must not survive alongside the current run's own output."""
    frames_dir = tmp_path / INPUT_FRAMES_DIRNAME
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "000003.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))

    out_dir = tmp_path / "preview"
    out_dir.mkdir()
    stale_file = out_dir / "000099_chest.jpg"
    stale_file.write_bytes(b"stale")

    K = np.array([[10.0, 0, 10.0], [0, 10.0, 10.0], [0, 0, 1.0]])
    joints = np.zeros((5, 22, 3))
    joints[3, 20] = [0.0, 0.0, 1.0]
    event = ContactEvent(
        regions=["left_arm"], joint="wrist", start_frame=1, end_frame=5, peak_frame=3, mean_confidence=0.9,
    )

    _render_contacts_preview([event], joints, K, frames_dir, out_dir)

    written = list(out_dir.iterdir())
    assert not stale_file.exists()
    assert [f.name for f in written] == ["000003_left_arm.jpg"]


def test_render_contacts_preview_joins_a_consolidated_event_regions_for_the_filename(tmp_path):
    frames_dir = tmp_path / INPUT_FRAMES_DIRNAME
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "000007.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))

    K = np.array([[10.0, 0, 10.0], [0, 10.0, 10.0], [0, 0, 1.0]])
    joints = np.zeros((10, 22, 3))
    joints[7, 20] = [0.0, 0.0, 1.0]

    event = ContactEvent(
        regions=["left_hand", "left_arm"], joint="wrist", start_frame=1, end_frame=9, peak_frame=7,
        mean_confidence=0.9,
    )

    out_dir = tmp_path / "preview"
    _render_contacts_preview([event], joints, K, frames_dir, out_dir)

    written = list(out_dir.iterdir())
    assert len(written) == 1
    assert written[0].name == "000007_left_hand+left_arm.jpg"


def test_event_to_dict_sets_is_low_confidence_below_threshold():
    event = ContactEvent(
        regions=["left_hand"], joint="wrist", start_frame=0, end_frame=5, peak_frame=2,
        mean_confidence=LOW_CONFIDENCE_THRESHOLD - 0.01,
    )
    assert _event_to_dict(event)["is_low_confidence"] is True


def test_event_to_dict_is_not_low_confidence_when_confident_and_depth_verified():
    event = ContactEvent(
        regions=["left_hand"], joint="wrist", start_frame=0, end_frame=5, peak_frame=2,
        mean_confidence=1.0, depth_gap_m=0.01,
    )
    assert _event_to_dict(event)["is_low_confidence"] is False


def test_event_to_dict_is_not_low_confidence_when_depth_confirms_contact_despite_weak_2d_confidence():
    """Real case from the coffee-mug clip that motivated this: a hand-grip
    event with weak 2D confidence (0.49, below LOW_CONFIDENCE_THRESHOLD)
    but a tiny depth gap (0.01m). Depth is the stronger, physically-grounded
    signal here and should override a merely-noisy 2D confidence score."""
    event = ContactEvent(
        regions=["right_hand"], joint="index_tip", start_frame=309, end_frame=319, peak_frame=312,
        mean_confidence=0.49, depth_gap_m=0.01,
    )
    assert _event_to_dict(event)["is_low_confidence"] is False


class _FakeDepthAdapter:
    """Stands in for DepthAnything3Adapter -- returns a fixed, caller-supplied
    depth map regardless of which frame is asked for, so the filtering logic
    in `_verify_events_with_depth` can be tested without a GPU/checkpoint."""

    def __init__(self, depth: np.ndarray):
        self._depth = depth

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def infer(self, frame_path: str, focal_length_px: float) -> dict:
        return {KEY_DEPTH: self._depth}


def test_verify_events_with_depth_drops_occlusion_keeps_real_contact(tmp_path, monkeypatch):
    """Regression test for the actual coffee-mug bug: a chest/leg event whose
    object mask merely overlapped the body in 2D, with no real proximity in
    depth, must be dropped outright rather than kept around flagged as
    uncertain -- while a real hand-grip event (small depth gap) survives."""
    native_hw = (20, 20)
    depth = np.ones(native_hw, dtype=np.float32)
    depth[0:3, 0:3] = 1.0    # frame 0's object region
    depth[0:3, 5:8] = 1.02   # frame 0's human region -- 0.02m gap, real contact
    depth[10:13, 0:3] = 1.0  # frame 1's object region
    depth[10:13, 5:8] = 2.0  # frame 1's human region -- 1.0m gap, incidental occlusion
    monkeypatch.setattr(stage_7_annotate_contacts, "DepthAnything3Adapter", lambda: _FakeDepthAdapter(depth))

    object_mask_0 = np.zeros(native_hw, dtype=bool); object_mask_0[0:3, 0:3] = True
    human_mask_0 = np.zeros(native_hw, dtype=bool); human_mask_0[0:3, 5:8] = True
    object_mask_1 = np.zeros(native_hw, dtype=bool); object_mask_1[10:13, 0:3] = True
    human_mask_1 = np.zeros(native_hw, dtype=bool); human_mask_1[10:13, 5:8] = True

    K = np.array([[5.0, 0.0, 10.0], [0.0, 5.0, 10.0], [0.0, 0.0, 1.0]])
    joints = np.zeros((2, 25, 3))
    joints[0, 20] = [0.0, 0.0, 1.0]  # left_hand's wrist candidate, frame 0
    joints[1, 6] = [0.0, 0.0, 1.0]   # chest's spine2 candidate, frame 1

    real_contact = ContactEvent(
        regions=["left_hand"], joint="wrist", start_frame=0, end_frame=0, peak_frame=0, mean_confidence=0.9,
    )
    occlusion = ContactEvent(
        regions=["chest"], joint="spine2", start_frame=1, end_frame=1, peak_frame=1, mean_confidence=0.9,
    )

    survivors = _verify_events_with_depth(
        [real_contact, occlusion], joints, K,
        [object_mask_0, object_mask_1], [human_mask_0, human_mask_1],
        tmp_path, native_hw,
    )

    assert len(survivors) == 1
    assert survivors[0].regions == ["left_hand"]
    assert survivors[0].depth_gap_m is not None and survivors[0].depth_gap_m <= DEPTH_GAP_OCCLUSION_THRESHOLD_M


def test_verify_events_with_depth_keeps_an_event_it_cannot_verify(tmp_path, monkeypatch):
    """A frame with no tracked object/human mask can't be depth-verified
    either way -- the event should survive unverified (depth_gap_m stays
    None), not be dropped for lack of evidence."""
    native_hw = (20, 20)
    monkeypatch.setattr(
        stage_7_annotate_contacts, "DepthAnything3Adapter",
        lambda: _FakeDepthAdapter(np.ones(native_hw, dtype=np.float32)),
    )

    K = np.eye(3)
    joints = np.zeros((1, 25, 3))
    joints[0, 20] = [0.0, 0.0, 1.0]
    event = ContactEvent(
        regions=["left_hand"], joint="wrist", start_frame=0, end_frame=0, peak_frame=0, mean_confidence=0.9,
    )

    survivors = _verify_events_with_depth([event], joints, K, [None], [None], tmp_path, native_hw)

    assert survivors == [event]
    assert event.depth_gap_m is None


def _save_packed_masks(path: Path, raw_masks: torch.Tensor) -> None:
    """`raw_masks`: (n_frames, 1, H, W) bool, W divisible by 8 -- same shape
    stage 1 itself writes, packed the same way (`pack_masks`)."""
    torch.save({KEY_PACKED_MASKS: pack_masks(raw_masks)}, path)


def test_lazy_mask_loader_decodes_and_resizes_each_frame_on_access(tmp_path):
    """Regression test for a real crash: the old eager loader unpacked and
    native-resized every frame up front, holding the whole clip's masks in
    memory at once -- on a real 4K/675-frame clip (two such lists: object +
    human) that was 10+GB simultaneously, and caused a hard Windows access
    violation later in this same stage. This loader must decode lazily,
    per-frame, instead.
    """
    working_h, working_w = 8, 16  # W must be divisible by 8 for pack_masks
    raw_masks = torch.zeros((3, 1, working_h, working_w), dtype=torch.bool)
    raw_masks[0, 0, 2:5, 4:10] = True  # frame 0: a real mask
    # frame 1 left all-False -- nothing tracked that frame
    raw_masks[2, 0, 0:3, 0:3] = True  # frame 2: another real mask

    masks_path = tmp_path / "masks.pt"
    _save_packed_masks(masks_path, raw_masks)

    native_hw = (32, 64)  # 4x upsample in both dims, like resizing to native resolution
    loader = _LazyMaskLoader(str(masks_path), n_frames=3, native_hw=native_hw)

    assert len(loader) == 3

    frame0 = loader[0]
    assert frame0 is not None
    assert frame0.shape == native_hw
    assert frame0.dtype == np.bool_
    assert frame0.any()

    assert loader[1] is None  # nothing tracked that frame

    frame2 = loader[2]
    assert frame2 is not None
    assert frame2.any()


def test_lazy_mask_loader_returns_none_past_the_end_of_the_packed_tensor(tmp_path):
    """The clip's masks can end before the retargeted motion does (an
    occlusion gap running to the end of the clip) -- indices beyond the
    packed tensor's own frame count should return None, not raise."""
    raw_masks = torch.zeros((2, 1, 8, 16), dtype=torch.bool)  # only 2 frames of masks
    raw_masks[0, 0, 0:2, 0:2] = True
    masks_path = tmp_path / "masks.pt"
    _save_packed_masks(masks_path, raw_masks)

    loader = _LazyMaskLoader(str(masks_path), n_frames=5, native_hw=(16, 32))

    assert loader[0] is not None
    assert loader[4] is None
