"""Stage 1 regression test: runs real SAM 3.1 tracking on the small test clip
and checks the tracked human/object masks look correct -- plausible per-frame
area, not empty, not the whole frame, and mask previews written at the
video's native resolution. Needs the real SAM 3.1 checkpoint and a CUDA GPU
-- skipped automatically otherwise (see conftest.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch

from pipeline.adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from conftest import TEST_VIDEO_FRAME_COUNT, TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH

# SAM 3.1's own fixed internal mask resolution (see gvhmr_adapter.py's docstring) --
# area fractions below are computed against this, not the video's native resolution.
SAM31_WORKING_SIZE = 1008

# Generous bounds around the real areas measured on this exact clip (human ~6.4-7%,
# object ~0.6-0.9%) -- wide enough to tolerate minor model/library version drift,
# tight enough to catch a genuinely broken or empty mask.
MIN_HUMAN_AREA_FRACTION = 0.02
MAX_HUMAN_AREA_FRACTION = 0.20
MIN_OBJECT_AREA_FRACTION = 0.001
MAX_OBJECT_AREA_FRACTION = 0.05


def _area_fractions(packed_masks: torch.Tensor) -> torch.Tensor:
    masks = unpack_masks(packed_masks).squeeze(1)
    areas = masks.sum(dim=(-1, -2)).float()
    return areas / (SAM31_WORKING_SIZE * SAM31_WORKING_SIZE)


def test_human_tracked_in_every_frame_with_plausible_area(stage_1_result):
    human = torch.load(Path(stage_1_result["human_masks"]), weights_only=False)
    assert human[KEY_PACKED_MASKS] is not None
    assert human[KEY_PACKED_MASKS].shape[0] == TEST_VIDEO_FRAME_COUNT

    fractions = _area_fractions(human[KEY_PACKED_MASKS])
    assert (fractions > MIN_HUMAN_AREA_FRACTION).all(), fractions.tolist()
    assert (fractions < MAX_HUMAN_AREA_FRACTION).all(), fractions.tolist()


def test_object_tracked_in_every_frame_with_plausible_area(stage_1_result):
    assert "object_masks" in stage_1_result, "the tennis racket should be tracked throughout this hand-picked clip"
    obj = torch.load(Path(stage_1_result["object_masks"]), weights_only=False)
    assert obj[KEY_PACKED_MASKS] is not None

    fractions = _area_fractions(obj[KEY_PACKED_MASKS])
    assert (fractions > MIN_OBJECT_AREA_FRACTION).all(), fractions.tolist()
    assert (fractions < MAX_OBJECT_AREA_FRACTION).all(), fractions.tolist()


def test_anchor_frame_resolved_within_bounds(progress, stage_1_result):
    assert 0 <= progress.scene.anchor_frame_index < TEST_VIDEO_FRAME_COUNT


def test_mask_previews_written_at_native_resolution(stage_1_result):
    for key in ("human_masks_preview", "object_masks_preview"):
        preview_dir = Path(stage_1_result[key])
        preview_paths = sorted(preview_dir.glob("*.jpg"))
        assert len(preview_paths) == TEST_VIDEO_FRAME_COUNT

        image = cv2.imread(str(preview_paths[0]))
        assert image is not None
        assert image.shape[:2] == (TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH)
