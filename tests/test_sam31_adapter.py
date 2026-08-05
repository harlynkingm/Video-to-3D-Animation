"""Unit tests for the pure-Python parts of `sam31_adapter.py` -- this whole
module had zero test coverage before (written before the project had a test
suite): checkpoint key-prefix splitting, frame loading/color conversion, the
multi-slot stitching fix, and `infer()`'s own prompt-routing control flow.
Nothing here needs the real checkpoint or a GPU.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from pipeline.adapters.sam31.sam31_adapter import (
    KEY_DETECTOR,
    KEY_HUMAN,
    KEY_OBJECT,
    KEY_PACKED_MASKS,
    KEY_SCORE,
    KEY_TEXT_TOWER,
    KEY_TRACKER,
    KEY_VISION_BACKBONE,
    Sam31Adapter,
    _LazyFrameLoader,
    _load_checkpoint_state,
    _stitch_tracked_slots,
)
from pipeline.adapters.sam31.sam31_tracker import KEY_N_FRAMES, pack_masks, unpack_masks
from pipeline.progress_tracker import StageName

H, W = 4, 8


def _slot_pattern(obj_idx: int) -> torch.Tensor:
    m = torch.zeros(H, W, dtype=torch.bool)
    m[obj_idx % H] = True
    return m


def _make_packed(slot_frame_active: list[list[bool]]) -> torch.Tensor:
    """slot_frame_active[obj_idx][frame_idx] -> whether that slot has its own
    distinguishing pattern active at that frame (all-zero otherwise)."""
    n_obj = len(slot_frame_active)
    n_frames = len(slot_frame_active[0])
    masks = torch.zeros(n_frames, n_obj, H, W, dtype=torch.bool)
    for obj_idx, frames in enumerate(slot_frame_active):
        pattern = _slot_pattern(obj_idx)
        for frame_idx, active in enumerate(frames):
            if active:
                masks[frame_idx, obj_idx] = pattern
    return pack_masks(masks)


def _merged_frame(merged: torch.Tensor, frame_idx: int) -> torch.Tensor:
    return unpack_masks(merged[frame_idx, 0])


def test_single_slot_no_gaps_passes_through_unchanged():
    packed = _make_packed([[True, True, True]])
    merged, best_score = _stitch_tracked_slots(packed, scores=[0.9], max_bridge_frames=2)
    for t in range(3):
        assert torch.equal(_merged_frame(merged, t), _slot_pattern(0))
    assert best_score == 0.9


def test_uses_whichever_slot_is_active_even_if_a_different_slot_scores_higher():
    # Slot 0 (score 0.93) only covers frames 0-1; slot 1 (score 0.68, LOWER)
    # covers frames 2-4. The old "keep the single best-scoring slot" behavior
    # would drop slot 1 entirely -- this is the actual basketball-clip bug.
    packed = _make_packed([
        [True, True, False, False, False],
        [False, False, True, True, True],
    ])
    merged, best_score = _stitch_tracked_slots(packed, scores=[0.93, 0.68], max_bridge_frames=0)
    assert torch.equal(_merged_frame(merged, 0), _slot_pattern(0))
    assert torch.equal(_merged_frame(merged, 1), _slot_pattern(0))
    assert torch.equal(_merged_frame(merged, 2), _slot_pattern(1))
    assert torch.equal(_merged_frame(merged, 3), _slot_pattern(1))
    assert torch.equal(_merged_frame(merged, 4), _slot_pattern(1))
    assert best_score == 0.93  # highest score among CONTRIBUTING slots, both did here


def test_prefers_higher_scoring_slot_when_both_active_the_same_frame():
    packed = _make_packed([
        [True, True],
        [False, True],
    ])
    merged, _ = _stitch_tracked_slots(packed, scores=[0.5, 0.9], max_bridge_frames=0)
    assert torch.equal(_merged_frame(merged, 0), _slot_pattern(0))  # only slot 0 active
    assert torch.equal(_merged_frame(merged, 1), _slot_pattern(1))  # both active, slot 1 scores higher


def test_bridges_a_short_interior_gap_by_holding_the_previous_winner():
    # slot 0 active 0-1, gap frames 2-3 (nothing active), slot 1 active 4-5.
    packed = _make_packed([
        [True, True, False, False, False, False],
        [False, False, False, False, True, True],
    ])
    merged, _ = _stitch_tracked_slots(packed, scores=[0.9, 0.6], max_bridge_frames=2)
    assert torch.equal(_merged_frame(merged, 2), _slot_pattern(0))  # held from slot 0, not slot 1
    assert torch.equal(_merged_frame(merged, 3), _slot_pattern(0))
    assert torch.equal(_merged_frame(merged, 4), _slot_pattern(1))


def test_leaves_a_gap_longer_than_max_bridge_frames_empty():
    packed = _make_packed([
        [True, True, False, False, False, True, True],
    ])
    merged, _ = _stitch_tracked_slots(packed, scores=[0.9], max_bridge_frames=2)
    assert torch.equal(_merged_frame(merged, 2), torch.zeros(H, W, dtype=torch.bool))
    assert torch.equal(_merged_frame(merged, 3), torch.zeros(H, W, dtype=torch.bool))
    assert torch.equal(_merged_frame(merged, 4), torch.zeros(H, W, dtype=torch.bool))


def test_leaves_leading_and_trailing_gaps_empty_regardless_of_bridge_setting():
    # Never detected until frame 2, and gone again after frame 3 -- no valid
    # mask exists on one side of either gap to hold from.
    packed = _make_packed([
        [False, False, True, True, False, False],
    ])
    merged, _ = _stitch_tracked_slots(packed, scores=[0.9], max_bridge_frames=10)
    assert torch.equal(_merged_frame(merged, 0), torch.zeros(H, W, dtype=torch.bool))
    assert torch.equal(_merged_frame(merged, 1), torch.zeros(H, W, dtype=torch.bool))
    assert torch.equal(_merged_frame(merged, 4), torch.zeros(H, W, dtype=torch.bool))
    assert torch.equal(_merged_frame(merged, 5), torch.zeros(H, W, dtype=torch.bool))


def test_best_score_ignores_a_slot_that_was_never_actually_active():
    # A high-scoring slot with an all-empty mask (e.g. a phantom detection
    # that never actually produced a mask) must never win, and must not
    # inflate best_score.
    packed = _make_packed([
        [True, True],
        [False, False],
    ])
    merged, best_score = _stitch_tracked_slots(packed, scores=[0.5, 0.99], max_bridge_frames=0)
    assert torch.equal(_merged_frame(merged, 0), _slot_pattern(0))
    assert torch.equal(_merged_frame(merged, 1), _slot_pattern(0))
    assert best_score == 0.5


# --- _load_checkpoint_state: pure key-prefix splitting, no real checkpoint needed ---


def test_load_checkpoint_state_splits_and_strips_prefixes(tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.safetensors"
    tensors = {
        "detector.backbone.vision_backbone.patch_embed.weight": torch.tensor([1.0]),
        "detector.backbone.language_backbone.encoder.layer0.weight": torch.tensor([2.0]),
        "detector.backbone.language_backbone.resizer.weight": torch.tensor([3.0]),
        "detector.mask_decoder.weight": torch.tensor([4.0]),
        "tracker.model.memory_encoder.weight": torch.tensor([5.0]),
        # Under detector.backbone.* but not vision_backbone/language_backbone --
        # the detector branch explicitly excludes anything under that prefix.
        "detector.backbone.some_other_submodule.weight": torch.tensor([6.0]),
        # Matches no recognized prefix at all -- silently dropped.
        "discriminator.unused.weight": torch.tensor([7.0]),
    }
    save_file(tensors, str(ckpt_path))

    state = _load_checkpoint_state(ckpt_path)

    assert set(state[KEY_VISION_BACKBONE].keys()) == {"patch_embed.weight"}
    assert set(state[KEY_TEXT_TOWER].keys()) == {"layer0.weight"}
    assert set(state[KEY_DETECTOR].keys()) == {"mask_decoder.weight", "text_resizer.weight"}
    assert state[KEY_DETECTOR]["text_resizer.weight"].item() == 3.0
    assert set(state[KEY_TRACKER].keys()) == {"memory_encoder.weight"}

    kept_values = {v.item() for d in state.values() for v in d.values()}
    assert kept_values == {1.0, 2.0, 3.0, 4.0, 5.0}  # 6.0 and 7.0 must land nowhere


# --- _LazyFrameLoader: BGR->RGB conversion, normalization, lazy slicing ---


def _write_solid_bgr_image(path: Path, bgr: tuple[int, int, int], size: int = 4) -> None:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr
    cv2.imwrite(str(path), img)  # .png -> lossless, exact round-trip


def test_lazy_frame_loader_converts_bgr_to_rgb_and_normalizes(tmp_path: Path):
    path = tmp_path / "000000.png"
    _write_solid_bgr_image(path, bgr=(255, 0, 0))  # pure blue in cv2's native BGR order
    loader = _LazyFrameLoader([path])

    assert loader.shape == (1,)
    frames = loader[0:1]
    assert frames.shape == (1, 3, 4, 4)
    assert frames.dtype == torch.float32
    # After BGR->RGB, pure blue is R=0, G=0, B=1 (normalized to [0, 1]).
    assert torch.allclose(frames[0, 0], torch.zeros(4, 4))
    assert torch.allclose(frames[0, 1], torch.zeros(4, 4))
    assert torch.allclose(frames[0, 2], torch.ones(4, 4))


def test_lazy_frame_loader_slicing_preserves_frame_order(tmp_path: Path):
    colors_bgr = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # blue, green, red frames
    paths = []
    for i, bgr in enumerate(colors_bgr):
        p = tmp_path / f"{i:06d}.png"
        _write_solid_bgr_image(p, bgr)
        paths.append(p)
    loader = _LazyFrameLoader(paths)
    assert loader.shape == (3,)

    subset = loader[1:3]  # green frame, then red frame
    assert subset.shape == (2, 3, 4, 4)
    assert torch.allclose(subset[0, 1], torch.ones(4, 4))   # green frame -> G channel high
    assert torch.allclose(subset[0, 0], torch.zeros(4, 4))  # green frame -> R channel low
    assert torch.allclose(subset[1, 0], torch.ones(4, 4))   # red frame -> R channel high


def test_lazy_frame_loader_raises_on_unreadable_file(tmp_path: Path):
    missing = tmp_path / "does_not_exist.png"
    loader = _LazyFrameLoader([missing])
    with pytest.raises(RuntimeError, match="Could not read frame"):
        loader[0:1]


# --- infer(): prompt-routing control flow, `_track_one_prompt` stubbed out
# so this needs no real model weights ---


def _stub_track_one_prompt(calls: list):
    def fake(self, images, prompt, progress_label, max_bridge_frames):
        calls.append((prompt, progress_label, max_bridge_frames))
        return {KEY_PACKED_MASKS: None, KEY_N_FRAMES: 5, KEY_SCORE: None}
    return fake


def test_infer_skips_object_tracking_when_no_object_prompt(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(Sam31Adapter, "_track_one_prompt", _stub_track_one_prompt(calls))
    adapter = Sam31Adapter()

    result = adapter.infer([tmp_path / "a.png"], human_prompt="a person", object_prompt=None, max_bridge_frames=7)

    assert result[KEY_OBJECT] is None
    assert calls == [("a person", StageName.STAGE_1A_HUMAN_MASK.label, 7)]


def test_infer_tracks_both_prompts_with_the_same_bridge_setting(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(Sam31Adapter, "_track_one_prompt", _stub_track_one_prompt(calls))
    adapter = Sam31Adapter()

    result = adapter.infer(
        [tmp_path / "a.png"], human_prompt="a person", object_prompt="a basketball", max_bridge_frames=12
    )

    assert result[KEY_HUMAN][KEY_PACKED_MASKS] is None  # stub's return value, just checking wiring
    assert calls == [
        ("a person", StageName.STAGE_1A_HUMAN_MASK.label, 12),
        ("a basketball", StageName.STAGE_1B_OBJECT_MASK.label, 12),
    ]
