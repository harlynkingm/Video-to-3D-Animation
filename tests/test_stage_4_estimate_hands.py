"""Stage 4 regression test: runs real HaMeR on the test clip and checks the
per-frame MANO hand pose looks correct, right shapes, no NaN, physically
plausible finger rotations, both hands detected on this clearly-two-handed
tennis clip, and smooth frame-to-frame motion (a broken crop/flip would produce
jumpy garbage). Needs the HaMeR + ViTPose checkpoints and a CUDA GPU, skipped
automatically otherwise (see conftest.py).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from pipeline.adapters.hamer import hamer_adapter
from pipeline.adapters.hamer.hamer_adapter import (
    HamerAdapter,
    KEY_LEFT_GLOBAL_ORIENT,
    KEY_LEFT_HAND_POSE,
    KEY_LEFT_VALID,
    KEY_RIGHT_GLOBAL_ORIENT,
    KEY_RIGHT_HAND_POSE,
    KEY_RIGHT_VALID,
    MIN_ROLLING_WRIST_CONFIDENCE,
    _hand_crop_has_object_contact,
    _hand_crop_is_ambiguous,
    _reject_low_confidence_stretches,
)
from pipeline.adapters.hamer.hamer_preprocess import COCO_L_ELBOW, COCO_L_WRIST, COCO_R_ELBOW, COCO_R_WRIST
from pipeline.adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, pack_masks
from pipeline.stages.stage_4_estimate_hands import (
    FINGER_AMBIGUITY_BETA_SCALE,
    FINGER_AMBIGUITY_RECOVERY_FRAMES,
    FINGER_OBJECT_CONTACT_BETA_SCALE,
    FINGER_MOTION_SETTINGS,
    _finger_ambiguity_beta_scale,
    _finger_object_contact_beta_scale,
    _finger_validity_after_sustained_wrist_failure,
    _finger_motion_settings,
    _smooth_hand_channel,
)
from pipeline.progress_tracker import FingerMotion
from conftest import TEST_VIDEO_FRAME_COUNT

MAX_PLAUSIBLE_FINGER_ROTATION_RAD = 3.15  # any single axis-angle rotation maxes out at pi
MAX_PLAUSIBLE_FRAME_DELTA = 2.0  # generous bound on frame-to-frame full-hand pose change


def test_sustained_wrist_failure_holds_finger_input_but_a_brief_one_does_not():
    finger_valid = np.ones(20, bool)
    wrist_valid = np.ones(20, bool)
    wrist_valid[2:5] = False  # isolated/brief rejection: fingers remain usable
    wrist_valid[8:15] = False  # sustained failure: treat fingers as occluded

    filtered = _finger_validity_after_sustained_wrist_failure(finger_valid, wrist_valid)

    assert filtered[2:5].all()
    assert not filtered[8:15].any()


def test_crop_ambiguity_is_a_soft_smoothing_cue_not_a_finger_hold():
    finger_valid = np.ones(12, bool)
    wrist_valid = np.ones(12, bool)
    ambiguous = np.zeros(12, bool)
    ambiguous[3:9] = True

    filtered = _finger_validity_after_sustained_wrist_failure(finger_valid, wrist_valid, ambiguous)

    assert filtered.all()


def test_hand_crop_ambiguity_detects_nearby_hands_but_not_a_visible_object_grip():
    hand = np.array([40.0, 40.0, 80.0, 80.0])
    wrist = np.array([60.0, 60.0, 0.9])
    object_mask = np.zeros((120, 120), bool)
    object_mask[50:70, 50:70] = True

    assert _hand_crop_has_object_contact(hand, object_mask)
    assert not _hand_crop_is_ambiguous(hand, None, wrist, None)
    assert _hand_crop_is_ambiguous(hand, np.array([50.0, 50.0, 90.0, 90.0]), wrist, wrist)


def test_hand_crop_ambiguity_leaves_a_clear_crop_usable():
    hand = np.array([40.0, 40.0, 80.0, 80.0])
    other_hand = np.array([50.0, 50.0, 90.0, 90.0])
    wrist = np.array([60.0, 60.0, 0.9])
    other_wrist = np.array([110.0, 60.0, 0.9])

    assert not _hand_crop_is_ambiguous(hand, other_hand, wrist, other_wrist)


def test_ambiguous_hand_crop_gets_a_smooth_reacquisition_ramp():
    ambiguous = np.zeros(50, bool)
    ambiguous[10:15] = True

    scale = _finger_ambiguity_beta_scale(ambiguous)

    assert np.all(scale[10:15] == FINGER_AMBIGUITY_BETA_SCALE)
    assert FINGER_AMBIGUITY_BETA_SCALE < scale[15] < 1.0
    assert np.all(np.diff(scale[15:15 + FINGER_AMBIGUITY_RECOVERY_FRAMES]) > 0.0)
    assert scale[14 + FINGER_AMBIGUITY_RECOVERY_FRAMES] == pytest.approx(1.0)
    assert np.all(scale[15 + FINGER_AMBIGUITY_RECOVERY_FRAMES:] == 1.0)


def test_object_contact_only_modestly_reduces_finger_bandwidth_without_holding():
    scale = _finger_object_contact_beta_scale(np.array([False, True, True] + [False] * 27))

    assert np.array_equal(scale[:3], [1.0, FINGER_OBJECT_CONTACT_BETA_SCALE, FINGER_OBJECT_CONTACT_BETA_SCALE])
    assert FINGER_OBJECT_CONTACT_BETA_SCALE < scale[3] < 1.0


def test_detailed_profile_preserves_a_short_small_finger_bend():
    """A detailed finger movement can be just a few frames of low-amplitude finger motion.

    The old shared 15-frame pre-pass erases that signal before One Euro gets a
    speed measurement from it. The finger profile must retain it while the
    wrist can continue using that stronger stabilization independently.
    """
    sequence = np.zeros((60, 3), np.float32)
    sequence[28:31, 0] = 0.35  # 20 degrees for three frames, a light keypress
    valid = np.ones(60, bool)

    legacy = _smooth_hand_channel(sequence, valid, 30.0, 15, 0.15, 0.3, 1.5)
    detailed = _smooth_hand_channel(sequence, valid, 30.0, 5, 0.225, 1.85, 0.375, 2.75)

    assert legacy[:, 0].max() < 0.10  # demonstrates the regression we are avoiding
    assert detailed[:, 0].max() > 0.28


def test_finger_motion_mode_selects_a_profile_before_applying_numeric_pins():
    smooth = _finger_motion_settings(SimpleNamespace(
        input=SimpleNamespace(finger_motion=FingerMotion.SMOOTH), fine_tuning_overrides={},
    ))
    detailed = _finger_motion_settings(SimpleNamespace(
        input=SimpleNamespace(finger_motion=FingerMotion.DETAILED), fine_tuning_overrides={},
    ))
    pinned = _finger_motion_settings(SimpleNamespace(
        input=SimpleNamespace(finger_motion=FingerMotion.DETAILED),
        fine_tuning_overrides={"hand_finger_beta": 1.25, "hand_wrist_min_cutoff_hz": 0.08},
    ))

    assert smooth == FINGER_MOTION_SETTINGS[FingerMotion.SMOOTH]
    assert detailed == FINGER_MOTION_SETTINGS[FingerMotion.DETAILED]
    assert smooth.hand_finger_smoothing_window > detailed.hand_finger_smoothing_window
    assert pinned.hand_finger_beta == 1.25
    assert pinned.hand_finger_min_cutoff_hz == detailed.hand_finger_min_cutoff_hz


def test_smooth_profile_strictly_prioritizes_resting_finger_stability():
    """The default profile is the no-visible-jitter path, not a compromise."""
    rng = np.random.default_rng(11)
    resting = rng.normal(0, 0.035, (120, 3)).astype(np.float32)
    valid = np.ones(120, bool)
    smooth_settings = FINGER_MOTION_SETTINGS[FingerMotion.SMOOTH]
    detailed_settings = FINGER_MOTION_SETTINGS[FingerMotion.DETAILED]

    smooth = _smooth_hand_channel(
        resting, valid, 30.0,
        smooth_settings.hand_finger_smoothing_window,
        smooth_settings.hand_finger_min_cutoff_hz,
        smooth_settings.hand_finger_beta,
        smooth_settings.hand_finger_decimate_deg,
        smooth_settings.hand_finger_derivative_cutoff_hz,
        suppress_transient_reversals=True,
    )
    detailed = _smooth_hand_channel(
        resting, valid, 30.0,
        detailed_settings.hand_finger_smoothing_window,
        detailed_settings.hand_finger_min_cutoff_hz,
        detailed_settings.hand_finger_beta,
        detailed_settings.hand_finger_decimate_deg,
        detailed_settings.hand_finger_derivative_cutoff_hz,
        suppress_transient_reversals=True,
    )
    jitter = lambda values: float(np.abs(np.diff(values, n=2, axis=0)).mean())

    assert jitter(smooth) < 0.25 * jitter(detailed)


def test_detailed_profile_reduces_resting_frame_jitter():
    """The detail-preserving finger profile must not look like noisy spider legs."""
    rng = np.random.default_rng(7)
    resting = rng.normal(0, 0.035, (120, 3)).astype(np.float32)
    valid = np.ones(120, bool)

    extra_detailed = _smooth_hand_channel(resting, valid, 30.0, 0, 0.30, 2.0, 0.35, 5.0)
    detailed = _smooth_hand_channel(resting, valid, 30.0, 5, 0.225, 1.85, 0.375, 2.75)
    jitter = lambda values: float(np.abs(np.diff(values, n=2, axis=0)).mean())

    assert jitter(detailed) < 0.25 * jitter(extra_detailed)


def test_adaptive_finger_path_detects_a_pop_before_the_centered_prepass():
    """The short Savitzky-Golay pass spreads a one-frame bad estimate over
    nearby frames, so the transient detector must inspect raw hand poses
    before that pass, not after it."""
    sequence = np.zeros((60, 3), np.float32)
    sequence[30, 0] = 0.9
    valid = np.ones(60, bool)

    ordinary = _smooth_hand_channel(sequence, valid, 30.0, 5, 0.225, 1.85, 0.375, 2.75)
    adaptive = _smooth_hand_channel(
        sequence, valid, 30.0, 5, 0.225, 1.85, 0.375, 2.75,
        suppress_transient_reversals=True,
    )

    assert adaptive[:, 0].max() < 0.8 * ordinary[:, 0].max()




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


def test_confidence_gate_rejects_a_low_confidence_stretch():
    """A body-occluded wrist (real HaMeR still returns *a* pose, confidently
    wrong) is caught by dips in wrist keypoint confidence, not by the pose
    itself, see hamer_adapter's module docstring for why a kinematic check
    was tried and rejected in favor of this."""
    n = 30
    valid = np.ones(n, bool)
    conf = np.full(n, 0.9)
    conf[12:20] = 0.4  # a stretch of low confidence, as if occluded

    gated = _reject_low_confidence_stretches(valid, conf)

    assert not gated[12:20].any()
    # Well outside the rolling window's reach from the low stretch (half-window
    # of 3 frames either side of it).
    assert gated[:9].all()
    assert gated[23:].all()


def test_confidence_gate_catches_a_blip_inside_an_occluded_stretch():
    """A single frame's confidence can bounce back above threshold in the
    middle of an otherwise-occluded stretch (ViTPose momentarily picking up a
    partial cue), the rolling MIN, not the frame's own raw confidence, is
    what still rejects it, since its neighbors are still low."""
    n = 30
    valid = np.ones(n, bool)
    conf = np.full(n, 0.9)
    conf[12:20] = 0.4
    conf[16] = 0.9  # one-frame blip back up to "confident", still mid-occlusion

    gated = _reject_low_confidence_stretches(valid, conf)

    assert not gated[16]  # still rejected, neighbors are still low
    assert not gated[12:20].any()


def test_confidence_gate_never_resurrects_an_already_invalid_frame():
    """A frame already marked invalid upstream (no box at all) must stay
    invalid regardless of confidence, the gate can only narrow validity, not
    widen it."""
    n = 10
    valid = np.zeros(n, bool)
    conf = np.full(n, 0.9)  # high confidence everywhere, doesn't matter
    gated = _reject_low_confidence_stretches(valid, conf)
    assert not gated.any()


def test_confidence_gate_passes_through_uniformly_confident_sequence():
    n = 20
    valid = np.ones(n, bool)
    conf = np.full(n, MIN_ROLLING_WRIST_CONFIDENCE + 0.1)
    gated = _reject_low_confidence_stretches(valid, conf)
    assert gated.all()


def test_adapter_batches_pose_and_hand_inference_without_reordering(tmp_path, monkeypatch):
    """Stage 4 should run one ViTPose batch per source-frame batch and one
    HaMeR batch across its left/right crops, while storing each output back at
    the source frame and hand side that produced it."""
    class FakeBackbone:
        def __init__(self):
            self.batch_sizes: list[int] = []

        def __call__(self, crops):
            self.batch_sizes.append(crops.shape[0])
            return crops

    class FakeHead:
        def __call__(self, features):
            batch_size = features.shape[0]
            return {
                "global_orient": torch.eye(3).repeat(batch_size, 1, 1),
                "hand_pose": torch.eye(3).repeat(batch_size, 15, 1, 1),
            }

    pose_batch_sizes: list[int] = []

    def fake_estimate_keypoints_batch(_model, frames_rgb, _bboxes, _device, _dtype):
        pose_batch_sizes.append(len(frames_rgb))
        keypoints = np.zeros((len(frames_rgb), 17, 3), np.float32)
        keypoints[:, COCO_R_ELBOW] = [30.0, 30.0, 0.9]
        keypoints[:, COCO_R_WRIST] = [50.0, 45.0, 0.9]
        keypoints[:, COCO_L_ELBOW] = [90.0, 30.0, 0.9]
        keypoints[:, COCO_L_WRIST] = [70.0, 45.0, 0.9]
        return keypoints

    monkeypatch.setattr(hamer_adapter, "INFERENCE_FRAME_BATCH_SIZE", 2)
    monkeypatch.setattr(hamer_adapter, "estimate_keypoints_batch", fake_estimate_keypoints_batch)
    frame_paths: list[Path] = []
    for frame in range(3):
        path = tmp_path / f"{frame:06d}.jpg"
        assert cv2.imwrite(str(path), np.zeros((128, 128, 3), dtype=np.uint8))
        frame_paths.append(path)
    raw_masks = torch.ones((3, 1, 8, 16), dtype=torch.bool)

    adapter = HamerAdapter(device=torch.device("cpu"), dtype=torch.float32)
    adapter._vitpose = object()
    adapter._backbone = FakeBackbone()
    adapter._head = FakeHead()
    result = adapter.infer(frame_paths, {KEY_PACKED_MASKS: pack_masks(raw_masks)})

    assert pose_batch_sizes == [2, 1]
    assert adapter._backbone.batch_sizes == [4, 2]
    assert result[KEY_LEFT_VALID].all()
    assert result[KEY_RIGHT_VALID].all()
    assert np.allclose(result[KEY_LEFT_HAND_POSE], 0.0)
    assert np.allclose(result[KEY_RIGHT_GLOBAL_ORIENT], 0.0)


def test_hands_bvh_preview_is_structurally_valid(tmp_path):
    """The optional BVH preview builds a valid two-hand animated skeleton. Needs
    only the SMPL-X model file (for the hand joint offsets), no GPU/checkpoints."""
    from pipeline.adapters.hamer.hamer_bvh_preview import render_hands_bvh
    from pipeline.helpers.smplx_bvh_preview import SMPLX_MODEL_PATH

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
    render_hands_bvh(synthetic, fps=30.0, out_path=out)

    text = out.read_text()
    assert text.startswith("HIERARCHY")
    assert "MOTION" in text
    assert f"Frames: {n}" in text
    # root + 2 hands x (wrist + 15 fingers) = 33 joints => 32 JOINT lines + 1 ROOT
    assert text.count("JOINT ") == 32
    assert text.count("End Site") == 10  # 5 fingertips x 2 hands
