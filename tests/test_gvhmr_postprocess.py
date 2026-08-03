"""Unit tests for `gvhmr_postprocess`'s incam foot-drift lock and low-
confidence root-motion bridge: all the hysteresis/drift/bridge math
(`_static_label`/`_static_joint_drift`/`_unreliable_pose_label`/
`pp_bridge_low_confidence_root_motion`) runs as pure tensor ops, no
GPU/checkpoints needed; `pp_static_joint_incam` itself needs a real
`EnDecoder` (-> `SmplxSkeleton` -> the registration-gated SMPL-X model file),
skipped automatically when that file isn't present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from pipeline.adapters.gvhmr.gvhmr_postprocess import (
    POSE_CONF_RELEASE,
    POSE_CONF_SEED,
    STATIC_CONF_RELEASE,
    STATIC_CONF_SEED,
    STATIC_JOINT_IDS,
    _static_joint_drift,
    _static_label,
    _unreliable_pose_label,
    pp_bridge_low_confidence_root_motion,
    pp_static_joint_incam,
)
from pipeline.algorithms.motion_smoothing import fill_invalid, hemisphere_aligned_quats

SMPLX_MODEL_PATH = Path(__file__).resolve().parents[1] / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"


def _logit(p: float) -> float:
    return torch.logit(torch.tensor(p)).item()


def _confidence_seq(values: list[float], n_joints: int = 6, static_joint_pos: int = 0) -> torch.Tensor:
    """(1, T, n_joints) logits: `static_joint_pos` follows `values` (as
    probabilities, converted to logits), every other joint stays flat at a
    low, never-locking confidence."""
    logits = torch.full((1, len(values), n_joints), _logit(0.1))
    logits[0, :, static_joint_pos] = torch.tensor([_logit(v) for v in values])
    return logits


def test_static_label_locks_a_steadily_confident_joint():
    logits = _confidence_seq([0.95] * 10)
    label = _static_label(logits)
    assert label[0, :, 0].all()
    assert not label[0, :, 1:].any()


def test_static_label_never_locks_a_joint_below_seed():
    # Steadily above STATIC_CONF_RELEASE but never above STATIC_CONF_SEED --
    # "elevated" without a "seed" frame should never start a lock, same as
    # contact_detection.detect_contact_events' own seed requirement.
    assert STATIC_CONF_RELEASE < 0.6 < STATIC_CONF_SEED
    logits = _confidence_seq([0.6] * 10)
    label = _static_label(logits)
    assert not label.any()


def test_static_label_hysteresis_prevents_chatter_at_the_seed_boundary():
    # Alternates just below/above STATIC_CONF_SEED (0.8) but always above
    # STATIC_CONF_RELEASE (0.5) -- a bare `> 0.8` threshold would flicker
    # locked/unlocked every other frame; hysteresis should hold the lock
    # through the whole run once any single frame clears the seed.
    assert 0.79 < STATIC_CONF_SEED < 0.81
    values = [0.79, 0.81] * 5
    logits = _confidence_seq(values)
    label = _static_label(logits)
    assert label[0, :, 0].all()

    bare_threshold = torch.tensor(values) > STATIC_CONF_SEED
    assert not bare_threshold.all()  # confirms this scenario really would have chattered pre-fix


def _joints_with_drift(n_frames: int, joint_idx: int, drift_per_frame: tuple[float, float, float]) -> torch.Tensor:
    """(1, n_frames, 22, 3) all-zero joint positions except `joint_idx`, which
    accumulates `drift_per_frame` every frame starting from frame 0."""
    joints = torch.zeros(1, n_frames, 22, 3)
    drift = torch.tensor(drift_per_frame)
    for t in range(n_frames):
        joints[0, t, joint_idx] = drift * t
    return joints


def test_static_joint_drift_cancels_a_labeled_joints_own_drift():
    n_frames = 5
    joint_idx = STATIC_JOINT_IDS[0]
    joints = _joints_with_drift(n_frames, joint_idx, (0.01, 0.02, 0.0))
    label = torch.zeros(1, n_frames - 1, len(STATIC_JOINT_IDS), dtype=torch.bool)
    label[0, :, 0] = True

    disp = _static_joint_drift(joints, label, zero_vertical=False)
    assert disp.shape == (1, n_frames - 1, 3)
    assert torch.allclose(disp, torch.tensor([0.01, 0.02, 0.0]).expand(1, n_frames - 1, 3))


def test_static_joint_drift_is_zero_when_nothing_is_labeled():
    n_frames = 5
    joints = _joints_with_drift(n_frames, STATIC_JOINT_IDS[0], (0.01, 0.02, 0.03))
    label = torch.zeros(1, n_frames - 1, len(STATIC_JOINT_IDS), dtype=torch.bool)

    disp = _static_joint_drift(joints, label, zero_vertical=False)
    assert torch.allclose(disp, torch.zeros_like(disp))


def test_static_joint_drift_zero_vertical_flag():
    n_frames = 5
    joint_idx = STATIC_JOINT_IDS[0]
    joints = _joints_with_drift(n_frames, joint_idx, (0.01, 0.02, 0.03))
    label = torch.ones(1, n_frames - 1, len(STATIC_JOINT_IDS), dtype=torch.bool)

    disp_with_vertical = _static_joint_drift(joints, label, zero_vertical=False)
    assert not torch.allclose(disp_with_vertical[:, :, 1], torch.zeros(1, n_frames - 1))

    disp_no_vertical = _static_joint_drift(joints, label, zero_vertical=True)
    assert torch.allclose(disp_no_vertical[:, :, 1], torch.zeros(1, n_frames - 1))


def _synthetic_incam_outputs(n_frames: int, drift_per_frame: tuple[float, float, float]) -> dict:
    """A rigid, unarticulated (all-zero body_pose/global_orient/betas) body
    whose root drifts by `drift_per_frame` every frame -- every joint,
    including both ankles, moves by exactly that same rigid displacement, so
    flagging one ankle confidently static gives `pp_static_joint_incam`
    everything it needs to recover the injected drift exactly."""
    transl = torch.zeros(1, n_frames, 3)
    drift = torch.tensor(drift_per_frame)
    for t in range(n_frames):
        transl[0, t] = drift * t

    return {
        "pred_smpl_params_incam": {
            "body_pose": torch.zeros(1, n_frames, 63),
            "betas": torch.zeros(1, n_frames, 10),
            "global_orient": torch.zeros(1, n_frames, 3),
            "transl": transl,
        },
        "static_conf_logits": _confidence_seq([0.95] * n_frames, static_joint_pos=0),
    }


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_pp_static_joint_incam_cancels_horizontal_and_vertical_drift():
    from pipeline.adapters.gvhmr.gvhmr_endecoder import EnDecoder

    endecoder = EnDecoder()
    n_frames = 6
    outputs = _synthetic_incam_outputs(n_frames, drift_per_frame=(0.02, 0.015, 0.0))

    corrected = pp_static_joint_incam(outputs, endecoder)

    frame_to_frame = corrected[0, 1:] - corrected[0, :-1]
    assert torch.allclose(frame_to_frame, torch.zeros_like(frame_to_frame), atol=1e-5)


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_pp_static_joint_incam_is_a_noop_when_nothing_is_confidently_static():
    from pipeline.adapters.gvhmr.gvhmr_endecoder import EnDecoder

    endecoder = EnDecoder()
    n_frames = 6
    outputs = _synthetic_incam_outputs(n_frames, drift_per_frame=(0.02, 0.015, 0.0))
    outputs["static_conf_logits"] = torch.full((1, n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    corrected = pp_static_joint_incam(outputs, endecoder)

    assert torch.allclose(corrected, outputs["pred_smpl_params_incam"]["transl"])


def test_unreliable_pose_label_flags_a_genuine_confidence_collapse():
    # Steadily below POSE_CONF_SEED for a whole run -- the real shape a fast,
    # motion-blurred tumble produces (mean body-keypoint confidence measured
    # on a real clip: ~0.75 baseline, ~0.25-0.5 through the bad stretch).
    assert POSE_CONF_SEED > 0.3
    confidence = torch.full((1, 10), 0.3)
    label = _unreliable_pose_label(confidence)
    assert label.all()


def test_unreliable_pose_label_never_flags_confidence_that_stays_above_release():
    assert 0.9 > POSE_CONF_RELEASE
    confidence = torch.full((1, 10), 0.9)
    label = _unreliable_pose_label(confidence)
    assert not label.any()


def test_unreliable_pose_label_ignores_a_dip_that_never_reaches_seed():
    # Dips below POSE_CONF_RELEASE (a "candidate" region) but never down to
    # POSE_CONF_SEED -- should not be confirmed as unreliable, same seed-
    # requirement logic as _static_label's own boundary test.
    midpoint = (POSE_CONF_SEED + POSE_CONF_RELEASE) / 2
    assert POSE_CONF_SEED < midpoint < POSE_CONF_RELEASE
    confidence = torch.full((1, 10), midpoint)
    label = _unreliable_pose_label(confidence)
    assert not label.any()


def _framewise_pose_params(n_frames: int) -> dict:
    """pred_smpl_params_incam where body_pose/betas/transl's value at frame t
    encodes t itself, so a bridged frame's post-fix value can be checked
    exactly against whichever real frame(s) it should have been
    interpolated/frozen from. global_orient is a small rotation about Z
    (0.05 * t radians) instead of the same frame-index encoding -- keeps
    quaternion interpolation well clear of any large-rotation wraparound
    edge case, which isn't what these tests are checking."""
    return {
        "body_pose": torch.arange(n_frames).float().view(1, n_frames, 1).expand(1, n_frames, 63).clone(),
        "betas": torch.arange(n_frames).float().view(1, n_frames, 1).expand(1, n_frames, 10).clone(),
        "global_orient": torch.stack(
            [torch.zeros(n_frames), torch.zeros(n_frames), torch.arange(n_frames).float() * 0.05], dim=-1
        ).view(1, n_frames, 3),
        "transl": torch.arange(n_frames).float().view(1, n_frames, 1).expand(1, n_frames, 3).clone(),
    }


def test_pp_bridge_low_confidence_root_motion_interpolates_an_interior_run():
    # Plenty of clearly-reliable buffer frames on each side of the dip, wider
    # than POSE_CONF_WINDOW, so the run's own start/end aren't obscured by the
    # rolling-window's edge-widening (see _unreliable_pose_label's own
    # boundary tests above for that widening behavior in isolation).
    n_frames = 16
    params = _framewise_pose_params(n_frames)
    # Corrupt the flagged run's own raw transl -- this is the actual point of
    # the fix (GVHMR's raw estimate during a real confidence collapse is
    # garbage, not just numerically off-trend), and a plain linear-in-frame-
    # index fixture can't otherwise prove interpolation actually overwrote
    # anything, since interpolating an already-linear run reproduces itself.
    params["transl"][0, 6:9] = torch.tensor([100.0, 100.0, 100.0])
    confidence = torch.full((1, n_frames), 0.9)
    confidence[0, 6:9] = POSE_CONF_SEED - 0.1  # frames 6-8 confirmed unreliable

    bridged, label = pp_bridge_low_confidence_root_motion(params, confidence)
    assert torch.equal(label, _unreliable_pose_label(confidence))  # returned label matches the real one, not a stub

    valid = ~label[0].numpy()

    # transl: linear interpolation is exactly what fill_invalid does --
    # check the real function's output directly against it, not a hand-
    # derived formula. Also confirms the corrupted raw values above were
    # actually discarded, not passed through.
    expected_transl = fill_invalid(params["transl"][0].numpy(), valid)
    assert np.allclose(bridged["transl"][0].numpy(), expected_transl)
    assert not torch.allclose(bridged["transl"][0, 6:9], params["transl"][0, 6:9])

    # global_orient: bridged result should exactly match calling the same
    # shared helper directly on the raw input -- confirms correct wiring,
    # not re-deriving slerp math by hand.
    quats = hemisphere_aligned_quats(params["global_orient"][0].numpy(), valid)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    expected_orient = Rotation.from_quat(quats).as_rotvec()
    assert np.allclose(bridged["global_orient"][0].numpy(), expected_orient, atol=1e-6)

    # body_pose: the user's own explicit request -- never touched, even
    # during the flagged run.
    assert torch.allclose(bridged["body_pose"], params["body_pose"])


def test_pp_bridge_low_confidence_root_motion_freezes_a_leading_run():
    n_frames = 8
    params = _framewise_pose_params(n_frames)
    confidence = torch.full((1, n_frames), 0.9)
    confidence[0, 0:3] = POSE_CONF_SEED - 0.1  # unreliable from frame 0 -- no earlier real frame

    bridged, label = pp_bridge_low_confidence_root_motion(params, confidence)
    label = label[0]

    first_valid = int((~label).nonzero()[0][0])
    for t in range(n_frames):
        if label[t]:
            assert torch.allclose(bridged["transl"][0, t], params["transl"][0, first_valid])
            assert torch.allclose(bridged["global_orient"][0, t], params["global_orient"][0, first_valid])
    assert torch.allclose(bridged["body_pose"], params["body_pose"])


def test_pp_bridge_low_confidence_root_motion_freezes_a_trailing_run():
    n_frames = 8
    params = _framewise_pose_params(n_frames)
    confidence = torch.full((1, n_frames), 0.9)
    confidence[0, 5:8] = POSE_CONF_SEED - 0.1  # unreliable through the clip's end

    bridged, label = pp_bridge_low_confidence_root_motion(params, confidence)
    label = label[0]

    last_valid = int((~label).nonzero()[-1][0])
    for t in range(n_frames):
        if label[t]:
            assert torch.allclose(bridged["transl"][0, t], params["transl"][0, last_valid])
            assert torch.allclose(bridged["global_orient"][0, t], params["global_orient"][0, last_valid])
    assert torch.allclose(bridged["body_pose"], params["body_pose"])


def test_pp_bridge_low_confidence_root_motion_leaves_betas_untouched():
    n_frames = 8
    params = _framewise_pose_params(n_frames)
    confidence = torch.full((1, n_frames), 0.9)
    confidence[0, 3:6] = POSE_CONF_SEED - 0.1

    bridged, _ = pp_bridge_low_confidence_root_motion(params, confidence)

    assert torch.allclose(bridged["betas"], params["betas"])  # shape param, not motion -- never touched


def test_pp_bridge_low_confidence_root_motion_is_a_noop_when_nothing_is_flagged():
    n_frames = 8
    params = _framewise_pose_params(n_frames)
    confidence = torch.full((1, n_frames), 0.9)

    bridged, label = pp_bridge_low_confidence_root_motion(params, confidence)

    assert not label.any()
    for key in params:
        assert torch.allclose(bridged[key], params[key])
