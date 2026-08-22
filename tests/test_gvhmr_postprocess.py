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

from pipeline.adapters.gvhmr.gvhmr_adapter import GVHMRAdapter
from pipeline.adapters.gvhmr.gvhmr_postprocess import (
    POSE_CONF_RELEASE,
    POSE_CONF_SEED,
    STATIC_CONF_RELEASE,
    STATIC_CONF_SEED,
    STANCE_MAX_CORRECTION_STEP_M,
    STATIC_JOINT_IDS,
    _rate_limited,
    _stance_edge_weight,
    _static_joint_drift,
    _static_label,
    _unreliable_pose_label,
    pp_bridge_low_confidence_root_motion,
    pp_static_joint_incam,
    relock_stance_feet_with_ik,
    stance_vertical_grounding_correction,
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
    # Steadily above STATIC_CONF_RELEASE but never above STATIC_CONF_SEED,
    # "elevated" without a "seed" frame should never start a lock, same as
    # contact_detection.detect_contact_events' own seed requirement.
    assert STATIC_CONF_RELEASE < 0.6 < STATIC_CONF_SEED
    logits = _confidence_seq([0.6] * 10)
    label = _static_label(logits)
    assert not label.any()


def test_static_label_hysteresis_prevents_chatter_at_the_seed_boundary():
    # Alternates just below/above STATIC_CONF_SEED (0.8) but always above
    # STATIC_CONF_RELEASE (0.5), a bare `> 0.8` threshold would flicker
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


def _stance_joints(foot_y: np.ndarray) -> torch.Tensor:
    """Minimal FK output whose four foot points share `foot_y` per frame."""
    joints = torch.zeros(1, len(foot_y), 22, 3)
    joints[0, :, [7, 10, 8, 11], 1] = torch.from_numpy(foot_y[:, None]).expand(-1, 4)
    return joints


def _per_foot_stance_joints(left_y: np.ndarray, right_y: np.ndarray) -> torch.Tensor:
    joints = torch.zeros(1, len(left_y), 22, 3)
    joints[0, :, [7, 10], 1] = torch.from_numpy(left_y[:, None]).expand(-1, 2)
    joints[0, :, [8, 11], 1] = torch.from_numpy(right_y[:, None]).expand(-1, 2)
    return joints


def test_stance_vertical_grounding_flattens_a_still_foot_without_static_logits():
    # Geometry fallback matters when GVHMR does not emit a static confidence:
    # a long, horizontally motionless stance should still suppress vertical
    # root wobble rather than wait for the classifier.
    n_frames = 600
    foot_y = (1.0 + 0.08 * np.sin(np.linspace(0, 4 * np.pi, n_frames))).astype(np.float32)
    joints = _stance_joints(foot_y)
    low_confidence = torch.full((1, n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    correction = stance_vertical_grounding_correction(joints, low_confidence, fps=30.0)
    # The correction is a distance along the camera's own up direction, which
    # for the default level camera is -Y, hence the subtraction from a Y-down
    # foot position.
    grounded_foot_y = foot_y - correction[0].numpy()

    assert correction.shape == (1, n_frames)
    assert grounded_foot_y.std() < 0.004
    assert np.abs(correction.numpy()).max() > 0.05


def test_stance_vertical_grounding_releases_an_airborne_foot():
    n_frames = 80
    foot_y = np.ones(n_frames, dtype=np.float32)
    foot_y[25:55] = 0.35  # camera Y-down: a foot well above its floor level
    joints = _stance_joints(foot_y)
    # It also travels horizontally while airborne, so neither stance signal is
    # available in the middle of the jump.
    joints[0, 25:55, [7, 10, 8, 11], 0] = torch.linspace(0.0, 0.6, 30).view(-1, 1)
    low_confidence = torch.full((1, n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    correction = stance_vertical_grounding_correction(joints, low_confidence, fps=30.0)

    assert torch.allclose(correction[0, 31:49], torch.zeros(18), atol=1e-6)


def test_stance_vertical_grounding_holds_height_when_one_foot_stays_near_floor():
    """A forward/waist bend can make the strict stance detector lose its
    support foot. Its height must remain anchored until *both* feet fly."""
    n_frames = 100
    left_y = np.concatenate([np.linspace(0.95, 1.10, 40), np.full(60, 1.10)]).astype(np.float32)
    right_y = np.concatenate([np.full(40, 1.10), np.full(60, 0.30)]).astype(np.float32)
    # After frame 40, this foot is lifted well above the camera-Y-down floor.
    joints = _per_foot_stance_joints(left_y, right_y)
    # After frame 40 the remaining support foot travels in X, deliberately
    # failing the strict horizontal-velocity stance test while still near floor.
    joints[0, 40:, [7, 10], 0] = torch.linspace(0.0, 0.6, 60).view(-1, 1)
    low_confidence = torch.full((1, n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    correction = stance_vertical_grounding_correction(joints, low_confidence, fps=30.0)[0].numpy()

    # The last strict stance establishes a nonzero vertical correction. It is
    # preserved during the single-support bend rather than decaying to zero.
    assert abs(correction[55]) > 0.005


def test_stance_vertical_grounding_never_injects_a_body_height_step():
    # A pose-estimation discontinuity can make two adjacent detected stances
    # prefer substantially different heights. The grounding correction itself
    # must never transmit that as a one-frame pelvis jump.
    n_frames = 80
    foot_y = np.ones(n_frames, dtype=np.float32)
    foot_y[30:] = 0.92
    joints = _stance_joints(foot_y)
    low_confidence = torch.full((1, n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    correction = stance_vertical_grounding_correction(joints, low_confidence, fps=30.0)[0].numpy()

    assert np.abs(np.diff(correction)).max() <= STANCE_MAX_CORRECTION_STEP_M + 1e-7


def _synthetic_incam_outputs(n_frames: int, drift_per_frame: tuple[float, float, float]) -> dict:
    """A rigid, unarticulated (all-zero body_pose/global_orient/betas) body
    whose root drifts by `drift_per_frame` every frame, every joint,
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


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_adapter_relocks_smoothed_unbatched_incam_params():
    """Stage 2 hands unbatched, already-smoothed params back to the adapter;
    verify its small batch-dimension bridge retains the vertical foot lock."""
    from pipeline.adapters.gvhmr.gvhmr_endecoder import EnDecoder

    outputs = _synthetic_incam_outputs(n_frames=6, drift_per_frame=(0.0, 0.015, 0.0))
    adapter = GVHMRAdapter.__new__(GVHMRAdapter)
    adapter._loaded = True
    adapter.endecoder = EnDecoder()

    corrected = adapter.relock_smoothed_incam_feet(
        {key: value[0] for key, value in outputs["pred_smpl_params_incam"].items()},
        outputs["static_conf_logits"][0],
    )

    assert torch.allclose(corrected[1:] - corrected[:-1], torch.zeros_like(corrected[1:]), atol=1e-5)


def test_unreliable_pose_label_flags_a_genuine_confidence_collapse():
    # Steadily below POSE_CONF_SEED for a whole run, the real shape a fast,
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
    # POSE_CONF_SEED, should not be confirmed as unreliable, same seed-
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
    (0.05 * t radians) instead of the same frame-index encoding, keeps
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
    # Corrupt the flagged run's own raw transl, this is the actual point of
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

    # transl: linear interpolation is exactly what fill_invalid does,
    # check the real function's output directly against it, not a hand-
    # derived formula. Also confirms the corrupted raw values above were
    # actually discarded, not passed through.
    expected_transl = fill_invalid(params["transl"][0].numpy(), valid)
    assert np.allclose(bridged["transl"][0].numpy(), expected_transl)
    assert not torch.allclose(bridged["transl"][0, 6:9], params["transl"][0, 6:9])

    # global_orient: bridged result should exactly match calling the same
    # shared helper directly on the raw input, confirms correct wiring,
    # not re-deriving slerp math by hand.
    quats = hemisphere_aligned_quats(params["global_orient"][0].numpy(), valid)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    expected_orient = Rotation.from_quat(quats).as_rotvec()
    assert np.allclose(bridged["global_orient"][0].numpy(), expected_orient, atol=1e-6)

    # body_pose is out of scope for root-motion bridging by design, never
    # touched, even during the flagged run.
    assert torch.allclose(bridged["body_pose"], params["body_pose"])


def test_pp_bridge_low_confidence_root_motion_freezes_a_leading_run():
    n_frames = 8
    params = _framewise_pose_params(n_frames)
    confidence = torch.full((1, n_frames), 0.9)
    confidence[0, 0:3] = POSE_CONF_SEED - 0.1  # unreliable from frame 0, no earlier real frame

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

    assert torch.allclose(bridged["betas"], params["betas"])  # shape param, not motion, never touched


def test_pp_bridge_low_confidence_root_motion_is_a_noop_when_nothing_is_flagged():
    n_frames = 8
    params = _framewise_pose_params(n_frames)
    confidence = torch.full((1, n_frames), 0.9)

    bridged, label = pp_bridge_low_confidence_root_motion(params, confidence)

    assert not label.any()
    for key in params:
        assert torch.allclose(bridged[key], params[key])


def test_pp_bridge_low_confidence_root_motion_handles_a_never_confident_clip():
    """Regression guard for a real crash: a shoulders-up clip framed for
    face capture gives GVHMR's whole-body pose network almost no
    real body to see. Confidence never once crosses POSE_CONF_SEED across
    the entire clip, so `valid` is all-False for that person and the old
    `fill_invalid(transl_np, valid[b])` call raised (`np.interp` on an empty
    sample-point array). There's no reliable frame to bridge from or to, so
    the whole clip should come back marked unreliable with transl/
    global_orient left exactly as GVHMR produced them"""
    n_frames = 8
    params = _framewise_pose_params(n_frames)
    confidence = torch.full((1, n_frames), POSE_CONF_SEED - 0.1)

    bridged, label = pp_bridge_low_confidence_root_motion(params, confidence)

    assert label.all()
    assert torch.allclose(bridged["transl"], params["transl"])
    assert torch.allclose(bridged["global_orient"], params["global_orient"])


def test_stance_vertical_grounding_measures_height_along_a_tilted_camera_up():
    """On a tilted camera, a foot planted on a level floor while the subject
    walks changes its camera Y even though its real height is constant. The
    correction must read height along the measured up direction, so that
    horizontal travel is not mistaken for float (and corrected away)."""
    n_frames = 300
    tilt = np.radians(17.0)
    camera_up = np.array([0.0, -np.cos(tilt), np.sin(tilt)])
    # A foot at constant true height, travelling along the floor.
    travel = np.linspace(0.0, 1.2, n_frames).astype(np.float32)
    floor_forward = np.array([0.0, np.sin(tilt), np.cos(tilt)], dtype=np.float32)
    joints = torch.zeros(1, n_frames, 22, 3)
    joints[0, :, [7, 10, 8, 11], :] = torch.from_numpy(travel[:, None] * floor_forward).unsqueeze(1).expand(-1, 4, -1)
    low_confidence = torch.full((1, n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    correction = stance_vertical_grounding_correction(joints, low_confidence, 30.0, camera_up)[0].numpy()

    # True height never changes, so there is nothing to correct.
    assert np.abs(correction).max() < 1e-3


def test_stance_vertical_grounding_follows_the_lower_of_two_planted_feet():
    """Two planted feet at different heights disagree about the needed root
    move. The weight-bearing (lower) foot is the one that defines the floor;
    following the smaller-magnitude candidate instead lets the two feet's
    differing stance medians cancel each other out to nearly nothing."""
    n_frames = 400
    # Camera Y-down: the larger value is the LOWER, weight-bearing foot. Only
    # it drifts; the raised foot is steady, and stays far enough below the
    # other that the two never swap which one is lower.
    lower_y = (1.0 + 0.06 * np.sin(np.linspace(0, 2 * np.pi, n_frames))).astype(np.float32)
    upper_y = np.full(n_frames, 0.92, dtype=np.float32)
    joints = _per_foot_stance_joints(upper_y, lower_y)
    low_confidence = torch.full((1, n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    correction = stance_vertical_grounding_correction(joints, low_confidence, 30.0)[0].numpy()

    grounded_lower = lower_y - correction
    assert grounded_lower.std() < 0.1 * lower_y.std()


def test_rate_limited_is_symmetric_in_time():
    """A single forward sweep lags every change it clamps. Limiting a signal
    and its own time reversal must give mirror-image results, or the pass
    injects a direction-dependent phase shift into the correction."""
    signal = np.concatenate([np.zeros(40), np.full(40, 0.05)]).astype(np.float32)

    forward = _rate_limited(signal.copy(), 0.0005)
    backward = _rate_limited(signal[::-1].copy(), 0.0005)[::-1]

    assert np.allclose(forward, backward, atol=1e-7)
    assert np.abs(np.diff(forward)).max() <= 0.0005 + 1e-7


def test_stance_edge_weight_eases_smoothly_at_an_interior_edge():
    """Smoothstep, not a straight line: a linear ramp's corners land on the
    corrected joints as acceleration spikes."""
    start, end, n_frames = 20, 60, 100
    weights = [_stance_edge_weight(f, start, end, n_frames) for f in range(start, end + 1)]

    assert weights[len(weights) // 2] == pytest.approx(1.0)
    assert weights == pytest.approx(weights[::-1])       # symmetric across the run
    # The S-curve signature, against the linear ramp this replaced: over a
    # 4-frame blend a straight line would pass through exactly 1/4, 2/4, 3/4,
    # so smoothstep must sit below early and above late.
    assert weights[0] < 0.25
    assert weights[1] == pytest.approx(0.5)              # the curves cross at the midpoint
    assert weights[2] > 0.75

    # A run touching a clip boundary has nothing to blend toward, so it keeps
    # the full correction there.
    assert _stance_edge_weight(0, 0, end, n_frames) == pytest.approx(1.0)
    assert _stance_edge_weight(n_frames - 1, start, n_frames - 1, n_frames) == pytest.approx(1.0)


def _incam_params_from_joint_motion(n_frames: int, transl: np.ndarray) -> dict:
    """A real, articulable SMPL body whose root follows `transl`; body pose and
    shape stay neutral so FK positions are driven purely by the root."""
    return {
        "body_pose": torch.zeros(n_frames, 63),
        "betas": torch.zeros(n_frames, 10),
        "global_orient": torch.zeros(n_frames, 3),
        "transl": torch.from_numpy(transl.astype(np.float32)),
    }


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_relock_stance_feet_never_moves_the_root():
    """The whole point of this pass is that it is not a root move: the root
    correction already ran, and moving the body again here would double-count
    it. Only leg rotations may change."""
    from pipeline.adapters.gvhmr.gvhmr_endecoder import EnDecoder

    endecoder = EnDecoder()
    n_frames = 200
    # A standing body whose root wobbles vertically: exactly the case a planted
    # foot should absorb into the legs.
    transl = np.zeros((n_frames, 3))
    transl[:, 1] = 0.04 * np.sin(np.linspace(0, 2 * np.pi, n_frames))
    params = _incam_params_from_joint_motion(n_frames, transl)
    low_confidence = torch.full((n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    body_pose = relock_stance_feet_with_ik(params, low_confidence, endecoder, 30.0)

    assert body_pose.shape == params["body_pose"].shape
    before = endecoder.fk_v2(**{k: v.unsqueeze(0) for k, v in params.items()})[0]
    after = endecoder.fk_v2(**{k: v.unsqueeze(0) for k, v in {**params, "body_pose": body_pose}.items()})[0]
    assert torch.allclose(before[:, 0], after[:, 0], atol=1e-6)


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_relock_stance_feet_pulls_a_planted_toe_toward_its_stance_position():
    from pipeline.adapters.gvhmr.gvhmr_endecoder import EnDecoder

    endecoder = EnDecoder()
    n_frames = 200
    # Root drifts horizontally as well as vertically, so both components of the
    # planted toe's drift are represented.
    transl = np.zeros((n_frames, 3))
    transl[:, 0] = np.linspace(0.0, 0.05, n_frames)
    transl[:, 1] = 0.04 * np.sin(np.linspace(0, 2 * np.pi, n_frames))
    params = _incam_params_from_joint_motion(n_frames, transl)
    low_confidence = torch.full((n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    body_pose = relock_stance_feet_with_ik(params, low_confidence, endecoder, 30.0)

    def toe_excursion(pose):
        joints = endecoder.fk_v2(**{k: v.unsqueeze(0) for k, v in {**params, "body_pose": pose}.items()})
        toe = joints[0, :, 10].detach().numpy()
        return float(np.linalg.norm(toe - toe.mean(0), axis=-1).max())

    assert toe_excursion(body_pose) < 0.6 * toe_excursion(params["body_pose"])


@pytest.mark.skipif(not SMPLX_MODEL_PATH.exists(), reason="needs the SMPL-X model file (registration-gated)")
def test_relock_stance_feet_leaves_an_unplanted_foot_alone():
    """A foot that is never planted (here: the whole body is travelling too
    fast for any stance) must keep its own trajectory exactly. This is the
    guard that kept a supported acrobatic lift untouched on real footage."""
    from pipeline.adapters.gvhmr.gvhmr_endecoder import EnDecoder

    endecoder = EnDecoder()
    n_frames = 120
    transl = np.zeros((n_frames, 3))
    transl[:, 0] = np.linspace(0.0, 6.0, n_frames)  # 1.5 m/s, far above stance speed
    params = _incam_params_from_joint_motion(n_frames, transl)
    low_confidence = torch.full((n_frames, len(STATIC_JOINT_IDS)), _logit(0.05))

    body_pose = relock_stance_feet_with_ik(params, low_confidence, endecoder, 30.0)

    assert torch.allclose(body_pose, params["body_pose"], atol=1e-6)
