"""estimate_hands: runs HaMeR on every frame to recover per-frame MANO hand
pose for both hands, using the SAM 3.1 human mask (stage 1) to locate the
person and our COCO-17 ViTPose to locate each hand.

Output is the *raw* per-hand MANO pose (finger articulation + wrist
orientation, both in each hand crop's camera frame) plus TWO per-frame validity
flags per hand: `*_valid` (detection-based, a hand may be off-screen, too
occluded to detect, or its keypoint confidence too low over a rolling window)
and `*_wrist_valid` (a further narrowing, wrist-only, see below). Grafting
these onto GVHMR's body, merging the wrist/forearm reconciliation into the
final SMPL-X body pose, is still stage 5's job (retarget_hands), not this one.

This stage depends on stage 2 (estimate_human_motion), not just stage 0/1: a
hand is an extension of the arm, not an independent tracked object, and a real
wrist has a rotational limit relative to the forearm that HaMeR's crop-only
view has no way to know about. Before any smoothing runs, every frame's raw
wrist estimate is checked against GVHMR's own elbow orientation
(`hand_retarget.reject_biomechanically_implausible_wrist`) and treated as an
occlusion if it's anatomically impossible, deliberately done here, before the
smoothing chain, rather than after in stage 5: a filter that's already blended
a bad value into its neighbors can't be un-blended by a later stage.

This wrist check produces its OWN validity array (`*_wrist_valid`), separate
from the base `*_valid` used for finger smoothing: an isolated biomechanically
implausible wrist estimate does not mean the finger articulation is wrong.
One exception is a *sustained* wrist failure: it is a strong occlusion signal,
so the stage temporarily holds finger articulation through the run rather than
letting HaMeR hallucinate a new pose from the occluder. This policy is symmetric
for both hands and only activates after a multi-frame run, preserving usable
fingers when a wrist rejection is brief or incidental.

"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..adapters.gvhmr.gvhmr_adapter import KEY_BODY_POSE, KEY_GLOBAL_ORIENT, KEY_PRED_SMPL_PARAMS_INCAM
from ..adapters.gvhmr.gvhmr_smplx_skeleton import SmplxSkeleton
from ..adapters.hamer.hamer_adapter import (
    HamerAdapter,
    KEY_LEFT_FINGER_AMBIGUOUS,
    KEY_LEFT_FINGER_OBJECT_CONTACT,
    KEY_LEFT_GLOBAL_ORIENT,
    KEY_LEFT_HAND_POSE,
    KEY_LEFT_VALID,
    KEY_LEFT_WRIST_VALID,
    KEY_RIGHT_GLOBAL_ORIENT,
    KEY_RIGHT_FINGER_AMBIGUOUS,
    KEY_RIGHT_FINGER_OBJECT_CONTACT,
    KEY_RIGHT_HAND_POSE,
    KEY_RIGHT_VALID,
    KEY_RIGHT_WRIST_VALID,
)
from ..algorithms.hand_retarget import (
    LEFT_ELBOW,
    LEFT_MIDDLE1,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_MIDDLE1,
    RIGHT_WRIST,
    body_joint_global_rotations,
    reject_biomechanically_implausible_wrist,
    reject_hand_swung_past_forearm,
    reject_hand_through_forearm,
    reject_wrist_velocity_spikes,
    rest_forearm_and_hand_offsets,
    rest_hand_direction,
    wrist_global_from_relative,
    wrist_relative_to_elbow,
)
from ..algorithms.contact_detection import contiguous_true_runs
from ..algorithms.motion_smoothing import (
    cap_long_gaps_with_hold,
    decimate_rotation_sequence,
    one_euro_filter_rotation_sequence,
    smooth_rotation_sequence,
    transient_rotation_reversal_mask,
)
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import FINGER_MOTION_SETTINGS, FingerMotionSettings, RunRecord, StageName
from ..stages.stage_1_mask_and_track import OUTPUT_HUMAN_MASKS, OUTPUT_OBJECT_MASKS
from ..stages.stage_2_estimate_human_motion import OUTPUT_HUMAN_MOTION

# stage_0_ingest_video.py's own output key, consumed here.
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

HANDS_DIRNAME = f"stage{StageName.STAGE_4_ESTIMATE_HANDS.stage_number}_hands"
HAND_POSE_FILENAME = "hand_pose.npz"
HANDS_PREVIEW_FILENAME = "hands_preview.bvh"

# This stage's own progress.json output keys.
OUTPUT_HAND_POSE = "hand_pose"
OUTPUT_HANDS_PREVIEW = "hands_preview"

# A single rejected wrist frame is not enough evidence to discard otherwise
# useful finger articulation. A sustained run is a reliable occlusion proxy:
# HaMeR has no native visibility score and will otherwise infer arbitrary MANO
# poses from the occluder. Keep most of that run at the last trusted pose, then
# use a brief recovery bridge if the hand reappears.
FINGER_WRIST_FAILURE_MIN_FRAMES = 6
FINGER_WRIST_RECOVERY_BRIDGE_FRAMES = 3
# HaMeR estimates each crop independently. Nearby hand overlap is a useful
# warning that the estimate may be less reliable, but it does not prove the
# fingers are hidden: it can also be a visible two-hand grip.
# Keep the adaptive filter conservative through such a warning and its short
# recovery, but reserve a held pose for an actual sustained wrist-confidence
# failure. This policy is symmetric across hands and clips.
FINGER_AMBIGUITY_RECOVERY_FRAMES = 24
FINGER_AMBIGUITY_BETA_SCALE = 0.12
# Object contact commonly still shows a hand's grip. It should be a little
# more conservative than a clear hand, but never be converted into a held
# pose; otherwise every prolonged tool/object grip becomes frozen.
FINGER_OBJECT_CONTACT_BETA_SCALE = 0.20


def _finger_motion_settings(runRecord: RunRecord) -> FingerMotionSettings:
    """Resolve a profile, then apply only explicit numeric user pins over it."""
    profile = FINGER_MOTION_SETTINGS[runRecord.input.finger_motion]
    profile_pins = {
        name: value
        for name, value in runRecord.fine_tuning_overrides.items()
        if name in FingerMotionSettings.__dataclass_fields__
    }
    return replace(profile, **profile_pins)


def _finger_validity_after_sustained_wrist_failure(
    finger_valid: np.ndarray, wrist_valid: np.ndarray, finger_ambiguous: np.ndarray | None = None,
) -> np.ndarray:
    """Demote only sustained wrist failures; crop ambiguity remains a soft cue.

    ``finger_ambiguous`` is intentionally accepted for compatibility with the
    caller's data shape, but is not a validity failure. Forearm-derived crops
    can overlap another hand while the fingers remain clearly visible, so
    treating that 2-D signal as missing data freezes real grips.
    """
    out = np.asarray(finger_valid, dtype=bool).copy()
    suspect = out & ~np.asarray(wrist_valid, dtype=bool)
    for start, end in contiguous_true_runs(suspect):
        if end - start + 1 >= FINGER_WRIST_FAILURE_MIN_FRAMES:
            out[start:end + 1] = False
    return out


def _finger_ambiguity_beta_scale(finger_ambiguous: np.ndarray) -> np.ndarray:
    """Return a conservative, smoothly recovering bandwidth for uncertain crops."""
    ambiguous = np.asarray(finger_ambiguous, dtype=bool)
    scale = np.ones(len(ambiguous), dtype=float)
    for start, end in contiguous_true_runs(ambiguous):
        scale[start:end + 1] = FINGER_AMBIGUITY_BETA_SCALE
        recovery_end = min(end + FINGER_AMBIGUITY_RECOVERY_FRAMES, len(ambiguous) - 1)
        recovery_count = recovery_end - end
        if recovery_count:
            recovery = np.linspace(FINGER_AMBIGUITY_BETA_SCALE, 1.0, recovery_count + 1)[1:]
            scale[end + 1:recovery_end + 1] = np.minimum(scale[end + 1:recovery_end + 1], recovery)
    return scale


def _finger_object_contact_beta_scale(finger_object_contact: np.ndarray) -> np.ndarray:
    """Lower bandwidth modestly for a visible hand-object grip, never hold it."""
    contact = np.asarray(finger_object_contact, dtype=bool)
    scale = np.ones(len(contact), dtype=float)
    for start, end in contiguous_true_runs(contact):
        scale[start:end + 1] = FINGER_OBJECT_CONTACT_BETA_SCALE
        recovery_end = min(end + FINGER_AMBIGUITY_RECOVERY_FRAMES, len(contact) - 1)
        recovery_count = recovery_end - end
        if recovery_count:
            recovery = np.linspace(FINGER_OBJECT_CONTACT_BETA_SCALE, 1.0, recovery_count + 1)[1:]
            scale[end + 1:recovery_end + 1] = np.minimum(scale[end + 1:recovery_end + 1], recovery)
    return scale


def _smooth_hand_channel(
    axis_angle: np.ndarray,
    valid: np.ndarray,
    fps: float,
    savgol_window: int,
    min_cutoff_hz: float,
    beta: float,
    decimate_deg: float,
    derivative_cutoff_hz: float = 1.0,
    suppress_transient_reversals: bool = False,
    beta_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth one hand channel while preserving the configured motion bandwidth.

    A zero/one-frame ``savgol_window`` deliberately disables the centered
    pre-pass. That is important for fingers: finger motion can be a small,
    two-to-three-frame bend, so a wide, fixed window would flatten it *before*
    the adaptive filter gets a chance to identify it as real motion. Wrists
    retain their wide pre-pass because their input is noisier and their motion
    is naturally broader. The one-euro and decimation passes still provide
    rest-state stability and remove residual high-frequency estimator noise.
    """
    # Detect brief reversals before the fixed-window pass; that pass is useful
    # for ordinary hand noise, but it would spread a one-frame pose pop across
    # its neighbours and hide the temporal evidence that identifies it.
    transient_reversals = (
        transient_rotation_reversal_mask(axis_angle, fps, valid)
        if suppress_transient_reversals else None
    )
    smoothed = axis_angle
    if savgol_window >= 3:
        smoothed = smooth_rotation_sequence(smoothed, savgol_window, valid=valid)
    smoothed = one_euro_filter_rotation_sequence(
        smoothed, fps, min_cutoff_hz, beta, derivative_cutoff_hz, valid=valid,
        transient_reversal_mask=transient_reversals,
        beta_scale=beta_scale,
    )
    return decimate_rotation_sequence(smoothed, decimate_deg, valid=valid)


def _smooth_hand_result(
    result: dict,
    global_rot: torch.Tensor,
    fps: float,
    finger_savgol_window: int,
    finger_beta: float,
    finger_derivative_cutoff_hz: float,
    wrist_savgol_window: int,
    wrist_beta: float,
    finger_min_cutoff_hz: float,
    wrist_min_cutoff_hz: float,
    finger_decimate_deg: float,
    wrist_decimate_deg: float,
    wrist_max_bridge_frames: int,
) -> None:
    """In place: temporally smooth each hand's finger articulation and wrist
    orientation (HaMeR has no temporal model, so raw hands are far jitterier
    than GVHMR's body). Fingers and wrists intentionally have independent
    temporal profiles: finger articulation can skip the fixed-window pass and
    react quickly to small motions, while the load-bearing wrist keeps stronger
    stabilization. Their validity arrays also differ (fingers: base detection
    validity; wrist: the narrower one from `_compute_wrist_validity`); neither
    array is modified here, only the pose values for invalid frames.

    **The wrist channel runs this entire chain in the forearm-relative frame,
    converting back after** (`hand_retarget.wrist_relative_to_elbow` /
    `wrist_global_from_relative`). HaMeR's wrist is a *global* orientation in
    its crop's camera frame, but gap-filling (holding an occlusion, blending
    a short one) is only anatomically meaningful relative to the forearm,
    holding the global orientation while the arm keeps moving drags the real
    forearm-relative angle away from anything a wrist can do (measured on a
    real clip: a held stretch with an unchanged stored value still swung from
    95 to 166 degrees, rest is 8, because the elbow moved underneath it).
    Working in the relative frame keeps held/interpolated poses anatomically
    legal by construction, and lets the filters see real articulation instead
    of articulation plus whole-arm motion.

    The wrist alone also gets `cap_long_gaps_with_hold`, capping how much of
    a long invalid stretch gets interpolated rather than held (see that
    function's own docstring). Fingers are relative to the wrist, but their
    smoothing input is temporarily held for sustained wrist failures."""
    for pose_key, global_orient_key, valid_key, wrist_valid_key, ambiguous_key, object_contact_key, elbow in (
        (KEY_LEFT_HAND_POSE, KEY_LEFT_GLOBAL_ORIENT, KEY_LEFT_VALID, KEY_LEFT_WRIST_VALID, KEY_LEFT_FINGER_AMBIGUOUS, KEY_LEFT_FINGER_OBJECT_CONTACT, LEFT_ELBOW),
        (KEY_RIGHT_HAND_POSE, KEY_RIGHT_GLOBAL_ORIENT, KEY_RIGHT_VALID, KEY_RIGHT_WRIST_VALID, KEY_RIGHT_FINGER_AMBIGUOUS, KEY_RIGHT_FINGER_OBJECT_CONTACT, RIGHT_ELBOW),
    ):
        finger_valid = _finger_validity_after_sustained_wrist_failure(
            result[valid_key], result[wrist_valid_key], result[ambiguous_key],
        )
        finger_values, finger_smoothing_valid = cap_long_gaps_with_hold(
            result[pose_key],
            finger_valid,
            FINGER_WRIST_RECOVERY_BRIDGE_FRAMES,
        )
        finger_beta_scale = np.minimum(
            _finger_ambiguity_beta_scale(result[ambiguous_key]),
            _finger_object_contact_beta_scale(result[object_contact_key]),
        )
        result[pose_key] = _smooth_hand_channel(
            finger_values,
            finger_smoothing_valid,
            fps,
            finger_savgol_window,
            finger_min_cutoff_hz,
            finger_beta,
            finger_decimate_deg,
            finger_derivative_cutoff_hz,
            suppress_transient_reversals=True,
            beta_scale=finger_beta_scale,
        )

        elbow_global = global_rot[:, elbow]
        wrist_local = wrist_relative_to_elbow(elbow_global, torch.as_tensor(result[global_orient_key]).float())
        held_local, wrist_valid = cap_long_gaps_with_hold(
            wrist_local.numpy(), result[wrist_valid_key], wrist_max_bridge_frames
        )
        smoothed_local = _smooth_hand_channel(
            held_local,
            wrist_valid,
            fps,
            wrist_savgol_window,
            wrist_min_cutoff_hz,
            wrist_beta,
            wrist_decimate_deg,
        )
        smoothed_global = wrist_global_from_relative(elbow_global, torch.as_tensor(smoothed_local).float())
        result[global_orient_key] = smoothed_global.numpy().astype(np.float32)


def _body_joint_rotations(runRecord: RunRecord) -> torch.Tensor:
    """(F, 22, 3, 3) global rotation of every SMPL-X body joint, from stage
    2's own incam body pose, the elbow orientation both the plausibility
    gates and the forearm-relative smoothing frame are defined against.
    Loaded once here rather than per consumer, since both need the identical
    tensor."""
    motion = torch.load(
        runRecord.stages[StageName.STAGE_2_ESTIMATE_HUMAN_MOTION].outputs[OUTPUT_HUMAN_MOTION],
        weights_only=False,
    )
    incam = motion[KEY_PRED_SMPL_PARAMS_INCAM]
    return body_joint_global_rotations(
        torch.as_tensor(incam[KEY_GLOBAL_ORIENT]).float(),
        torch.as_tensor(incam[KEY_BODY_POSE]).float(),
        SmplxSkeleton().parents,
    )


def _compute_wrist_validity(result: dict, global_rot: torch.Tensor, runRecord: RunRecord) -> None:
    """In place: adds `KEY_LEFT_WRIST_VALID`/`KEY_RIGHT_WRIST_VALID` to `result`
   , each hand's base `*_valid` narrowed further wherever the raw wrist
    estimate is anatomically impossible relative to GVHMR's own elbow, changed
    implausibly fast from the previous frame, has the hand swung too far from
    its own rest-pose pointing direction, or has the hand geometrically folded
    back into the forearm's own space (see this module's docstring and
    `hand_retarget.reject_biomechanically_implausible_wrist`/`reject_wrist_
    velocity_spikes`/`reject_hand_swung_past_forearm`/`reject_hand_through_
    forearm` for why these run here, before any smoothing, rather than
    downstream in stage 5, and why the result is a separate array rather than
    overwriting `*_valid`)."""
    max_deg = runRecord.fine_tuning.hand_wrist_max_deviation_deg
    release_deg = runRecord.fine_tuning.hand_wrist_release_deviation_deg
    window = runRecord.fine_tuning.hand_wrist_deviation_window
    max_expansion_frames = runRecord.fine_tuning.hand_wrist_max_expansion_frames
    max_velocity_deg_per_sec = runRecord.fine_tuning.hand_wrist_max_velocity_deg_per_sec
    max_swing_deg = runRecord.fine_tuning.hand_wrist_max_swing_deg
    forearm_radius_m = runRecord.fine_tuning.hand_forearm_radius_m
    forearm_interior_max_t = runRecord.fine_tuning.hand_forearm_interior_max_t
    for elbow, wrist, middle1, global_orient_key, valid_key, wrist_valid_key in (
        (LEFT_ELBOW, LEFT_WRIST, LEFT_MIDDLE1, KEY_LEFT_GLOBAL_ORIENT, KEY_LEFT_VALID, KEY_LEFT_WRIST_VALID),
        (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_MIDDLE1, KEY_RIGHT_GLOBAL_ORIENT, KEY_RIGHT_VALID, KEY_RIGHT_WRIST_VALID),
    ):
        wrist_global_aa = torch.as_tensor(result[global_orient_key]).float()
        gated = reject_biomechanically_implausible_wrist(
            torch.from_numpy(result[valid_key]),
            wrist_global_aa,
            global_rot[:, elbow],
            max_deg,
            release_deg,
            window,
            max_expansion_frames,
        )
        gated = reject_wrist_velocity_spikes(gated, wrist_global_aa, runRecord.scene.fps, max_velocity_deg_per_sec)
        gated = reject_hand_swung_past_forearm(
            gated,
            wrist_global_aa,
            global_rot[:, elbow],
            rest_hand_direction(wrist, middle1),
            max_swing_deg,
        )
        forearm_offset, hand_offset = rest_forearm_and_hand_offsets(elbow, wrist, middle1)
        gated = reject_hand_through_forearm(
            gated,
            wrist_global_aa,
            global_rot[:, elbow],
            forearm_offset,
            hand_offset,
            forearm_radius_m,
            forearm_interior_max_t,
        )
        result[wrist_valid_key] = gated.numpy()


def run(runRecord: RunRecord) -> dict[str, str]:
    frames_dir = Path(runRecord.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    human_masks = torch.load(
        runRecord.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs[OUTPUT_HUMAN_MASKS],
        weights_only=False,
    )
    object_masks = None
    stage_1_outputs = runRecord.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs
    if OUTPUT_OBJECT_MASKS in stage_1_outputs:
        object_masks = torch.load(stage_1_outputs[OUTPUT_OBJECT_MASKS], weights_only=False)

    adapter = HamerAdapter()
    adapter.load()
    try:
        result = adapter.infer(frame_paths, human_masks, object_masks)
    finally:
        adapter.unload()

    global_rot = _body_joint_rotations(runRecord)
    _compute_wrist_validity(result, global_rot, runRecord)
    finger_settings = _finger_motion_settings(runRecord)

    _smooth_hand_result(
        result,
        global_rot,
        runRecord.scene.fps,
        finger_settings.hand_finger_smoothing_window,
        finger_settings.hand_finger_beta,
        finger_settings.hand_finger_derivative_cutoff_hz,
        runRecord.fine_tuning.hand_smoothing_window,
        runRecord.fine_tuning.hand_beta,
        finger_settings.hand_finger_min_cutoff_hz,
        runRecord.fine_tuning.hand_wrist_min_cutoff_hz,
        finger_settings.hand_finger_decimate_deg,
        runRecord.fine_tuning.hand_wrist_decimate_deg,
        runRecord.fine_tuning.hand_wrist_max_bridge_frames,
    )

    hands_dir = Path(runRecord.progress_dir) / HANDS_DIRNAME
    hands_dir.mkdir(parents=True, exist_ok=True)
    hand_pose_path = hands_dir / HAND_POSE_FILENAME
    np.savez(
        hand_pose_path,
        **{
            KEY_LEFT_HAND_POSE: result[KEY_LEFT_HAND_POSE],
            KEY_RIGHT_HAND_POSE: result[KEY_RIGHT_HAND_POSE],
            KEY_LEFT_GLOBAL_ORIENT: result[KEY_LEFT_GLOBAL_ORIENT],
            KEY_RIGHT_GLOBAL_ORIENT: result[KEY_RIGHT_GLOBAL_ORIENT],
            KEY_LEFT_VALID: result[KEY_LEFT_VALID],
            KEY_RIGHT_VALID: result[KEY_RIGHT_VALID],
            KEY_LEFT_WRIST_VALID: result[KEY_LEFT_WRIST_VALID],
            KEY_RIGHT_WRIST_VALID: result[KEY_RIGHT_WRIST_VALID],
            KEY_LEFT_FINGER_AMBIGUOUS: result[KEY_LEFT_FINGER_AMBIGUOUS],
            KEY_RIGHT_FINGER_AMBIGUOUS: result[KEY_RIGHT_FINGER_AMBIGUOUS],
            KEY_LEFT_FINGER_OBJECT_CONTACT: result[KEY_LEFT_FINGER_OBJECT_CONTACT],
            KEY_RIGHT_FINGER_OBJECT_CONTACT: result[KEY_RIGHT_FINGER_OBJECT_CONTACT],
        },
    )

    outputs = {OUTPUT_HAND_POSE: str(hand_pose_path)}

    if runRecord.input.render_hands_preview:
        from ..adapters.hamer.hamer_bvh_preview import render_hands_bvh

        preview_path = hands_dir / HANDS_PREVIEW_FILENAME
        render_hands_bvh(result, runRecord.scene.fps, preview_path)
        outputs[OUTPUT_HANDS_PREVIEW] = str(preview_path)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_4_ESTIMATE_HANDS)
