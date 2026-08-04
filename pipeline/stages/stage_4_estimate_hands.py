"""estimate_hands: runs HaMeR on every frame to recover per-frame MANO hand
pose for both hands, using the SAM 3.1 human mask (stage 1) to locate the
person and our COCO-17 ViTPose to locate each hand.

Output is the *raw* per-hand MANO pose (finger articulation + wrist
orientation, both in each hand crop's camera frame) plus TWO per-frame validity
flags per hand: `*_valid` (detection-based -- a hand may be off-screen, too
occluded to detect, or its keypoint confidence too low over a rolling window)
and `*_wrist_valid` (a further narrowing, wrist-only -- see below). Grafting
these onto GVHMR's body -- merging the wrist/forearm reconciliation into the
final SMPL-X body pose -- is still stage 5's job (retarget_hands), not this one.

This stage depends on stage 2 (estimate_human_motion), not just stage 0/1: a
hand is an extension of the arm, not an independent tracked object, and a real
wrist has a rotational limit relative to the forearm that HaMeR's crop-only
view has no way to know about. Before any smoothing runs, every frame's raw
wrist estimate is checked against GVHMR's own elbow orientation
(`hand_retarget.reject_biomechanically_implausible_wrist`) and treated as an
occlusion if it's anatomically impossible -- deliberately done here, before the
smoothing chain, rather than after in stage 5: a filter that's already blended
a bad value into its neighbors can't be un-blended by a later stage.

This wrist check produces its OWN validity array (`*_wrist_valid`), separate
from the base `*_valid` used for finger smoothing: a biomechanically
implausible wrist estimate doesn't mean the finger articulation from the same
frame is untrustworthy too (checked on a real clip -- finger jitter during a
bad wrist stretch was only modestly elevated, nowhere near the wrist's own
near-total breakdown). Coupling them would throw away real, usable finger
motion for every frame the wrist gate rejects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..adapters.gvhmr.gvhmr_adapter import KEY_BODY_POSE, KEY_GLOBAL_ORIENT, KEY_PRED_SMPL_PARAMS_INCAM
from ..adapters.gvhmr.gvhmr_smplx_skeleton import SmplxSkeleton
from ..adapters.hamer.hamer_adapter import (
    HamerAdapter,
    KEY_LEFT_GLOBAL_ORIENT,
    KEY_LEFT_HAND_POSE,
    KEY_LEFT_VALID,
    KEY_LEFT_WRIST_VALID,
    KEY_RIGHT_GLOBAL_ORIENT,
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
from ..algorithms.motion_smoothing import (
    cap_long_gaps_with_hold,
    decimate_rotation_sequence,
    one_euro_filter_rotation_sequence,
    smooth_rotation_sequence,
)
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName
from ..stages.stage_1_mask_and_track import OUTPUT_HUMAN_MASKS
from ..stages.stage_2_estimate_human_motion import OUTPUT_HUMAN_MOTION

# stage_0_ingest_video.py's own output key, consumed here.
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

HANDS_DIRNAME = f"stage{StageName.STAGE_4_ESTIMATE_HANDS.stage_number}_hands"
HAND_POSE_FILENAME = "hand_pose.npz"
HANDS_PREVIEW_FILENAME = "hands_preview.bvh"

# This stage's own progress.json output keys.
OUTPUT_HAND_POSE = "hand_pose"
OUTPUT_HANDS_PREVIEW = "hands_preview"


def _smooth_hand_channel(
    axis_angle: np.ndarray,
    valid: np.ndarray,
    fps: float,
    savgol_window: int,
    min_cutoff_hz: float,
    beta: float,
    decimate_deg: float,
) -> np.ndarray:
    """One hand channel (finger articulation or wrist orientation) through the
    full three-pass chain: a zero-phase savgol pre-pass to knock down HaMeR's
    broadband per-frame jitter, then the adaptive one-euro filter to hold
    nearly-still joints tight (killing the rest-state wobble/precession the savgol
    pass leaves) while tracking real motion, then quaternion-space decimation to
    refit the result through sparse keyframes so it's mathematically smooth --
    removing residual jitter outright rather than averaging it down. The order
    matters: decimation before the one-euro pass would place keyframes on the
    noise and lock it in. Occlusion handling (interpolate a gap that recovers,
    freeze one that runs to the clip's end) is carried by every pass via `valid`;
    see `motion_smoothing.fill_invalid`."""
    smoothed = smooth_rotation_sequence(axis_angle, savgol_window, valid=valid)
    smoothed = one_euro_filter_rotation_sequence(smoothed, fps, min_cutoff_hz, beta, valid=valid)
    return decimate_rotation_sequence(smoothed, decimate_deg, valid=valid)


def _smooth_hand_result(
    result: dict,
    global_rot: torch.Tensor,
    fps: float,
    savgol_window: int,
    beta: float,
    finger_min_cutoff_hz: float,
    wrist_min_cutoff_hz: float,
    finger_decimate_deg: float,
    wrist_decimate_deg: float,
    wrist_max_bridge_frames: int,
) -> None:
    """In place: temporally smooth each hand's finger articulation and wrist
    orientation (HaMeR has no temporal model, so raw hands are far jitterier
    than GVHMR's body). Both run the same `_smooth_hand_channel` chain with
    different knobs and validity arrays (fingers: base detection validity;
    wrist: the narrower one from `_compute_wrist_validity`) -- neither array
    is modified here, only the pose values for invalid frames.

    **The wrist channel runs this entire chain in the forearm-relative frame,
    converting back after** (`hand_retarget.wrist_relative_to_elbow` /
    `wrist_global_from_relative`). HaMeR's wrist is a *global* orientation in
    its crop's camera frame, but gap-filling (holding an occlusion, blending
    a short one) is only anatomically meaningful relative to the forearm --
    holding the global orientation while the arm keeps moving drags the real
    forearm-relative angle away from anything a wrist can do (measured on a
    real clip: a held stretch with an unchanged stored value still swung from
    95 to 166 degrees, rest is 8, because the elbow moved underneath it).
    Working in the relative frame keeps held/interpolated poses anatomically
    legal by construction, and lets the filters see real articulation instead
    of articulation plus whole-arm motion.

    The wrist alone also gets `cap_long_gaps_with_hold`, capping how much of
    a long invalid stretch gets interpolated rather than held (see that
    function's own docstring). Fingers get neither treatment -- they're
    relative to the wrist already and stay mostly trustworthy through a
    rejected wrist stretch (see `hand_retarget`'s own docstring)."""
    for pose_key, global_orient_key, valid_key, wrist_valid_key, elbow in (
        (KEY_LEFT_HAND_POSE, KEY_LEFT_GLOBAL_ORIENT, KEY_LEFT_VALID, KEY_LEFT_WRIST_VALID, LEFT_ELBOW),
        (KEY_RIGHT_HAND_POSE, KEY_RIGHT_GLOBAL_ORIENT, KEY_RIGHT_VALID, KEY_RIGHT_WRIST_VALID, RIGHT_ELBOW),
    ):
        result[pose_key] = _smooth_hand_channel(
            result[pose_key], result[valid_key], fps, savgol_window, finger_min_cutoff_hz, beta, finger_decimate_deg
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
            savgol_window,
            wrist_min_cutoff_hz,
            beta,
            wrist_decimate_deg,
        )
        smoothed_global = wrist_global_from_relative(elbow_global, torch.as_tensor(smoothed_local).float())
        result[global_orient_key] = smoothed_global.numpy().astype(np.float32)


def _body_joint_rotations(runRecord: RunRecord) -> torch.Tensor:
    """(F, 22, 3, 3) global rotation of every SMPL-X body joint, from stage
    2's own incam body pose -- the elbow orientation both the plausibility
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
    -- each hand's base `*_valid` narrowed further wherever the raw wrist
    estimate is anatomically impossible relative to GVHMR's own elbow, changed
    implausibly fast from the previous frame, has the hand swung too far from
    its own rest-pose pointing direction, or has the hand geometrically folded
    back into the forearm's own space (see this module's docstring and
    `hand_retarget.reject_biomechanically_implausible_wrist`/`reject_wrist_
    velocity_spikes`/`reject_hand_swung_past_forearm`/`reject_hand_through_
    forearm` for why these run here, before any smoothing, rather than
    downstream in stage 5, and why the result is a separate array rather than
    overwriting `*_valid`)."""
    max_deg = runRecord.input.hand_wrist_max_deviation_deg
    release_deg = runRecord.input.hand_wrist_release_deviation_deg
    window = runRecord.input.hand_wrist_deviation_window
    max_expansion_frames = runRecord.input.hand_wrist_max_expansion_frames
    max_velocity_deg_per_sec = runRecord.input.hand_wrist_max_velocity_deg_per_sec
    max_swing_deg = runRecord.input.hand_wrist_max_swing_deg
    forearm_radius_m = runRecord.input.hand_forearm_radius_m
    forearm_interior_max_t = runRecord.input.hand_forearm_interior_max_t
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

    adapter = HamerAdapter()
    adapter.load()
    try:
        result = adapter.infer(frame_paths, human_masks)
    finally:
        adapter.unload()

    global_rot = _body_joint_rotations(runRecord)
    _compute_wrist_validity(result, global_rot, runRecord)

    _smooth_hand_result(
        result,
        global_rot,
        runRecord.scene.fps,
        runRecord.input.hand_smoothing_window,
        runRecord.input.hand_beta,
        runRecord.input.hand_finger_min_cutoff_hz,
        runRecord.input.hand_wrist_min_cutoff_hz,
        runRecord.input.hand_finger_decimate_deg,
        runRecord.input.hand_wrist_decimate_deg,
        runRecord.input.hand_wrist_max_bridge_frames,
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
