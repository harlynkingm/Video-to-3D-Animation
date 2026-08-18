"""capture_face: runs DECA, MICA, and MediaPipe's FaceLandmarker on every
frame, then fits FLAME against MediaPipe's detected 2D landmarks
(`face_landmark_fit.fit_clip`) using DECA/MICA as the initial guess. Early-
returns `{}` when `RunInput.skip_face_capture` is set, downstream stages
(export) already tolerate a missing face output the same way they tolerate a
missing object track (see `stage_10_export.py`'s `object_shape_path is None`
branch).

Needs frames (stage 0), the human mask (stage 1), and body motion (stage 2).
One shared ViTPose pass over the human-mask bounding box derives face boxes
for both DECA and MediaPipe; stage 2's tracked head rotation feeds `fit_clip`
as a body-based orientation prior (see `face_landmark_fit.calibrate_rotation_
offset`). MediaPipe's own dense landmarks also feed a second, independent
orientation signal,
`face_pose_stabilization.stabilize_orientation`'s rigid-landmark Kabsch
rotation, anchoring `fit_clip`'s global_orient without ever seeing
chin/jaw/mouth geometry, see that module's own docstring.

ViTPose runs on the GPU while a single image-mode MediaPipe task runs on the
CPU in a bounded worker thread. DECA and MICA still load and unload in
sequence, retaining the safe GPU-memory pattern exercised against real
footage during this stage's development. This avoids the cross-*stage*
segfault risk `pipeline/run.py` documents, which is why each pipeline stage
still gets its own subprocess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pipeline.helpers.torch_helpers import sys_torch_device

from ..adapters.deca.deca_adapter import DecaAdapter, KEY_EXP, KEY_POSE, KEY_SHAPE as DECA_KEY_SHAPE, KEY_VALID as DECA_KEY_VALID
from ..adapters.face_prepass_adapter import FacePrepassAdapter
from ..adapters.face_landmarks.face_landmarks_adapter import FaceLandmarksWorker, KEY_BLENDSHAPES as MP_KEY_BLENDSHAPES, KEY_LANDMARKS, KEY_VALID as MP_KEY_VALID
from ..adapters.face_landmarks.mp2dlib import dlib68_to_arcface5, dlib68_to_flame51, mediapipe_to_dlib68
from ..adapters.gvhmr.gvhmr_adapter import (
    KEY_BODY_POSE, KEY_GLOBAL_ORIENT as GVHMR_KEY_GLOBAL_ORIENT, KEY_PRED_SMPL_PARAMS_INCAM, KEY_ROOT_MOTION_UNRELIABLE,
)
from ..adapters.gvhmr.gvhmr_smplx_skeleton import SmplxSkeleton
from ..adapters.mica.mica_adapter import MicaAdapter, KEY_SHAPE as MICA_KEY_SHAPE, KEY_VALID as MICA_KEY_VALID
from ..algorithms.face.face_blendshapes import direct_arkit_jaw_channels, solve_arkit_weights
from ..algorithms.face.face_eyelid import eyelid_arkit_channels
from ..algorithms.face.face_gaze import direct_arkit_gaze_channels, eye_euler_degrees
from ..algorithms.face.face_landmark_fit import (
    FACE_ONE_EURO_EXPR_BETA, FACE_ONE_EURO_EXPR_MIN_CUTOFF_HZ,
    FitInputs, KEY_BETAS, KEY_EXPRESSION, KEY_GLOBAL_ORIENT, KEY_JAW_POSE, KEY_TRANSL, KEY_VALID as FIT_KEY_VALID,
    fit_clip, local_landmark_delta,
)
from ..algorithms.face.face_pose_stabilization import stabilize_orientation
from ..algorithms.face.face_preview import write_arkit_preview, write_flame_preview, write_landmark_preview
from ..algorithms.hand_retarget import body_joint_global_rotations
from ..algorithms.motion_smoothing import fill_invalid, one_euro_filter_sequence, smooth_position_sequence
from ..helpers.livelink_csv import ARKIT_BLENDSHAPE_NAMES, HEAD_EYE_COLUMN_NAMES, write_livelink_csv
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName
from ..stages.stage_1_mask_and_track import OUTPUT_HUMAN_MASKS
from ..stages.stage_2_estimate_human_motion import OUTPUT_HUMAN_MOTION

from scripts.diagnostic.build_ground_truth_report import build as build_ground_truth_report

# stage_0_ingest_video.py's own output key, consumed here (not exported as a
# named constant there, see stage_4_estimate_hands.py's identical pattern).
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

# SMPL-X body-joint index of "Head" (smplx_bvh_preview.BODY_JOINT_NAMES[15]),
# used to pull the body-based-orientation-prior rotation out of
# body_joint_global_rotations' full (F, 22, 3, 3) per-joint output.
HEAD_JOINT_INDEX = 15

FACE_DIRNAME = f"stage{StageName.STAGE_9_CAPTURE_FACE.stage_number}_face"
FACE_PARAMS_FILENAME = "face_params.npz"
FACE_MOTION_FILENAME = "face_motion.npz"

# This stage's own progress.json output keys. The face_preview_* pair is
# only present when RunInput.render_face_preview is set, see
# face_preview.write_flame_preview's own docstring for what they contain.
OUTPUT_FACE_PARAMS = "face_params"
OUTPUT_FACE_MOTION = "face_motion"
OUTPUT_FACE_CSV = "face_csv"


def _run_landmark_adapters(frame_paths: list[Path], human_masks: dict, device: torch.device) -> dict[str, np.ndarray]:
    """Pipelines GPU ViTPose with one CPU MediaPipe worker, then DECA/MICA."""
    prepass = FacePrepassAdapter(device=device)
    mp_worker = FaceLandmarksWorker(len(frame_paths))
    mp_worker.start()
    try:
        prepass.load()
        try:
            face_boxes = prepass.infer(frame_paths, human_masks, on_face_boxes=mp_worker.submit)
        finally:
            prepass.unload()
    finally:
        # Finish drains the bounded queue and re-raises a worker error.
        # It also guarantees the task is closed if localization fails.
        mp_out = mp_worker.finish()

    deca = DecaAdapter(device=device)
    deca.load()
    try:
        deca_out = deca.infer(frame_paths, face_boxes)
    finally:
        deca.unload()

    # MICA needs 5-point ArcFace landmarks, derived from MediaPipe's 478 via
    # the shared Dlib-68 correspondence (see mp2dlib.py), the same table
    # that gives FLAME's own 51-point set for the fitting loop below.
    n = len(frame_paths)
    flame51 = np.zeros((n, 51, 2), dtype=np.float32)
    arcface5 = np.zeros((n, 5, 2), dtype=np.float32)
    for i in range(n):
        if not mp_out[MP_KEY_VALID][i]:
            continue
        dlib68 = mediapipe_to_dlib68(mp_out[KEY_LANDMARKS][i][:, :2])
        flame51[i] = dlib68_to_flame51(dlib68)
        arcface5[i] = dlib68_to_arcface5(dlib68)

    mica = MicaAdapter(device=device)
    mica.load()
    try:
        mica_out = mica.infer(frame_paths, arcface5, mp_out[MP_KEY_VALID])
    finally:
        mica.unload()

    return {
        "deca_shape": deca_out[DECA_KEY_SHAPE], "deca_exp": deca_out[KEY_EXP], "deca_pose": deca_out[KEY_POSE],
        "deca_valid": deca_out[DECA_KEY_VALID],
        "mica_shape": mica_out[MICA_KEY_SHAPE], "mica_valid": mica_out[MICA_KEY_VALID],
        "mp_landmarks": mp_out[KEY_LANDMARKS], "mp_valid": mp_out[MP_KEY_VALID],
        "mp_blendshapes": mp_out[MP_KEY_BLENDSHAPES],
        "flame51": flame51,
    }


def _body_head_rotation(runRecord: RunRecord) -> tuple[np.ndarray, np.ndarray]:
    """Body-based orientation prior: stage 2's own tracked Head-joint rotation,
    already in the same camera frame `fit_clip` projects into, since both
    this stage and stage 2 consume the same `runRecord.scene.intrinsics_K`
   , plus a per-frame trust flag. Reuses `hand_retarget.body_joint_global_
    rotations`, the same helper stages 4/5 use to interpret the elbow for
    wrist reconciliation, just indexed at the Head joint instead."""
    motion = torch.load(
        runRecord.stages[StageName.STAGE_2_ESTIMATE_HUMAN_MOTION].outputs[OUTPUT_HUMAN_MOTION],
        weights_only=False,
    )
    incam = motion[KEY_PRED_SMPL_PARAMS_INCAM]
    global_orient = torch.as_tensor(incam[GVHMR_KEY_GLOBAL_ORIENT]).float()
    body_pose = torch.as_tensor(incam[KEY_BODY_POSE]).float()
    global_rotmats = body_joint_global_rotations(global_orient, body_pose, SmplxSkeleton().parents)
    head_rotmat = global_rotmats[:, HEAD_JOINT_INDEX].numpy()

    root_motion_unreliable = np.asarray(motion[KEY_ROOT_MOTION_UNRELIABLE])
    return head_rotmat, ~root_motion_unreliable


def _optionally_render_ground_truth_comparison(runRecord: RunRecord) -> None:
    """Render an HTML comparison to the ground-truth LiveLink CSV, if one exists in the folder."""
    run_dir = Path(runRecord.progress_dir)
    # Only render ground truth report if a raw csv is detected in the run folder
    matches = sorted(run_dir.glob("*_raw.csv"))
    if len(matches) > 0:
        build_ground_truth_report(run_dir)


def run(runRecord: RunRecord) -> dict[str, str]:
    if runRecord.input.skip_face_capture:
        return {}

    frames_dir = Path(runRecord.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    human_masks = torch.load(
        runRecord.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs[OUTPUT_HUMAN_MASKS],
        weights_only=False,
    )

    device = sys_torch_device()
    params = _run_landmark_adapters(frame_paths, human_masks, device)

    # Smoothing the raw landmarks here (before the fit ever sees them) is the
    # correct place for it: `fit_clip`'s own temporal_weight penalizes the
    # *fitted parameters'* deltas, which structurally can't tell a real fast
    # transient (a blink) from single-frame detection noise, see
    # `FineTuningOptions.face_smoothing_window`'s own comment.
    smoothed_landmarks_51 = smooth_position_sequence(
        params["flame51"], runRecord.fine_tuning.face_smoothing_window, valid=params["mp_valid"],
    )

    head_rotmat, head_confidence = _body_head_rotation(runRecord)
    kabsch_rotmat = stabilize_orientation(params["mp_landmarks"], params["mp_valid"])

    fit_inputs = FitInputs(
        landmarks_51=smoothed_landmarks_51,
        landmarks_valid=params["mp_valid"],
        deca_exp=params["deca_exp"],
        deca_pose=params["deca_pose"],
        deca_shape=params["deca_shape"],
        deca_valid=params["deca_valid"],
        mica_shape=params["mica_shape"],
        mica_valid=params["mica_valid"],
        intrinsics_k=np.array(runRecord.scene.intrinsics_K),
        head_rotmat=head_rotmat,
        head_confidence=head_confidence,
        kabsch_rotmat=kabsch_rotmat,
        kabsch_confidence=params["mp_valid"],
    )
    motion = fit_clip(fit_inputs, device=device, fps=runRecord.scene.fps)

    face_dir = Path(runRecord.progress_dir) / FACE_DIRNAME
    face_dir.mkdir(parents=True, exist_ok=True)

    params_path = face_dir / FACE_PARAMS_FILENAME
    np.savez(
        params_path,
        deca_shape=params["deca_shape"], deca_exp=params["deca_exp"], deca_pose=params["deca_pose"],
        deca_valid=params["deca_valid"],
        mica_shape=params["mica_shape"], mica_valid=params["mica_valid"],
        mp_landmarks=params["mp_landmarks"], mp_valid=params["mp_valid"],
        mp_blendshapes=params["mp_blendshapes"],
    )

    motion_path = face_dir / FACE_MOTION_FILENAME
    np.savez(
        motion_path,
        **{
            KEY_BETAS: motion[KEY_BETAS], KEY_EXPRESSION: motion[KEY_EXPRESSION],
            KEY_GLOBAL_ORIENT: motion[KEY_GLOBAL_ORIENT], KEY_JAW_POSE: motion[KEY_JAW_POSE],
            KEY_TRANSL: motion[KEY_TRANSL], FIT_KEY_VALID: motion[FIT_KEY_VALID],
        },
    )

    # Computed once, shared by the CSV and (if requested) the ARKit preview
    # below, the preview must show exactly what the CSV contains, never a
    # second, potentially-drifted recomputation.
    arkit_weights, head_eye_euler = _compute_arkit_channels(motion, params, device, runRecord.scene.fps)
    csv_path = Path(runRecord.progress_dir) / "output_face.csv"
    write_livelink_csv(csv_path, arkit_weights, head_eye_euler, fps=runRecord.scene.fps)
    runRecord.outputs.final_face_csv = str(csv_path)

    outputs = {OUTPUT_FACE_PARAMS: str(params_path), OUTPUT_FACE_MOTION: str(motion_path), OUTPUT_FACE_CSV: str(csv_path)}
    if runRecord.input.render_face_preview:
        outputs.update(write_flame_preview(motion, face_dir, device=device))
        outputs.update(write_landmark_preview(
            params["mp_landmarks"], params["mp_valid"], runRecord.fine_tuning.face_smoothing_window, face_dir,
        ))
        outputs.update(write_arkit_preview(arkit_weights, head_eye_euler, face_dir))
        # Render HTML comparison to ground-truth LiveLink CSV, if one exists in the folder
        _optionally_render_ground_truth_comparison(runRecord)
    return outputs


# ARKit-52 channels that stay hard zero even with MediaPipe's own native
# blendshapes available, see this function's own docstring for why each
# one specifically. Every other name in ARKIT_BLENDSHAPE_NAMES not already
# covered by this project's own tracked state falls through to MediaPipe.
HARD_ZERO_CHANNELS = frozenset({
    "TongueOut",  # no source anywhere: MediaPipe's own 52 categories omit it too (Google's model never predicts tongue)
    "CheekPuff",  # MediaPipe's own value correlates *negatively* against real ground truth on all 3 real
                  # clips tested (-0.42 to -0.61), worse than flat zero, so kept hard zero (Group W) on purpose
})

# `eyelid_arkit_channels` still computes both blink and wide from dense
# landmarks, but the six-capture LiveLink comparison found native, smoothed
# MediaPipe is consistently the better EyeWide source. Keep this narrow
# allow-list so the merge below makes the remaining pipeline-owned eyelid
# channels explicit rather than relying on an incidental dictionary shape.
PIPELINE_EYELID_CHANNELS = frozenset({"EyeBlinkLeft", "EyeBlinkRight"})

# The FLAME jaw pose remains saved in face_motion.npz and still supplies the
# lateral jaw channels. Six paired LiveLink captures consistently favor the
# native, smoothed MediaPipe score for open/close, so JawOpen joins the other
# MediaPipe-sourced ARKit channels in the exported CSV.
FLAME_JAW_CHANNELS = frozenset({"JawLeft", "JawRight"})


def _smoothed_mediapipe_blendshapes(mp_blendshapes: np.ndarray, mp_valid: np.ndarray, fps: float) -> np.ndarray:
    """Gap-fills MediaPipe detection dropout and one-euro smooths its own
    raw per-frame blendshape scores, reusing the exact constants already
    shipped for FLAME's own `expression` (0.6Hz/beta=1) rather than a
    separate tuning pass, measured directly against real ground truth
    (not assumed) that this roughly halves MediaPipe's own raw jitter
    (0.037->0.017 on a representative channel) while correlation holds
    steady or improves slightly on most channels."""
    filled = fill_invalid(mp_blendshapes, mp_valid)
    return one_euro_filter_sequence(filled, fps, FACE_ONE_EURO_EXPR_MIN_CUTOFF_HZ, FACE_ONE_EURO_EXPR_BETA)


def _compute_arkit_channels(
    motion: dict, params: dict, device: torch.device, fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assembles all 52 ARKit channels plus the 9 head/eye Euler columns,
    the shared computation behind both `output_face.csv` and the
    `ARKit_face_preview.blend` visualization (`face_preview.write_arkit_
    preview`), so the preview always shows exactly what the CSV contains,
    never a second, potentially-drifted recomputation. Returns
    `(arkit_weights (F, 52), head_eye_euler (F, 9))`, both in
    `livelink_csv`'s own column order.

    Each channel's source is chosen per measured accuracy against MediaPipe's
    native blendshape output (see `face_blendshapes.py`'s docstring):
    - Lateral jaw (`JawLeft`/`Right`), horizontal gaze (`EyeLookIn/Out*`),
      blink (`EyeBlink*`), and `NoseSneer*`/`CheekSquint*`/`EyeSquint*`:
      this project's own tracked state.
    - Everything else (including `JawOpen`, `JawForward`, `EyeWide*`, all
      `Brow*`, and most `Mouth*`): MediaPipe's own native blendshape output, smoothed with
      the same one-euro filter this project's other temporal signals use
      (`_smoothed_mediapipe_blendshapes`). The six paired LiveLink captures
      compared in `ground_truth_report.html` consistently favor this source
      for both EyeWide channels and JawOpen. The FLAME jaw pose is still
      saved unchanged in `face_motion.npz` for diagnostics and future fusion.
      `JawForward` is a genuine new capability here regardless, this project's
      own tracked state structurally can't produce it (SMPL-X's jaw joint has
      no translation DOF).
    - `TongueOut`/`CheekPuff`: hard zero either way (see
      `HARD_ZERO_CHANNELS`).
    """
    jaw_pose, expression = motion[KEY_JAW_POSE], motion[KEY_EXPRESSION]
    mp_landmarks = params["mp_landmarks"]
    # `flame_valid` contains the long-gap recovery gate from the fitter, as
    # well as DECA's own detector validity. Only its *newly rejected* frames
    # should gate MediaPipe channels: a brief DECA-only miss must retain the
    # pre-existing MediaPipe export behavior. This shared recovery mask keeps
    # a failed re-acquisition out of every face shape while FLAME jaw and
    # expression are being held at the same time.
    base_fit_valid = params["mp_valid"] & params["deca_valid"]
    recovery_rejected = base_fit_valid & ~motion[FIT_KEY_VALID]
    mp_valid = params["mp_valid"] & ~recovery_rejected

    jaw_channels = direct_arkit_jaw_channels(jaw_pose)
    lateral_jaw_channels = {name: jaw_channels[name] for name in FLAME_JAW_CHANNELS}
    eyelid_channels = eyelid_arkit_channels(mp_landmarks, mp_valid, fps)
    blink_channels = {name: eyelid_channels[name] for name in PIPELINE_EYELID_CHANNELS}
    gaze_channels = direct_arkit_gaze_channels(mp_landmarks, mp_valid, fps)

    landmark_delta = local_landmark_delta(jaw_pose, expression, device=device)
    solved_channels = solve_arkit_weights(landmark_delta, jaw_pose, valid=motion[FIT_KEY_VALID])

    mp_blendshapes = _smoothed_mediapipe_blendshapes(params["mp_blendshapes"], mp_valid, fps)

    n = len(jaw_pose)
    ours = {**lateral_jaw_channels, **gaze_channels, **blink_channels, **solved_channels}
    arkit_weights = np.zeros((n, len(ARKIT_BLENDSHAPE_NAMES)), dtype=np.float32)
    for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
        if name in ours:
            arkit_weights[:, i] = ours[name]
        elif name not in HARD_ZERO_CHANNELS:
            arkit_weights[:, i] = mp_blendshapes[:, i]
        # else: TongueOut/CheekPuff, stays 0.

    euler = eye_euler_degrees(mp_landmarks, mp_valid, fps)
    eye_euler = {f"eye{i}": euler[:, i] for i in range(6)}
    head_eye_euler = np.zeros((n, len(HEAD_EYE_COLUMN_NAMES)), dtype=np.float32)
    head_eye_euler[:, 3:] = np.stack([eye_euler[f"eye{i}"] for i in range(6)], axis=1)  # HeadYaw/Pitch/Roll stay 0

    return arkit_weights, head_eye_euler


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_9_CAPTURE_FACE)
