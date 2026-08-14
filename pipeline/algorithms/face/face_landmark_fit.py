"""Per-frame FLAME fitting against 2D landmark reprojection error.

Given DECA's per-frame initial guess (shape/expression/pose), MICA's
per-frame identity estimate, and MediaPipe's per-frame detected FLAME-51
landmarks (via `mp2dlib.py`) in full-frame pixel coordinates, this solves for
one shared identity (`betas`) across the whole clip plus per-frame
expression/jaw/head-rotation/head-translation, by gradient descent (Adam)
through `smplx.FLAME`'s own differentiable forward pass, the one place in
this pipeline that needs an actual optimization loop rather than a
closed-form fit, since a landmark's position is a nonlinear function of
rotation and expression.

Two sequential stages, never one joint optimization: Stage 1 optimizes only
`global_orient`/`transl`, with `jaw_pose`/`expression` frozen at the DECA
init; Stage 2, with orientation now frozen, optimizes `jaw_pose`/`expression`
as bounded deltas from that same init. Orientation and jaw/expression never
share a gradient step, so neither can trade off against the other to absorb
residual reprojection error at the other's expense.

Scope, deliberately: gaze (`leye_pose`/`reye_pose`) is not fitted here and
stays at zero. FLAME's static 51-point landmark set has no iris/pupil
points, so it can't constrain gaze direction at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...adapters.gvhmr.gvhmr_rotation_math import axis_angle_to_matrix
from ..contact_detection import contiguous_true_runs
from ...helpers.progress_reporter import frame_progress
from ..motion_smoothing import (
    cap_long_gaps_with_hold,
    fill_invalid,
    hemisphere_aligned_quats,
    one_euro_filter_rotation_sequence,
    one_euro_filter_sequence,
)
from pipeline.progress_tracker import StageName

FLAME_MODEL_DIR = Path(__file__).resolve().parents[3] / "body_models"
FLAME_MODEL_TYPE = "flame"
FLAME_GENDER = "neutral"
FLAME_NUM_BETAS = 300  # FLAME 2020's full shape-identity space (matches MICA's own output dim)
FLAME_NUM_EXPRESSION = 50  # matches DECA's own n_exp
NUM_FLAME_LANDMARKS = 51  # `use_face_contour=False`: FLAME's static embedding only

_FIT_ORIENTATION_LABEL = f"{StageName.STAGE_9_CAPTURE_FACE.label} 4/5 (FLAME fit, head pose)"
_FIT_EXPRESSION_LABEL = f"{StageName.STAGE_9_CAPTURE_FACE.label} 5/5 (FLAME fit, expression)"

# Reference points within the 51-point FLAME/dlib-inner set (0-indexed within
# the 51, i.e. dlib index - 17) used to initialize head depth via the
# interocular-distance heuristic below.
_LEYE_51, _REYE_51 = 19 - 17, 28 - 17  # dlib 19 (left eye outer corner), 28 (right eye outer corner)

# Stage 1 (orientation only): global_orient + transl, jaw_pose/expression
# frozen at the DECA init. Iteration count/LR-decay schedule matches
# `flame-head-tracker`'s own validated rigid-pose stage (`tracker_base.py`'s
# `run_rigid_camera_pose_fitting`, 1500 iters, LR dropped 5x at iter 1000),
# confirmed on the hardest real frames tested: every one converges to
# 1.5-2.4px mean reprojection residual, not just "doesn't blow up".
DEFAULT_STAGE1_ITERS = 1500
DEFAULT_STAGE1_LR = 0.01
STAGE1_LR_DECAY_ITER = 1000
STAGE1_LR_DECAY_FACTOR = 0.2

# Stage 2 (jaw_pose + expression only): global_orient/transl frozen from
# Stage 1. Both parameters are bounded-tanh reparameterizations initialized
# at the DECA estimate (see `_bounded_tanh`/`_inverse_bounded_tanh`),
# iteration count/LR schedule again matches `flame-head-tracker`'s own
# validated fine-fitting stage (200 iters, LR dropped 2x at iter 100).
DEFAULT_STAGE2_ITERS = 200
DEFAULT_STAGE2_LR = 0.01
STAGE2_LR_DECAY_ITER = 100
STAGE2_LR_DECAY_FACTOR = 0.5

# Hard per-axis bounds on jaw_pose (axis-angle), enforced by construction via
# a tanh reparameterization in Stage 2, not a soft penalty, so no anchor
# weight, however large, can be outvoted the way a soft jaw anchor can.
# Axis0 (opening) has real, large-magnitude motion in practice; axis1 (yaw)
# and axis2 (roll) barely move at all, so they're bounded much tighter to
# keep a weakly-constrained fit from swinging them into an anatomically
# implausible pose (e.g. the jaw rotating in yaw/roll well past what a real
# mouth does). Calibrated against real per-clip DECA ranges, not a
# physiological constant.
JAW_AXIS0_BOUND_RAD = 0.8  # opening/closing
JAW_AXIS1_BOUND_RAD = 0.08  # yaw (side-to-side)
JAW_AXIS2_BOUND_RAD = 0.4  # roll/twist

# Hard bound on `expression` (per component, same tanh mechanism as jaw
# above). Without it, an underdetermined reprojection fit can push
# `expression`'s norm arbitrarily high with nothing structurally stopping
# it. A safety net, not the primary regularizer, a well-fit clip's own
# expression norm stays well inside this bound on its own.
EXPRESSION_BOUND = 6.0

# Gaussian prior pulling expression toward neutral, applied on top of the
# hard bound above as an additional soft term: FLAME's expression
# coefficients are PCA units and a real expression sits within a modest
# range, but 2D landmark reprojection alone is too weak a constraint to keep
# the optimizer from drifting well past that to shave off residual pixel
# error, producing a mesh that reprojects correctly but looks like a
# caricature. This weight trades expression magnitude off against blink
# depth specifically, too high and a real blink's closure gets shallower.
# Frame-to-frame jitter is handled separately, by
# `RunInput.face_smoothing_window` and the one-euro post-filter below, not
# by this weight.
DEFAULT_EXPRESSION_WEIGHT = 20.0

# Soft regularizer anchoring global_orient to DECA's own per-frame estimate,
# in Stage 1 (where global_orient is the only free rotation, so this and the
# Kabsch anchor below are genuine complementary priors, not competing votes
# in a coupled system the way they were in the original single-stage
# design). Uses a chordal-distance (relative-rotation) loss, not a raw
# axis-angle vector difference: global_orient can sit near the axis-angle
# singularity, a real failure mode, hit twice during this anchor's own
# development, not hypothetical.
DEFAULT_GLOBAL_ORIENT_ANCHOR_WEIGHT = 10.0

# Soft regularizer anchoring global_orient to `face_pose_stabilization.
# stabilize_orientation`'s rigid-landmark Kabsch rotation (see that module's
# own docstring), fit against a fixed 19-point subset that excludes all
# chin/lips/brow/nose-wing points by construction, so a foreshortened jaw or
# wide-open mouth literally isn't in its input. Same chordal-distance
# mechanism as the DECA anchor above.
DEFAULT_KABSCH_ANCHOR_WEIGHT = 10.0
# Below this many confidently-detected frames, calibrating the fixed offset
# against DECA isn't trustworthy, same reasoning as
# MIN_HEAD_CALIBRATION_FRAMES below (defined separately since the two
# calibrations gate on different confidence signals: mp_valid here, GVHMR's
# root_motion_unreliable there).
MIN_KABSCH_CALIBRATION_FRAMES = 3

# Soft regularizer pulling global_orient toward the body-as-prior signal
# (GVHMR's own Head-joint rotation, offset-corrected, see
# calibrate_rotation_offset). Same order of magnitude as temporal_weight
# since both regularize a per-frame axis-angle quantity, but this one should
# be trusted more under occlusion, GVHMR's head track stays reliable
# exactly where the face landmarks don't.
DEFAULT_HEAD_PRIOR_WEIGHT = 10.0
# Below this many confidently-tracked frames, a calibrated offset isn't
# trustworthy (could be one lucky/unlucky outlier), skip the whole prior for
# that clip rather than risk a bad calibration actively misleading the fit.
MIN_HEAD_CALIBRATION_FRAMES = 3
# GVHMR's incam global_orient convention isn't camera-relative, so the
# calibrated head-rotation prior can land close enough to a 180 deg rotation
# that axis-angle's [0, pi]-clamped representation becomes ill-conditioned:
# small, real per-frame noise gets amplified into large swings in the
# composed rotation even though the underlying motion is smooth. Rather than
# fix the ill-conditioning itself, this detects it and falls back to
# DECA-only for that clip, never worse than not having the prior at all.
MAX_SAFE_PRIOR_ANGLE_DEG = 150.0

# A plain L2 delta penalty on frame-to-frame parameter changes was tried and
# rejected: it suppresses the *largest* frame-to-frame jumps hardest, which
# are exactly real fast transients like a blink, not small landmark noise,
# a uniform quadratic penalty can't separate "one frame of real motion" from
# "many frames of small noise". The Huber loss below has a narrower job:
# `_bridge_short_gaps` already handles brief detection dropouts and the
# one-euro post-filter (below) already handles ambient noise on valid
# frames, so this term only needs to damp in-loop single-frame Adam outliers
# during optimization, quadratic below its own delta threshold, linear
# (not crushing) above it. Per-quantity deltas are set near each quantity's
# own normal frame-to-frame variation range, below where outliers start.
STAGE1_ORIENT_TEMPORAL_HUBER_DELTA = 0.5  # rad, global_orient's own axis-angle vector delta
STAGE1_TRANSL_TEMPORAL_HUBER_DELTA = 0.15  # meters
STAGE2_JAW_TEMPORAL_HUBER_DELTA = 0.15  # rad
STAGE2_EXPR_TEMPORAL_HUBER_DELTA = 1.5  # PCA units

# A single global temporal weight can't tell "many frames of small noise"
# from "one frame of real fast motion", pushed high enough to smooth
# ambient jitter, it also flattens real transients like a blink.
# `_detect_outlier_frames` handles single-frame optimization outliers
# deterministically instead (a frame whose fitted value disagrees sharply
# with both its immediate neighbors, unlike genuine fast motion, which moves
# consistently across consecutive frames), so this weight only needs to
# stabilize the optimization itself and feed that detector a sane series to
# compare against, the one-euro post-filter (below) carries the actual
# ambient-smoothness job.
DEFAULT_TEMPORAL_WEIGHT = 600.0

# Thresholds are real-data-derived the same way as MAX_BRIDGE_FRAMES: on a
# real clip's actual output, "deviation from the linear/slerp midpoint of a
# frame's own immediate neighbors" is small and gently-growing up to ~p99
# for every quantity, then confirmed real outliers sit clearly past it (jaw
# 0.25/0.15 vs. p99 0.094; expression 0.50/0.30 vs. p99 0.233; global_orient
# 2.08 rad vs. p99 1.13), while known-already-fixed outliers measure
# near-zero on this same metric, confirming it only fires on frames still
# genuinely wrong, not already-healthy ones. Set between p99 and the
# confirmed outliers' own values.
JAW_OUTLIER_DEVIATION_THRESHOLD = 0.12  # rad
EXPR_OUTLIER_DEVIATION_THRESHOLD = 0.27  # PCA units
# Deliberately conservative (well above global_orient's own p95=0.53, closer
# to p99=1.13), a real fast head turn can legitimately produce a larger
# midpoint deviation than jaw/expression's own everyday range, so this
# threshold only needs to catch the unambiguous case (idx190's 2.08), not
# the ambiguous middle ground jaw/expression don't have to worry about.
GO_OUTLIER_DEVIATION_THRESHOLD = 1.3  # rad
# A real single-frame optimization spike is virtually always exactly one
# frame, capped low so this mechanism can't be mistaken for a second,
# looser gap-bridge and start smoothing over genuine short bursts of fast
# motion.
MAX_OUTLIER_BRIDGE_FRAMES = 2

# Jaw-specific overrides: a jaw optimization outlier can spike in pairs or
# triples of consecutive frames, not just single frames, which
# `_detect_outlier_frames`'s default 2-point midpoint (zero tolerance for a
# corrupted immediate neighbor) misses entirely. `JAW_OUTLIER_WINDOW=3`
# widens the reference to a median over up to 3 frames on each side,
# tolerating up to 2 corrupted neighbors on one side without being pulled
# toward them. `JAW_OUTLIER_BRIDGE_FRAMES` is sized to bridge a burst that
# long once flagged.
JAW_OUTLIER_WINDOW = 3
JAW_OUTLIER_BRIDGE_FRAMES = 3

# A second, independent outlier signal for jaw specifically, catching what
# `_detect_outlier_frames`'s neighbor-midpoint check structurally can't: a
# *sustained* multi-frame drift away from DECA, where each frame agrees with
# its own immediate neighbors (so it's invisible to a neighbor-comparison
# check) while the whole stretch drifts far from DECA's own stable estimate.
# Scoped to axis1 (yaw) specifically, axis0/axis2 both have legitimate
# motion that overlaps this kind of deviation on real data, so neither has a
# safe separating threshold, while axis1's real motion stays small enough
# that a drifted stretch is unambiguous. Flagged frames are snapped directly
# to DECA's own jaw_pose (all 3 axes).
JAW_AXIS1_DECA_DEVIATION_THRESHOLD = 0.05  # rad

# `_bridge_short_gaps`'s own length cutoff: a detection dropout at or under
# this many frames is treated as a brief flicker and bridged via
# interpolation between the flanking valid frames' own fitted values;
# anything longer is left to the anchor-driven fit (DECA/Kabsch/body-based
# orientation prior). For jaw/expression specifically, a long gap has no
# anchor pulling it back toward anything during the optimization itself, so
# it's frozen at the entry value via `cap_long_gaps_with_hold` instead,
# blending into the real recovery only over the final `MAX_BRIDGE_FRAMES`.
# Real detection dropouts on a typical clip fall into two clearly separated
# clusters, brief flicker vs. real occlusion, so this sits in the gap
# between them.
MAX_BRIDGE_FRAMES = 8

# A valid run shorter than this is too brief to trust as an independent
# anchor point, an isolated detection (or a couple) surrounded by invalid
# frames, with no neighbors to cross-check against, the same "isolated
# frame, don't trust it alone" reasoning as the per-frame optimization-
# outlier check, applied to detection noise instead. Demoted to invalid and
# folded into its surrounding gap, handled by the existing long-gap hold or
# short-gap bridge like any other occlusion.
MIN_VALID_RUN_FRAMES = 6

# Adaptive one-euro smoothing for jaw_pose/expression, applied as a post-hoc
# pass on Stage 2's already bridged/held/outlier-corrected output. The
# in-loop Huber term only stabilizes the optimization itself and feeds the
# outlier detector a sane series, at a weight low enough not to flatten
# real transients, it's too weak to absorb ordinary per-frame landmark
# noise on its own. Mirrors the adaptive filter already used for finger
# joints (`motion_smoothing.one_euro_filter_rotation_sequence`): heavy
# smoothing while a quantity is nearly still, loosening automatically once
# it starts moving fast, so ambient jitter is damped without adding lag to
# real motion. Runs last, after every bridging/hold/outlier pass, so it only
# ever sees an already-continuous series with no discontinuity at a gap
# boundary. min_cutoff_hz/beta are calibrated per quantity, not shared with
# the hand-joint filter's own values, jaw and expression each have a
# different jitter-vs-lag trade-off.
FACE_ONE_EURO_JAW_MIN_CUTOFF_HZ = 0.15
FACE_ONE_EURO_JAW_BETA = 4.0
FACE_ONE_EURO_EXPR_MIN_CUTOFF_HZ = 0.6
FACE_ONE_EURO_EXPR_BETA = 1.0

# A soft ridge pulling jaw_pose/expression toward DECA's own per-frame
# estimate throughout Stage 2's optimization, masked to valid frames only,
# a genuine during-optimization regularizer, distinct from the long-
# occlusion hold above (an invalid frame already stays at its own DECA init
# with no gradient at all, so this term never substitutes for missing data).
# Without it, the underdetermined jaw/expression fit can diverge to an
# implausible solution even on fully-valid, well-behaved frames, since
# reprojection alone doesn't sufficiently constrain either parameter. A
# uniform weight, so it pulls equally everywhere rather than only where the
# fit actually drifts, a future pass could scope it to just the frames
# that need it, the same way `_detect_outlier_frames` scopes other
# corrections.
DEFAULT_JAW_DECA_ANCHOR_WEIGHT = 3000.0
DEFAULT_EXPR_DECA_ANCHOR_WEIGHT = 3000.0

# fit_clip() output keys.
KEY_BETAS = "flame_betas"  # (300,), one shared identity for the whole clip
KEY_EXPRESSION = "flame_expression"  # (F, 50)
KEY_GLOBAL_ORIENT = "flame_global_orient"  # (F, 3) axis-angle
KEY_JAW_POSE = "flame_jaw_pose"  # (F, 3) axis-angle
KEY_TRANSL = "flame_transl"  # (F, 3) camera-space head translation
KEY_VALID = "flame_valid"


@dataclass
class FitInputs:
    """Per-frame inputs to `fit_clip`, already aligned to the same F frames."""

    landmarks_51: np.ndarray  # (F, 51, 2) full-frame pixel coords, from mp2dlib.dlib68_to_flame51
    landmarks_valid: np.ndarray  # (F,) bool
    deca_exp: np.ndarray  # (F, 50)
    deca_pose: np.ndarray  # (F, 6): [:3] global_orient, [3:6] jaw_pose
    deca_shape: np.ndarray  # (F, 100)
    deca_valid: np.ndarray  # (F,) bool
    mica_shape: np.ndarray  # (F, 300)
    mica_valid: np.ndarray  # (F,) bool
    intrinsics_k: np.ndarray  # (3, 3)
    # Body-based orientation prior, both optional (None disables it entirely, e.g.
    # for standalone/test use): GVHMR's own Head-joint global rotation for
    # this clip, already in the same camera frame this module projects into
    # (both consume runRecord.scene.intrinsics_K, see
    # stage_9_capture_face.py for how these are produced), and a per-frame
    # trust flag (e.g. ~root_motion_unreliable) gating where GVHMR's own head
    # track is itself confident.
    head_rotmat: np.ndarray | None = None  # (F, 3, 3)
    head_confidence: np.ndarray | None = None  # (F,) bool
    # `face_pose_stabilization.stabilize_orientation`'s rigid-landmark Kabsch
    # rotation, relative to this clip's own neutral-frame template, and the
    # mp_valid mask it was computed from. Optional for the same reason as the
    # two fields above (standalone/test use).
    kabsch_rotmat: np.ndarray | None = None  # (F, 3, 3)
    kabsch_confidence: np.ndarray | None = None  # (F,) bool


def _build_flame_model(device: torch.device, batch_size: int):
    import smplx

    return smplx.create(
        str(FLAME_MODEL_DIR),
        model_type=FLAME_MODEL_TYPE,
        gender=FLAME_GENDER,
        ext="npz",
        num_betas=FLAME_NUM_BETAS,
        num_expression_coeffs=FLAME_NUM_EXPRESSION,
        use_face_contour=False,
        create_betas=False, create_expression=False, create_global_orient=False,
        create_neck_pose=False, create_jaw_pose=False, create_leye_pose=False, create_reye_pose=False,
        create_transl=False,
        batch_size=batch_size,
    ).to(device)


def _project_points(points_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """points_cam: (..., 3) camera-space points, Z > 0. K: (3, 3). Returns (..., 2) pixel coords."""
    projected = points_cam @ K.T
    return projected[..., :2] / projected[..., 2:3].clamp_min(1e-6)


def _bounded_tanh(raw: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Maps an unconstrained `raw` tensor to `(-bounds, bounds)`, elementwise.
    The forward half of Stage 2's hard-bound reparameterization, see
    JAW_AXIS0_BOUND_RAD's own comment for why this exists instead of a soft
    penalty. Gradients stay usable (never exactly zero) even far from the
    origin, confirmed directly: raw=5 (far beyond any bound reached in
    practice) still has gradient ~1.5e-5, not 0."""
    return bounds * torch.tanh(raw / bounds)


def _inverse_bounded_tanh(x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Inverts `_bounded_tanh`, used once, to initialize Stage 2's raw
    parameter so the bounded value starts exactly at `x` (DECA's own
    estimate), not at 0. Clamped just inside +-1 before `atanh` since values
    at exactly the bound would otherwise map to +-inf."""
    return bounds * torch.atanh(torch.clamp(x / bounds, -0.999, 0.999))


def _init_shared_betas(mica_shape: np.ndarray, mica_valid: np.ndarray, deca_shape: np.ndarray, deca_valid: np.ndarray) -> np.ndarray:
    """One (300,) identity for the whole clip. MICA's own full 300-dim shape
    estimate is preferred (it specializes in identity; DECA's own 100-dim
    output is truncated to fewer of the same PCA basis' components, both
    regress coefficients of FLAME's shared `shapedirs`, so they're directly
    poolable). Falls back to DECA's 100 (zero-padded) if no frame has a valid
    MICA estimate."""
    if mica_valid.any():
        return mica_shape[mica_valid].mean(axis=0)
    betas = np.zeros(FLAME_NUM_BETAS, dtype=np.float32)
    if deca_valid.any():
        betas[:100] = deca_shape[deca_valid].mean(axis=0)
    return betas


def _init_translation(landmarks_51: np.ndarray, valid: np.ndarray, template_landmarks: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-frame (F, 3) initial head translation via a similar-triangles depth
    estimate from interocular distance, then unprojecting the landmark
    centroid at that depth. Invalid frames get the nearest valid frame's
    translation (better than a fixed guess, and gets overwritten by the
    temporal-smoothness term's pull during optimization anyway)."""
    template_interocular = float(np.linalg.norm(template_landmarks[_LEYE_51] - template_landmarks[_REYE_51]))
    fx = float(K[0, 0])
    n = len(landmarks_51)
    transl = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        if not valid[i]:
            continue
        observed_interocular = float(np.linalg.norm(landmarks_51[i, _LEYE_51] - landmarks_51[i, _REYE_51]))
        depth = fx * template_interocular / max(observed_interocular, 1e-3)
        centroid_px = landmarks_51[i].mean(axis=0)
        xy = np.array([(centroid_px[0] - K[0, 2]) * depth / fx, (centroid_px[1] - K[1, 2]) * depth / K[1, 1]])
        transl[i] = [xy[0], xy[1], depth]

    if valid.any():
        valid_idx = np.where(valid)[0]
        for i in np.where(~valid)[0]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
            transl[i] = transl[nearest]
    return transl


def calibrate_rotation_offset(deca_global_orient: np.ndarray, candidate_rotmat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """A fixed (3, 3) rotation reconciling some other per-frame rotation
    signal's own convention with FLAME's `global_orient` convention, so that
    signal can be used as a prior/anchor. Two callers today: the body-based
    orientation prior's `head_rotmat` (GVHMR's tracked Head-joint rotation)
    and `face_pose_stabilization.stabilize_orientation`'s Kabsch rotation.

    Calibrated per clip from data: for each trustworthy frame,
    `deca_global_orient @ candidate_rotmat.T` is a candidate offset (DECA's
    own per-frame estimate stands in for "the true head orientation" here),
    averaged via quaternion mean (sign-resolved against the first candidate,
    then renormalized) rather than a naive matrix mean, which isn't a
    rotation. Not circular: the win isn't that the other signal is more
    accurate than DECA, but that it stays reliable exactly where DECA
    isn't, calibrating the offset against DECA's average behavior, then
    applying the other signal through it, gets the best of both."""
    from scipy.spatial.transform import Rotation

    if not valid.any():
        return np.eye(3, dtype=np.float32)

    r_deca = Rotation.from_rotvec(deca_global_orient[valid]).as_matrix()
    r_candidate = candidate_rotmat[valid]
    r_offset_candidates = r_deca @ np.transpose(r_candidate, (0, 2, 1))

    quats = Rotation.from_matrix(r_offset_candidates).as_quat()
    signs = np.sign((quats * quats[0]).sum(axis=1))
    signs[signs == 0] = 1.0
    mean_quat = (quats * signs[:, None]).mean(axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    return Rotation.from_quat(mean_quat).as_matrix().astype(np.float32)


def _calibrate_kabsch(inputs: FitInputs, valid: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Calibrates `face_pose_stabilization`'s Kabsch rotation into FLAME's
    `global_orient` convention, for the Stage 1 soft anchor loss. Returns
    (calibrated_rotmat (n, 3, 3) or None, confidence mask (n,) or None),
    None/None when the signal isn't available or there aren't enough
    trustworthy frames to calibrate from (see MIN_KABSCH_CALIBRATION_FRAMES)."""
    if inputs.kabsch_rotmat is None or inputs.kabsch_confidence is None:
        return None, None
    kabsch_finite = ~np.isnan(inputs.kabsch_rotmat).any(axis=(1, 2))
    kabsch_confidence_effective = inputs.kabsch_confidence & kabsch_finite
    calib_valid_kabsch = valid & kabsch_confidence_effective
    if int(calib_valid_kabsch.sum()) < MIN_KABSCH_CALIBRATION_FRAMES:
        return None, None
    # NaN rows (undetected frames) are replaced with identity before
    # calibration, excluded by the confidence mask everywhere this is
    # used, but 0 * NaN is still NaN, so a literal NaN would poison a masked
    # sum without this.
    kabsch_filled = np.where(kabsch_finite[:, None, None], inputs.kabsch_rotmat, np.eye(3, dtype=np.float32))
    r_offset_kabsch = calibrate_rotation_offset(inputs.deca_pose[:, :3], kabsch_filled, calib_valid_kabsch)
    calibrated_kabsch = r_offset_kabsch[None] @ kabsch_filled
    return calibrated_kabsch.astype(np.float32), kabsch_confidence_effective


def _demote_short_valid_runs(valid: np.ndarray, min_valid_run_frames: int) -> np.ndarray:
    """A valid run shorter than `min_valid_run_frames` is too brief a
    detection to trust as an independent anchor point, demoted to
    invalid so it's folded into its surrounding gap (handled by the
    existing long-gap hold or short-gap bridge like any other occlusion)
    instead of the fit treating a single isolated frame's own reprojection
    result as ground truth. See MIN_VALID_RUN_FRAMES's own comment.

    Only *interior* runs (bounded by invalid frames on both sides) are ever
    demoted, same convention `_bridge_keep_mask` already uses for the
    opposite case (invalid runs), and for the same reason: a leading/
    trailing run (or, degenerately, a run spanning the *entire* clip, real
    on a short synthetic test clip) has no invalid neighbor on one side to
    be suspicious of it *relative to*, so there's nothing here to demote it
    in favor of."""
    demoted = valid.copy()
    for start, end in contiguous_true_runs(valid):
        if start == 0 or end == len(valid) - 1:
            continue
        if (end - start + 1) < min_valid_run_frames:
            demoted[start:end + 1] = False
    return demoted


def _lead_from_neutral(values: np.ndarray, valid: np.ndarray, max_bridge_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """The pipeline's other freeze mechanisms hold at a real neighboring
    value (an entry pose for an interior gap, the last known pose for a
    trailing one), but a leading gap has no real prior value to hold, so
    `fill_invalid`'s own boundary behavior would instead freeze it at the
    first-detected frame's value for the whole gap, which can hold on an
    arbitrary mid-articulation pose well before any real detection.

    Instead, holds at neutral (all-zero, FLAME's own rest pose/expression)
    and glides into the real first-detected pose only over the final
    `max_bridge_frames`, the same hold-then-blend contract
    `cap_long_gaps_with_hold` uses for interior gaps, just sourced from
    neutral since no real entry value exists here.

    A no-op if the clip doesn't start with an invalid run. Returns (values,
    valid) copies; the returned `valid` marks the whole leading run
    trustworthy so downstream steps leave it alone, callers must keep
    using the *original* `valid` for those steps regardless, so this
    function's own neutral-hold/glide values aren't second-guessed."""
    values = values.copy()
    valid = np.asarray(valid, dtype=bool).copy()
    runs = contiguous_true_runs(~valid)
    if not runs or runs[0][0] != 0:
        return values, valid

    _, end = runs[0]  # start is 0 by construction
    run_length = end + 1
    glide_frames = min(run_length, max_bridge_frames)
    glide_start = run_length - glide_frames

    values[:glide_start] = 0.0
    recovery = values[end + 1] if end + 1 < len(values) else np.zeros_like(values[0])
    for i in range(glide_frames):
        frac = (i + 1) / (glide_frames + 1)  # strictly in (0, 1): lands just short of the real recovery value
        values[glide_start + i] = frac * recovery
    valid[:end + 1] = True
    return values, valid


def _bridge_keep_mask(valid: np.ndarray, max_bridge_frames: int, trustworthy_anchor: np.ndarray | None = None) -> np.ndarray:
    """True wherever the fit's own output should be trusted as-is: every
    originally-valid frame, every frame `trustworthy_anchor` marks as having
    a real independent signal despite invalid landmarks (Kabsch/the
    body-based orientation prior, when confidently available), plus any
    interior invalid run longer than `max_bridge_frames` (real occlusion,
    left to the existing anchor-driven fit) and any leading/trailing run (no
    second real endpoint to interpolate toward). False only for interior
    invalid runs, with no trustworthy anchor covering them, at or under
    `max_bridge_frames`, see `MAX_BRIDGE_FRAMES`.

    `trustworthy_anchor` exists because gap length alone isn't enough to
    decide "prefer interpolation": a short dropout covered by a confident
    Kabsch/body-based-orientation-prior signal should recover the real
    rotation change through it, not interpolate it away. DECA is
    deliberately not included: it has no confidence signal distinguishing a
    good per-frame estimate from a bad one, so it can't be trusted as a
    reason to skip bridging."""
    keep = valid.copy()
    if trustworthy_anchor is not None:
        keep = keep | trustworthy_anchor
    for start, end in contiguous_true_runs(~keep):
        if start == 0 or end == len(keep) - 1 or (end - start + 1) > max_bridge_frames:
            keep[start:end + 1] = True
    return keep


def _bridge_short_gaps(
    rotvec: np.ndarray | None, linear: dict[str, np.ndarray], valid: np.ndarray, max_bridge_frames: int,
    trustworthy_anchor: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Replaces the fit's own output on short (`<= max_bridge_frames`)
    interior detection dropouts with an interpolation between the flanking
    valid frames' own fitted values, instead of whatever the anchor-driven
    optimization produced there, same short-gap-interpolate/long-gap-leave
    -alone contract as `gvhmr_postprocess.pp_bridge_low_confidence_root_
    motion` and `motion_smoothing.cap_long_gaps_with_hold`. `rotvec` (F, 3),
    if given, is bridged via quaternion slerp (avoids axis-angle wraparound),
    pass None to skip it (jaw_pose is a small per-axis articulation, not a
    composed rotation, so it belongs in `linear` instead). Each array in
    `linear` (F, C) is bridged via plain per-channel linear interpolation,
    safe for translation/jaw/expression's small, non-wrapping magnitudes.
    `trustworthy_anchor` is forwarded to `_bridge_keep_mask`. A no-op
    wherever `_bridge_keep_mask` is already True."""
    keep = _bridge_keep_mask(valid, max_bridge_frames, trustworthy_anchor)
    if keep.all():
        return rotvec, linear

    bridged_rotvec = rotvec
    if rotvec is not None:
        from scipy.spatial.transform import Rotation

        quats = hemisphere_aligned_quats(rotvec, keep)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        bridged_rotvec = Rotation.from_quat(quats).as_rotvec().astype(rotvec.dtype)

    bridged_linear = {key: fill_invalid(values, keep).astype(values.dtype) for key, values in linear.items()}
    return bridged_rotvec, bridged_linear


def _detect_outlier_frames(
    values: np.ndarray, valid: np.ndarray, threshold: float, is_rotation: bool = False, window: int = 1,
) -> np.ndarray:
    """Flags originally-valid interior frames whose own fitted value
    deviates sharply from a reference built from their own nearby neighbors
   , a real per-frame optimization outlier, as opposed to a landmark-
    dropout artifact (already handled by `_bridge_keep_mask`'s validity-
    driven pass, which should run first). Only originally-valid frames are
    ever flagged, an already-bridged frame is itself an interpolation
    result with no independent "own value" to judge.

    `window` (non-rotation path only): at the default 1, the reference is
    the linear midpoint of the immediate +-1 neighbors. When an outlier
    spans more than one consecutive frame, an immediate neighbor can itself
    already be corrupted, pulling that 2-point midpoint toward the outlier
    instead of away from it. `window > 1` widens the reference to the
    median of `window` frames on each side (excluding the frame itself),
    tolerating up to `window - 1` corrupted neighbors on one side without
    being pulled toward them."""
    n = len(values)
    outlier = np.zeros(n, dtype=bool)
    if n < 3:
        return outlier

    if is_rotation:
        from scipy.spatial.transform import Rotation, Slerp

        r = Rotation.from_rotvec(values)
        for i in range(1, n - 1):
            if not valid[i]:
                continue
            r_mid = Slerp([0, 1], Rotation.concatenate([r[i - 1], r[i + 1]]))([0.5])[0]
            if (r_mid.inv() * r[i]).magnitude() > threshold:
                outlier[i] = True
    elif window == 1:
        midpoint = 0.5 * (values[:-2] + values[2:])
        deviation = np.linalg.norm(values[1:-1] - midpoint, axis=-1)
        outlier[1:-1] = valid[1:-1] & (deviation > threshold)
    else:
        for i in range(window, n - window):
            if not valid[i]:
                continue
            neighborhood = np.concatenate([values[i - window:i], values[i + 1:i + 1 + window]])
            reference = np.median(neighborhood, axis=0)
            if np.linalg.norm(values[i] - reference) > threshold:
                outlier[i] = True

    return outlier


def _snap_jaw_to_deca_on_axis_deviation(
    jaw_pose: np.ndarray, deca_jaw: np.ndarray, valid: np.ndarray, axis: int, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replaces the fitted `jaw_pose` (all 3 axes) with DECA's own per-frame
    estimate on any valid frame where `axis` alone deviates from DECA by more
    than `threshold`, see JAW_AXIS1_DECA_DEVIATION_THRESHOLD's own comment
    for why this exists and why axis1 specifically. Returns (values,
    flagged), `flagged` for callers that want to know which frames were
    touched."""
    deviation = np.abs(jaw_pose[:, axis] - deca_jaw[:, axis])
    flagged = valid & (deviation > threshold)
    snapped = jaw_pose.copy()
    snapped[flagged] = deca_jaw[flagged]
    return snapped, flagged


def fit_clip(
    inputs: FitInputs,
    fps: float,
    device: torch.device | None = None,
    stage1_iters: int = DEFAULT_STAGE1_ITERS,
    stage1_lr: float = DEFAULT_STAGE1_LR,
    stage2_iters: int = DEFAULT_STAGE2_ITERS,
    stage2_lr: float = DEFAULT_STAGE2_LR,
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
    expression_weight: float = DEFAULT_EXPRESSION_WEIGHT,
    head_prior_weight: float = DEFAULT_HEAD_PRIOR_WEIGHT,
    global_orient_anchor_weight: float = DEFAULT_GLOBAL_ORIENT_ANCHOR_WEIGHT,
    kabsch_anchor_weight: float = DEFAULT_KABSCH_ANCHOR_WEIGHT,
    jaw_deca_anchor_weight: float = DEFAULT_JAW_DECA_ANCHOR_WEIGHT,
    expr_deca_anchor_weight: float = DEFAULT_EXPR_DECA_ANCHOR_WEIGHT,
) -> dict[str, np.ndarray]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(inputs.landmarks_51)
    valid = inputs.landmarks_valid & inputs.deca_valid
    valid = _demote_short_valid_runs(valid, MIN_VALID_RUN_FRAMES)
    if not valid.any():
        zeros3 = np.zeros((n, 3), dtype=np.float32)
        return {
            KEY_BETAS: np.zeros(FLAME_NUM_BETAS, dtype=np.float32),
            KEY_EXPRESSION: np.zeros((n, FLAME_NUM_EXPRESSION), dtype=np.float32),
            KEY_GLOBAL_ORIENT: zeros3, KEY_JAW_POSE: zeros3, KEY_TRANSL: zeros3,
            KEY_VALID: valid,
        }

    model = _build_flame_model(device, batch_size=n)
    with torch.no_grad():
        template = model(
            betas=torch.zeros(1, FLAME_NUM_BETAS, device=device),
            expression=torch.zeros(1, FLAME_NUM_EXPRESSION, device=device),
            global_orient=torch.zeros(1, 3, device=device), neck_pose=torch.zeros(1, 3, device=device),
            jaw_pose=torch.zeros(1, 3, device=device),
            leye_pose=torch.zeros(1, 3, device=device), reye_pose=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
        )
        template_landmarks = template.joints[0, -NUM_FLAME_LANDMARKS:].cpu().numpy()

    betas_init = _init_shared_betas(inputs.mica_shape, inputs.mica_valid, inputs.deca_shape, inputs.deca_valid)
    transl_init = _init_translation(inputs.landmarks_51, valid, template_landmarks, inputs.intrinsics_k)

    # Body-based orientation prior: on frames where GVHMR's own head track is
    # confident, its (offset-corrected) rotation replaces DECA's own
    # global_orient init for Stage 1 and pulls the fit toward it during
    # optimization, DECA's per-frame estimate is the weaker signal exactly
    # where this matters (occlusion, extreme turns), while GVHMR's temporal
    # model stays stable through both. See calibrate_rotation_offset's own
    # docstring.
    global_orient_init = inputs.deca_pose[:, :3].copy()
    prior_rotmat = None
    prior_confidence = None
    if inputs.head_rotmat is not None and inputs.head_confidence is not None:
        from scipy.spatial.transform import Rotation

        calib_valid = valid & inputs.head_confidence
        if int(calib_valid.sum()) >= MIN_HEAD_CALIBRATION_FRAMES:
            r_offset = calibrate_rotation_offset(inputs.deca_pose[:, :3], inputs.head_rotmat, calib_valid)
            candidate_prior_rotmat = r_offset[None] @ inputs.head_rotmat

            # Ill-conditioning gate, see MAX_SAFE_PRIOR_ANGLE_DEG's own
            # comment. Checked on every confidence-eligible frame, not just
            # the calibration subset: the danger is in how close the composed
            # rotation gets to 180 deg wherever the prior would actually be
            # applied, not in the calibration itself (which measured stable
            # here even on the clip that triggered this gate).
            angle_deg = np.degrees(Rotation.from_matrix(candidate_prior_rotmat[inputs.head_confidence]).magnitude())
            if angle_deg.max() <= MAX_SAFE_PRIOR_ANGLE_DEG:
                prior_rotmat = candidate_prior_rotmat
                prior_confidence = inputs.head_confidence
                # A single, independent per-frame conversion is fine for an
                # init (each frame just needs *a* valid representation to
                # start Adam from), unlike the loss below, this never
                # differences two frames' vectors against each other, so it
                # isn't exposed to the axis-angle wraparound issue that
                # motivates the geodesic-distance formulation there.
                init_rotvec = Rotation.from_matrix(prior_rotmat).as_rotvec().astype(np.float32)
                global_orient_init[prior_confidence] = init_rotvec[prior_confidence]

    # betas is fixed at MICA's estimate, not optimized: a 2D landmark reprojection
    # is a weak, ambiguous signal for identity/shape (a bigger face further away
    # projects to the same 2D points as a smaller face closer up, the classic
    # monocular scale/depth ambiguity), but a strong one for pose/expression given
    # a *fixed* shape. Letting betas drift measurably changed head depth in
    # testing without improving the fit; MICA is the dedicated identity signal
    # (see this module's docstring), so it should win outright, not get diluted.
    betas = torch.tensor(betas_init, device=device, dtype=torch.float32)
    neck_pose = torch.zeros(n, 3, device=device)
    eye_pose = torch.zeros(n, 3, device=device)  # not optimized, see module docstring
    K = torch.tensor(inputs.intrinsics_k, device=device, dtype=torch.float32)
    landmarks_target = torch.tensor(inputs.landmarks_51, device=device, dtype=torch.float32)
    valid_mask = torch.tensor(valid, device=device, dtype=torch.float32).unsqueeze(-1)
    betas_batch = betas.unsqueeze(0).expand(n, -1)

    # Stage 1: global_orient + transl only. jaw_pose/expression are FIXED
    # at the DECA init (no_grad), completely out of this optimization, so
    # orientation gets full, uncompeted reprojection-driven refinement. A
    # clean, easy clip's own healthy convergence was never about orientation
    # being "easy", it was about jaw/expression never getting the chance to
    # compete for the same gradient in the first place.
    jaw_fixed = torch.tensor(inputs.deca_pose[:, 3:6].copy(), device=device, dtype=torch.float32)
    expression_fixed = torch.tensor(inputs.deca_exp, device=device, dtype=torch.float32)

    global_orient = torch.tensor(global_orient_init, device=device, dtype=torch.float32, requires_grad=True)
    transl = torch.tensor(transl_init, device=device, dtype=torch.float32, requires_grad=True)

    deca_global_orient_mat = axis_angle_to_matrix(torch.tensor(inputs.deca_pose[:, :3], device=device, dtype=torch.float32))
    deca_valid_flat = torch.tensor(inputs.deca_valid, device=device, dtype=torch.float32)
    identity3 = torch.eye(3, device=device, dtype=torch.float32)

    calibrated_kabsch_rotmat, kabsch_confidence_effective = _calibrate_kabsch(inputs, valid)
    kabsch_rotmat_tensor = kabsch_valid_flat = None
    if calibrated_kabsch_rotmat is not None:
        kabsch_rotmat_tensor = torch.tensor(calibrated_kabsch_rotmat, device=device, dtype=torch.float32)
        kabsch_valid_flat = torch.tensor(kabsch_confidence_effective, device=device, dtype=torch.float32)

    prior_rotmat_tensor = prior_mask = None
    if prior_rotmat is not None:
        prior_rotmat_tensor = torch.tensor(prior_rotmat, device=device, dtype=torch.float32)
        prior_mask = torch.tensor(prior_confidence, device=device, dtype=torch.float32)

    optimizer1 = torch.optim.Adam([global_orient, transl], lr=stage1_lr)
    for it in frame_progress(
        range(stage1_iters), total=stage1_iters, label=_FIT_ORIENTATION_LABEL, unit="iteration",
    ):
        if it == STAGE1_LR_DECAY_ITER:
            for group in optimizer1.param_groups:
                group["lr"] = stage1_lr * STAGE1_LR_DECAY_FACTOR
        optimizer1.zero_grad()
        out = model(
            betas=betas_batch, expression=expression_fixed, global_orient=global_orient, neck_pose=neck_pose,
            jaw_pose=jaw_fixed, leye_pose=eye_pose, reye_pose=eye_pose, transl=transl,
        )
        landmarks_pred = out.joints[:, -NUM_FLAME_LANDMARKS:, :]
        landmarks_pixels = _project_points(landmarks_pred, K)

        reprojection_error = (landmarks_pixels - landmarks_target).pow(2).sum(-1) * valid_mask
        loss = reprojection_error.sum() / valid_mask.sum().clamp_min(1.0)

        # Chordal distance (||R_rel - I||_F^2 = 8*sin^2(angle/2)), not a raw
        # axis-angle vector difference or its extracted angle: global_orient
        # can sit near the axis-angle singularity on real clips (GVHMR's own
        # rest-pose convention isn't camera-relative), where a raw difference
        # explodes by ~2*pi despite barely-changed physical rotation, and
        # angle-extraction has an undefined gradient exactly at the identity
        # rotation, both real failure modes hit and fixed during this
        # anchor's own development, not hypothetical.
        global_orient_mat = axis_angle_to_matrix(global_orient)
        global_orient_relative_deca = global_orient_mat.transpose(-1, -2) @ deca_global_orient_mat
        global_orient_anchor_error = (global_orient_relative_deca - identity3).pow(2).sum(dim=(-2, -1)) * deca_valid_flat
        loss = loss + global_orient_anchor_weight * global_orient_anchor_error.sum() / deca_valid_flat.sum().clamp_min(1.0)

        if kabsch_rotmat_tensor is not None:
            global_orient_relative_kabsch = global_orient_mat.transpose(-1, -2) @ kabsch_rotmat_tensor
            kabsch_anchor_error = (global_orient_relative_kabsch - identity3).pow(2).sum(dim=(-2, -1)) * kabsch_valid_flat
            loss = loss + kabsch_anchor_weight * kabsch_anchor_error.sum() / kabsch_valid_flat.sum().clamp_min(1.0)

        if prior_rotmat_tensor is not None:
            relative = global_orient_mat.transpose(-1, -2) @ prior_rotmat_tensor
            head_prior_error = (relative - identity3).pow(2).sum(dim=(-2, -1)) * prior_mask
            loss = loss + head_prior_weight * head_prior_error.sum() / prior_mask.sum().clamp_min(1.0)

        if n > 1:  # frame-to-frame deltas are undefined for a single-frame clip
            temporal_loss = (
                F.huber_loss(global_orient[1:], global_orient[:-1], delta=STAGE1_ORIENT_TEMPORAL_HUBER_DELTA)
                + F.huber_loss(transl[1:], transl[:-1], delta=STAGE1_TRANSL_TEMPORAL_HUBER_DELTA)
            )
            loss = loss + temporal_weight * temporal_loss
        loss.backward()
        optimizer1.step()

    global_orient_final = global_orient.detach()
    transl_final = transl.detach()

    # Bridge short (<= MAX_BRIDGE_FRAMES) detection dropouts before Stage 2
    # ever sees this orientation, see MAX_BRIDGE_FRAMES/_bridge_short_gaps'
    # own comments. Feeding Stage 2 the bridged orientation (rather than
    # bridging only at the very end) matters: Stage 2's own reprojection
    # term for a bridged frame should react to the clean, interpolated pose,
    # not the anchor-driven one it's about to be thrown away in favor of.
    # A short gap covered by a confident Kabsch/body-based-orientation-prior
    # signal is left to that anchor instead of being interpolated away, see
    # `_bridge_keep_mask`'s own docstring for why (a real regression caught
    # against exactly this: `test_fit_clip_head_prior_recovers_
    # occluded_rotation_spike`/the Kabsch equivalent both failed before this
    # was added, since unconditional bridging discarded the very recovered
    # rotation spike those tests exist to check for).
    trustworthy_anchor = np.zeros(n, dtype=bool)
    if kabsch_valid_flat is not None:
        trustworthy_anchor |= kabsch_confidence_effective
    if prior_mask is not None:
        trustworthy_anchor |= prior_confidence

    bridged_go_np, bridged_linear = _bridge_short_gaps(
        global_orient_final.cpu().numpy(), {"transl": transl_final.cpu().numpy()}, valid, MAX_BRIDGE_FRAMES,
        trustworthy_anchor=trustworthy_anchor,
    )

    # Second pass: a real per-frame optimization outlier on an otherwise-
    # valid frame (see DEFAULT_TEMPORAL_WEIGHT's own comment), bridged the
    # same way, driven by global_orient's own outlier flag for
    # both quantities (Stage 1 fits them jointly, so a bad orientation frame
    # is the relevant signal for transl too, same as the gap-bridge above).
    go_outliers = _detect_outlier_frames(bridged_go_np, valid, GO_OUTLIER_DEVIATION_THRESHOLD, is_rotation=True)
    if go_outliers.any():
        bridged_go_np, bridged_linear = _bridge_short_gaps(
            bridged_go_np, bridged_linear, ~go_outliers, MAX_OUTLIER_BRIDGE_FRAMES,
        )

    global_orient_final = torch.tensor(bridged_go_np, device=device, dtype=torch.float32)
    transl_final = torch.tensor(bridged_linear["transl"], device=device, dtype=torch.float32)

    # Stage 2: jaw_pose + expression only. global_orient/transl are FIXED
    # at Stage 1's converged result, so nothing in this stage can trade
    # jaw/expression off against orientation, structurally, not by
    # out-weighting a competing gradient the way every soft anchor tried
    # before this redesign had to. Both parameters are bounded-tanh
    # reparameterizations (see JAW_AXIS0_BOUND_RAD/EXPRESSION_BOUND's own
    # comments), initialized so they start exactly at DECA's own estimate.
    jaw_bounds = torch.tensor([JAW_AXIS0_BOUND_RAD, JAW_AXIS1_BOUND_RAD, JAW_AXIS2_BOUND_RAD], device=device)
    expr_bounds = torch.full((FLAME_NUM_EXPRESSION,), EXPRESSION_BOUND, device=device)

    jaw_raw = _inverse_bounded_tanh(jaw_fixed, jaw_bounds).clone().requires_grad_(True)
    expression_clamped_init = expression_fixed.clamp(-expr_bounds * 0.99, expr_bounds * 0.99)
    expr_raw = _inverse_bounded_tanh(expression_clamped_init, expr_bounds).clone().requires_grad_(True)

    # Stage 2's temporal Huber loss, unlike Stage 1's, has nothing else
    # anchoring an invalid frame to anything real (no jaw/expression anchor
    # exists any more, see MAX_OUTLIER_BRIDGE_FRAMES's own comment above)
    #, so masking it to literal array-adjacent pairs
    # regardless of validity let a chain of unconstrained frames span an
    # entire long occlusion and pull even the sparse *valid* frames at
    # either end off their own correct reprojection-driven fit, confirmed
    # directly on a real 111-frame invalid stretch (idx574-684) containing
    # only two real detections: both of those detections' own fitted values
    # were themselves contaminated by the drift, not just the frames
    # between them, no amount of *post-hoc* bridging can fix that, since
    # it's already baked into what Stage 2 optimized. Gating the temporal
    # term to only connect pairs where *both* frames are valid fixes this
    # at the source: an invalid frame gets no temporal pull at all (falls
    # back cleanly to its own DECA-based init, never dragged anywhere), and
    # two valid frames are only ever compared to each other, never to the
    # unconstrained noise in between.
    both_valid_pairs = torch.tensor(valid[1:] & valid[:-1], device=device, dtype=torch.float32) if n > 1 else None

    optimizer2 = torch.optim.Adam([jaw_raw, expr_raw], lr=stage2_lr)
    for it in frame_progress(
        range(stage2_iters), total=stage2_iters, label=_FIT_EXPRESSION_LABEL, unit="iteration",
    ):
        if it == STAGE2_LR_DECAY_ITER:
            for group in optimizer2.param_groups:
                group["lr"] = stage2_lr * STAGE2_LR_DECAY_FACTOR
        optimizer2.zero_grad()
        jaw_pose = _bounded_tanh(jaw_raw, jaw_bounds)
        expression = _bounded_tanh(expr_raw, expr_bounds)
        out = model(
            betas=betas_batch, expression=expression, global_orient=global_orient_final, neck_pose=neck_pose,
            jaw_pose=jaw_pose, leye_pose=eye_pose, reye_pose=eye_pose, transl=transl_final,
        )
        landmarks_pred = out.joints[:, -NUM_FLAME_LANDMARKS:, :]
        landmarks_pixels = _project_points(landmarks_pred, K)

        reprojection_error = (landmarks_pixels - landmarks_target).pow(2).sum(-1) * valid_mask
        loss = reprojection_error.sum() / valid_mask.sum().clamp_min(1.0)
        loss = loss + expression_weight * expression.pow(2).sum(-1).mean()

        # DECA-anchor ridge (jaw and expression both), see
        # DEFAULT_JAW_DECA_ANCHOR_WEIGHT's own comment for the real-data
        # evidence that motivated this. Masked to valid frames only: an
        # invalid frame already starts and stays exactly at its own DECA
        # init (nothing else touches it, per the temporal-loss masking above),
        # so this term is a genuine *during-real-optimization* regularizer,
        # never a substitute for missing data the way the reverted
        # long-occlusion DECA anchor above was.
        valid_mask_flat = valid_mask.squeeze(-1)
        jaw_deca_anchor_error = (jaw_pose - jaw_fixed).pow(2).sum(-1) * valid_mask_flat
        loss = loss + jaw_deca_anchor_weight * jaw_deca_anchor_error.sum() / valid_mask_flat.sum().clamp_min(1.0)
        expr_deca_anchor_error = (expression - expression_fixed).pow(2).sum(-1) * valid_mask_flat
        loss = loss + expr_deca_anchor_weight * expr_deca_anchor_error.sum() / valid_mask_flat.sum().clamp_min(1.0)

        if n > 1:
            expr_delta = F.huber_loss(
                expression[1:], expression[:-1], delta=STAGE2_EXPR_TEMPORAL_HUBER_DELTA, reduction="none",
            ).mean(dim=-1)
            jaw_delta = F.huber_loss(
                jaw_pose[1:], jaw_pose[:-1], delta=STAGE2_JAW_TEMPORAL_HUBER_DELTA, reduction="none",
            ).mean(dim=-1)
            temporal_loss = ((expr_delta + jaw_delta) * both_valid_pairs).sum() / both_valid_pairs.sum().clamp_min(1.0)
            loss = loss + temporal_weight * temporal_loss
        loss.backward()
        optimizer2.step()

    jaw_pose_final = _bounded_tanh(jaw_raw, jaw_bounds).detach()
    expression_final = _bounded_tanh(expr_raw, expr_bounds).detach()

    # Sustained-drift pass: catches what the neighbor-midpoint outlier check
    # below structurally can't, see JAW_AXIS1_DECA_DEVIATION_THRESHOLD's
    # own comment. Runs here, directly on Stage 2's own raw per-frame output
    # and before any gap-handling below, so a snapped frame is
    # indistinguishable from a real Stage 2 result to every downstream
    # mechanism (leading-glide, hold, short-gap interpolation, the
    # neighbor-outlier check), running it later, after those had already
    # computed their own hold/interpolation values from the *pre-snap*
    # flanking frame, left a real seam where a short gap's own bridge still
    # aimed at the old (wrong) endpoint one frame after that endpoint had
    # already been corrected.
    jaw_pose_snapped_np, _ = _snap_jaw_to_deca_on_axis_deviation(
        jaw_pose_final.cpu().numpy(), inputs.deca_pose[:, 3:6], valid, axis=1, threshold=JAW_AXIS1_DECA_DEVIATION_THRESHOLD,
    )

    # A leading or trailing invalid run needs the same freeze-at-a-real-value
    # treatment as an interior gap, but nothing upstream does that for
    # jaw_pose/expression automatically: with zero reprojection signal and
    # the temporal loss masked to both-valid pairs (see above), an invalid
    # frame gets zero gradient all optimization long and never moves off its
    # own noisy per-frame DECA init. Fixed by freezing at the nearest real
    # neighbor via `fill_invalid`, except the leading run, which
    # `_lead_from_neutral` handles first since it has no real prior value to
    # freeze at (see its own docstring).
    jaw_lead_np, jaw_lead_valid = _lead_from_neutral(jaw_pose_snapped_np, valid, MAX_BRIDGE_FRAMES)
    expr_lead_np, expr_lead_valid = _lead_from_neutral(expression_final.cpu().numpy(), valid, MAX_BRIDGE_FRAMES)
    jaw_pose_final_np = fill_invalid(jaw_lead_np, jaw_lead_valid)
    expression_final_np = fill_invalid(expr_lead_np, expr_lead_valid)

    # Long (> MAX_BRIDGE_FRAMES) *interior* invalid runs: Stage 2's own raw
    # output has nothing pulling it back toward anything for the whole
    # stretch (see MAX_BRIDGE_FRAMES's own comment above), so freeze at the
    # entry value instead of trusting it, same
    # `cap_long_gaps_with_hold` + `fill_invalid` composition already
    # validated for hands: hold the bulk of the gap, blend into the real
    # recovery only over the final MAX_BRIDGE_FRAMES, which is exactly what
    # the short-gap bridge below already does once `held_valid` marks the
    # held portion trustworthy. Leading/trailing runs are already frozen by
    # the `fill_invalid` pass just above; this only ever touches interior ones.
    jaw_held_values, jaw_held_valid = cap_long_gaps_with_hold(jaw_pose_final_np, valid, MAX_BRIDGE_FRAMES)
    expr_held_values, expr_held_valid = cap_long_gaps_with_hold(expression_final_np, valid, MAX_BRIDGE_FRAMES)

    _, jaw_bridged_init = _bridge_short_gaps(None, {"jaw_pose": jaw_held_values}, jaw_held_valid, MAX_BRIDGE_FRAMES)
    _, expr_bridged_init = _bridge_short_gaps(None, {"expression": expr_held_values}, expr_held_valid, MAX_BRIDGE_FRAMES)
    bridged_final = {"jaw_pose": jaw_bridged_init["jaw_pose"], "expression": expr_bridged_init["expression"]}

    # Second pass: real per-frame optimization outliers on otherwise-valid
    # frames, detected independently for jaw and expression (unlike Stage 1's
    # joint go/transl fit, Stage 2 has no single shared signal to drive both
    # from), see JAW_OUTLIER_DEVIATION_THRESHOLD's own comment.
    jaw_outliers = _detect_outlier_frames(
        bridged_final["jaw_pose"], valid, JAW_OUTLIER_DEVIATION_THRESHOLD, window=JAW_OUTLIER_WINDOW,
    )
    expr_outliers = _detect_outlier_frames(bridged_final["expression"], valid, EXPR_OUTLIER_DEVIATION_THRESHOLD)
    if jaw_outliers.any():
        _, jaw_bridged = _bridge_short_gaps(
            None, {"jaw_pose": bridged_final["jaw_pose"]}, ~jaw_outliers, JAW_OUTLIER_BRIDGE_FRAMES,
        )
        bridged_final["jaw_pose"] = jaw_bridged["jaw_pose"]
    if expr_outliers.any():
        _, expr_bridged = _bridge_short_gaps(
            None, {"expression": bridged_final["expression"]}, ~expr_outliers, MAX_OUTLIER_BRIDGE_FRAMES,
        )
        bridged_final["expression"] = expr_bridged["expression"]

    # Adaptive ambient-jitter smoothing, see FACE_ONE_EURO_JAW_MIN_CUTOFF_HZ's
    # own comment. Runs last, over the whole already-continuous
    # series (no `valid` mask needed: every gap has already been bridged,
    # held, or demoted above, so there's no discontinuity left for the filter
    # to trip over).
    jaw_pose_final_np = one_euro_filter_rotation_sequence(
        bridged_final["jaw_pose"], fps, FACE_ONE_EURO_JAW_MIN_CUTOFF_HZ, FACE_ONE_EURO_JAW_BETA,
    )
    expression_final_np = one_euro_filter_sequence(
        bridged_final["expression"], fps, FACE_ONE_EURO_EXPR_MIN_CUTOFF_HZ, FACE_ONE_EURO_EXPR_BETA,
    )

    return {
        KEY_BETAS: betas.detach().cpu().numpy(),
        KEY_EXPRESSION: expression_final_np,
        KEY_GLOBAL_ORIENT: global_orient_final.cpu().numpy(),
        KEY_JAW_POSE: jaw_pose_final_np,
        KEY_TRANSL: transl_final.cpu().numpy(),
        KEY_VALID: valid,
    }


def local_landmark_delta(
    jaw_pose: np.ndarray, expression: np.ndarray, device: torch.device | None = None,
) -> np.ndarray:
    """Per-frame FLAME landmarks in their own local frame (zero
    `global_orient`/`transl`, head rotation/translation deliberately
    excluded, since this feeds `face_blendshapes.solve_arkit_weights`,
    which needs the expression-driven deformation only, in the same
    canonical frame `scripts/build_face_bases.py`'s own offline basis was
    built in), minus that same zero-pose/zero-expression neutral. `betas`
    is fixed at zero rather than the clip's own tracked identity: FLAME's
    expression blendshapes add before skinning, so the expression-driven
    component of this delta is betas-invariant by construction, and jaw's
    own (small, second-order) betas-dependence isn't worth a second
    per-clip model build for. `jaw_pose`/`expression`: (F, 3)/(F, 50).
    Returns (F, 51, 3).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(jaw_pose)
    model = _build_flame_model(device, batch_size=n)

    with torch.no_grad():
        neutral = model(
            betas=torch.zeros(1, FLAME_NUM_BETAS, device=device),
            expression=torch.zeros(1, FLAME_NUM_EXPRESSION, device=device),
            global_orient=torch.zeros(1, 3, device=device), neck_pose=torch.zeros(1, 3, device=device),
            jaw_pose=torch.zeros(1, 3, device=device),
            leye_pose=torch.zeros(1, 3, device=device), reye_pose=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
        )
        neutral_landmarks = neutral.joints[0, -NUM_FLAME_LANDMARKS:].cpu().numpy()

        out = model(
            betas=torch.zeros(n, FLAME_NUM_BETAS, device=device),
            expression=torch.tensor(expression, device=device, dtype=torch.float32),
            global_orient=torch.zeros(n, 3, device=device), neck_pose=torch.zeros(n, 3, device=device),
            jaw_pose=torch.tensor(jaw_pose, device=device, dtype=torch.float32),
            leye_pose=torch.zeros(n, 3, device=device), reye_pose=torch.zeros(n, 3, device=device),
            transl=torch.zeros(n, 3, device=device),
        )
        landmarks = out.joints[:, -NUM_FLAME_LANDMARKS:].cpu().numpy()

    return landmarks - neutral_landmarks[None]
