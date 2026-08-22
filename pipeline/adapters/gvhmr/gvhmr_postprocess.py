"""Post-processing passes applied to GVHMR's raw per-frame predictions:
`pp_static_joint_cam` corrects drift in the world-space ("global") translation
using the static-camera assumption plus predicted foot/wrist "static"
confidence; `pp_static_joint_incam` applies that same static-joint drift lock
directly to the camera-space ("incam") translation this project actually uses
downstream (see its own docstring for why `pp_static_joint_cam` alone isn't
enough); `process_ik` runs a small CCD-IK cleanup so limbs actually reach the
corrected target positions instead of just moving the root;
`pp_bridge_low_confidence_root_motion` bridges incam's own root
(global_orient/transl) across frames where the 2D keypoint detector itself
lost track (see its own docstring).

Ported from `comfyui-motioncapture/nodes/gvhmr/postprocess.py`. **Only
`pp_static_joint_cam` is ported, not `pp_static_joint`**: GVHMR's own pipeline
picks between them based on `static_cam`, and this project is static-camera
only (confirmed at `Pipeline.forward`'s call site), `pp_static_joint` is the
moving-camera variant, never reached here. `pp_static_joint_incam`/
`pp_bridge_low_confidence_root_motion` and their helpers are this project's
own addition, not part of that port.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter, minimum_filter1d
from scipy.spatial.transform import Rotation

from ...algorithms.camera_gravity import LEVEL_CAMERA_UP
from ...algorithms.contact_detection import contiguous_true_runs
from ...algorithms.motion_smoothing import fill_invalid, hemisphere_aligned_quats
from .gvhmr_ccd_ik import CCD_IK
from .gvhmr_forward_kinematics import get_rotation
from .gvhmr_rotation_math import axis_angle_to_matrix, matrix_to_axis_angle

# The 6 joints whose predicted "static" confidence drives drift correction:
# left/right ankle, left/right foot, left/right wrist.
STATIC_JOINT_IDS = [7, 10, 8, 11, 20, 21]

# Hysteresis over static_conf_logits' own per-frame sigmoid confidence, same
# seed/release/rolling-window shape as contact_detection.py's own contact
# hysteresis (CONTACT_SEED_CONFIDENCE/CONTACT_RELEASE_CONFIDENCE/CONTACT_WINDOW)
#, a bare instantaneous `> 0.8` threshold chatters right at the boundary,
# locking and unlocking single frames at a time; a rolling-max release with a
# stricter seed only starts a lock on an unambiguous frame, then holds it
# through neighbouring frames that dip just under 0.8 but stay above 0.5.
STATIC_CONF_SEED = 0.8  # unchanged from the prior bare threshold
STATIC_CONF_RELEASE = 0.5
STATIC_CONF_WINDOW = 5

# Same seed/release/rolling-window hysteresis family, applied to mean 2D
# body-keypoint confidence (see `_unreliable_pose_label`) instead of the
# network's own static-joint logits, identifies runs where the *root's* own
# tracked motion should be distrusted (see `pp_bridge_low_confidence_root_motion`).
# Thresholds set from a real clip's own numbers: mean body-joint confidence sits ~0.75 in normally-tracked
# frames and collapses to ~0.25-0.5 (individual frames bottoming out under
# 0.1) for the ~20 frames the detector genuinely lost her. RELEASE=0.6 is
# comfortably below the normal baseline (so ordinary frames never enter the
# candidate window); SEED=0.45 sits inside the observed collapse but above
# the noise floor of a single bad frame, so an isolated frame dipping near
# the release band doesn't get flagged on its own.
POSE_CONF_SEED = 0.45
POSE_CONF_RELEASE = 0.6
POSE_CONF_WINDOW = 5

# SMPL body kinematic chains used by the IK cleanup pass (root + hip->knee->ankle->foot,
# root + shoulder->elbow->wrist), matching gvhmr_forward_kinematics.py's joint indexing.
LEFT_LEG_CHAIN = [0, 1, 4, 7, 10]
RIGHT_LEG_CHAIN = [0, 2, 5, 8, 11]
LEFT_HAND_CHAIN = [9, 13, 16, 18, 20]
RIGHT_HAND_CHAIN = [9, 14, 17, 19, 21]

# Stage 2's final root-height cleanup. These deliberately describe a stance
# rather than a generic "low foot": a walking foot should be free in flight,
# but a foot with little horizontal velocity near its local floor is useful
# evidence that any vertical root wobble is estimation noise.
#
# "Height" and "horizontal" here are measured against a clip's own measured
# gravity direction (`camera_gravity.estimate_camera_up`), not against camera
# Y.
STANCE_HORIZONTAL_SPEED_MPS = 0.12
STANCE_HEIGHT_ABOVE_FLOOR_M = 0.15
STANCE_MIN_FRAMES = 5
# A root may be allowed to move freely in height only after both feet have
# been clearly above their floor bands for this long. This is intentionally
# the same duration as a stance confirmation so a single uncertain frame
# cannot turn a supported bend into apparent flight.
FLIGHT_MIN_FRAMES = 5
STANCE_EDGE_BLEND_FRAMES = 4
STANCE_FOOT_MEDIAN_FRAMES = 5
STANCE_CORRECTION_SMOOTH_SIGMA = 3.0
# This is a correction-rate guard, not a physical root-speed limit. The
# ordinary Stage 2 filter owns normal body motion; this only prevents a noisy
# change in stance classification from injecting a visible whole-body step.
# It is deliberately slack: on real clips it does not bind at all (raising it
# by three orders of magnitude changes the correction by 0.05cm), so it acts
# only against a pathological classification flip, never against ordinary
# stance-to-stance handover.
STANCE_MAX_CORRECTION_STEP_M = 0.0015

# Per-foot IK relock (`relock_stance_feet_with_ik`). The contact point held
# during a stance is the toe (SMPL's "foot" joint), not the ankle: measured on
# BEHAVE ground truth and on this project's own output, the toe joint is the
# lowest point of the foot in 100% of detected plants, so it is the contact
# patch, and holding it while leaving the ankle free is what lets a heel still
# lift. These are the last two entries of LEFT_LEG_CHAIN/RIGHT_LEG_CHAIN, and
# TOE_CHAIN_INDEX is that chain's own index for the toe.
LEFT_TOE_JOINT, RIGHT_TOE_JOINT = 10, 11
TOE_CHAIN_INDEX = 4

# How far the IK pass may move a planted toe back toward its stance target.
# Not 1.0, because a real planted foot genuinely does move: over BEHAVE's
# ground-truth plants the contact point travels 1.2-1.8cm horizontally and
# ~1.0cm vertically, so driving the residual to zero would be fitting past the
# real signal. This pulls most of the excess out while leaving the estimator's
# own foot motion recognisable.
STANCE_IK_CORRECTION_WEIGHT = 0.7

# CCD sweeps per frame. The pass only ever asks for a few centimetres, well
# inside the leg's reach, so this converges almost immediately; the existing
# `process_ik` cleanup uses the same budget.
STANCE_IK_MAX_ITERATIONS = 2

# Temporal smoothing of the IK *target*, in frames. Small on purpose: it exists
# to take the corners off stance-run junctions, not to filter the motion (the
# stage's own smoother already owns that).
STANCE_IK_TARGET_SMOOTH_SIGMA = 2.0


def _gaussian_smooth(x: torch.Tensor, sigma: float = 3.0, dim: int = -1) -> torch.Tensor:
    """1D Gaussian smoothing along `dim`, edge-replicated at the boundary.
    Reimplements the standard Gaussian-kernel formula directly (mean 0,
    normalized to sum to 1) rather than reaching into scipy's own
    underscore-prefixed internal helper, which the source calls directly."""
    radius = int(4 * sigma + 0.5)
    xs = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    kernel = (kernel / kernel.sum()).to(dtype=x.dtype, device=x.device).view(1, 1, -1)

    x = x.transpose(dim, -1)
    lead_shape = x.shape[:-1]
    flat = x.reshape(-1, 1, x.shape[-1])
    flat = F.pad(flat, (radius, radius), mode="replicate")
    smoothed = F.conv1d(flat, kernel)
    return smoothed.reshape(*lead_shape, -1).transpose(-1, dim)


def _transform_mat(rot: torch.Tensor, transl: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation + (..., 3) translation -> (..., 4, 4) transform."""
    return torch.cat([F.pad(rot, [0, 0, 0, 1]), F.pad(transl[..., None], [0, 0, 0, 1], value=1)], dim=-1)


def _apply_transform_to_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """points: (..., N, 3), transform: (..., 4, 4) -> (..., N, 3)."""
    return torch.einsum("...ki,...ji->...jk", transform[..., :3, :3], points) + transform[..., None, :3, 3]


def _static_label(static_conf_logits: torch.Tensor) -> torch.Tensor:
    """(B, T, J) sigmoid-able confidence logits -> (B, T, J) bool lock state,
    hysteresis applied independently per joint (see STATIC_CONF_SEED/RELEASE/
    WINDOW above) instead of a bare instantaneous threshold, which chatters
    right at the boundary, reuses contact_detection.py's own seed/release/
    run-finding logic (`contiguous_true_runs`), the same hysteresis family
    this project already applies to hand-object contact confidence, here
    applied per static-candidate joint instead of per contact region."""
    confidence = static_conf_logits.sigmoid().cpu().numpy()
    label = np.zeros_like(confidence, dtype=bool)
    B, _, J = confidence.shape
    for b in range(B):
        for j in range(J):
            c = confidence[b, :, j]
            elevated = maximum_filter1d(c, size=STATIC_CONF_WINDOW, mode="nearest") > STATIC_CONF_RELEASE
            seed = c > STATIC_CONF_SEED
            for start, end in contiguous_true_runs(elevated):
                if seed[start:end + 1].any():
                    label[b, start:end + 1, j] = True
    return torch.from_numpy(label).to(static_conf_logits.device)


def _static_joint_drift(post_j3d: torch.Tensor, static_label: torch.Tensor, zero_vertical: bool) -> torch.Tensor:
    """Per-frame-transition (B, L-1, 3) displacement to subtract from
    translation so joints the network is confident are stationary stop
    drifting, the shared lock math behind both `pp_static_joint_cam`
    (`zero_vertical=True`: that frame's own whole-clip floor snap already
    handles vertical placement, so per-frame vertical correction is left to
    it) and `pp_static_joint_incam` (`zero_vertical=False`, see that
    function's own docstring for why incam's vertical drift needs this same
    per-frame treatment instead)."""
    pred_j3d_static = post_j3d[:, :, STATIC_JOINT_IDS]
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]
    static_label_sumJ = torch.clamp_min(static_label.sum(-1, keepdim=True), 1)
    pred_disp = (pred_j_disp * static_label[..., None]).sum(-2) / static_label_sumJ
    if zero_vertical:
        pred_disp = pred_disp.clone()
        pred_disp[:, :, 1] = 0
    return pred_disp


def _stance_runs(label: np.ndarray, min_frames: int = STANCE_MIN_FRAMES) -> list[tuple[int, int]]:
    """Inclusive runs of a per-frame stance label, excluding short flickers."""
    return [run for run in contiguous_true_runs(label) if run[1] - run[0] + 1 >= min_frames]


def resolved_camera_up(camera_up: np.ndarray | list[float] | None) -> np.ndarray:
    """The clip's unit camera-space up direction, defaulting to a level camera."""
    up = np.asarray(LEVEL_CAMERA_UP if camera_up is None else camera_up, dtype=np.float64)
    return up / np.linalg.norm(up)


def _static_feet_label(static_conf_logits: torch.Tensor) -> np.ndarray:
    """(B, T, left/right) "GVHMR is confident this foot is static".
    STATIC_JOINT_IDS is [left ankle, left foot, right ankle, right foot, left
    wrist, right wrist]; only the four foot columns matter for a stance."""
    static = _static_label(static_conf_logits).detach().cpu().numpy()
    return np.stack([static[..., 0] | static[..., 1], static[..., 2] | static[..., 3]], axis=-1)


def _foot_stance_state(
    joints: np.ndarray, static_feet: np.ndarray, fps: float, up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(stance, flight, filtered_foot_height)` for one clip's (T, 22, 3) joints.

    `stance` is (T, left/right): this foot is bearing weight. `flight` is (T,):
    neither foot is anywhere near the floor for a sustained interval, the only
    state in which the body's height is genuinely free. `filtered_foot_height`
    is (T, left/right), the temporally median-filtered height of each side's
    lowest point.

    Shared by the root-height correction and the per-foot IK relock so the two
    can never disagree about which frames are a stance.
    """
    T = len(joints)
    height = joints @ up  # (T, 22), larger is higher
    # min(ankle, foot) is the lowest point this skeleton represents on each
    # side. Averaging the ankle/foot positions in the ground plane gives a
    # plant-position signal that is less sensitive to a toe roll.
    foot_height = np.stack([
        height[:, [7, 10]].min(axis=-1),
        height[:, [8, 11]].min(axis=-1),
    ], axis=-1)
    ground_plane = joints - height[..., None] * up
    foot_ground = np.stack([
        ground_plane[:, [7, 10]].mean(axis=1),
        ground_plane[:, [8, 11]].mean(axis=1),
    ], axis=1)  # (T, left/right, XYZ in the plane orthogonal to `up`)
    horizontal_speed = np.zeros((T, 2), dtype=np.float32)
    horizontal_speed[1:] = np.linalg.norm(np.diff(foot_ground, axis=0), axis=-1) * fps
    horizontal_speed[0] = horizontal_speed[1]

    filtered_foot_height = median_filter(foot_height, size=(STANCE_FOOT_MEDIAN_FRAMES, 1), mode="nearest")
    # A low percentile is a conservative floor estimate: it is near the lowest
    # observed foot point but cannot be pulled downward by a single bad frame,
    # nor upward by an airborne step. A foot more than this distance above it
    # is not a plausible support foot unless GVHMR itself says it is static.
    local_floor = np.percentile(filtered_foot_height, 25, axis=0)
    near_floor = filtered_foot_height <= local_floor + STANCE_HEIGHT_ABOVE_FLOOR_M
    stance = static_feet | ((horizontal_speed <= STANCE_HORIZONTAL_SPEED_MPS) & near_floor)

    # An entire contiguous run must be high above the local floor before it is
    # flight. In particular, one support foot keeps the root height anchored
    # while the other foot swings or the torso bends forward.
    flight = np.zeros(T, dtype=bool)
    for start, end in _stance_runs((~near_floor).all(axis=-1), min_frames=FLIGHT_MIN_FRAMES):
        flight[start:end + 1] = True
    return stance, flight, filtered_foot_height


def _stance_edge_weight(frame: int, start: int, end: int, n_frames: int) -> float:
    """Ease-in/ease-out weight for a correction across one stance run. At an
    interior edge the correction fades over `STANCE_EDGE_BLEND_FRAMES` so
    takeoff and landing do not snap; a clip boundary has no adjacent
    unconstrained pose to blend toward, so the full correction is kept there.

    Smoothstep rather than a straight line: a linear ramp is continuous in
    position but not in velocity, so each of its two corners lands as an
    acceleration spike on the joints being corrected. Measured on real clips,
    switching to smoothstep is what keeps the per-foot IK relock from raising
    peak foot acceleration even while it lowers the typical value.
    """
    fraction = 1.0
    if start > 0:
        fraction = min(fraction, (frame - start + 1) / STANCE_EDGE_BLEND_FRAMES)
    if end < n_frames - 1:
        fraction = min(fraction, (end - frame + 1) / STANCE_EDGE_BLEND_FRAMES)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _slew_sweep(signal: np.ndarray, max_step: float, direction: int) -> np.ndarray:
    """One in-place slew-limiting sweep over `signal`, forwards or backwards."""
    view = signal[::direction]
    for frame in range(1, len(view)):
        view[frame] = np.clip(view[frame], view[frame - 1] - max_step, view[frame - 1] + max_step)
    return signal


def _rate_limited(signal: np.ndarray, max_step: float) -> np.ndarray:
    """Slew-limit `signal` without biasing it in time.

    A single sweep lags every change it clamps, in whichever direction it
    happens to run; so does a sweep pair, since clamping is not linear and
    forwards-then-backwards is not backwards-then-forwards. Averaging the two
    orderings removes that bias by construction, and the average still obeys
    the limit because the set of signals satisfying it is convex.
    """
    forward_first = _slew_sweep(_slew_sweep(signal.copy(), max_step, 1), max_step, -1)
    backward_first = _slew_sweep(_slew_sweep(signal.copy(), max_step, -1), max_step, 1)
    return 0.5 * (forward_first + backward_first)


def stance_vertical_grounding_correction(
    post_c_j3d: torch.Tensor, static_conf_logits: torch.Tensor, fps: float,
    camera_up: np.ndarray | list[float] | None = None,
) -> torch.Tensor:
    """Return (B, T) root height corrections that make planted feet define height.

    This is the small, stance-aware part of a foot-grounding pass: unlike the
    ordinary translation low-pass, it does not guess that every vertical
    change is noise. A correction is only present while a foot is confidently
    static according to GVHMR, or is independently detected as nearly still
    horizontally and close to its own low-height baseline. Within each
    qualifying stance, the foot's temporally-median-filtered bottom is held to
    that stance's median height. That removes the low-frequency pelvis wobble
    a planted foot cannot physically cause; four-frame edge fades avoid a snap
    on takeoff/landing. If neither foot meets the stricter stance test but
    either remains near the floor, the last correction is held rather than
    releasing the root; a release needs both feet clearly airborne for a
    sustained interval, preserving true jumps while keeping waist bends
    grounded.

    `camera_up` is the clip's measured camera-space up direction, defaulting
    to a level camera's own -Y. Height and horizontal speed are both measured
    against it, so a tilted camera does not turn the subject's horizontal
    travel into apparent height. The returned scalar is a distance along that
    same direction, so a caller applies it as `transl += correction * up`.

    A root-only correction is knowingly a partial fix. The two planted feet drift
    almost independently of each other, which splits the drift about evenly into
    a common-mode part a single root offset can remove and a differential part only
    per-leg IK can. Absolute floor placement remains stage 10's responsibility,
    because camera-space motion itself has no floor origin.
    """
    if post_c_j3d.ndim != 4 or post_c_j3d.shape[-2:] != (22, 3):
        raise ValueError("post_c_j3d must have shape (B, T, 22, 3)")
    if static_conf_logits.shape[:2] != post_c_j3d.shape[:2] or static_conf_logits.shape[-1] != len(STATIC_JOINT_IDS):
        raise ValueError("static_conf_logits must have shape (B, T, 6) matching post_c_j3d")
    if fps <= 0:
        raise ValueError("fps must be positive")

    B, T = post_c_j3d.shape[:2]
    correction = np.zeros((B, T), dtype=np.float32)
    if T < STANCE_MIN_FRAMES:
        return torch.from_numpy(correction).to(device=post_c_j3d.device, dtype=post_c_j3d.dtype)

    up = resolved_camera_up(camera_up)
    joints = post_c_j3d.detach().cpu().numpy()
    static_feet = _static_feet_label(static_conf_logits)

    for b in range(B):
        stance, flight, filtered_foot_height = _foot_stance_state(joints[b], static_feet[b], fps, up)

        per_side = np.full((T, 2), np.nan, dtype=np.float32)
        for side in range(2):
            for start, end in _stance_runs(stance[:, side]):
                target = float(np.median(filtered_foot_height[start:end + 1, side]))
                values = target - filtered_foot_height[start:end + 1, side]
                for frame in range(start, end + 1):
                    per_side[frame, side] = values[frame - start] * _stance_edge_weight(frame, start, end, T)

        last_supported_correction = 0.0
        for frame in range(T):
            planted = ~np.isnan(per_side[frame])
            if planted.any():
                # Two planted feet can have incompatible estimates. Follow the
                # lower one: it is the foot actually bearing weight, and the
                # higher one is the one leg IK can still reach down to. The
                # smaller-magnitude choice looks safer but is not, because the
                # two feet's stance medians differ, so it oscillates between
                # them and collapses toward no correction at all.
                lower_side = np.argmin(np.where(planted, filtered_foot_height[frame], np.inf))
                correction[b, frame] = per_side[frame, lower_side]
                last_supported_correction = correction[b, frame]
            elif not flight[frame]:
                correction[b, frame] = last_supported_correction

        # Smooth this *new* correction signal before constraining its rate.
        # That removes a one-frame foot-pose glitch without re-filtering the
        # underlying root trajectory or leaking a correction far into flight.
        correction[b] = gaussian_filter1d(
            correction[b], sigma=STANCE_CORRECTION_SMOOTH_SIGMA, mode="nearest",
        )

        # Per-stance target medians can legitimately differ, but a classifier
        # switching from one foot to another must never become an instantaneous
        # whole-body translation. Rate-limit only this new correction signal;
        # the pre-grounding, already-smoothed root trajectory remains intact.
        correction[b] = _rate_limited(correction[b], STANCE_MAX_CORRECTION_STEP_M)

    return torch.from_numpy(correction).to(device=post_c_j3d.device, dtype=post_c_j3d.dtype)


def _stance_toe_targets(
    joints: np.ndarray, static_feet: np.ndarray, fps: float, up: np.ndarray,
) -> np.ndarray:
    """(T, left/right, 3) per-frame target position for each toe joint.

    Outside a stance the target is the joint's own current position, so the IK
    pass has nothing to do there. Inside one it is the stance's median toe
    position, eased in and out at the run's edges, which constrains the plant
    in all three dimensions rather than in height alone.

    Using a whole-stance median (not the previous frame's position) means the
    constraint cannot integrate its own error into a slow walk-away, which is
    the failure mode the root-height pass was already built to avoid.
    """
    T = len(joints)
    stance, _, _ = _foot_stance_state(joints, static_feet, fps, up)
    toes = joints[:, [LEFT_TOE_JOINT, RIGHT_TOE_JOINT]]  # (T, left/right, 3)

    # Built as an offset from the current position rather than as an absolute
    # target, so that it is exactly zero wherever no foot is planted. That
    # matters for the smoothing below: smoothing absolute positions would
    # low-pass the foot's real trajectory everywhere, dragging a fast-swinging
    # airborne foot centimetres off its own path, while smoothing the offset
    # only ever softens the correction's own edges.
    offset = np.zeros_like(toes)
    for side in range(2):
        for start, end in _stance_runs(stance[:, side]):
            planted = np.median(toes[start:end + 1, side], axis=0)
            for frame in range(start, end + 1):
                blend = STANCE_IK_CORRECTION_WEIGHT * _stance_edge_weight(frame, start, end, T)
                offset[frame, side] = (planted - toes[frame, side]) * blend
    # Two stance runs separated by a gap shorter than a stance can hold
    # noticeably different medians, and IK is solved per frame, so an unsmoothed
    # correction hands the solver a step at every such junction and shows up as
    # a foot acceleration spike.
    offset = gaussian_filter1d(offset, sigma=STANCE_IK_TARGET_SMOOTH_SIGMA, axis=0, mode="constant", cval=0.0)
    return toes + offset


def relock_stance_feet_with_ik(
    pred_smpl_params_incam: dict[str, torch.Tensor], static_conf_logits: torch.Tensor, endecoder,
    fps: float, camera_up: np.ndarray | list[float] | None = None,
) -> torch.Tensor:
    """Return a body_pose whose planted feet hold position in all three axes.

    The root-height pass (`stance_vertical_grounding_correction`) can only move
    the whole body, so it removes the part of the drift both feet share and
    nothing else, and it does not address horizontal drift at all. This pass
    takes the rest: it leaves the root exactly where it is and rotates each
    leg so that a planted toe stays put, which is the only way to fix two feet
    that are drifting differently at the same time.

    Only leg joint rotations change; `transl` and `global_orient` are
    untouched, so the body does not move as a whole and no other stage's view
    of the root changes.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    up = resolved_camera_up(camera_up)
    batched = {key: value.unsqueeze(0) for key, value in pred_smpl_params_incam.items()}
    joints, local_mat, world_mat = endecoder.fk_v2(**batched, get_intermediate=True)
    if joints.shape[1] < STANCE_MIN_FRAMES:
        # Raw key, like `process_ik` below: `gvhmr_adapter` owns the KEY_*
        # constants and imports this module, so it cannot be imported here.
        return pred_smpl_params_incam["body_pose"]

    static_feet = _static_feet_label(static_conf_logits.unsqueeze(0))
    targets = _stance_toe_targets(joints[0].detach().cpu().numpy(), static_feet[0], fps, up)
    targets = torch.from_numpy(targets).to(device=joints.device, dtype=joints.dtype).unsqueeze(0)

    # Rotation targets keep each foot's own orientation as CCD solves for
    # position, so holding the toe in place cannot silently twist the foot.
    global_rot = get_rotation(world_mat)
    for chain, toe in ((LEFT_LEG_CHAIN, LEFT_TOE_JOINT), (RIGHT_LEG_CHAIN, RIGHT_TOE_JOINT)):
        side = 0 if toe == LEFT_TOE_JOINT else 1
        solved = CCD_IK(
            local_mat, endecoder.parents, [TOE_CHAIN_INDEX],
            targets[:, :, [side]], global_rot[:, :, [toe]],
            kinematic_chain=chain, max_iter=STANCE_IK_MAX_ITERATIONS,
        ).solve()
        local_mat = local_mat.clone()
        local_mat[:, :, chain[1:], :-1, :-1] = get_rotation(solved)[:, :, 1:]

    return matrix_to_axis_angle(get_rotation(local_mat[:, :, 1:])).flatten(2)[0]


def _unreliable_pose_label(pose_confidence: torch.Tensor) -> torch.Tensor:
    """(B, T) mean body-keypoint confidence -> (B, T) bool "the root's own
    tracked motion here is unreliable" state. Same seed/release/rolling-window
    hysteresis family as `_static_label` (see POSE_CONF_SEED/RELEASE/WINDOW
    above), with the comparisons flipped since this labels *low*-confidence
    runs rather than high-confidence ones: a run is only confirmed unreliable
    if confidence genuinely bottoms out somewhere inside it (POSE_CONF_SEED),
    not merely dips near the release band, avoids bridging over ordinary
    single-frame noise."""
    confidence = pose_confidence.cpu().numpy()
    label = np.zeros_like(confidence, dtype=bool)
    B, _ = confidence.shape
    for b in range(B):
        c = confidence[b]
        dipped = minimum_filter1d(c, size=POSE_CONF_WINDOW, mode="nearest") < POSE_CONF_RELEASE
        seed = c < POSE_CONF_SEED
        for start, end in contiguous_true_runs(dipped):
            if seed[start:end + 1].any():
                label[b, start:end + 1] = True
    return torch.from_numpy(label).to(pose_confidence.device)


def pp_bridge_low_confidence_root_motion(pred_smpl_params: dict, pose_confidence: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Bridge global_orient/transl, the pelvis's own root orientation and
    world position, across any run flagged unreliable by
    `_unreliable_pose_label`, using the identical interior-interpolate/edge-
    freeze occlusion contract `motion_smoothing.fill_invalid` already
    established for hand-tracking gaps: a run bounded by reliable frames on
    both sides is bridged by interpolating directly between them (global_orient
    via `hemisphere_aligned_quats`, the same quaternion gap-fill
    `smooth_rotation_sequence` uses; transl via plain per-channel linear
    interpolation); a run touching either end of the clip has no second real
    endpoint to interpolate toward and is instead held constant at whichever
    single real value it does have.

    `body_pose` (the other 21 joints' own local rotations, elbows, knees,
    spine, etc.) is deliberately left untouched, and so is `betas` (body
    shape, not motion). An earlier version of this fix also froze body_pose,
    but during a genuine 2D-tracking dropout only the pelvis's own root
    motion actually reads as visually wrong, the other joints stay
    plausible even though the network's confidence in them is measured low
    too, and freezing them made the result look worse, not better.

    Returns `(bridged_params, label)`, `label` (the same (B, T) bool from
    `_unreliable_pose_label`) is returned too, not just used internally, so
    stage 10's own export can delete the pelvis bone's real Blender keyframes
    at these frames instead of just baking this function's own interpolated
    numbers into them: the values computed here still matter for every
    stage between this one and export (stage 6's scale fit, stage 7's contact
    projection, stage 8's attachment search all need dense, plausible
    per-frame numbers, not a gap), but the final exported file should show a
    real gap Blender itself interpolates across, rather than a baked
    keyframe that merely looks similar."""
    label = _unreliable_pose_label(pose_confidence)
    valid = (~label).cpu().numpy()
    bridged = {k: v.clone() for k, v in pred_smpl_params.items()}
    B = label.shape[0]
    for b in range(B):
        if valid[b].all():
            continue

        transl_np = bridged["transl"][b].cpu().numpy()
        bridged["transl"][b] = torch.from_numpy(fill_invalid(transl_np, valid[b])).to(bridged["transl"].dtype)

        orient_np = bridged["global_orient"][b].cpu().numpy()
        quats = hemisphere_aligned_quats(orient_np, valid[b])
        quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
        rotvec = Rotation.from_quat(quats).as_rotvec()
        bridged["global_orient"][b] = torch.from_numpy(rotvec).to(bridged["global_orient"].dtype)
    return bridged, label


def pp_static_joint_cam(outputs: dict, endecoder) -> torch.Tensor:
    """Correct the "global" (world-grounded) translation using the static-camera
    assumption: a genuinely static camera means the "incam" (camera-space)
    prediction's own joint motion, once aligned into world space via the first
    frame, is a second independent estimate of world motion, disagreements
    between the two beyond a small threshold get pulled back, and joints
    predicted as "static" this frame get locked in place to remove foot sliding.
    """
    pred_smpl_params_incam = dict(outputs["pred_smpl_params_incam"])
    pred_smpl_params_global = outputs["pred_smpl_params_global"]
    static_conf_logits = outputs["static_conf_logits"][:, :-1].clone()
    B, L = pred_smpl_params_incam["transl"].shape[:2]
    assert B == 1

    pred_w_j3d = endecoder.fk_v2(**pred_smpl_params_global)
    # The incam prediction is noisier (no temporal smoothing baked in like the
    # global prediction has); smooth its translation before using it as a
    # cross-check signal.
    pred_smpl_params_incam["transl"] = _gaussian_smooth(pred_smpl_params_incam["transl"], sigma=5, dim=-2)
    pred_c_j3d = endecoder.fk_v2(**pred_smpl_params_incam)

    # Align the camera-space skeleton into world space via a single rigid
    # transform computed from frame 0 (where both predictions must agree on
    # the root, by definition).
    R_gv = axis_angle_to_matrix(pred_smpl_params_global["global_orient"][:, 0])
    R_c = axis_angle_to_matrix(pred_smpl_params_incam["global_orient"][:, 0])
    R_c2w = R_gv @ R_c.transpose(-1, -2)
    t_c2w = pred_w_j3d[:, 0, 0] - torch.einsum("bij,bj->bi", R_c2w, pred_c_j3d[:, 0, 0])
    T_c2w = _transform_mat(R_c2w, t_c2w)
    pred_c_j3d_in_w = _apply_transform_to_points(pred_c_j3d, T_c2w[:, None])

    post_w_transl = pred_smpl_params_global["transl"].clone()
    post_w_j3d = pred_w_j3d.clone()
    cp_thr = torch.tensor([0.25, 0.25, 0.25], device=post_w_j3d.device, dtype=post_w_j3d.dtype)
    for i in range(1, L):
        cp_diff = post_w_j3d[:, i, 0] - pred_c_j3d_in_w[:, i, 0]
        cp_diff = cp_diff * ~((cp_diff > -cp_thr) * (cp_diff < cp_thr))  # only correct genuinely large disagreements
        cp_diff = torch.clamp(cp_diff, -0.02, 0.02)  # small per-frame correction, not a snap
        post_w_transl[:, i:] -= cp_diff
        post_w_j3d[:, i:] -= cp_diff[:, None, None]

    # Lock joints the network is confident are stationary this frame, removing
    # foot-sliding drift that would otherwise accumulate frame over frame.
    static_label = _static_label(static_conf_logits)
    pred_disp = _static_joint_drift(post_w_j3d, static_label, zero_vertical=True)

    for i in range(1, L):
        post_w_transl[:, i:] -= pred_disp[:, [i - 1]]
        post_w_j3d[:, i:] -= pred_disp[:, [i - 1], None]

    # Put the sequence on the ground (does not account for actual foot height).
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]
    post_w_transl[..., 1] -= ground_y
    return post_w_transl


def pp_static_joint_incam(outputs: dict, endecoder) -> torch.Tensor:
    """Cancels drift in incam's own translation the same way `pp_static_joint_cam`
    does for `global`, lock joints the network is confident are stationary,
    but self-contained: incam's own FK positions, incam's own translation, no
    camera cross-check (that function's `cp_diff` correction is inherently a
    global-vs-incam agreement check with no incam-only equivalent, and isn't
    needed here). This is the fix that actually reaches a real run: every
    stage past stage 2 consumes incam exclusively (see stage_10_export.py's own
    module docstring), so `pp_static_joint_cam`'s identical-looking correction
    on `global` never reaches anything downstream, `global` is vestigial
    past this point, feeding only one optional debug preview.

    Unlike `pp_static_joint_cam`, this also corrects vertical (Y) drift
    per-frame instead of zeroing it out: incam is camera-space (X-right/
    Y-down/Z-forward), and `bvh_export.CAMERA_TO_BVH_ROOT_ROTATION` (this
    project's own camera-space -> upright change of basis, applied downstream
    at export) maps output Y to exactly `-input Y` with no mixing from X/Z,
    confirming incam's own Y axis already *is* the real vertical axis, just
    sign-flipped, not merely a convenient approximation. Locking a confidently
    -static ankle's own height the identical way its horizontal position is
    locked reproduces a real squat-vs-jump distinction for free, with no
    separate classifier needed: a squat keeps that ankle's own static
    confidence high (the correction fires), a real jump drops it as the foot
    leaves the ground (the correction doesn't). No whole-clip "put on the
    ground" snap is applied here either, that step assumes a gravity-
    aligned, floored world incam doesn't have; the equivalent already exists,
    correctly, downstream in stage_10_export.py's own `_lowest_foot_z`/
    `floor_offset`.
    """
    pred_smpl_params_incam = outputs["pred_smpl_params_incam"]
    static_conf_logits = outputs["static_conf_logits"][:, :-1]
    L = pred_smpl_params_incam["transl"].shape[1]

    post_c_j3d = endecoder.fk_v2(**pred_smpl_params_incam)
    post_c_transl = pred_smpl_params_incam["transl"].clone()

    static_label = _static_label(static_conf_logits)
    pred_disp = _static_joint_drift(post_c_j3d, static_label, zero_vertical=False)
    for i in range(1, L):
        post_c_transl[:, i:] -= pred_disp[:, [i - 1]]
    return post_c_transl


def process_ik(outputs: dict, endecoder) -> torch.Tensor:
    """Nudge each limb's joints via CCD-IK toward a target that blends the
    previous frame's position (weighted by static confidence) with this
    frame's raw FK position, cleans up the small pops/jitters that `
    pp_static_joint_cam`'s translation-only correction can't fix, since that
    pass never touches individual joint rotations."""
    static_conf = outputs["static_conf_logits"].sigmoid()
    post_w_j3d, local_mat, post_w_mat = endecoder.fk_v2(**outputs["pred_smpl_params_global"], get_intermediate=True)

    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, STATIC_JOINT_IDS]
        this = post_w_j3d[:, i, STATIC_JOINT_IDS]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, STATIC_JOINT_IDS] = prev * c_prev + this * (1 - c_prev)

    global_rot = get_rotation(post_w_mat)

    def _ik(local_mat: torch.Tensor, target_pos: torch.Tensor, target_rot: torch.Tensor,
            target_ind: list[int], chain: list[int]) -> torch.Tensor:
        local_mat = local_mat.clone()
        solved_chain = CCD_IK(local_mat, endecoder.parents, target_ind, target_pos, target_rot,
                               kinematic_chain=chain, max_iter=2).solve()
        chain_rotmat = get_rotation(solved_chain)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]
        return local_mat

    local_mat = _ik(local_mat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], LEFT_LEG_CHAIN)
    local_mat = _ik(local_mat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], RIGHT_LEG_CHAIN)
    local_mat = _ik(local_mat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], LEFT_HAND_CHAIN)
    local_mat = _ik(local_mat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], RIGHT_HAND_CHAIN)

    body_pose = matrix_to_axis_angle(get_rotation(local_mat[:, :, 1:]))
    return body_pose.flatten(2)
