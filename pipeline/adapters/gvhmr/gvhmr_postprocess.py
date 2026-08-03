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
only (confirmed at `Pipeline.forward`'s call site) -- `pp_static_joint` is the
moving-camera variant, never reached here. `pp_static_joint_incam`/
`pp_bridge_low_confidence_root_motion` and their helpers are this project's
own addition, not part of that port.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.spatial.transform import Rotation

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
# -- a bare instantaneous `> 0.8` threshold chatters right at the boundary,
# locking and unlocking single frames at a time; a rolling-max release with a
# stricter seed only starts a lock on an unambiguous frame, then holds it
# through neighbouring frames that dip just under 0.8 but stay above 0.5.
STATIC_CONF_SEED = 0.8  # unchanged from the prior bare threshold
STATIC_CONF_RELEASE = 0.5
STATIC_CONF_WINDOW = 5

# Same seed/release/rolling-window hysteresis family, applied to mean 2D
# body-keypoint confidence (see `_unreliable_pose_label`) instead of the
# network's own static-joint logits -- identifies runs where the *root's* own
# tracked motion should be distrusted (see `pp_bridge_low_confidence_root_motion`).
# Thresholds set from a real clip's own numbers (a gymnast mid-tumble,
# motion-blurred): mean body-joint confidence sits ~0.75 in normally-tracked
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
    right at the boundary -- reuses contact_detection.py's own seed/release/
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
    drifting -- the shared lock math behind both `pp_static_joint_cam`
    (`zero_vertical=True`: that frame's own whole-clip floor snap already
    handles vertical placement, so per-frame vertical correction is left to
    it) and `pp_static_joint_incam` (`zero_vertical=False` -- see that
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


def _unreliable_pose_label(pose_confidence: torch.Tensor) -> torch.Tensor:
    """(B, T) mean body-keypoint confidence -> (B, T) bool "the root's own
    tracked motion here is unreliable" state. Same seed/release/rolling-window
    hysteresis family as `_static_label` (see POSE_CONF_SEED/RELEASE/WINDOW
    above), with the comparisons flipped since this labels *low*-confidence
    runs rather than high-confidence ones: a run is only confirmed unreliable
    if confidence genuinely bottoms out somewhere inside it (POSE_CONF_SEED),
    not merely dips near the release band -- avoids bridging over ordinary
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
    """Bridge global_orient/transl -- the pelvis's own root orientation and
    world position -- across any run flagged unreliable by
    `_unreliable_pose_label`, using the identical interior-interpolate/edge-
    freeze occlusion contract `motion_smoothing.fill_invalid` already
    established for hand-tracking gaps: a run bounded by reliable frames on
    both sides is bridged by interpolating directly between them (global_orient
    via `hemisphere_aligned_quats`, the same quaternion gap-fill
    `smooth_rotation_sequence` uses; transl via plain per-channel linear
    interpolation); a run touching either end of the clip has no second real
    endpoint to interpolate toward and is instead held constant at whichever
    single real value it does have.

    `body_pose` (the other 21 joints' own local rotations -- elbows, knees,
    spine, etc.) is deliberately left untouched, and so is `betas` (body
    shape, not motion). An earlier version of this fix also froze body_pose
    -- reviewing a real export, the user found only the pelvis's own root
    motion actually looked wrong during a genuine 2D-tracking dropout; the
    other joints stayed visually plausible even though the network's
    confidence in them was measured low too, and freezing them made the
    result look worse, not better. This mirrors exactly how the user resolved
    it by hand in Blender: deleting the pelvis bone's own keyframes across the
    bad stretch (which Blender then interpolates across from its own
    surrounding keyframes) while leaving every other bone's keyframes alone.

    Returns `(bridged_params, label)` -- `label` (the same (B, T) bool from
    `_unreliable_pose_label`) is returned too, not just used internally, so
    stage 9's own export can delete the pelvis bone's real Blender keyframes
    at these frames instead of just baking this function's own interpolated
    numbers into them: the values computed here still matter for every
    stage between this one and export (stage 6's scale fit, stage 7's contact
    projection, stage 8's attachment search all need dense, plausible
    per-frame numbers, not a gap), but the final exported file should show a
    real gap Blender itself interpolates across, per the user's own explicit
    request -- matching their own manual keyframe-deletion approach exactly,
    rather than a baked keyframe that merely looks similar."""
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
    frame, is a second independent estimate of world motion -- disagreements
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
    does for `global` -- lock joints the network is confident are stationary --
    but self-contained: incam's own FK positions, incam's own translation, no
    camera cross-check (that function's `cp_diff` correction is inherently a
    global-vs-incam agreement check with no incam-only equivalent, and isn't
    needed here). This is the fix that actually reaches a real run: every
    stage past stage 2 consumes incam exclusively (see stage_9_export.py's own
    module docstring), so `pp_static_joint_cam`'s identical-looking correction
    on `global` never reaches anything downstream -- `global` is vestigial
    past this point, feeding only one optional debug preview.

    Unlike `pp_static_joint_cam`, this also corrects vertical (Y) drift
    per-frame instead of zeroing it out: incam is camera-space (X-right/
    Y-down/Z-forward), and `bvh_export.CAMERA_TO_BVH_ROOT_ROTATION` (this
    project's own camera-space -> upright change of basis, applied downstream
    at export) maps output Y to exactly `-input Y` with no mixing from X/Z --
    confirming incam's own Y axis already *is* the real vertical axis, just
    sign-flipped, not merely a convenient approximation. Locking a confidently
    -static ankle's own height the identical way its horizontal position is
    locked reproduces a real squat-vs-jump distinction for free, with no
    separate classifier needed: a squat keeps that ankle's own static
    confidence high (the correction fires), a real jump drops it as the foot
    leaves the ground (the correction doesn't). No whole-clip "put on the
    ground" snap is applied here either -- that step assumes a gravity-
    aligned, floored world incam doesn't have; the equivalent already exists,
    correctly, downstream in stage_9_export.py's own `_lowest_foot_z`/
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
    frame's raw FK position -- cleans up the small pops/jitters that `
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
