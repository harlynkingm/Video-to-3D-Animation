"""Two post-processing passes applied to GVHMR's raw per-frame predictions:
`pp_static_joint_cam` corrects drift in the world-space translation using the
static-camera assumption plus predicted foot/wrist "static" confidence, and
`process_ik` runs a small CCD-IK cleanup so limbs actually reach the corrected
target positions instead of just moving the root.

Ported from `comfyui-motioncapture/nodes/gvhmr/postprocess.py`. **Only
`pp_static_joint_cam` is ported, not `pp_static_joint`**: GVHMR's own pipeline
picks between them based on `static_cam`, and this project is static-camera
only (confirmed at `Pipeline.forward`'s call site) -- `pp_static_joint` is the
moving-camera variant, never reached here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .gvhmr_ccd_ik import CCD_IK
from .gvhmr_forward_kinematics import get_rotation
from .gvhmr_rotation_math import axis_angle_to_matrix, matrix_to_axis_angle

# The 6 joints whose predicted "static" confidence drives drift correction:
# left/right ankle, left/right foot, left/right wrist.
STATIC_JOINT_IDS = [7, 10, 8, 11, 20, 21]

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
    pred_j3d_static = post_w_j3d[:, :, STATIC_JOINT_IDS]
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]

    static_label = static_conf_logits.sigmoid() > 0.8
    static_label_sumJ = torch.clamp_min(static_label.sum(-1, keepdim=True), 1)
    pred_disp = (pred_j_disp * static_label[..., None]).sum(-2) / static_label_sumJ
    pred_disp[:, :, 1] = 0  # never adjust vertical (height) drift this way

    for i in range(1, L):
        post_w_transl[:, i:] -= pred_disp[:, [i - 1]]
        post_w_j3d[:, i:] -= pred_disp[:, [i - 1], None]

    # Put the sequence on the ground (does not account for actual foot height).
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]
    post_w_transl[..., 1] -= ground_y
    return post_w_transl


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
