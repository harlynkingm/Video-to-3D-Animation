"""World-frame translation math for GVHMR's "global" (world-grounded) pose:
turning the network's predicted per-frame local translation *velocity* into
absolute translation, and reorienting a pose into the pipeline's target
gravity-aligned coordinate convention.

Ported from `comfyui-motioncapture/nodes/motion_utils/hmr_global.py`,
restricted to the two functions the actual call site
(`gvhmr/model.py`'s `get_smpl_params_w_Rt_v2`) uses: `rollout_local_transl_vel`
and `get_tgtcoord_rootparam`. Everything else in the source -- `get_local_transl_vel`
and its many "alignhead"/"absy"/"absgy" variants -- is training-data-preparation
code (the forward/inverse of what's needed here, for different coordinate
conventions never used by this project's inference path) and isn't ported.
`get_tgtcoord_rootparam`'s `gravity_vec`/`tgt_gravity_vec` branch is dropped
too: the only call site never passes them, and that branch is unreachable in
the source itself (it starts with `raise NotImplementedError`).
"""

from __future__ import annotations

import torch

from .gvhmr_rotation_math import axis_angle_to_matrix, matrix_to_axis_angle

# Fixed axis-angle rotations for each supported gravity-convention change. Only
# "any->ay" is ever actually requested by this project's call site; the rest
# are kept since they're just data, not logic, matching the source's own table.
_TSF_AXISANGLE = {
    "ay->ay": [0.0, 0.0, 0.0],
    "any->ay": [0.0, 0.0, torch.pi],
    "az->ay": [-torch.pi / 2, 0.0, 0.0],
    "ay->any": [0.0, 0.0, torch.pi],
}


def get_tgtcoord_rootparam(
    global_orient: torch.Tensor, transl: torch.Tensor, tsf: str = "ay->ay",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate a pose's root orientation and translation around the origin to
    match a different gravity-direction convention (e.g. "any" world-tracking
    axes -> "ay", this project's y-up gravity-aligned convention).

    Args:
        global_orient: (..., 3) axis-angle root orientation.
        transl: (..., 3) translation, in the same frame as global_orient.
        tsf: one of `_TSF_AXISANGLE`'s keys.
    Returns:
        (tgt_global_orient, tgt_transl, R_g2tg) -- reoriented orientation/translation
        and the rotation matrix used, in case a caller needs it too.
    """
    aa = torch.tensor(_TSF_AXISANGLE[tsf], device=global_orient.device, dtype=global_orient.dtype)
    R_g2tg = axis_angle_to_matrix(aa)  # (3, 3)

    global_orient_R = axis_angle_to_matrix(global_orient)
    tgt_global_orient = matrix_to_axis_angle(R_g2tg @ global_orient_R)
    tgt_transl = torch.einsum("...ij,...j->...i", R_g2tg, transl)
    return tgt_global_orient, tgt_transl, R_g2tg


def rollout_local_transl_vel(
    local_transl_vel: torch.Tensor, global_orient: torch.Tensor, transl_0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate per-frame local (root-relative) translation velocity into
    absolute (world-frame) translation.

    Args:
        local_transl_vel: (..., L, 3) predicted velocity, in the root's own local frame.
        global_orient: (..., L, 3) axis-angle root orientation, same frame per-frame.
        transl_0: (..., 1, 3) starting position; defaults to the origin.
    Returns:
        (..., L, 3) absolute translation per frame.
    """
    global_orient_R = axis_angle_to_matrix(global_orient)
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel)

    if transl_0 is None:
        transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    shifted = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)
    return torch.cumsum(shifted, dim=-2)
