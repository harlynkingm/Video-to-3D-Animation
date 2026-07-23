"""Skeletal forward kinematics: given each joint's own local rotation + its
offset from its parent (in a rest pose), compute every joint's actual
world-space position by composing transforms down the kinematic chain.
Needed by `EnDecoder.fk_v2` (turning predicted joint rotations into 3D joint
positions for the static-foot-lock postprocessing) and `process_ik` (the
CCD-IK cleanup pass).

Ported from `comfyui-motioncapture/nodes/motion_utils/matrix.py`, which is a
much larger, generic transform/quaternion toolkit (`matrix.py`, hence the
generic name) -- restricted here to just the four functions GVHMR's own code
actually calls (`get_TRS`, `forward_kinematics`, `get_position`,
`get_rotation`), renamed to this file to make that scope explicit rather than
implying a general-purpose transform library. The many
quaternion/euler/velocity helpers in the source file are for other parts of
GVHMR's training pipeline this project never reaches.
"""

from __future__ import annotations

import torch


def normalize_transform(mat: torch.Tensor) -> torch.Tensor:
    """Rescale a 4x4 (or 3x3) transform's rotation columns back to unit length.
    An approximate re-orthonormalization (not full Gram-Schmidt) -- matches the
    source exactly, used after every matrix multiply below to keep small
    floating-point drift from accumulating into a non-rotation matrix over a
    long kinematic chain."""
    if mat.shape[-1] == 4:
        rot = mat[..., :-1, :-1]
    else:
        rot = mat
    rot_norm = rot / (rot.norm(2, dim=-2, keepdim=True) + 1e-9)

    if mat.shape[-1] == 4:
        out = torch.zeros_like(mat)
        out[..., :-1, :-1] = rot_norm
        out[..., :-1, -1] = mat[..., :-1, -1]
        out[..., -1, -1] = 1.0
        return out
    return rot_norm


def get_TRS(rot_mat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation + (..., 3) position -> (..., 4, 4) homogeneous transform."""
    mat = torch.eye(4, device=pos.device, dtype=pos.dtype).repeat(pos.shape[:-1] + (1, 1))
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    return normalize_transform(mat)


def get_rotation(mat: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) -> (..., 3, 3) rotation part."""
    return mat[..., :-1, :-1]


def get_position(mat: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) -> (..., 3) translation part."""
    return mat[..., :-1, 3]


def _compose(parent_world: torch.Tensor, local_to_parent: torch.Tensor) -> torch.Tensor:
    """World transform of a child, given its parent's world transform and the
    child's own transform relative to that parent."""
    return normalize_transform(parent_world @ local_to_parent)


def get_mat_BtoA(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """Given two transforms in the same (e.g. world) space, return B expressed
    relative to A -- i.e. `inverse(A) @ B`. Used by the CCD-IK solver to turn a
    solved world-space rotation back into a joint's local rotation relative to
    its (unchanged) parent."""
    return normalize_transform(torch.inverse(mat_a) @ mat_b)


def forward_kinematics(local_transforms: torch.Tensor, parents: list[int]) -> torch.Tensor:
    """Walk a kinematic tree, turning each joint's local transform (relative to
    its parent) into its actual world transform.

    Args:
        local_transforms: (..., J, 4, 4) -- joint `i`'s own rotation + its
            rest-pose offset from joint `parents[i]`.
        parents: length-J list, `parents[i]` is joint i's parent index, or -1
            for the root. Must list each joint after its own parent (true for
            SMPL-X's own joint ordering, which this is always called with).
    Returns:
        (..., J, 4, 4) world transforms.
    """
    world = torch.eye(local_transforms.shape[-1], device=local_transforms.device, dtype=local_transforms.dtype)
    world = world.repeat(local_transforms.shape[:-2] + (1, 1))

    for i in range(local_transforms.shape[-3]):
        if parents[i] != -1:
            new_joint = _compose(world[..., parents[i], :, :], local_transforms[..., i, :, :])
            world = torch.cat([world[..., :i, :, :], new_joint[..., None, :, :], world[..., i + 1:, :, :]], dim=-3)
        else:
            world = torch.cat([local_transforms[..., :i + 1, :, :], world[..., i + 1:, :, :]], dim=-3)

    return world
