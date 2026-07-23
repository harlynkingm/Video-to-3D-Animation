"""Rotation-representation conversions (axis-angle / quaternion / matrix / 6D)
needed by the GVHMR port's decode and postprocessing steps.

Copied (not reimplemented from scratch) from `comfyui-motioncapture/nodes/
motion_utils/pytorch3d_shim.py`, itself a pure-PyTorch port of the same
functions in Meta's `facebookresearch/pytorch3d` -- BSD-3-Clause, unlike
GVHMR's own restricted-use license, so directly reusable here with
attribution (see the notice below), not a clean-room rewrite.

Restricted to the one code path this project's port actually exercises:
every call site in the reference uses the default `fast=False` (quaternion-
based) conversion, never the alternate Rodrigues-formula fast path -- so
`axis_angle_to_matrix`/`matrix_to_axis_angle` below only implement that one
path, and the euler-angle/so3-exp-log aliases (never called anywhere in the
GVHMR code this project ports) are dropped entirely.

Original source: https://github.com/facebookresearch/pytorch3d
License: BSD-3-Clause
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of the pytorch3d source tree.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """sqrt(max(0, x)), with a zero (not NaN) subgradient where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Flip a unit quaternion so its real part is non-negative (a unique
    representative, since q and -q represent the same rotation)."""
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) real-part-first quaternion -> (..., 3, 3) rotation matrix."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 4) real-part-first quaternion."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(torch.stack(
        [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1,
    ))

    # The desired quaternion, multiplied by each of r, i, j, k -- picking the
    # best-conditioned candidate (largest q_abs) avoids the near-zero-denominator
    # instability any single formula has near specific rotation angles.
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 4) real-part-first quaternion."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat([torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) real-part-first quaternion -> (..., 3) axis-angle."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    sin_half_angles_over_angles = 0.5 * torch.sinc(half_angles / torch.pi)
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 3) axis-angle."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) continuous 6D rotation representation (Zhou et al. 2019) ->
    (..., 3, 3) rotation matrix, via Gram-Schmidt orthogonalization."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 6) 6D representation (drops the last row;
    not a unique inverse of `rotation_6d_to_matrix`, but a valid one)."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))
