"""A small iterative (CCD-style) inverse-kinematics solver used only by
`gvhmr_postprocess.py`'s limb cleanup pass: given a kinematic chain (e.g. hip
-> knee -> ankle -> foot) and a target world position/rotation for its end
joint, nudge every joint's local rotation along the chain so the end joint
gets closer to the target, without moving the root.

Ported from `comfyui-motioncapture/nodes/motion_utils/ccd_ik.py`. That file
pulls in a handful of quaternion helpers (`qinv`/`qmul`/`qrot`/`qslerp`/
`qbetween`) from a separate, much larger `quaternion.py` module -- ported
directly below as private helpers instead, since nothing else in this port
needs a general-purpose quaternion library. Quaternions here are (w, x, y, z),
real part first -- the same convention `gvhmr_rotation_math.py` uses, so the
two mix freely.
"""

from __future__ import annotations

import torch

from .gvhmr_forward_kinematics import forward_kinematics, get_mat_BtoA, get_position, get_rotation
from .gvhmr_rotation_math import matrix_to_quaternion


def _qinv(q: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(q)
    mask[..., 1:] = -1
    return q * mask


def _qnormalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)


def _qmul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    original_shape = q.shape
    terms = torch.bmm(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def _qpow(q0: torch.Tensor, t: float) -> torch.Tensor:
    q0 = _qnormalize(q0)
    theta0 = torch.acos(q0[..., :1])
    near_zero = (theta0.abs() <= 1e-9).float()
    theta0 = (1 - near_zero) * theta0 + near_zero * 1e-9
    v0 = q0[..., 1:] / torch.sin(theta0)

    theta = t * theta0
    q = torch.zeros_like(q0)
    q[..., :1] = torch.cos(theta)
    q[..., 1:] = v0 * torch.sin(theta)
    return q


def _qslerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    q0 = _qnormalize(q0)
    q1 = _qnormalize(q1)
    q_ = _qpow(_qmul(q1, _qinv(q0)), t)
    return _qmul(q_, q0)


def _qbetween(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """The quaternion that rotates v0 onto v1."""
    v = torch.cross(v0, v1, dim=-1)
    w = torch.sqrt((v0 ** 2).sum(-1, keepdim=True) * (v1 ** 2).sum(-1, keepdim=True)) + (v0 * v1).sum(-1, keepdim=True)
    y_axis = torch.zeros_like(v)
    y_axis[..., 1] = 1.0
    # v0 and v1 exactly opposite (or exactly equal): cross product degenerates to
    # zero, so pick an arbitrary perpendicular axis (y) as the rotation axis instead.
    degenerate = (v.norm(dim=-1) == 0) & (w.sum(-1).abs() <= 1e-4)
    v[degenerate] = y_axis[degenerate]
    return _qnormalize(torch.cat([w, v], dim=-1))


class CCD_IK:
    """Solves one kinematic chain at a time (e.g. one leg or one arm), toward
    a single target end-effector position (and optionally rotation).
    """

    def __init__(
        self,
        local_mat: torch.Tensor,
        parents: list[int],
        target_ind: list[int],
        target_pos: torch.Tensor | None = None,
        target_rot: torch.Tensor | None = None,
        kinematic_chain: list[int] | None = None,
        max_iter: int = 2,
        pos_weight: float = 1.0,
        rot_weight: float = 0.0,
    ):
        if kinematic_chain is None:
            kinematic_chain = list(range(local_mat.shape[-3]))
        global_mat = forward_kinematics(local_mat, parents)

        # Work on just this chain's joints, re-rooted at the chain's own first
        # joint (its real world transform, so the rest of the skeleton is
        # unaffected) -- iteration below never touches index 0, so the root
        # never moves.
        local_mat = local_mat.clone()[..., kinematic_chain, :, :]
        local_mat[..., 0, :, :] = global_mat[..., kinematic_chain[0], :, :]
        chain_parents = [i - 1 for i in range(len(kinematic_chain))]

        self.local_mat = local_mat
        self.global_mat = forward_kinematics(local_mat, chain_parents)
        self.parent = chain_parents

        self.target_ind = target_ind
        self.target_pos = target_pos
        self.target_q = matrix_to_quaternion(target_rot) if target_rot is not None else None

        self.J_N = self.local_mat.shape[-3]
        self.target_N = len(target_ind)
        self.max_iter = max_iter
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight

    def solve(self) -> torch.Tensor:
        for _ in range(self.max_iter):
            self._optimize(1)  # never touches joint 0 (the chain's root)
        return self.local_mat

    def _optimize(self, i: int) -> None:
        if i == self.J_N - 1:
            return
        pos = get_position(self.global_mat)[..., i, :]
        quat = matrix_to_quaternion(get_rotation(self.global_mat)[..., i, :, :])

        x_axis = torch.zeros(quat.shape[:-1] + (3,), device=quat.device)
        x_axis[..., 0] = 1.0
        y_axis = torch.zeros_like(x_axis)
        y_axis[..., 1] = 1.0
        x_sum, y_sum, count = torch.zeros_like(x_axis), torch.zeros_like(y_axis), 0

        for target_i, j in enumerate(self.target_ind):
            if i >= j:
                continue  # never optimize a joint at or past its own target

            if self.target_pos is not None:
                end_pos = get_position(self.global_mat)[..., j, :]
                target_pos = self.target_pos[..., target_i, :]
                # Rotate this joint just enough to point the end-effector at the target.
                aim_quat = _qslerp(quat, _qmul(_qbetween(end_pos - pos, target_pos - pos), quat), self._weight(i))
                x_sum += _qrot(aim_quat, x_axis)
                y_sum += _qrot(aim_quat, y_axis)
                if self.pos_weight > 0:
                    count += 1

            if self.target_q is not None:
                if target_i < self.target_N - 1:
                    continue  # multiple rotation targets are unstable; only honor the last
                end_quat = matrix_to_quaternion(get_rotation(self.global_mat)[..., j, :, :])
                target_q = self.target_q[..., target_i, :]
                aim_quat = _qslerp(quat, _qmul(_qmul(target_q, _qinv(end_quat)), quat), self._weight(i))
                x_sum += _qrot(aim_quat, x_axis) * self.rot_weight
                y_sum += _qrot(aim_quat, y_axis) * self.rot_weight
                if self.rot_weight > 0:
                    count += 1

        if count > 0:
            x_avg = _normalize(x_sum / count)
            y_avg = _normalize(y_sum / count)
            z_avg = torch.cross(x_avg, y_avg, dim=-1)
            solved_world_rot = torch.stack([x_avg, y_avg, z_avg], dim=-1)  # columns = new basis vectors

            parent_rot = get_rotation(self.global_mat)[..., self.parent[i], :, :]
            self.local_mat[..., i, :-1, :-1] = get_mat_BtoA(parent_rot, solved_world_rot)
            self.global_mat = forward_kinematics(self.local_mat, self.parent)

        self._optimize(i + 1)

    def _weight(self, i: int) -> float:
        """Later joints in the chain get moved more than earlier ones, so the
        base of the chain (e.g. the hip) stays closer to its original pose than
        the tip (e.g. the ankle) does."""
        return (i + 1) / self.J_N


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
