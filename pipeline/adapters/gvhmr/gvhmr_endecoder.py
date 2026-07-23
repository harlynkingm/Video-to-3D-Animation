"""Turns the GVHMR transformer's raw 151-dim per-frame output into actual pose
parameters (`decode`), and those parameters into real 3D joint positions
(`fk_v2`) -- the latter is what the static-foot-lock postprocessing and CCD-IK
cleanup both need to actually see where the body is in space.

Ported from `comfyui-motioncapture/nodes/gvhmr/endecoder.py`. Has no learned
weights of its own -- `mean`/`std` are the checkpoint-independent
`MM_V1_AMASS_LOCAL_BEDLAM_CAM` normalization table (confirmed by checking the
real `inference_node.py` call site, not the class's own misleading
`stats_name="DEFAULT_01"` default, which turns out to be an unused no-op),
extracted once from the real reference module into `gvhmr_endecoder_stats.json`
rather than hand-transcribed (that table is ~300 floats composed from five
different named entries in the source's data file -- retyping it by hand would
be a real transcription-error risk with no way to catch a wrong digit later).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .gvhmr_forward_kinematics import forward_kinematics, get_position, get_TRS
from .gvhmr_rotation_math import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix
from .gvhmr_smplx_skeleton import NUM_BODY_JOINTS, SmplxSkeleton

_STATS_PATH = Path(__file__).parent / "gvhmr_endecoder_stats.json"


class EnDecoder:
    """Not an `nn.Module` -- no learned parameters, just fixed normalization
    stats and a `SmplxSkeleton` instance, matching the source's own treatment
    of the equivalent object (its `nn.Module` base there exists only so
    `mean`/`std` ride along with `.to(device)`, which this port does explicitly
    instead via `device`)."""

    def __init__(self, device: torch.device | None = None):
        stats = json.loads(_STATS_PATH.read_text())
        self.mean = torch.tensor(stats["mean"], device=device)
        self.std = torch.tensor(stats["std"], device=device)
        self.skeleton = SmplxSkeleton(device=device)
        self.parents = self.skeleton.parents
        self.parents_tensor = torch.tensor(self.parents, device=device)

    def decode(self, x_norm: torch.Tensor) -> dict:
        """x_norm: (B, L, 151) network output. Returns axis-angle body_pose (B,L,63),
        betas (B,L,10), global_orient in camera space and in the network's
        internal "gravity view" frame (B,L,3 each), and local_transl_vel (B,L,3)."""
        B, L, _ = x_norm.shape
        x = x_norm * self.std + self.mean

        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]
        global_orient_gv_r6d = x[:, :, 142:148]
        local_transl_vel = x[:, :, 148:151]

        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6)))
        body_pose = body_pose.flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))

        return {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
        }

    def fk_v2(
        self, body_pose: torch.Tensor, betas: torch.Tensor,
        global_orient: torch.Tensor | None = None, transl: torch.Tensor | None = None,
        get_intermediate: bool = False,
    ):
        """Forward kinematics: axis-angle joint rotations + body shape -> actual
        3D joint positions.

        Args:
            body_pose: (B, L, 63) axis-angle, 21 body joints.
            betas: (B, L, 10).
            global_orient: (B, L, 3) axis-angle root orientation; zero if omitted.
            transl: (B, L, 3) root translation, added directly to the root joint.
            get_intermediate: if True, also return the per-joint local and world transforms.
        Returns:
            (B, L, 22, 3) joint positions (or a (joints, local_mat, world_mat) tuple).
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device, dtype=body_pose.dtype)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)

        skeleton = self.skeleton.get_skeleton(betas)[..., :NUM_BODY_JOINTS, :]
        # Rest-pose position -> parent-relative offset (except the root, which has
        # no parent, so it keeps its own rest-pose position as its "local" transform).
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        if transl is not None:
            local_skeleton = local_skeleton.clone()
            local_skeleton[..., 0, :] += transl

        local_mat = get_TRS(rotmat, local_skeleton)
        world_mat = forward_kinematics(local_mat, self.parents)
        joints = get_position(world_mat)

        if not get_intermediate:
            return joints
        return joints, local_mat, world_mat
