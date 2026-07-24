"""Graft HaMeR's per-hand MANO pose onto GVHMR's SMPL-X body by reconciling the
wrist orientation.

Two pieces of information come out of stage 4 (HaMeR) per hand: the finger
articulation (`hand_pose`, 15 joints relative to the wrist) and the wrist's
global orientation (`global_orient`, in the hand crop's camera frame). The
fingers transfer directly -- MANO's joint order matches SMPL-X's hand-joint
order, and relative rotations are frame-independent, so they drop straight into
SMPL-X's `left_hand_pose`/`right_hand_pose`.

The wrist is the part that needs work. HaMeR gives the wrist's *global*
orientation; SMPL-X wants it as a rotation *relative to the forearm* (the elbow
joint, the wrist's parent). So we forward-kinematics the GVHMR body to get the
elbow's global rotation, then express HaMeR's wrist in that frame:

    R_wrist_local = R_elbow_global^T @ R_wrist_global

and overwrite the wrist slot of the body pose with it. This is the arm-retarget
step of `open4dhoi`'s `preprocessing/scripts/make_hand_sam3d.py` (which does the
same `R_new_local = gvhmr_globals[parent].T @ R_child_target`), specialized to
HaMeR as the hand source. That reference also applies an `R_align` rotation to
bring the hand estimator's coordinate frame into GVHMR's before the change of
basis; whether HaMeR's crop-frame `global_orient` needs one is left to real-data
verification rather than assumed here -- the seam is the wrist-global term below.

Stage 4 already fills in the frames it couldn't detect (interpolating an
occlusion that recovers, freezing one that runs to either end of the clip --
see `motion_smoothing._fill_invalid`), so every frame of `left/right_wrist_global`
and `left/right_hand_pose` is a usable pose whenever the hand was detected at
least once anywhere in the clip. Only a hand that was *never once* detected in
the whole clip has nothing usable to reconcile -- that one keeps GVHMR's own
wrist and flat fingers for its entire duration, so a hand that's off-screen or
too occluded the entire time degrades gracefully instead of reconciling against
noise.
"""

from __future__ import annotations

import torch

from ..adapters.gvhmr.gvhmr_rotation_math import axis_angle_to_matrix, matrix_to_axis_angle

# SMPL-X kinematic-tree indices (full-skeleton numbering): the root plus the 21
# body-pose joints make up the first 22 joints, and the wrists are the last two
# of those, parented to the elbows.
NUM_BODY_JOINTS = 22  # root (global_orient) + 21 body-pose joints
POSE_AXIS_DIM = 3
LEFT_ELBOW, RIGHT_ELBOW = 18, 19
LEFT_WRIST, RIGHT_WRIST = 20, 21


def _global_joint_rotations(local_rotmats: torch.Tensor, parents: list[int]) -> torch.Tensor:
    """(F, J, 3, 3) per-joint local rotations -> (F, J, 3, 3) global rotations,
    composing down the kinematic tree. This is the rotation-only slice of forward
    kinematics: joint *positions* aren't needed to reconcile the wrist, only how
    each joint is oriented in camera space, so the rest-pose offsets that the full
    `gvhmr_forward_kinematics` carries are irrelevant here."""
    globals_: list[torch.Tensor | None] = [None] * len(parents)
    for joint, parent in enumerate(parents):
        if parent == -1:
            globals_[joint] = local_rotmats[:, joint]
        else:
            globals_[joint] = globals_[parent] @ local_rotmats[:, joint]
    return torch.stack(globals_, dim=1)  # type: ignore[arg-type]


def retarget_hands(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    parents: list[int],
    left_wrist_global: torch.Tensor,
    right_wrist_global: torch.Tensor,
    left_hand_pose: torch.Tensor,
    right_hand_pose: torch.Tensor,
    left_valid: torch.Tensor,
    right_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconcile both wrists and assemble the SMPL-X hand params.

    Args (all per-frame, F frames; axis-angle):
        global_orient: (F, 3) GVHMR body root orientation (camera space).
        body_pose: (F, 63) GVHMR body pose (21 joints).
        parents: length-22 SMPL-X body kinematic tree (`parents[i]` before `i`).
        left/right_wrist_global: (F, 3) HaMeR wrist global orientation, already
            gap-filled by stage 4 for any frame the hand wasn't itself detected.
        left/right_hand_pose: (F, 45) HaMeR finger articulation (15 joints), same.
        left/right_valid: (F,) bool, the *raw* per-frame HaMeR detection record
            (not gap-filled) -- used only to decide whether this hand has any
            real data at all anywhere in the clip, not to gate individual frames.

    Returns:
        (merged_body_pose (F, 63), left_hand_pose (F, 45), right_hand_pose (F, 45)).
        `merged_body_pose` is `body_pose` with the two wrist slots replaced by the
        HaMeR-reconciled rotations, for every frame of any hand detected at least
        once; a hand never detected anywhere keeps GVHMR's own wrist throughout.
    """
    n_frames = global_orient.shape[0]
    local_aa = torch.cat([global_orient, body_pose], dim=1).reshape(n_frames, NUM_BODY_JOINTS, POSE_AXIS_DIM)
    global_rot = _global_joint_rotations(axis_angle_to_matrix(local_aa), parents)  # (F, 22, 3, 3)

    merged_body_pose = body_pose.clone()
    left_hand_out = torch.zeros_like(left_hand_pose)
    right_hand_out = torch.zeros_like(right_hand_pose)

    for wrist, elbow, wrist_global, hand_pose, hand_out, valid in (
        (LEFT_WRIST, LEFT_ELBOW, left_wrist_global, left_hand_pose, left_hand_out, left_valid),
        (RIGHT_WRIST, RIGHT_ELBOW, right_wrist_global, right_hand_pose, right_hand_out, right_valid),
    ):
        if not bool(valid.any()):
            continue  # never detected anywhere in the clip -- nothing usable to reconcile

        # Express HaMeR's global wrist orientation relative to GVHMR's forearm.
        # Every frame is used, not just the originally-detected ones: stage 4
        # already interpolated/froze the undetected frames into a usable pose.
        elbow_global = global_rot[:, elbow]  # (F, 3, 3)
        wrist_local = elbow_global.transpose(-1, -2) @ axis_angle_to_matrix(wrist_global)
        wrist_local_aa = matrix_to_axis_angle(wrist_local)  # (F, 3)

        start = (wrist - 1) * POSE_AXIS_DIM  # wrist joint j -> body_pose slot (j-1)
        merged_body_pose[:, start : start + POSE_AXIS_DIM] = wrist_local_aa
        hand_out[:] = hand_pose

    return merged_body_pose, left_hand_out, right_hand_out
