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

The wrist-relative-to-elbow math below is also used, standalone, by stage 4
*before* this reconciliation even happens: `reject_biomechanically_implausible_wrist`
catches a HaMeR wrist estimate that's anatomically impossible relative to the
forearm (see that function's docstring) so it can be treated as an occlusion
before stage 4's own smoothing chain ever sees it, rather than only being
visible once it reaches this stage's full merge.

Wrist and finger validity are deliberately kept as two SEPARATE per-frame
arrays throughout the pipeline (stage 4's `hand_pose.npz`, and this function's
own `left/right_wrist_valid` vs `left/right_finger_valid` params), not one
shared flag. A frame whose wrist estimate is biomechanically implausible
doesn't mean the finger articulation from the same HaMeR inference is
untrustworthy too -- checked on a real clip: finger joint jitter during a bad
wrist stretch was only modestly elevated (nowhere near the wrist's own
near-total breakdown), consistent with a wrist-orientation-specific failure
(monocular rotation estimation is a harder, more ambiguous read than finger
curl), not a wholesale bad-crop failure. Coupling them would throw away real,
usable finger motion for every frame the wrist gate rejects.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d

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


def body_joint_global_rotations(global_orient: torch.Tensor, body_pose: torch.Tensor, parents: list[int]) -> torch.Tensor:
    """(F, 22, 3, 3) global rotation of every SMPL-X body joint (root + 21 pose
    joints), from GVHMR's own local axis-angle body pose. Shared by this
    module's own wrist reconciliation and by stage 4's pre-smoothing plausibility
    gate -- both need the true elbow orientation to interpret HaMeR's wrist."""
    n_frames = global_orient.shape[0]
    local_aa = torch.cat([global_orient, body_pose], dim=1).reshape(n_frames, NUM_BODY_JOINTS, POSE_AXIS_DIM)
    return _global_joint_rotations(axis_angle_to_matrix(local_aa), parents)


def wrist_relative_to_elbow(elbow_global: torch.Tensor, wrist_global_aa: torch.Tensor) -> torch.Tensor:
    """HaMeR's wrist orientation (axis-angle, in the hand crop's own camera
    frame) expressed relative to the forearm (the elbow's global rotation) --
    SMPL-X's own convention for a wrist joint's local pose, and the quantity
    that has a real biomechanical ceiling. Neither input alone does: the whole
    arm can legitimately point anywhere, so only the wrist-relative-to-forearm
    angle is meaningful to check against anatomical limits (see
    `reject_biomechanically_implausible_wrist`)."""
    wrist_local = elbow_global.transpose(-1, -2) @ axis_angle_to_matrix(wrist_global_aa)
    return matrix_to_axis_angle(wrist_local)


def reject_biomechanically_implausible_wrist(
    valid: torch.Tensor,
    wrist_global_aa: torch.Tensor,
    elbow_global: torch.Tensor,
    max_deg: float,
    release_deg: float,
    window: int,
) -> torch.Tensor:
    """Demote to invalid every frame in a stretch where the wrist-relative-to-
    elbow rotation magnitude (`wrist_relative_to_elbow`) is implausibly large --
    a real human wrist cannot bend this far relative to the forearm, so this is
    HaMeR regressing a pose from an ambiguous view (e.g. a foreshortened
    forearm, or a genuine rotation-from-monocular-view ambiguity), not real
    motion. Two real clips showed two different shapes of this failure, so a
    single instantaneous threshold isn't enough on its own:

      - A slow, single-frame-smooth drift into impossible territory (a wrist
        that gradually and continuously rotates past ~150 degrees, then back).
      - A CHAOTIC stretch, jumping frame-to-frame between clearly-implausible
        values (>150 degrees) and moderate ones (80-100 degrees) that, in
        isolation, aren't distinguishable from a real deep bend elsewhere in
        the clip (a clean reference clip's own legitimate max reached ~95-103
        degrees) -- an instantaneous-only check catches the extreme frames but
        leaves these moderate "shoulder" frames of the same bad stretch
        looking individually plausible, so they still anchor the smoothing
        chain and the excursion survives as a milder-but-still-wrong bump.

    Handled with hysteresis, the same lock/release pattern as a noise gate (or,
    concretely, the same idea as this project's own root-motion-lock Blender
    addon): `max_deg` is the strict threshold that SEEDS detection (a frame
    exceeding it is unambiguously bad); from any seed frame, the invalid region
    expands outward in both time directions while a `window`-frame ROLLING MAX
    of the magnitude stays above the lower `release_deg` -- the rolling max
    (not the instantaneous value) is what keeps a single-frame dip inside an
    otherwise-bad chaotic stretch from prematurely ending the region, the same
    reason `hamer_adapter`'s confidence gate uses a rolling min rather than a
    raw per-frame check. This can only ever expand FROM a confirmed-bad seed,
    so a clip that never crosses `max_deg` (any clean, real clip observed so
    far) is completely unaffected regardless of how `release_deg` is tuned.

    Deliberately a validity gate, not a clamp: a clamped value would still look
    wrong (the wrist visibly stopping dead at a ceiling instead of continuing
    to move), whereas marking frames invalid lets the existing occlusion
    gap-fill (`motion_smoothing._fill_invalid`, run by stage 4's own smoothing
    chain right after this) bridge across the whole excursion with a plausible
    transition, the same as any other occlusion. Called from stage 4 on
    HaMeR's raw per-frame estimate, before that smoothing chain runs --
    catching a bad value before any filter blends it into its neighbors,
    rather than after, since a filter that's already blended a bad value in
    can't be un-blended downstream."""
    wrist_local_aa = wrist_relative_to_elbow(elbow_global, wrist_global_aa)
    magnitude_deg = torch.rad2deg(torch.linalg.norm(wrist_local_aa, dim=-1)).numpy()

    rolling_max_deg = maximum_filter1d(magnitude_deg, size=window, mode="nearest")
    elevated = rolling_max_deg > release_deg
    seed = magnitude_deg > max_deg

    reject = np.zeros_like(elevated)
    n, i = len(elevated), 0
    while i < n:
        if not elevated[i]:
            i += 1
            continue
        j = i
        while j < n and elevated[j]:
            j += 1
        if seed[i:j].any():
            reject[i:j] = True
        i = j

    return valid & torch.from_numpy(~reject)


def retarget_hands(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    parents: list[int],
    left_wrist_global: torch.Tensor,
    right_wrist_global: torch.Tensor,
    left_hand_pose: torch.Tensor,
    right_hand_pose: torch.Tensor,
    left_wrist_valid: torch.Tensor,
    right_wrist_valid: torch.Tensor,
    left_finger_valid: torch.Tensor,
    right_finger_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconcile both wrists and assemble the SMPL-X hand params.

    Args (all per-frame, F frames; axis-angle):
        global_orient: (F, 3) GVHMR body root orientation (camera space).
        body_pose: (F, 63) GVHMR body pose (21 joints).
        parents: length-22 SMPL-X body kinematic tree (`parents[i]` before `i`).
        left/right_wrist_global: (F, 3) HaMeR wrist global orientation, already
            gap-filled by stage 4 for any frame the hand wasn't itself detected.
        left/right_hand_pose: (F, 45) HaMeR finger articulation (15 joints), same.
        left/right_wrist_valid: (F,) bool, whether this hand's wrist has any
            usable estimate anywhere in the clip (detected AND biomechanically
            plausible) -- decides only whether to write the reconciled wrist at
            all, not which individual frames to trust (stage 4 has already
            gap-filled those).
        left/right_finger_valid: (F,) bool, same question for the finger
            articulation -- kept separate from wrist validity on purpose (see
            this module's docstring): a hand can have usable fingers even on
            frames its wrist estimate was rejected, or vice versa.

    Returns:
        (merged_body_pose (F, 63), left_hand_pose (F, 45), right_hand_pose (F, 45)).
        `merged_body_pose` is `body_pose` with a wrist slot replaced by the
        HaMeR-reconciled rotation wherever that wrist has any usable data;
        a wrist with none keeps GVHMR's own throughout. Same independently for
        fingers: a hand with no usable finger data keeps flat (zero) fingers.
    """
    global_rot = body_joint_global_rotations(global_orient, body_pose, parents)  # (F, 22, 3, 3)

    merged_body_pose = body_pose.clone()
    left_hand_out = torch.zeros_like(left_hand_pose)
    right_hand_out = torch.zeros_like(right_hand_pose)

    for wrist, elbow, wrist_global, wrist_valid in (
        (LEFT_WRIST, LEFT_ELBOW, left_wrist_global, left_wrist_valid),
        (RIGHT_WRIST, RIGHT_ELBOW, right_wrist_global, right_wrist_valid),
    ):
        if not bool(wrist_valid.any()):
            continue  # never usable anywhere in the clip -- nothing to reconcile

        # Express HaMeR's global wrist orientation relative to GVHMR's forearm.
        # Every frame is used, not just the originally-detected ones: stage 4
        # already interpolated/froze the undetected/rejected frames into a
        # usable pose.
        wrist_local_aa = wrist_relative_to_elbow(global_rot[:, elbow], wrist_global)

        start = (wrist - 1) * POSE_AXIS_DIM  # wrist joint j -> body_pose slot (j-1)
        merged_body_pose[:, start : start + POSE_AXIS_DIM] = wrist_local_aa

    for hand_pose, hand_out, finger_valid in (
        (left_hand_pose, left_hand_out, left_finger_valid),
        (right_hand_pose, right_hand_out, right_finger_valid),
    ):
        if bool(finger_valid.any()):
            hand_out[:] = hand_pose

    return merged_body_pose, left_hand_out, right_hand_out
