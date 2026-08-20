"""Graft HaMeR's per-hand MANO pose onto GVHMR's SMPL-X body by reconciling the
wrist orientation.

Finger articulation (`hand_pose`, 15 joints relative to the wrist) transfers
directly, MANO's joint order matches SMPL-X's, and relative rotations are
frame-independent. The wrist needs work: HaMeR gives its *global* orientation,
SMPL-X wants it relative to the forearm (the elbow, its parent):

    R_wrist_local = R_elbow_global^T @ R_wrist_global

-- the arm-retarget step of `open4dhoi`'s `make_hand_sam3d.py`, specialized to
HaMeR. That reference also applies an `R_align` correction for its own hand
estimator's frame; verified unneeded here (see `wrist_relative_to_elbow`).

Stage 4 gap-fills every detected-at-least-once hand (interior occlusions
interpolated, edge ones frozen, `motion_smoothing.fill_invalid`), so every
frame is usable except a hand never once detected, which keeps GVHMR's own
wrist and flat fingers for its whole duration.

Four independent, complementary checks gate wrist validity before any of this
runs (each function's own docstring has the specifics): `reject_biomechanically
_implausible_wrist` (static magnitude vs. the forearm), `reject_wrist_velocity
_spikes` (frame-to-frame rotation speed), `reject_hand_swung_past_forearm`
(pointing-direction deviation from rest, excluding twist), and `reject_hand_
through_forearm` (real 3D geometry, does the hand's position overlap the
forearm segment). Each catches a failure shape the others structurally can't.

Wrist and finger validity are kept as separate per-frame arrays throughout the
pipeline, not one shared flag, a biomechanically bad wrist doesn't mean the
same frame's fingers are untrustworthy too (confirmed on real data: finger
jitter during a bad wrist stretch stays only modestly elevated). Coupling them
would throw away real, usable finger motion for every wrist-gate rejection.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d

from ..adapters.gvhmr.gvhmr_rotation_math import axis_angle_to_matrix, matrix_to_axis_angle
from ..helpers.smplx_bvh_preview import smplx_rest_joints_and_parents

# SMPL-X kinematic-tree indices (full-skeleton numbering): the root plus the 21
# body-pose joints make up the first 22 joints, and the wrists are the last two
# of those, parented to the elbows.
NUM_BODY_JOINTS = 22  # root (global_orient) + 21 body-pose joints
POSE_AXIS_DIM = 3
LEFT_ELBOW, RIGHT_ELBOW = 18, 19
LEFT_WRIST, RIGHT_WRIST = 20, 21
# Middle finger's first knuckle, used only to define "which way the hand
# points" for `rest_hand_direction`/`reject_hand_swung_past_forearm` below,
# any finger's own first knuckle would do equally well, middle was picked
# arbitrarily for being roughly central.
LEFT_MIDDLE1, RIGHT_MIDDLE1 = 28, 43
# Index/pinky first knuckles, used only by `palm_normal_direction` below,
# these two specifically (not e.g. middle/ring) because they're the widest
# lateral spread across the palm, giving the most numerically stable normal.
LEFT_INDEX1, RIGHT_INDEX1 = 25, 40
LEFT_PINKY1, RIGHT_PINKY1 = 31, 46


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
    gate, both need the true elbow orientation to interpret HaMeR's wrist."""
    n_frames = global_orient.shape[0]
    local_aa = torch.cat([global_orient, body_pose], dim=1).reshape(n_frames, NUM_BODY_JOINTS, POSE_AXIS_DIM)
    return _global_joint_rotations(axis_angle_to_matrix(local_aa), parents)


def _wrist_relative_to_elbow_matrix(elbow_global: torch.Tensor, wrist_global_aa: torch.Tensor) -> torch.Tensor:
    """The rotation-matrix form of `wrist_relative_to_elbow`, shared by that
    function (which converts to axis-angle for a magnitude check) and
    `reject_hand_swung_past_forearm` (which needs the matrix directly, to
    rotate the rest-pose hand direction)."""
    return elbow_global.transpose(-1, -2) @ axis_angle_to_matrix(wrist_global_aa)


def wrist_relative_to_elbow(elbow_global: torch.Tensor, wrist_global_aa: torch.Tensor) -> torch.Tensor:
    """HaMeR's wrist orientation (axis-angle, in the hand crop's own camera
    frame) expressed relative to the forearm (the elbow's global rotation),
    SMPL-X's own convention for a wrist joint's local pose, and the quantity
    that has a real biomechanical ceiling. Neither input alone does: the whole
    arm can legitimately point anywhere, so only the wrist-relative-to-forearm
    angle is meaningful to check against anatomical limits (see
    `reject_biomechanically_implausible_wrist`)."""
    return matrix_to_axis_angle(_wrist_relative_to_elbow_matrix(elbow_global, wrist_global_aa))


def wrist_global_from_relative(elbow_global: torch.Tensor, wrist_local_aa: torch.Tensor) -> torch.Tensor:
    """Inverse of `wrist_relative_to_elbow`: takes the wrist's rotation
    relative to the forearm back to the crop-camera-frame global orientation
    the rest of the pipeline stores. Lets stage 4 do its gap-filling and
    smoothing in the forearm-relative frame (where "hold this pose" means
    what it anatomically should) and convert back afterward."""
    return matrix_to_axis_angle(elbow_global @ axis_angle_to_matrix(wrist_local_aa))


def reject_biomechanically_implausible_wrist(
    valid: torch.Tensor,
    wrist_global_aa: torch.Tensor,
    elbow_global: torch.Tensor,
    max_deg: float,
    release_deg: float,
    window: int,
    max_expansion_frames: int,
) -> torch.Tensor:
    """Demote to invalid every frame near a seed where the wrist-relative-
    to-elbow magnitude (`wrist_relative_to_elbow`) is implausibly large, a
    real wrist can't bend this far (clean-clip legitimate max ~95-103 deg),
    so this is HaMeR misreading an ambiguous view, not real motion.

    Hysteresis, not a single threshold (same lock/release pattern as a noise
    gate): `max_deg` strictly SEEDS detection; from each seed, the invalid
    region expands up to `max_expansion_frames` either direction while a
    `window`-frame rolling max stays above the lower `release_deg` (a rolling
    max, not the instantaneous value, so a single-frame dip inside a bad
    chaotic stretch doesn't end the region early). Never expands from
    anything but a confirmed-bad seed. `max_expansion_frames` caps it because
    a long stretch of real large motion (e.g. gripping a strap) can sit above
    `release_deg` for many seconds without itself seeding.

    A validity gate, not a clamp, marking frames invalid lets the existing
    gap-fill (`motion_smoothing.fill_invalid`) bridge across with a plausible
    transition, the same as any other occlusion, instead of visibly stopping
    dead at a ceiling. Runs on HaMeR's raw estimate before any smoothing,
    a filter that's already blended a bad value in can't be un-blended after.
    """
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
        for s in np.flatnonzero(seed[i:j]) + i:
            lo = max(i, s - max_expansion_frames)
            hi = min(j, s + max_expansion_frames + 1)
            reject[lo:hi] = True
        i = j

    return valid & torch.from_numpy(~reject)


def reject_wrist_velocity_spikes(
    valid: torch.Tensor,
    wrist_global_aa: torch.Tensor,
    fps: float,
    max_velocity_deg_per_sec: float,
) -> torch.Tensor:
    """Demote to invalid any frame whose wrist orientation changed implausibly
    fast from its neighbor, a real wrist can't rotate a large fraction of a
    full turn in one frame, so a huge frame-to-frame delta is HaMeR flipping
    to a wrong orientation for an isolated frame or two, not real motion.
    Catches what `reject_biomechanically_implausible_wrist` structurally
    can't: a flip can pass entirely through a "plausible" magnitude on its way
    up and back down. Different question from the velocity check already tried and
    rejected for `hamer_adapter`'s confidence gate, that needed to separate
    *sustained* noise from *sustained* real fast motion and couldn't; this is
    an isolated single-frame reversal, a different signature entirely.

    Both endpoints of an excessive transition are marked (a flip looks
    anomalous relative to both neighbors, so marking only one leaves the
    other spike unexplained). `max_velocity_deg_per_sec` is degrees/second,
    not degrees/frame, so the same value means the same physical speed
    regardless of fps. `valid` only narrows (never resurrects); a velocity
    is only meaningful between two already-valid frames, skipped across gaps.
    """
    wrist_mats = axis_angle_to_matrix(wrist_global_aa)  # (T, 3, 3)
    relative = wrist_mats[1:] @ wrist_mats[:-1].transpose(-1, -2)  # (T-1, 3, 3)
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    delta_deg = torch.rad2deg(torch.arccos(torch.clamp((trace - 1) / 2, -1.0, 1.0)))  # (T-1,)
    velocity_deg_per_sec = delta_deg * fps

    both_valid = valid[1:] & valid[:-1]
    spike = both_valid & (velocity_deg_per_sec > max_velocity_deg_per_sec)

    reject = torch.zeros_like(valid)
    reject[:-1] |= spike
    reject[1:] |= spike
    return valid & ~reject


def rest_hand_direction(wrist_joint: int, middle1_joint: int) -> np.ndarray:
    """Unit vector from the wrist to the middle finger's first knuckle, in
    SMPL-X's own neutral rest pose, the hand's own "pointing straight out
    from the forearm" direction, the reference `reject_hand_swung_past_
    forearm` measures every frame's actual hand direction against. Pass
    `(LEFT_WRIST, LEFT_MIDDLE1)`/`(RIGHT_WRIST, RIGHT_MIDDLE1)`."""
    rest_joints, _ = smplx_rest_joints_and_parents()
    v = rest_joints[middle1_joint] - rest_joints[wrist_joint]
    return (v / np.linalg.norm(v)).astype(np.float32)


def palm_normal_direction(wrist_joint: int, index1_joint: int, pinky1_joint: int, mirror: bool) -> np.ndarray:
    """Unit vector perpendicular to the palm (the plane spanned by the
    wrist->index-knuckle and wrist->pinky-knuckle rest-pose vectors),
    roughly the direction a held object nestles into when the fingers curl
    around it, unlike `rest_hand_direction` (which points along the fingers
    when *extended*, not the grip direction of a curled hand). Derived
    purely from SMPL-X's own rest-pose hand geometry, not fit to any
    particular clip; cross-checked against real reference points across
    several grip events and found ~59 degrees closer to the real grip
    direction on average than `rest_hand_direction` was (see `hoi_object_
    pose.py`'s `GRIP_NORMAL_SCALE` comment for the offset this pairs with).

    `mirror` corrects for a real chirality flip, confirmed on that same
    data: a cross product's sign encodes handedness, so the identical
    `index1_joint`/`pinky1_joint` order that reads correctly on one hand
    reads ~180 degrees backwards on its mirror-image counterpart. Pass
    `mirror=False` for the hand the un-mirrored convention (`cross(pinky,
    index)`) was validated against, `mirror=True` for its mirror image,
    concretely, `(LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, mirror=False)` /
    `(RIGHT_WRIST, RIGHT_INDEX1, RIGHT_PINKY1, mirror=True)`.
    """
    rest_joints, _ = smplx_rest_joints_and_parents()
    wrist = rest_joints[wrist_joint]
    v_index = rest_joints[index1_joint] - wrist
    v_pinky = rest_joints[pinky1_joint] - wrist
    normal = np.cross(v_index, v_pinky) if mirror else np.cross(v_pinky, v_index)
    return (normal / np.linalg.norm(normal)).astype(np.float32)


def palm_distal_direction(wrist_joint: int, index1_joint: int, pinky1_joint: int, middle1_joint: int) -> np.ndarray:
    """Unit vector from the wrist toward the middle knuckle (see `rest_hand_
    direction`), re-orthogonalized against the palm normal (the plane spanned
    by wrist->index-knuckle/wrist->pinky-knuckle) so the two form a clean
    basis for a held object's offset (see `hoi_object_pose.py`, which
    combines a distance along each). Without this correction the raw
    wrist->middle-knuckle vector isn't exactly perpendicular to the palm
    normal (~87 degrees, not 90, in SMPL-X's own rest geometry), which would
    leak a small stray palm-normal component into what's meant to be a purely
    distal (along-the-hand) term.

    No `mirror` parameter, unlike `palm_normal_direction`, this function's
    own orthogonal projection is provably invariant to the palm normal's
    sign (`v - (v.n)n == v - (v.(-n))(-n)` for any unit `n`), so the
    chirality correction that matters for the normal's own *direction*
    doesn't matter here at all.
    """
    rest_joints, _ = smplx_rest_joints_and_parents()
    wrist = rest_joints[wrist_joint]
    v_index = rest_joints[index1_joint] - wrist
    v_pinky = rest_joints[pinky1_joint] - wrist
    normal = np.cross(v_pinky, v_index)
    normal = normal / np.linalg.norm(normal)
    distal_raw = rest_joints[middle1_joint] - wrist
    orthogonal = distal_raw - np.dot(distal_raw, normal) * normal
    return (orthogonal / np.linalg.norm(orthogonal)).astype(np.float32)


def reject_hand_swung_past_forearm(
    valid: torch.Tensor,
    wrist_global_aa: torch.Tensor,
    elbow_global: torch.Tensor,
    rest_direction: np.ndarray,
    max_swing_deg: float,
) -> torch.Tensor:
    """Demote to invalid any frame where the hand's own pointing direction
    (wrist-to-middle-finger) has swung more than `max_swing_deg` from its
    rest-pose direction, a real wrist can't swing much past ~90 deg without
    the hand folding back over the forearm.

    Measures SWING only, deliberately ignoring TWIST (rotation about that
    direction, e.g. forearm pronation while the hand keeps pointing the same
    way): rejecting on twist directly doesn't work, since composing two
    legitimate swing rotations (pure flex + pure deviate) produces large
    *apparent* twist purely from how compound 3D rotations compose, not from
    real axial rotation.
    Swing has none of that artifact, it's just the angle between
    two directions, not a decomposition of one rotation into two parts.

    Single instantaneous threshold, not hysteresis like the other wrist gates,
    swing isn't contaminated by twist the way raw magnitude is, so it
    doesn't have the "plausible shoulder frames of a chaotic excursion"
    problem hysteresis exists for. `rest_direction`: (3,) unit vector from
    `rest_hand_direction`, fixed per side.
    """
    wrist_local_mat = _wrist_relative_to_elbow_matrix(elbow_global, wrist_global_aa)
    rest = torch.as_tensor(rest_direction, dtype=wrist_local_mat.dtype, device=wrist_local_mat.device)
    current_direction = torch.einsum("...ij,j->...i", wrist_local_mat, rest)
    cos_swing = torch.clamp((current_direction * rest).sum(-1), -1.0, 1.0)
    swing_deg = torch.rad2deg(torch.arccos(cos_swing))
    return valid & (swing_deg <= max_swing_deg)


def rest_forearm_and_hand_offsets(
    elbow_joint: int, wrist_joint: int, middle1_joint: int
) -> tuple[np.ndarray, np.ndarray]:
    """(elbow->wrist rest offset, wrist->middle1 rest offset), both real
    metric (meters) vectors from SMPL-X's own neutral mesh, the fixed local
    geometry `reject_hand_through_forearm` needs (forearm ~25cm, hand ~11cm
    on the neutral mesh; a real person's own proportions vary, but this is
    the same neutral-mesh approximation `smplx_bvh_preview.py` already uses
    for skeleton geometry without needing per-person betas)."""
    rest_joints, _ = smplx_rest_joints_and_parents()
    forearm = rest_joints[wrist_joint] - rest_joints[elbow_joint]
    hand = rest_joints[middle1_joint] - rest_joints[wrist_joint]
    return forearm.astype(np.float32), hand.astype(np.float32)


def reject_hand_through_forearm(
    valid: torch.Tensor,
    wrist_global_aa: torch.Tensor,
    elbow_global: torch.Tensor,
    forearm_offset: np.ndarray,
    hand_offset: np.ndarray,
    forearm_radius_m: float,
    forearm_interior_max_t: float,
) -> torch.Tensor:
    """Demote to invalid any frame where the hand has folded back far enough
    to sit inside the forearm's own segment, a direct geometric check, not
    another rotation-angle heuristic, for what "the hand goes through the
    arm" describes.

    Works in the elbow's own local frame (same convention as `wrist_relative_
    to_elbow`): the forearm bone is rigid, so the elbow sits at the origin
    and the wrist always sits at the fixed rest offset `forearm_offset`; only
    the hand's position (`forearm_offset + wrist_local_rotation @ hand_
    offset`) moves per frame. Checked against the fixed [elbow, wrist]
    segment: `t` is where the closest point falls along it (0=elbow, 1=wrist,
    unclamped so below-1 is meaningful), `forearm_interior_max_t` is how far
    below the wrist end still counts as "inside" rather than merely "near the
    wrist" (fingers curled toward the palm sit close to the wrist in plenty
    of normal poses), and `forearm_radius_m` requires genuine proximity to
    the forearm's own axis so a hand held out to the side isn't penalized.

    Calibrated on a real clip: a confirmed hand-through-forearm stretch
    dropped `t` to 0.67-0.99 at 7-11cm from the axis; a confirmed real
    (if extreme-looking) grip and 1900+ other frames never dropped below
    ~1.1, wide margin either side. `forearm_offset`/`hand_offset`: (3,)
    rest-pose vectors from `rest_forearm_and_hand_offsets`, fixed per side.
    """
    wrist_local_mat = _wrist_relative_to_elbow_matrix(elbow_global, wrist_global_aa)
    forearm = torch.as_tensor(forearm_offset, dtype=wrist_local_mat.dtype, device=wrist_local_mat.device)
    hand_vec = torch.as_tensor(hand_offset, dtype=wrist_local_mat.dtype, device=wrist_local_mat.device)

    hand_point = forearm + torch.einsum("...ij,j->...i", wrist_local_mat, hand_vec)

    seg_len_sq = (forearm * forearm).sum()
    t = (hand_point * forearm).sum(-1) / seg_len_sq
    closest = torch.clamp(t, 0.0, 1.0).unsqueeze(-1) * forearm
    distance = torch.linalg.norm(hand_point - closest, dim=-1)

    penetrating = (t < forearm_interior_max_t) & (distance < forearm_radius_m)
    return valid & ~penetrating


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
            plausible), decides only whether to write the reconciled wrist at
            all, not which individual frames to trust (stage 4 has already
            gap-filled those).
        left/right_finger_valid: (F,) bool, same question for the finger
            articulation, kept separate from wrist validity on purpose (see
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
            continue  # never usable anywhere in the clip, nothing to reconcile

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
