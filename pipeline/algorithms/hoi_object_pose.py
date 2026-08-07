"""Per-frame object 6DoF pose for stage 8 (`optimize_hoi`): given a body's
already-computed motion and a set of qualifying contact events, produces the
tracked object's own pose for every frame of the clip.

Two states, not the two-regime design this module started with: **attached**
(rigid, follows a body joint) during a genuine sustained hold, and **held**
(perfectly frozen, wherever the object last was) everywhere else -- before the
first hold, between two holds, and after the last one. There is no more
independent per-frame depth-tracking of the object while it isn't being held.
That was tried first and abandoned after reviewing a real export: even after
smoothing, a real object sitting perfectly still on a real table showed
chaotic, unusable frame-to-frame X/Z jitter. Monocular per-frame depth is
just not reliable enough, for a small object, to be worth independently
tracking at all -- so this module no longer tries to.

This module is pure algorithm -- no bpy, no DA3, no mask/frame I/O. The one
callback callers must supply (`object_position_fn`) hides that I/O behind a
plain `frame -> value` interface, so every function here is testable against
synthetic data without a GPU or checkpoints.
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..adapters.gvhmr.gvhmr_forward_kinematics import forward_kinematics
from .contact_detection import attachment_joint_index
from .hand_retarget import (
    LEFT_INDEX1,
    LEFT_MIDDLE1,
    LEFT_PINKY1,
    LEFT_WRIST,
    RIGHT_INDEX1,
    RIGHT_MIDDLE1,
    RIGHT_PINKY1,
    RIGHT_WRIST,
    palm_distal_direction,
    palm_normal_direction,
)

# SMPL-X pelvis + 21 body joints -- SmplxSkeleton's own scope (see that
# module's docstring); every real REGION_JOINTS attachment joint (wrist,
# ankle, head, spine3) falls within this range, so no hand/finger extension
# is needed. "head_top" is not itself in range (a synthetic mesh-vertex
# index, not a skeletal joint) -- attachment_joint_index redirects it to the
# real HEAD_JOINT instead, see that function's own comment.
NUM_BODY_JOINTS = 22

# How many frames to search outward from an event's own start_frame (both
# directions) for a usable object measurement, before giving up and treating
# the whole hold as low-confidence (see `_find_snap_measurement`). Small and
# deliberately bounded to stay near the hold's own start -- unlike this
# module's old free-tracking design, only a handful of DA3 calls happen per
# *event* now, not one per frame of the whole clip, so there's no cost
# pressure to search wide; the pressure runs the other way; searching too far
# into an already-underway hold risks the exact occlusion bias the reference-
# frame redesign above already fixed once. Uncalibrated -- a reasonable
# starting point.
MAX_SNAP_SEARCH_FRAMES = 10

# How far a held object's center sits from the attaching joint, split into
# two anatomical terms: a palm-normal component (perpendicular to the palm,
# scaled by `GRIP_NORMAL_SCALE * object_radius`) and a distal component
# (along the hand, from wrist toward the fingers, scaled by `GRIP_DISTAL_
# SCALE * palm_length`, where `palm_length` is the person's own real
# wrist-to-middle-knuckle rest distance from their SMPL-X betas). Both
# terms are needed because the palm normal is ~87 degrees from the distal
# axis by construction (see `hand_retarget.palm_distal_direction`), so it
# alone can only push an object away from the palm surface, not toward the
# fingers.
#
# Both scale constants are `real_offset_component / (radius or palm_
# length)`, averaged across 24 manually-placed reference points spanning 3
# real clips (a cradled grip, a handle grip, and a utensil grip), and
# cross-validated by fitting on 2 clips and testing on the third held out:
# 13.6/8.0/5.4 degrees mean error respectively.
GRIP_NORMAL_SCALE = 1.055
GRIP_DISTAL_SCALE = 0.861


def _ensure_proper_rotation(rotation: np.ndarray) -> np.ndarray:
    """Flip the last axis's sign if needed so det=+1. PCA's own SVD-derived
    rotation can come out as a reflection (fine for a shape-fitter using it
    only as a local coordinate basis, since a box/ellipsoid/cylinder's own
    dimensions don't care about handedness) -- but the signed-permutation
    search below assumes a proper rotation group, so it's corrected here,
    local to this module, rather than changing `object_extent_fit`'s shared
    fitting code for a need only this module has."""
    if np.linalg.det(rotation) < 0:
        rotation = rotation.copy()
        rotation[:, -1] *= -1
    return rotation


def _signed_permutation_matrices() -> np.ndarray:
    """The 24 proper (det=+1) signed 3x3 permutation matrices -- the cube's
    own rotation symmetry group. Composing one onto a rotation matrix
    relabels its columns (which principal axis is "0"/"1"/"2") and/or flips
    their signs, without changing which physical directions they span."""
    matrices = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if np.linalg.det(m) > 0:
                matrices.append(m)
    return np.stack(matrices)


_SIGNED_PERMUTATIONS = _signed_permutation_matrices()  # (24, 3, 3), built once at import


def disambiguate_rotation(fresh_rotation: np.ndarray, reference_rotation: np.ndarray) -> np.ndarray:
    """Among the 24 signed-axis-relabelings of `fresh_rotation` (a PCA fit's
    own axis order/sign has no canonical convention -- SVD ties broken by
    float noise), pick whichever is closest, by rotation angle, to
    `reference_rotation` -- the most recent known-good object orientation.

    A genuine large real rotation (an object flipped over) is not erased by
    this: every candidate *is* `fresh_rotation`, just relabeled/resigned, so
    the true newly-measured orientation survives regardless of which
    relabeling wins -- only the axis bookkeeping is corrected, never the
    measured rotation itself.
    """
    fresh_rotation = _ensure_proper_rotation(fresh_rotation)
    candidates = fresh_rotation @ _SIGNED_PERMUTATIONS  # (24, 3, 3)
    # trace(reference.T @ candidate) = sum(reference * candidate) element-wise
    # = 1 + 2*cos(angle) -- maximizing it minimizes the rotation angle between them.
    traces = np.einsum("ij,kij->k", reference_rotation, candidates)
    return candidates[int(np.argmax(traces))]


# "Up" in this module's own incam/body space (GVHMR's raw camera-space,
# X-right/Y-down/Z-forward -- see stage_10_export.py's own module docstring):
# the same -Y_cam convention already established for the root's own
# camera-to-upright correction (`CAMERA_TO_BVH_ROOT_ROTATION`'s own comment),
# reused here rather than re-derived, so an object snapped "upright" by this
# module ends up actually upright once stage 10 applies that same conversion
# to the whole scene.
_INCAM_UP = np.array([0.0, -1.0, 0.0])


def _snap_axis_to_up(rotation: np.ndarray, up: np.ndarray = _INCAM_UP) -> np.ndarray:
    """Rotates `rotation` (3x3, object-local axes expressed in this module's
    own incam/body space) by the minimal amount needed to point whichever of
    its own three local axes (either sign -- a fit's own axis directions have
    no canonical sign, see `disambiguate_rotation`) already comes closest to
    `up` exactly along `up`, leaving the other two axes wherever that
    rotation leaves them.

    Used to give a held object a plausible-looking rest orientation instead
    of PCA's own raw fitted one: a real object generally settles resting on
    some flat side/base when not being held, and "whichever of its own axes
    is already closest to vertical, snapped exactly vertical" approximates
    that without needing to know in advance which axis a given shape kind
    calls "up" (`object_extent_fit` doesn't label one). For a rotationally
    symmetric shape (a cylinder's radius has no preferred direction), the
    rotation *around* the now-vertical axis is left exactly as PCA happened
    to produce it -- there is no real signal in a partial point cloud to
    recover that from, so an arbitrary-but-stable choice is the best
    available rather than a defect to fix further.
    """
    up = up / np.linalg.norm(up)
    candidate_axes = np.concatenate([rotation.T, -rotation.T])  # 6: +/- each local axis
    best_axis = candidate_axes[int(np.argmax(candidate_axes @ up))]
    align, _ = Rotation.align_vectors([up], [best_axis])
    return align.as_matrix() @ rotation


def _joint_world_transforms(body_motion: dict, skeleton) -> np.ndarray:
    """(F, 22, 4, 4) world transform of every body joint (pelvis + 21 body
    joints, ending at both wrists), every frame -- via `SmplxSkeleton`'s
    rest-pose offsets (betas-dependent, pooled from frame 0 like every other
    consumer of this motion) composed down the kinematic chain with
    `gvhmr_forward_kinematics.forward_kinematics`, using each frame's own
    `global_orient`/`body_pose` axis-angle values for the per-joint rotation
    and `transl` for the root's own per-frame world position (SMPL-X's own
    convention, confirmed empirically during stage 10's own work: the root's
    world position is `transl + pelvis_rest`, unaffected by `global_orient`
    itself -- rotation only matters for joints further down the chain).
    """
    global_orient = np.asarray(body_motion["global_orient"])  # (F, 3)
    body_pose = np.asarray(body_motion["body_pose"]).reshape(-1, 21, 3)  # (F, 21, 3)
    betas = np.asarray(body_motion["betas"])[0]  # pooled -- identical every frame
    transl = np.asarray(body_motion["transl"])  # (F, 3)
    n_frames = global_orient.shape[0]

    all_axis_angle = np.concatenate([global_orient[:, None, :], body_pose], axis=1)  # (F, 22, 3)
    rotations = Rotation.from_rotvec(all_axis_angle.reshape(-1, 3)).as_matrix().reshape(n_frames, NUM_BODY_JOINTS, 3, 3)

    rest_positions = skeleton.get_skeleton(torch.from_numpy(betas).float()).numpy()[:NUM_BODY_JOINTS]  # (22, 3)
    offsets = np.zeros((NUM_BODY_JOINTS, 3))
    offsets[0] = rest_positions[0]  # root: offset from the world origin, not a parent
    for j in range(1, NUM_BODY_JOINTS):
        offsets[j] = rest_positions[j] - rest_positions[skeleton.parents[j]]

    local = np.zeros((n_frames, NUM_BODY_JOINTS, 4, 4))
    local[:, :, 3, 3] = 1.0
    local[:, :, :3, :3] = rotations
    local[:, :, :3, 3] = offsets[None, :, :]
    local[:, 0, :3, 3] += transl  # root only -- every other joint's offset is parent-relative, not world

    world = forward_kinematics(torch.from_numpy(local).float(), skeleton.parents)
    return world.numpy()


def _to_transform(center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    return transform


def _find_snap_measurement(
    start_frame: int,
    end_frame: int,
    object_position_fn: Callable[[int], tuple[np.ndarray, np.ndarray] | None],
) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Searches for a usable `object_position_fn` measurement near an
    attachment event's own `start_frame` -- tried first, since it's the
    frame the object is least occluded by the gripping body part (occlusion
    only grows once a hold is underway), then alternating outward (+1, -1,
    +2, -2, ...) up to `MAX_SNAP_SEARCH_FRAMES`, never past `end_frame` and
    never before `start_frame - MAX_SNAP_SEARCH_FRAMES`. Returns
    `(frame, center, rotation) | None`.
    """
    lo_bound = max(0, start_frame - MAX_SNAP_SEARCH_FRAMES)
    hi_bound = min(end_frame, start_frame + MAX_SNAP_SEARCH_FRAMES)
    for delta in range(0, MAX_SNAP_SEARCH_FRAMES + 1):
        candidates = {start_frame + delta, start_frame - delta} if delta else {start_frame}
        for frame in sorted(candidates):
            if frame < lo_bound or frame > hi_bound:
                continue
            fitted = object_position_fn(frame)
            if fitted is not None:
                center, rotation = fitted
                return frame, center, rotation
    return None


def compute_object_pose_sequence(
    n_frames: int,
    attachment_events: list[dict],
    body_motion: dict,
    initial_center: np.ndarray,
    initial_rotation: np.ndarray,
    object_position_fn: Callable[[int], tuple[np.ndarray, np.ndarray] | None],
    skeleton=None,
    object_radius: float = 0.0,
) -> dict:
    """The full per-frame object pose for a clip.

    For each attachment event: position contact-point-anchors to the
    attaching joint's own world position at a snap frame near the event's
    `start_frame` (`_find_snap_measurement` picks the frame; only its
    rotation is used), axis-disambiguated against the last known-good
    orientation and snapped upright (`_snap_axis_to_up`). For a `left_hand`/
    `right_hand` event with `object_radius > 0`, the joint-coincident
    position is pushed outward by a two-term anatomical offset -- see
    `GRIP_NORMAL_SCALE`'s own comment. Other regions get no offset.
    `object_radius=0` (the default) reproduces plain joint-coincident
    behavior.

    The reference pose fixes a rigid joint-relative offset for the event's
    whole span, so rotation evolves with the joint while translation stays
    fixed relative to it. After an event ends, its final pose freezes until
    the next event starts; before the first event, the object holds that
    event's own reference pose. `initial_center`/`initial_rotation` are the
    last-resort fallback for a clip with no events.

    Args: object_position_fn is `frame -> (center, rotation) | None` -- its
    center is unused here (position comes from the joint); object_radius is
    the object's own fitted size, or 0 to disable the hand-relative offset.

    Returns `{"translation", "rotation", "is_low_confidence", "resolved_
    events"}` -- `resolved_events` carries each event's own raw reference
    measurement, for a caller that needs the object to stay attached
    through a later retarget.
    """
    if skeleton is None:
        from ..adapters.gvhmr.gvhmr_smplx_skeleton import SmplxSkeleton
        skeleton = SmplxSkeleton()

    joint_world = _joint_world_transforms(body_motion, skeleton)  # (F, 22, 4, 4)

    initial_center = np.asarray(initial_center, dtype=float)
    initial_rotation = _snap_axis_to_up(np.asarray(initial_rotation, dtype=float))

    events_sorted = sorted(attachment_events, key=lambda e: e["start_frame"])

    translation = np.tile(initial_center, (n_frames, 1))
    rotation = np.tile(initial_rotation, (n_frames, 1, 1))
    is_low_confidence = np.ones(n_frames, dtype=bool)

    hold_translation = initial_center
    hold_rotation = initial_rotation
    last_rotation_reference = initial_rotation
    resolved_events: list[dict] = []

    # Only loaded when actually needed -- palm_normal_direction/palm_distal_
    # direction read the SMPL-X model file, no reason to require it when
    # object_radius=0. palm_length uses this body's own real betas (pooled,
    # identical every frame -- same convention _joint_world_transforms uses),
    # not the generic neutral template, since it's a body dimension, not a
    # direction. Precomputed once per hand here, not per event: radius,
    # betas, and rest geometry are all constant across the whole clip.
    hand_grip_offset: dict[str, np.ndarray] = {}
    if object_radius > 0:
        betas = np.asarray(body_motion["betas"])[0]
        person_rest_joints = skeleton.get_skeleton(torch.from_numpy(betas).float()).numpy()
        for region, wrist, index1, pinky1, middle1, mirror in (
            ("left_hand", LEFT_WRIST, LEFT_INDEX1, LEFT_PINKY1, LEFT_MIDDLE1, False),
            ("right_hand", RIGHT_WRIST, RIGHT_INDEX1, RIGHT_PINKY1, RIGHT_MIDDLE1, True),
        ):
            palm_length = float(np.linalg.norm(person_rest_joints[middle1] - person_rest_joints[wrist]))
            normal = palm_normal_direction(wrist, index1, pinky1, mirror=mirror)
            distal = palm_distal_direction(wrist, index1, pinky1, middle1)
            hand_grip_offset[region] = object_radius * GRIP_NORMAL_SCALE * normal + palm_length * GRIP_DISTAL_SCALE * distal

    for i, event in enumerate(events_sorted):
        start, end = event["start_frame"], event["end_frame"]
        joint_idx = attachment_joint_index(event["region"])

        found = _find_snap_measurement(start, end, object_position_fn)
        if found is None:
            # No fresh orientation measurement nearby -- hold whatever
            # rotation is already active. Position doesn't need this
            # fallback at all (see below), so it's no longer part of this
            # branch.
            snap_frame = start
            ref_rotation = hold_rotation
            event_low_confidence = True
        else:
            snap_frame, _raw_center, raw_rotation = found
            ref_rotation = disambiguate_rotation(raw_rotation, last_rotation_reference)
            event_low_confidence = False

        # Contact-point anchoring: the object's position comes from the
        # attaching joint's own already-tracked body position at the snap
        # frame, not a fresh per-event depth measurement of the object
        # itself -- a single-frame monocular depth fit is too noisy to
        # trust for position. The joint-relative offset's translation
        # component is exactly zero (object_radius=0) or a fixed hand-
        # relative vector (object_radius>0, see below) -- determined once
        # at snap time, never re-measured mid-hold; only the rotational
        # offset carries real per-event information, taken from the
        # object's own fresh DA3-measured orientation.
        ref_center = joint_world[snap_frame, joint_idx, :3, 3].copy()
        local_offset = hand_grip_offset.get(event["region"])
        if local_offset is not None:
            world_direction = joint_world[snap_frame, joint_idx, :3, :3] @ local_offset
            ref_center = ref_center + world_direction

        # A fresh grip's own rest orientation snapped upright, not left at
        # PCA's raw fit -- see `_snap_axis_to_up`'s own docstring. Applied to
        # every event alike (not just the first): each event's `ref_rotation`
        # is already its own independent re-measurement, not carried over
        # from where the previous grip ended, so there was never cross-event
        # rotation continuity to preserve here the way there is for position.
        ref_rotation = _snap_axis_to_up(ref_rotation)
        last_rotation_reference = ref_rotation

        if i == 0:
            # Before the first event, hold this same snap-corrected reference
            # pose -- not a separately-measured "early resting position".
            # Both would be measuring the same physically-stationary object,
            # so any difference between them is pure measurement noise (two
            # independent DA3 reads at different frames), and the earlier
            # design's separate early measurement produced a real, jarring
            # multi-meter pop right at the first contact frame as a result.
            # Reusing the event's own reference pose instead makes entry as
            # pop-free as exit already is (see below) -- both sides of a
            # transition now come from the *same* single trusted measurement.
            translation[:start] = ref_center
            rotation[:start] = ref_rotation

        resolved_events.append({
            "region": event["region"],
            "joint_idx": joint_idx,
            "start_frame": start,
            "end_frame": end,
            "snap_frame": snap_frame,
            "ref_center": ref_center,
            "ref_rotation": ref_rotation,
            "is_low_confidence": event_low_confidence,
        })

        object_transform = _to_transform(ref_center, ref_rotation)
        joint_transform_at_snap = joint_world[snap_frame, joint_idx]
        offset = np.linalg.inv(joint_transform_at_snap) @ object_transform

        joint_transforms = joint_world[start:end + 1, joint_idx]  # (K, 4, 4)
        object_transforms = joint_transforms @ offset
        translation[start:end + 1] = object_transforms[:, :3, 3]
        rotation[start:end + 1] = object_transforms[:, :3, :3]
        is_low_confidence[start:end + 1] = event_low_confidence

        hold_translation = object_transforms[-1, :3, 3]
        hold_rotation = object_transforms[-1, :3, :3]
        translation[end + 1:] = hold_translation
        rotation[end + 1:] = hold_rotation
        is_low_confidence[end + 1:] = True

    return {
        "translation": translation,
        "rotation": rotation,
        "is_low_confidence": is_low_confidence,
        "resolved_events": resolved_events,
    }
