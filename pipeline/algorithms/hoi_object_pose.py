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
# X-right/Y-down/Z-forward -- see stage_9_export.py's own module docstring):
# the same -Y_cam convention already established for the root's own
# camera-to-upright correction (`CAMERA_TO_BVH_ROOT_ROTATION`'s own comment),
# reused here rather than re-derived, so an object snapped "upright" by this
# module ends up actually upright once stage 9 applies that same conversion
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
    convention, confirmed empirically during stage 9's own work: the root's
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
) -> dict:
    """The full per-frame object pose for a clip.

    For each attachment event, in order: a single reference measurement is
    taken near the event's own `start_frame` (see `_find_snap_measurement`),
    its rotation axis-disambiguated against the most recently known-good
    orientation, and its position **depth-corrected**: the object's own
    lateral (X/Y) measurement is trusted, but its own depth (Z, the camera's
    forward axis) is replaced with the attaching joint's own Z that same
    frame. This isn't a compromise -- `similarity_transform.py`'s own fit
    already established, on real data, that monocular per-frame depth is
    measurably less reliable along Z specifically than laterally, while the
    body's own motion-capture Z is comparatively trustworthy. The reference
    rotation is also snapped so whichever of its own fitted axes is already
    closest to vertical points exactly upright (see `_snap_axis_to_up`) --
    PCA's own raw fit has no reason to already look that way, especially for
    a rotationally symmetric shape. The corrected reference pose fixes a
    rigid joint-relative offset, which then propagates rigidly for the
    event's whole `[start_frame, end_frame]` range -- so the object's
    rotation during the hold itself still evolves naturally as the joint
    rotates, starting from that upright baseline rather than PCA's raw one.
    Once the event ends, that final frame's own resulting absolute pose
    becomes the new held pose, frozen forward until the next event (or the
    end of the clip) -- the object stays wherever it was last placed, the
    same way a real object would, rather than continuing to move or snapping
    back to some earlier default. The *next* event, when it comes, measures
    and snaps its own fresh reference pose independently -- there was never
    cross-event rotation continuity to preserve (unlike position, a real
    object could plausibly be reoriented by an unseen hand between two
    holds), so this isn't a new discontinuity.

    Before the *first* event, the object holds that same first event's own
    reference pose, not a separately-measured "early resting position" (an
    earlier design measured one independently, scanning forward from frame
    0). Both would be measuring the same physically-stationary object at
    different times, so any disagreement between them is pure measurement
    noise -- two independent depth reads, not two different real positions
    -- and that earlier design produced a real, jarring multi-meter pop
    right at the first contact frame as a result. Reusing the same reference
    pose the event itself already trusts makes entry exactly as pop-free as
    exit already is, for the same reason: both sides of the transition come
    from one single measurement, not two independent ones. `initial_center`/
    `initial_rotation` are used only as a last-resort fallback, for a clip
    with no attachment events at all, or one whose very first event's own
    snap search finds nothing usable.

    There is no cross-fade/blend at transitions (an earlier design had one)
    -- both entry and exit are now already continuous by construction (see
    above), so there's no discontinuity left to blend across.

    Args:
        n_frames: clip length.
        attachment_events: already filtered by the caller to qualifying
            (duration-gated) events, each `{"region": str, "start_frame": int,
            "end_frame": int}` (inclusive), sorted by `start_frame`, with no
            two events overlapping (the caller resolves cross-region/cross-
            hand overlap -- e.g. by `mean_confidence` -- before calling this).
        body_motion: `global_orient (F,3)`, `body_pose (F,63)`, `betas (F,10)`,
            `transl (F,3)` -- same schema as `retarget_hands`'s own output.
        initial_center: fallback held pose center, used only if there are no
            attachment events at all, or the first one's own snap search
            finds nothing usable.
        initial_rotation: fallback held pose rotation (and fallback axis-
            disambiguation reference), used in the same last-resort case as
            `initial_center`.
        object_position_fn: `frame -> (center, rotation) | None` -- the
            expensive one (a fresh depth pass + back-projection +
            `object_extent_fit.fit_position_and_orientation`), or `None` if
            unusable that frame (no mask, too few points).
        skeleton: an `SmplxSkeleton` instance (constructed once by the
            caller, reused across a whole run); defaults to constructing one
            here if not supplied (tests may inject a fake/stub).

    Returns `{"translation": (F,3), "rotation": (F,3,3), "is_low_confidence": (F,) bool,
    "resolved_events": list[dict]}`. `is_low_confidence` is True for every
    held (non-attached) frame, plus any attachment event whose own snap
    search found nothing usable. `resolved_events` carries each event's own
    `{region, joint_idx, start_frame, end_frame, snap_frame, ref_center,
    ref_rotation, is_low_confidence}` -- the raw reference measurement, not
    the rigid offset derived from it. A caller that wants the object to stay
    correctly attached to the joint even after the skeleton itself is later
    retargeted onto a different rig (bone lengths/proportions changed) needs
    a *live* parent/constraint relationship to that joint, not the baked
    `translation`/`rotation` values above (which are frozen for the joint's
    original proportions) -- this is what `resolved_events` is for; the
    caller re-derives the offset itself, in whatever space it needs (e.g.
    stage 9 does this in Blender's own live coordinate space, reusing
    `_object_pose_to_blender_world`, not this module's own incam-space
    `offset`).
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

    for i, event in enumerate(events_sorted):
        start, end = event["start_frame"], event["end_frame"]
        joint_idx = attachment_joint_index(event["region"])

        found = _find_snap_measurement(start, end, object_position_fn)
        if found is None:
            # Never measurable near this event at all -- anchor to whatever
            # pose is currently being held rather than a fresh snap.
            snap_frame = start
            ref_center = hold_translation
            ref_rotation = hold_rotation
            event_low_confidence = True
        else:
            snap_frame, raw_center, raw_rotation = found
            ref_rotation = disambiguate_rotation(raw_rotation, last_rotation_reference)
            ref_center = np.array([raw_center[0], raw_center[1], joint_world[snap_frame, joint_idx, 2, 3]])
            event_low_confidence = False

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
