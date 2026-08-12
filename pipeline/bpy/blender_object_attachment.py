"""Adds the interacted-object mesh and drives its pose: baked world-space
keyframes while held, a live Child Of constraint while attached to a bone.
"""

from __future__ import annotations

import math
import types
from typing import TYPE_CHECKING

import numpy as np

from ..algorithms.object_extent_fit import KEY_KIND, KIND_BOX, KIND_CYLINDER, KIND_ELLIPSOID
from ..helpers.bvh_export import CAMERA_TO_BVH_ROOT_ROTATION
from .blender_constants import _FIRST_MOTION_BLENDER_FRAME
from .blender_scene import _iter_action_fcurves

if TYPE_CHECKING:
    import bpy

# Prefix used for the interacted object name. Used as "object_{kind}"
OBJECT_MESH_PREFIX = "object_"

# Standard SMPL-X body joint order (indices 0-21), matches both
# `SmplxSkeleton`'s own joint numbering (`hoi_object_pose.py`'s own
# `REGION_JOINTS`, not importable here, see stage_10_export's module
# docstring) and the addon's own bone names exactly, confirmed against a
# real exported armature. Used to look up which bone an attachment event's
# own `joint_idx` refers to.
_SMPLX_BODY_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)

# Maps the AMASS-file convention `stage_10_export._write_body_amass` writes
# (X, Y-up, Z) into Blender's own live-scene world space (X, Y, Z-up),
# confirmed empirically, not derived: a pose-bone's `.location` (set
# directly from the npz's own `trans`, no reinterpretation, see the addon's
# `animation.py`) lands at world position (x, -z, y), not (x, y, z), because
# every bone's own rest orientation (baked into the addon's bundled rig
# template) already carries this same Y-up -> Z-up change of basis. A
# property of the addon's own rig asset, not of any export operator,
# applies to the live scene as built by `smplx_add_animation`, independent
# of whether anything is ever exported.
_AMASS_TO_BLENDER_WORLD_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def _object_pose_to_blender_world(
    center: np.ndarray, rotation: np.ndarray, pelvis_rest: np.ndarray, floor_offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Maps one frame of the object's pose (`center`/`rotation`, in stage
    6/8's shared incam body-space) into Blender's live-scene world space:
    the same camera->upright change of basis as the body root, the same
    `floor_offset`, then the same AMASS-file->Blender-world mapping the
    body's own pose-bone location implicitly goes through (see
    `_AMASS_TO_BLENDER_WORLD_ROTATION`).

    `center` needs one correction the body's own joints don't: SMPL-X's
    `global_orient` rotates the whole body about the pelvis's own rest
    position, not the world origin. A skeleton joint gets this for free via
    forward kinematics; `center` is a standalone point with no kinematic
    chain of its own, so it has to pivot around that same point explicitly
    to end up correctly placed relative to the body.
    """
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)

    upright_center = CAMERA_TO_BVH_ROOT_ROTATION @ (center - pelvis_rest) + pelvis_rest
    upright_center[1] += floor_offset  # AMASS's own up axis, same as the body's
    upright_rotation = CAMERA_TO_BVH_ROOT_ROTATION @ rotation

    blender_location = _AMASS_TO_BLENDER_WORLD_ROTATION @ upright_center
    blender_rotation = _AMASS_TO_BLENDER_WORLD_ROTATION @ upright_rotation
    return blender_location, blender_rotation


def _add_object_mesh(bpy: types.ModuleType, object_shape: dict) -> bpy.types.Object:
    """Builds a plain, untextured primitive mesh matching `object_shape`'s
    own dimensions, left at the origin; its real pose is set separately,
    held frames via `_keyframe_held_object_pose` and attached ones via
    `_add_attachment_constraint`, not here.
    """
    kind = object_shape[KEY_KIND]
    if kind == KIND_BOX:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.object
        obj.scale = tuple(object_shape["half_extents"])
    elif kind == KIND_ELLIPSOID:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.object
        obj.scale = tuple(object_shape["semi_axes"])
    elif kind == KIND_CYLINDER:
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.object
        # object_extent_fit's own convention uses local axis 0 for length;
        # Blender's primitive defaults to length along local Z, bake a
        # fixed 90-degree remap into the mesh data itself so the fitted
        # rotation (composed in by `_keyframe_held_object_pose`/
        # `_add_attachment_constraint`) can be applied directly afterward
        # without a second, hand-composed correction.
        obj.rotation_euler = (0.0, math.radians(90.0), 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        radius = object_shape["radius"]
        obj.scale = (object_shape["half_height"], radius, radius)
    else:
        raise ValueError(f"unknown object shape kind: {kind!r}")

    obj.name = OBJECT_MESH_PREFIX + kind
    return obj


def _held_frame_mask(n_frames: int, attachment_events: list[dict]) -> np.ndarray:
    """`True` for every frame not inside any attachment event's own
    `[start_frame, end_frame]`, exactly the frames `_keyframe_held_object_
    pose` should bake as absolute world-space keyframes; the rest are driven
    live by a Child Of constraint instead (see `_add_attachment_constraint`).
    """
    held = np.ones(n_frames, dtype=bool)
    for event in attachment_events:
        held[event["start_frame"]:event["end_frame"] + 1] = False
    return held


def _keyframe_held_object_pose(
    obj: bpy.types.Object, translations: np.ndarray, rotations: np.ndarray, held_mask: np.ndarray,
    pelvis_rest: np.ndarray, floor_offset: float,
) -> None:
    """Keyframes `obj`'s own location/rotation for every *held* frame only
    (see `_held_frame_mask`); an attached frame must NOT also carry a baked
    keyframe, since a Child Of constraint composes its own live result with
    the object's own base transform, and a stale held value underneath
    would corrupt it (`_add_attachment_constraint` resets the base
    transform to identity for each attached window instead).

    Frame 1 (the body's own prepended rest-pose frame) has no equivalent
    concept for the object, so it's held at the object's own first real
    frame's pose too.

    A held frame is skipped (no keyframe) when it's bit-for-bit identical
    to both neighbors and both are themselves held, a flat stretch's own
    first/last held frame already reproduces the whole stretch exactly
    under linear interpolation.
    """
    from mathutils import Matrix

    def same(i: int, j: int) -> bool:
        return bool(np.array_equal(translations[i], translations[j]) and np.array_equal(rotations[i], rotations[j]))

    obj.rotation_mode = "QUATERNION"
    n = len(translations)
    for i in range(n):
        if not held_mask[i]:
            continue
        prev_same = i > 0 and held_mask[i - 1] and same(i, i - 1)
        next_same = i < n - 1 and held_mask[i + 1] and same(i, i + 1)
        if prev_same and next_same:
            continue  # strictly redundant, both neighbors already pin this exact value

        blender_location, blender_rotation = _object_pose_to_blender_world(
            translations[i], rotations[i], pelvis_rest, floor_offset,
        )
        obj.location = tuple(blender_location)
        obj.rotation_quaternion = Matrix(blender_rotation.tolist()).to_quaternion()
        frame = i + _FIRST_MOTION_BLENDER_FRAME
        if i == 0:
            obj.keyframe_insert(data_path="location", frame=frame - 1)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame - 1)
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    # Real motion (an attached window, before the constraint switch above
    # existed) used to be keyframed every frame here too; now only held
    # frames ever are, but the same reasoning still applies to them: Bezier's
    # default overshoot/smoothing between adjacent keyframes only ever
    # hurts, so linear interpolation is used instead (reproduces a flat hold
    # exactly, interpolating between two identical keyframed values is
    # constant).
    if obj.animation_data and obj.animation_data.action:
        for fcurve in _iter_action_fcurves(obj.animation_data.action):
            if fcurve.data_path in ("location", "rotation_quaternion"):
                for keyframe_point in fcurve.keyframe_points:
                    keyframe_point.interpolation = "LINEAR"


def _reset_object_base_transform(obj: bpy.types.Object, frame: int) -> None:
    """Keyframes `obj`'s own location/rotation to identity at `frame`,
    called at *both* the start and end frame of each attached window (two
    matching identity keyframes, same "flat span" principle
    `_keyframe_held_object_pose` already uses) so the object's own base
    transform, composed underneath a Child Of constraint's own live result
    (see that function's docstring), stays at identity for the *whole*
    window, not just its first frame. Calling this only at the start was
    tried first and was a real bug: with only one identity keyframe and the
    next real keyframe being the "held" value the window hands back to
    afterward, Blender linearly interpolates *between* them across the
    entire window instead of holding flat, confirmed by a test that checks
    the object's own position relative to the bone stays rigid across
    several frames within a window, which failed exactly this way (matching
    at the start frame, drifting further off it deeper into the window)."""
    from mathutils import Quaternion

    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def _add_attachment_constraint(
    bpy: types.ModuleType, obj: bpy.types.Object, armature: bpy.types.Object, event: dict, n_frames: int,
    pelvis_rest: np.ndarray, floor_offset: float,
) -> None:
    """Adds one Child Of constraint driving `obj` rigidly from the event's
    own attaching bone during `[start_frame, end_frame]`, instead of baking
    per-frame world-space keyframes: the constraint re-derives the object's
    world transform from whatever pose the bone is actually in at playback
    time, so the attachment survives retargeting onto a different rig (see
    `hoi_object_pose.compute_object_pose_sequence`'s own docstring for the
    full reasoning).

    One constraint per event, not one shared constraint, naturally handles
    a hand-off between different attaching body parts across events: each
    event's own constraint only ever targets its own bone, active only
    during its own window.
    """
    from mathutils import Matrix

    scene = bpy.context.scene
    start, end = event["start_frame"], event["end_frame"]
    bone_name = _SMPLX_BODY_JOINT_NAMES[event["joint_idx"]]

    # The bone's own *live* world transform at the snap frame, read
    # directly from the already-built, already-keyframed armature, not
    # recomputed. This already reflects the same camera->upright/floor-
    # offset chain `_object_pose_to_blender_world` applies to the object
    # below (the armature was built from AMASS data that chain already
    # produced), so composing the two gives a constraint offset expressed
    # correctly in Blender's own live space, not stage 8's incam space.
    scene.frame_set(event["snap_frame"] + _FIRST_MOTION_BLENDER_FRAME)
    bone_world = armature.matrix_world @ armature.pose.bones[bone_name].matrix

    blender_location, blender_rotation = _object_pose_to_blender_world(
        np.array(event["ref_center"]), np.array(event["ref_rotation"]), pelvis_rest, floor_offset,
    )
    ref_transform = Matrix.Translation(blender_location) @ Matrix(blender_rotation.tolist()).to_4x4()

    constraint = obj.constraints.new("CHILD_OF")
    constraint.name = f"attach_{start}_{end}"
    constraint.target = armature
    constraint.subtarget = bone_name
    constraint.inverse_matrix = bone_world.inverted() @ ref_transform

    frame_start = start + _FIRST_MOTION_BLENDER_FRAME
    frame_end = end + _FIRST_MOTION_BLENDER_FRAME
    data_path = f'constraints["{constraint.name}"].influence'
    if start > 0:
        constraint.influence = 0.0
        obj.keyframe_insert(data_path=data_path, frame=frame_start - 1)
    constraint.influence = 1.0
    obj.keyframe_insert(data_path=data_path, frame=frame_start)
    obj.keyframe_insert(data_path=data_path, frame=frame_end)
    if end + 1 < n_frames:
        constraint.influence = 0.0
        obj.keyframe_insert(data_path=data_path, frame=frame_end + 1)

    # A hard on/off step, not a ramp, matches the hard-cut philosophy
    # already used for the held pose itself (see hoi_object_pose.py's own
    # module docstring for why: a blend would just draw out a transition
    # that should resolve the instant real data (contact) is available).
    if obj.animation_data and obj.animation_data.action:
        for fcurve in _iter_action_fcurves(obj.animation_data.action):
            if fcurve.data_path == data_path:
                for keyframe_point in fcurve.keyframe_points:
                    keyframe_point.interpolation = "CONSTANT"
