"""export: combines the retargeted SMPL-X body+hands motion into a single
`output.blend` -- this pipeline's actual final deliverable, written to the
run's top level (not nested under a per-stage subdirectory like every other
stage's output, since this is the actual end result, not an intermediate
artifact). A native Blender file, not FBX -- see "Why .blend, not FBX"
below.

If `align_scene_scale` tracked an object, it's added as a second, independent
top-level mesh: sized/shaped from `object_shape.json` (stage 6). During a
genuine hold, it's rigidly attached to the holding bone via a live **Child Of
constraint** (target the armature, subtarget the bone, keyframed influence
0/1 -- see `_add_attachment_constraint`), not baked per-frame world-space
keyframes -- so the object stays correctly attached even if the skeleton's
own animation is later retargeted onto a different rig (different bone
lengths/proportions), the same way the retargeted rig's own bone rotations
reconstruct its animation, rather than replaying a recording made for this
rig's own proportions. Held perfectly still (baked world-space keyframes,
unparented) everywhere else -- a real-world resting position has nothing to
do with which character's hand eventually reaches for it, so it doesn't need
to be rig-relative at all. See stage 8's own module docstring for the
attached/held design this reflects. Human-only runs (no object tracked) fall
back to the body-only export exactly as before.

The object mesh is added directly into the same live scene `smplx_add_
animation` builds and keyframed there -- no separate export/reimport step,
no duplicate scene. Earlier versions of this file went through the SMPL-X
addon's own `smplx_export_fbx` operator first (even for a body-only run) to
get its shape-key-baking/Unity-axis-remap side effects; reading that
operator's own source (`operators/export.py`) shows it actually duplicates
the mesh/armature, bakes/rotates the *duplicate*, exports it, then deletes it
-- the original scene objects it leaves behind are never touched. So that
call was never actually necessary for correctness here, and calling it at all
would have baked shape keys that a native `.blend` save doesn't need (Blender
handles live shape keys and drivers natively, no baking required to view or
render them correctly).

**Why .blend, not FBX**: stage 8 now holds the object perfectly still for
long stretches between attachment events (see hoi_object_pose.py's own module
docstring) and keyframes those flat stretches sparsely -- two keyframes at a
stretch's own first/last frame, not one per frame, since linear interpolation
between two identical values is already exactly flat. That sparse
keyframing surfaced a real, previously-invisible imprecision in Blender's own
FBX exporter: baking the animation for FBX export (its own default,
`bake_anim=True`) introduced a ~1.3cm single-frame error right at a flat
stretch's own first frame, even though the source data and the in-memory
F-curve were both confirmed exactly correct beforehand -- confirmed by
writing the exact same keyframed object straight to `.blend` instead (no bake
step involved in that path at all) and finding it reproduced the source data
exactly, with no artifact. This was a smaller version of an already-known
FBX-export quirk this file used to work around (combining two independently-
keyframed actions in one FBX export could silently reintroduce the body's
pre-floor-offset height) -- rather than keep chasing FBX-bake-specific
precision issues, the fix is to stop asking Blender to bake anything at all:
write the scene's own native representation directly.

Needs `bpy`, which lives only in the separate `export` pixi environment --
every bpy touch point here is inside `run()` (imported lazily, not at module
load time) so this file stays importable under the `main` environment too
(needed for `pipeline.run`'s own dispatch logic and so this module's own
constants are readable without bpy installed). That environment has no torch
at all (kept minimal on purpose, see `pixi.toml`), so this reads
`retarget_hands`'s plain-numpy `.npz` companion output, never its `.pt` --
and deliberately never imports `gvhmr_adapter`/`hamer_adapter`, even just for
their KEY_* string constants, since both pull in `torch` at module load time
(`object_extent_fit.py` is plain numpy underneath and is imported directly
for its shape-descriptor constants -- no torch involved there).

Retarget_hands' motion is in GVHMR's own incam (raw camera-space, X-right/
Y-down/Z-forward) frame, never GVHMR's separate "global" one -- see this
project's own depth-reliability investigation for why staying in incam
avoids a fragile, unnecessary incam-to-global reconciliation for the
eventual per-frame object pose. The AMASS/SMPL-X-addon import path expects
an upright, gravity-aligned root, though (confirmed empirically: feeding
incam's raw root orientation/translation straight through put the head
below the feet), so the root gets the exact same camera-space -> upright
change of basis already proven for the BVH previews
(`bvh_export.CAMERA_TO_BVH_ROOT_ROTATION`) before being written out --
non-root joints are unaffected (their rotations are relative to their
parent, not the world).

Two more real gaps found by the user inspecting a real export: incam space
has no inherent floor, so the character's absolute height after the
rotation above was wherever GVHMR's raw numbers happened to put it (~1.7m
underground on a real clip) -- `run()` builds the animation once to measure
the lowest foot height, then rebuilds with a compensating offset baked into
the root's own translation (see `_lowest_foot_z`/`floor_offset`). Separately,
retargeting this onto another rig has no bind/reference pose to work from, so
`_prepend_rest_pose_frame` always adds one neutral SMPL-X rest-stance frame
before the real motion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..algorithms.object_extent_fit import KEY_KIND, KIND_BOX, KIND_CYLINDER, KIND_ELLIPSOID
from ..helpers.amass_export_helper import write_amass_npz
from ..helpers.bvh_export import CAMERA_TO_BVH_ROOT_ROTATION
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName

EXPORT_DIRNAME = f"stage{StageName.STAGE_9_EXPORT.stage_number}_export"
BODY_AMASS_FILENAME = "body.npz"
OUTPUT_BLEND_FILENAME = "output.blend"

# Prefix used for the interacted object name. Used as "object_{kind}"
OBJECT_MESH_PREFIX = "object_"

# The addon's own default names (e.g. "SMPLX-lh-neutral_body") are renamed to
# these -- readable in the Outliner without needing to know the addon's own
# internal model-spec naming convention.
PERSON_ARMATURE_NAME = "person"
PERSON_ARMATURE_DATA_NAME = "person_armature"
PERSON_MESH_NAME = "person_mesh"

# This stage's own progress.json output key.
OUTPUT_BLEND = "output_blend"

# Matches stage_6_align_scene_scale.OUTPUT_OBJECT_SHAPE/OUTPUT_SCENE_SCALE
# and scene_scale.json's own KEY_PELVIS_REST -- defined locally, not
# imported, since that module transitively pulls in torch (see module
# docstring). Absent from a human-only run's outputs (no object tracked).
_OBJECT_SHAPE_OUTPUT_KEY = "object_shape"
_SCENE_SCALE_OUTPUT_KEY = "scene_scale"
_PELVIS_REST_KEY = "pelvis_rest_incam"

# Matches stage_8_optimize_hoi.OUTPUT_OBJECT_POSE_NPZ/OUTPUT_ATTACHMENT_EVENTS
# and their own array/field names exactly -- defined locally, not imported,
# since that module transitively pulls in torch (see module docstring).
_OBJECT_POSE_NPZ_OUTPUT_KEY = "object_pose_npz"
_KEY_TRANSLATION = "translation"
_KEY_ROTATION = "rotation"
_ATTACHMENT_EVENTS_OUTPUT_KEY = "attachment_events"

# Standard SMPL-X body joint order (indices 0-21) -- matches both
# `SmplxSkeleton`'s own joint numbering (`hoi_object_pose.py`'s own
# `REGION_JOINTS`, not importable here -- see module docstring) and the
# addon's own bone names exactly, confirmed against a real exported
# armature. Used to look up which bone an attachment event's own `joint_idx`
# refers to.
_SMPLX_BODY_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)

# Stage 8's object_pose.npz index 0 (the source video's own first frame)
# lands on this Blender frame, one later than the index alone would suggest
# -- Blender frame 1 is always the body's own prepended rest-pose frame
# (`_prepend_rest_pose_frame`), which has no equivalent concept for the
# object. The object is just held at its own first frame's pose for that
# one extra frame too (see `_keyframe_held_object_pose`).
_OBJECT_FIRST_MOTION_BLENDER_FRAME = 2

# Matches retarget_hands.OUTPUT_RETARGET_MOTION_NPZ and its own npz key names
# exactly (the latter same strings as gvhmr_adapter.KEY_GLOBAL_ORIENT/
# KEY_BODY_POSE/KEY_BETAS/KEY_TRANSL and hamer_adapter.KEY_LEFT_HAND_POSE/
# KEY_RIGHT_HAND_POSE) -- defined locally, not imported, since importing
# `stage_5_retarget_hands` (or the adapter modules) transitively pulls in
# torch at module load time (see module docstring).
_RETARGET_MOTION_NPZ_OUTPUT_KEY = "retarget_motion_npz"
_KEY_GLOBAL_ORIENT = "global_orient"
_KEY_BODY_POSE = "body_pose"
_KEY_BETAS = "betas"
_KEY_TRANSL = "transl"
_KEY_LEFT_HAND_POSE = "left_hand_pose"
_KEY_RIGHT_HAND_POSE = "right_hand_pose"

# The addon's own AMASS-import operator over-rotates 90 degrees for this
# installed addon/Blender version specifically when told anim_format="AMASS"
# -- the file itself stays real AMASS-formatted, only this operator argument
# differs (see docs/ARCHITECTURE.md's stage 2 addon-quirk note).
_ADDON_ANIM_FORMAT = "SMPL-X"

# Installed via Blender 4.2+'s Extensions system (not the legacy addons
# folder), hence this module name rather than a plain "smplx_blender_addon".
# `bpy.ops.object.smplx_*` exists once discovered (bpy.utils.script_paths
# finds the on-disk files) but is NOT auto-enabled by a bare embedded-bpy
# session the way a real Blender.app session would restore its own saved
# preferences -- `hasattr` on a bpy.ops operator is also not a reliable
# existence check either way (it returns True for literally any name; only
# calling it actually validates), so this always explicitly enables the
# addon rather than checking first.
_ADDON_MODULE_NAME = "bl_ext.user_default.smplx_blender_addon"


def _root_camera_to_upright(global_orient: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorients the root from camera space into the addon's expected
    upright frame -- the same change of basis already proven for the BVH
    previews, left-multiplied onto the root's own rotation matrix (not the
    other joints, whose rotations are relative to their parent already) and
    applied to the root translation the same way a direction vector would
    need (see `smplx_bvh_preview.py`'s identical pattern)."""
    root_matrix = Rotation.from_rotvec(global_orient).as_matrix()
    root_matrix = CAMERA_TO_BVH_ROOT_ROTATION @ root_matrix
    corrected_orient = Rotation.from_matrix(root_matrix).as_rotvec()
    corrected_transl = transl @ CAMERA_TO_BVH_ROOT_ROTATION.T
    return corrected_orient, corrected_transl


# Maps the AMASS-file convention `_write_body_amass` writes (X, Y-up, Z) into
# Blender's own live-scene world space (X, Y, Z-up) -- confirmed empirically,
# not derived: a pose-bone's `.location` (set directly from the npz's own
# `trans`, no reinterpretation -- see the addon's `animation.py`) lands at
# world position (x, -z, y), not (x, y, z), because every bone's own rest
# orientation (baked into the addon's bundled rig template) already carries
# this same Y-up -> Z-up change of basis. A property of the addon's own rig
# asset, not of any export operator -- applies to the live scene as built by
# `smplx_add_animation`, independent of whether anything is ever exported.
_AMASS_TO_BLENDER_WORLD_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def _object_pose_to_blender_world(
    center: np.ndarray, rotation: np.ndarray, pelvis_rest: np.ndarray, floor_offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Maps one frame of the object's pose (`center`/`rotation`, in stage
    6/8's shared incam body-space -- the same space the body root starts in)
    into Blender's live-scene world space: the same camera->upright change of
    basis as the body root, the same `floor_offset` the body's own
    translation gets, then the same AMASS-file->Blender-world mapping the
    body's own pose-bone location implicitly goes through (see
    `_AMASS_TO_BLENDER_WORLD_ROTATION`). Called once per held frame by
    `_keyframe_held_object_pose`, and once per event (at its own snap frame)
    by `_add_attachment_constraint` -- stage 8 provides a real reference
    pose either way, not just one whole-clip anchor frame.

    `center` needs one correction the body's own joints don't: SMPL-X's
    `global_orient` rotates the whole body *about the pelvis's own rest
    position*, not the world origin (confirmed empirically -- with
    `transl=0`, changing `global_orient` leaves the reported pelvis position
    unchanged). A skeleton joint gets this for free via forward kinematics;
    `center` is a standalone point with no kinematic chain of its own, so it
    has to pivot around that same point explicitly to end up correctly
    placed relative to the body -- a real bug found here: without this, a
    real clip's object landed ~0.7m away from the hand it was fit next to,
    instead of the ~0.2m the raw (untransformed) data itself shows.
    """
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)

    upright_center = CAMERA_TO_BVH_ROOT_ROTATION @ (center - pelvis_rest) + pelvis_rest
    upright_center[1] += floor_offset  # AMASS's own up axis, same as the body's
    upright_rotation = CAMERA_TO_BVH_ROOT_ROTATION @ rotation

    blender_location = _AMASS_TO_BLENDER_WORLD_ROTATION @ upright_center
    blender_rotation = _AMASS_TO_BLENDER_WORLD_ROTATION @ upright_rotation
    return blender_location, blender_rotation


def _add_object_mesh(bpy, object_shape: dict):
    """Builds a plain, untextured primitive mesh matching `object_shape`'s
    own dimensions -- left at the origin; its real pose is set separately,
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
        # Blender's primitive defaults to length along local Z -- bake a
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


def _iter_action_fcurves(action):
    """Blender 4.4+ (this addon's target, 5.x here) moved F-curves off of
    `Action.fcurves` directly and into a layered layers/strips/channelbags
    structure -- `Action.fcurves` doesn't exist at all anymore on a plain
    `bpy.types.Action`. Falls back to the old direct attribute in case a
    future/older bpy build still has it, so this doesn't silently stop
    working across a bpy version bump."""
    if hasattr(action, "fcurves"):
        yield from action.fcurves
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                yield from channelbag.fcurves


def _held_frame_mask(n_frames: int, attachment_events: list[dict]) -> np.ndarray:
    """`True` for every frame not inside any attachment event's own
    `[start_frame, end_frame]` -- exactly the frames `_keyframe_held_object_
    pose` should bake as absolute world-space keyframes; the rest are driven
    live by a Child Of constraint instead (see `_add_attachment_constraint`).
    """
    held = np.ones(n_frames, dtype=bool)
    for event in attachment_events:
        held[event["start_frame"]:event["end_frame"] + 1] = False
    return held


def _keyframe_held_object_pose(
    obj, translations: np.ndarray, rotations: np.ndarray, held_mask: np.ndarray,
    pelvis_rest: np.ndarray, floor_offset: float,
) -> None:
    """Keyframes `obj`'s own location/rotation for every *held* frame only
    (see `_held_frame_mask`) -- an attached frame must NOT also carry a
    baked location/rotation keyframe, since a Child Of constraint composes
    its own live result with the object's own base transform
    (`constraint_result @ object_own_matrix_basis`); a stale or interpolated
    base transform left over from the surrounding held keyframes would
    corrupt it. `_add_attachment_constraint` resets the base transform to
    identity for the duration of each attached window separately.

    Frame 1 (the body's own prepended rest-pose frame) has no equivalent
    concept for the object, so it's just held at the object's own first real
    frame's pose too (see `_OBJECT_FIRST_MOTION_BLENDER_FRAME`).

    A held frame is skipped (no keyframe) when it's bit-for-bit identical to
    both neighbors *and* both neighbors are themselves held -- a flat
    stretch's own first/last held frame already reproduces the whole stretch
    exactly under linear interpolation, so only those boundary frames need
    to actually exist. The frame right before an attached window always
    forces a keyframe here (its "next" neighbor isn't held), and the frame
    right after always forces one too (its "previous" neighbor isn't held)
    -- exactly the two frames needed to correctly hold flat right up to the
    transition on both sides.
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
            continue  # strictly redundant -- both neighbors already pin this exact value

        blender_location, blender_rotation = _object_pose_to_blender_world(
            translations[i], rotations[i], pelvis_rest, floor_offset,
        )
        obj.location = tuple(blender_location)
        obj.rotation_quaternion = Matrix(blender_rotation.tolist()).to_quaternion()
        frame = i + _OBJECT_FIRST_MOTION_BLENDER_FRAME
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
    # exactly -- interpolating between two identical keyframed values is
    # constant).
    if obj.animation_data and obj.animation_data.action:
        for fcurve in _iter_action_fcurves(obj.animation_data.action):
            if fcurve.data_path in ("location", "rotation_quaternion"):
                for keyframe_point in fcurve.keyframe_points:
                    keyframe_point.interpolation = "LINEAR"


def _reset_object_base_transform(obj, frame: int) -> None:
    """Keyframes `obj`'s own location/rotation to identity at `frame` --
    called at *both* the start and end frame of each attached window (two
    matching identity keyframes, same "flat span" principle
    `_keyframe_held_object_pose` already uses) so the object's own base
    transform, composed underneath a Child Of constraint's own live result
    (see that function's docstring), stays at identity for the *whole*
    window, not just its first frame. Calling this only at the start was
    tried first and was a real bug: with only one identity keyframe and the
    next real keyframe being the "held" value the window hands back to
    afterward, Blender linearly interpolates *between* them across the
    entire window instead of holding flat -- confirmed by a test that checks
    the object's own position relative to the bone stays rigid across
    several frames within a window, which failed exactly this way (matching
    at the start frame, drifting further off it deeper into the window)."""
    from mathutils import Quaternion

    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def _add_attachment_constraint(
    bpy, obj, armature, event: dict, n_frames: int, pelvis_rest: np.ndarray, floor_offset: float,
) -> None:
    """Adds one Child Of constraint driving `obj` rigidly from the event's
    own attaching bone during `[start_frame, end_frame]`, instead of baking
    per-frame world-space keyframes for that window -- this is what lets the
    object stay correctly attached if the skeleton's own animation is later
    retargeted onto a different rig (different bone lengths/proportions):
    the constraint re-derives the object's world transform from whatever
    pose the bone is *actually* in at playback time, the same way a
    retargeted rig's own bone rotations reconstruct its animation, rather
    than replaying a recording made for this rig's own proportions (see
    `hoi_object_pose.compute_object_pose_sequence`'s own docstring for the
    full reasoning, and `Args` below for why the constraint's own
    `inverse_matrix` is computed here, in Blender's live coordinate space,
    rather than reusing stage 8's own incam-space offset directly).

    One constraint per event (not one shared, retargeted constraint) --
    naturally handles a hand-off between different attaching body parts
    across events (e.g. the right hand picks the object up, sets it down,
    the left hand picks it up later): each event's own constraint only ever
    targets its own bone, active only during its own window.
    """
    from mathutils import Matrix

    scene = bpy.context.scene
    start, end = event["start_frame"], event["end_frame"]
    bone_name = _SMPLX_BODY_JOINT_NAMES[event["joint_idx"]]

    # The bone's own *live* world transform at the snap frame -- read
    # directly from the already-built, already-keyframed armature, not
    # recomputed. This already reflects the same camera->upright/floor-
    # offset chain `_object_pose_to_blender_world` applies to the object
    # below (the armature was built from AMASS data that chain already
    # produced), so composing the two gives a constraint offset expressed
    # correctly in Blender's own live space, not stage 8's incam space.
    scene.frame_set(event["snap_frame"] + _OBJECT_FIRST_MOTION_BLENDER_FRAME)
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

    frame_start = start + _OBJECT_FIRST_MOTION_BLENDER_FRAME
    frame_end = end + _OBJECT_FIRST_MOTION_BLENDER_FRAME
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

    # A hard on/off step, not a ramp -- matches the hard-cut philosophy
    # already used for the held pose itself (see hoi_object_pose.py's own
    # module docstring for why: a blend would just draw out a transition
    # that should resolve the instant real data (contact) is available).
    if obj.animation_data and obj.animation_data.action:
        for fcurve in _iter_action_fcurves(obj.animation_data.action):
            if fcurve.data_path == data_path:
                for keyframe_point in fcurve.keyframe_points:
                    keyframe_point.interpolation = "CONSTANT"


def _prepend_rest_pose_frame(
    global_orient: np.ndarray, body_pose: np.ndarray, transl: np.ndarray,
    left_hand_pose: np.ndarray, right_hand_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adds one neutral (all-zero-rotation, SMPL-X's own canonical rest
    stance) frame before the real motion. Retargeting this onto another rig
    otherwise has no bind/reference pose to work from -- a real, concrete
    usability gap the user hit trying to do exactly that. Positioned at the
    real motion's own first frame's location, so only the *pose* jumps at
    the frame 0->1 boundary, not the character's position -- a static
    calibration frame most retargeting tools skip or trim anyway. Always
    included; not currently exposed as an opt-out flag (the existing
    `RunInput` bool-flag CLI plumbing is wired for opt-in/default-off flags
    like the `--render-*-preview` family, not default-on ones -- add one if
    a real need to disable this ever comes up).
    """
    rest_global_orient = np.zeros((1, 3), dtype=global_orient.dtype)
    rest_body_pose = np.zeros((1, body_pose.shape[1]), dtype=body_pose.dtype)
    rest_left_hand_pose = np.zeros((1, left_hand_pose.shape[1]), dtype=left_hand_pose.dtype)
    rest_right_hand_pose = np.zeros((1, right_hand_pose.shape[1]), dtype=right_hand_pose.dtype)
    rest_transl = transl[:1]

    return (
        np.concatenate([rest_global_orient, global_orient], axis=0),
        np.concatenate([rest_body_pose, body_pose], axis=0),
        np.concatenate([rest_transl, transl], axis=0),
        np.concatenate([rest_left_hand_pose, left_hand_pose], axis=0),
        np.concatenate([rest_right_hand_pose, right_hand_pose], axis=0),
    )


def _write_body_amass(motion: dict, fps: float, out_path: Path, *, floor_offset: float = 0.0) -> None:
    """Maps retarget_hands' npz schema into the AMASS format the addon
    expects. Pure numpy (+ scipy for the root's rotation-matrix round trip),
    no bpy needed -- kept separate from `run()` so it's testable under any
    environment, not just `export`.

    `floor_offset` shifts the root's own up-axis translation by a constant
    (added post-rotation, in the same upright frame `_root_camera_to_upright`
    already produces) -- incam space has no inherent floor reference, so
    without this the character's absolute height is wherever GVHMR's raw
    numbers happened to put it (confirmed on a real clip: ~1.7m underground).
    `run()` computes the real value empirically (there's no way to derive it
    from the source data alone) and passes it in; callers that don't care
    about floor placement can omit it.
    """
    global_orient, transl = _root_camera_to_upright(motion[_KEY_GLOBAL_ORIENT], motion[_KEY_TRANSL])
    transl = transl.copy()
    transl[:, 1] += floor_offset  # AMASS's own up axis

    global_orient, body_pose, transl, left_hand_pose, right_hand_pose = _prepend_rest_pose_frame(
        global_orient, motion[_KEY_BODY_POSE], transl, motion[_KEY_LEFT_HAND_POSE], motion[_KEY_RIGHT_HAND_POSE],
    )

    write_amass_npz(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=motion[_KEY_BETAS][0],  # pooled -- identical every frame
        transl=transl,
        fps=fps,
        out_path=out_path,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
    )


def _lowest_foot_z(armature) -> float:
    """Scans every frame of the already-keyframed armature and returns the
    lowest world-space Z any foot/ankle bone reaches across the whole clip
    -- Blender's own native internal convention is always Z-up regardless of
    the eventual export's own target axis convention (e.g. Unity's Y-up is
    applied only at export time), so this reads Z directly rather than
    guessing which axis is "up" at this point in the pipeline.

    The frame range comes from the armature's own action
    (`animation_data.action.frame_range`), NOT `scene.frame_start`/
    `frame_end` -- confirmed on a real 676-frame clip that the addon never
    updates the scene's own frame range when it builds a longer animation
    (it stayed at Blender's stock default, 1-250), so trusting the scene
    would silently scan only the first ~37% of a clip like that one.
    """
    import bpy

    scene = bpy.context.scene
    frame_start, frame_end = (int(v) for v in armature.animation_data.action.frame_range)
    foot_bones = [b for b in armature.pose.bones if "foot" in b.name.lower() or "ankle" in b.name.lower()]
    lowest = float("inf")
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        for bone in foot_bones:
            world_z = (armature.matrix_world @ bone.head).z
            lowest = min(lowest, world_z)
    return lowest


def _clear_scene(bpy) -> None:
    """Clears whatever the current scene contains (a stray cube/camera/
    light, or a previous pass's own armature+mesh) without touching
    preferences -- `bpy.ops.wm.read_factory_settings` also resets which
    addons are enabled, which would disable the SMPL-X addon itself.

    Also purges orphaned data-blocks (`do_recursive=True` catches the
    cascade: removing an object orphans its mesh/armature/action data,
    which can itself then orphan more) -- a real bug found while building
    the floor-grounding fix: this stage runs `smplx_add_animation` twice
    (the floor-offset two-pass build, see `run()`), and leftover orphaned
    action data-blocks from the *first* (uncorrected) pass were somehow
    resurfacing in place of the correct, second-pass one. Purging after
    every clear removes the ambiguity entirely rather than chasing the
    exact reuse mechanism.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def run(runRecord: RunRecord) -> dict[str, str]:
    # `bpy` must be imported before `addon_utils` -- importing bpy is what
    # puts Blender's own bundled scripts directory on sys.path in the first
    # place, which is where addon_utils itself lives.
    import bpy
    import addon_utils

    addon_utils.enable(_ADDON_MODULE_NAME, default_set=True, persistent=True)

    motion = np.load(runRecord.stages[StageName.STAGE_5_RETARGET_HANDS].outputs[_RETARGET_MOTION_NPZ_OUTPUT_KEY])

    export_dir = Path(runRecord.progress_dir) / EXPORT_DIRNAME
    amass_path = export_dir / BODY_AMASS_FILENAME

    # First pass, floor_offset=0: incam space has no inherent floor
    # reference, so how far below (or above) Y=0 the character's own feet
    # land can only be measured empirically, not derived from the source
    # data -- build once to find out.
    _write_body_amass(motion, runRecord.scene.fps, amass_path)
    _clear_scene(bpy)
    bpy.ops.object.smplx_add_animation(filepath=str(amass_path), anim_format=_ADDON_ANIM_FORMAT)
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    floor_offset = -_lowest_foot_z(armature)

    # Second pass: rebuild with that offset baked into the root's own
    # per-frame translation curve -- keeps the addon's separate, always-
    # static "root" bone at true world origin (per the user's own request),
    # since only the *pelvis*'s translation is shifted, not an object-level
    # transform that would move everything including "root".
    _write_body_amass(motion, runRecord.scene.fps, amass_path, floor_offset=floor_offset)
    _clear_scene(bpy)
    bpy.ops.object.smplx_add_animation(filepath=str(amass_path), anim_format=_ADDON_ANIM_FORMAT)

    # The addon names these after its own internal model spec (e.g.
    # "SMPLX-lh-neutral_body"/"SMPLX-mesh-neutral") -- renamed to something
    # a downstream Blender user actually wants to see in the Outliner. Only
    # these two objects exist at this point (`_clear_scene` wiped everything
    # else, and the tracked object mesh, if any, hasn't been added yet).
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    armature.name = PERSON_ARMATURE_NAME
    armature.data.name = PERSON_ARMATURE_DATA_NAME
    body_mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    body_mesh.name = PERSON_MESH_NAME

    object_shape_path = runRecord.stages[StageName.STAGE_6_ALIGN_SCENE_SCALE].outputs.get(_OBJECT_SHAPE_OUTPUT_KEY)
    output_path = Path(runRecord.progress_dir) / OUTPUT_BLEND_FILENAME

    if object_shape_path is not None:
        object_shape = json.loads(Path(object_shape_path).read_text())
        scene_scale_path = runRecord.stages[StageName.STAGE_6_ALIGN_SCENE_SCALE].outputs[_SCENE_SCALE_OUTPUT_KEY]
        pelvis_rest = np.array(json.loads(Path(scene_scale_path).read_text())[_PELVIS_REST_KEY])

        object_pose_npz_path = runRecord.stages[StageName.STAGE_8_OPTIMIZE_HOI].outputs[_OBJECT_POSE_NPZ_OUTPUT_KEY]
        with np.load(object_pose_npz_path) as object_pose:
            translations = object_pose[_KEY_TRANSLATION]
            rotations = object_pose[_KEY_ROTATION]
        n_frames = len(translations)

        attachment_events_path = runRecord.stages[StageName.STAGE_8_OPTIMIZE_HOI].outputs[_ATTACHMENT_EVENTS_OUTPUT_KEY]
        attachment_events = json.loads(Path(attachment_events_path).read_text())

        obj = _add_object_mesh(bpy, object_shape)
        held_mask = _held_frame_mask(n_frames, attachment_events)
        _keyframe_held_object_pose(obj, translations, rotations, held_mask, pelvis_rest, floor_offset)
        for event in attachment_events:
            _reset_object_base_transform(obj, event["start_frame"] + _OBJECT_FIRST_MOTION_BLENDER_FRAME)
            _reset_object_base_transform(obj, event["end_frame"] + _OBJECT_FIRST_MOTION_BLENDER_FRAME)
            _add_attachment_constraint(bpy, obj, armature, event, n_frames, pelvis_rest, floor_offset)

    # Building the animation above (attachment constraints, foot-grounding)
    # moves the scene's own current frame around as a side effect of reading
    # bone/object transforms at specific frames -- reset to frame 0 before
    # saving so the file doesn't open with the playhead sitting wherever that
    # process happened to leave it.
    bpy.context.scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

    # The run's own overall deliverable (docs/PROGRESS_SCHEMA.md's own
    # `RunOutputs.final_blend`), distinct from this stage's own `outputs`
    # dict entry below -- lets a caller find the final file without knowing
    # which stage produced it.
    runRecord.outputs.final_blend = str(output_path)

    return {OUTPUT_BLEND: str(output_path)}


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_9_EXPORT)
