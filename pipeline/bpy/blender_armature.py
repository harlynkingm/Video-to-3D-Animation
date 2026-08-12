"""Post-build armature fixups: bone-tail orientation (cosmetic), deleting
unreliable pelvis keyframes, and quaternion hemisphere continuity.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import numpy as np

from ..helpers.progress_reporter import frame_progress
from ..progress_tracker import StageName
from .blender_constants import _FIRST_MOTION_BLENDER_FRAME
from .blender_scene import _iter_action_fcurves

if TYPE_CHECKING:
    import bpy

# The root/pelvis bone's own name in the addon's rig template, matches
# `stage_10_export._SMPLX_BODY_JOINT_NAMES[0]` (SMPL-X's own joint-0
# convention), but this module already needed a standalone name here since
# that tuple is defined for attachment-event joint lookup, a different,
# unrelated purpose.
_PELVIS_BONE_NAME = "pelvis"
_QUATERNION_COMPONENTS = 4  # w, x, y, z, one F-curve each


def _lowest_foot_z(armature: bpy.types.Object) -> float:
    """Scans every frame of the already-keyframed armature and returns the
    lowest world-space Z any foot/ankle bone reaches across the whole clip,
    Blender's own native internal convention is always Z-up regardless of
    the eventual export's own target axis convention (e.g. Unity's Y-up is
    applied only at export time), so this reads Z directly rather than
    guessing which axis is "up" at this point in the pipeline.

    The frame range comes from the armature's own action
    (`animation_data.action.frame_range`), NOT `scene.frame_start`/
    `frame_end`, confirmed on a real 676-frame clip that the addon never
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


def _primary_child(edit_bone: bpy.types.EditBone) -> bpy.types.EditBone | None:
    """Which child a bone's tail should point at, when it has more than one
    (a branch point, e.g. pelvis -> {left_hip, right_hip, spine1}, or
    spine3 -> {neck, left_collar, right_collar}). Every bone in the addon's
    own template rig currently points straight up (+Z) in this project's
    T-pose (see `_orient_bones_toward_children`'s own docstring), picking
    whichever child continues furthest in that same +Z direction naturally
    selects the spine's own continuation over a sideways branch (a hip or a
    collar), with no bone-name-specific logic needed. `None` for a leaf bone
    (no children, e.g. a fingertip, or the last bone of a chain)."""
    children = list(edit_bone.children)
    if not children:
        return None
    return max(children, key=lambda c: c.head.z - edit_bone.head.z)


def _hierarchy_order(armature: bpy.types.Object) -> list[str]:
    """Pose bone names, parent before every child, a bone's own pose
    correction (see `_orient_bones_toward_children`) must be applied before
    its children's, since a child's correction reads its *current* (i.e.
    already-corrected, if processed in this order) parent pose to derive the
    right local rotation."""
    order: list[str] = []
    queue = [b for b in armature.pose.bones if b.parent is None]
    while queue:
        bone = queue.pop(0)
        order.append(bone.name)
        queue.extend(bone.children)
    return order


def _animated_bone_names(action: bpy.types.Action | None) -> set[str]:
    """Bone names with at least one existing pose keyframe, everything
    else (e.g. the addon's own always-static "root" bone) gets its own
    orientation correction applied once, directly, with no keyframe
    inserted, rather than adding animation data to a bone that never had
    any."""
    if action is None:
        return set()
    names = set()
    for fcurve in _iter_action_fcurves(action):
        path = fcurve.data_path  # e.g. 'pose.bones["left_elbow"].rotation_quaternion'
        if path.startswith('pose.bones["'):
            names.add(path.split('"')[1])
    return names


def _rotation_keyframe_data_path(pose_bone: bpy.types.PoseBone) -> str:
    return {
        "QUATERNION": "rotation_quaternion",
        "AXIS_ANGLE": "rotation_axis_angle",
    }.get(pose_bone.rotation_mode, "rotation_euler")


def _orient_bones_toward_children(bpy: types.ModuleType, armature: bpy.types.Object) -> None:
    """Points each bone's tail at its own primary child (see
    `_primary_child`), or, for a leaf bone, continues its parent's own
    direction, instead of the addon's own decorative "point straight up"
    template tail. Purely a display/Edit-mode change.

    Retailing a bone changes its own rest transform (`Bone.matrix_local`),
    which the mesh's deform matrix (`PoseBone.matrix @ Bone.matrix_local.
    inverted()`) depends on, so every bone's existing pose animation is
    rewritten afterward (`old_pose @ old_rest^-1 @ new_rest`, assigned back
    to `PoseBone.matrix`) to keep that deform matrix unchanged, otherwise
    retailing would visibly move the mesh, not just the bone display.

    Processes bones parent-first (`_hierarchy_order`) so a child's own
    correction reads its parent's already-corrected pose, not the stale
    pre-retail one.
    """
    from mathutils import Matrix

    scene = bpy.context.scene
    action = armature.animation_data.action if armature.animation_data else None

    keyframed_frames: list[int] = []
    if action is not None:
        frame_set: set[int] = set()
        for fcurve in _iter_action_fcurves(action):
            for kp in fcurve.keyframe_points:
                frame_set.add(int(round(kp.co.x)))
        keyframed_frames = sorted(frame_set)

    animated = _animated_bone_names(action)
    order = _hierarchy_order(armature)
    print(f"[{StageName.STAGE_10B_ALIGN_BONES.label}] re-orienting {len(order)} bones across {len(keyframed_frames)} keyframes...")

    # Ground truth: every bone's own real, evaluated world *pose* transform
    # at every already-keyframed frame, captured BEFORE any edit-mode
    # change, plus each bone's own *rest* transform (`Bone.matrix_local`),
    # also captured before. The rest transform is what actually changes here
    # (that's the whole point of retailing); pose alone preserving its old
    # value is the wrong target, see this function's own docstring for why
    # both are needed to get the actual invariant (the mesh's own deform
    # matrix) right. Unanimated bones only need one pose sample (their pose
    # is constant by definition, no fcurve means matrix_basis never changes).
    sample_frames = keyframed_frames or [int(scene.frame_current)]
    world_before: dict[str, dict[int, Matrix]] = {name: {} for name in order}
    for frame in sample_frames:
        scene.frame_set(frame)
        for name in order:
            if name in animated or frame == sample_frames[0]:
                world_before[name][frame] = armature.pose.bones[name].matrix.copy()
    rest_before = {name: armature.data.bones[name].matrix_local.copy() for name in order}

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones

    # Pass 1: every bone with a child points at it (see _primary_child).
    child_targets: dict[str, Matrix] = {}
    for bone in edit_bones:
        primary = _primary_child(bone)
        if primary is None:
            continue
        new_tail = primary.head.copy()
        if (new_tail - bone.head).length > 1e-6:  # skip a degenerate (near-zero-length) result
            child_targets[bone.name] = new_tail
    for name, new_tail in child_targets.items():
        edit_bones[name].tail = new_tail

    # Pass 2: a leaf bone (no children, still true after pass 1, which
    # only ever moves a *parent's* tail, never a bone's own head) continues
    # its own parent's direction instead. Run as its own pass, after pass 1
    # is fully applied, so it reads the parent's real (already-retailed)
    # direction, not the stale decorative default, otherwise a fingertip
    # would inherit the same "point straight up" look pass 1 exists to fix.
    leaf_targets: dict[str, Matrix] = {}
    for bone in edit_bones:
        if _primary_child(bone) is not None:
            continue
        parent = bone.parent
        if parent is None:
            continue  # an isolated leaf, nothing to continue the direction of
        parent_direction = parent.tail - parent.head
        if parent_direction.length < 1e-9:
            continue
        bone_length = (bone.tail - bone.head).length
        new_tail = bone.head + parent_direction.normalized() * bone_length
        if (new_tail - bone.head).length > 1e-6:
            leaf_targets[bone.name] = new_tail
    for name, new_tail in leaf_targets.items():
        edit_bones[name].tail = new_tail
    bpy.ops.object.mode_set(mode="OBJECT")

    # Blender's own mesh-deform formula for a bone is `PoseBone.matrix @
    # Bone.matrix_local.inverted()` (in armature space, this is what the
    # Armature modifier actually applies to every vertex weighted to this
    # bone, independent of parent-chain depth, since `PoseBone.matrix`
    # already recursively encodes the whole chain). Retailing changes
    # `Bone.matrix_local` (the rest transform); for the *deform* matrix to
    # stay the same, the actual requirement, since that's what the mesh
    # itself follows, `PoseBone.matrix` has to change by exactly the same
    # rest-transform delta: `new_pose = old_pose @ old_rest^-1 @ new_rest`.
    # Simply holding `PoseBone.matrix` fixed at its old value would keep the
    # bone's own world *pose* unchanged while silently changing the deform
    # matrix underneath it, since `old_rest^-1 @ new_rest` isn't identity.
    rest_delta = {name: rest_before[name].inverted() @ armature.data.bones[name].matrix_local for name in order}

    # The dominant cost of this whole function: one `scene.frame_set` (a full
    # dependency-graph re-evaluation) per (animated bone, keyframed frame)
    # pair, for a long clip with a full-body rig this can run into the tens
    # of thousands of evaluations and take minutes with nothing else printed
    # on the way, so this loop gets its own progress bar (reusing the same
    # helper the per-frame pipeline stages use, just counting bones instead).
    for name in frame_progress(order, total=len(order), label=StageName.STAGE_10B_ALIGN_BONES.label, unit="bone"):
        pose_bone = armature.pose.bones[name]
        rotation_path = _rotation_keyframe_data_path(pose_bone)
        delta = rest_delta[name]
        if name in animated:
            for frame in keyframed_frames:
                scene.frame_set(frame)
                pose_bone.matrix = world_before[name][frame] @ delta
                pose_bone.keyframe_insert(data_path=rotation_path, frame=frame)
                pose_bone.keyframe_insert(data_path="location", frame=frame)
        else:
            scene.frame_set(sample_frames[0])
            pose_bone.matrix = world_before[name][sample_frames[0]] @ delta


def _delete_unreliable_root_keyframes(
    bpy: types.ModuleType, armature: bpy.types.Object, root_motion_unreliable: np.ndarray,
) -> None:
    """Deletes the pelvis bone's own location/rotation keyframes at every
    frame `pp_bridge_low_confidence_root_motion` (stage 2) flagged as
    unreliable, so the exported file shows a real gap there instead of a
    baked-but-fabricated keyframe, the same effect as deleting the pelvis
    bone's own keyframes directly in Blender and letting it interpolate the
    resulting gap from its own surrounding real keyframes.
    Blender's default extrapolation (constant, before the first/after the
    last keyframe) naturally reproduces the same freeze behavior stage 2's
    own bridging already uses for a run touching either end of the clip, so
    no special-casing is needed for that case here.

    Must run *after* `_orient_bones_toward_children`, that function
    rewrites every existing keyframe on every animated bone (pelvis
    included) to preserve the mesh's own deform matrix once bone tails are
    retailed, so deleting first would just have those keyframes reinserted
    right back by that pass. Follow with
    `_fix_rotation_hemisphere_continuity`, deleting a run can leave
    two now-distant keyframes whose quaternions Blender's own interpolation
    doesn't bridge correctly on its own; see that function's own docstring.
    """
    pose_bone = armature.pose.bones[_PELVIS_BONE_NAME]
    rotation_path = _rotation_keyframe_data_path(pose_bone)
    for i in np.flatnonzero(root_motion_unreliable):
        frame = int(i) + _FIRST_MOTION_BLENDER_FRAME
        pose_bone.keyframe_delete(data_path=rotation_path, frame=frame)
        pose_bone.keyframe_delete(data_path="location", frame=frame)


def _fix_rotation_hemisphere_continuity(armature: bpy.types.Object) -> None:
    """Blender's `rotation_quaternion` F-curve interpolation is NOT
    hemisphere-aware, it interpolates w/x/y/z as independent scalar curves,
    not a proper slerp. `q` and its negation `-q` are the same rotation but
    look nothing alike numerically, so when two consecutive keyframes land
    on opposite sides of that double-cover, the naive interpolation sweeps
    every component through zero, rendering as the joint swinging through
    an impossible orientation.

    Walks every quaternion-mode bone's keyframes and flips any with a
    negative dot product with the previous one, the same algorithm
    `motion_smoothing.hemisphere_aligned_quats` applies numpy-side. Doesn't
    change the exported pose, only removes the interpolation artifact
    between keyframes. Two real triggers: the pelvis across a keyframe gap
    left by `_delete_unreliable_root_keyframes`, and wrists on adjacent
    keyframes with no gap at all (a wrist rotation often sits near 180
    degrees, where `w` is near zero and `q`/`-q` are equally canonical, so
    numerical noise picks either).

    Works directly on F-curve keyframe values rather than stepping the
    scene frame by frame, since covering every bone that way would be tens
    of thousands of `frame_set` calls on a long clip. Bezier handles are
    negated alongside each flipped value so the curve shape mirrors with
    it."""
    action = armature.animation_data.action

    for pose_bone in armature.pose.bones:
        if pose_bone.rotation_mode != "QUATERNION":
            continue
        data_path = f'pose.bones["{pose_bone.name}"].rotation_quaternion'
        curves = {
            fcurve.array_index: fcurve
            for fcurve in _iter_action_fcurves(action)
            if fcurve.data_path == data_path
        }
        if len(curves) != _QUATERNION_COMPONENTS:
            continue  # not fully keyed (or not keyed at all), nothing to align

        components = [curves[i].keyframe_points for i in range(_QUATERNION_COMPONENTS)]
        n_keyframes = len(components[0])
        if any(len(points) != n_keyframes for points in components):
            continue  # components keyed at different frames, can't pair them up safely

        last = None
        for k in range(n_keyframes):
            quat = np.array([points[k].co[1] for points in components])
            if last is not None and float(quat @ last) < 0:
                quat = -quat
                for value, points in zip(quat, components):
                    keyframe = points[k]
                    keyframe.co[1] = value
                    keyframe.handle_left[1] = -keyframe.handle_left[1]
                    keyframe.handle_right[1] = -keyframe.handle_right[1]
            last = quat

        for fcurve in curves.values():
            fcurve.update()
