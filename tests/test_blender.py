"""Unit tests for the bpy-touching helpers in `pipeline/bpy/`, one function
at a time, gated behind `bpy` importability (`tests.conftest.HAS_BPY`) since
these only ever actually run in the separate `export` pixi environment. Run
the real ones with: `pixi run -e export python -m pytest tests/test_blender.py -v`.

Stage 10's own orchestration tests (`run()` end to end, plus the pure-numpy
AMASS-prep logic) live in `tests/test_stage_10_export.py` instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.helpers.bvh_export import CAMERA_TO_BVH_ROOT_ROTATION
from pipeline.algorithms.face.face_preview import _write_pc2, _write_template_npz, flame_to_blender, write_landmark_preview
from pipeline.bpy.blender_armature import _fix_rotation_hemisphere_continuity, _orient_bones_toward_children
from pipeline.bpy.blender_constants import _FIRST_MOTION_BLENDER_FRAME
from pipeline.bpy.blender_face_expression import _EXPRESSION_SHAPEKEY_RANGE, _keyframe_face_expression
from pipeline.bpy.blender_object_attachment import _object_pose_to_blender_world
from pipeline.bpy.blender_preview import VIDEO_PLANE_HEIGHT_M, VIDEO_PLANE_POSITION_M, _add_video_reference_plane, _build_face_preview_blend, _build_landmark_preview_blend
from pipeline.bpy.blender_scene import _iter_action_fcurves
from tests.conftest import HAS_BPY


def test_object_pose_to_blender_world_pivots_around_pelvis_rest_not_the_origin():
    """Regression test for a real bug: SMPL-X's `global_orient` rotates the
    body about the pelvis's own rest position, not the world origin (a
    skeleton joint gets this for free via forward kinematics; a standalone
    point like the object's own `center` does not). Ignoring this placed a
    real clip's object ~0.7m away from the hand it was fit next to, instead
    of the ~0.2m the raw (untransformed) data itself shows.

    Picks `center` so `center - pelvis_rest` is a simple "1 unit up in
    camera space" vector, reusing the same known mapping
    `test_root_camera_to_upright_maps_camera_up_to_target_plus_y` already
    established (camera-space up -> target +Y) to hand-derive the expected
    result, rather than trusting a from-real-data magic number.
    """
    pelvis_rest = np.array([1.0, 2.0, 3.0])
    center = np.array([1.0, 1.0, 3.0])  # pelvis_rest + (0, -1, 0): "1 unit up" in camera space
    rotation = np.eye(3)

    blender_location, _ = _object_pose_to_blender_world(center, rotation, pelvis_rest, floor_offset=0.0)

    # upright_center = CAMERA_TO_BVH_ROOT_ROTATION @ (0, -1, 0) + pelvis_rest = (0, 1, 0) + pelvis_rest = (1, 3, 3)
    # blender_location maps (x, y, z) -> (x, -z, y): (1, -3, 3)
    assert np.allclose(blender_location, [1.0, -3.0, 3.0], atol=1e-6)

    # Guard against regressing to the naive (pivot-at-origin) formula, which
    # would give a visibly different, wrong answer for this same input.
    naive_wrong = CAMERA_TO_BVH_ROOT_ROTATION @ center
    assert not np.allclose(blender_location, naive_wrong, atol=1e-3)


def test_object_pose_to_blender_world_applies_floor_offset():
    pelvis_rest = np.zeros(3)
    center = np.zeros(3)
    rotation = np.eye(3)

    blender_location, _ = _object_pose_to_blender_world(center, rotation, pelvis_rest, floor_offset=5.0)

    # floor_offset lands on the AMASS-frame's own up axis (index 1), which
    # `_AMASS_TO_BLENDER_WORLD_ROTATION` maps to Blender's own Z.
    assert np.allclose(blender_location, [0.0, 0.0, 5.0], atol=1e-6)


def _make_shape_key_mesh(names: list[str]):
    """A minimal real mesh with a `Basis` key plus one shape key per `names`
   , everything `_keyframe_face_expression` touches (`data.shape_keys.
    key_blocks[...]`), without going through the full addon/AMASS-import
    machinery just to get real `Exp###` keys to test against."""
    import bpy

    mesh_data = bpy.data.meshes.new("test_face_mesh")
    mesh_data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    obj = bpy.data.objects.new("test_face_mesh_obj", mesh_data)
    bpy.context.scene.collection.objects.link(obj)
    obj.shape_key_add(name="Basis")
    for name in names:
        obj.shape_key_add(name=name)
    return obj


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_keyframe_face_expression_skips_all_zero_components():
    """A component that's zero on every frame is never touched at all,
    default slider range untouched, no keyframes, no fcurve, since the
    shape key's own default value (0.0) already reproduces it."""
    obj = _make_shape_key_mesh(["Exp000", "Exp001"])
    expression = np.zeros((4, 2), dtype=np.float32)
    expression[:, 0] = [0.0, 0.3, 0.5, 0.2]  # Exp000 moves
    # Exp001 stays all-zero

    _keyframe_face_expression(obj, expression)

    key_blocks = obj.data.shape_keys.key_blocks
    assert key_blocks["Exp000"].slider_min == -_EXPRESSION_SHAPEKEY_RANGE
    assert key_blocks["Exp001"].slider_min != -_EXPRESSION_SHAPEKEY_RANGE  # untouched, still the addon-less default

    action = obj.data.shape_keys.animation_data.action
    touched_paths = {fcurve.data_path for fcurve in _iter_action_fcurves(action)}
    assert 'key_blocks["Exp000"].value' in touched_paths
    assert 'key_blocks["Exp001"].value' not in touched_paths


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_keyframe_face_expression_widens_range_and_uses_linear_interpolation():
    obj = _make_shape_key_mesh(["Exp000"])
    expression = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)

    _keyframe_face_expression(obj, expression)

    key_block = obj.data.shape_keys.key_blocks["Exp000"]
    assert key_block.slider_min == -_EXPRESSION_SHAPEKEY_RANGE
    assert key_block.slider_max == _EXPRESSION_SHAPEKEY_RANGE

    action = obj.data.shape_keys.animation_data.action
    fcurve = next(fc for fc in _iter_action_fcurves(action) if fc.data_path == 'key_blocks["Exp000"].value')
    assert all(kp.interpolation == "LINEAR" for kp in fcurve.keyframe_points)


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_keyframe_face_expression_seeds_a_neutral_rest_frame():
    """Frame 1 (the prepended rest-pose frame, see `_prepend_rest_pose_frame`)
    gets a neutral (0.0) keyframe for any component that moves at all, the
    same jump-not-blend treatment the body pose itself gets across that same
    boundary, not an interpolated blend into the real first frame's value."""
    obj = _make_shape_key_mesh(["Exp000"])
    expression = np.array([[0.8], [0.8]], dtype=np.float32)

    _keyframe_face_expression(obj, expression)

    action = obj.data.shape_keys.animation_data.action
    fcurve = next(fc for fc in _iter_action_fcurves(action) if fc.data_path == 'key_blocks["Exp000"].value')
    points = {int(round(kp.co.x)): kp.co.y for kp in fcurve.keyframe_points}
    assert points[_FIRST_MOTION_BLENDER_FRAME - 1] == pytest.approx(0.0)
    assert points[_FIRST_MOTION_BLENDER_FRAME] == pytest.approx(0.8)


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_keyframe_face_expression_skips_redundant_middle_of_a_flat_stretch():
    """Mirrors `_keyframe_held_object_pose`'s own redundant-keyframe skip: a
    flat run's own first/last frame already reproduces the whole stretch
    under linear interpolation, so only those boundary frames need a real
    keyframe."""
    obj = _make_shape_key_mesh(["Exp000"])
    expression = np.array([[0.0], [1.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    _keyframe_face_expression(obj, expression)

    action = obj.data.shape_keys.animation_data.action
    fcurve = next(fc for fc in _iter_action_fcurves(action) if fc.data_path == 'key_blocks["Exp000"].value')
    keyframed_frames = {int(round(kp.co.x)) for kp in fcurve.keyframe_points}

    base = _FIRST_MOTION_BLENDER_FRAME
    assert keyframed_frames == {base - 1, base, base + 1, base + 3, base + 4}  # base + 2 (index 2) skipped


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_build_face_preview_blend_roundtrips_flame_axes_and_frame_offset(tmp_path):
    """Confirms the two things `face_preview.py`'s own docstring claims were
    verified empirically, not assumed: FLAME's Y-up axis lands on Blender's
    Z-up with no mirroring, and PC2 sample `i` plays back at Blender frame
    `i + _FIRST_MOTION_BLENDER_FRAME`."""
    import bpy

    n_frames = 3
    # A single triangle in FLAME's own (x, y=up, z) convention, vertex 0
    # rises along FLAME's Y axis by a known, frame-distinguishable amount.
    flame_verts = np.zeros((n_frames, 3, 3), dtype=np.float32)
    flame_verts[:, 1] = [0.1, 0.0, 0.0]
    flame_verts[:, 2] = [0.0, 0.1, 0.0]
    for i in range(n_frames):
        flame_verts[i, 0] = [0.0, 0.2 * i, 0.0]  # vertex 0: FLAME-Y = 0.0, 0.2, 0.4
    faces = np.array([[0, 1, 2]])

    blender_verts = flame_to_blender(flame_verts)
    template_path = tmp_path / "template.npz"
    _write_template_npz(template_path, blender_verts[0], faces)
    pc2_path = tmp_path / "animation.pc2"
    _write_pc2(pc2_path, blender_verts)

    blend_path = tmp_path / "FLAME_face_preview.blend"
    _build_face_preview_blend(bpy, template_path, pc2_path, n_frames, 59.917638984214136, blend_path)

    assert bpy.context.scene.render.fps == 60  # rounded; regression guard for the never-set-at-all bug
    assert bpy.context.scene.render.fps_base == 1.0

    obj = next(o for o in bpy.data.objects if o.type == "MESH")
    # matrix_world must stay identity: a previous version imported the
    # template via bpy.ops.wm.obj_import, whose own Y-up -> Z-up conversion
    # landed here as a 90-degree object rotation and silently compounded
    # with the conversion already baked into the vertex data.
    assert tuple(obj.matrix_world.to_euler()) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    for i in range(n_frames):
        bpy.context.scene.frame_set(i + _FIRST_MOTION_BLENDER_FRAME)
        depsgraph.update()
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        # Checked in WORLD space, not local: the object transform is exactly
        # where the mis-orientation hid before.
        world_co = eval_obj.matrix_world @ eval_mesh.vertices[0].co
        # FLAME's Y-up (vertex 0's own rising axis) lands on Blender's Z.
        assert world_co.z == pytest.approx(0.2 * i, abs=1e-5)
        eval_obj.to_mesh_clear()


def _write_fake_frames(frames_dir: Path, n: int, width: int = 40, height: int = 60) -> None:
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (width, height), color=(i % 256, 0, 0)).save(frames_dir / f"{i:06d}.jpg")


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_add_video_reference_plane_adds_a_correctly_aspected_textured_plane(tmp_path):
    import bpy

    n_frames = 3
    frames_dir = tmp_path / "input_frames"
    _write_fake_frames(frames_dir, n_frames, width=40, height=60)  # 2:3 aspect

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    _add_video_reference_plane(bpy, frames_dir, n_frames)

    obj = next(o for o in bpy.data.objects if o.type == "MESH")
    assert len(obj.data.vertices) == 4
    xs = {round(v.co.x, 6) for v in obj.data.vertices}
    ys = sorted({round(v.co.y, 6) for v in obj.data.vertices})
    zs = sorted({round(v.co.z, 6) for v in obj.data.vertices})
    px, py, pz = VIDEO_PLANE_POSITION_M
    assert xs == {px}  # a flat plane at the fixed X position
    assert zs == pytest.approx([pz - VIDEO_PLANE_HEIGHT_M / 2, pz + VIDEO_PLANE_HEIGHT_M / 2])
    # width/height matches the fake frames' own 40x60 (2:3) aspect ratio
    half_w = (VIDEO_PLANE_HEIGHT_M / 2) * (40 / 60)
    assert ys == pytest.approx([py - half_w, py + half_w])

    assert len(obj.data.materials) == 1
    image_nodes = [n for n in obj.data.materials[0].node_tree.nodes if n.type == "TEX_IMAGE"]
    assert len(image_nodes) == 1
    assert image_nodes[0].image.source == "SEQUENCE"
    assert image_nodes[0].image_user.frame_start == _FIRST_MOTION_BLENDER_FRAME
    assert image_nodes[0].image_user.frame_duration == n_frames


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_add_video_reference_plane_no_op_when_frames_dir_empty(tmp_path):
    import bpy

    frames_dir = tmp_path / "no_frames_here"
    frames_dir.mkdir()

    before = set(bpy.data.objects)
    _add_video_reference_plane(bpy, frames_dir, 3)
    assert set(bpy.data.objects) == before  # nothing added, no frames to show


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_build_landmark_preview_blend_builds_two_separated_objects(tmp_path):
    """`write_landmark_preview`'s real output, assembled end to end: two
    objects (raw, smoothed) exist, both with identity `matrix_world` (same
    hazard `_build_face_preview_blend`'s own test guards against), and the
    smoothed object's known-noisy trajectory reads visibly calmer at the
    real evaluated mesh, confirming the wiring actually smooths something
    rather than passing the raw data through twice."""
    import bpy

    n = 12
    v = 10
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, n)
    clean = np.zeros((n, v, 3), dtype=np.float32)
    clean[:, :, 0] = 50.0 * np.sin(2 * np.pi * t)[:, None]
    mp_landmarks = clean + rng.normal(0, 5.0, clean.shape).astype(np.float32)
    mp_valid = np.ones(n, dtype=bool)

    outputs = write_landmark_preview(mp_landmarks, mp_valid, smoothing_window=7, out_dir=tmp_path)
    blend_path = tmp_path / "landmark_preview.blend"
    _build_landmark_preview_blend(
        bpy, Path(outputs["landmark_preview_raw_template"]), Path(outputs["landmark_preview_raw_pc2"]),
        Path(outputs["landmark_preview_smoothed_template"]), Path(outputs["landmark_preview_smoothed_pc2"]),
        n, 24.0, blend_path,
    )

    assert bpy.context.scene.render.fps == 24

    objs = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
    assert set(objs) == {"landmark_preview_raw", "landmark_preview_smoothed"}
    for obj in objs.values():
        assert tuple(obj.matrix_world.to_euler()) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    def trajectory(obj):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        xs = []
        for i in range(n):
            bpy.context.scene.frame_set(i + _FIRST_MOTION_BLENDER_FRAME)
            depsgraph.update()
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            xs.append((eval_obj.matrix_world @ eval_mesh.vertices[0].co).x)
            eval_obj.to_mesh_clear()
        return np.array(xs)

    raw_jitter = np.abs(np.diff(trajectory(objs["landmark_preview_raw"]), n=2)).mean()
    smoothed_jitter = np.abs(np.diff(trajectory(objs["landmark_preview_smoothed"]), n=2)).mean()
    assert smoothed_jitter < raw_jitter


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
@pytest.mark.parametrize("bone_name", ["pelvis", "left_wrist"])
def test_fix_rotation_hemisphere_continuity_flips_opposite_sign_keyframe(bone_name):
    """Regression test for a real bug found on real exports, in two distinct
    forms this one fix covers: the pelvis across a keyframe *gap* (deleting a
    run leaves two surviving keyframes far enough apart that a large real
    rotation between them crosses the q/-q double-cover), and the wrists on
    *adjacent* keyframes with no gap at all (a wrist near 180 degrees has a
    near-zero `w`, where tiny numerical noise flips which representation the
    conversion picks). Blender's own per-component quaternion interpolation
    isn't hemisphere-aware either way, so it sweeps through a degenerate
    near-zero quaternion between the two. Constructs that exact scenario and
    confirms the fix flips the second keyframe back onto the first's own
    hemisphere. Parameterized over a non-pelvis bone specifically: the fix
    was originally pelvis-only, which is why the wrists kept glitching."""
    import math

    import bpy
    from mathutils import Quaternion, Vector

    arm_data = bpy.data.armatures.new("test_armature")
    arm_obj = bpy.data.objects.new("test_armature_obj", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones.new(bone_name)
    eb.head, eb.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.1)
    bpy.ops.object.mode_set(mode="OBJECT")

    pose_bone = arm_obj.pose.bones[bone_name]
    pose_bone.rotation_mode = "QUATERNION"

    axis = Vector((0.3, 0.7, 0.2)).normalized()
    quat_a = Quaternion(axis, math.radians(40))
    quat_a_negated = Quaternion((-quat_a.w, -quat_a.x, -quat_a.y, -quat_a.z))
    assert quat_a.dot(quat_a_negated) < 0  # confirms the two are on opposite hemispheres

    scene = bpy.context.scene
    scene.frame_set(1)
    pose_bone.rotation_quaternion = quat_a
    pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=1)

    scene.frame_set(5)
    pose_bone.rotation_quaternion = quat_a_negated  # same real rotation, opposite numeric sign
    pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=5)

    _fix_rotation_hemisphere_continuity(arm_obj)

    scene.frame_set(5)
    fixed = arm_obj.pose.bones[bone_name].rotation_quaternion
    assert fixed.dot(quat_a) > 0  # now on the same hemisphere as frame 1
    assert tuple(fixed) == pytest.approx(tuple(quat_a), abs=1e-6)


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_fix_rotation_hemisphere_continuity_leaves_already_continuous_keyframes_alone():
    """The fix must be a no-op on a bone whose keyframes already share a
    hemisphere, it rewrites keyframe values in place, so a false positive
    would silently corrupt good animation."""
    import math

    import bpy
    from mathutils import Quaternion, Vector

    arm_data = bpy.data.armatures.new("test_armature")
    arm_obj = bpy.data.objects.new("test_armature_obj", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones.new("right_wrist")
    eb.head, eb.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.1)
    bpy.ops.object.mode_set(mode="OBJECT")

    pose_bone = arm_obj.pose.bones["right_wrist"]
    pose_bone.rotation_mode = "QUATERNION"

    axis = Vector((0.1, 0.2, 0.97)).normalized()
    scene = bpy.context.scene
    expected = []
    for frame, degrees in ((1, 10.0), (2, 25.0), (3, 40.0)):
        quat = Quaternion(axis, math.radians(degrees))
        scene.frame_set(frame)
        pose_bone.rotation_quaternion = quat
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        expected.append((frame, tuple(quat)))

    _fix_rotation_hemisphere_continuity(arm_obj)

    for frame, original in expected:
        scene.frame_set(frame)
        assert tuple(arm_obj.pose.bones["right_wrist"].rotation_quaternion) == pytest.approx(original, abs=1e-6)


def _evaluated_mesh_world_positions(bpy, mesh_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    positions = [tuple(eval_obj.matrix_world @ v.co) for v in eval_mesh.vertices]
    eval_obj.to_mesh_clear()
    return positions


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_orient_bones_toward_children_does_not_move_the_skinned_mesh():
    """The correctness property `_orient_bones_toward_children` must have:
    the mesh is skinned to each bone via its *deform* matrix (`PoseBone.
    matrix @ Bone.matrix_local.inverted()`), not just its world *pose*
    matrix, so retailing a bone's rest transform without compensating the
    pose half would move the mesh even if bone matrices look unchanged.
    This test checks real evaluated mesh vertex positions, not bone
    matrices, which can't distinguish a correct fix from that bug.

    Builds a small synthetic armature with a branch point (exercises
    `_primary_child`'s own selection), a mix of animated/unanimated bones,
    and two leaf bones with mismatched original directions, to exercise the
    "continue the parent's direction" leaf behavior distinctly from a
    coincidental match. A real skinned mesh (one vertex per bone, 100%
    weighted) checks every vertex's own real, dependency-graph-evaluated
    world position stays unchanged at every already-keyframed frame.
    """
    import math as _math

    import bpy
    from mathutils import Quaternion

    arm_data = bpy.data.armatures.new("test_armature")
    arm_obj = bpy.data.objects.new("test_armature_obj", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    eb_root = arm_data.edit_bones.new("root")
    eb_root.head, eb_root.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.1)

    eb_spine = arm_data.edit_bones.new("spine")
    eb_spine.head, eb_spine.tail = (0.0, 0.0, 1.0), (0.0, 0.0, 1.1)
    eb_spine.parent = eb_root
    eb_spine.use_connect = False

    # A leaf (no children), with its own original tail deliberately NOT
    # pointing straight up (root's own future direction), so a redirect
    # to follow root's new direction is visible, not coincidental.
    eb_hip = arm_data.edit_bones.new("hip")
    eb_hip.head, eb_hip.tail = (0.3, 0.0, 0.9), (0.3, 0.2, 0.85)
    eb_hip.parent = eb_root
    eb_hip.use_connect = False

    eb_neck = arm_data.edit_bones.new("neck")  # sideways from spine, not straight up, see docstring
    eb_neck.head, eb_neck.tail = (0.6, 0.0, 1.0), (0.7, 0.0, 1.0)
    eb_neck.parent = eb_spine
    eb_neck.use_connect = False

    # A leaf two levels below an *animated* retailed bone, own original
    # tail again deliberately off of neck's own future (sideways) direction.
    eb_fingertip = arm_data.edit_bones.new("fingertip")
    eb_fingertip.head, eb_fingertip.tail = (0.7, 0.0, 1.0), (0.7, 0.15, 1.05)
    eb_fingertip.parent = eb_neck
    eb_fingertip.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    # Offset a fixed amount off of each bone's own head, not sitting exactly
    # on top of it: a vertex placed exactly at a bone's rotation origin can't
    # detect a rotational change in that bone's deform matrix, since rotating
    # a zero-offset point around itself doesn't move it, only a translation
    # difference would show up there, which isn't the property this test
    # needs to check.
    bone_names = ["root", "spine", "hip", "neck"]
    vertex_offset = (0.0, 0.2, 0.0)
    bone_heads = {
        "root": (0.0, 0.0, 0.0), "spine": (0.0, 0.0, 1.0), "hip": (0.3, 0.0, 0.9), "neck": (0.6, 0.0, 1.0),
    }
    vertex_positions = {
        name: tuple(h + o for h, o in zip(head, vertex_offset)) for name, head in bone_heads.items()
    }

    mesh_data = bpy.data.meshes.new("test_mesh")
    mesh_data.from_pydata([vertex_positions[name] for name in bone_names], [], [])
    mesh_data.update()
    mesh_obj = bpy.data.objects.new("test_mesh_obj", mesh_data)
    bpy.context.scene.collection.objects.link(mesh_obj)
    for i, name in enumerate(bone_names):
        mesh_obj.vertex_groups.new(name=name).add([i], 1.0, "REPLACE")
    modifier = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = arm_obj

    spine_pb = arm_obj.pose.bones["spine"]
    neck_pb = arm_obj.pose.bones["neck"]
    spine_pb.rotation_mode = "QUATERNION"
    neck_pb.rotation_mode = "QUATERNION"

    scene = bpy.context.scene
    scene.frame_set(1)
    spine_pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0), _math.radians(15))
    spine_pb.keyframe_insert(data_path="rotation_quaternion", frame=1)
    neck_pb.rotation_quaternion = Quaternion((0.0, 1.0, 0.0), _math.radians(-10))
    neck_pb.keyframe_insert(data_path="rotation_quaternion", frame=1)

    scene.frame_set(5)
    spine_pb.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), _math.radians(-20))
    spine_pb.keyframe_insert(data_path="rotation_quaternion", frame=5)
    neck_pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0), _math.radians(25))
    neck_pb.keyframe_insert(data_path="rotation_quaternion", frame=5)

    mesh_before = {}
    # Include interpolated frames too. The alignment pass rewrites the pose
    # action in bulk, so preserving only the source keyframes would miss a
    # regression in the generated Bezier handles between them.
    for frame in range(1, 6):
        scene.frame_set(frame)
        mesh_before[frame] = _evaluated_mesh_world_positions(bpy, mesh_obj)

    _orient_bones_toward_children(bpy, arm_obj)

    # The whole point: bones with children got retailed toward the right one.
    assert tuple(arm_data.bones["root"].tail_local) == pytest.approx(tuple(arm_data.bones["spine"].head_local), abs=1e-6)
    assert tuple(arm_data.bones["spine"].tail_local) == pytest.approx(tuple(arm_data.bones["neck"].head_local), abs=1e-6)
    assert tuple(arm_data.bones["neck"].tail_local) == pytest.approx(tuple(arm_data.bones["fingertip"].head_local), abs=1e-6)
    # Leaves (no children) continue their own parent's, now-corrected,
    # direction instead of the decorative default, preserving their own
    # original bone length (only the direction changes).
    hip_length = (0.2 ** 2 + 0.05 ** 2) ** 0.5  # |original hip tail - hip head|
    assert tuple(arm_data.bones["hip"].tail_local) == pytest.approx((0.3, 0.0, 0.9 + hip_length), abs=1e-5)
    fingertip_length = (0.15 ** 2 + 0.05 ** 2) ** 0.5  # |original fingertip tail - fingertip head|
    assert tuple(arm_data.bones["fingertip"].tail_local) == pytest.approx((0.7 + fingertip_length, 0.0, 1.0), abs=1e-5)

    # The property that actually matters: the mesh itself hasn't moved.
    for frame in range(1, 6):
        scene.frame_set(frame)
        after = _evaluated_mesh_world_positions(bpy, mesh_obj)
        for name, before_pos, after_pos in zip(bone_names, mesh_before[frame], after):
            assert after_pos == pytest.approx(before_pos, abs=1e-5), (name, frame)
