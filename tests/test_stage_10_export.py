"""Stage 10 (export) tests: fast pure-numpy tests of the AMASS-preparation
logic (always run, no bpy needed) plus a real bpy smoke test gated behind
`bpy` importability, since this stage only ever actually runs in the
separate `export` pixi environment. Run the real one with:
`pixi run -e export python -m pytest tests/test_stage_10_export.py -v`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline.helpers.bvh_export import CAMERA_TO_BVH_ROOT_ROTATION
from pipeline.progress_tracker import RunInput, RunRecord, SceneInfo, StageName, StageRecord
from pipeline.stages.stage_10_export import (
    OUTPUT_BLEND,
    _FIRST_MOTION_BLENDER_FRAME,
    _REST_POSE_YAW_RADIANS,
    _fix_rotation_hemisphere_continuity,
    _iter_action_fcurves,
    _lowest_foot_z,
    _object_pose_to_blender_world,
    _orient_bones_toward_children,
    _prepend_rest_pose_frame,
    _write_body_amass,
    run,
)
from tests.conftest import make_run_input

try:
    import bpy  # noqa: F401
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


if HAS_BPY:
    @pytest.fixture(autouse=True)
    def _isolated_bpy_scene():
        """A real, order-dependent bug was found running this file's own
        tests together: a prior test's large (300-frame) build left bpy in a
        state that corrupted a *later* test's otherwise-correct output, even
        though `run()` already clears the scene at its own start (something
        more subtle than orphaned objects/actions -- never fully root-caused,
        and not worth chasing further since it's purely a test-harness
        artifact: every real invocation of this stage is its own fresh
        `pixi run -e export python -m ...` subprocess, so bpy state never
        actually carries over between runs in production). Clearing before
        *and* after each test, rather than trusting `run()`'s own single
        clear at the start, is what actually made the suite order-independent.
        """
        from pipeline.stages.stage_10_export import _clear_scene
        _clear_scene(bpy)
        yield
        _clear_scene(bpy)

N_FRAMES = 5


def _fake_motion() -> dict:
    rng = np.random.default_rng(0)
    betas = np.tile(rng.normal(size=(1, 10)).astype(np.float32), (N_FRAMES, 1))
    return {
        "global_orient": rng.normal(size=(N_FRAMES, 3)).astype(np.float32),
        "body_pose": rng.normal(size=(N_FRAMES, 63)).astype(np.float32),
        "betas": betas,
        "transl": rng.normal(size=(N_FRAMES, 3)).astype(np.float32),
        "left_hand_pose": rng.normal(size=(N_FRAMES, 45)).astype(np.float32),
        "right_hand_pose": rng.normal(size=(N_FRAMES, 45)).astype(np.float32),
        "root_motion_unreliable": np.zeros(N_FRAMES, dtype=bool),
    }


def test_prepend_rest_pose_frame_adds_one_all_zero_frame():
    global_orient = np.full((N_FRAMES, 3), 2.0, dtype=np.float32)
    body_pose = np.full((N_FRAMES, 63), 3.0, dtype=np.float32)
    transl = np.arange(N_FRAMES * 3, dtype=np.float32).reshape(N_FRAMES, 3)
    left_hand_pose = np.full((N_FRAMES, 45), 4.0, dtype=np.float32)
    right_hand_pose = np.full((N_FRAMES, 45), 5.0, dtype=np.float32)

    go, bp, tr, lh, rh = _prepend_rest_pose_frame(global_orient, body_pose, transl, left_hand_pose, right_hand_pose)

    assert go.shape == (N_FRAMES + 1, 3)
    # Root orientation is yawed (see _REST_POSE_YAW_RADIANS), not literal
    # zero -- everything else about the rest frame (body/hand pose) stays a
    # true T-pose.
    assert np.allclose(go[0], [0.0, _REST_POSE_YAW_RADIANS, 0.0]) and np.allclose(go[1:], global_orient)
    assert np.allclose(bp[0], 0.0) and np.allclose(bp[1:], body_pose)
    assert np.allclose(lh[0], 0.0) and np.allclose(lh[1:], left_hand_pose)
    assert np.allclose(rh[0], 0.0) and np.allclose(rh[1:], right_hand_pose)
    # The rest frame sits at the real motion's own first-frame position, not
    # the origin -- only the pose should jump at the 0->1 boundary.
    assert np.allclose(tr[0], transl[0])
    assert np.allclose(tr[1:], transl)


def test_write_body_amass_embeds_real_hand_pose(tmp_path):
    motion = _fake_motion()
    out_path = tmp_path / "body.npz"

    _write_body_amass(motion, fps=30.0, out_path=out_path)

    with np.load(out_path) as data:
        assert data["poses"].shape == (N_FRAMES + 1, 55 * 3)
        hand_start = 3 + 63 + 9  # global_orient + body_pose + jaw/eyes
        # Frame 0 is the prepended rest-pose frame -- real hand data starts at 1.
        assert np.allclose(data["poses"][0, hand_start:hand_start + 90], 0.0)
        assert np.allclose(data["poses"][1:, hand_start:hand_start + 45], motion["left_hand_pose"])
        assert np.allclose(data["poses"][1:, hand_start + 45:hand_start + 90], motion["right_hand_pose"])
        expected_trans = motion["transl"] @ CAMERA_TO_BVH_ROOT_ROTATION.T
        assert np.allclose(data["trans"][1:], expected_trans, atol=1e-5)
        assert np.allclose(data["trans"][0], expected_trans[0], atol=1e-5)


def test_write_body_amass_applies_the_floor_offset_to_every_frame_including_rest(tmp_path):
    motion = _fake_motion()
    out_path = tmp_path / "body.npz"

    _write_body_amass(motion, fps=30.0, out_path=out_path, floor_offset=5.0)

    with np.load(out_path) as data:
        baseline = motion["transl"] @ CAMERA_TO_BVH_ROOT_ROTATION.T
        assert np.allclose(data["trans"][1:, 1], baseline[:, 1] + 5.0, atol=1e-5)
        assert np.allclose(data["trans"][0, 1], baseline[0, 1] + 5.0, atol=1e-5)


def test_write_body_amass_pools_betas_from_the_first_frame(tmp_path):
    motion = _fake_motion()
    motion["betas"] = motion["betas"].copy()
    motion["betas"][1:] = 0.0  # only frame 0 carries the "real" repeated value

    out_path = tmp_path / "body.npz"
    _write_body_amass(motion, fps=30.0, out_path=out_path)

    with np.load(out_path) as data:
        assert np.allclose(data["betas"], motion["betas"][0])


def test_object_pose_to_blender_world_pivots_around_pelvis_rest_not_the_origin():
    """Regression test for a real bug: SMPL-X's `global_orient` rotates the
    body about the pelvis's own rest position, not the world origin (a
    skeleton joint gets this for free via forward kinematics; a standalone
    point like the object's own `center` does not). Ignoring this placed a
    real clip's object ~0.7m away from the hand it was fit next to, instead
    of the ~0.2m the raw (untransformed) data itself shows.

    Picks `center` so `center - pelvis_rest` is a simple "1 unit up in
    camera space" vector -- reusing the same known mapping
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


def _make_runRecord(
    tmp_path, motion_npz_path, object_shape=None, pelvis_rest=None, object_pose_npz_path=None,
    attachment_events=None,
) -> RunRecord:
    """A real run always has a `STAGE_6_ALIGN_SCENE_SCALE` entry (a hard DAG
    dependency of `export`) -- its own `scene_scale.json` always exists,
    but only carries `object_shape`/`pelvis_rest_incam` when an object was
    actually tracked. Mirrored here rather than omitting the stage entirely
    for the human-only case, which `run()` doesn't expect. `object_pose_npz_path`
    (stage 8's real per-frame pose) and `attachment_events` (stage 8's own
    `attachment_events.json`, defaults to no events -- see
    `hoi_object_pose.compute_object_pose_sequence`'s own docstring for what
    each event carries) are only read by `run()` when an object was tracked,
    so the human-only tests never need to pass either."""
    run_input = make_run_input()
    scene_scale_path = tmp_path / "scene_scale.json"
    scene_scale_json = {}
    stage_6_outputs = {"scene_scale": str(scene_scale_path)}
    if object_shape is not None:
        object_shape_path = tmp_path / "object_shape.json"
        object_shape_path.write_text(json.dumps(object_shape))
        scene_scale_json["pelvis_rest_incam"] = pelvis_rest.tolist()
        stage_6_outputs["object_shape"] = str(object_shape_path)
    scene_scale_path.write_text(json.dumps(scene_scale_json))

    stages = {
        StageName.STAGE_5_RETARGET_HANDS.value: StageRecord(
            outputs={"retarget_motion_npz": str(motion_npz_path)},
        ),
        StageName.STAGE_6_ALIGN_SCENE_SCALE.value: StageRecord(outputs=stage_6_outputs),
    }
    if object_pose_npz_path is not None:
        attachment_events_path = tmp_path / "attachment_events.json"
        attachment_events_path.write_text(json.dumps(attachment_events or []))
        stages[StageName.STAGE_8_OPTIMIZE_HOI.value] = StageRecord(
            outputs={
                "object_pose_npz": str(object_pose_npz_path),
                "attachment_events": str(attachment_events_path),
            },
        )
    return RunRecord(
        run_id="test",
        progress_dir=str(tmp_path),
        input=run_input,
        scene=SceneInfo(fps=30.0),
        stages=stages,
    )


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_run_produces_a_real_blend_file(tmp_path):
    motion_npz_path = tmp_path / "retargeted_motion.npz"
    np.savez(motion_npz_path, **_fake_motion())
    runRecord = _make_runRecord(tmp_path, motion_npz_path)

    outputs = run(runRecord)

    output_path = tmp_path / "output.blend"
    assert outputs[OUTPUT_BLEND] == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert runRecord.outputs.final_blend == str(output_path)


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_run_grounds_the_lowest_foot_position_near_zero(tmp_path):
    """Regression test for the real bug the user found: incam space has no
    floor reference, so without the offset the character sits wherever
    GVHMR's raw numbers happened to put it (~1.7m underground on a real
    clip). After `run()`, the lowest point any foot/ankle bone reaches
    across the whole clip should land at (approximately) Z=0.

    Uses 300 frames (not the usual N_FRAMES=5) deliberately -- a real bug
    was found and fixed at this exact frame count boundary: Blender's own
    scene.frame_end defaults to 250 and the addon never extends it to match
    a longer animation, so trusting it (instead of the armature's own
    action.frame_range, what `_lowest_foot_z` actually uses) would silently
    stop scanning partway through a clip like this one.
    """
    import bpy

    n_frames = 300
    rng = np.random.default_rng(1)
    long_motion = {
        "global_orient": rng.normal(size=(n_frames, 3)).astype(np.float32),
        "body_pose": rng.normal(size=(n_frames, 63)).astype(np.float32),
        "betas": np.tile(rng.normal(size=(1, 10)).astype(np.float32), (n_frames, 1)),
        "transl": rng.normal(size=(n_frames, 3)).astype(np.float32),
        "left_hand_pose": rng.normal(size=(n_frames, 45)).astype(np.float32),
        "right_hand_pose": rng.normal(size=(n_frames, 45)).astype(np.float32),
        "root_motion_unreliable": np.zeros(n_frames, dtype=bool),
    }
    motion_npz_path = tmp_path / "retargeted_motion.npz"
    np.savez(motion_npz_path, **long_motion)
    runRecord = _make_runRecord(tmp_path, motion_npz_path)

    run(runRecord)

    # Saving doesn't touch the live scene, so the just-built armature is
    # still right here -- no need to reopen the file to check it. A separate
    # test below (`test_run_with_an_object_shape...`) covers a real
    # save->reopen round trip.
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    assert abs(_lowest_foot_z(armature)) < 0.01


def _bone_keyframed_frames(action, bone_name: str) -> set[int]:
    frames: set[int] = set()
    for fcurve in _iter_action_fcurves(action):
        if fcurve.data_path.startswith(f'pose.bones["{bone_name}"]'):
            frames.update(int(round(kp.co.x)) for kp in fcurve.keyframe_points)
    return frames


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_run_deletes_pelvis_keyframes_at_unreliable_root_motion_frames(tmp_path):
    """Rather than exporting stage 2's bridged/interpolated root numbers as
    real keyframes, delete the pelvis bone's own keyframes at those frames
    entirely, so Blender's own curve interpolation bridges the resulting gap
    live -- the same fix a Blender artist would apply by hand (deleting the
    pelvis bone's keyframes and letting it interpolate). Every other bone
    must be completely unaffected."""
    import bpy

    n_frames = 10
    rng = np.random.default_rng(3)
    root_motion_unreliable = np.array([False, False, False, True, True, True, False, False, False, False])
    motion = {
        "global_orient": rng.normal(size=(n_frames, 3)).astype(np.float32) * 0.1,
        "body_pose": rng.normal(size=(n_frames, 63)).astype(np.float32) * 0.1,
        "betas": np.tile(rng.normal(size=(1, 10)).astype(np.float32), (n_frames, 1)),
        "transl": rng.normal(size=(n_frames, 3)).astype(np.float32) * 0.1,
        "left_hand_pose": np.zeros((n_frames, 45), dtype=np.float32),
        "right_hand_pose": np.zeros((n_frames, 45), dtype=np.float32),
        "root_motion_unreliable": root_motion_unreliable,
    }
    motion_npz_path = tmp_path / "retargeted_motion.npz"
    np.savez(motion_npz_path, **motion)
    runRecord = _make_runRecord(tmp_path, motion_npz_path)

    run(runRecord)

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    action = armature.animation_data.action
    pelvis_frames = _bone_keyframed_frames(action, "pelvis")
    other_frames = _bone_keyframed_frames(action, "spine1")  # any non-root bone

    for source_idx in range(n_frames):
        blender_frame = source_idx + _FIRST_MOTION_BLENDER_FRAME
        if root_motion_unreliable[source_idx]:
            assert blender_frame not in pelvis_frames
        else:
            assert blender_frame in pelvis_frames
        # A completely different bone is untouched -- every frame still keyframed.
        assert blender_frame in other_frames
    assert 1 in pelvis_frames  # the prepended rest-pose frame itself, never flagged


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
    hemisphere -- it rewrites keyframe values in place, so a false positive
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


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_run_with_an_object_shape_combines_body_and_object_into_one_blend_file(tmp_path):
    """When `align_scene_scale` tracked an object, the object is added
    directly into the same live scene as the body and the whole scene is
    saved together in one `output.blend` (see module docstring for why this
    doesn't need the addon's own body-only export operator at all). Held
    frames get baked world-space keyframes; an attachment event instead gets
    a live Child Of constraint targeting the attaching bone (see module
    docstring for why: this is what lets the object stay correctly attached
    if the skeleton's own animation is later retargeted onto a different
    rig). This test covers both: a flat held value before/after one
    attachment event on the right wrist.

    Also a regression test for a real bug: leftover orphaned data-blocks
    from this stage's own two-pass floor-offset build resurfacing in the
    saved file in place of the corrected body, silently undoing the
    floor-grounding fix (`_clear_scene`'s own `orphans_purge` call is what
    this asserts against).

    Runs `run()` in a genuinely separate process rather than in-process like
    this file's other `run()` tests: a *second*, distinct order-dependent bpy
    quirk was found chaining this test after `test_run_grounds_the_lowest_foot_
    position_near_zero` in the same pytest session (a large, 300-frame build
    left something -- never fully root-caused, and not orphaned objects/
    actions/`scene.frame_end`, all of which were checked directly -- that
    corrupted this test's own otherwise-correct output). Not worth chasing
    further: it's purely a test-harness artifact, since every real invocation
    of this stage is its own fresh subprocess (`pixi run -e export python
    -m pipeline.stages.stage_10_export`) that never shares bpy state with a
    prior run to begin with -- so making *this* test match that same
    real-world shape sidesteps the whole class of bug rather than chasing an
    uncharacterized one.
    """
    n_frames = 60
    rng = np.random.default_rng(2)
    motion = {
        "global_orient": rng.normal(size=(n_frames, 3)).astype(np.float32),
        "body_pose": rng.normal(size=(n_frames, 63)).astype(np.float32),
        "betas": np.tile(rng.normal(size=(1, 10)).astype(np.float32), (n_frames, 1)),
        "transl": rng.normal(size=(n_frames, 3)).astype(np.float32),
        "left_hand_pose": np.zeros((n_frames, 45), dtype=np.float32),
        "right_hand_pose": np.zeros((n_frames, 45), dtype=np.float32),
        "root_motion_unreliable": np.zeros(n_frames, dtype=bool),
    }
    motion_npz_path = tmp_path / "retargeted_motion.npz"
    np.savez(motion_npz_path, **motion)

    object_kind = "box"
    object_shape = {"kind": object_kind, "half_extents": [0.05, 0.06, 0.04]}
    pelvis_rest = np.array([0.01, -0.3, 0.02])

    # A flat held value, far from the body -- stage 8's own real contract
    # for every non-attached frame (see hoi_object_pose.py's own module
    # docstring). The attachment event below (right wrist, joint_idx=21)
    # covers frames 20-40; these array values are never actually read for
    # that range (see `_held_frame_mask`) -- left flat here specifically so
    # that if stage 10 ever regressed to reading them for attached frames
    # too, the check below (which compares against the *live* wrist bone,
    # not this array) would catch it by disagreeing wildly.
    held_value = np.array([5.0, 5.0, 5.0])
    translations = np.tile(held_value, (n_frames, 1))
    rotations = np.tile(np.eye(3), (n_frames, 1, 1))
    object_pose_npz_path = tmp_path / "object_pose.npz"
    np.savez(
        object_pose_npz_path,
        translation=translations,
        rotation=rotations,
        is_low_confidence=np.zeros(n_frames, dtype=bool),
    )

    event = {
        "region": "right_hand", "joint_idx": 21,  # right_wrist
        "start_frame": 20, "end_frame": 40, "snap_frame": 20,
        "ref_center": [0.05, 0.02, 1.3], "ref_rotation": np.eye(3).tolist(),
        "is_low_confidence": False,
    }

    runRecord = _make_runRecord(
        tmp_path, motion_npz_path, object_shape=object_shape, pelvis_rest=pelvis_rest,
        object_pose_npz_path=object_pose_npz_path, attachment_events=[event],
    )
    runRecord.save()

    # No `check=True`: bpy has a reproducible access-violation exit code on
    # interpreter teardown *after* the file is fully written (a known
    # standalone-bpy quirk, already documented for this stage's own bpy
    # smoke test) -- verify success via the output artifact instead.
    subprocess.run(
        [
            sys.executable, "-c",
            "from pipeline.progress_tracker import RunRecord\n"
            "from pipeline.stages.stage_10_export import run\n"
            f"run(RunRecord.load(r{str(tmp_path)!r}))\n",
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    output_path = tmp_path / "output.blend"
    assert output_path.exists()

    import bpy

    from pipeline.stages.stage_10_export import OBJECT_MESH_PREFIX

    # A real save->reopen round trip (not just reading the live in-memory
    # scene, which the subprocess above doesn't share with this process
    # anyway) -- confirms the saved file itself, not just Blender's live
    # state, is correct.
    bpy.ops.wm.open_mainfile(filepath=str(output_path))

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    object_mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.name == OBJECT_MESH_PREFIX + object_kind)
    wrist_bone = armature.pose.bones["right_wrist"]
    scene = bpy.context.scene

    assert abs(_lowest_foot_z(armature)) < 0.01
    floor_offset = -_lowest_foot_z(armature)

    def blender_frame(raw_frame: int) -> int:
        return raw_frame + _FIRST_MOTION_BLENDER_FRAME

    # `floor_offset` re-derived this way is near-zero (grounding is already
    # correct post-reload), NOT the original, generally-nonzero value `run()`
    # itself used internally when it keyframed the object -- re-deriving the
    # true value isn't possible from the saved file alone. floor_offset only
    # ever shifts Blender's own Z (see `_object_pose_to_blender_world`), so
    # the checks below only compare the two axes it never touches: X and Y
    # (same reasoning this test used before the constraint-based redesign).
    expected_held, _ = _object_pose_to_blender_world(held_value, np.eye(3), pelvis_rest, floor_offset)

    # Held, before the event.
    scene.frame_set(blender_frame(0))
    assert np.allclose(tuple(object_mesh.location)[:2], expected_held[:2], atol=1e-4)

    # At the snap frame itself, the object's live (constrained) position
    # exactly reproduces the raw reference measurement -- true by
    # construction of how the constraint's own inverse_matrix is derived
    # (see `_add_attachment_constraint`), a direct check that the derivation
    # is wired correctly end to end, not just internally self-consistent.
    scene.frame_set(blender_frame(event["snap_frame"]))
    expected_snap_loc, _ = _object_pose_to_blender_world(
        np.array(event["ref_center"]), np.array(event["ref_rotation"]), pelvis_rest, floor_offset,
    )
    assert np.allclose(tuple(object_mesh.matrix_world.translation)[:2], expected_snap_loc[:2], atol=1e-4)

    # Attached: the object's own position relative to the wrist bone stays
    # perfectly rigid across several frames within the event -- only
    # possible if the constraint is live-tracking the bone's own real
    # motion every frame (a frozen/baked value would only coincidentally
    # match the bone's position at the one frame it was captured, not
    # consistently across several different ones). This is also the
    # concrete property that makes the object survive a retarget onto a
    # different rig: the *relationship* to the bone is what's preserved,
    # not an absolute recording.
    def relative_offset(raw_frame: int) -> np.ndarray:
        scene.frame_set(blender_frame(raw_frame))
        bone_world = armature.matrix_world @ wrist_bone.matrix
        return np.array(bone_world.inverted() @ object_mesh.matrix_world)

    offsets = [relative_offset(f) for f in [21, 25, 30, 35, 40]]
    for later in offsets[1:]:
        assert np.allclose(later, offsets[0], atol=1e-4)

    # Held again after the event -- back to the same flat value (proves the
    # constraint's own influence actually turns back off, and the object's
    # own base transform correctly resumes rather than staying wherever the
    # constraint last left it).
    scene.frame_set(blender_frame(event["end_frame"] + 1))
    assert np.allclose(tuple(object_mesh.location)[:2], expected_held[:2], atol=1e-4)

    assert np.allclose(tuple(object_mesh.dimensions), (0.1, 0.12, 0.08), atol=1e-4)


def _evaluated_mesh_world_positions(bpy, mesh_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    positions = [tuple(eval_obj.matrix_world @ v.co) for v in eval_mesh.vertices]
    eval_obj.to_mesh_clear()
    return positions


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_orient_bones_toward_children_does_not_move_the_skinned_mesh():
    """The actual correctness property `_orient_bones_toward_children` must
    have: the mesh is skinned to each bone via that bone's own *deform*
    matrix (`PoseBone.matrix @ Bone.matrix_local.inverted()`, the formula
    the Armature modifier itself applies to every weighted vertex) -- not
    just its world *pose* matrix. Retailing a bone changes its own rest
    transform (`Bone.matrix_local`); holding the bone's own world pose fixed
    at its pre-retail value leaves the deform matrix changed anyway, since
    the rest half of that formula moved out from under it and nothing
    compensates for that half. This test checks the deform matrix's own
    real effect directly -- actual evaluated mesh vertex positions -- not
    bone matrices, which can't distinguish a correct fix from this bug.

    Builds a small synthetic armature with a branch point (exercises
    `_primary_child`'s own selection: "spine" continues further up than its
    sibling "hip", so it should win), a mix of animated/unanimated bones
    ("root" is retailed but never animated, like the real addon's own
    always-static root bone), and two leaf bones with deliberately mismatched
    original directions -- "hip" (child of unanimated "root") and
    "fingertip" (child of animated "neck", itself two levels deep) -- to
    exercise the "continue the parent's own direction" leaf behavior
    distinctly from a coincidental match. "neck" sits sideways from "spine",
    not directly above it -- deliberately, so retailing spine actually
    changes its own rest rotation (from the decorative default's straight up
    to sideways toward neck), not just its length: a geometry where old and
    new tail directions happen to coincide can't tell a correct fix apart
    from the deform-matrix bug this guards against, since there'd be no real
    rest-rotation delta either way. A real skinned mesh (one vertex per bone,
    100% weighted, no shared influence -- isolates each bone's own deform
    contribution) checks every vertex's own real, dependency-graph-evaluated
    world position is unchanged at every already-keyframed frame -- the
    literal thing a user looking at the export would see move if this were
    wrong.
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
    # pointing straight up (root's own future direction) -- so a redirect
    # to follow root's new direction is visible, not coincidental.
    eb_hip = arm_data.edit_bones.new("hip")
    eb_hip.head, eb_hip.tail = (0.3, 0.0, 0.9), (0.3, 0.2, 0.85)
    eb_hip.parent = eb_root
    eb_hip.use_connect = False

    eb_neck = arm_data.edit_bones.new("neck")  # sideways from spine, not straight up -- see docstring
    eb_neck.head, eb_neck.tail = (0.6, 0.0, 1.0), (0.7, 0.0, 1.0)
    eb_neck.parent = eb_spine
    eb_neck.use_connect = False

    # A leaf two levels below an *animated* retailed bone -- own original
    # tail again deliberately off of neck's own future (sideways) direction.
    eb_fingertip = arm_data.edit_bones.new("fingertip")
    eb_fingertip.head, eb_fingertip.tail = (0.7, 0.0, 1.0), (0.7, 0.15, 1.05)
    eb_fingertip.parent = eb_neck
    eb_fingertip.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    # Offset a fixed amount off of each bone's own head, not sitting exactly
    # on top of it: a vertex placed exactly at a bone's rotation origin can't
    # detect a rotational change in that bone's deform matrix, since rotating
    # a zero-offset point around itself doesn't move it -- only a translation
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
    for frame in (1, 5):
        scene.frame_set(frame)
        mesh_before[frame] = _evaluated_mesh_world_positions(bpy, mesh_obj)

    _orient_bones_toward_children(bpy, arm_obj)

    # The whole point: bones with children got retailed toward the right one.
    assert tuple(arm_data.bones["root"].tail_local) == pytest.approx(tuple(arm_data.bones["spine"].head_local), abs=1e-6)
    assert tuple(arm_data.bones["spine"].tail_local) == pytest.approx(tuple(arm_data.bones["neck"].head_local), abs=1e-6)
    assert tuple(arm_data.bones["neck"].tail_local) == pytest.approx(tuple(arm_data.bones["fingertip"].head_local), abs=1e-6)
    # Leaves (no children) continue their own parent's -- now-corrected --
    # direction instead of the decorative default, preserving their own
    # original bone length (only the direction changes).
    hip_length = (0.2 ** 2 + 0.05 ** 2) ** 0.5  # |original hip tail - hip head|
    assert tuple(arm_data.bones["hip"].tail_local) == pytest.approx((0.3, 0.0, 0.9 + hip_length), abs=1e-5)
    fingertip_length = (0.15 ** 2 + 0.05 ** 2) ** 0.5  # |original fingertip tail - fingertip head|
    assert tuple(arm_data.bones["fingertip"].tail_local) == pytest.approx((0.7 + fingertip_length, 0.0, 1.0), abs=1e-5)

    # The property that actually matters: the mesh itself hasn't moved.
    for frame in (1, 5):
        scene.frame_set(frame)
        after = _evaluated_mesh_world_positions(bpy, mesh_obj)
        for name, before_pos, after_pos in zip(bone_names, mesh_before[frame], after):
            assert after_pos == pytest.approx(before_pos, abs=1e-5), (name, frame)
