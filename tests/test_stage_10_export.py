"""Stage 10 (export) orchestration tests: fast pure-numpy tests of the
AMASS-preparation logic (always run, no bpy needed) plus `run()` end-to-end
integration tests, gated behind `bpy` importability (`tests.conftest.
HAS_BPY`) since this stage only ever actually runs in the separate `export`
pixi environment. Run the real ones with:
`pixi run -e export python -m pytest tests/test_stage_10_export.py -v`.

Unit tests for the individual bpy-touching helpers (`pipeline/bpy/`) live
in `tests/test_blender.py` instead.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline.helpers.bvh_export import CAMERA_TO_BVH_ROOT_ROTATION
from pipeline.progress_tracker import RunRecord, SceneInfo, StageName, StageRecord
from pipeline.bpy.blender_armature import _lowest_foot_z
from pipeline.bpy.blender_constants import _FIRST_MOTION_BLENDER_FRAME
from pipeline.bpy.blender_object_attachment import OBJECT_MESH_PREFIX, _object_pose_to_blender_world
from pipeline.bpy.blender_preview import VIDEO_PLANE_HEIGHT_M, VIDEO_PLANE_POSITION_M
from pipeline.bpy.blender_scene import _iter_action_fcurves
from pipeline.stages.stage_10_export import OUTPUT_BLEND, _REST_POSE_YAW_RADIANS, _prepend_rest_pose_frame, _write_body_amass, run
from tests.conftest import FLAME_MODEL_PATH, HAS_BPY, SMPLX_MODEL_PATH, make_run_input

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
    # zero, everything else about the rest frame (body/hand pose) stays a
    # true T-pose.
    assert np.allclose(go[0], [0.0, _REST_POSE_YAW_RADIANS, 0.0]) and np.allclose(go[1:], global_orient)
    assert np.allclose(bp[0], 0.0) and np.allclose(bp[1:], body_pose)
    assert np.allclose(lh[0], 0.0) and np.allclose(lh[1:], left_hand_pose)
    assert np.allclose(rh[0], 0.0) and np.allclose(rh[1:], right_hand_pose)
    # The rest frame sits at the real motion's own first-frame position, not
    # the origin, only the pose should jump at the 0->1 boundary.
    assert np.allclose(tr[0], transl[0])
    assert np.allclose(tr[1:], transl)


def test_write_body_amass_embeds_real_hand_pose(tmp_path):
    motion = _fake_motion()
    out_path = tmp_path / "body.npz"

    _write_body_amass(motion, fps=30.0, out_path=out_path)

    with np.load(out_path) as data:
        assert data["poses"].shape == (N_FRAMES + 1, 55 * 3)
        hand_start = 3 + 63 + 9  # global_orient + body_pose + jaw/eyes
        # Frame 0 is the prepended rest-pose frame, real hand data starts at 1.
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


def _make_runRecord(
    tmp_path, motion_npz_path, object_shape=None, pelvis_rest=None, object_pose_npz_path=None,
    attachment_events=None, face_motion_npz_path=None, frames_dir=None,
) -> RunRecord:
    """A real run always has a `STAGE_6_ALIGN_SCENE_SCALE` and
    `STAGE_9_CAPTURE_FACE` entry (both hard DAG dependencies of `export`),
    `scene_scale.json` always exists but only carries `object_shape`/
    `pelvis_rest_incam` when an object was actually tracked, and
    `face_motion` is only present when face capture wasn't skipped. Both
    mirrored here rather than omitted, which `run()` doesn't expect.
    `object_pose_npz_path` (stage 8's real per-frame pose) and
    `attachment_events` (stage 8's own `attachment_events.json`, defaults to
    no events, see `hoi_object_pose.compute_object_pose_sequence`'s own
    docstring for what each event carries) are only read by `run()` when an
    object was tracked, so the human-only tests never need to pass either."""
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

    stage_9_outputs = {}
    if face_motion_npz_path is not None:
        stage_9_outputs["face_motion"] = str(face_motion_npz_path)

    stages = {
        StageName.STAGE_5_RETARGET_HANDS.value: StageRecord(
            outputs={"retarget_motion_npz": str(motion_npz_path)},
        ),
        StageName.STAGE_6_ALIGN_SCENE_SCALE.value: StageRecord(outputs=stage_6_outputs),
        StageName.STAGE_9_CAPTURE_FACE.value: StageRecord(outputs=stage_9_outputs),
    }
    if frames_dir is not None:
        stages[StageName.STAGE_0_INGEST_VIDEO.value] = StageRecord(outputs={"frames_dir": str(frames_dir)})
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
def test_run_opens_and_renders_from_the_first_motion_frame(tmp_path):
    """Frame 1 retains the T-pose, but output.blend starts at frame 2."""
    import bpy

    motion_npz_path = tmp_path / "retargeted_motion.npz"
    np.savez(motion_npz_path, **_fake_motion())
    runRecord = _make_runRecord(tmp_path, motion_npz_path)

    run(runRecord)

    bpy.ops.wm.open_mainfile(filepath=str(tmp_path / "output.blend"))
    scene = bpy.context.scene
    assert scene.frame_start == _FIRST_MOTION_BLENDER_FRAME
    assert scene.frame_current == _FIRST_MOTION_BLENDER_FRAME


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_run_embeds_the_source_video_plane_in_the_final_blend(tmp_path):
    """The final deliverable uses the exact same video-reference object as
    the face previews, and it is saved with the final body scene rather than
    merely lingering in the live Blender session after export."""
    import bpy
    from PIL import Image

    motion_npz_path = tmp_path / "retargeted_motion.npz"
    np.savez(motion_npz_path, **_fake_motion())
    frames_dir = tmp_path / "input_frames"
    frames_dir.mkdir()
    for i in range(N_FRAMES):
        Image.new("RGB", (40, 60), color=(i, 0, 0)).save(frames_dir / f"{i:06d}.jpg")

    runRecord = _make_runRecord(tmp_path, motion_npz_path, frames_dir=frames_dir)
    run(runRecord)

    bpy.ops.wm.open_mainfile(filepath=str(tmp_path / "output.blend"))
    video_plane = bpy.data.objects["video_reference"]
    x, y, z = VIDEO_PLANE_POSITION_M
    half_width = (VIDEO_PLANE_HEIGHT_M / 2.0) * (40 / 60)
    assert tuple(video_plane.data.vertices[0].co) == pytest.approx((x, y - half_width, z - VIDEO_PLANE_HEIGHT_M / 2.0))
    image_node = next(node for node in video_plane.data.materials[0].node_tree.nodes if node.type == "TEX_IMAGE")
    assert image_node.image.source == "SEQUENCE"
    assert image_node.image_user.frame_start == _FIRST_MOTION_BLENDER_FRAME
    assert image_node.image_user.frame_duration == N_FRAMES


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
@pytest.mark.skipif(
    not (FLAME_MODEL_PATH.exists() and SMPLX_MODEL_PATH.exists()),
    reason="needs the FLAME/SMPL-X model files (see README's Setup section)",
)
def test_run_with_face_motion_keyframes_jaw_and_expression_but_never_eyes(tmp_path):
    """End-to-end wiring check for stage 9's `face_motion.npz`: not a
    re-test of the FLAME-to-SMPL-X expression mapping itself (see
    test_face_blendshapes.py for that), just confirmation that when it's
    present, `run()` actually keyframes both the jaw bone (`write_amass_npz`'s
    `jaw_pose` param) and at least one `Exp###` shape key
    (`flame_to_smplx_expression` + `_keyframe_face_expression`) with real,
    per-frame-varying values, not just present but trivially flat/zero.
    Also a regression guard: eye bones must stay at the armature's own rest
    rotation always, even when `face_motion.npz` happens to carry gaze data
    (stage 9 always saves `head_eye_euler` for the CSV; stage 10 must never
    consume it for eye bones)."""
    import bpy

    n_frames = 5
    motion_npz_path = tmp_path / "retargeted_motion.npz"
    np.savez(motion_npz_path, **_fake_motion())

    face_motion_npz_path = tmp_path / "face_motion.npz"
    flame_expression = np.zeros((n_frames, 50), dtype=np.float32)
    flame_expression[:, 0] = np.linspace(0.0, 1.0, n_frames)  # a real, moving expression component
    flame_jaw_pose = np.zeros((n_frames, 3), dtype=np.float32)
    flame_jaw_pose[:, 0] = np.linspace(0.0, 0.4, n_frames)  # a real jaw-open rotation
    head_eye_euler = np.zeros((n_frames, 9), dtype=np.float32)
    head_eye_euler[:, 3] = np.linspace(0.0, 30.0, n_frames)  # LeftEyeYaw, present, must still be ignored
    np.savez(
        face_motion_npz_path, flame_expression=flame_expression, flame_jaw_pose=flame_jaw_pose,
        head_eye_euler=head_eye_euler,
    )

    runRecord = _make_runRecord(tmp_path, motion_npz_path, face_motion_npz_path=face_motion_npz_path)

    run(runRecord)

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    scene = bpy.context.scene
    scene.frame_set(_FIRST_MOTION_BLENDER_FRAME - 1)  # the neutral rest frame
    rest_jaw_quat = tuple(armature.pose.bones["jaw"].rotation_quaternion)
    left_eye_bone = next(b for b in armature.pose.bones if "eye" in b.name.lower() and "left" in b.name.lower())
    rest_eye_quat = tuple(left_eye_bone.rotation_quaternion)
    scene.frame_set(n_frames - 1 + _FIRST_MOTION_BLENDER_FRAME)  # the largest jaw-open frame
    open_jaw_quat = tuple(armature.pose.bones["jaw"].rotation_quaternion)
    open_eye_quat = tuple(left_eye_bone.rotation_quaternion)
    assert not rest_jaw_quat == pytest.approx(open_jaw_quat, abs=1e-4)  # real, non-trivial jaw rotation, not just present
    assert rest_eye_quat == pytest.approx(open_eye_quat, abs=1e-6)  # eyes never move, regardless of gaze data

    body_mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    action = body_mesh.data.shape_keys.animation_data.action
    assert any(fc.data_path.startswith("key_blocks[") for fc in _iter_action_fcurves(action))


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_run_grounds_the_first_motion_frame_foot_position_near_zero(tmp_path):
    """Regression test for a real bug: incam space has no
    floor reference, so without the offset the character sits wherever
    GVHMR's raw numbers happened to put it (~1.7m underground on a real
    clip). After `run()`, the lowest point of a foot/ankle on the first
    real motion frame should land at (approximately) Z=0.

    Uses 300 frames (not the usual N_FRAMES=5) so the opening reference is
    verified independently of a long clip's later motion. Frame 1 is the
    prepended rest pose; frame 2 is the first source frame.
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
    # still right here, no need to reopen the file to check it. A separate
    # test below (`test_run_with_an_object_shape...`) covers a real
    # save->reopen round trip.
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    assert abs(_lowest_foot_z(
        armature, frame_start=_FIRST_MOTION_BLENDER_FRAME, frame_end=_FIRST_MOTION_BLENDER_FRAME,
    )) < 0.01


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
    live, the same fix a Blender artist would apply by hand (deleting the
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
        # A completely different bone is untouched, every frame still keyframed.
        assert blender_frame in other_frames
    assert 1 in pelvis_frames  # the prepended rest-pose frame itself, never flagged


@pytest.mark.skipif(not HAS_BPY, reason="needs the export pixi environment")
def test_run_with_an_object_shape_combines_body_and_object_into_one_blend_file(tmp_path):
    """When `align_scene_scale` tracked an object, it's added directly into
    the same live scene as the body and saved together in one `output.blend`
    (see module docstring for why). Held frames get baked world-space
    keyframes; an attachment event gets a live Child Of constraint instead,
    so the attachment survives a retarget onto a different rig. This test
    covers both: a flat held value before/after one attachment event on the
    right wrist.

    Also a regression test for orphaned data-blocks from this stage's own
    two-pass floor-offset build resurfacing in the saved file in place of
    the corrected body (`_clear_scene`'s own `orphans_purge` call is what
    this asserts against).

    Runs `run()` in a genuinely separate process, not in-process like this
    file's other `run()` tests: an order-dependent bpy state quirk was found
    when chaining this test after another large build in the same pytest
    session. Every real invocation of this stage is already its own fresh
    subprocess, so matching that shape here sidesteps the whole bug class
    rather than chasing it.
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

    # A flat held value, far from the body, stage 8's own real contract
    # for every non-attached frame (see hoi_object_pose.py's own module
    # docstring). The attachment event below (right wrist, joint_idx=21)
    # covers frames 20-40; these array values are never actually read for
    # that range (see `_held_frame_mask`), left flat here specifically so
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
    # smoke test), verify success via the output artifact instead.
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

    # A real save->reopen round trip (not just reading the live in-memory
    # scene, which the subprocess above doesn't share with this process
    # anyway), confirms the saved file itself, not just Blender's live
    # state, is correct.
    bpy.ops.wm.open_mainfile(filepath=str(output_path))

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    object_mesh = next(o for o in bpy.data.objects if o.type == "MESH" and o.name == OBJECT_MESH_PREFIX + object_kind)
    wrist_bone = armature.pose.bones["right_wrist"]
    scene = bpy.context.scene

    assert abs(_lowest_foot_z(
        armature, frame_start=_FIRST_MOTION_BLENDER_FRAME, frame_end=_FIRST_MOTION_BLENDER_FRAME,
    )) < 0.01
    floor_offset = -_lowest_foot_z(armature)

    def blender_frame(raw_frame: int) -> int:
        return raw_frame + _FIRST_MOTION_BLENDER_FRAME

    # This global-minimum-derived value need not be zero: grounding now uses
    # the opening motion frame, not a later foot position. It still only
    # shifts Blender's own Z (see `_object_pose_to_blender_world`), so the
    # checks below compare the two axes it never touches: X and Y.
    expected_held, _ = _object_pose_to_blender_world(held_value, np.eye(3), pelvis_rest, floor_offset)

    # Held, before the event.
    scene.frame_set(blender_frame(0))
    assert np.allclose(tuple(object_mesh.location)[:2], expected_held[:2], atol=1e-4)

    # At the snap frame itself, the object's live (constrained) position
    # exactly reproduces the raw reference measurement, true by
    # construction of how the constraint's own inverse_matrix is derived
    # (see `_add_attachment_constraint`), a direct check that the derivation
    # is wired correctly end to end, not just internally self-consistent.
    scene.frame_set(blender_frame(event["snap_frame"]))
    expected_snap_loc, _ = _object_pose_to_blender_world(
        np.array(event["ref_center"]), np.array(event["ref_rotation"]), pelvis_rest, floor_offset,
    )
    assert np.allclose(tuple(object_mesh.matrix_world.translation)[:2], expected_snap_loc[:2], atol=1e-4)

    # Attached: the object's own position relative to the wrist bone stays
    # perfectly rigid across several frames within the event, only
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

    # Held again after the event, back to the same flat value (proves the
    # constraint's own influence actually turns back off, and the object's
    # own base transform correctly resumes rather than staying wherever the
    # constraint last left it).
    scene.frame_set(blender_frame(event["end_frame"] + 1))
    assert np.allclose(tuple(object_mesh.location)[:2], expected_held[:2], atol=1e-4)

    assert np.allclose(tuple(object_mesh.dimensions), (0.1, 0.12, 0.08), atol=1e-4)
