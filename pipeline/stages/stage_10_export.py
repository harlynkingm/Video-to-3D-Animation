"""export: combines the retargeted SMPL-X body+hands motion into
`output.blend`, this pipeline's final deliverable, written to the run's
top level. Native Blender file, not FBX (its own animation-baking
introduces real precision errors on stage 8's sparsely-keyframed object
data). Runs only in the separate `export` pixi env (needs `bpy`, no
torch).

If `align_scene_scale` tracked an object, it's added as a second mesh: a
live Child Of constraint rigidly attaches it to the holding bone during a
genuine hold, held still and unparented otherwise (see stage 8's own
docstring for the attached/held design). Human-only runs export body-only.

Retargeted motion is GVHMR's own incam frame, reoriented upright for the
addon's import path; floor height and a prepended rest-pose frame cover
what incam has no reference for. Frames flagged unreliable (stage 2) get
pelvis keyframes deleted, not exported as fabricated data.

If `capture_face` tracked a face, jaw rotation rides alongside the
body/hands, from Group D's own tracked `jaw_pose`. Eye rotation stays
zero, SMPL-X's eye bones don't skin cleanly against tracked gaze angles;
gaze is consumed from `output_face.csv` downstream instead. The mesh's own
FLAME expression is mapped into its `Exp000-099` shape keys
(`face_blendshapes.flame_to_smplx_expression`), a loose, preview-quality
approximation, not the real character remap, which happens downstream
from `output_face.csv` directly. Skipped (`--skip-face-capture`) runs
export an inert face, same as always."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Deliberately never imports gvhmr_adapter/hamer_adapter/face_landmark_fit/
# stage_9_capture_face, even just for their KEY_* string constants, all
# pull in torch at module load time, and the `export` pixi env this module
# actually runs in has no torch installed (face_blendshapes is plain numpy
# underneath, safe to import directly).
from ..algorithms.face.face_blendshapes import flame_to_smplx_expression
from ..bpy.blender_armature import (
    _delete_unreliable_root_keyframes, _fix_rotation_hemisphere_continuity, _lowest_foot_z, _orient_bones_toward_children,
)
from ..bpy.blender_constants import _FIRST_MOTION_BLENDER_FRAME
from ..bpy.blender_face_expression import _keyframe_face_expression
from ..bpy.blender_object_attachment import (
    _add_attachment_constraint, _add_object_mesh, _held_frame_mask, _keyframe_held_object_pose, _reset_object_base_transform,
)
from ..bpy.blender_preview import _add_video_reference_plane, _build_face_preview_blend, _build_landmark_preview_blend
from ..bpy.blender_scene import _clear_scene
from ..helpers.amass_export_helper import write_amass_npz
from ..helpers.bvh_export import root_camera_to_upright
from ..helpers.progress_reporter import frame_progress, report_single_shot
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName

EXPORT_DIRNAME = f"stage{StageName.STAGE_10_EXPORT.stage_number}_export"
# Matches stage_9_capture_face.FACE_DIRNAME, reconstructed locally rather
# than imported, per this module's own no-torch-imports rule above.
FACE_DIRNAME = f"stage{StageName.STAGE_9_CAPTURE_FACE.stage_number}_face"
BODY_AMASS_FILENAME = "body.npz"
OUTPUT_BLEND_FILENAME = "output.blend"

# Matches every other stage's own FRAMES_DIR_OUTPUT_KEY (stage_1/2/3/4/6/7/8
# each define an identical local copy rather than import stage_0_ingest_
# video, its own established pattern, this module just follows it).
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

# The addon's own default names (e.g. "SMPLX-lh-neutral_body") are renamed to
# these, readable in the Outliner without needing to know the addon's own
# internal model-spec naming convention.
PERSON_ARMATURE_NAME = "person"
PERSON_ARMATURE_DATA_NAME = "person_armature"
PERSON_MESH_NAME = "person_mesh"

# This stage's own progress.json output key.
OUTPUT_BLEND = "output_blend"

# Matches stage_6_align_scene_scale.OUTPUT_OBJECT_SHAPE/OUTPUT_SCENE_SCALE
# and scene_scale.json's own KEY_PELVIS_REST, defined locally, not
# imported, since that module transitively pulls in torch (see module
# docstring). Absent from a human-only run's outputs (no object tracked).
_OBJECT_SHAPE_OUTPUT_KEY = "object_shape"
_SCENE_SCALE_OUTPUT_KEY = "scene_scale"
_PELVIS_REST_KEY = "pelvis_rest_incam"

# Matches stage_8_optimize_hoi.OUTPUT_OBJECT_POSE_NPZ/OUTPUT_ATTACHMENT_EVENTS
# and their own array/field names exactly, defined locally, not imported,
# since that module transitively pulls in torch (see module docstring).
_OBJECT_POSE_NPZ_OUTPUT_KEY = "object_pose_npz"
_KEY_TRANSLATION = "translation"
_KEY_ROTATION = "rotation"
_ATTACHMENT_EVENTS_OUTPUT_KEY = "attachment_events"

# Matches retarget_hands.OUTPUT_RETARGET_MOTION_NPZ and its own npz key names
# exactly (the latter same strings as gvhmr_adapter.KEY_GLOBAL_ORIENT/
# KEY_BODY_POSE/KEY_BETAS/KEY_TRANSL/KEY_ROOT_MOTION_UNRELIABLE and
# hamer_adapter.KEY_LEFT_HAND_POSE/KEY_RIGHT_HAND_POSE), defined locally,
# not imported, since importing `stage_5_retarget_hands` (or the adapter
# modules) transitively pulls in torch at module load time (see module
# docstring).
_RETARGET_MOTION_NPZ_OUTPUT_KEY = "retarget_motion_npz"
_KEY_GLOBAL_ORIENT = "global_orient"
_KEY_BODY_POSE = "body_pose"
_KEY_BETAS = "betas"
_KEY_TRANSL = "transl"
_KEY_LEFT_HAND_POSE = "left_hand_pose"
_KEY_RIGHT_HAND_POSE = "right_hand_pose"
_KEY_ROOT_MOTION_UNRELIABLE = "root_motion_unreliable"

# Matches stage_9_capture_face.OUTPUT_FACE_MOTION and face_landmark_fit's own
# KEY_EXPRESSION/KEY_JAW_POSE npz key names, defined locally, not imported,
# since either module transitively pulls in torch (see module docstring).
# Absent from a run where face capture was skipped (--skip-face-capture).
_FACE_MOTION_NPZ_OUTPUT_KEY = "face_motion"
_KEY_FLAME_EXPRESSION = "flame_expression"
_KEY_FLAME_JAW_POSE = "flame_jaw_pose"
_FACE_PREVIEW_TEMPLATE_OUTPUT_KEY = "face_preview_template"
_FACE_PREVIEW_PC2_OUTPUT_KEY = "face_preview_pc2"
FACE_PREVIEW_BLEND_FILENAME = "FLAME_face_preview.blend"

# Matches face_preview's own LANDMARK_PREVIEW_*_FILENAME output keys.
_LANDMARK_PREVIEW_RAW_TEMPLATE_OUTPUT_KEY = "landmark_preview_raw_template"
_LANDMARK_PREVIEW_RAW_PC2_OUTPUT_KEY = "landmark_preview_raw_pc2"
_LANDMARK_PREVIEW_SMOOTHED_TEMPLATE_OUTPUT_KEY = "landmark_preview_smoothed_template"
_LANDMARK_PREVIEW_SMOOTHED_PC2_OUTPUT_KEY = "landmark_preview_smoothed_pc2"
LANDMARK_PREVIEW_BLEND_FILENAME = "landmark_preview.blend"

# Matches face_preview.write_arkit_preview's own output keys, the
# output_face.csv deliverable, visualized as a real 3D face performance
# (see that function's own docstring). Absent when body_models/arkit/
# face_preview_shapes.npz was never generated locally.
_ARKIT_PREVIEW_TEMPLATE_OUTPUT_KEY = "arkit_preview_template"
_ARKIT_PREVIEW_PC2_OUTPUT_KEY = "arkit_preview_pc2"
ARKIT_PREVIEW_BLEND_FILENAME = "ARKit_face_preview.blend"

# The addon's own AMASS-import operator over-rotates 90 degrees for this
# installed addon/Blender version specifically when told anim_format="AMASS",
# the file itself stays real AMASS-formatted, only this operator argument
# differs.
_ADDON_ANIM_FORMAT = "SMPL-X"

# Installed via Blender 4.2+'s Extensions system (not the legacy addons
# folder), hence this module name rather than a plain "smplx_blender_addon".
# `bpy.ops.object.smplx_*` exists once discovered (bpy.utils.script_paths
# finds the on-disk files) but is NOT auto-enabled by a bare embedded-bpy
# session the way a real Blender.app session would restore its own saved
# preferences, `hasattr` on a bpy.ops operator is also not a reliable
# existence check either way (it returns True for literally any name; only
# calling it actually validates), so this always explicitly enables the
# addon rather than checking first.
_ADDON_MODULE_NAME = "bl_ext.user_default.smplx_blender_addon"


# The rest frame's own root orientation is the identity SMPL-X pose *plus*
# this yaw, not literal zero. Zero, written directly into this function's
# already-upright AMASS space (it bypasses root_camera_to_upright entirely,
# unlike every real frame), renders as a fixed "-Y facing" in Blender's own
# world space, a mismatch against where the real motion's own first frame
# actually faces (+X, the direction CAMERA_TO_BVH_ROOT_ROTATION's own
# "forward" maps to, see bvh_export.py's own docstring). This mismatch is
# architecturally fixed (not clip-specific): CAMERA_TO_BVH_ROOT_ROTATION is a
# constant, so every clip's real motion faces the same consistent direction
# once corrected, while the rest frame's identity orientation never passes
# through that correction at all. A rotation about AMASS's own up axis
# (index 1, Y, same axis `_write_body_amass`'s own `floor_offset` uses)
# maps, under the same up-axis-preserving change of basis
# `_AMASS_TO_BLENDER_WORLD_ROTATION` uses for everything else, to the
# same-angle same-handedness rotation about Blender's own up axis (Z), so
# +90 degrees here rotates the rest pose's fixed -Y facing to +X, matching
# the real motion's own starting direction. Not yet independently visually
# reconfirmed in Blender, if this turns out backwards, flip the sign.
_REST_POSE_YAW_RADIANS = math.pi / 2


def _prepend_rest_pose_frame(
    global_orient: np.ndarray, body_pose: np.ndarray, transl: np.ndarray,
    left_hand_pose: np.ndarray, right_hand_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adds one neutral (SMPL-X's own canonical rest stance, yawed to face
    the same direction the real motion's own first frame does, see
    `_REST_POSE_YAW_RADIANS`) frame before the real motion. Retargeting this
    onto another rig otherwise has no bind/reference pose to work from.
    Positioned at the real motion's own first frame's location, so only the
    *pose* jumps at the frame 0->1 boundary, not the character's position,
    a static calibration frame most retargeting tools skip or trim anyway.
    Always included; not currently exposed as an opt-out flag (the existing
    `RunInput` bool-flag CLI plumbing is wired for opt-in/default-off flags
    like the `--render-*-preview` family, not default-on ones, add one if
    a real need to disable this ever comes up).
    """
    rest_global_orient = np.zeros((1, 3), dtype=global_orient.dtype)
    rest_global_orient[0, 1] = _REST_POSE_YAW_RADIANS
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


def _write_body_amass(
    motion: dict, fps: float, out_path: Path, *, floor_offset: float = 0.0, jaw_pose: np.ndarray | None = None,
    camera_up: list[float] | None = None,
) -> None:
    """Maps retarget_hands' npz schema into the AMASS format the addon
    expects. Pure numpy (+ scipy for the root's rotation-matrix round trip),
    no bpy needed, kept separate from `run()` so it's testable under any
    environment, not just `export`.

    `camera_up` is the clip's measured camera-space up direction
    (`SceneInfo.camera_up`); omitting it assumes a level camera, see
    `bvh_export.camera_to_upright_rotation`.

    `floor_offset` shifts the root's own up-axis translation by a constant
    (added post-rotation, in the same upright frame `root_camera_to_upright`
    already produces), incam space has no inherent floor reference, so
    without this the character's absolute height is wherever GVHMR's raw
    numbers happened to put it. `run()` computes the real value empirically
    (there's no way to derive it from the source data alone) and passes it
    in; callers that don't care about floor placement can omit it.

    `jaw_pose` (F, 3), from stage 9's fitting loop, omitted (stays zero,
    see `write_amass_npz`'s own default) when face capture was skipped. Eye
    pose is never passed: SMPL-X's own `left_eye_smplhf`/`right_eye_smplhf`
    bones don't skin cleanly against tracked gaze angles, and gaze is
    consumed from `output_face.csv` downstream instead, not from
    `output.blend`'s own skeleton.
    """
    global_orient, transl = root_camera_to_upright(motion[_KEY_GLOBAL_ORIENT], motion[_KEY_TRANSL], camera_up)
    transl = transl.copy()
    transl[:, 1] += floor_offset  # AMASS's own up axis

    global_orient, body_pose, transl, left_hand_pose, right_hand_pose = _prepend_rest_pose_frame(
        global_orient, motion[_KEY_BODY_POSE], transl, motion[_KEY_LEFT_HAND_POSE], motion[_KEY_RIGHT_HAND_POSE],
    )

    if jaw_pose is not None:
        # The prepended rest frame (see _prepend_rest_pose_frame) has no
        # facial pose of its own, neutral (zero), same treatment the body
        # pose itself gets across that same frame boundary.
        jaw_pose = np.concatenate([np.zeros((1, 3), dtype=jaw_pose.dtype), jaw_pose], axis=0)

    write_amass_npz(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=motion[_KEY_BETAS][0],  # pooled, identical every frame
        transl=transl,
        fps=fps,
        out_path=out_path,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
    )


def run(runRecord: RunRecord) -> dict[str, str]:
    # `bpy` must be imported before `addon_utils`, importing bpy is what
    # puts Blender's own bundled scripts directory on sys.path in the first
    # place, which is where addon_utils itself lives.
    import bpy
    import addon_utils

    print(f"[{StageName.STAGE_10_EXPORT.label}] running...")

    addon_utils.enable(_ADDON_MODULE_NAME, default_set=True, persistent=True)

    motion = np.load(runRecord.stages[StageName.STAGE_5_RETARGET_HANDS].outputs[_RETARGET_MOTION_NPZ_OUTPUT_KEY])

    # Absent when face capture was skipped (--skip-face-capture), export
    # then falls back to write_amass_npz's own zero jaw/eye default and
    # simply never calls _keyframe_face_expression.
    face_motion_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_FACE_MOTION_NPZ_OUTPUT_KEY)
    jaw_pose = None
    smplx_expression = None
    face_preview_n_frames = None
    if face_motion_path is not None:
        with np.load(face_motion_path) as face_motion:
            jaw_pose = face_motion[_KEY_FLAME_JAW_POSE]
            smplx_expression = flame_to_smplx_expression(face_motion[_KEY_FLAME_EXPRESSION])
            face_preview_n_frames = face_motion[_KEY_FLAME_JAW_POSE].shape[0]

    # Only present when RunInput.render_face_preview was set, see
    # face_preview.write_flame_preview's own docstring.
    face_preview_template_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_FACE_PREVIEW_TEMPLATE_OUTPUT_KEY)
    face_preview_pc2_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_FACE_PREVIEW_PC2_OUTPUT_KEY)
    landmark_preview_raw_template_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_LANDMARK_PREVIEW_RAW_TEMPLATE_OUTPUT_KEY)
    landmark_preview_raw_pc2_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_LANDMARK_PREVIEW_RAW_PC2_OUTPUT_KEY)
    landmark_preview_smoothed_template_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_LANDMARK_PREVIEW_SMOOTHED_TEMPLATE_OUTPUT_KEY)
    landmark_preview_smoothed_pc2_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_LANDMARK_PREVIEW_SMOOTHED_PC2_OUTPUT_KEY)
    arkit_preview_template_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_ARKIT_PREVIEW_TEMPLATE_OUTPUT_KEY)
    arkit_preview_pc2_path = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE].outputs.get(_ARKIT_PREVIEW_PC2_OUTPUT_KEY)

    export_dir = Path(runRecord.progress_dir) / EXPORT_DIRNAME
    face_export_dir = Path(runRecord.progress_dir) / FACE_DIRNAME
    amass_path = export_dir / BODY_AMASS_FILENAME

    # For the face preview blends' own reference video plane (see
    # _add_video_reference_plane), absent stays None, same as every
    # optional face_preview_*_path above, and the plane is just skipped.
    # `.get(...)` on `stages` itself too: a synthetic/test RunRecord may
    # have no STAGE_0_INGEST_VIDEO entry at all, unlike a real run (which
    # always starts there).
    ingest_video_stage = runRecord.stages.get(StageName.STAGE_0_INGEST_VIDEO)
    frames_dir_str = ingest_video_stage.outputs.get(FRAMES_DIR_OUTPUT_KEY) if ingest_video_stage is not None else None
    frames_dir = Path(frames_dir_str) if frames_dir_str is not None else None

    # First pass, floor_offset=0: incam space has no inherent floor
    # reference, so how far below (or above) Y=0 the character's own feet
    # land can only be measured empirically, not derived from the source
    # data, build once to find out. The first *real* source frame is the
    # grounding reference, rather than a later, lowest foot in the clip: the
    # latter made otherwise stationary opening poses visibly float. Jaw/eye
    # pose do not affect foot height, so they're omitted here and only passed
    # on the real second pass below.
    camera_up = runRecord.scene.camera_up or None
    _write_body_amass(motion, runRecord.scene.fps, amass_path, camera_up=camera_up)
    _clear_scene(bpy)
    bpy.ops.object.smplx_add_animation(filepath=str(amass_path), anim_format=_ADDON_ANIM_FORMAT)
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    floor_offset = -_lowest_foot_z(
        armature, frame_start=_FIRST_MOTION_BLENDER_FRAME, frame_end=_FIRST_MOTION_BLENDER_FRAME,
    )

    # Second pass: rebuild with that offset baked into the root's own
    # per-frame translation curve, keeps the addon's separate, always-
    # static "root" bone at true world origin, since only the *pelvis*'s
    # translation is shifted, not an object-level transform that would move
    # everything including "root".
    _write_body_amass(
        motion, runRecord.scene.fps, amass_path, floor_offset=floor_offset, jaw_pose=jaw_pose, camera_up=camera_up,
    )
    _clear_scene(bpy)
    bpy.ops.object.smplx_add_animation(filepath=str(amass_path), anim_format=_ADDON_ANIM_FORMAT)

    # The addon names these after its own internal model spec (e.g.
    # "SMPLX-lh-neutral_body"/"SMPLX-mesh-neutral"), renamed to something
    # a downstream Blender user actually wants to see in the Outliner. Only
    # these two objects exist at this point (`_clear_scene` wiped everything
    # else, and the tracked object mesh, if any, hasn't been added yet).
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    armature.name = PERSON_ARMATURE_NAME
    armature.data.name = PERSON_ARMATURE_DATA_NAME
    body_mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    body_mesh.name = PERSON_MESH_NAME

    # Cosmetic only (see that function's own docstring for why it needs to
    # rewrite every existing pose keyframe to stay that way), must run
    # before any attachment constraint is added below, since those read this
    # same armature's own live bone transforms at a specific frame.
    _orient_bones_toward_children(bpy, armature)

    # Must run after _orient_bones_toward_children, see this function's own
    # docstring for why (that pass would otherwise reinsert exactly the
    # keyframes deleted here). Followed immediately by the hemisphere-
    # continuity fix, which has to see the final keyframe set: deleting a run
    # is one of its two real triggers, and it rewrites keyframe values, so
    # anything that reinserts keyframes afterward would undo it.
    with report_single_shot(StageName.STAGE_10C_CONTINUITY.label):
        _delete_unreliable_root_keyframes(bpy, armature, motion[_KEY_ROOT_MOTION_UNRELIABLE])
        _fix_rotation_hemisphere_continuity(armature)

    if smplx_expression is not None:
        _keyframe_face_expression(body_mesh, smplx_expression)

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
        _keyframe_held_object_pose(obj, translations, rotations, held_mask, pelvis_rest, floor_offset, camera_up)
        for event in frame_progress(attachment_events, total=len(attachment_events),
                                     label=StageName.STAGE_10D_ATTACH_TRACKED_OBJECT.label, unit="event"):
            _reset_object_base_transform(obj, event["start_frame"] + _FIRST_MOTION_BLENDER_FRAME)
            _reset_object_base_transform(obj, event["end_frame"] + _FIRST_MOTION_BLENDER_FRAME)
            _add_attachment_constraint(bpy, obj, armature, event, n_frames, pelvis_rest, floor_offset, camera_up)

    # Include the source footage in the final deliverable too. This reuses
    # the preview blends' own plane builder, including its fixed world-space
    # placement, image-sequence setup, and source-frame -> Blender-frame
    # alignment. Unlike the face preview data, retargeted motion is always
    # available, so this remains useful on --skip-face-capture runs.
    if frames_dir is not None:
        _add_video_reference_plane(bpy, frames_dir, len(motion[_KEY_GLOBAL_ORIENT]))

    # Building the animation above (attachment constraints, foot-grounding)
    # moves the scene's current frame around as a side effect of reading
    # bone/object transforms at specific frames. Keep frame 1 as the
    # intentionally preserved T-pose, but open and render/playback the
    # deliverable from the first actual motion frame.
    scene = bpy.context.scene
    scene.frame_start = _FIRST_MOTION_BLENDER_FRAME
    scene.frame_set(_FIRST_MOTION_BLENDER_FRAME)
    with report_single_shot(StageName.STAGE_10E_SAVE_FILE.label):
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

    # The run's own overall deliverable (`RunOutputs.final_blend`),
    # distinct from this stage's own `outputs` dict entry below, lets a
    # caller find the final file without knowing which stage produced it.
    runRecord.outputs.final_blend = str(output_path)

    outputs = {OUTPUT_BLEND: str(output_path)}

    # Built after output.blend is already saved: it clears and rebuilds the
    # whole scene for its own separate, unrelated content (see
    # _build_face_preview_blend's own docstring), which must not touch the
    # already-saved main scene.
    if face_preview_template_path is not None:
        face_preview_path = face_export_dir / FACE_PREVIEW_BLEND_FILENAME
        _build_face_preview_blend(
            bpy, Path(face_preview_template_path), Path(face_preview_pc2_path), face_preview_n_frames,
            runRecord.scene.fps, face_preview_path, frames_dir=frames_dir,
        )
        outputs["face_preview_blend"] = str(face_preview_path)

    if landmark_preview_raw_template_path is not None:
        landmark_preview_path = face_export_dir / LANDMARK_PREVIEW_BLEND_FILENAME
        _build_landmark_preview_blend(
            bpy, Path(landmark_preview_raw_template_path), Path(landmark_preview_raw_pc2_path),
            Path(landmark_preview_smoothed_template_path), Path(landmark_preview_smoothed_pc2_path),
            face_preview_n_frames, runRecord.scene.fps, landmark_preview_path,
        )
        outputs["landmark_preview_blend"] = str(landmark_preview_path)

    if arkit_preview_template_path is not None:
        arkit_preview_path = face_export_dir / ARKIT_PREVIEW_BLEND_FILENAME
        _build_face_preview_blend(
            bpy, Path(arkit_preview_template_path), Path(arkit_preview_pc2_path), face_preview_n_frames,
            runRecord.scene.fps, arkit_preview_path, frames_dir=frames_dir,
        )
        outputs["arkit_preview_blend"] = str(arkit_preview_path)

    print(f"[{StageName.STAGE_10_EXPORT.label}] done")

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_10_EXPORT)
