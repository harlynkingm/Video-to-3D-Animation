"""optimize_hoi: solves the tracked object's own per-frame 6DoF pose --
rigidly attached to a body joint during a genuine sustained hold, held
perfectly still everywhere else (before the first grip, between two grips,
and after the last one). The actual algorithm lives in
`pipeline/algorithms/hoi_object_pose.py`, kept as pure math/no I/O; this file
is the orchestration around it -- load stage 6/7's outputs, wire up the one
callback that pure module needs (an expensive per-frame depth-based position
fit), run it, write the result.

Stays in the same incam/body-metric space as every upstream stage (5-7) and
`object_shape.json` -- no camera-to-upright pivot correction here, that's
purely a stage-9 export concern.

A run without a tracked object (human-only) has nothing for this stage to
solve -- writes an all-identity, all-flagged-low-confidence pose sequence
rather than skipping the stage's own output entirely, so stage 9's eventual
consumption of this file doesn't need a separate no-object code path.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..adapters.depth_anything3_adapter import KEY_DEPTH, DepthAnything3Adapter
from ..adapters.gvhmr.gvhmr_adapter import KEY_BETAS, KEY_BODY_POSE, KEY_GLOBAL_ORIENT, KEY_TRANSL
from ..adapters.gvhmr.gvhmr_smplx_skeleton import SmplxSkeleton
from ..algorithms.depth_unprojection import unproject_depth_to_points
from ..algorithms.hoi_object_pose import compute_object_pose_sequence
from ..algorithms.object_extent_fit import fit_position_and_orientation
from ..helpers.progress_reporter import report_single_shot
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName
from ..stages.stage_1_mask_and_track import OUTPUT_OBJECT_MASKS
from ..stages.stage_5_retarget_hands import OUTPUT_RETARGET_MOTION
from ..stages.stage_6_align_scene_scale import KEY_SCALE, KEY_TRANSLATION, OUTPUT_OBJECT_SHAPE, OUTPUT_SCENE_SCALE
from ..stages.stage_7_annotate_contacts import OUTPUT_CONTACT_EVENTS, _LazyMaskLoader

# stage_0_ingest_video.py's own output key, consumed here (same convention
# every other stage reading frames uses).
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

HOI_DIRNAME = f"stage{StageName.STAGE_8_OPTIMIZE_HOI.stage_number}_interaction"
OBJECT_POSE_FILENAME = "object_pose.pt"
OBJECT_POSE_NPZ_FILENAME = "object_pose.npz"
ATTACHMENT_EVENTS_FILENAME = "attachment_events.json"

# This stage's own progress.json output keys.
OUTPUT_OBJECT_POSE = "object_pose"
OUTPUT_OBJECT_POSE_NPZ = "object_pose_npz"
OUTPUT_ATTACHMENT_EVENTS = "attachment_events"

# Below this duration, a contact event reads as a brief collision/kick, not a
# sustained hold worth rigidly attaching the object to -- treated as no
# attachment at all, so the object just stays at whatever pose it's currently
# holding through the brief touch (this module no longer independently
# tracks the object while it isn't attached -- see hoi_object_pose.py's own
# module docstring for why). Expressed in seconds (not a fixed frame count)
# so it scales correctly across clips at different frame rates. Uncalibrated:
# a rough starting point, needs real footage with an actual kick/bounce to
# validate (see docs/ARCHITECTURE.md's stage 8 section).
MIN_ATTACHMENT_DURATION_SECONDS = 0.4


def _min_attachment_duration_frames(fps: float) -> int:
    return max(1, round(MIN_ATTACHMENT_DURATION_SECONDS * fps))


def _resolve_overlapping_events(events: list[dict]) -> list[dict]:
    """Same-joint overlaps are already merged upstream by `contact_detection.
    consolidate_overlapping_events`; anything left here is a genuine cross-
    joint overlap (a two-handed grip, or a hand-to-hand pass), which needs its
    own resolution since `compute_object_pose_sequence` assumes no two
    attachment events cover the same frame. Resolved deterministically:
    process events highest-`mean_confidence`-first, each claiming its own
    largest still-unclaimed contiguous sub-range; an event left with nothing
    unclaimed is dropped rather than kept as a zero-length no-op.
    """
    if not events:
        return []
    n_frames = max(e["end_frame"] for e in events) + 1
    claimed = np.zeros(n_frames, dtype=bool)
    resolved = []
    for event in sorted(events, key=lambda e: -e["mean_confidence"]):
        start, end = event["start_frame"], event["end_frame"]
        free = ~claimed[start:end + 1]
        if not free.any():
            continue
        best_len, best_start, run_len, run_start = 0, None, 0, None
        for i, is_free in enumerate(free):
            if is_free:
                if run_len == 0:
                    run_start = i
                run_len += 1
                if run_len > best_len:
                    best_len, best_start = run_len, run_start
            else:
                run_len = 0
        new_start = start + best_start
        new_end = new_start + best_len - 1
        claimed[new_start:new_end + 1] = True
        resolved.append({**event, "start_frame": new_start, "end_frame": new_end})
    return resolved


def _qualifying_attachment_events(contact_events: list[dict], fps: float) -> list[dict]:
    """Filters `contact_events.json`'s own events down to genuine sustained
    holds (any region -- a hat on the head uses the same mechanism as a mug
    in a hand, just a different attachment joint; see `hoi_object_pose`'s own
    docstring), then resolves any remaining cross-joint overlap."""
    min_frames = _min_attachment_duration_frames(fps)
    candidates = [
        {
            "region": event["regions"][0],  # every region in the list resolves to the same joint (consolidated)
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "mean_confidence": event["mean_confidence"],
        }
        for event in contact_events
        if event["end_frame"] - event["start_frame"] + 1 >= min_frames
    ]
    return _resolve_overlapping_events(candidates)


def run(runRecord: RunRecord) -> dict[str, str]:
    motion = torch.load(
        runRecord.stages[StageName.STAGE_5_RETARGET_HANDS].outputs[OUTPUT_RETARGET_MOTION], weights_only=False,
    )
    body_motion = {
        "global_orient": motion[KEY_GLOBAL_ORIENT].numpy(),
        "body_pose": motion[KEY_BODY_POSE].numpy(),
        "betas": motion[KEY_BETAS].numpy(),
        "transl": motion[KEY_TRANSL].numpy(),
    }
    n_frames = body_motion["global_orient"].shape[0]

    stage_6_outputs = runRecord.stages[StageName.STAGE_6_ALIGN_SCENE_SCALE].outputs
    object_shape_path = stage_6_outputs.get(OUTPUT_OBJECT_SHAPE)

    if object_shape_path is None:
        # Human-only run -- nothing for this stage to solve.
        translation = np.zeros((n_frames, 3))
        rotation = np.tile(np.eye(3), (n_frames, 1, 1))
        is_low_confidence = np.ones(n_frames, dtype=bool)
        resolved_events: list[dict] = []
    else:
        object_shape = json.loads(Path(object_shape_path).read_text())
        scene_scale = json.loads(Path(stage_6_outputs[OUTPUT_SCENE_SCALE]).read_text())
        scale = np.array(scene_scale[KEY_SCALE])
        translation_fit = np.array(scene_scale[KEY_TRANSLATION])
        initial_center = np.array(object_shape["center"])
        initial_rotation = np.array(object_shape["rotation"])

        contact_events = json.loads(
            Path(runRecord.stages[StageName.STAGE_7_ANNOTATE_CONTACTS].outputs[OUTPUT_CONTACT_EVENTS]).read_text()
        )
        attachment_events = _qualifying_attachment_events(contact_events, runRecord.scene.fps)

        stage_1_outputs = runRecord.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs
        native_hw = (runRecord.scene.height, runRecord.scene.width)
        object_masks = _LazyMaskLoader(stage_1_outputs[OUTPUT_OBJECT_MASKS], n_frames, native_hw)
        frames_dir = Path(runRecord.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])
        K = np.array(runRecord.scene.intrinsics_K)
        focal_length_px = K[0, 0]

        # `object_position_fn` is the expensive one (a fresh DA3 pass per
        # call) -- memoized since `compute_object_pose_sequence`'s own
        # free-track already asks for every frame (including ones inside
        # attachment windows, for the cross-fade blend's own use).
        position_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        adapter = DepthAnything3Adapter()
        adapter.load()
        try:
            def object_position_fn(frame: int) -> tuple[np.ndarray, np.ndarray] | None:
                if frame in position_cache:
                    return position_cache[frame]
                object_mask = object_masks[frame]
                if object_mask is None:
                    position_cache[frame] = None
                    return None
                frame_path = frames_dir / f"{frame:06d}.jpg"
                result = adapter.infer(str(frame_path), focal_length_px)
                depth = cv2.resize(result[KEY_DEPTH], (native_hw[1], native_hw[0]), interpolation=cv2.INTER_LINEAR)
                scene_cloud = unproject_depth_to_points(depth, K)
                object_points = (scene_cloud[object_mask.reshape(-1)] - translation_fit) / scale
                fitted = fit_position_and_orientation(object_points)
                position_cache[frame] = fitted
                return fitted

            skeleton = SmplxSkeleton()
            with report_single_shot(StageName.STAGE_8_OPTIMIZE_HOI.label):
                pose_sequence = compute_object_pose_sequence(
                    n_frames=n_frames,
                    attachment_events=attachment_events,
                    body_motion=body_motion,
                    initial_center=initial_center,
                    initial_rotation=initial_rotation,
                    object_position_fn=object_position_fn,
                    skeleton=skeleton,
                )
        finally:
            adapter.unload()

        translation = pose_sequence["translation"]
        rotation = pose_sequence["rotation"]
        is_low_confidence = pose_sequence["is_low_confidence"]
        resolved_events = pose_sequence["resolved_events"]

    hoi_dir = Path(runRecord.progress_dir) / HOI_DIRNAME
    hoi_dir.mkdir(parents=True, exist_ok=True)
    object_pose_path = hoi_dir / OBJECT_POSE_FILENAME
    object_pose_npz_path = hoi_dir / OBJECT_POSE_NPZ_FILENAME

    torch.save(
        {
            "translation": torch.from_numpy(translation),
            "rotation": torch.from_numpy(rotation),
            "is_low_confidence": torch.from_numpy(is_low_confidence),
        },
        object_pose_path,
    )
    np.savez(object_pose_npz_path, translation=translation, rotation=rotation, is_low_confidence=is_low_confidence)

    # A caller that wants the object to stay correctly attached after the
    # skeleton itself is later retargeted onto a different rig (see
    # hoi_object_pose.compute_object_pose_sequence's own docstring) needs a
    # live parent/constraint relationship to the joint, not the baked
    # translation/rotation above -- this is the raw per-event reference data
    # that lets a caller (stage 9) build one. A plain JSON, not part of the
    # .pt/.npz above: an irregular-length list of small per-event records,
    # not a fixed-size per-frame array.
    attachment_events_path = hoi_dir / ATTACHMENT_EVENTS_FILENAME
    attachment_events_path.write_text(json.dumps([
        {
            "region": event["region"],
            "joint_idx": event["joint_idx"],
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "snap_frame": event["snap_frame"],
            "ref_center": np.asarray(event["ref_center"]).tolist(),
            "ref_rotation": np.asarray(event["ref_rotation"]).tolist(),
            "is_low_confidence": bool(event["is_low_confidence"]),
        }
        for event in resolved_events
    ]))

    return {
        OUTPUT_OBJECT_POSE: str(object_pose_path),
        OUTPUT_OBJECT_POSE_NPZ: str(object_pose_npz_path),
        OUTPUT_ATTACHMENT_EVENTS: str(attachment_events_path),
    }


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_8_OPTIMIZE_HOI)
