"""annotate_contacts: detects per-frame body-to-object contact primarily from
geometry (no learned model, no GPU for this part): projects each body
region's candidate joints (fingertips + wrist per hand; a couple of
representative joints for head/chest/arms/legs) into image space and checks
proximity to the tracked object's 2D mask, using GVHMR/HaMeR's own retargeted
positions -- deliberately not compared against real-world depth for this
first pass (see `pipeline/algorithms/contact_detection.py`'s module docstring
for why). Hysteresis (the same lock/release pattern as the stage 4
wrist-plausibility gate) turns raw per-frame confidence into contact events.

Two second passes over the same-region-detected events, both in
`contact_detection.py`: `consolidate_overlapping_events` merges events from
different regions that share an underlying joint (a hand and its own arm
both treat the wrist as a candidate, so one real grip can otherwise produce
two region-events); `depth_gap_for_joint` DOES use real depth, but
Depth-Anything-3's own per-frame estimate (found reliable by this project's
depth investigation), never GVHMR's. It's only ever invoked on the specific
frames an event was already detected on -- a real but bounded GPU cost, not
a per-clip one -- to catch the one thing pure 2D-proximity can't: a body part
passing in front of or behind the object in the image, with no actual
contact, still overlaps its mask exactly like a real touch would.

Human-in-the-loop, per this project's own philosophy (machine-triggered
prompts only, never an open-ended manual review pile): a large depth gap is
not ambiguous -- it's DA3 confidently saying the object and body were never
actually close -- so `_verify_events_with_depth` drops that event outright
rather than keeping it around flagged as uncertain. Conversely, a small gap is
strong positive evidence of contact and overrides a merely-noisy 2D
confidence score, so `_event_to_dict` only sets `is_low_confidence` when the 2D
confidence is low AND depth verification couldn't settle it either way (no
mask that frame). If `RunInput.render_contacts_preview` is set, also writes
one annotated JPEG per surviving event (see `_render_contacts_preview`) as a
visual spot-check of the same signal.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ..adapters.depth_anything3_adapter import KEY_DEPTH, DepthAnything3Adapter
from ..adapters.gvhmr.gvhmr_adapter import KEY_BETAS, KEY_BODY_POSE, KEY_GLOBAL_ORIENT, KEY_TRANSL
from ..adapters.hamer.hamer_adapter import KEY_LEFT_HAND_POSE, KEY_RIGHT_HAND_POSE
from ..adapters.sam31.sam31_tracker import KEY_PACKED_MASKS, unpack_masks
from ..algorithms.contact_detection import (
    REGION_JOINT_NAMES,
    REGION_JOINTS,
    REGION_NAMES,
    ContactEvent,
    consolidate_overlapping_events,
    depth_gap_for_joint,
    detect_contact_events,
    per_frame_region_confidence,
    project_to_pixels,
)
from ..helpers.progress_reporter import frame_progress
from ..pipeline_stage_base import cli_entrypoint
from ..progress_tracker import RunRecord, StageName
from ..stages.stage_1_mask_and_track import OUTPUT_HUMAN_MASKS, OUTPUT_OBJECT_MASKS
from ..stages.stage_5_retarget_hands import OUTPUT_RETARGET_MOTION

# stage_0_ingest_video.py's own output key, consumed here (same convention
# stage_1_mask_and_track.py uses for the same key).
FRAMES_DIR_OUTPUT_KEY = "frames_dir"

# Repo root is 2 levels up from this file (stages/ -> pipeline/ -> root). Same
# SMPL-X model file the other stages use.
SMPLX_MODEL_PATH = Path(__file__).resolve().parents[2] / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
SMPLX_MODEL_TYPE = "smplx"
SMPLX_GENDER = "neutral"
SMPLX_NUM_BETAS = 10

CONTACTS_DIRNAME = f"stage{StageName.STAGE_7_ANNOTATE_CONTACTS.stage_number}_contacts"
CONTACT_EVENTS_FILENAME = "contact_events.json"
CONTACTS_PREVIEW_DIRNAME = "contacts_preview"

# Preview circle styling -- BGR (cv2's own channel order), a bright cyan/orange
# that reads clearly against most skin tones and object colors alike.
CONTACT_PREVIEW_CIRCLE_COLOR = (0, 220, 255)
CONTACT_PREVIEW_CIRCLE_RADIUS = 12
CONTACT_PREVIEW_CIRCLE_THICKNESS = 3

# This stage's own progress.json output keys.
OUTPUT_CONTACT_EVENTS = "contact_events"
OUTPUT_CONTACTS_PREVIEW = "contacts_preview"

# Below this mean confidence, an event is flagged as low-confidence rather than
# trusted outright -- see this module's docstring for why there's no actual
# interactive prompt here, just a passive data flag.
LOW_CONFIDENCE_THRESHOLD = 0.85

# Beyond this metric depth gap (meters), a 2D-detected event is confidently
# incidental occlusion, not real contact (see depth_gap_for_joint's own
# docstring), and _verify_events_with_depth drops it outright -- DA3 has
# already resolved the ambiguity, so there's nothing left to review. At or
# below it, the gap is treated as confirming real contact, which overrides a
# merely-noisy 2D confidence score in `_event_to_dict`. An initial estimate:
# on real test data, genuine contact measured 0.01-0.05m and false-positive
# occlusion measured 0.25-1.7m, a wide enough margin that this cutoff isn't
# finely tuned -- revisit if a real clip ever lands in between.
DEPTH_GAP_OCCLUSION_THRESHOLD_M = 0.15


def _all_frame_joints(motion: dict) -> np.ndarray:
    """Forward-kinematics the full retargeted body+hands sequence in one
    batched call. Returns (F, J, 3) camera-space joint positions -- GVHMR/
    HaMeR's own space, not real-world/depth-aligned (see this module's
    docstring)."""
    import smplx

    n_frames = motion[KEY_GLOBAL_ORIENT].shape[0]
    model = smplx.create(
        str(SMPLX_MODEL_PATH), model_type=SMPLX_MODEL_TYPE, gender=SMPLX_GENDER,
        num_betas=SMPLX_NUM_BETAS, use_pca=False, flat_hand_mean=True, batch_size=n_frames,
    )
    output = model(
        global_orient=motion[KEY_GLOBAL_ORIENT].float(),
        body_pose=motion[KEY_BODY_POSE].float(),
        betas=motion[KEY_BETAS].float(),
        transl=motion[KEY_TRANSL].float(),
        left_hand_pose=motion[KEY_LEFT_HAND_POSE].float(),
        right_hand_pose=motion[KEY_RIGHT_HAND_POSE].float(),
    )
    return output.joints.detach().numpy()


class _LazyMaskLoader:
    """`[frame_idx] -> (H, W) bool mask at native resolution, or None`,
    decoded one frame at a time instead of unpacking a whole clip's worth of
    native-resolution masks into memory up front (mirrors
    `sam31_adapter._LazyFrameLoader`'s own fix for the same class of bug in
    stage 1).

    The eager version of this (unpack every frame, resize to native, hold
    the whole list) is a real, demonstrated memory bottleneck at high
    resolution: a single native mask is ~7.9MB at 3840x2160, and this stage
    needs two independent mask tracks (object + human) -- on a real
    675-frame 4K clip that's over 10GB held simultaneously, which caused a
    hard Windows access violation (not a clean Python `MemoryError`) later
    in this same stage, on a 16GB machine, right after GVHMR and HaMeR had
    already run in the same process. The packed tensor this reads from is
    both bit-packed (8 pixels/byte, `sam31_tracker.pack_masks`) AND stored
    at SAM 3.1's own small working resolution, not native -- tiny by
    comparison, so keeping it packed until each frame is actually needed
    removes the bottleneck entirely; each frame's mask is used at most twice
    (once during detection, once more only if that exact frame becomes an
    event's `peak_frame` during depth verification), so re-decoding on
    access rather than caching costs nothing meaningful.
    """

    def __init__(self, masks_path: str, n_frames: int, native_hw: tuple[int, int]):
        self._packed = torch.load(masks_path, weights_only=False)[KEY_PACKED_MASKS]
        self._n_frames = n_frames
        self._native_hw = native_hw

    def __len__(self) -> int:
        return self._n_frames

    def __getitem__(self, index: int) -> np.ndarray | None:
        if index >= self._packed.shape[0]:
            return None
        unpacked = unpack_masks(self._packed[index])[0]
        if not unpacked.any():
            return None
        height, width = self._native_hw
        return cv2.resize(
            unpacked.numpy().astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)


def _event_to_dict(event: ContactEvent) -> dict:
    depth_confirms_contact = event.depth_gap_m is not None and event.depth_gap_m <= DEPTH_GAP_OCCLUSION_THRESHOLD_M
    is_low_confidence = event.mean_confidence < LOW_CONFIDENCE_THRESHOLD and not depth_confirms_contact
    return {**asdict(event), "is_low_confidence": is_low_confidence}


def _verify_events_with_depth(
    events: list[ContactEvent], joints: np.ndarray, K: np.ndarray,
    object_masks: _LazyMaskLoader, human_masks: _LazyMaskLoader,
    frames_dir: Path, native_hw: tuple[int, int],
) -> list[ContactEvent]:
    """Second-pass verification, only for already-detected candidate events
    (never the whole clip): at each event's own peak_frame, runs
    Depth-Anything-3 fresh on just that one frame and measures the metric gap
    between the object and the body surface at the contact joint's
    projection (`contact_detection.depth_gap_for_joint`). Returns the events
    that survive: a gap over `DEPTH_GAP_OCCLUSION_THRESHOLD_M` means the 2D
    mask overlap that triggered detection was confidently incidental
    occlusion, not real contact, so that event is dropped outright rather
    than kept around flagged as uncertain -- DA3 already resolved it, there's
    nothing left to flag. An event whose masks were missing that frame
    (can't verify either way) is kept as-is, its `depth_gap_m` left None, so
    `_event_to_dict` falls back to confidence alone. Loads the DA3 model once
    for the whole batch of events, not once per event.
    """
    if not events:
        return events

    focal_length_px = K[0, 0]
    adapter = DepthAnything3Adapter()
    adapter.load()
    try:
        survivors = []
        for event in events:
            frame = event.peak_frame
            object_mask = object_masks[frame]
            human_mask = human_masks[frame]
            if object_mask is None or human_mask is None:
                survivors.append(event)
                continue

            frame_path = frames_dir / f"{frame:06d}.jpg"
            result = adapter.infer(str(frame_path), focal_length_px)
            depth = cv2.resize(
                result[KEY_DEPTH], (native_hw[1], native_hw[0]), interpolation=cv2.INTER_LINEAR,
            )

            region = event.regions[0]
            joint_id = REGION_JOINTS[region][REGION_JOINT_NAMES[region].index(event.joint)]
            joint_pixel = project_to_pixels(joints[frame, joint_id:joint_id + 1], K)[0]
            event.depth_gap_m = depth_gap_for_joint(depth, object_mask, human_mask, joint_pixel)
            if event.depth_gap_m is not None and event.depth_gap_m > DEPTH_GAP_OCCLUSION_THRESHOLD_M:
                continue  # confidently occlusion, not contact -- drop
            survivors.append(event)
    finally:
        adapter.unload()
    return survivors


def _render_contacts_preview(
    events: list[ContactEvent], joints: np.ndarray, K: np.ndarray, frames_dir: Path, out_dir: Path,
) -> None:
    """One annotated JPEG per contact event, at that event's own peak-
    confidence frame -- a human-reviewable spot-check, since this stage has no
    learned model whose confidence can otherwise be sanity-checked visually.
    Draws a circle around whichever candidate joint triggered the event,
    projected to its actual pixel location via the same intrinsics used for
    detection, on top of the real source frame (stage 0's own extracted
    frames, not a mask or a rendered scene).

    Clears out `out_dir` first: unlike this stage's other outputs, the number
    of preview images is data-dependent (one per surviving event), so a
    re-run producing fewer events than a previous one would otherwise leave
    stale images from the removed events sitting alongside the current ones.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        region = event.regions[0]  # every region in the list resolves to the same joint index
        joint_names = REGION_JOINT_NAMES[region]
        joint_id = REGION_JOINTS[region][joint_names.index(event.joint)]
        pixel = project_to_pixels(joints[event.peak_frame, joint_id:joint_id + 1], K)[0]

        image = cv2.imread(str(frames_dir / f"{event.peak_frame:06d}.jpg"))
        if image is None:
            continue

        center = (int(round(pixel[0])), int(round(pixel[1])))
        cv2.circle(image, center, CONTACT_PREVIEW_CIRCLE_RADIUS, CONTACT_PREVIEW_CIRCLE_COLOR, CONTACT_PREVIEW_CIRCLE_THICKNESS)
        label = f"{'+'.join(event.regions)} ({event.joint})"
        cv2.putText(
            image, label, (center[0] + CONTACT_PREVIEW_CIRCLE_RADIUS + 4, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONTACT_PREVIEW_CIRCLE_COLOR, 2, cv2.LINE_AA,
        )

        out_path = out_dir / f"{event.peak_frame:06d}_{'+'.join(event.regions)}.jpg"
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def run(runRecord: RunRecord) -> dict[str, str]:
    motion = torch.load(
        runRecord.stages[StageName.STAGE_5_RETARGET_HANDS].outputs[OUTPUT_RETARGET_MOTION], weights_only=False,
    )
    joints = _all_frame_joints(motion)
    n_frames = joints.shape[0]
    frames_dir = Path(runRecord.stages[StageName.STAGE_0_INGEST_VIDEO].outputs[FRAMES_DIR_OUTPUT_KEY])

    events: list[ContactEvent] = []
    K = None
    stage_1_outputs = runRecord.stages[StageName.STAGE_1_MASK_AND_TRACK].outputs
    if OUTPUT_OBJECT_MASKS in stage_1_outputs:
        native_hw = (runRecord.scene.height, runRecord.scene.width)
        object_masks = _LazyMaskLoader(stage_1_outputs[OUTPUT_OBJECT_MASKS], n_frames, native_hw)
        human_masks = _LazyMaskLoader(stage_1_outputs[OUTPUT_HUMAN_MASKS], n_frames, native_hw)
        K = np.array(runRecord.scene.intrinsics_K)

        region_confidence = {region: np.zeros(n_frames) for region in REGION_NAMES}
        region_joint_idx = {region: np.full(n_frames, -1, dtype=int) for region in REGION_NAMES}

        for f in frame_progress(range(n_frames), total=n_frames, label=StageName.STAGE_7_ANNOTATE_CONTACTS.label):
            for region, (confidence, joint_idx) in per_frame_region_confidence(joints[f], K, object_masks[f]).items():
                region_confidence[region][f] = confidence
                region_joint_idx[region][f] = joint_idx

        for region in REGION_NAMES:
            events.extend(detect_contact_events(
                region_confidence[region], region_joint_idx[region], region, REGION_JOINT_NAMES[region],
            ))

        events = consolidate_overlapping_events(events)
        events = _verify_events_with_depth(events, joints, K, object_masks, human_masks, frames_dir, native_hw)

    contacts_dir = Path(runRecord.progress_dir) / CONTACTS_DIRNAME
    contacts_dir.mkdir(parents=True, exist_ok=True)
    events_path = contacts_dir / CONTACT_EVENTS_FILENAME
    events_path.write_text(json.dumps([_event_to_dict(e) for e in events], indent=2))

    outputs = {OUTPUT_CONTACT_EVENTS: str(events_path)}

    if runRecord.input.render_contacts_preview and events and K is not None:
        preview_dir = contacts_dir / CONTACTS_PREVIEW_DIRNAME
        _render_contacts_preview(events, joints, K, frames_dir, preview_dir)
        outputs[OUTPUT_CONTACTS_PREVIEW] = str(preview_dir)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_7_ANNOTATE_CONTACTS)
