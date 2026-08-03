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

A large depth gap does NOT necessarily mean incidental occlusion, though: a
real grip commonly wraps the touching joint around or behind the object's
near-facing visible surface (the surface its own mask actually shows), which
a single frame's monocular depth can read as a genuine multi-centimeter gap
even during unambiguous, sustained contact. So `_verify_events_with_depth`
never drops an event outright on a large gap -- it flags it
`is_low_confidence` instead (same flag a low, unsettled 2D confidence score
gets), leaving the actual accept/reject call to whatever consumes that flag,
rather than silently discarding real interactions the depth check got wrong.
A small gap remains strong positive evidence of contact and overrides a
merely-noisy 2D confidence score. If `RunInput.render_contacts_preview` is
set, also writes one annotated JPEG per event (see
`_render_contacts_preview`), colored and labeled with its confidence
percentage, as a visual spot-check of the same signal -- low-confidence
events are still used exactly like any other event, this is purely for a
human who chooses to go look.
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

# The SMPL-X neutral mesh's own fixed topology (vertex indices are the same
# across every body shape/pose, only the topology itself would change them)
# -- found empirically as the highest (+Y, this project's up axis) vertex
# within 15cm of the rest-pose HEAD_JOINT, i.e. the top of the scalp. Used
# instead of a skeletal joint for the "head_top" region (see
# contact_detection.HEAD_TOP_JOINT_INDEX's own comment for why): a mesh
# vertex rides the actual head surface, so it lands where a hat/cap
# physically rests and stays correct under head tilts/rotations, unlike a
# fixed offset from a joint would.
SMPLX_HEAD_TOP_VERTEX = 9003

CONTACTS_DIRNAME = f"stage{StageName.STAGE_7_ANNOTATE_CONTACTS.stage_number}_contacts"
CONTACT_EVENTS_FILENAME = "contact_events.json"
CONTACTS_PREVIEW_DIRNAME = "contacts_preview"

# Preview circle/text styling -- BGR (cv2's own channel order). A normal event
# gets a bright cyan/orange that reads clearly against most skin tones and
# object colors alike; a low-confidence one gets a distinct red so it stands
# out as worth a second look while flipping through the preview folder.
CONTACT_PREVIEW_CIRCLE_COLOR = (0, 220, 255)
CONTACT_PREVIEW_CIRCLE_COLOR_LOW_CONFIDENCE = (0, 0, 255)
CONTACT_PREVIEW_CIRCLE_RADIUS = 12
CONTACT_PREVIEW_CIRCLE_THICKNESS = 3

# This stage's own progress.json output keys.
OUTPUT_CONTACT_EVENTS = "contact_events"
OUTPUT_CONTACTS_PREVIEW = "contacts_preview"

# Below this mean confidence, an event is flagged as low-confidence rather than
# trusted outright -- purely a passive data flag (see this module's docstring),
# not a gate on whether the event gets used.
LOW_CONFIDENCE_THRESHOLD = 0.85

# An event whose 2D mask overlap stays at/above LOW_CONFIDENCE_THRESHOLD for at
# least this long is trusted outright, without the depth gap able to override
# it -- a person's body part briefly passing in front of/behind a stationary
# object is implausible to hold that level of sustained overlap this long, so
# duration + confidence alone already rule out the incidental-occlusion case
# depth verification exists to catch (see this module's docstring for why the
# depth gap itself is unreliable for a real full-grip contact). Well above
# `stage_8_optimize_hoi.MIN_ATTACHMENT_DURATION_SECONDS` (which only filters
# brief kicks/collisions, a different concern). An initial estimate, same
# caveat as the other thresholds here: not yet calibrated against footage with
# a real brief incidental occlusion to find where this should actually sit.
SUSTAINED_CONTACT_DURATION_SECONDS = 1.0

# Beyond this metric depth gap (meters), a 2D-detected event reads as
# confidently incidental occlusion by depth alone (see depth_gap_for_joint's
# own docstring) -- but a real grip's own near-side-of-object-vs-wrapped-
# behind-it geometry can trigger this just as easily as an actual pass-through
# occlusion can (see this module's docstring), so this only ever flags
# `is_low_confidence`, never drops the event. At or below it, the gap is
# treated as confirming real contact, which overrides a merely-noisy 2D
# confidence score in `_event_to_dict`. An initial estimate: on real test
# data, genuine contact measured 0.01-0.05m and false-positive occlusion
# measured 0.25-1.7m -- but real full-grip contact on other footage has also
# measured up to ~0.39m, overlapping that same "occlusion" range, so this
# cutoff alone can't reliably separate the two on its own.
DEPTH_GAP_OCCLUSION_THRESHOLD_M = 0.15


def _all_frame_joints(motion: dict) -> np.ndarray:
    """Forward-kinematics the full retargeted body+hands sequence in one
    batched call. Returns (F, J, 3) camera-space joint positions -- GVHMR/
    HaMeR's own space, not real-world/depth-aligned (see this module's
    docstring). The last column is not a real skeletal joint: it's
    SMPLX_HEAD_TOP_VERTEX, appended so contact_detection.HEAD_TOP_JOINT_INDEX
    can address it the same way every real joint is addressed."""
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
    joints = output.joints.detach().numpy()
    assert joints.shape[1] == HEAD_TOP_JOINT_INDEX, (
        f"smplx returned {joints.shape[1]} joints, expected {HEAD_TOP_JOINT_INDEX} -- "
        "contact_detection.HEAD_TOP_JOINT_INDEX needs updating to match."
    )
    head_top = output.vertices[:, SMPLX_HEAD_TOP_VERTEX, :].detach().numpy()[:, None, :]
    return np.concatenate([joints, head_top], axis=1)


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


def _is_low_confidence(event: ContactEvent, fps: float) -> bool:
    """A large depth gap flags an event as low-confidence -- unless the
    event's own sustained 2D confidence already rules out incidental
    occlusion on its own (see `SUSTAINED_CONTACT_DURATION_SECONDS`), in which
    case the depth gap is ignored entirely rather than overriding strong 2D
    evidence with an unreliable single-frame reading. Absent a depth reading
    (mask missing that frame), falls back to 2D confidence alone; a small gap
    overrides a merely-noisy 2D score."""
    duration_frames = event.end_frame - event.start_frame + 1
    if event.mean_confidence >= LOW_CONFIDENCE_THRESHOLD and duration_frames >= SUSTAINED_CONTACT_DURATION_SECONDS * fps:
        return False
    if event.depth_gap_m is None:
        return event.mean_confidence < LOW_CONFIDENCE_THRESHOLD
    return event.depth_gap_m > DEPTH_GAP_OCCLUSION_THRESHOLD_M


def _event_to_dict(event: ContactEvent, fps: float) -> dict:
    return {**asdict(event), "is_low_confidence": _is_low_confidence(event, fps)}


def _verify_events_with_depth(
    events: list[ContactEvent], joints: np.ndarray, K: np.ndarray,
    object_masks: _LazyMaskLoader, human_masks: _LazyMaskLoader,
    frames_dir: Path, native_hw: tuple[int, int],
) -> list[ContactEvent]:
    """Second-pass verification, only for already-detected candidate events
    (never the whole clip): at each event's own peak_frame, runs
    Depth-Anything-3 fresh on just that one frame and measures the metric gap
    between the object and the body surface at the contact joint's
    projection (`contact_detection.depth_gap_for_joint`), stashing it on the
    event as `depth_gap_m` for `_event_to_dict`/`_is_low_confidence` to read
    later. Never drops an event -- see this module's docstring for why a
    large gap isn't reliable evidence of incidental occlusion on its own; the
    accept/reject call belongs to whatever consumes `is_low_confidence`, not
    here. An event whose masks were missing that frame (can't verify either
    way) is left with `depth_gap_m` at None, so `_is_low_confidence` falls
    back to 2D confidence alone. Loads the DA3 model once for the whole batch
    of events, not once per event.
    """
    if not events:
        return events

    focal_length_px = K[0, 0]
    adapter = DepthAnything3Adapter()
    adapter.load()
    try:
        for event in events:
            frame = event.peak_frame
            object_mask = object_masks[frame]
            human_mask = human_masks[frame]
            if object_mask is None or human_mask is None:
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
    finally:
        adapter.unload()
    return events


def _render_contacts_preview(
    events: list[ContactEvent], joints: np.ndarray, K: np.ndarray, frames_dir: Path, out_dir: Path, fps: float,
) -> None:
    """One annotated JPEG per contact event, at that event's own peak-
    confidence frame -- a human-reviewable spot-check, since this stage has no
    learned model whose confidence can otherwise be sanity-checked visually.
    Draws a circle around whichever candidate joint triggered the event,
    projected to its actual pixel location via the same intrinsics used for
    detection, on top of the real source frame (stage 0's own extracted
    frames, not a mask or a rendered scene). Labeled with the event's own
    mean confidence as a percentage, and colored/marked distinctly when
    `is_low_confidence` -- purely a visual aid for a human who chooses to
    look, not a gate on which events actually get used (see this module's
    docstring).

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

        low_confidence = _is_low_confidence(event, fps)
        color = CONTACT_PREVIEW_CIRCLE_COLOR_LOW_CONFIDENCE if low_confidence else CONTACT_PREVIEW_CIRCLE_COLOR

        center = (int(round(pixel[0])), int(round(pixel[1])))
        cv2.circle(image, center, CONTACT_PREVIEW_CIRCLE_RADIUS, color, CONTACT_PREVIEW_CIRCLE_THICKNESS)
        label = f"{'+'.join(event.regions)} ({event.joint})"
        confidence_label = f"confidence: {event.mean_confidence * 100:.0f}%"
        if low_confidence:
            confidence_label += " (LOW CONFIDENCE)"
        text_origin = (center[0] + CONTACT_PREVIEW_CIRCLE_RADIUS + 4, center[1])
        cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(
            image, confidence_label, (text_origin[0], text_origin[1] + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
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
    events_path.write_text(json.dumps([_event_to_dict(e, runRecord.scene.fps) for e in events], indent=2))

    outputs = {OUTPUT_CONTACT_EVENTS: str(events_path)}

    if runRecord.input.render_contacts_preview and events and K is not None:
        preview_dir = contacts_dir / CONTACTS_PREVIEW_DIRNAME
        _render_contacts_preview(events, joints, K, frames_dir, preview_dir, runRecord.scene.fps)
        outputs[OUTPUT_CONTACTS_PREVIEW] = str(preview_dir)

    return outputs


if __name__ == "__main__":
    cli_entrypoint(run, stage_name=StageName.STAGE_7_ANNOTATE_CONTACTS)
