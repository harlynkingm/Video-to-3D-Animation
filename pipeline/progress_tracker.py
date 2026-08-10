"""Tracks a pipeline run's progress in a single JSON file (progress.json).

This is what makes a run resumable: each stage checks the progress record
before running to see whether its dependencies are already complete, and
updates it when it finishes. Killing the process and rerunning the same
command picks up where it left off instead of starting over.
"""

from __future__ import annotations

import argparse
import enum
import json
import time
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1

FIELD_INPUT = "input"
FIELD_SCENE = "scene"
FIELD_STAGES = "stages"
FIELD_STATUS = "status"
FIELD_OUTPUTS = "outputs"
FIELD_OBJECT_SHAPE_HINT = "object_shape_hint"

PROGRESS_JSON_NAME = "progress.json"


class StageName(enum.StrEnum):
    def __new__(cls, value: str, label: str, stage_number: int) -> StageName:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.label = label
        obj.stage_number = stage_number
        return obj

    STAGE_0_INGEST_VIDEO = "ingest_video", "initial stage: process video", 0
    STAGE_1_MASK_AND_TRACK = "mask_and_track", "stage 1: generate masks", 1
    STAGE_1A_HUMAN_MASK = "generate_human_mask", "stage 1: generate human mask", 1
    STAGE_1B_OBJECT_MASK = "generate_object_mask", "stage 1: generate object mask", 1
    STAGE_2_ESTIMATE_HUMAN_MOTION = "estimate_human_motion", "stage 2: estimate human motion", 2
    STAGE_3_ESTIMATE_DEPTH = "estimate_depth", "stage 3: estimate scene depth", 3
    STAGE_4_ESTIMATE_HANDS = "estimate_hands", "stage 4: estimate hands motion", 4
    STAGE_5_RETARGET_HANDS = "retarget_hands","stage 5: attach hands to body", 5
    STAGE_6_ALIGN_SCENE_SCALE = "align_scene_scale", "stage 6: detect scene scale", 6
    STAGE_6B_SAMPLE_OBJECT_SHAPE = "sample_object_shape", "stage 6: detect object shape", 6
    STAGE_7_ANNOTATE_CONTACTS = "annotate_contacts", "stage 7: detect object interactions", 7
    STAGE_8_OPTIMIZE_HOI = "optimize_hoi", "stage 8: optimize object interactions", 8
    STAGE_9_CAPTURE_FACE = "capture_face", "stage 9: face capture", 9
    STAGE_10_EXPORT = "export", "stage 10: export animation", 10
    STAGE_10B_ALIGN_BONES = "align_bones", "stage 10: align bones for export", 10
    STAGE_10C_CONTINUITY = "continuity_fix", "stage 10: clean up motion and rotation continuity", 10
    STAGE_10D_ATTACH_TRACKED_OBJECT = "attach_tracked_object", "stage 10: attach object for export", 10
    STAGE_10E_SAVE_FILE = "save_file", "stage 10: save exported file", 10

def ordered_stages() -> list[StageName]:
    """One `StageName` per distinct `.stage_number`, in ascending stage-number
    order. Picks whichever member is declared FIRST
    for a given number, so each stage's own top-level member must stay
    declared before any of its sub-labels above for this to keep picking the
    right one.
    """
    stage_numbers: set[int] = set()
    stages: list[StageName] = []
    for stage in StageName:
        if stage.stage_number not in stage_numbers:
            stage_numbers.add(stage.stage_number)
            stages.append(stage)
    return stages


def stage_by_number(stage_number: int) -> StageName | None:
    """The single top-level stage (see `ordered_stages()`) with this
    `.stage_number`, or `None` if no stage has it.
    """
    return next((stage for stage in ordered_stages() if stage.stage_number == stage_number), None)


# The pipeline's dependency DAG. Stages not yet implemented are included here too
# (as pending, never-run records) so the full chain is visible in `progress.json`
# from the start, and so a later stage's `depends_on` doesn't need editing in once
# its own file is finally written.
STAGE_DEPENDS_ON: dict[StageName, list[StageName]] = {
    StageName.STAGE_0_INGEST_VIDEO: [],
    StageName.STAGE_1_MASK_AND_TRACK: [StageName.STAGE_0_INGEST_VIDEO],
    StageName.STAGE_2_ESTIMATE_HUMAN_MOTION: [StageName.STAGE_0_INGEST_VIDEO, StageName.STAGE_1_MASK_AND_TRACK],
    StageName.STAGE_3_ESTIMATE_DEPTH: [StageName.STAGE_0_INGEST_VIDEO, StageName.STAGE_1_MASK_AND_TRACK],
    # estimate_hands needs the person mask (stage 1) to locate the person, its
    # own ViTPose pass to locate the hands, AND the body motion (stage 2): a
    # hand is an extension of the arm, not an independent tracked object, and
    # stage 4 checks every raw wrist estimate against GVHMR's own elbow
    # orientation for biomechanical plausibility before its own smoothing runs
    # (see stage_4_estimate_hands.py's module docstring). This is a real
    # kinematic dependency, not just an implementation convenience, so hands and
    # body motion no longer run in parallel.
    StageName.STAGE_4_ESTIMATE_HANDS: [
        StageName.STAGE_0_INGEST_VIDEO,
        StageName.STAGE_1_MASK_AND_TRACK,
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
    ],
    # retarget_hands attaches the stage-4 hands onto the stage-2 body -- it needs
    # both, and nothing else.
    StageName.STAGE_5_RETARGET_HANDS: [
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
        StageName.STAGE_4_ESTIMATE_HANDS,
    ],
    # align_scene_scale's eventual DAG position routes its SMPL-X input through
    # retarget_hands, but scene *scale* only needs the body's overall size,
    # which the body-only estimate_human_motion already gives -- so it depends
    # on that directly and does not wait on the (not-yet-built) hand stages.
    StageName.STAGE_6_ALIGN_SCENE_SCALE: [
        StageName.STAGE_1_MASK_AND_TRACK,
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
        StageName.STAGE_3_ESTIMATE_DEPTH,
    ],
    # annotate_contacts detects hand-to-object proximity in 2D image space,
    # using the retargeted hand joints (stage 5) and the object's own 2D mask
    # (stage 1) -- deliberately NOT align_scene_scale's 3D object primitive or
    # real-world depth, since GVHMR's own Z estimate is measurably unreliable
    # for a reaching/foreshortened arm and an animation only needs the hand and
    # object to agree with each other in their own shared space, not with
    # absolute ground truth (see contact_detection.py's module docstring). Also
    # reads stage 2 directly (not just transitively via stage 5) for its own
    # pre-foot-lock incam translation -- see stage_7_annotate_contacts.py's own
    # module docstring for why contact detection wants that instead of stage
    # 5's (foot-lock-corrected) translation.
    StageName.STAGE_7_ANNOTATE_CONTACTS: [
        StageName.STAGE_1_MASK_AND_TRACK,
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
        StageName.STAGE_5_RETARGET_HANDS,
    ],
    # optimize_hoi solves the tracked object's own per-frame pose: rigidly
    # attached to a body joint during a qualifying contact event (needs
    # annotate_contacts), against the object's fitted shape and the scale/
    # translation that reconciles depth with the body (needs
    # align_scene_scale). It never touches retarget_hands directly -- the
    # body/hand motion is trusted as-is, not refined.
    StageName.STAGE_8_OPTIMIZE_HOI: [
        StageName.STAGE_6_ALIGN_SCENE_SCALE,
        StageName.STAGE_7_ANNOTATE_CONTACTS,
    ],
    # capture_face locates the face itself (its own ViTPose pass, same as
    # DECA/MICA/MediaPipe each doing their own crop) so frames + the human
    # mask are its only *required* inputs. Not a data dependency beyond
    # stage 2 itself; dict insertion order alone is what makes it run before
    # export.
    StageName.STAGE_9_CAPTURE_FACE: [
        StageName.STAGE_0_INGEST_VIDEO,
        StageName.STAGE_1_MASK_AND_TRACK,
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
    ],
    # export also reads retarget_hands' body motion (stage 5) and
    # align_scene_scale's object shape (stage 6) directly, but doesn't list
    # either here -- optimize_hoi (its only real dependency now, for the
    # object's real per-frame pose) already transitively guarantees both are
    # complete by the time it finishes: directly via align_scene_scale, and
    # via annotate_contacts -> retarget_hands. capture_face IS listed
    # directly, though: it's a DAG sibling of optimize_hoi (nothing upstream
    # of export transitively depends on it), and export now reads its
    # face_motion.npz output (jaw pose + bridge-1-mapped expression) whenever
    # face capture wasn't skipped.
    StageName.STAGE_10_EXPORT: [
        StageName.STAGE_8_OPTIMIZE_HOI,
        StageName.STAGE_9_CAPTURE_FACE,
    ],
}


class StageStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ObjectShapeHint(enum.StrEnum):
    AUTO = "auto"
    BOX = "box"
    ELLIPSOID = "ellipsoid"
    CYLINDER = "cylinder"


def cli_field(
    *,
    flag: str,
    short_flag: str | None = None,
    default: Any = MISSING,
    help: str = "",
    required: bool = False,
    bool_flag: bool = False,
    is_render_preview: bool = True,
    value_type: Callable[[str], Any] = str,
    choices: list[str] | None = None,
    parse: Callable[[str], Any] | None = None,
    metavar: str | None = None,
) -> Any:
    """Declares a dataclass field's CLI presentation (flag name, help text,
    choices, type) as part of the field's own declaration, instead of a
    separate hand-maintained list living elsewhere -- `add_dataclass_cli_arguments`/
    `build_dataclass_from_args` below read this metadata back via
    `dataclasses.fields(...)`, so a new field only needs to be declared once,
    here, to show up correctly everywhere it's used. Not `RunInput`-specific:
    `RunLocation`/`NewRunLocation` below use it too, for the `-o/--output-dir`
    and `--run-id` flags shared across `create_run`/`pipeline.run`/`update_run`.

    `bool_flag=True` fields (`RunInput`'s `--render-*-preview` flags) double
    as `--render-previews`-shorthand members automatically in
    `run_input_from_args`/`apply_run_input_overrides`. `is_render_preview=False`
    excludes a boolean field from that sweep

    `parse`, if given, converts the raw parsed string into the field's real
    type (e.g. `ObjectShapeHint`). It's applied by `build_dataclass_from_args`/
    `update_run`, not registered as argparse's own `type=`, so a `None` value
    (meaning "flag not passed", in `update_run`'s optional mode) never runs
    through it.
    """
    metadata = {"cli": {
        "flag": flag, "short_flag": short_flag, "help": help, "required": required, "bool_flag": bool_flag,
        "is_render_preview": is_render_preview,
        "value_type": value_type, "choices": choices, "parse": parse, "metavar": metavar,
    }}
    if default is MISSING:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


@dataclass
class RunInput:
    video_path: str = cli_field(
        flag="--input-video", metavar="INPUT_VIDEO", required=True,
        help="Path to the source video file (MP4, MOV, MPEG, FLV, or WMV), or a directory of "
             "already-extracted JPEG/PNG frames (sorted by filename) -- "
             "--source-fps is required for the directory case.",
    )
    human_prompt: str = cli_field(flag="--human-prompt", required=True, help='e.g. "a person"')
    object_prompt: str | None = cli_field(
        flag="--object-prompt", default=None, help='e.g. "a teddy bear" (omit if there is no object)',
    )
    object_shape_hint: ObjectShapeHint = cli_field(
        flag="--object-shape-hint", default=ObjectShapeHint.AUTO, parse=ObjectShapeHint,
        choices=[hint.value for hint in ObjectShapeHint],
    )
    # Camera intrinsics: either lens/sensor specs (the common case -- a real
    # phone/camera shoot, no calibration available) or a raw K matrix (real
    # calibration data, e.g. a research dataset's own published intrinsics --
    # also strictly more accurate than the lens-spec path even when both are
    # available, since compute_intrinsics_matrix assumes a perfectly centered
    # principal point, which a raw K doesn't have to). Exactly one path must
    # be given -- see validate_camera_input, called by both create_run.py and
    # pipeline.run.py's own main()s (the two places a fresh RunInput gets
    # built from CLI args).
    focal_length_mm: float = cli_field(flag="--focal-length-mm", default=0.0, required=False, value_type=float)
    sensor_width_mm: float = cli_field(flag="--sensor-width-mm", default=0.0, required=False, value_type=float)
    intrinsics_k: list[list[float]] | None = cli_field(
        flag="--intrinsics-k", default=None, value_type=json.loads,
        help='Raw 3x3 intrinsics matrix as JSON, e.g. \'[[fx,0,cx],[0,fy,cy],[0,0,1]]\' -- '
             "an alternative to --focal-length-mm/--sensor-width-mm for real calibration data.",
    )
    # Only meaningful when video_path (above) is a directory of images rather
    # than a real video file: a frame sequence has no embedded frame rate the
    # way a video container does, so this is the only way stage 0 can know
    # it. Ignored for a real video file (its own container fps is used
    # instead). Enforced required-when-a-directory-is-given by
    # validate_video_input, not argparse itself -- same shape as the camera-
    # intrinsics fields above and their own validate_camera_input.
    source_fps: float | None = cli_field(
        flag="--source-fps", default=None, value_type=float,
        help="Frame rate for an --input-video image-sequence directory. This is required in that case, but ignored otherwise.",
    )
    anchor_frame_override: int | None = cli_field(flag="--anchor-frame-override", default=None, value_type=int)
    render_mask_previews: bool = cli_field(
        flag="--render-mask-previews", default=False, bool_flag=True,
        help="Stage 1 also writes black/white JPEG mask previews for visual spot-checking",
    )
    render_motion_preview: bool = cli_field(
        flag="--render-motion-preview", default=False, bool_flag=True,
        help="Stage 2 also writes an AMASS .npz importable into Blender for visual spot-checking",
    )
    render_depth_preview: bool = cli_field(
        flag="--render-depth-preview", default=False, bool_flag=True,
        help="Stage 3 also writes a colored .ply point cloud importable into Blender for visual spot-checking",
    )
    render_scene_preview: bool = cli_field(
        flag="--render-scene-preview", default=False, bool_flag=True,
        help="Stage 6 also writes a .ply combining human, object, and scene in one aligned space for visual spot-checking",
    )
    render_hands_preview: bool = cli_field(
        flag="--render-hands-preview", default=False, bool_flag=True,
        help="Stage 4 also writes a .bvh hand skeleton animation importable into Blender for visual spot-checking",
    )
    render_retarget_preview: bool = cli_field(
        flag="--render-retarget-preview", default=False, bool_flag=True,
        help="Stage 5 also writes a .bvh full-body-plus-hands skeleton importable into Blender for visual spot-checking",
    )
    render_contacts_preview: bool = cli_field(
        flag="--render-contacts-preview", default=False, bool_flag=True,
        help="Stage 7 also writes one annotated JPEG per contact event for visual spot-checking",
    )

    # Stage 1's tracker can lose and re-detect the same physical object (or the
    # human) several times in one clip, each re-detection landing as its own
    # internal track with its own (lower, first-detection) confidence -- bridges
    # a real gap of up to this many frames between two such tracks by holding
    # the last tracked mask forward, so a brief re-occlusion doesn't leave a
    # true empty gap once the object is back in view. First-pass value (~0.5s
    # at 30fps), calibrated against one real clip's own observed gaps (2-59
    # frames between re-detections); longer gaps stay genuinely empty rather
    # than holding a stale mask across an uncertain-duration real occlusion.
    # See `sam31_adapter._stitch_tracked_slots`.
    sam_track_max_bridge_frames: int = 15

    # A single anchor frame's own object-shape fit can be badly wrong when
    # every frame of a clip is either motion-blurred (in flight) or occluded
    # (gripped) -- confirmed on a real clip: three different
    # anchor-frame-selection heuristics (largest mask area, visual stillness,
    # highest 2D mask circularity) each produced a wrong overall size (over-
    # elongated or flattened-to-a-disc). `align_scene_scale` corrects the
    # anchor fit's own proportions (not its orientation/center, which stay
    # from the anchor) using the median of this many independently-measured
    # candidate frames' own 2D mask extents (see
    # `stage_6_align_scene_scale._aggregate_object_shape_proportions`),
    # ranked by how unoccluded they look -- median specifically because real
    # measurements showed errors in both directions (blur inflates,
    # occlusion shrinks), so a robust central estimate rejects outliers
    # either way, unlike mean or max. First-pass value, one clip.
    object_shape_candidate_frames: int = 15

    # Temporal-smoothing knobs. Not exposed as create_run CLI flags on purpose --
    # the defaults are tuned to need no adjustment; a power user can override them
    # by hand-editing these fields in a run's progress.json before running stage
    # 2/4. Body needs only light polish (GVHMR already runs a temporal model over
    # the whole clip); the hands are far jitterier because HaMeR infers each frame
    # independently. See pipeline/algorithms/motion_smoothing.py.
    body_smoothing_window: int = 9
    body_translation_cutoff: float = 0.15

    # Both hand parts (finger articulation and wrist orientation) go through the
    # same three-pass chain, differing only in the per-part knobs below:
    #   1. savgol pre-pass (`hand_smoothing_window`) -- zero-phase, no lag; knocks
    #      down HaMeR's broadband per-frame jitter, the temporal pre-conditioning
    #      GVHMR gives the body for free but HaMeR never does.
    #   2. one-euro adaptive filter (`*_min_cutoff_hz`, shared `hand_beta`) -- holds
    #      a nearly-still joint tight (killing the rest-state wobble/precession the
    #      savgol pass leaves) and loosens automatically once it moves. It replaced
    #      an earlier hard hold/snap deadzone that snapped visibly on release.
    #   3. decimation (`*_decimate_deg`) -- keyframe reduction in quaternion space:
    #      refit the curve through a sparse set of keyframes so the result is
    #      mathematically smooth between them, removing residual jitter outright
    #      rather than just averaging it down. Without pass 2 first, decimation
    #      would place keyframes on the noise and lock it in.
    # The wrist gets a lower min_cutoff (heavier hold) and a looser decimate
    # tolerance than the fingers: it starts from noisier global-orientation data
    # and is a load-bearing joint, so it needs more smoothing. Tuned on a real
    # clip -- fingers land below the body's own jitter, the wrist a few times above
    # it (its noisier input floors higher without risking arm-detachment lag).
    hand_smoothing_window: int = 15  # savgol pre-pass window, both wrist and fingers
    hand_beta: float = 0.3  # one-euro speed responsiveness, both wrist and fingers
    hand_finger_min_cutoff_hz: float = 0.15
    hand_wrist_min_cutoff_hz: float = 0.10
    hand_finger_decimate_deg: float = 1.5
    hand_wrist_decimate_deg: float = 3.0

    # A real human wrist cannot rotate further than roughly `hand_wrist_max_deviation_deg`
    # relative to the forearm in any direction -- calibrated against two clean,
    # previously-verified real clips (max ever observed combined: ~95 degrees
    # across 1200+ frames of legitimate motion, on two different people/
    # activities). HaMeR sometimes regresses a wrist orientation well past this
    # on an ambiguous frame (e.g. a foreshortened forearm mid-reach, or a
    # genuine rotation-from-monocular-view ambiguity) -- sometimes a slow,
    # smooth drift, sometimes a CHAOTIC stretch bouncing between clearly-
    # implausible values and moderate ones that look individually plausible in
    # isolation (a clean clip's own legitimate motion also reaches ~95-103
    # degrees at its peak). The moderate "shoulder" frames of a chaotic bad
    # stretch would otherwise still anchor the smoothing chain, so an
    # instantaneous-only threshold isn't enough on its own: this uses hysteresis
    # (the same lock/release pattern as a noise gate) -- `_max_deviation_deg` is
    # the strict threshold that seeds detection, then the invalid region expands
    # outward while a `_deviation_window`-frame rolling max stays above the
    # lower `_release_deviation_deg`, so it can only ever expand from a
    # confirmed-bad seed frame; a clip that never crosses the strict threshold
    # is unaffected regardless of the release value. `_deviation_window` needs
    # care too: a window that's too wide relative to a clip's own natural
    # busyness merges the rolling max across genuinely separate local peaks --
    # confirmed on a short, energetic reference clip, where window=7 let one
    # isolated real spike's rolling max touch nearly the whole clip (every
    # frame had SOME elevated neighbor within reach) and reject all of it;
    # window=5 isolates just the frames actually near that spike while still
    # capturing the full multi-frame chaotic stretch on the clip this was
    # designed for. Checked in stage 4 against GVHMR's own elbow orientation
    # (stage 4 depends on stage 2's output for this), before any smoothing
    # runs -- a filter that's already blended a bad value into its neighbors
    # can't be un-blended by a later stage. See
    # hand_retarget.reject_biomechanically_implausible_wrist.
    hand_wrist_max_deviation_deg: float = 110.0
    hand_wrist_release_deviation_deg: float = 55.0
    hand_wrist_deviation_window: int = 5
    # How many frames the rejected region may expand outward from a seed
    # frame, in either direction. Without a cap, a long stretch of real
    # sustained motion that stays above `hand_wrist_release_deviation_deg`
    # (e.g. gripping a strap near the shoulder for several seconds) gets
    # entirely rejected the moment one bad frame anywhere in it seeds
    # detection -- confirmed on a real clip, an 80+ frame stretch reading a
    # smooth, plausible 60-100 degrees throughout was thrown out this way.
    # 10 frames (~0.33s at 30fps) is a first-pass value calibrated against
    # one clip; revisit once more clips are available to check it against.
    hand_wrist_max_expansion_frames: int = 10

    # A separate check from the deviation gate above: catches a HaMeR wrist
    # estimate that flips to a different (wrong) orientation for an isolated
    # frame or two while staying within a plausible magnitude relative to the
    # elbow the whole time -- invisible to the deviation gate, which only
    # looks at the static magnitude, never at how fast it's changing. A real
    # wrist cannot rotate a large fraction of a full turn within one frame.
    # Calibrated against a real clip: genuine fast motion topped out around
    # 40 degrees/frame (1200 degrees/sec at 30fps) even during active
    # reaching, while flip instances read 100-175 degrees/frame (3000+
    # degrees/sec) -- a clean gap between the two. In degrees/second (not
    # degrees/frame) so the same value means the same physical speed
    # regardless of a clip's fps. See
    # hand_retarget.reject_wrist_velocity_spikes.
    hand_wrist_max_velocity_deg_per_sec: float = 2400.0

    # How much of a long invalid wrist stretch (real occlusion, or one
    # rejected by the gates above) is left for straight-line interpolation
    # before the rest gets held at the last known-good orientation instead --
    # confirmed on a real clip, interpolating across a 40-90+ frame gap can
    # visibly sweep through a large, physically-impossible-looking rotation.
    # 15 frames (~0.5s at 30fps) is a first-pass value; revisit once more
    # clips are available to check it against. See
    # motion_smoothing.cap_long_gaps_with_hold.
    hand_wrist_max_bridge_frames: int = 15

    # A third, independent wrist check: how far the hand's own pointing
    # direction (wrist to middle-finger) may swing away from its rest-pose
    # direction before a frame is rejected -- catches a sustained
    # anatomically-impossible pose that changes too slowly to trip the
    # velocity check and reads a moderate enough blended magnitude to dodge
    # the deviation check too. Deliberately measures swing only, never twist
    # (rotation about the hand's own pointing direction) -- twist isn't a
    # reliable signal here, since composing two legitimate swing-only
    # rotations produces large apparent twist as a pure artifact of how
    # compound 3D rotations compose, confirmed both synthetically and on real
    # motion (a strap-grip clip read swing under 30 degrees the whole time
    # while its blended magnitude spiked past 150 from twist alone). ~90-100
    # degrees is roughly where a real wrist starts folding the hand back over
    # the forearm; 95 is a first-pass value from one real clip, not a
    # multi-clip calibration. See hand_retarget.reject_hand_swung_past_forearm.
    hand_wrist_max_swing_deg: float = 95.0

    # A fourth wrist check, genuinely different from the three above: real 3D
    # geometry (does the hand's own reference point sit inside the forearm's
    # fixed rest-pose segment) instead of any rotation angle -- catches a hand
    # folded back into the forearm's own space even when every rotation-based
    # check reads within range. `hand_forearm_interior_max_t` is how far
    # inside the segment (0 = elbow, 1 = wrist) still counts as "inside the
    # forearm" rather than merely "near the wrist," which happens in plenty
    # of normal poses; `hand_forearm_radius_m` additionally requires genuine
    # proximity to the forearm's own axis, not just alignment along its
    # length. Calibrated on a real clip: a confirmed hand-through-forearm
    # stretch dropped to t=0.67-0.99 at 7-11cm from the axis, while 1900+
    # other real frames (including a confirmed genuine extreme-looking grip)
    # never dropped below ~1.1 -- comfortable margin either side of these
    # defaults, but from one clip only. See hand_retarget.reject_hand_
    # through_forearm.
    hand_forearm_interior_max_t: float = 0.95
    hand_forearm_radius_m: float = 0.10


def add_dataclass_cli_arguments(parser: argparse.ArgumentParser, dataclass_type: type, *, required: bool = True) -> None:
    """Registers every `cli_field(...)`-tagged field of `dataclass_type` onto
    `parser`, read directly off each field's own metadata -- the shared
    engine `add_run_input_arguments` (for `RunInput`) builds on, and that
    `create_run`/`pipeline.run`/`update_run` also use directly for
    `RunLocation`/`NewRunID` (the `-o/--output-dir`/`--run-id` flags),
    so no CLI entrypoint hand-writes its own `parser.add_argument` calls for
    a field any of these dataclasses already declares.

    `required=False` (used only by `update_run`, for `RunInput`'s own fields)
    makes every otherwise-required flag optional and switches every flag's
    "not passed" value to `None` instead of a real default -- `update_run`
    treats `None` as "leave this field's existing value alone", so a flag
    only overrides anything when the caller actually passes it. Boolean
    flags can therefore only be turned ON via `update_run`, never explicitly
    back off; that's an acceptable limitation for a dev-only migration tool
    (hand-edit progress.json to turn one off).
    """
    for f in fields(dataclass_type):
        cli = f.metadata.get("cli")
        if cli is None:
            continue  # a field with no CLI presence (RunInput's smoothing knobs)

        flags = (cli["short_flag"], cli["flag"]) if cli["short_flag"] else (cli["flag"],)
        kwargs: dict[str, Any] = {"dest": f.name, "help": cli["help"]}
        if cli["bool_flag"]:
            kwargs["action"] = "store_true"
            kwargs["default"] = False if required else None
        else:
            kwargs["type"] = cli["value_type"]
            kwargs["required"] = cli["required"] and required
            default = f.default
            if isinstance(default, enum.Enum):
                default = default.value
            elif default is MISSING:
                default = None
            kwargs["default"] = default if required else None
            if cli["choices"] is not None:
                kwargs["choices"] = cli["choices"]
            if cli["metavar"] is not None:
                kwargs["metavar"] = cli["metavar"]
        parser.add_argument(*flags, **kwargs)


def resolve_cli_value(f: Any, args: argparse.Namespace) -> Any:
    """The parsed value for one `cli_field(...)`-tagged field, with its
    `parse` transform (if any) applied -- shared by `build_dataclass_from_args`
    and `apply_run_input_overrides` so "convert the raw string" can't drift
    between fresh-construction and override-merge.
    """
    cli = f.metadata["cli"]
    value = getattr(args, f.name)
    if value is not None and cli["parse"] is not None:
        value = cli["parse"](value)
    return value


def build_dataclass_from_args(dataclass_type: type, args: argparse.Namespace) -> Any:
    """Constructs a fresh instance of `dataclass_type` from parsed CLI args --
    any field with no CLI flag is simply omitted here, so it falls back to
    its own dataclass default.
    """
    kwargs: dict[str, Any] = {}
    for f in fields(dataclass_type):
        if f.metadata.get("cli") is None:
            continue
        kwargs[f.name] = resolve_cli_value(f, args)
    return dataclass_type(**kwargs)


def add_run_input_arguments(parser: argparse.ArgumentParser, required: bool = True) -> None:
    """`RunInput`'s own wrapper around `add_dataclass_cli_arguments`, adding
    the one thing that isn't a plain per-field concern: the
    `--render-previews` shorthand, which has no `RunInput` field of its own
    (it toggles several other fields at once) so it can't be declared via
    `cli_field` like everything else.
    """
    add_dataclass_cli_arguments(parser, RunInput, required=required)
    preview_default = False if required else None
    parser.add_argument("--render-previews", action="store_true", default=preview_default,
                         help="Shorthand for every --render-*-preview flag above at once")


def run_input_from_args(args: argparse.Namespace) -> RunInput:
    """Builds a fresh `RunInput` from parsed CLI args, then applies the
    `--render-previews` shorthand on top of the generic per-field
    construction (see `add_run_input_arguments` for why that flag can't just
    be another `cli_field`).
    """
    run_input = build_dataclass_from_args(RunInput, args)
    if args.render_previews:
        for f in fields(RunInput):
            cli = f.metadata.get("cli")
            if cli and cli["bool_flag"] and cli["is_render_preview"]:
                setattr(run_input, f.name, True)
    return run_input


def validate_camera_input(run_input: RunInput) -> str | None:
    """Returns an error message if `run_input`'s camera intrinsics aren't
    resolvable -- neither path given, or both given at once (ambiguous which
    should win) -- else `None`. `focal_length_mm`/`sensor_width_mm` are
    otherwise-required-looking fields that are no longer enforced by argparse
    itself (see their own `cli_field` comment in `RunInput`), so this is the
    one place that actually enforces a fresh run has SOME usable camera
    input. Only meaningful for a *fresh* run: `apply_run_input_overrides`'s
    resume/update path already only touches fields the caller explicitly
    passed, so an existing run's own already-valid camera input is never at
    risk there. Shared by create_run.py and pipeline.run.py's own main()s,
    the two places a fresh RunInput gets built from CLI args."""
    has_k = run_input.intrinsics_k is not None
    has_focal = run_input.focal_length_mm > 0 and run_input.sensor_width_mm > 0
    if has_k and has_focal:
        return "--intrinsics-k cannot be combined with --focal-length-mm/--sensor-width-mm - provide exactly one"
    if not has_k and not has_focal:
        return "camera intrinsics required: either --intrinsics-k, or both --focal-length-mm and --sensor-width-mm"
    return None


def validate_video_input(run_input: RunInput) -> str | None:
    """Returns an error message if `run_input.video_path` is a directory
    (an image-sequence input, see that field's own comment) without a usable
    `--source-fps`, else `None`. A real video file's own container fps makes
    `source_fps` unnecessary, so this only fires for the directory case --
    stage 0 can't derive a frame rate from a folder of otherwise-unordered-
    in-time still images. Same shape as validate_camera_input: only meaningful
    for a *fresh* run, shared by create_run.py and pipeline.run.py's own
    main()s."""
    if not Path(run_input.video_path).is_dir():
        return None
    if run_input.source_fps is None or run_input.source_fps <= 0:
        return "--source-fps is required when --input-video is a directory of images"
    return None


def apply_run_input_overrides(existing: RunInput, args: argparse.Namespace) -> RunInput:
    """Builds an updated `RunInput` via `dataclasses.replace`, applying only
    the flags `args` actually carries a value for (parsed with
    `add_run_input_arguments(parser, required=False)`, where an omitted flag
    comes back as `None`) -- every other field, whether its flag simply
    wasn't passed or it's a smoothing knob with no CLI flag at all, passes
    through from `existing` untouched. Shared by `update_run` (its whole
    purpose) and `pipeline.run`'s resume path (letting a resumed run still
    pick up a flag like `--render-contacts-preview` without needing to
    re-supply the ones that are already fixed, like `--input-video`).
    """
    render_all = args.render_previews
    overrides: dict = {}
    for f in fields(RunInput):
        cli = f.metadata.get("cli")
        if cli is None:
            continue
        value = resolve_cli_value(f, args)
        if cli["bool_flag"]:
            forced_by_render_all = render_all and cli["is_render_preview"]
            if forced_by_render_all or value is not None:
                overrides[f.name] = bool(forced_by_render_all or value)
        elif value is not None:
            overrides[f.name] = value

    return replace(existing, **overrides)


@dataclass
class RunLocation:
    """Where a run lives on disk -- shared by `create_run`, `pipeline.run`,
    and `update_run` via `add_dataclass_cli_arguments`, as opposed to
    `RunInput`'s own video/prompt/camera parameters (which get persisted
    separately, under progress.json's own `input` key, not here).
    """
    progress_dir: Path = cli_field(
        flag="--output-dir", short_flag="-o", metavar="OUTPUT_DIR", required=True, value_type=Path,
        help="Directory for this run's progress.json and outputs",
    )


@dataclass
class NewRunID(RunLocation):
    """Adds `--run-id`, registered only where a run might not exist yet
    (`create_run`, `pipeline.run`) -- an existing run's id is already fixed,
    so `update_run` doesn't offer a way to change it.
    """
    run_id: str | None = cli_field(flag="--run-id", default=None, help="Defaults to --output-dir's own folder name")


@dataclass
class SceneInfo:
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    intrinsics_K: list[list[float]] = field(default_factory=list)
    anchor_frame_index: int = 0


@dataclass
class StageRecord:
    status: StageStatus = StageStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)  # output name -> path
    error: str | None = None


@dataclass
class RunOutputs:
    final_blend: str | None = None


@dataclass
class RunRecord:
    run_id: str
    progress_dir: str
    input: RunInput
    scene: SceneInfo = field(default_factory=SceneInfo)
    stages: dict[str, StageRecord] = field(default_factory=dict)
    outputs: RunOutputs = field(default_factory=RunOutputs)
    schema_version: int = SCHEMA_VERSION
    # Unix timestamps (seconds). created_at is stamped once, by create_run();
    # updated_at is refreshed on every save() below, regardless of call site,
    # so it always reflects when this run last actually progressed. A run
    # directory from before this field existed loads with both at 0.0 (the
    # dataclass default) rather than failing -- there's no real "created"
    # timestamp to recover for those, and 0.0 reads unambiguously as "unknown"
    # rather than a plausible-looking but fabricated date.
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def path(self) -> Path:
        return Path(self.progress_dir) / PROGRESS_JSON_NAME

    def save(self) -> None:
        self.updated_at = time.time()
        data = asdict(self)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(self.path)  # atomic rename on the same filesystem

    @classmethod
    def load(cls, progress_dir: str | Path) -> RunRecord:
        data = json.loads((Path(progress_dir) / PROGRESS_JSON_NAME).read_text())
        data[FIELD_INPUT] = RunInput(
            **{
                **data[FIELD_INPUT],
                FIELD_OBJECT_SHAPE_HINT: ObjectShapeHint(data[FIELD_INPUT][FIELD_OBJECT_SHAPE_HINT]),
            }
        )
        data[FIELD_SCENE] = SceneInfo(**data[FIELD_SCENE])
        data[FIELD_STAGES] = {
            name: StageRecord(**{**rec, FIELD_STATUS: StageStatus(rec[FIELD_STATUS])})
            for name, rec in data[FIELD_STAGES].items()
        }
        data[FIELD_OUTPUTS] = RunOutputs(**data[FIELD_OUTPUTS])
        return cls(**data)

    def is_complete(self, stage_name: StageName) -> bool:
        record = self.stages.get(stage_name)
        return record is not None and record.status == StageStatus.COMPLETE

    def dependencies_met(self, stage_name: StageName) -> bool:
        return all(self.is_complete(dep) for dep in self.stages[stage_name].depends_on)

    def mark_progress(
        self,
        stage_name: StageName,
        status: StageStatus,
        outputs: dict[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        record = self.stages[stage_name]
        record.status = status
        record.error = error
        if outputs is not None:
            record.outputs = outputs
        self.save()

    def update_schema(self) -> None:
        """Migrates this record's `stages` dict to the current `STAGE_DEPENDS_ON`
        DAG in place -- adds a record (defaulted to PENDING) for any stage the
        current code knows about that this run predates, and refreshes
        `depends_on` for every stage (the DAG can change after a stage's own
        record was first written). Never touches an existing record's own
        status/outputs/error, and never removes a stage this run has a record
        for even if the current DAG no longer lists it, so no recorded
        progress is ever discarded. A stale, no-longer-listed stage's record
        just sits unread -- nothing keys off `self.stages` except by a
        `StageName` the current DAG still knows about.

        Deliberately NOT called from `load()` itself: this saves (see below),
        and `load()` must stay a pure read with no disk side effect. Callers that
        load a record to actually *act* on it (`update_run`, `pipeline.run`,
        `pipeline_stage_base.cli_entrypoint`) call this explicitly instead, once,
        right after their own `load()` -- safe there because each is a one-shot
        call at the start of a single process's lifetime, not a poll.
        """
        for stage, deps in STAGE_DEPENDS_ON.items():
            depends_on = [dep.value for dep in deps]
            if stage.value in self.stages:
                self.stages[stage.value].depends_on = depends_on
            else:
                self.stages[stage.value] = StageRecord(depends_on=depends_on)

        self.schema_version = SCHEMA_VERSION
        self.save()
