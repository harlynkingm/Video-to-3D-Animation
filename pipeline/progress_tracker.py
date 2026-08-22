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

SCHEMA_VERSION = 2

FIELD_INPUT = "input"
FIELD_SCENE = "scene"
FIELD_STAGES = "stages"
FIELD_STATUS = "status"
FIELD_OUTPUTS = "outputs"
FIELD_OBJECT_SHAPE_HINT = "object_shape_hint"
FIELD_FINGER_MOTION = "finger_motion"
FIELD_FINE_TUNING = "fine_tuning"
FIELD_FINE_TUNING_OVERRIDES = "fine_tuning_overrides"

PROGRESS_JSON_NAME = "progress.json"

# On Windows, a just-written file can briefly be held by the system
# Keep the write atomic, but give that transient lock a moment to
# clear before reporting a real save failure.
PROGRESS_REPLACE_MAX_ATTEMPTS = 6
PROGRESS_REPLACE_INITIAL_RETRY_SECONDS = 0.05


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
    # retarget_hands attaches the stage-4 hands onto the stage-2 body, it needs
    # both, and nothing else.
    StageName.STAGE_5_RETARGET_HANDS: [
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
        StageName.STAGE_4_ESTIMATE_HANDS,
    ],
    # align_scene_scale's eventual DAG position routes its SMPL-X input through
    # retarget_hands, but scene *scale* only needs the body's overall size,
    # which the body-only estimate_human_motion already gives, so it depends
    # on that directly and does not wait on the (not-yet-built) hand stages.
    StageName.STAGE_6_ALIGN_SCENE_SCALE: [
        StageName.STAGE_1_MASK_AND_TRACK,
        StageName.STAGE_2_ESTIMATE_HUMAN_MOTION,
        StageName.STAGE_3_ESTIMATE_DEPTH,
    ],
    # annotate_contacts detects hand-to-object proximity in 2D image space,
    # using the retargeted hand joints (stage 5) and the object's own 2D mask
    # (stage 1), deliberately NOT align_scene_scale's 3D object primitive or
    # real-world depth, since GVHMR's own Z estimate is measurably unreliable
    # for a reaching/foreshortened arm and an animation only needs the hand and
    # object to agree with each other in their own shared space, not with
    # absolute ground truth (see contact_detection.py's module docstring). Also
    # reads stage 2 directly (not just transitively via stage 5) for its own
    # pre-foot-lock incam translation, see stage_7_annotate_contacts.py's own
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
    # align_scene_scale). It never touches retarget_hands directly, the
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
    # either here, optimize_hoi (its only real dependency now, for the
    # object's real per-frame pose) already transitively guarantees both are
    # complete by the time it finishes: directly via align_scene_scale, and
    # via annotate_contacts -> retarget_hands. capture_face IS listed
    # directly, though: it's a DAG sibling of optimize_hoi (nothing upstream
    # of export transitively depends on it), and export now reads its
    # face_motion.npz output (jaw pose + SMPL-X-mapped expression) whenever
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


class FingerMotion(enum.StrEnum):
    """The run-level trade-off between stable and highly articulated fingers."""

    SMOOTH = "smooth"
    DETAILED = "detailed"


@dataclass(frozen=True)
class FingerMotionSettings:
    """The base filter controls selected by one semantic finger-motion mode."""

    hand_finger_smoothing_window: int
    hand_finger_beta: float
    hand_finger_derivative_cutoff_hz: float
    hand_finger_min_cutoff_hz: float
    hand_finger_decimate_deg: float


# Smooth deliberately uses the original broad HaMeR cleanup pass: it removes
# most occlusion/interlocking jitter at the cost of small, fast articulations.
# Detailed is the balanced sharp movement profile, retaining a short pre-pass and a
# responsive One Euro path. Fine-tuning pins layer over either base below.
FINGER_MOTION_SETTINGS = {
    FingerMotion.SMOOTH: FingerMotionSettings(15, 0.3, 1.0, 0.15, 1.5),
    FingerMotion.DETAILED: FingerMotionSettings(5, 1.85, 2.75, 0.225, 0.375),
}


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
    separate hand-maintained list living elsewhere, `add_dataclass_cli_arguments`/
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
             "already-extracted JPEG/PNG frames (sorted by filename), "
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
    # HaMeR estimates each frame independently, so a clip with occluded or
    # interlocked hands needs a different temporal trade-off from a clear
    # performance clip. Smooth is the reliable default; detailed preserves
    # more fast, subtle articulation when that detail is the point of the run.
    finger_motion: FingerMotion = cli_field(
        flag="--finger-motion", default=FingerMotion.SMOOTH, parse=FingerMotion,
        choices=[motion.value for motion in FingerMotion],
        help="Finger-motion profile: 'smooth' (stable default) or 'detailed' (preserves fast subtle articulation)",
    )
    # Camera intrinsics: either lens/sensor specs (the common case, a real
    # phone/camera shoot, no calibration available) or a raw K matrix (real
    # calibration data, e.g. a research dataset's own published intrinsics,
    # also strictly more accurate than the lens-spec path even when both are
    # available, since compute_intrinsics_matrix assumes a perfectly centered
    # principal point, which a raw K doesn't have to). Exactly one path must
    # be given, see validate_camera_input, called by both create_run.py and
    # pipeline.run.py's own main()s (the two places a fresh RunInput gets
    # built from CLI args).
    focal_length_mm: float = cli_field(flag="--focal-length-mm", default=0.0, required=False, value_type=float)
    sensor_width_mm: float = cli_field(flag="--sensor-width-mm", default=0.0, required=False, value_type=float)
    intrinsics_k: list[list[float]] | None = cli_field(
        flag="--intrinsics-k", default=None, value_type=json.loads,
        help='Raw 3x3 intrinsics matrix as JSON, e.g. \'[[fx,0,cx],[0,fy,cy],[0,0,1]]\', '
             "an alternative to --focal-length-mm/--sensor-width-mm for real calibration data.",
    )
    # Only meaningful when video_path (above) is a directory of images rather
    # than a real video file: a frame sequence has no embedded frame rate the
    # way a video container does, so this is the only way stage 0 can know
    # it. Ignored for a real video file (its own container fps is used
    # instead). Enforced required-when-a-directory-is-given by
    # validate_video_input, not argparse itself, same shape as the camera-
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
    skip_face_capture: bool = cli_field(
        flag="--skip-face-capture", default=False, bool_flag=True, is_render_preview=False,
        help="Disable stage 9 (capture_face) entirely, with no output_face.csv, no facial animation",
    )
    render_face_preview: bool = cli_field(
        flag="--render-face-preview", default=False, bool_flag=True,
        help="Stage 9/10 also write FLAME_face_preview.blend (raw tracked FLAME mesh, no SMPL-X expression mapping), "
             "landmark_preview.blend (raw vs. smoothed MediaPipe landmarks, upstream of DECA/MICA/FLAME entirely), and "
             "ARKit_face_preview.blend (the ARKit-52 channels that feed output_face.csv) for visual spot-checking",
    )

@dataclass
class FineTuningOptions:
    """Algorithm controls resolved from code defaults plus numeric overrides.

    These options are intentionally separate from ``RunInput``: source and
    preview choices stay visible in every run record, while these controls only
    appear in progress.json when a user deliberately pins one.
    """

    # Smooth landmarks before face fitting, which preserves large real blinks
    # better than penalizing the fitted parameters' own frame deltas.
    face_smoothing_window: int = 7

    # Bridge brief SAM re-detection gaps by holding the last trusted mask; do
    # not bridge longer occlusions, where stale geometry is less trustworthy.
    sam_track_max_bridge_frames: int = 15
    # Robust median number of unoccluded object candidates used to correct an
    # anchor-frame shape fit that may be blurred or gripped.
    object_shape_candidate_frames: int = 15
    # GVHMR is already temporal; the body therefore needs only light rotation
    # and root-position cleanup.
    body_smoothing_window: int = 9
    body_translation_cutoff: float = 0.15
    # HaMeR is per-frame. The default finger profile is deliberately stable;
    # `RunInput.finger_motion=DETAILED` selects its own responsive base values.
    # Explicit numeric fine-tuning overrides always take precedence over either
    # profile, so they remain useful for calibration without changing a run's
    # semantic smooth/detailed choice.
    hand_smoothing_window: int = 15
    hand_beta: float = 0.3
    hand_finger_smoothing_window: int = FINGER_MOTION_SETTINGS[FingerMotion.SMOOTH].hand_finger_smoothing_window
    hand_finger_beta: float = FINGER_MOTION_SETTINGS[FingerMotion.SMOOTH].hand_finger_beta
    hand_finger_derivative_cutoff_hz: float = FINGER_MOTION_SETTINGS[FingerMotion.SMOOTH].hand_finger_derivative_cutoff_hz
    hand_finger_min_cutoff_hz: float = FINGER_MOTION_SETTINGS[FingerMotion.SMOOTH].hand_finger_min_cutoff_hz
    hand_wrist_min_cutoff_hz: float = 0.10
    hand_finger_decimate_deg: float = FINGER_MOTION_SETTINGS[FingerMotion.SMOOTH].hand_finger_decimate_deg
    hand_wrist_decimate_deg: float = 3.0

    # Wrist plausibility uses a strict angular seed plus a lower hysteretic
    # release threshold, so isolated ambiguity is rejected without spreading a
    # real sustained pose across the whole clip.
    hand_wrist_max_deviation_deg: float = 110.0
    hand_wrist_release_deviation_deg: float = 55.0
    hand_wrist_deviation_window: int = 5
    # Limits how far a confirmed-invalid wrist run can grow in either direction.
    hand_wrist_max_expansion_frames: int = 10
    # Rejects isolated HaMeR orientation flips that remain angle-plausible.
    hand_wrist_max_velocity_deg_per_sec: float = 2400.0

    # Long invalid wrist stretches hold after this brief recovery blend instead
    # of sweeping through an uncertain large rotation.
    hand_wrist_max_bridge_frames: int = 15

    # Measures hand swing from the forearm, deliberately not unreliable twist.
    hand_wrist_max_swing_deg: float = 95.0
    # A geometry check for a hand folded into the forearm's own space.
    hand_forearm_interior_max_t: float = 0.95
    hand_forearm_radius_m: float = 0.10


def add_dataclass_cli_arguments(parser: argparse.ArgumentParser, dataclass_type: type, *, required: bool = True) -> None:
    """Registers every `cli_field(...)`-tagged field of `dataclass_type` onto
    `parser`, read directly off each field's own metadata, the shared
    engine `add_run_input_arguments` (for `RunInput`) builds on, and that
    `create_run`/`pipeline.run`/`update_run` also use directly for
    `RunLocation`/`NewRunID` (the `-o/--output-dir`/`--run-id` flags),
    so no CLI entrypoint hand-writes its own `parser.add_argument` calls for
    a field any of these dataclasses already declares.

    `required=False` (used only by `update_run`, for `RunInput`'s own fields)
    makes every otherwise-required flag optional and switches every flag's
    "not passed" value to `None` instead of a real default, `update_run`
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
    `parse` transform (if any) applied, shared by `build_dataclass_from_args`
    and `apply_run_input_overrides` so "convert the raw string" can't drift
    between fresh-construction and override-merge.
    """
    cli = f.metadata["cli"]
    value = getattr(args, f.name)
    if value is not None and cli["parse"] is not None:
        value = cli["parse"](value)
    return value


def build_dataclass_from_args(dataclass_type: type, args: argparse.Namespace) -> Any:
    """Constructs a fresh instance of `dataclass_type` from parsed CLI args,
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
    resolvable, neither path given, or both given at once (ambiguous which
    should win), else `None`. `focal_length_mm`/`sensor_width_mm` are
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
    `source_fps` unnecessary, so this only fires for the directory case,
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
    comes back as `None`), every other field, whether its flag simply
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
    """Where a run lives on disk, shared by `create_run`, `pipeline.run`,
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
    (`create_run`, `pipeline.run`), an existing run's id is already fixed,
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
    # Measured camera-space up direction (`camera_gravity.estimate_camera_up`),
    # empty when the scene has no measurable vertical structure or
    # for a run whose progress.json predates the measurement. Consumers treat
    # empty as "assume a level camera"
    camera_up: list[float] = field(default_factory=list)


@dataclass
class StageRecord:
    status: StageStatus = StageStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)  # output name -> path
    error: str | None = None


@dataclass
class RunOutputs:
    final_blend: str | None = None
    final_face_csv: str | None = None


@dataclass
class RunRecord:
    run_id: str
    progress_dir: str
    input: RunInput
    scene: SceneInfo = field(default_factory=SceneInfo)
    stages: dict[str, StageRecord] = field(default_factory=dict)
    outputs: RunOutputs = field(default_factory=RunOutputs)
    # Resolved temporal-filter settings. They are intentionally omitted from
    # progress.json; only explicit pins below are persisted there.
    fine_tuning: FineTuningOptions = field(default_factory=FineTuningOptions, repr=False)
    # Explicitly pinned fine-tuning values. They override the code defaults.
    fine_tuning_overrides: dict[str, int | float] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    # Unix timestamps (seconds). created_at is stamped once, by create_run();
    # updated_at is refreshed on every save() below, regardless of call site,
    # so it always reflects when this run last actually progressed. A run
    # directory from before this field existed loads with both at 0.0 (the
    # dataclass default) rather than failing, there's no real "created"
    # timestamp to recover for those, and 0.0 reads unambiguously as "unknown"
    # rather than a plausible-looking but fabricated date.
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def path(self) -> Path:
        return Path(self.progress_dir) / PROGRESS_JSON_NAME

    def save(self) -> None:
        self.updated_at = time.time()
        unknown_overrides = set(self.fine_tuning_overrides) - {f.name for f in fields(FineTuningOptions)}
        if unknown_overrides:
            raise ValueError(f"Unknown fine-tuning override(s): {sorted(unknown_overrides)}")
        for name, value in self.fine_tuning_overrides.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Fine-tuning override {name!r} must be an int or float")
            setattr(self.fine_tuning, name, value)
        data = asdict(self)
        # `fine_tuning` is the resolved runtime object (code defaults plus
        # explicit pins). Do not serialize it: only the pins belong in a run.
        data.pop(FIELD_FINE_TUNING)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        for attempt in range(PROGRESS_REPLACE_MAX_ATTEMPTS):
            try:
                # Atomic rename on the same filesystem.
                tmp_path.replace(self.path)
                break
            except PermissionError:
                if attempt == PROGRESS_REPLACE_MAX_ATTEMPTS - 1:
                    raise
                time.sleep(PROGRESS_REPLACE_INITIAL_RETRY_SECONDS * (2 ** attempt))

    @classmethod
    def load(cls, progress_dir: str | Path) -> RunRecord:
        # ``utf-8-sig`` consumes an optional UTF-8 BOM while decoding ordinary
        # UTF-8 unchanged. Some Windows tools save JSON with a BOM, whereas
        # pipeline-created and older records have none.
        path = Path(progress_dir) / PROGRESS_JSON_NAME
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            input_data = dict(data[FIELD_INPUT])
            # Older records stored smoothing fields under `input`; do not let those
            # stale defaults pin a clip to an old profile.
            for f in fields(FineTuningOptions):
                input_data.pop(f.name, None)
            raw_fine_tuning_overrides = data.get(FIELD_FINE_TUNING_OVERRIDES)
            if raw_fine_tuning_overrides is None:
                fine_tuning_overrides = {}
            elif isinstance(raw_fine_tuning_overrides, dict):
                fine_tuning_overrides = dict(raw_fine_tuning_overrides)
            else:
                raise ValueError("fine_tuning_overrides must be an object")
            unknown_overrides = set(fine_tuning_overrides) - {f.name for f in fields(FineTuningOptions)}
            if unknown_overrides:
                raise ValueError(f"Unknown fine-tuning override(s): {sorted(unknown_overrides)}")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in fine_tuning_overrides.values()):
                raise ValueError("fine_tuning_overrides values must be ints or floats")
            data[FIELD_FINE_TUNING_OVERRIDES] = fine_tuning_overrides
            data[FIELD_INPUT] = RunInput(
                **{
                    **input_data,
                    **(
                        {
                            name: enum_type(input_data[name])
                            for name, enum_type in (
                                (FIELD_OBJECT_SHAPE_HINT, ObjectShapeHint),
                                (FIELD_FINGER_MOTION, FingerMotion),
                            )
                            if name in input_data
                        }
                    ),
                }
            )
            data[FIELD_FINE_TUNING] = FineTuningOptions(**fine_tuning_overrides)
            data[FIELD_SCENE] = SceneInfo(**data[FIELD_SCENE])
            data[FIELD_STAGES] = {
                name: StageRecord(**{**rec, FIELD_STATUS: StageStatus(rec[FIELD_STATUS])})
                for name, rec in data[FIELD_STAGES].items()
            }
            data[FIELD_OUTPUTS] = RunOutputs(**data[FIELD_OUTPUTS])
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid progress record at {path}: {exc}") from exc

    def is_complete(self, stage_name: StageName) -> bool:
        record = self.stages.get(stage_name)
        return record is not None and record.status == StageStatus.COMPLETE

    def dependencies_met(self, stage_name: StageName) -> bool:
        return all(self.is_complete(dep) for dep in self.stages[stage_name].depends_on)

    def incomplete_dependencies(self, stage_name: StageName) -> list[tuple[StageName, StageStatus]]:
        """Returns this stage's direct prerequisites that still need attention.

        The order is the declared DAG order, which is also the order a user
        should normally run the prerequisite stages in.
        """
        return [
            (StageName(dependency), self.stages[dependency].status)
            for dependency in self.stages[stage_name].depends_on
            if not self.is_complete(dependency)
        ]

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
        DAG in place, adds a record (defaulted to PENDING) for any stage the
        current code knows about that this run predates, and refreshes
        `depends_on` for every stage (the DAG can change after a stage's own
        record was first written). Never touches an existing record's own
        status/outputs/error, and never removes a stage this run has a record
        for even if the current DAG no longer lists it, so no recorded
        progress is ever discarded. A stale, no-longer-listed stage's record
        just sits unread, nothing keys off `self.stages` except by a
        `StageName` the current DAG still knows about.

        Deliberately NOT called from `load()` itself: this saves (see below),
        and `load()` must stay a pure read with no disk side effect. Callers that
        load a record to actually *act* on it (`update_run`, `pipeline.run`,
        `pipeline_stage_base.cli_entrypoint`) call this explicitly instead, once,
        right after their own `load()`, safe there because each is a one-shot
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
