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

    STAGE_0_INGEST_VIDEO = "ingest_video", "initial stage: processing video", 0
    STAGE_1_MASK_AND_TRACK = "mask_and_track", "stage 1: generate masks", 1
    STAGE_1A_HUMAN_MASK = "generate_human_mask", "stage 1: generate human mask", 1
    STAGE_1B_OBJECT_MASK = "generate_object_mask", "stage 1: generate object mask", 1
    STAGE_2_ESTIMATE_HUMAN_MOTION = "estimate_human_motion", "stage 2: estimate human motion", 2
    STAGE_3_ESTIMATE_DEPTH = "estimate_depth", "stage 3: estimate scene depth", 3
    STAGE_4_ESTIMATE_HANDS = "estimate_hands", "stage 4: estimate hands motion", 4
    STAGE_5_RETARGET_HANDS = "retarget_hands","stage 5: attach hands to body", 5
    STAGE_6_ALIGN_SCENE_SCALE = "align_scene_scale", "stage 6: detect scene scale", 6
    STAGE_7_ANNOTATE_CONTACTS = "annotate_contacts", "stage 7: detect contact points", 7
    STAGE_8_OPTIMIZE_HOI = "optimize_hoi", "stage 8: optimize human-object interaction", 8
    STAGE_9_EXPORT = "export", "stage 9: export animation", 9


class StageStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


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
    `run_input_from_args` -- every current boolean field is a preview flag,
    so there's no separate marker for that.

    `parse`, if given, converts the raw parsed string into the field's real
    type (e.g. `ObjectShapeHint`). It's applied by `build_dataclass_from_args`/
    `update_run`, not registered as argparse's own `type=`, so a `None` value
    (meaning "flag not passed", in `update_run`'s optional mode) never runs
    through it.
    """
    metadata = {"cli": {
        "flag": flag, "short_flag": short_flag, "help": help, "required": required, "bool_flag": bool_flag,
        "value_type": value_type, "choices": choices, "parse": parse, "metavar": metavar,
    }}
    if default is MISSING:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


@dataclass
class RunInput:
    video_path: str = cli_field(
        flag="--input-video", metavar="INPUT_VIDEO", required=True,
        help="Path to the source video file (MP4, MOV, MPEG, FLV, or WMV)",
    )
    human_prompt: str = cli_field(flag="--human-prompt", required=True, help='e.g. "a person"')
    object_prompt: str | None = cli_field(
        flag="--object-prompt", default=None, help='e.g. "a teddy bear" (omit if there is no object)',
    )
    object_shape_hint: ObjectShapeHint = cli_field(
        flag="--object-shape-hint", default=ObjectShapeHint.AUTO, parse=ObjectShapeHint,
        choices=[hint.value for hint in ObjectShapeHint],
    )
    focal_length_mm: float = cli_field(flag="--focal-length-mm", default=0.0, required=True, value_type=float)
    sensor_width_mm: float = cli_field(flag="--sensor-width-mm", default=0.0, required=True, value_type=float)
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
            if cli and cli["bool_flag"]:
                setattr(run_input, f.name, True)
    return run_input


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
            if render_all or value is not None:
                overrides[f.name] = bool(render_all or value)
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

    @property
    def path(self) -> Path:
        return Path(self.progress_dir) / PROGRESS_JSON_NAME

    def save(self) -> None:
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
