"""create_run: bootstraps a new pipeline run -- a fresh progress directory with
a `progress.json` seeded from the user's input (video path, prompts, camera
info). Every stage script expects this file to already exist (see
`pipeline_stage_base.cli_entrypoint`), so this runs first, once per clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .progress_tracker import ObjectShapeHint, ProgressRecord, RunInput, StageName, StageRecord

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
}


def create_run(progress_dir: Path, run_input: RunInput, run_id: str | None = None) -> ProgressRecord:
    """`run_id` is just a human-readable label stored alongside the run's data --
    `progress_dir` is what actually identifies a run on disk -- so it defaults to
    the directory's own name rather than requiring the caller to repeat it.
    """
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressRecord(
        run_id=run_id or progress_dir.name,
        progress_dir=str(progress_dir),
        input=run_input,
        stages={
            stage.value: StageRecord(depends_on=[dep.value for dep in deps])
            for stage, deps in STAGE_DEPENDS_ON.items()
        },
    )
    progress.save()
    return progress


def add_run_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Registers every `RunInput` CLI flag onto `parser` -- shared by this
    module's own `main()` and `pipeline.run` (the one-shot command) so the two
    can never drift apart.
    """
    parser.add_argument("--input-video", dest="video_path", metavar="INPUT_VIDEO", required=True,
                         help="Path to the source video file (MP4, MOV, MPEG, FLV, or WMV)")
    parser.add_argument("--human-prompt", required=True, help='e.g. "a tennis player"')
    parser.add_argument("--object-prompt", default=None, help='e.g. "a tennis racket" (omit if there is no object)')
    parser.add_argument("--object-shape-hint", default=ObjectShapeHint.AUTO.value,
                         choices=[hint.value for hint in ObjectShapeHint])
    parser.add_argument("--focal-length-mm", required=True, type=float)
    parser.add_argument("--sensor-width-mm", required=True, type=float)
    parser.add_argument("--anchor-frame-override", default=None, type=int)
    parser.add_argument("--render-mask-previews", action="store_true",
                         help="Stage 1 also writes black/white JPEG mask previews for visual spot-checking")
    parser.add_argument("--render-motion-preview", action="store_true",
                         help="Stage 2 also writes an AMASS .npz importable into Blender for visual spot-checking")
    parser.add_argument("--render-depth-preview", action="store_true",
                         help="Stage 3 also writes a colored .ply point cloud importable into Blender for visual spot-checking")
    parser.add_argument("--render-hands-preview", action="store_true",
                         help="Stage 4 also writes a .bvh hand skeleton animation importable into Blender for visual spot-checking")
    parser.add_argument("--render-retarget-preview", action="store_true",
                         help="Stage 5 also writes a .bvh full-body-plus-hands skeleton importable into Blender for visual spot-checking")
    parser.add_argument("--render-scene-preview", action="store_true",
                         help="Stage 6 also writes a .ply combining human, object, and scene in one aligned space for visual spot-checking")
    parser.add_argument("--render-previews", action="store_true",
                         help="Shorthand for every --render-*-preview flag above at once")


def run_input_from_args(args: argparse.Namespace) -> RunInput:
    render_all = args.render_previews
    return RunInput(
        video_path=args.video_path,
        human_prompt=args.human_prompt,
        object_prompt=args.object_prompt,
        object_shape_hint=ObjectShapeHint(args.object_shape_hint),
        focal_length_mm=args.focal_length_mm,
        sensor_width_mm=args.sensor_width_mm,
        anchor_frame_override=args.anchor_frame_override,
        render_mask_previews=render_all or args.render_mask_previews,
        render_motion_preview=render_all or args.render_motion_preview,
        render_depth_preview=render_all or args.render_depth_preview,
        render_scene_preview=render_all or args.render_scene_preview,
        render_hands_preview=render_all or args.render_hands_preview,
        render_retarget_preview=render_all or args.render_retarget_preview,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new pipeline run")
    parser.add_argument("-o", "--output-dir", dest="progress_dir", metavar="OUTPUT_DIR", required=True,
                         help="Directory to create for this run's state and outputs")
    parser.add_argument("--run-id", default=None, help="Defaults to --output-dir's own folder name")
    add_run_input_arguments(parser)
    args = parser.parse_args()

    run_input = run_input_from_args(args)
    progress = create_run(Path(args.progress_dir), run_input, run_id=args.run_id)
    print(f"Created run {progress.run_id!r} at {args.progress_dir}")


if __name__ == "__main__":
    main()
