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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new pipeline run")
    parser.add_argument("--progress-dir", required=True, help="Directory to create for this run's state and outputs")
    parser.add_argument("--run-id", default=None, help="Defaults to --progress-dir's own folder name")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--human-prompt", required=True, help='e.g. "a tennis player"')
    parser.add_argument("--object-prompt", default=None, help='e.g. "a tennis racket" (omit if there is no object)')
    parser.add_argument("--object-shape-hint", default=ObjectShapeHint.AUTO.value,
                         choices=[hint.value for hint in ObjectShapeHint])
    parser.add_argument("--focal-length-mm", required=True, type=float)
    parser.add_argument("--sensor-width-mm", required=True, type=float)
    parser.add_argument("--anchor-frame-override", default=None, type=int)
    parser.add_argument("--dump-mask-previews", action="store_true",
                         help="Stage 1 also writes black/white JPEG mask previews for visual spot-checking")
    parser.add_argument("--dump-motion-preview", action="store_true",
                         help="Stage 2 also writes an AMASS .npz importable into Blender for visual spot-checking")
    parser.add_argument("--dump-depth-preview", action="store_true",
                         help="Stage 3 also writes a colored .ply point cloud importable into Blender for visual spot-checking")
    args = parser.parse_args()

    run_input = RunInput(
        video_path=args.video_path,
        human_prompt=args.human_prompt,
        object_prompt=args.object_prompt,
        object_shape_hint=ObjectShapeHint(args.object_shape_hint),
        focal_length_mm=args.focal_length_mm,
        sensor_width_mm=args.sensor_width_mm,
        anchor_frame_override=args.anchor_frame_override,
        dump_mask_previews=args.dump_mask_previews,
        dump_motion_preview=args.dump_motion_preview,
        dump_depth_preview=args.dump_depth_preview,
    )
    progress = create_run(Path(args.progress_dir), run_input, run_id=args.run_id)
    print(f"Created run {progress.run_id!r} at {args.progress_dir}")


if __name__ == "__main__":
    main()
