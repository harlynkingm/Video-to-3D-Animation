"""create_run: bootstraps a new pipeline run, a fresh progress directory with
a `progress.json` seeded from the user's input (video path, prompts, camera
info). Every stage script expects this file to already exist (see
`pipeline_stage_base.cli_entrypoint`), so this runs first, once per clip.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .progress_tracker import (
    STAGE_DEPENDS_ON,
    NewRunID,
    RunRecord,
    RunInput,
    StageRecord,
    add_dataclass_cli_arguments,
    add_run_input_arguments,
    run_input_from_args,
    validate_camera_input,
    validate_video_input,
)


def create_run(progress_dir: Path, run_input: RunInput, run_id: str | None = None) -> RunRecord:
    """`run_id` is just a human-readable label stored alongside the run's data,
    `progress_dir` is what actually identifies a run on disk, so it defaults to
    the directory's own name rather than requiring the caller to repeat it.
    """
    progress_dir.mkdir(parents=True, exist_ok=True)
    runRecord = RunRecord(
        run_id=run_id or progress_dir.name,
        progress_dir=str(progress_dir),
        input=run_input,
        stages={
            stage.value: StageRecord(depends_on=[dep.value for dep in deps])
            for stage, deps in STAGE_DEPENDS_ON.items()
        },
        created_at=time.time(),
    )
    runRecord.save()
    return runRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new pipeline run")
    add_dataclass_cli_arguments(parser, NewRunID)
    add_run_input_arguments(parser)
    args = parser.parse_args()

    run_input = run_input_from_args(args)
    camera_error = validate_camera_input(run_input)
    if camera_error:
        parser.error(camera_error)
    video_error = validate_video_input(run_input)
    if video_error:
        parser.error(video_error)
    runRecord = create_run(args.progress_dir, run_input, run_id=args.run_id)
    print(f"Created run {runRecord.run_id!r} at {args.progress_dir}")


if __name__ == "__main__":
    main()
