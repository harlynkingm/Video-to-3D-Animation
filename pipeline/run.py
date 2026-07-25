"""run: one-shot command that creates a run (if one doesn't already exist at
--output-dir) and then executes every implemented stage in order, start to
finish, in this one process.

Stages run in-process (each stage module's own `run(progress)`, not a
subprocess per stage) via dynamic import keyed off the existing
`stage_{index}_{StageName.value}` file-naming convention -- so as later
stages (7, 8, 9...) get real files following that same convention, this loop
picks them up automatically; the only thing that needs updating is
`create_run.STAGE_DEPENDS_ON`, which is already the existing convention for
registering a new stage.

Resumable exactly like the per-stage manual workflow: if `progress.json`
already exists at --output-dir, this skips `create_run` and just resumes
against it (each stage still skips itself if already complete).
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .create_run import STAGE_DEPENDS_ON, add_run_input_arguments, create_run, run_input_from_args
from .pipeline_stage_base import run_stage
from .progress_tracker import PROGRESS_JSON_NAME, ProgressRecord, StageName

ORDERED_STAGES: list[StageName] = list(STAGE_DEPENDS_ON.keys())


def _load_stage_run(index: int, stage_name: StageName):
    module = importlib.import_module(f"pipeline.stages.stage_{index}_{stage_name.value}")
    return module.run


def run_pipeline(progress: ProgressRecord, stop_after_stage: int | None = None) -> None:
    max_stage_index = len(ORDERED_STAGES) - 1
    if stop_after_stage is None:
        last_index = max_stage_index
    else:
        if stop_after_stage > max_stage_index:
            print(f"Only stages 0-{max_stage_index} are implemented; running through stage {max_stage_index}")
        last_index = min(stop_after_stage, max_stage_index)

    for index, stage_name in enumerate(ORDERED_STAGES):
        if index > last_index:
            break
        run_stage(progress, _load_stage_run(index, stage_name), stage_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a run and execute every implemented stage in sequence, start to finish"
    )
    parser.add_argument("-o", "--output-dir", dest="progress_dir", metavar="OUTPUT_DIR", required=True,
                         help="Directory for this run's state and outputs")
    parser.add_argument("--run-id", default=None, help="Defaults to --output-dir's own folder name")
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        default=None,
        help="Run only through this stage number, inclusive (e.g. 5 runs stages 0-5, skipping 6+). "
             "Omit to run every implemented stage.",
    )
    add_run_input_arguments(parser)
    args = parser.parse_args()

    if args.stop_after_stage is not None and args.stop_after_stage < 0:
        parser.error("--stop-after-stage must be >= 0")

    progress_dir = Path(args.progress_dir)
    if (progress_dir / PROGRESS_JSON_NAME).exists():
        print(f"Found an existing run at {progress_dir}, resuming")
        progress = ProgressRecord.load(progress_dir)
    else:
        run_input = run_input_from_args(args)
        progress = create_run(progress_dir, run_input, run_id=args.run_id)
        print(f"Created run {progress.run_id!r} at {progress_dir}")

    run_pipeline(progress, stop_after_stage=args.stop_after_stage)


if __name__ == "__main__":
    main()
