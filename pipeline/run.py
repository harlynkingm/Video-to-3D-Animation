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
against it (each stage still skips itself if already complete). Resuming
doesn't require re-supplying `--input-video`/`--human-prompt`/etc. -- those
only matter for a fresh run, since an existing one already has them stored.
A RunInput flag passed while resuming still applies, though (via the same
override mechanism `update_run` uses), so e.g. `--render-contacts-preview`
works equally well tacked onto a resumed run as a fresh one.
"""

from __future__ import annotations

import argparse
import importlib

from .create_run import STAGE_DEPENDS_ON, create_run
from .pipeline_stage_base import run_stage
from .progress_tracker import (
    PROGRESS_JSON_NAME,
    NewRunID,
    RunRecord,
    StageName,
    add_dataclass_cli_arguments,
    add_run_input_arguments,
    apply_run_input_overrides,
    run_input_from_args,
)

ORDERED_STAGES: list[StageName] = list(STAGE_DEPENDS_ON.keys())


def stage_module_name(index: int, stage_name: StageName) -> str:
    """The `stage_{index}_{StageName.value}` naming convention every stage
    module follows, as an importable dotted path -- pulled out of
    `_load_stage_run` so tests (which need to invoke the same modules as
    real subprocesses) can reuse this exact convention instead of
    re-deriving it by hand.
    """
    return f"pipeline.stages.stage_{index}_{stage_name.value}"


def _load_stage_run(index: int, stage_name: StageName):
    module = importlib.import_module(stage_module_name(index, stage_name))
    return module.run


def run_pipeline(runRecord: RunRecord, stop_after_stage: int | None = None) -> None:
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
        run_stage(runRecord, _load_stage_run(index, stage_name), stage_name)


def main() -> None:
    # Whether RunInput's own flags (--input-video, --human-prompt, etc.) are
    # required depends on whether a run already exists at --output-dir: a
    # fresh run needs them, resuming an existing one doesn't (its RunInput is
    # already stored) -- but that isn't knowable until --output-dir itself
    # has been parsed, so this resolves just that much first with a
    # throwaway parser (add_help=False so a real --help still goes to the
    # full parser below, not this partial one) before building the real one.
    location_parser = argparse.ArgumentParser(add_help=False)
    add_dataclass_cli_arguments(location_parser, NewRunID)
    location_args, _ = location_parser.parse_known_args()
    resuming = (location_args.progress_dir / PROGRESS_JSON_NAME).exists()

    parser = argparse.ArgumentParser(
        description="Create a run and execute every implemented stage in sequence, start to finish"
    )
    add_dataclass_cli_arguments(parser, NewRunID)
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        default=None,
        help="Run only through this stage number, inclusive (e.g. 5 runs stages 0-5, skipping 6+). "
             "Omit to run every implemented stage.",
    )
    add_run_input_arguments(parser, required=not resuming)
    args = parser.parse_args()

    if args.stop_after_stage is not None and args.stop_after_stage < 0:
        parser.error("--stop-after-stage must be >= 0")

    progress_dir = args.progress_dir
    if resuming:
        print(f"Found an existing run at {progress_dir}, resuming")
        runRecord = RunRecord.load(progress_dir)
        runRecord.input = apply_run_input_overrides(runRecord.input, args)
        runRecord.save()
    else:
        run_input = run_input_from_args(args)
        runRecord = create_run(progress_dir, run_input, run_id=args.run_id)
        print(f"Created run {runRecord.run_id!r} at {progress_dir}")

    run_pipeline(runRecord, stop_after_stage=args.stop_after_stage)


if __name__ == "__main__":
    main()
