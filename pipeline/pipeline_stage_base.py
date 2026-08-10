"""Shared harness so each pipeline stage file only has to contain its own logic.

A stage file should look like:

    from .pipeline_stage_base import cli_entrypoint
    from .progress_tracker import StageName

    def run(runRecord: RunRecord) -> dict[str, str]:
        ...  # the actual stage logic
        return {"some_output": "path/to/it"}

    if __name__ == "__main__":
        cli_entrypoint(run, stage_name=StageName.STAGE_0_INGEST_VIDEO)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable

from .progress_tracker import RunRecord, StageName, StageStatus


class StageDependenciesNotMetError(RuntimeError):
    pass


def run_stage(
    runRecord: RunRecord,
    run: Callable[[RunRecord], dict[str, str]],
    stage_name: StageName,
    force: bool = False,
) -> bool:
    """The dependency-check/skip/mark-progress logic shared by `cli_entrypoint`
    (one stage per process, via each stage's own `--output-dir`/`--force` CLI)
    and `pipeline.run` (the one-shot command, calling every stage's `run` in one
    process). Returns whether the stage actually ran (False means it was already
    complete and got skipped); raises `StageDependenciesNotMetError` if this
    stage's dependencies aren't complete yet.
    """
    if not runRecord.dependencies_met(stage_name):
        raise StageDependenciesNotMetError(f"{stage_name}: dependencies not met")

    if runRecord.is_complete(stage_name) and not force:
        print(f"[{stage_name.label}] already complete, skipping (use --force to re-run)")
        return False

    runRecord.mark_progress(stage_name, StageStatus.RUNNING)
    try:
        outputs = run(runRecord)
    except Exception:
        runRecord.mark_progress(stage_name, StageStatus.FAILED, error=traceback.format_exc())
        raise

    runRecord.mark_progress(stage_name, StageStatus.COMPLETE, outputs=outputs)
    return True


def cli_entrypoint(run: Callable[[RunRecord], dict[str, str]], stage_name: StageName) -> None:
    parser = argparse.ArgumentParser(description=f"Run the {stage_name} stage")
    parser.add_argument("-o", "--output-dir", dest="progress_dir", metavar="OUTPUT_DIR", required=True,
                         help="Path to the run directory containing progress.json")
    parser.add_argument("-f", "--force", action="store_true", help="Re-run even if this stage is already marked complete")
    args = parser.parse_args()

    # A stage run directly (not via `pipeline.run`/`update_run`, both of which
    # already call this themselves) is otherwise the one load path with no
    # chance to migrate an old progress.json before `run_stage` below looks up
    # `self.stages[stage_name]`. A genuinely new stage that predates this run
    # would KeyError there instead of just getting a fresh PENDING record.
    runRecord = RunRecord.load(args.progress_dir)
    runRecord.update_schema()

    try:
        run_stage(runRecord, run, stage_name, force=args.force)
    except StageDependenciesNotMetError as e:
        print(f"{e}, aborting", file=sys.stderr)
        sys.exit(1)
