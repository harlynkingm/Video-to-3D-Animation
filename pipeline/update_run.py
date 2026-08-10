"""update_run: migrates an existing run's progress.json to the current code's
schema without discarding any recorded progress -- a completed stage's own
status/outputs/error survive untouched; this only adds what's missing (a
newer stage the run predates) and refreshes what can safely change without
losing information (a stage's `depends_on` list, in case the DAG itself
moved since that stage's own record was written).

Useful after a code change adds a new `RunInput` field or a
new pipeline stage, and an in-progress run's progress.json predates it --
loading that file straight into the current `RunInput`/`RunRecord`
dataclasses already tolerates new *fields* fine (they all have defaults), but
a *new stage* has no record at all yet, and a stage's `depends_on` can go
stale if the DAG changed after that stage last ran.

Takes the same --input-video/--human-prompt/etc. flags as `create_run`, but
every one is optional here: passing a flag overwrites that field on the
existing run; omitting it keeps whatever was already stored. See
`progress_tracker.add_run_input_arguments`'s own docstring for the one real
limitation this implies (a boolean --render-*-preview flag can only be
turned on this way, never explicitly back off).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .progress_tracker import (
    PROGRESS_JSON_NAME,
    RunRecord,
    RunLocation,
    add_dataclass_cli_arguments,
    add_run_input_arguments,
    apply_run_input_overrides,
)


def update_run(progress_dir: Path, args: argparse.Namespace) -> RunRecord:
    progress_path = progress_dir / PROGRESS_JSON_NAME
    if not progress_path.exists():
        raise SystemExit(
            f"No progress.json found at {progress_dir}, update_run only migrates an existing run "
            "(use create_run or pipeline.run to start a new one)"
        )

    runRecord = RunRecord.load(progress_dir)
    shutil.copy2(progress_path, progress_path.with_suffix(".json.bak"))

    runRecord.input = apply_run_input_overrides(runRecord.input, args)

    runRecord.update_schema()
    return runRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate an existing run's progress.json to the current schema without losing recorded progress"
    )
    add_dataclass_cli_arguments(parser, RunLocation)
    add_run_input_arguments(parser, required=False)
    args = parser.parse_args()

    runRecord = update_run(args.progress_dir, args)
    print(f"Updated run {runRecord.run_id!r} at {args.progress_dir} (backup saved as progress.json.bak)")


if __name__ == "__main__":
    main()
