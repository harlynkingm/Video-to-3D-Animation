"""update_run: migrates an existing run's progress.json to the current code's
schema without discarding any recorded progress -- a completed stage's own
status/outputs/error survive untouched; this only adds what's missing (a
newer stage the run predates) and refreshes what can safely change without
losing information (a stage's `depends_on` list, in case the DAG itself
moved since that stage's own record was written).

Useful after a code change adds a new `RunInput` field or a
new pipeline stage, and an in-progress run's progress.json predates it --
loading that file straight into the current `RunInput`/`ProgressRecord`
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
from dataclasses import fields, replace
from pathlib import Path

from .create_run import STAGE_DEPENDS_ON
from .progress_tracker import (
    PROGRESS_JSON_NAME,
    SCHEMA_VERSION,
    ProgressRecord,
    RunInput,
    RunLocation,
    StageRecord,
    add_dataclass_cli_arguments,
    add_run_input_arguments,
    resolve_cli_value,
)


def _apply_overrides(existing: RunInput, args: argparse.Namespace) -> RunInput:
    """Builds the updated `RunInput` via `dataclasses.replace`, reading the
    same `cli_field(...)` metadata `add_run_input_arguments` registered the
    parser from (see `progress_tracker.py`) -- so every field this doesn't
    touch, whether because its flag wasn't passed or because it's a smoothing
    knob with no CLI flag at all, passes through from `existing` untouched,
    not reset to some default. Adding a new `RunInput` field never requires
    touching this function.
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


def update_run(progress_dir: Path, args: argparse.Namespace) -> ProgressRecord:
    progress_path = progress_dir / PROGRESS_JSON_NAME
    if not progress_path.exists():
        raise SystemExit(
            f"No progress.json found at {progress_dir}, update_run only migrates an existing run "
            "(use create_run or pipeline.run to start a new one)"
        )

    progress = ProgressRecord.load(progress_dir)
    shutil.copy2(progress_path, progress_path.with_suffix("-backup.json"))

    progress.input = _apply_overrides(progress.input, args)

    # Add any stage the current DAG knows about that this run predates; for a
    # stage that already has a record, only refresh its depends_on (the DAG
    # can change after a stage's own file is written -- see
    # create_run.STAGE_DEPENDS_ON's own comments for real examples of this).
    # Never touches an existing record's status/outputs/error.
    for stage, deps in STAGE_DEPENDS_ON.items():
        depends_on = [dep.value for dep in deps]
        if stage.value in progress.stages:
            progress.stages[stage.value].depends_on = depends_on
        else:
            progress.stages[stage.value] = StageRecord(depends_on=depends_on)

    progress.schema_version = SCHEMA_VERSION
    progress.save()
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate an existing run's progress.json to the current schema without losing recorded progress"
    )
    add_dataclass_cli_arguments(parser, RunLocation)
    add_run_input_arguments(parser, required=False)
    args = parser.parse_args()

    progress = update_run(args.progress_dir, args)
    print(f"Updated run {progress.run_id!r} at {args.progress_dir} (backup saved as progress-backup.json)")


if __name__ == "__main__":
    main()
