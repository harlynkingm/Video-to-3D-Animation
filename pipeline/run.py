"""run: one-shot command that creates a run (if one doesn't already exist at
--output-dir) and then executes every implemented stage in order, start to
finish.

Every stage except `export_fbx` runs in-process (each stage module's own
`run(progress)`, not a subprocess per stage) via dynamic import keyed off the
existing `stage_{number}_{StageName.value}` file-naming convention -- so as
later stages get real files following that same convention, this loop picks
them up automatically; the only thing that needs updating is
`create_run.STAGE_DEPENDS_ON`, which is already the existing convention for
registering a new stage. `export_fbx` is the one exception: it needs `bpy`,
which lives only in a separate pixi environment that can't be imported into
this process, so it's dispatched as its own subprocess instead (see
`_run_in_fbx_export_env`).

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
import subprocess
from pathlib import Path

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

# repo root is 1 level up from this file (pipeline/ -> root) -- needed so the
# fbx-export subprocess below finds pixi.toml regardless of the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def stage_module_name(stage_name: StageName) -> str:
    """The `stage_{number}_{StageName.value}` naming convention every stage
    module follows, as an importable dotted path -- pulled out of
    `_load_stage_run` so tests (which need to invoke the same modules as
    real subprocesses) can reuse this exact convention instead of
    re-deriving it by hand.
    """
    return f"pipeline.stages.stage_{stage_name.stage_number}_{stage_name.value}"


def _load_stage_run(stage_name: StageName):
    module = importlib.import_module(stage_module_name(stage_name))
    return module.run


def _run_in_fbx_export_env(runRecord: RunRecord) -> None:
    """`export_fbx` needs `bpy`, which lives only in the separate
    `fbx-export` pixi environment (kept separate from `main` since bpy pins
    hard to its own Python release) -- it can't be dynamically imported into
    this process the way every other stage is. Shells out via `pixi run`
    itself, rather than hand-deriving the environment's own python.exe path,
    so the subprocess resolves the right interpreter the same way a person
    would run it manually. The subprocess's own `cli_entrypoint`/`run_stage`
    already does the right dependency/skip/mark-progress bookkeeping against
    `progress.json`, so nothing is duplicated here.
    """
    module_name = stage_module_name(StageName.STAGE_9_EXPORT_FBX)
    subprocess.run(
        ["pixi", "run", "-e", "fbx-export", "python", "-m", module_name,
         "--output-dir", str(runRecord.progress_dir)],
        check=True, cwd=_REPO_ROOT,
    )


def run_pipeline(runRecord: RunRecord, stop_after_stage: int | None = None) -> None:
    max_stage_number = max(s.stage_number for s in ORDERED_STAGES)
    if stop_after_stage is None:
        last_stage_number = max_stage_number
    else:
        if stop_after_stage > max_stage_number:
            print(f"Only stages 0-{max_stage_number} are implemented; running through stage {max_stage_number}")
        last_stage_number = min(stop_after_stage, max_stage_number)

    # Assumes ORDERED_STAGES is declared in ascending stage-number order
    # (true today) so a number exceeding the bound can short-circuit the
    # rest via `break` instead of scanning every remaining stage.
    for stage_name in ORDERED_STAGES:
        if stage_name.stage_number > last_stage_number:
            break
        if stage_name == StageName.STAGE_9_EXPORT_FBX:
            _run_in_fbx_export_env(runRecord)
            runRecord = RunRecord.load(runRecord.progress_dir)
            continue
        run_stage(runRecord, _load_stage_run(stage_name), stage_name)


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
