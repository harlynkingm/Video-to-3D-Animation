"""run: one-shot command that creates a run (if one doesn't already exist at
--output-dir) and then executes every implemented stage in order, start to
finish.

Every stage runs as its own subprocess (`python -m
pipeline.stages.stage_{number}_{name} --output-dir ...`), dynamically named
off the existing `stage_{number}_{StageName.value}` file-naming convention --
so as later stages get real files following that same convention, this loop
picks them up automatically; the only thing that needs updating is
`create_run.STAGE_DEPENDS_ON`, which is already the existing convention for
registering a new stage.

Stages are deliberately NOT called in-process one after another. Some of the
heavier models here (SAM 3.1, GVHMR, HAMER) are separately-compiled
native/CUDA stacks, and loading one after another inside a single process has
been observed to segfault deep inside torch, even though each stage's own
adapter correctly `del`s its model and calls `torch.cuda.empty_cache()` on
unload -- the corruption is at the native/CUDA level, not a Python-visible
leak, and gets more likely to actually trigger the more frames/GPU memory
churn a stage does. Running each stage as its own process sidesteps that
class of bug entirely, at the cost of a per-stage process-startup/checkpoint-
reload overhead. `export` already worked this way (it needs `bpy`, which
lives only in a separate pixi environment), so this just extends the same
mechanism to every stage.

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
import subprocess
import sys
from pathlib import Path

from .create_run import STAGE_DEPENDS_ON, create_run
from .progress_tracker import (
    PROGRESS_JSON_NAME,
    NewRunID,
    RunRecord,
    StageName,
    add_dataclass_cli_arguments,
    add_run_input_arguments,
    apply_run_input_overrides,
    run_input_from_args,
    validate_camera_input,
    validate_video_input,
)

ORDERED_STAGES: list[StageName] = list(STAGE_DEPENDS_ON.keys())

# repo root is 1 level up from this file (pipeline/ -> root) -- needed so
# subprocess stages find pixi.toml regardless of the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def stage_module_name(stage_name: StageName) -> str:
    """The `stage_{number}_{StageName.value}` naming convention every stage
    module follows, as an importable dotted path -- pulled out so tests
    (which need to invoke the same modules as real subprocesses) can reuse
    this exact convention instead of re-deriving it by hand.
    """
    return f"pipeline.stages.stage_{stage_name.stage_number}_{stage_name.value}"


def _run_subprocess_tolerating_crash_after_success(
    cmd: list[str], progress_dir: Path, stage_name: StageName,
) -> None:
    """Runs a stage's own CLI entrypoint and raises only if the stage didn't
    actually complete. A subprocess can crash during its own interpreter
    teardown *after* already saving its real output and marking the stage
    complete in `progress.json` -- `bpy` in particular is known to sometimes
    do this on exit, well after the export stage's own `run()` has already
    returned successfully. Judging success by the subprocess's exit code
    alone would treat that the same as a genuine mid-stage failure; checking
    `progress.json`'s own recorded status instead tells the two apart.
    """
    result = subprocess.run(cmd, cwd=_REPO_ROOT)
    if result.returncode != 0 and not RunRecord.load(progress_dir).is_complete(stage_name):
        result.check_returncode()


def _run_stage_subprocess(stage_name: StageName, progress_dir: Path, force: bool) -> None:
    """Runs one stage's own CLI entrypoint (`cli_entrypoint`/`run_stage`) as a
    fresh subprocess under this same `main`-env interpreter -- that
    entrypoint already does the right dependency/skip/mark-progress
    bookkeeping against `progress.json`, so nothing is duplicated here.
    """
    cmd = [sys.executable, "-m", stage_module_name(stage_name), "--output-dir", str(progress_dir)]
    if force:
        cmd.append("--force")
    _run_subprocess_tolerating_crash_after_success(cmd, progress_dir, stage_name)


def _run_export_subprocess(progress_dir: Path, force: bool) -> None:
    """`export` needs `bpy`, which lives only in the separate `export` pixi
    environment (kept separate from `main` since bpy pins hard to its own
    Python release), so unlike every other stage it can't run under this
    process's own `sys.executable`. Shells out via `pixi run` itself, rather
    than hand-deriving the environment's own python.exe path, so the
    subprocess resolves the right interpreter the same way a person would
    run it manually.
    """
    cmd = ["pixi", "run", "-e", "export", "python", "-m", stage_module_name(StageName.STAGE_9_EXPORT),
           "--output-dir", str(progress_dir)]
    if force:
        cmd.append("--force")
    _run_subprocess_tolerating_crash_after_success(cmd, progress_dir, StageName.STAGE_9_EXPORT)


def run_pipeline(runRecord: RunRecord, start_on_stage: int | None = None, stop_after_stage: int | None = None, force_all: bool = False) -> None:
    first_stage_number = 0
    if start_on_stage is not None:
        first_stage_number = start_on_stage
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
        if stage_name.stage_number < first_stage_number:
            continue
        if stage_name.stage_number > last_stage_number:
            break
        if stage_name == StageName.STAGE_9_EXPORT:
            _run_export_subprocess(Path(runRecord.progress_dir), force_all)
        else:
            _run_stage_subprocess(stage_name, Path(runRecord.progress_dir), force_all)


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
        "--start-on-stage",
        type=int,
        default=None,
        help="Start the run on this stage number",
    )
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        default=None,
        help="Run only through this stage number, inclusive (e.g. 5 runs stages 0-5, skipping 6+). "
             "Omit to run every implemented stage.",
    )
    parser.add_argument(
        "-f", "--force-all",
        action="store_true",
        help="Force all stages to re-run. The equivalent of passing `--force` to each stage individually.",
    )
    add_run_input_arguments(parser, required=not resuming)
    args = parser.parse_args()

    max_stage_number = max(s.stage_number for s in ORDERED_STAGES)
    if args.start_on_stage is not None:
        if args.start_on_stage < 0:
            parser.error("--start-on-stage must be >= 0")
        elif args.start_on_stage > max_stage_number:
            parser.error(f"--start-on-stage must be <= {max_stage_number}")
    if args.stop_after_stage is not None:
        if args.stop_after_stage < 0:
            parser.error("--stop-after-stage must be >= 0")
        elif args.stop_after_stage > max_stage_number:
            parser.error(f"--stop-after-stage must be <= {max_stage_number}")
        elif args.start_on_stage is not None and args.stop_after_stage < args.start_on_stage:
            parser.error("--stop-after-stage must be >= --start-on-stage")

    progress_dir = args.progress_dir
    if resuming:
        print(f"Found an existing run at {progress_dir}, resuming")
        runRecord = RunRecord.load(progress_dir)
        runRecord.input = apply_run_input_overrides(runRecord.input, args)
        runRecord.save()
    else:
        run_input = run_input_from_args(args)
        camera_error = validate_camera_input(run_input)
        if camera_error:
            parser.error(camera_error)
        video_error = validate_video_input(run_input)
        if video_error:
            parser.error(video_error)
        runRecord = create_run(progress_dir, run_input, run_id=args.run_id)
        print(f"Created run {runRecord.run_id!r} at {progress_dir}")

    run_pipeline(runRecord, start_on_stage=args.start_on_stage, stop_after_stage=args.stop_after_stage, force_all=args.force_all)



if __name__ == "__main__":
    main()
