"""Builds and launches the exact `pipeline.run` subprocess the CLI itself
uses (see pipeline/run.py), driven by the UI's own form state instead of
argv. `RunFormState`/`build_run_argv` are kept free of PySide6 so they're
testable without a Qt application; `PipelineRunner` is the Qt-facing wrapper
around `QProcess` that streams that subprocess's output back to the window.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from pipeline.progress_tracker import ObjectShapeHint, RunRecord, StageName, StageStatus
from pipeline.run import MAX_ORDERED_STAGE_NUMBER, ORDERED_STAGES

# ui/ -> repo root, matching pipeline/run.py's own _REPO_ROOT (pipeline/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]

UI_STAGES_COMPLETE = "All stages complete"
UI_PREPARING_NEXT_STAGE = "Preparing to run next stage..."

@dataclass
class RunFormState:
    destination_folder: str
    # video_path/human_prompt/focal_length_mm/sensor_width_mm are only required
    # for a fresh run, pipeline.run itself already resumes an existing run
    # (progress.json already present at destination_folder) from its own
    # stored RunInput, treating any of these as an override only when given.
    # Left as None, they're simply omitted from argv (see build_run_argv)
    # rather than sent as empty/zero values that would fail argparse or
    # overwrite the stored run's real input.
    video_path: str | None = None
    human_prompt: str | None = None
    focal_length_mm: float | None = None
    sensor_width_mm: float | None = None
    object_prompt: str | None = None
    is_image_sequence: bool = False
    source_fps: float | None = None
    start_stage: int = 0
    stop_stage: int = MAX_ORDERED_STAGE_NUMBER
    object_shape: str = "auto"
    force_all: bool = False
    render_previews: bool = False


def build_run_argv(state: RunFormState) -> list[str]:
    """`state` -> the same argv `pixi run -e main python -m pipeline.run ...`
    would receive by hand. Uses `sys.executable` rather than shelling out
    through `pixi run` again, the UI process itself already runs under the
    `main` env's interpreter, so it's the same interpreter either way.
    """
    argv = [
        sys.executable, "-m", "pipeline.run",
        "--output-dir", state.destination_folder,
        "--start-on-stage", str(state.start_stage),
        "--stop-after-stage", str(state.stop_stage),
    ]
    # `auto` is the UI's no-override choice.  Omitting the flag lets a new
    # run use the pipeline default and, more importantly, preserves an
    # existing run's recorded object_shape_hint when resuming.
    if state.object_shape != ObjectShapeHint.AUTO.value:
        argv += ["--object-shape-hint", state.object_shape]
    if state.video_path:
        argv += ["--input-video", state.video_path]
    if state.human_prompt:
        argv += ["--human-prompt", state.human_prompt]
    if state.object_prompt:
        argv += ["--object-prompt", state.object_prompt]
    if state.focal_length_mm is not None:
        argv += ["--focal-length-mm", str(state.focal_length_mm)]
    if state.sensor_width_mm is not None:
        argv += ["--sensor-width-mm", str(state.sensor_width_mm)]
    if state.is_image_sequence and state.source_fps is not None:
        argv += ["--source-fps", str(state.source_fps)]
    if state.force_all:
        argv.append("--force-all")
    if state.render_previews:
        argv.append("--render-previews")
    return argv


@dataclass
class StageProgress:
    completed: int
    total: int
    status_text: str
    overall_status_text: str | None


def _stages_in_range(start_stage: int, stop_stage: int) -> list[StageName]:
    return [s for s in ORDERED_STAGES if start_stage <= s.stage_number <= stop_stage]


def stage_count_in_range(start_stage: int, stop_stage: int) -> int:
    """How many stages `pipeline.run` will actually execute for
    [start_stage, stop_stage], used by the queue (ui/queue.py) to size its
    own aggregate progress bar for items that haven't started yet (no
    progress.json to read from) without duplicating ORDERED_STAGES'
    filtering logic.
    """
    return len(_stages_in_range(start_stage, stop_stage))


def compute_stage_progress(run_record: RunRecord, start_stage: int, stop_stage: int) -> StageProgress:
    """Summarizes a run's on-disk progress.json against the stages
    `pipeline.run` actually executes for [start_stage, stop_stage],
    `ORDERED_STAGES`, the same list pipeline.run itself iterates, into one
    bar-fill fraction and one human-readable status line, reusing each
    stage's own `StageName.label` rather than inventing new wording.
    """
    stages_in_range = _stages_in_range(start_stage, stop_stage)
    total = len(stages_in_range)
    completed = 0
    running_stage = None
    failed_stage = None
    for stage in stages_in_range:
        record = run_record.stages.get(stage.value)
        if record is None:
            continue
        if record.status == StageStatus.COMPLETE:
            completed += 1
        elif record.status == StageStatus.RUNNING and running_stage is None:
            running_stage = stage
        elif record.status == StageStatus.FAILED and failed_stage is None:
            failed_stage = stage

    if failed_stage is not None:
        status_text = f"Failed: {failed_stage.label}"
    elif running_stage is not None:
        status_text = f"Running [{running_stage.label}]..."
    elif total > 0 and completed == total:
        status_text = UI_STAGES_COMPLETE
    else:
        status_text = UI_PREPARING_NEXT_STAGE

    return StageProgress(completed=completed, total=total, status_text=status_text, overall_status_text=None)


class PipelineRunner(QObject):
    """Thin QProcess wrapper: launches a `pipeline.run` argv, merges
    stdout/stderr into one stream (the console log shows both together), and
    re-emits it as Qt signals the window can connect to directly.
    """

    output_received = Signal(str)
    finished = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    def start(self, argv: list[str]) -> None:
        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(REPO_ROOT))
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        # Python only line-buffers stdout when it's a real terminal, piped to
        # QProcess like this, it's fully block-buffered by default, so a print
        # can sit unflushed for a long time (confirmed: pipeline.run's own
        # early "Found an existing run..." print was arriving after every
        # stage's output, only flushed at process exit). PYTHONUNBUFFERED
        # forces unbuffered stdout/stderr for this process AND everything it
        # subprocess.run()s in turn (stages 0-8 directly, stage 10 through its
        # `pixi run -e export ...` wrapper), since env vars inherit down the
        # whole chain, restoring live output without touching pipeline code.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(env)
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)
        program, *args = argv
        self._process.start(program, args)

    def stop(self) -> None:
        if self._process is not None:
            self._process.kill()

    def _on_ready_read(self) -> None:
        assert self._process is not None
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.output_received.emit(data)

    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.finished.emit(exit_code)
