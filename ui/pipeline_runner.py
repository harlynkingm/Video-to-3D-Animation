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

from PySide6.QtCore import QObject, QProcess, Signal

# ui/ -> repo root, matching pipeline/run.py's own _REPO_ROOT (pipeline/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunFormState:
    video_path: str
    destination_folder: str
    human_prompt: str
    focal_length_mm: float
    sensor_width_mm: float
    object_prompt: str | None = None
    is_image_sequence: bool = False
    source_fps: float | None = None
    start_stage: int = 0
    stop_stage: int = 9
    object_shape: str = "auto"
    force_all: bool = False
    render_previews: bool = False


def build_run_argv(state: RunFormState) -> list[str]:
    """`state` -> the same argv `pixi run -e main python -m pipeline.run ...`
    would receive by hand. Uses `sys.executable` rather than shelling out
    through `pixi run` again -- the UI process itself already runs under the
    `main` env's interpreter, so it's the same interpreter either way.
    """
    argv = [
        sys.executable, "-m", "pipeline.run",
        "--input-video", state.video_path,
        "--output-dir", state.destination_folder,
        "--human-prompt", state.human_prompt,
        "--focal-length-mm", str(state.focal_length_mm),
        "--sensor-width-mm", str(state.sensor_width_mm),
        "--start-on-stage", str(state.start_stage),
        "--stop-after-stage", str(state.stop_stage),
        "--object-shape-hint", state.object_shape,
    ]
    if state.object_prompt:
        argv += ["--object-prompt", state.object_prompt]
    if state.is_image_sequence and state.source_fps is not None:
        argv += ["--source-fps", str(state.source_fps)]
    if state.force_all:
        argv.append("--force-all")
    if state.render_previews:
        argv.append("--render-previews")
    return argv


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
