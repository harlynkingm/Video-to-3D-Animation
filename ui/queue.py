"""The run queue: a sequence of RunFormState snapshots executed one after
another through a single reused PipelineRunner. QueueItem/the pure helpers
below are Qt-free and testable on their own (same split as pipeline_runner.py);
QueueRunner is the thin Qt-facing orchestrator that drives them.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from pipeline.progress_tracker import RunRecord
from ui.pipeline_runner import (
    PipelineRunner,
    RunFormState,
    StageProgress,
    build_run_argv,
    compute_stage_progress,
    stage_count_in_range,
)

UI_QUEUE_IDLE = "Queue idle"
UI_QUEUE_STOPPED = "Queue stopped"
UI_QUEUE_COMPLETE = "Queue complete"

# What a freshly-constructed RunFormState looks like, summarize_run_form_state
# below only reports a field when it differs from this, so "what counts as
# worth showing" can never drift out of sync with RunFormState's own defaults.
_DEFAULT_STATE = RunFormState(destination_folder="")


class QueueItemStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class QueueItem:
    state: RunFormState
    status: QueueItemStatus = QueueItemStatus.PENDING
    # Frozen at add/edit time (see build_queue_item), not recomputed while the
    # item sits in the queue or runs: once a run starts, pipeline.run itself
    # overwrites progress.json's input to match this item's own state, so a
    # live re-diff against it would collapse out from under the user right as
    # the run they just kicked off begins, with no visible cause.
    description: str = ""


def build_queue_item(state: RunFormState, status: QueueItemStatus = QueueItemStatus.PENDING) -> QueueItem:
    try:
        existing_run_record = RunRecord.load(state.destination_folder)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        existing_run_record = None  # progress.json doesn't exist yet, or is mid-write
    return QueueItem(state=state, status=status, description=summarize_run_form_state(state, existing_run_record))


def summarize_run_form_state(state: RunFormState, existing_run_record: RunRecord | None = None) -> str:
    # Ends up being a single line summary of the form state
    lines: list[str] = []
    if (state.start_stage, state.stop_stage) != (_DEFAULT_STATE.start_stage, _DEFAULT_STATE.stop_stage):
        if state.start_stage == state.stop_stage:
            lines.append(f"stage {state.start_stage}")
        else:
            lines.append(f"stages {state.start_stage}-{state.stop_stage}")
    else:
        lines.append("all stages")
    if state.human_prompt:
        # Only add human prompt if different than existing run
        if existing_run_record is None:
            lines.append(f"\"{state.human_prompt}\"")
        elif state.human_prompt != existing_run_record.input.human_prompt:
            lines.append(f"\"{state.human_prompt}\"")
    if state.object_prompt:
        # Only add object prompt if different than existing run
        if existing_run_record is None:
            lines.append(f"\"{state.object_prompt}\"")
        elif state.object_prompt != existing_run_record.input.object_prompt:
            lines.append(f"\"{state.object_prompt}\"")
    if state.focal_length_mm is not None and state.sensor_width_mm is not None:
        # Only add focal length/sensor width if different than existing run
        if existing_run_record is None:
            lines.append(f"focal length {state.focal_length_mm}mm, sensor width {state.sensor_width_mm}mm")
        elif state.focal_length_mm != existing_run_record.input.focal_length_mm or state.sensor_width_mm != existing_run_record.input.sensor_width_mm:
            lines.append(f"focal length {state.focal_length_mm}mm, sensor width {state.sensor_width_mm}mm")
    if state.is_image_sequence and state.source_fps is not None:
        # Only add source fps if different than existing run
        if existing_run_record is None:
            lines.append(f"{state.source_fps}fps")
        elif state.source_fps != existing_run_record.input.source_fps:
            lines.append(f"{state.source_fps}fps")
    if state.object_shape != _DEFAULT_STATE.object_shape:
        # Only add object shape if different than existing run
        if existing_run_record is None:
            lines.append(f"object shape {state.object_shape}")
        elif state.object_shape != existing_run_record.input.object_shape_hint:
            lines.append(f"object shape {state.object_shape}")
    if state.finger_motion != _DEFAULT_STATE.finger_motion:
        # Only add finger motion if different than existing run
        if existing_run_record is None:
            lines.append(f"finger motion {state.finger_motion}")
        elif state.finger_motion != existing_run_record.input.finger_motion:
            lines.append(f"finger motion {state.finger_motion}")
    if state.force_all:
        lines.append("force re-run")
    if state.render_previews:
        lines.append("render previews")
    return ", ".join(lines)


def next_pending_index(queue: list[QueueItem]) -> int | None:
    """The item the queue runs next: the first still-PENDING one. COMPLETE
    items are done; FAILED items are never auto-resumed (matching
    halt-on-failure), retrying one means re-queueing it or fixing the issue
    and running again.
    """
    for index, item in enumerate(queue):
        if item.status == QueueItemStatus.PENDING:
            return index
    return None


def compute_queue_progress(
    queue: list[QueueItem],
    current_index: int | None,
    current_run_record: RunRecord | None,
) -> StageProgress:
    """Whole-queue equivalent of compute_stage_progress: completed/total
    stage counts summed across every item, a full item is worth its own
    stage count, not a flat 1, so the bar "fragments by stage AND queue
    item", plus one status line naming whichever item is currently
    running (or the failure that halted the queue, if any).
    """
    completed = 0
    total = 0
    failed_item: QueueItem | None = None
    for item in queue:
        item_total = stage_count_in_range(item.state.start_stage, item.state.stop_stage)
        total += item_total
        if item.status == QueueItemStatus.COMPLETE:
            completed += item_total
        elif item.status == QueueItemStatus.FAILED and failed_item is None:
            failed_item = item

    overall_status_text: str | None = None
    if current_index is not None and current_run_record is not None and 0 <= current_index < len(queue):
        current_item = queue[current_index]
        item_progress = compute_stage_progress(
            current_run_record, current_item.state.start_stage, current_item.state.stop_stage,
        )
        completed += item_progress.completed
        title = Path(current_item.state.destination_folder).name
        status_text = item_progress.status_text
        overall_status_text =  f"{title} [{current_index + 1}/{len(queue)}]"
    elif failed_item is not None:
        status_text = f"Failed: {Path(failed_item.state.destination_folder).name}"
    elif total > 0 and completed == total:
        status_text = UI_QUEUE_COMPLETE
    elif completed > 0:
        status_text = UI_QUEUE_STOPPED
    else:
        status_text = UI_QUEUE_IDLE

    return StageProgress(completed=completed, total=total, status_text=status_text, overall_status_text=overall_status_text)


class QueueRunner(QObject):
    """Runs each PENDING QueueItem in `self.queue` in order through one
    reused PipelineRunner (never recreated, the same "one subprocess at a
    time" shape PipelineRunner itself already has for a single run), halting
   , not advancing, on a non-zero exit.
    """

    output_received = Signal(str)
    item_started = Signal(int)
    item_finished = Signal(int, bool)  # index, success
    all_finished = Signal()
    running_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pipeline_runner = PipelineRunner(self)
        self._pipeline_runner.output_received.connect(self.output_received)
        self._pipeline_runner.finished.connect(self._on_item_finished)
        self.queue: list[QueueItem] = []
        self.current_index: int | None = None
        self._stop_requested = False
        # Tracks the queue's own logical running state explicitly, rather
        # than deriving it from `_pipeline_runner.is_running()`, that
        # process isn't actually started until partway through
        # `_run_next_pending()`, so a listener reading is_running() during
        # the *same* running_changed(True) emit (e.g. to enable/disable
        # buttons) would otherwise still see False
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self.is_running() or next_pending_index(self.queue) is None:
            return
        self._stop_requested = False
        self._running = True
        self.running_changed.emit(True)
        self._run_next_pending()

    def stop(self) -> None:
        self._stop_requested = True
        self._pipeline_runner.stop()

    def _run_next_pending(self) -> None:
        index = next_pending_index(self.queue)
        if index is None:
            self.current_index = None
            self._running = False
            self.running_changed.emit(False)
            self.all_finished.emit()
            return
        self.current_index = index
        item = self.queue[index]
        item.status = QueueItemStatus.RUNNING
        self.item_started.emit(index)
        self._pipeline_runner.start(build_run_argv(item.state))

    def _on_item_finished(self, exit_code: int) -> None:
        index = self.current_index
        assert index is not None
        success = exit_code == 0 and not self._stop_requested
        self.queue[index].status = QueueItemStatus.COMPLETE if success else QueueItemStatus.FAILED
        self.item_finished.emit(index, success)
        if self._stop_requested or not success:
            self.current_index = None
            self._running = False
            self.running_changed.emit(False)
            self.all_finished.emit()
            return
        self._run_next_pending()
