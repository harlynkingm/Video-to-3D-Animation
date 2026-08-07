from pipeline.progress_tracker import RunInput, RunRecord, StageRecord, StageStatus
from ui.pipeline_runner import RunFormState
from ui.queue import (
    UI_QUEUE_COMPLETE,
    UI_QUEUE_IDLE,
    UI_QUEUE_STOPPED,
    QueueItem,
    QueueItemStatus,
    compute_queue_progress,
    next_pending_index,
    summarize_run_form_state,
)


def _state(**overrides) -> RunFormState:
    defaults = dict(destination_folder="C:/runs/my_clip")
    defaults.update(overrides)
    return RunFormState(**defaults)


# -- summarize_run_form_state --------------------------------------------


def test_summarize_run_form_state_all_default_mentions_only_all_stages():
    # Always a single joined string; with nothing else set, it's just the
    # "all stages" fallback (no specific range, no other non-default field).
    assert summarize_run_form_state(_state()) == ["all stages"]


def test_summarize_run_form_state_includes_human_and_object_prompt():
    lines = summarize_run_form_state(_state(human_prompt="a person", object_prompt="a teddy bear"))

    assert any("a person" in line for line in lines)
    assert any("a teddy bear" in line for line in lines)


def test_summarize_run_form_state_includes_camera_when_set():
    lines = summarize_run_form_state(_state(focal_length_mm=26.0, sensor_width_mm=36.0))

    assert any("26.0" in line and "36.0" in line for line in lines)


def test_summarize_run_form_state_omits_camera_when_only_one_value_set():
    # Half-populated camera info isn't meaningful to show -- both or neither.
    lines = summarize_run_form_state(_state(focal_length_mm=26.0))

    assert not any("focal" in line.lower() for line in lines)


def test_summarize_run_form_state_includes_image_sequence_with_fps():
    lines = summarize_run_form_state(_state(is_image_sequence=True, source_fps=29.97))

    assert any("29.97" in line for line in lines)


def test_summarize_run_form_state_includes_stage_range_when_not_default():
    lines = summarize_run_form_state(_state(start_stage=4, stop_stage=6))

    assert any("4" in line and "6" in line for line in lines)


def test_summarize_run_form_state_says_all_stages_instead_of_a_specific_range_when_default():
    lines = summarize_run_form_state(_state(start_stage=0, stop_stage=10))

    assert "all stages" in lines[0]
    assert "0-10" not in lines[0]


def test_summarize_run_form_state_includes_object_shape_when_not_auto():
    lines = summarize_run_form_state(_state(object_shape="box"))

    assert any("box" in line for line in lines)


def test_summarize_run_form_state_includes_force_rerun_and_render_previews_when_checked():
    lines = summarize_run_form_state(_state(force_all=True, render_previews=True))

    assert any("force" in line.lower() for line in lines)
    assert any("preview" in line.lower() for line in lines)

    # ...and omits both when unchecked (the default).
    default_lines = summarize_run_form_state(_state())
    assert "force" not in default_lines[0].lower()
    assert "preview" not in default_lines[0].lower()


# -- next_pending_index ----------------------------------------------------


def test_next_pending_index_empty_queue():
    assert next_pending_index([]) is None


def test_next_pending_index_all_complete():
    queue = [QueueItem(state=_state(), status=QueueItemStatus.COMPLETE) for _ in range(3)]
    assert next_pending_index(queue) is None


def test_next_pending_index_returns_first_pending():
    queue = [
        QueueItem(state=_state(), status=QueueItemStatus.COMPLETE),
        QueueItem(state=_state(), status=QueueItemStatus.PENDING),
        QueueItem(state=_state(), status=QueueItemStatus.PENDING),
    ]
    assert next_pending_index(queue) == 1


def test_next_pending_index_never_returns_a_failed_item():
    queue = [
        QueueItem(state=_state(), status=QueueItemStatus.FAILED),
        QueueItem(state=_state(), status=QueueItemStatus.PENDING),
    ]
    assert next_pending_index(queue) == 1


# -- compute_queue_progress -------------------------------------------------


def _run_record(**stage_statuses: StageStatus) -> RunRecord:
    return RunRecord(
        run_id="test",
        progress_dir="runs/test",
        input=RunInput(video_path="v.mp4", human_prompt="a person"),
        stages={name: StageRecord(status=status) for name, status in stage_statuses.items()},
    )


def test_compute_queue_progress_nothing_started():
    queue = [
        QueueItem(state=_state(destination_folder="a", start_stage=0, stop_stage=10)),
        QueueItem(state=_state(destination_folder="b", start_stage=0, stop_stage=4)),
    ]
    progress = compute_queue_progress(queue, current_index=None, current_run_record=None)

    assert progress.completed == 0
    assert progress.total == 10 + 5
    assert progress.status_text == UI_QUEUE_IDLE


def test_compute_queue_progress_one_complete_one_running():
    queue = [
        QueueItem(state=_state(destination_folder="a", start_stage=0, stop_stage=10), status=QueueItemStatus.COMPLETE),
        QueueItem(state=_state(destination_folder="b", start_stage=0, stop_stage=10), status=QueueItemStatus.RUNNING),
    ]
    run_record = _run_record(ingest_video=StageStatus.COMPLETE, mask_and_track=StageStatus.RUNNING)
    progress = compute_queue_progress(queue, current_index=1, current_run_record=run_record)

    # item 0 fully complete (10 stages) + item 1's own live progress (1 of 10 complete)
    assert progress.completed == 10 + 1
    assert progress.total == 10 + 10
    # status_text is the running item's own stage progress; overall_status_text
    # names which queue item that is and where it sits in the queue.
    assert "generate masks" in progress.status_text
    assert progress.overall_status_text is not None
    assert "b" in progress.overall_status_text
    assert "[2/2]" in progress.overall_status_text


def test_compute_queue_progress_all_complete():
    queue = [QueueItem(state=_state(destination_folder=name), status=QueueItemStatus.COMPLETE) for name in "ab"]
    progress = compute_queue_progress(queue, current_index=None, current_run_record=None)

    assert progress.completed == progress.total
    assert progress.status_text == UI_QUEUE_COMPLETE


def test_compute_queue_progress_reports_the_failure_that_halted_it():
    queue = [
        QueueItem(state=_state(destination_folder="a"), status=QueueItemStatus.COMPLETE),
        QueueItem(state=_state(destination_folder="broken_run"), status=QueueItemStatus.FAILED),
        QueueItem(state=_state(destination_folder="c"), status=QueueItemStatus.PENDING),
    ]
    progress = compute_queue_progress(queue, current_index=None, current_run_record=None)

    assert "broken_run" in progress.status_text
    assert "Failed" in progress.status_text


def test_compute_queue_progress_stopped_partway_with_no_failure():
    queue = [
        QueueItem(state=_state(destination_folder="a"), status=QueueItemStatus.COMPLETE),
        QueueItem(state=_state(destination_folder="b"), status=QueueItemStatus.PENDING),
    ]
    progress = compute_queue_progress(queue, current_index=None, current_run_record=None)

    assert progress.completed > 0
    assert progress.status_text == UI_QUEUE_STOPPED
