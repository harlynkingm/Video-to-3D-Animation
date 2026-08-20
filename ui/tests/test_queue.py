from pipeline.progress_tracker import RunInput, RunRecord, StageRecord, StageStatus
from ui.pipeline_runner import RunFormState
from ui.queue import (
    UI_QUEUE_COMPLETE,
    UI_QUEUE_IDLE,
    UI_QUEUE_STOPPED,
    QueueItem,
    QueueItemStatus,
    build_queue_item,
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
    assert summarize_run_form_state(_state()) == "all stages"


def test_summarize_run_form_state_includes_human_and_object_prompt():
    output = summarize_run_form_state(_state(human_prompt="a person", object_prompt="a teddy bear"))

    assert "a person" in output
    assert "a teddy bear" in output


def test_summarize_run_form_state_ignores_existing_run_record_with_same_values():
    existing_run_record = _run_record()
    state = _state(human_prompt="a person")
    output = summarize_run_form_state(state, existing_run_record)

    assert output == "all stages"


def test_summarize_run_form_state_includes_human_prompt_when_different_from_existing_run_record():
    existing_run_record = _run_record()
    state = _state(human_prompt="a dog")
    output = summarize_run_form_state(state, existing_run_record)

    assert "a dog" in output


def test_summarize_run_form_state_includes_camera_when_only_focal_length_differs_from_existing_run_record():
    # A half-changed camera pair (only one of the two values actually moved)
    # must still surface, since the displayed pair as a whole differs from
    # what's on record.
    existing_run_record = _run_record(focal_length_mm=26.0, sensor_width_mm=36.0)
    state = _state(focal_length_mm=50.0, sensor_width_mm=36.0)
    output = summarize_run_form_state(state, existing_run_record)

    assert "50.0" in output


def test_summarize_run_form_state_omits_camera_when_unchanged_from_existing_run_record():
    existing_run_record = _run_record(focal_length_mm=26.0, sensor_width_mm=36.0)
    state = _state(focal_length_mm=26.0, sensor_width_mm=36.0)
    output = summarize_run_form_state(state, existing_run_record)

    assert "focal" not in output.lower()


def test_summarize_run_form_state_includes_camera_when_set():
    output = summarize_run_form_state(_state(focal_length_mm=26.0, sensor_width_mm=36.0))

    assert "26.0" in output


def test_summarize_run_form_state_omits_camera_when_only_one_value_set():
    # Half-populated camera info isn't meaningful to show, both or neither.
    output = summarize_run_form_state(_state(focal_length_mm=26.0))

    assert not "focal" in output.lower()


def test_summarize_run_form_state_includes_image_sequence_with_fps():
    output = summarize_run_form_state(_state(is_image_sequence=True, source_fps=29.97))

    assert "29.97" in output.lower()


def test_summarize_run_form_state_includes_stage_range_when_not_default():
    output = summarize_run_form_state(_state(start_stage=4, stop_stage=6))

    assert "4" in output.lower()


def test_summarize_run_form_state_says_all_stages_instead_of_a_specific_range_when_default():
    output = summarize_run_form_state(_state(start_stage=0, stop_stage=10))

    assert "all stages" in output
    assert "0-10" not in output


def test_summarize_run_form_state_includes_object_shape_when_not_auto():
    output = summarize_run_form_state(_state(object_shape="box"))

    assert "box" in output


def test_summarize_run_form_state_includes_force_rerun_and_render_previews_when_checked():
    output = summarize_run_form_state(_state(force_all=True, render_previews=True))

    assert "force" in output.lower()
    assert "preview" in output.lower()

    # ...and omits both when unchecked (the default).
    default_output = summarize_run_form_state(_state())
    assert "force" not in default_output.lower()
    assert "preview" not in default_output.lower()


# -- build_queue_item --------------------------------------------------------


def test_build_queue_item_uses_the_existing_progress_json_for_its_description(tmp_path):
    RunRecord(
        run_id="test", progress_dir=str(tmp_path), input=RunInput(video_path="v.mp4", human_prompt="a person"),
    ).save()
    item = build_queue_item(_state(destination_folder=str(tmp_path), human_prompt="a dog"))

    assert "a dog" in item.description


def test_build_queue_item_tolerates_a_destination_with_no_progress_json_yet(tmp_path):
    item = build_queue_item(_state(destination_folder=str(tmp_path / "not_yet_run"), human_prompt="a person"))

    assert "a person" in item.description


def test_build_queue_item_description_is_frozen_even_if_progress_json_changes_later(tmp_path):
    RunRecord(
        run_id="test", progress_dir=str(tmp_path), input=RunInput(video_path="v.mp4", human_prompt="a person"),
    ).save()
    item = build_queue_item(_state(destination_folder=str(tmp_path), human_prompt="a dog"))
    assert "a dog" in item.description

    # progress.json changes underneath it (e.g. the run this item kicked off
    # just started and overwrote its own recorded input to match) -- the
    # cached description must not silently change out from under the user.
    RunRecord(
        run_id="test", progress_dir=str(tmp_path), input=RunInput(video_path="v.mp4", human_prompt="a dog"),
    ).save()

    assert "a dog" in item.description


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


def _run_record(focal_length_mm: float = 0.0, sensor_width_mm: float = 0.0, **stage_statuses: StageStatus) -> RunRecord:
    return RunRecord(
        run_id="test",
        progress_dir="runs/test",
        input=RunInput(
            video_path="v.mp4", human_prompt="a person",
            focal_length_mm=focal_length_mm, sensor_width_mm=sensor_width_mm,
        ),
        stages={name: StageRecord(status=status) for name, status in stage_statuses.items()},
    )


def test_compute_queue_progress_nothing_started():
    queue = [
        QueueItem(state=_state(destination_folder="a", start_stage=0, stop_stage=10)),
        QueueItem(state=_state(destination_folder="b", start_stage=0, stop_stage=4)),
    ]
    progress = compute_queue_progress(queue, current_index=None, current_run_record=None)

    assert progress.completed == 0
    assert progress.total == 11 + 5
    assert progress.status_text == UI_QUEUE_IDLE


def test_compute_queue_progress_one_complete_one_running():
    queue = [
        QueueItem(state=_state(destination_folder="a", start_stage=0, stop_stage=10), status=QueueItemStatus.COMPLETE),
        QueueItem(state=_state(destination_folder="b", start_stage=0, stop_stage=10), status=QueueItemStatus.RUNNING),
    ]
    run_record = _run_record(ingest_video=StageStatus.COMPLETE, mask_and_track=StageStatus.RUNNING)
    progress = compute_queue_progress(queue, current_index=1, current_run_record=run_record)

    # item 0 fully complete (11 stages) + item 1's own live progress (1 of 11 complete)
    assert progress.completed == 11 + 1
    assert progress.total == 11 + 11
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
