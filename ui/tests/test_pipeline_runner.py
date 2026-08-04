import sys

from pipeline.progress_tracker import RunInput, RunRecord, StageName, StageRecord, StageStatus
from ui.pipeline_runner import RunFormState, build_run_argv, compute_stage_progress


def _base_state(**overrides) -> RunFormState:
    defaults = dict(
        video_path="C:/clips/video.mp4",
        destination_folder="C:/runs/my_clip",
        human_prompt="a person",
        focal_length_mm=26.0,
        sensor_width_mm=36.0,
    )
    defaults.update(overrides)
    return RunFormState(**defaults)


def test_build_run_argv_basic_video_run():
    argv = build_run_argv(_base_state())

    assert argv[:3] == [sys.executable, "-m", "pipeline.run"]
    assert "--input-video" in argv and argv[argv.index("--input-video") + 1] == "C:/clips/video.mp4"
    assert "--output-dir" in argv and argv[argv.index("--output-dir") + 1] == "C:/runs/my_clip"
    assert "--human-prompt" in argv and argv[argv.index("--human-prompt") + 1] == "a person"
    assert "--focal-length-mm" in argv and argv[argv.index("--focal-length-mm") + 1] == "26.0"
    assert "--sensor-width-mm" in argv and argv[argv.index("--sensor-width-mm") + 1] == "36.0"
    assert "--start-on-stage" in argv and argv[argv.index("--start-on-stage") + 1] == "0"
    assert "--stop-after-stage" in argv and argv[argv.index("--stop-after-stage") + 1] == "9"
    assert "--object-shape-hint" in argv and argv[argv.index("--object-shape-hint") + 1] == "auto"
    assert "--object-prompt" not in argv
    assert "--source-fps" not in argv
    assert "--force-all" not in argv
    assert "--render-previews" not in argv


def test_build_run_argv_includes_object_prompt_when_set():
    argv = build_run_argv(_base_state(object_prompt="a teddy bear"))

    assert "--object-prompt" in argv
    assert argv[argv.index("--object-prompt") + 1] == "a teddy bear"


def test_build_run_argv_image_sequence_includes_source_fps():
    argv = build_run_argv(_base_state(
        video_path="C:/clips/frames", is_image_sequence=True, source_fps=29.97,
    ))

    assert "--source-fps" in argv
    assert argv[argv.index("--source-fps") + 1] == "29.97"


def test_build_run_argv_omits_source_fps_when_not_image_sequence():
    # A stray source_fps value should be ignored unless is_image_sequence is set.
    argv = build_run_argv(_base_state(is_image_sequence=False, source_fps=29.97))

    assert "--source-fps" not in argv


def test_build_run_argv_force_all_and_render_previews_flags():
    argv = build_run_argv(_base_state(force_all=True, render_previews=True))

    assert "--force-all" in argv
    assert "--render-previews" in argv


def test_build_run_argv_stage_range_and_object_shape():
    argv = build_run_argv(_base_state(start_stage=4, stop_stage=6, object_shape="box"))

    assert argv[argv.index("--start-on-stage") + 1] == "4"
    assert argv[argv.index("--stop-after-stage") + 1] == "6"
    assert argv[argv.index("--object-shape-hint") + 1] == "box"


def test_build_run_argv_omits_optional_fields_when_resuming():
    # Only destination_folder given (as when resuming/re-running an existing
    # output folder without re-filling the rest of the form) -- pipeline.run
    # itself fills in everything else from the existing run's stored input,
    # so none of these should be sent as empty/zero overrides.
    argv = build_run_argv(RunFormState(destination_folder="C:/runs/my_clip"))

    assert argv[argv.index("--output-dir") + 1] == "C:/runs/my_clip"
    assert "--input-video" not in argv
    assert "--human-prompt" not in argv
    assert "--object-prompt" not in argv
    assert "--focal-length-mm" not in argv
    assert "--sensor-width-mm" not in argv
    assert "--source-fps" not in argv
    # Stage range and object shape are pipeline.run's own flags/defaults,
    # unrelated to resuming, so they're always sent.
    assert "--start-on-stage" in argv
    assert "--stop-after-stage" in argv
    assert "--object-shape-hint" in argv


def _run_record(**stage_statuses: StageStatus) -> RunRecord:
    return RunRecord(
        run_id="test",
        progress_dir="runs/test",
        input=RunInput(video_path="v.mp4", human_prompt="a person"),
        stages={name: StageRecord(status=status) for name, status in stage_statuses.items()},
    )


def test_compute_stage_progress_nothing_started():
    progress = compute_stage_progress(_run_record(), start_stage=0, stop_stage=9)

    assert progress.completed == 0
    assert progress.total == 10
    assert progress.status_text == "Preparing next stage..."


def test_compute_stage_progress_partial_complete_and_running():
    progress = compute_stage_progress(
        _run_record(
            ingest_video=StageStatus.COMPLETE,
            mask_and_track=StageStatus.COMPLETE,
            estimate_human_motion=StageStatus.RUNNING,
        ),
        start_stage=0, stop_stage=9,
    )

    assert progress.completed == 2
    assert progress.total == 10
    label = StageName.STAGE_2_ESTIMATE_HUMAN_MOTION.label
    assert progress.status_text == f"Running {label}..."


def test_compute_stage_progress_respects_stage_range():
    # Every stage complete, but only a 3-stage sub-range was actually run.
    progress = compute_stage_progress(
        _run_record(**{stage.value: StageStatus.COMPLETE for stage in StageName if stage.stage_number <= 9}),
        start_stage=2, stop_stage=4,
    )

    assert progress.completed == 3
    assert progress.total == 3
    assert progress.status_text == "All stages complete"


def test_compute_stage_progress_reports_failure():
    progress = compute_stage_progress(
        _run_record(ingest_video=StageStatus.COMPLETE, mask_and_track=StageStatus.FAILED),
        start_stage=0, stop_stage=9,
    )

    label = StageName.STAGE_1_MASK_AND_TRACK.label
    assert progress.status_text == f"Failed: {label}"
