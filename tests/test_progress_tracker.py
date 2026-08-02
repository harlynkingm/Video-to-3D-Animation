"""Unit tests for progress_tracker.py's own pure logic -- no GPU/checkpoints
needed, always runs.
"""

from __future__ import annotations

import json

from conftest import TEST_VIDEO_PATH, make_run_input
from pipeline.create_run import create_run
from pipeline.progress_tracker import RunRecord, StageName, StageStatus, validate_camera_input, validate_video_input


def test_validate_camera_input_accepts_focal_length_path():
    run_input = make_run_input()  # conftest's own default: focal_length_mm/sensor_width_mm set, no intrinsics_k
    assert validate_camera_input(run_input) is None


def test_validate_camera_input_accepts_intrinsics_k_path():
    run_input = make_run_input(focal_length_mm=0.0, sensor_width_mm=0.0, intrinsics_k=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
    assert validate_camera_input(run_input) is None


def test_validate_camera_input_rejects_neither_given():
    run_input = make_run_input(focal_length_mm=0.0, sensor_width_mm=0.0)
    assert validate_camera_input(run_input) is not None


def test_validate_camera_input_rejects_both_given():
    run_input = make_run_input(intrinsics_k=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])  # focal/sensor still set
    assert validate_camera_input(run_input) is not None


def test_validate_camera_input_rejects_only_one_of_focal_or_sensor():
    run_input = make_run_input(sensor_width_mm=0.0)  # focal_length_mm still set, sensor_width_mm zeroed
    assert validate_camera_input(run_input) is not None


def test_validate_video_input_accepts_a_real_video_file_with_no_source_fps():
    run_input = make_run_input(video_path=str(TEST_VIDEO_PATH))  # source_fps left at its default (None)
    assert validate_video_input(run_input) is None


def test_validate_video_input_accepts_a_directory_with_source_fps(tmp_path):
    run_input = make_run_input(video_path=str(tmp_path), source_fps=24.0)
    assert validate_video_input(run_input) is None


def test_validate_video_input_rejects_a_directory_without_source_fps(tmp_path):
    run_input = make_run_input(video_path=str(tmp_path))  # source_fps left at its default (None)
    assert validate_video_input(run_input) is not None


def test_validate_video_input_rejects_a_directory_with_a_non_positive_source_fps(tmp_path):
    run_input = make_run_input(video_path=str(tmp_path), source_fps=0.0)
    assert validate_video_input(run_input) is not None


def test_create_run_stamps_created_at_and_updated_at(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    assert runRecord.created_at > 0
    assert runRecord.updated_at > 0


def test_save_refreshes_updated_at_but_leaves_created_at_alone(tmp_path, monkeypatch):
    times = iter([100.0, 100.0, 200.0])
    monkeypatch.setattr("time.time", lambda: next(times))

    # First time.time() call: create_run()'s own created_at=time.time(). Second:
    # the save() create_run() makes internally, which also stamps updated_at.
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    assert runRecord.created_at == 100.0
    assert runRecord.updated_at == 100.0

    # A later save (via mark_progress, same as every stage's own completion)
    # refreshes updated_at again but must never touch created_at.
    runRecord.mark_progress(StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs={})
    assert runRecord.created_at == 100.0
    assert runRecord.updated_at == 200.0


def test_load_defaults_missing_timestamps_to_zero(tmp_path):
    # A progress.json from before this field existed -- confirms an old run
    # directory still loads instead of erroring on the newly-required keys.
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    data = json.loads(runRecord.path.read_text())
    del data["created_at"]
    del data["updated_at"]
    runRecord.path.write_text(json.dumps(data))

    reloaded = RunRecord.load(runRecord.progress_dir)
    assert reloaded.created_at == 0.0
    assert reloaded.updated_at == 0.0
