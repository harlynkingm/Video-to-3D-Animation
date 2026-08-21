"""Unit tests for progress_tracker.py's own pure logic, no GPU/checkpoints
needed, always runs.
"""

from __future__ import annotations

import argparse
import json

import pytest

from conftest import TEST_VIDEO_PATH, make_run_input
from pipeline.create_run import create_run
from pipeline.progress_tracker import (
    SCHEMA_VERSION,
    STAGE_DEPENDS_ON,
    RunInput,
    RunRecord,
    FingerMotion,
    FineTuningOptions,
    StageName,
    StageRecord,
    StageStatus,
    add_run_input_arguments,
    apply_run_input_overrides,
    ordered_stages,
    run_input_from_args,
    stage_by_number,
    validate_camera_input,
    validate_video_input,
)

_BASE_ARGV = [
    "--input-video", "v.mp4", "--human-prompt", "a person",
    "--focal-length-mm", "26", "--sensor-width-mm", "36",
]


def _parse_run_input(extra_argv: list[str]) -> RunInput:
    parser = argparse.ArgumentParser()
    add_run_input_arguments(parser)
    return run_input_from_args(parser.parse_args(_BASE_ARGV + extra_argv))


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
    # A progress.json from before this field existed, confirms an old run
    # directory still loads instead of erroring on the newly-required keys.
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    data = json.loads(runRecord.path.read_text())
    del data["created_at"]
    del data["updated_at"]
    runRecord.path.write_text(json.dumps(data))

    reloaded = RunRecord.load(runRecord.progress_dir)
    assert reloaded.created_at == 0.0
    assert reloaded.updated_at == 0.0


def test_load_accepts_utf8_progress_json_with_or_without_a_bom(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    original_json = runRecord.path.read_text(encoding="utf-8")

    # The pipeline writes ordinary UTF-8. External Windows tools may rewrite
    # the same JSON as UTF-8-with-BOM; both must remain resumable.
    for encoding in ("utf-8", "utf-8-sig"):
        runRecord.path.write_text(original_json, encoding=encoding)

        reloaded = RunRecord.load(runRecord.progress_dir)

        assert reloaded.run_id == "test"
        assert reloaded.input == runRecord.input


def test_load_rejects_an_obsolete_output_field_with_value_error(tmp_path):
    """An old, unsupported schema should be reportable by UI callers, not
    leak a dataclass-constructor TypeError from the loader."""
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    data = json.loads(runRecord.path.read_text())
    data["outputs"]["final_fbx"] = "output.fbx"
    runRecord.path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Invalid progress record"):
        RunRecord.load(runRecord.progress_dir)


def test_save_writes_utf8_without_a_bom(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")

    assert not runRecord.path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_progress_json_keeps_run_defaults_but_omits_smoothness_defaults(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    data = json.loads(runRecord.path.read_text())

    assert "hand_finger_min_cutoff_hz" not in data["input"]
    assert "hand_finger_decimate_deg" not in data["input"]
    assert data["input"]["render_hands_preview"] is False
    assert data["input"]["finger_motion"] == "smooth"
    assert "sam_track_max_bridge_frames" not in data["input"]
    assert "hand_wrist_max_deviation_deg" not in data["input"]
    assert data["fine_tuning_overrides"] == {}
    assert RunRecord.load(runRecord.progress_dir).fine_tuning.hand_finger_min_cutoff_hz == 0.15


def test_loading_legacy_record_drops_its_implicit_runtime_defaults(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    data = json.loads(runRecord.path.read_text())
    data.pop("fine_tuning_overrides")  # reproduce the version-1 serialization
    data["input"]["hand_finger_min_cutoff_hz"] = 0.15
    data["input"]["hand_finger_decimate_deg"] = 1.5
    data["input"].pop("finger_motion")  # predates the run-level profile field
    runRecord.path.write_text(json.dumps(data))

    reloaded = RunRecord.load(runRecord.progress_dir)

    assert reloaded.fine_tuning.hand_finger_min_cutoff_hz == 0.15
    assert reloaded.fine_tuning.hand_finger_decimate_deg == 1.5
    assert reloaded.fine_tuning_overrides == {}
    assert reloaded.input.finger_motion is FingerMotion.SMOOTH


def test_explicit_runtime_override_is_preserved_even_when_it_matches_a_default(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.fine_tuning_overrides = {"hand_finger_min_cutoff_hz": 0.18}
    runRecord.save()
    data = json.loads(runRecord.path.read_text())

    assert "hand_finger_min_cutoff_hz" not in data["input"]
    assert data["fine_tuning_overrides"] == {"hand_finger_min_cutoff_hz": 0.18}
    assert RunRecord.load(runRecord.progress_dir).fine_tuning.hand_finger_min_cutoff_hz == 0.18


def test_finger_motion_uses_a_string_enum_and_persists_as_its_string_value(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(finger_motion=FingerMotion.DETAILED), run_id="test")
    data = json.loads(runRecord.path.read_text())

    assert data["input"]["finger_motion"] == "detailed"
    assert RunRecord.load(runRecord.progress_dir).input.finger_motion is FingerMotion.DETAILED


def test_finger_motion_cli_default_and_override_use_the_string_enum():
    assert _parse_run_input([]).finger_motion is FingerMotion.SMOOTH
    assert _parse_run_input(["--finger-motion", "detailed"]).finger_motion is FingerMotion.DETAILED


def test_apply_run_input_overrides_changes_finger_motion_only_when_supplied():
    existing = make_run_input()
    parser = argparse.ArgumentParser()
    add_run_input_arguments(parser, required=False)

    unchanged = apply_run_input_overrides(existing, parser.parse_args([]))
    detailed = apply_run_input_overrides(existing, parser.parse_args(["--finger-motion", "detailed"]))

    assert unchanged.finger_motion is FingerMotion.SMOOTH
    assert detailed.finger_motion is FingerMotion.DETAILED


def test_non_smoothing_fine_tuning_override_uses_the_same_key_value_object(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.fine_tuning_overrides = {"sam_track_max_bridge_frames": 12}
    runRecord.save()

    reloaded = RunRecord.load(runRecord.progress_dir)

    assert isinstance(reloaded.fine_tuning, FineTuningOptions)
    assert reloaded.fine_tuning.sam_track_max_bridge_frames == 12


def test_smoothness_override_rejects_a_non_numeric_value(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.fine_tuning_overrides = {"hand_finger_min_cutoff_hz": "0.18"}  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="int or float"):
        runRecord.save()


def test_update_schema_adds_a_stage_the_run_predates(tmp_path):
    # Simulates an old progress.json written before a stage existed in the
    # DAG at all, create_run() itself always writes every stage the current
    # DAG knows about, so drop one afterward to reproduce that shape.
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    del runRecord.stages[StageName.STAGE_9_CAPTURE_FACE.value]

    runRecord.update_schema()

    added = runRecord.stages[StageName.STAGE_9_CAPTURE_FACE.value]
    assert added.status == StageStatus.PENDING
    assert added.depends_on == [dep.value for dep in STAGE_DEPENDS_ON[StageName.STAGE_9_CAPTURE_FACE]]


def test_update_schema_refreshes_depends_on_for_an_existing_stage(tmp_path):
    # The DAG can change after a stage's own record was first written (see
    # STAGE_DEPENDS_ON's own comment for real examples), update_schema
    # must overwrite a stale depends_on, not just leave whatever a stage
    # happened to be created with.
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.stages[StageName.STAGE_9_CAPTURE_FACE.value].depends_on = ["some_stale_dependency"]

    runRecord.update_schema()

    current = [dep.value for dep in STAGE_DEPENDS_ON[StageName.STAGE_9_CAPTURE_FACE]]
    assert runRecord.stages[StageName.STAGE_9_CAPTURE_FACE.value].depends_on == current


def test_incomplete_dependencies_reports_direct_prerequisites_and_statuses(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.mark_progress(StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs={})
    runRecord.mark_progress(StageName.STAGE_2_ESTIMATE_HUMAN_MOTION, StageStatus.FAILED)

    assert runRecord.incomplete_dependencies(StageName.STAGE_9_CAPTURE_FACE) == [
        (StageName.STAGE_1_MASK_AND_TRACK, StageStatus.PENDING),
        (StageName.STAGE_2_ESTIMATE_HUMAN_MOTION, StageStatus.FAILED),
    ]


def test_update_schema_never_touches_an_existing_stage_own_progress(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.mark_progress(
        StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs={"frames": "frames.mp4"},
    )

    runRecord.update_schema()

    record = runRecord.stages[StageName.STAGE_0_INGEST_VIDEO.value]
    assert record.status == StageStatus.COMPLETE
    assert record.outputs == {"frames": "frames.mp4"}
    assert record.error is None


def test_update_schema_never_discards_a_stage_no_longer_in_the_dag(tmp_path):
    # A stage this run has a real record for but the current code no longer
    # lists (e.g. a renamed/removed stage) must survive, no recorded
    # progress should ever be silently dropped by a schema migration.
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.stages["some_removed_stage"] = StageRecord(status=StageStatus.COMPLETE)

    runRecord.update_schema()

    assert runRecord.stages["some_removed_stage"].status == StageStatus.COMPLETE


def test_update_schema_bumps_schema_version_and_persists_to_disk(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.schema_version = 0  # simulate a run written by an older schema

    runRecord.update_schema()
    assert runRecord.schema_version == SCHEMA_VERSION

    # update_schema() saves internally, confirm the migration actually
    # reached disk, not just the in-memory object.
    reloaded = RunRecord.load(runRecord.progress_dir)
    assert reloaded.schema_version == SCHEMA_VERSION
    assert StageName.STAGE_9_CAPTURE_FACE.value in reloaded.stages


def test_ordered_stages_has_one_entry_per_stage_number_in_ascending_order():
    stages = ordered_stages()
    numbers = [stage.stage_number for stage in stages]

    assert numbers == sorted(numbers)
    assert len(numbers) == len(set(numbers))  # no stage_number repeated
    assert numbers == list(range(11))  # stages 0-10, no gaps


def test_ordered_stages_prefers_each_stage_own_top_level_member():
    # Stage 1 and stage 6 both have sub-progress labels sharing their
    # stage_number (STAGE_1A/1B_*, STAGE_6B_*), ordered_stages() must
    # still surface the real top-level stage pipeline.run actually invokes,
    # not one of those internal-reporting-only sub-labels.
    stages_by_number = {stage.stage_number: stage for stage in ordered_stages()}

    assert stages_by_number[1] == StageName.STAGE_1_MASK_AND_TRACK
    assert stages_by_number[6] == StageName.STAGE_6_ALIGN_SCENE_SCALE


def test_stage_by_number_finds_the_boundary_and_a_middle_stage():
    assert stage_by_number(0) == StageName.STAGE_0_INGEST_VIDEO
    assert stage_by_number(4) == StageName.STAGE_4_ESTIMATE_HANDS
    assert stage_by_number(9) == StageName.STAGE_9_CAPTURE_FACE
    assert stage_by_number(10) == StageName.STAGE_10_EXPORT


def test_stage_by_number_also_prefers_the_top_level_member():
    assert stage_by_number(1) == StageName.STAGE_1_MASK_AND_TRACK
    assert stage_by_number(6) == StageName.STAGE_6_ALIGN_SCENE_SCALE


def test_stage_by_number_returns_none_when_out_of_range():
    assert stage_by_number(-1) is None
    assert stage_by_number(11) is None


def test_render_previews_sweeps_real_preview_flags():
    run_input = _parse_run_input(["--render-previews"])
    assert run_input.render_mask_previews is True
    assert run_input.render_contacts_preview is True


def test_render_previews_does_not_disable_face_capture():
    # skip_face_capture is a control flag, not a preview output, sweeping
    # it to True under --render-previews would silently mean "disable face
    # capture", the opposite of "render every preview" (see cli_field's own
    # is_render_preview docstring).
    run_input = _parse_run_input(["--render-previews"])
    assert run_input.skip_face_capture is False


def test_skip_face_capture_flag_works_standalone():
    run_input = _parse_run_input(["--skip-face-capture"])
    assert run_input.skip_face_capture is True
    assert run_input.render_mask_previews is False  # untouched


def test_apply_run_input_overrides_render_previews_also_spares_skip_face_capture():
    # Same guarantee as run_input_from_args above, but through update_run's
    # optional-args merge path instead of fresh construction.
    existing = make_run_input()
    parser = argparse.ArgumentParser()
    add_run_input_arguments(parser, required=False)
    args = parser.parse_args(["--render-previews"])

    updated = apply_run_input_overrides(existing, args)

    assert updated.render_mask_previews is True
    assert updated.skip_face_capture is False
