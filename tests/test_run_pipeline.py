"""Tests for pipeline.run, the one-shot create+run-every-stage command.

Split into two kinds: fast, GPU-free tests of the stage-sequencing plumbing
itself (--stop-after-stage bound, resume-skips-create_run, dynamic
stage-module naming) using a monkeypatched stage runner so no real model/GPU
is touched, plus real end-to-end subprocess tests (mirrors
test_pipeline_end_to_end.py's style) that need the actual checkpoints/GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import (
    FOCAL_LENGTH_MM,
    GVHMR_CHECKPOINTS,
    HAMER_CHECKPOINT,
    HUMAN_PROMPT,
    OBJECT_PROMPT,
    SAM31_CHECKPOINT,
    SENSOR_WIDTH_MM,
    SMPLX_MODEL_PATH,
    TEST_VIDEO_PATH,
    assert_stages_complete,
    make_run_input,
)
from pipeline import run as run_pipeline_module
from pipeline.create_run import create_run
from pipeline.progress_tracker import RunRecord, RunInput, StageName, StageStatus


def _make_runRecord(tmp_path: Path) -> RunRecord:
    return create_run(tmp_path / "run", make_run_input(), run_id="test")


def _fake_run_stage_subprocess(calls: list[StageName]):
    """Stands in for the real `_run_stage_subprocess` (which shells out to a
    real `python -m pipeline.stages...` subprocess) so plumbing-only tests
    don't need any GPU/checkpoints. Marks the stage complete itself,
    mirroring what the real subprocess's own `cli_entrypoint`/`run_stage`
    would do -- reloading from disk first since that's genuinely a separate
    process there, not the same `runRecord` object the test holds."""
    def fake(stage_name: StageName, progress_dir: Path, force: bool) -> None:
        calls.append(stage_name)
        runRecord = RunRecord.load(progress_dir)
        runRecord.mark_progress(stage_name, StageStatus.COMPLETE, outputs={})
    return fake


def _fake_run_export_subprocess(calls: list[StageName]):
    """Same idea as `_fake_run_stage_subprocess`, for `_run_export_subprocess`
    (the separate `pixi run -e export` dispatch stage 9 needs)."""
    def fake(progress_dir: Path, force: bool) -> None:
        calls.append(StageName.STAGE_9_EXPORT)
        runRecord = RunRecord.load(progress_dir)
        runRecord.mark_progress(StageName.STAGE_9_EXPORT, StageStatus.COMPLETE, outputs={})
    return fake


def test_run_pipeline_runs_every_stage_when_no_bound_given(tmp_path, monkeypatch):
    runRecord = _make_runRecord(tmp_path)
    calls: list[StageName] = []
    monkeypatch.setattr(run_pipeline_module, "_run_stage_subprocess", _fake_run_stage_subprocess(calls))
    monkeypatch.setattr(run_pipeline_module, "_run_export_subprocess", _fake_run_export_subprocess(calls))

    run_pipeline_module.run_pipeline(runRecord)

    assert calls == run_pipeline_module.ORDERED_STAGES
    reloaded = RunRecord.load(runRecord.progress_dir)
    assert all(reloaded.is_complete(s) for s in run_pipeline_module.ORDERED_STAGES)


def test_run_pipeline_starts_on_the_requested_stage(tmp_path, monkeypatch):
    runRecord = _make_runRecord(tmp_path)
    calls: list[StageName] = []
    monkeypatch.setattr(run_pipeline_module, "_run_stage_subprocess", _fake_run_stage_subprocess(calls))
    monkeypatch.setattr(run_pipeline_module, "_run_export_subprocess", _fake_run_export_subprocess(calls))

    run_pipeline_module.run_pipeline(runRecord, start_on_stage=2)

    assert calls == run_pipeline_module.ORDERED_STAGES[2:]
    reloaded = RunRecord.load(runRecord.progress_dir)
    assert not reloaded.is_complete(StageName.STAGE_1_MASK_AND_TRACK)
    assert all(reloaded.is_complete(s) for s in run_pipeline_module.ORDERED_STAGES[2:])


def test_run_pipeline_stops_after_the_requested_stage(tmp_path, monkeypatch):
    runRecord = _make_runRecord(tmp_path)
    calls: list[StageName] = []
    monkeypatch.setattr(run_pipeline_module, "_run_stage_subprocess", _fake_run_stage_subprocess(calls))

    run_pipeline_module.run_pipeline(runRecord, stop_after_stage=1)

    assert calls == run_pipeline_module.ORDERED_STAGES[:2]
    reloaded = RunRecord.load(runRecord.progress_dir)
    assert reloaded.is_complete(StageName.STAGE_1_MASK_AND_TRACK)
    assert not reloaded.is_complete(StageName.STAGE_2_ESTIMATE_HUMAN_MOTION)

def test_run_pipeline_starts_on_and_stops_after_the_requested_stages(tmp_path, monkeypatch):
    runRecord = _make_runRecord(tmp_path)
    calls: list[StageName] = []
    monkeypatch.setattr(run_pipeline_module, "_run_stage_subprocess", _fake_run_stage_subprocess(calls))

    run_pipeline_module.run_pipeline(runRecord, start_on_stage=2, stop_after_stage=3)

    assert calls == run_pipeline_module.ORDERED_STAGES[2:4]
    reloaded = RunRecord.load(runRecord.progress_dir)
    assert not reloaded.is_complete(StageName.STAGE_1_MASK_AND_TRACK)
    assert reloaded.is_complete(StageName.STAGE_2_ESTIMATE_HUMAN_MOTION)


def test_run_pipeline_clamps_a_stop_after_stage_beyond_what_is_implemented(tmp_path, monkeypatch, capsys):
    runRecord = _make_runRecord(tmp_path)
    calls: list[StageName] = []
    monkeypatch.setattr(run_pipeline_module, "_run_stage_subprocess", _fake_run_stage_subprocess(calls))
    monkeypatch.setattr(run_pipeline_module, "_run_export_subprocess", _fake_run_export_subprocess(calls))

    run_pipeline_module.run_pipeline(runRecord, stop_after_stage=99)

    assert calls == run_pipeline_module.ORDERED_STAGES
    assert "Only stages 0-" in capsys.readouterr().out


def test_run_pipeline_passes_force_through_to_each_stage_subprocess(tmp_path, monkeypatch):
    runRecord = _make_runRecord(tmp_path)
    seen_force: list[bool] = []
    monkeypatch.setattr(
        run_pipeline_module, "_run_stage_subprocess",
        lambda stage_name, progress_dir, force: seen_force.append(force),
    )
    monkeypatch.setattr(
        run_pipeline_module, "_run_export_subprocess",
        lambda progress_dir, force: seen_force.append(force),
    )

    run_pipeline_module.run_pipeline(runRecord, force_all=True)

    assert seen_force and all(seen_force)


def test_run_stage_subprocess_tolerates_a_crash_after_the_stage_already_completed(tmp_path, monkeypatch):
    """Regression test: a stage's own subprocess (`bpy` in particular is
    known to do this) can crash during its own interpreter teardown *after*
    already saving its real output and marking the stage complete in
    `progress.json` -- that shouldn't be treated as a failure."""
    runRecord = _make_runRecord(tmp_path)
    runRecord.mark_progress(StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs={"fake": "output"})
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(args=cmd, returncode=3221225477)
    )

    run_pipeline_module._run_stage_subprocess(StageName.STAGE_0_INGEST_VIDEO, Path(runRecord.progress_dir), force=False)


def test_run_stage_subprocess_still_raises_when_the_stage_never_completed(tmp_path, monkeypatch):
    runRecord = _make_runRecord(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(args=cmd, returncode=1))

    with pytest.raises(subprocess.CalledProcessError):
        run_pipeline_module._run_stage_subprocess(StageName.STAGE_0_INGEST_VIDEO, Path(runRecord.progress_dir), force=False)


def test_main_resumes_an_existing_run_instead_of_recreating_it(tmp_path, monkeypatch, capsys):
    progress_dir = tmp_path / "run"
    original = create_run(
        progress_dir,
        RunInput(
            video_path=str(TEST_VIDEO_PATH),
            human_prompt=HUMAN_PROMPT,
            focal_length_mm=FOCAL_LENGTH_MM,
            sensor_width_mm=SENSOR_WIDTH_MM,
        ),
        run_id="original-run-id",
    )
    original.mark_progress(StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs={"fake": "output"})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("create_run should not be called for an already-existing run")

    monkeypatch.setattr(run_pipeline_module, "create_run", fail_if_called)
    monkeypatch.setattr(
        run_pipeline_module, "run_pipeline",
        lambda progress, start_on_stage=None, stop_after_stage=None, force_all=False: None,
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "run.py",
            "--output-dir", str(progress_dir),
            "--run-id", "a-different-run-id-that-should-be-ignored",
            "--input-video", str(TEST_VIDEO_PATH),
            "--human-prompt", HUMAN_PROMPT,
            "--focal-length-mm", str(FOCAL_LENGTH_MM),
            "--sensor-width-mm", str(SENSOR_WIDTH_MM),
        ],
    )

    run_pipeline_module.main()

    assert "existing run" in capsys.readouterr().out
    reloaded = RunRecord.load(progress_dir)
    assert reloaded.run_id == "original-run-id"


def test_main_resumes_without_requiring_run_input_flags(tmp_path, monkeypatch, capsys):
    """Regression test: resuming an existing run used to fail with argparse's
    own "required" error for --input-video/--human-prompt/etc., since those
    flags were registered as required regardless of whether a run already
    existed at --output-dir -- only --output-dir itself should still be
    required when resuming.
    """
    progress_dir = tmp_path / "run"
    create_run(
        progress_dir,
        RunInput(
            video_path=str(TEST_VIDEO_PATH), human_prompt=HUMAN_PROMPT,
            focal_length_mm=FOCAL_LENGTH_MM, sensor_width_mm=SENSOR_WIDTH_MM,
        ),
        run_id="original-run-id",
    )

    monkeypatch.setattr(
        run_pipeline_module, "run_pipeline",
        lambda progress, start_on_stage=None, stop_after_stage=None, force_all=False: None,
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--output-dir", str(progress_dir)])

    run_pipeline_module.main()  # would previously raise SystemExit from argparse

    assert "existing run" in capsys.readouterr().out


def test_main_applies_an_override_flag_when_resuming(tmp_path, monkeypatch, capsys):
    progress_dir = tmp_path / "run"
    create_run(
        progress_dir,
        RunInput(
            video_path=str(TEST_VIDEO_PATH), human_prompt=HUMAN_PROMPT,
            focal_length_mm=FOCAL_LENGTH_MM, sensor_width_mm=SENSOR_WIDTH_MM,
        ),
        run_id="original-run-id",
    )

    monkeypatch.setattr(
        run_pipeline_module, "run_pipeline",
        lambda progress, start_on_stage=None, stop_after_stage=None, force_all=False: None,
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--output-dir", str(progress_dir), "--render-mask-previews"])

    run_pipeline_module.main()
    capsys.readouterr()

    reloaded = RunRecord.load(progress_dir)
    assert reloaded.input.render_mask_previews is True
    assert reloaded.input.video_path == str(TEST_VIDEO_PATH)  # untouched fields carry over


def test_main_still_requires_run_input_flags_for_a_fresh_run(tmp_path, monkeypatch):
    """The other half of the regression: creating a genuinely new run must
    still fail fast with a clear argparse error when the required flags are
    missing, not silently proceed with an incomplete RunInput."""
    progress_dir = tmp_path / "does-not-exist-yet"
    monkeypatch.setattr(sys, "argv", ["run.py", "--output-dir", str(progress_dir)])

    with pytest.raises(SystemExit):
        run_pipeline_module.main()


pytestmark_end_to_end = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not SAM31_CHECKPOINT.exists()
    or not HAMER_CHECKPOINT.exists()
    or not SMPLX_MODEL_PATH.exists()
    or any(not p.exists() for p in GVHMR_CHECKPOINTS),
    reason="needs a CUDA GPU, all model checkpoints, and the SMPL-X model file (see README's Setup section)",
)


@pytestmark_end_to_end
def test_pipeline_run_end_to_end(tmp_path):
    """Capped at stage 7 -- `export` needs `bpy` from a separate pixi
    environment this test's own `main`-env interpreter can't provide (see
    `tests/test_stage_9_export.py` for its own dedicated coverage)."""
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable, "-m", "pipeline.run",
            "--output-dir", str(run_dir),
            "--input-video", str(TEST_VIDEO_PATH),
            "--human-prompt", HUMAN_PROMPT,
            "--object-prompt", OBJECT_PROMPT,
            "--focal-length-mm", str(FOCAL_LENGTH_MM),
            "--sensor-width-mm", str(SENSOR_WIDTH_MM),
            "--stop-after-stage", "7",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    progress_json = json.loads((run_dir / "progress.json").read_text())
    assert_stages_complete(progress_json, through=StageName.STAGE_7_ANNOTATE_CONTACTS)


@pytestmark_end_to_end
def test_pipeline_run_stop_after_stage(tmp_path):
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable, "-m", "pipeline.run",
            "--output-dir", str(run_dir),
            "--input-video", str(TEST_VIDEO_PATH),
            "--human-prompt", HUMAN_PROMPT,
            "--object-prompt", OBJECT_PROMPT,
            "--focal-length-mm", str(FOCAL_LENGTH_MM),
            "--sensor-width-mm", str(SENSOR_WIDTH_MM),
            "--stop-after-stage", "1",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    progress_json = json.loads((run_dir / "progress.json").read_text())
    assert_stages_complete(progress_json, through=StageName.STAGE_1_MASK_AND_TRACK)
    assert progress_json["stages"][StageName.STAGE_2_ESTIMATE_HUMAN_MOTION.value]["status"] == "pending"
