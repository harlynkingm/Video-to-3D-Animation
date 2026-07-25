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
)
from pipeline import run as run_pipeline_module
from pipeline.create_run import create_run
from pipeline.progress_tracker import ProgressRecord, RunInput, StageName, StageStatus


def _make_progress(tmp_path: Path) -> ProgressRecord:
    run_input = RunInput(
        video_path=str(TEST_VIDEO_PATH),
        human_prompt=HUMAN_PROMPT,
        object_prompt=OBJECT_PROMPT,
        focal_length_mm=FOCAL_LENGTH_MM,
        sensor_width_mm=SENSOR_WIDTH_MM,
    )
    return create_run(tmp_path / "run", run_input, run_id="test")


def _fake_load_stage_run(calls: list[StageName]):
    def load(index: int, stage_name: StageName):
        def fake_run(progress: ProgressRecord) -> dict[str, str]:
            calls.append(stage_name)
            return {}
        return fake_run
    return load


def test_load_stage_run_resolves_the_real_stage_module():
    from pipeline.stages import stage_0_ingest_video

    run = run_pipeline_module._load_stage_run(0, StageName.STAGE_0_INGEST_VIDEO)
    assert run is stage_0_ingest_video.run


def test_run_pipeline_runs_every_stage_when_no_bound_given(tmp_path, monkeypatch):
    progress = _make_progress(tmp_path)
    calls: list[StageName] = []
    monkeypatch.setattr(run_pipeline_module, "_load_stage_run", _fake_load_stage_run(calls))

    run_pipeline_module.run_pipeline(progress)

    assert calls == run_pipeline_module.ORDERED_STAGES
    assert all(progress.is_complete(s) for s in run_pipeline_module.ORDERED_STAGES)


def test_run_pipeline_stops_after_the_requested_stage(tmp_path, monkeypatch):
    progress = _make_progress(tmp_path)
    calls: list[StageName] = []
    monkeypatch.setattr(run_pipeline_module, "_load_stage_run", _fake_load_stage_run(calls))

    run_pipeline_module.run_pipeline(progress, stop_after_stage=1)

    assert calls == run_pipeline_module.ORDERED_STAGES[:2]
    assert progress.is_complete(StageName.STAGE_1_MASK_AND_TRACK)
    assert not progress.is_complete(StageName.STAGE_2_ESTIMATE_HUMAN_MOTION)


def test_run_pipeline_clamps_a_stop_after_stage_beyond_what_is_implemented(tmp_path, monkeypatch, capsys):
    progress = _make_progress(tmp_path)
    calls: list[StageName] = []
    monkeypatch.setattr(run_pipeline_module, "_load_stage_run", _fake_load_stage_run(calls))

    run_pipeline_module.run_pipeline(progress, stop_after_stage=99)

    assert calls == run_pipeline_module.ORDERED_STAGES
    assert "Only stages 0-" in capsys.readouterr().out


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
    monkeypatch.setattr(run_pipeline_module, "run_pipeline", lambda progress, stop_after_stage=None: None)
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

    assert "resuming" in capsys.readouterr().out
    reloaded = ProgressRecord.load(progress_dir)
    assert reloaded.run_id == "original-run-id"


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
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    progress_json = json.loads((run_dir / "progress.json").read_text())
    for stage_name in (
        "ingest_video", "mask_and_track", "estimate_human_motion",
        "estimate_depth", "estimate_hands", "retarget_hands", "align_scene_scale",
    ):
        assert progress_json["stages"][stage_name]["status"] == "complete"


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
    assert progress_json["stages"]["ingest_video"]["status"] == "complete"
    assert progress_json["stages"]["mask_and_track"]["status"] == "complete"
    assert progress_json["stages"]["estimate_human_motion"]["status"] == "pending"
