"""Unit tests for the shared stage execution harness."""

from __future__ import annotations

import pytest

from conftest import make_run_input
from pipeline.create_run import create_run
from pipeline.pipeline_stage_base import StageDependenciesNotMetError, run_stage
from pipeline.progress_tracker import StageName, StageStatus


def test_dependency_error_names_each_missing_stage_and_its_required_action(tmp_path):
    runRecord = create_run(tmp_path / "run", make_run_input(), run_id="test")
    runRecord.mark_progress(StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs={})
    runRecord.mark_progress(StageName.STAGE_2_ESTIMATE_HUMAN_MOTION, StageStatus.FAILED)

    with pytest.raises(StageDependenciesNotMetError) as exc_info:
        run_stage(runRecord, lambda _: {}, StageName.STAGE_9_CAPTURE_FACE)

    assert str(exc_info.value) == (
        "capture_face cannot run because these required stages are incomplete:\n"
        "  - stage 1: generate masks (mask_and_track) is pending: run it\n"
        "  - stage 2: estimate human motion (estimate_human_motion) is failed: rerun it"
    )
