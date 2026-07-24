"""Full pipeline regression test: runs every implemented stage as a real
subprocess (`python -m pipeline.stages...`), exactly how a user runs them,
against the small test clip -- exercising `create_run.py`, each stage's CLI
entrypoint, `progress.json` persistence, and the skip-if-already-complete/
`--force` behavior. The per-stage test files call each stage's `run()`
directly and check its own output in detail; this file is the only one that
exercises the actual subprocess/CLI dispatch path a real user goes through.

Needs the real checkpoints and a CUDA GPU -- skipped automatically otherwise.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pipeline.stages.stage_6_align_scene_scale import SMPLX_MODEL_PATH

from conftest import (
    FOCAL_LENGTH_MM,
    GVHMR_CHECKPOINTS,
    HAMER_CHECKPOINT,
    HUMAN_PROMPT,
    OBJECT_PROMPT,
    SAM31_CHECKPOINT,
    SENSOR_WIDTH_MM,
    TEST_VIDEO_PATH,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not SAM31_CHECKPOINT.exists()
    or not HAMER_CHECKPOINT.exists()
    or not SMPLX_MODEL_PATH.exists()
    or any(not p.exists() for p in GVHMR_CHECKPOINTS),
    reason="needs a CUDA GPU, all model checkpoints, and the SMPL-X model file (see README's Setup section)",
)


def _create_run(progress_dir: Path, *, object_prompt: str | None = OBJECT_PROMPT) -> subprocess.CompletedProcess:
    args = [
        sys.executable, "-m", "pipeline.create_run",
        "--progress-dir", str(progress_dir),
        "--video-path", str(TEST_VIDEO_PATH),
        "--human-prompt", HUMAN_PROMPT,
        "--focal-length-mm", str(FOCAL_LENGTH_MM),
        "--sensor-width-mm", str(SENSOR_WIDTH_MM),
    ]
    if object_prompt:
        args += ["--object-prompt", object_prompt]
    return subprocess.run(args, capture_output=True, text=True)


def _run_stage(module: str, progress_dir: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, "--progress-dir", str(progress_dir), *extra_args],
        capture_output=True, text=True,
    )


def test_full_pipeline_runs_end_to_end(tmp_path):
    run_dir = tmp_path / "run"
    created = _create_run(run_dir)
    assert created.returncode == 0, created.stderr

    for module in (
        "pipeline.stages.stage_0_ingest_video",
        "pipeline.stages.stage_1_mask_and_track",
        "pipeline.stages.stage_2_estimate_human_motion",
        "pipeline.stages.stage_3_estimate_depth",
        "pipeline.stages.stage_4_estimate_hands",
        "pipeline.stages.stage_5_retarget_hands",
        "pipeline.stages.stage_6_align_scene_scale",
    ):
        result = _run_stage(module, run_dir)
        assert result.returncode == 0, result.stderr

    progress_json = json.loads((run_dir / "progress.json").read_text())
    for stage_name in ("ingest_video", "mask_and_track", "estimate_human_motion", "estimate_depth", "estimate_hands", "retarget_hands", "align_scene_scale"):
        assert progress_json["stages"][stage_name]["status"] == "complete"

    motion_path = Path(progress_json["stages"]["estimate_human_motion"]["outputs"]["human_motion"])
    assert motion_path.exists()


def test_rerunning_a_completed_stage_skips_unless_forced(tmp_path):
    run_dir = tmp_path / "run"
    created = _create_run(run_dir, object_prompt=None)
    assert created.returncode == 0, created.stderr

    first = _run_stage("pipeline.stages.stage_0_ingest_video", run_dir)
    assert first.returncode == 0, first.stderr

    second = _run_stage("pipeline.stages.stage_0_ingest_video", run_dir)
    assert second.returncode == 0, second.stderr
    assert "already complete, skipping" in second.stdout

    forced = _run_stage("pipeline.stages.stage_0_ingest_video", run_dir, "--force")
    assert forced.returncode == 0, forced.stderr
    assert "already complete, skipping" not in forced.stdout
