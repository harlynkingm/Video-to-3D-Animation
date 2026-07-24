"""Shared fixtures for the pipeline's stage regression tests.

These are whole-stage regression tests, not unit tests of individual
functions: each fixture actually runs a real stage's `run()` against a small,
committed test video (`assets/tiny_tennis_clip.mp4` -- 20 frames, both the
tracked human and object stay clearly visible throughout) and hands the real
output forward to the next stage, exactly like a real pipeline run. Stages
are session-scoped so checkpoints only load once per test session, not once
per test function.

Stage 1/2 need the real SAM 3.1/GVHMR checkpoints (gitignored, see README's
Setup section) and a CUDA GPU -- fixtures that need them call `pytest.skip()`
rather than failing when either is missing, so this suite still runs
(partially) on a machine that hasn't set those up yet. Stage 3's checkpoint
auto-downloads on first use (see depth_anything3_adapter.py), so its fixture
only gates on a CUDA GPU, not on the checkpoint already being present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pipeline.create_run import create_run
from pipeline.progress_tracker import ProgressRecord, RunInput, StageName, StageStatus
from pipeline.stages import (
    stage_0_ingest_video,
    stage_1_mask_and_track,
    stage_2_estimate_human_motion,
    stage_3_estimate_depth,
    stage_4_estimate_hands,
    stage_5_retarget_hands,
    stage_6_align_scene_scale,
)

TESTS_DIR = Path(__file__).parent
TEST_VIDEO_PATH = TESTS_DIR / "assets" / "tiny_tennis_clip.mp4"

# The exact frame count and resolution of the committed test clip (frames
# 73-92 of the reference tennis clip used throughout this project's own
# development) -- update these if the fixture video is ever replaced.
TEST_VIDEO_FRAME_COUNT = 20
TEST_VIDEO_WIDTH = 812
TEST_VIDEO_HEIGHT = 720
TEST_VIDEO_FPS = 29.83

HUMAN_PROMPT = "a tennis player"
OBJECT_PROMPT = "a tennis racket"
FOCAL_LENGTH_MM = 35.0
SENSOR_WIDTH_MM = 36.0

CHECKPOINTS_DIR = TESTS_DIR.parent / "checkpoints"
SAM31_CHECKPOINT = CHECKPOINTS_DIR / "sam3.1_multiplex_fp16.safetensors"
VITPOSE_CHECKPOINT = CHECKPOINTS_DIR / "vitpose.safetensors"
GVHMR_CHECKPOINTS = (
    VITPOSE_CHECKPOINT,
    CHECKPOINTS_DIR / "hmr2.safetensors",
    CHECKPOINTS_DIR / "gvhmr.safetensors",
)
HAMER_CHECKPOINT = CHECKPOINTS_DIR / "hamer.safetensors"


@pytest.fixture(scope="session")
def progress(tmp_path_factory) -> ProgressRecord:
    run_dir = tmp_path_factory.mktemp("pipeline_test_run")
    run_input = RunInput(
        video_path=str(TEST_VIDEO_PATH),
        human_prompt=HUMAN_PROMPT,
        object_prompt=OBJECT_PROMPT,
        focal_length_mm=FOCAL_LENGTH_MM,
        sensor_width_mm=SENSOR_WIDTH_MM,
        dump_mask_previews=True,
        dump_motion_preview=True,
        dump_depth_preview=True,
        dump_scene_preview=True,
        dump_retarget_preview=True,
    )
    return create_run(run_dir, run_input, run_id="test")


@pytest.fixture(scope="session")
def stage_0_result(progress: ProgressRecord) -> dict[str, str]:
    outputs = stage_0_ingest_video.run(progress)
    progress.mark_progress(StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_1_result(progress: ProgressRecord, stage_0_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not SAM31_CHECKPOINT.exists():
        pytest.skip("needs the SAM 3.1 checkpoint (see README's Setup section)")

    outputs = stage_1_mask_and_track.run(progress)
    progress.mark_progress(StageName.STAGE_1_MASK_AND_TRACK, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_2_result(progress: ProgressRecord, stage_1_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    missing = [p.name for p in GVHMR_CHECKPOINTS if not p.exists()]
    if missing:
        pytest.skip(f"needs the GVHMR checkpoints (missing: {missing}; see README's Setup section)")

    outputs = stage_2_estimate_human_motion.run(progress)
    progress.mark_progress(StageName.STAGE_2_ESTIMATE_HUMAN_MOTION, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_3_result(progress: ProgressRecord, stage_1_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    outputs = stage_3_estimate_depth.run(progress)
    progress.mark_progress(StageName.STAGE_3_ESTIMATE_DEPTH, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_4_result(progress: ProgressRecord, stage_1_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    missing = [p.name for p in (HAMER_CHECKPOINT, VITPOSE_CHECKPOINT) if not p.exists()]
    if missing:
        pytest.skip(f"needs the HaMeR + ViTPose checkpoints (missing: {missing}; see README's Setup section)")

    outputs = stage_4_estimate_hands.run(progress)
    progress.mark_progress(StageName.STAGE_4_ESTIMATE_HANDS, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_5_result(
    progress: ProgressRecord, stage_2_result: dict[str, str], stage_4_result: dict[str, str]
) -> dict[str, str]:
    # No GPU/checkpoints of its own, but SmplxSkeleton (for the kinematic tree)
    # and the optional preview both need the registration-gated SMPL-X model file.
    if not stage_6_align_scene_scale.SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    outputs = stage_5_retarget_hands.run(progress)
    progress.mark_progress(StageName.STAGE_5_RETARGET_HANDS, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_6_result(
    progress: ProgressRecord, stage_2_result: dict[str, str], stage_3_result: dict[str, str]
) -> dict[str, str]:
    if not stage_6_align_scene_scale.SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    outputs = stage_6_align_scene_scale.run(progress)
    progress.mark_progress(StageName.STAGE_6_ALIGN_SCENE_SCALE, StageStatus.COMPLETE, outputs=outputs)
    return outputs
