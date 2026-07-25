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

Each stage module is imported *inside* its own fixture, not at the top of this
file, deliberately: every stage module imports its adapter at that module's own
top level (e.g. `stage_3_estimate_depth` imports `depth_anything3_adapter`,
which pulls in `depth_anything_3` and transitively `xformers`), and a plain
CPU-only machine (no NVIDIA driver at all -- not just "no GPU", a genuinely
different situation from a dev machine that always has a real driver present)
turned out unable to even *import* one of those packages without crashing
natively -- no Python exception, no traceback, just an instant, silent process
death. Importing all six stage modules unconditionally at collection time (as
this file used to) forced that crash on every CI run, before any test's own
skip-gate on `torch.cuda.is_available()` ever got a chance to run. Deferring
each import into its own fixture means a driver-less machine only ever imports
`stage_0_ingest_video` (needs no heavy ML libraries) for real; every other
stage's risky import happens only after that fixture's own GPU check passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pipeline.create_run import create_run
from pipeline.progress_tracker import ProgressRecord, RunInput, StageName, StageStatus

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

# Same path stage_6_align_scene_scale.py computes on its own -- defined here
# too (not imported from there) so any test file can check for the SMPL-X
# model file's presence without importing that module and its heavy transitive
# chain (see this file's own docstring above for why that chain is unsafe to
# import unconditionally).
SMPLX_MODEL_PATH = TESTS_DIR.parent / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"


@pytest.fixture(scope="session")
def progress(tmp_path_factory) -> ProgressRecord:
    run_dir = tmp_path_factory.mktemp("pipeline_test_run")
    run_input = RunInput(
        video_path=str(TEST_VIDEO_PATH),
        human_prompt=HUMAN_PROMPT,
        object_prompt=OBJECT_PROMPT,
        focal_length_mm=FOCAL_LENGTH_MM,
        sensor_width_mm=SENSOR_WIDTH_MM,
        render_mask_previews=True,
        render_motion_preview=True,
        render_depth_preview=True,
        render_scene_preview=True,
        render_retarget_preview=True,
    )
    return create_run(run_dir, run_input, run_id="test")


@pytest.fixture(scope="session")
def stage_0_result(progress: ProgressRecord) -> dict[str, str]:
    from pipeline.stages import stage_0_ingest_video

    outputs = stage_0_ingest_video.run(progress)
    progress.mark_progress(StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_1_result(progress: ProgressRecord, stage_0_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not SAM31_CHECKPOINT.exists():
        pytest.skip("needs the SAM 3.1 checkpoint (see README's Setup section)")

    from pipeline.stages import stage_1_mask_and_track

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

    from pipeline.stages import stage_2_estimate_human_motion

    outputs = stage_2_estimate_human_motion.run(progress)
    progress.mark_progress(StageName.STAGE_2_ESTIMATE_HUMAN_MOTION, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_3_result(progress: ProgressRecord, stage_1_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    from pipeline.stages import stage_3_estimate_depth

    outputs = stage_3_estimate_depth.run(progress)
    progress.mark_progress(StageName.STAGE_3_ESTIMATE_DEPTH, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_4_result(
    progress: ProgressRecord, stage_1_result: dict[str, str], stage_2_result: dict[str, str]
) -> dict[str, str]:
    # Depends on stage_2_result (not just stage_1_result) now: stage 4 checks
    # every raw wrist estimate against GVHMR's own elbow orientation for
    # biomechanical plausibility before its own smoothing runs, so it needs the
    # body motion, and (via SmplxSkeleton) the SMPL-X model file for the
    # kinematic tree.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    missing = [p.name for p in (HAMER_CHECKPOINT, VITPOSE_CHECKPOINT) if not p.exists()]
    if missing:
        pytest.skip(f"needs the HaMeR + ViTPose checkpoints (missing: {missing}; see README's Setup section)")
    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    from pipeline.stages import stage_4_estimate_hands

    outputs = stage_4_estimate_hands.run(progress)
    progress.mark_progress(StageName.STAGE_4_ESTIMATE_HANDS, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_5_result(
    progress: ProgressRecord, stage_2_result: dict[str, str], stage_4_result: dict[str, str]
) -> dict[str, str]:
    # No GPU/checkpoints of its own, but SmplxSkeleton (for the kinematic tree)
    # and the optional preview both need the registration-gated SMPL-X model
    # file -- checked via this file's own SMPLX_MODEL_PATH, not by importing
    # stage_6's module, to avoid its heavy transitive chain.
    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    from pipeline.stages import stage_5_retarget_hands

    outputs = stage_5_retarget_hands.run(progress)
    progress.mark_progress(StageName.STAGE_5_RETARGET_HANDS, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_6_result(
    progress: ProgressRecord, stage_2_result: dict[str, str], stage_3_result: dict[str, str]
) -> dict[str, str]:
    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    from pipeline.stages import stage_6_align_scene_scale

    outputs = stage_6_align_scene_scale.run(progress)
    progress.mark_progress(StageName.STAGE_6_ALIGN_SCENE_SCALE, StageStatus.COMPLETE, outputs=outputs)
    return outputs
