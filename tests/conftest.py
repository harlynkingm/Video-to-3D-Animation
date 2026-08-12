"""Shared fixtures for the pipeline's stage regression tests.

These are whole-stage regression tests, not unit tests of individual
functions: each fixture actually runs a real stage's `run()` against a small,
committed test video (`assets/tiny_tennis_clip.mp4`, 20 frames, both the
tracked human and object stay clearly visible throughout) and hands the real
output forward to the next stage, exactly like a real pipeline run. Stages
are session-scoped so checkpoints only load once per test session, not once
per test function.

Stage 1/2 need the real SAM 3.1/GVHMR checkpoints (gitignored, see README's
Setup section) and a CUDA GPU, fixtures that need them call `pytest.skip()`
rather than failing when either is missing, so this suite still runs
(partially) on a machine that hasn't set those up yet. Stage 3's checkpoint
auto-downloads on first use (see depth_anything3_adapter.py), so its fixture
only gates on a CUDA GPU, not on the checkpoint already being present.

Stages 1-4 (mask_and_track, estimate_human_motion, estimate_depth,
estimate_hands) each load a separately-compiled native/CUDA model stack (SAM
3.1, GVHMR, DepthAnything3, HaMeR respectively). Loading more than one of
these into a single long-lived process, which is what session-scoped
fixtures calling each stage's `run()` directly would do, since later stages'
fixtures depend on earlier ones already having run in this same pytest
session, has been observed to segfault deep inside torch (see
`pipeline/run.py`'s own module docstring for the full story; the same crash
class shows up there when `pipeline.run` calls stages in-process). Their
fixtures below run the real stage through `pipeline.run`'s own
`_run_stage_subprocess` instead, exactly the same per-stage subprocess
dispatch a real user's `pipeline.run` invocation goes through, so this
process itself only ever holds torch (for reading saved tensors back) and
never any of those heavier stacks directly. Stages 5-8 don't load a heavy
model of their own (light numpy/torch math against already-computed stage
1-4 output), so they still run in-process.

Each of those still-in-process stage modules (0, 5, 6, 7, 8) is imported
*inside* its own fixture, not at the top of this file, deliberately: a plain
CPU-only machine (no NVIDIA driver at all, not just "no GPU", a genuinely
different situation from a dev machine that always has a real driver present)
turned out unable to even *import* some of these adapters' transitive
dependencies without crashing natively, no Python exception, no traceback,
just an instant, silent process death. Importing every stage module
unconditionally at collection time (as this file used to) forced that crash
on every CI run, before any test's own skip-gate on `torch.cuda.is_available()`
ever got a chance to run. Deferring each import into its own fixture means a
driver-less machine only ever imports `stage_0_ingest_video` (needs no heavy
ML libraries) for real.

`torch` itself is deferred the same way, for the same reason at one level up:
`tests/test_stage_10_export.py` and `tests/test_blender.py` run under the
separate `export` pixi environment, which has no torch installed at all
(kept minimal on purpose, see `pixi.toml`), but pytest loads this
conftest.py regardless of which test file it's collecting. A missing torch
only actually matters to the fixtures below that call
`torch.cuda.is_available()`, none of which those two files' own tests ever
request. `HAS_BPY`/`_isolated_bpy_scene` are the mirror case: `bpy` only
exists in `export`, so every other test file just sees `HAS_BPY = False`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.create_run import create_run
from pipeline.progress_tracker import RunRecord, RunInput, StageName, StageStatus
from pipeline.run import ORDERED_STAGES, _run_stage_subprocess

try:
    import torch
except ImportError:  # e.g. the export env, see module docstring
    torch = None

try:
    import bpy
    HAS_BPY = True
except ImportError:  # every env except `export`
    HAS_BPY = False

TESTS_DIR = Path(__file__).parent
TEST_VIDEO_PATH = TESTS_DIR / "assets" / "tiny_tennis_clip.mp4"


def assert_stages_complete(progress_json: dict, through: StageName | None = None) -> None:
    """Every stage up to and including `through` (default: every stage in
    `ORDERED_STAGES`, i.e. everything currently implemented) must show
    `status == "complete"` in an already-loaded progress.json. Derives the
    stage list from the same `ORDERED_STAGES` `pipeline.run` itself iterates,
    instead of a hand-typed list of stage-name strings, that list had
    already silently fallen out of sync once stage 7 shipped, missing from
    both of this project's own full-pipeline end-to-end tests (neither had
    been updated to check it).
    """
    stages = ORDERED_STAGES if through is None else ORDERED_STAGES[:ORDERED_STAGES.index(through) + 1]
    for stage in stages:
        assert progress_json["stages"][stage.value]["status"] == "complete", f"{stage.value} did not complete"


def _run_heavy_stage(runRecord: RunRecord, stage_name: StageName) -> dict[str, str]:
    """Runs one of stages 1-4 the same way a real user's `pipeline.run` does
   , as its own subprocess, rather than importing and calling `run()`
    directly in this pytest process; see this file's own module docstring
    for why. `runRecord` is mutated in place (not just returned) via
    `__dict__.update` from the subprocess's own saved `progress.json`, since
    every fixture/test downstream already holds a reference to this exact
    session-scoped object, not just this function's own return value,
    e.g. stage 1 setting `scene.anchor_frame_index` needs to reach them too.
    """
    _run_stage_subprocess(stage_name, Path(runRecord.progress_dir), force=False)
    reloaded = RunRecord.load(runRecord.progress_dir)
    runRecord.__dict__.update(reloaded.__dict__)
    return runRecord.stages[stage_name].outputs

# The exact frame count and resolution of the committed test clip (frames
# 73-92 of the reference tennis clip used throughout this project's own
# development), update these if the fixture video is ever replaced.
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

# Same path stage_6_align_scene_scale.py computes on its own, defined here
# too (not imported from there) so any test file can check for the SMPL-X
# model file's presence without importing that module and its heavy transitive
# chain (see this file's own docstring above for why that chain is unsafe to
# import unconditionally).
SMPLX_MODEL_PATH = TESTS_DIR.parent / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"

DECA_CHECKPOINT = CHECKPOINTS_DIR / "deca.safetensors"
MICA_CHECKPOINT = CHECKPOINTS_DIR / "mica.safetensors"
FACE_LANDMARKER_CHECKPOINT = CHECKPOINTS_DIR / "face_landmarker.task"
FLAME_MODEL_PATH = TESTS_DIR.parent / "body_models" / "flame" / "FLAME_NEUTRAL.npz"


if HAS_BPY:
    @pytest.fixture(autouse=True)
    def _isolated_bpy_scene():
        """A real, order-dependent bug was found running stage 10's own bpy
        tests together: a prior test's large build left bpy in a state that
        corrupted a *later* test's otherwise-correct output, even though
        `run()` already clears the scene at its own start (something more
        subtle than orphaned objects/actions). Clearing before *and*
        after each test, rather than trusting `run()`'s own single clear at
        the start, is what actually made the suite order-independent.
        """
        from pipeline.bpy.blender_scene import _clear_scene
        _clear_scene(bpy)
        yield
        _clear_scene(bpy)


def make_run_input(**overrides) -> RunInput:
    # A plain merge (not literal duplicate kwargs) so a caller can override
    # any of these defaults too, e.g. zeroing focal_length_mm/sensor_width_mm
    # to test the --intrinsics-k alternative camera-input path.
    defaults = dict(
        video_path=str(TEST_VIDEO_PATH),
        human_prompt=HUMAN_PROMPT,
        object_prompt=OBJECT_PROMPT,
        focal_length_mm=FOCAL_LENGTH_MM,
        sensor_width_mm=SENSOR_WIDTH_MM,
    )
    defaults.update(overrides)
    return RunInput(**defaults)


@pytest.fixture(scope="session")
def runRecord(tmp_path_factory) -> RunRecord:
    run_dir = tmp_path_factory.mktemp("pipeline_test_run")
    run_input = make_run_input(
        render_mask_previews=True,
        render_motion_preview=True,
        render_depth_preview=True,
        render_scene_preview=True,
        render_retarget_preview=True,
        render_contacts_preview=True,
    )
    return create_run(run_dir, run_input, run_id="test")


@pytest.fixture(scope="session")
def stage_0_result(runRecord: RunRecord) -> dict[str, str]:
    from pipeline.stages import stage_0_ingest_video

    outputs = stage_0_ingest_video.run(runRecord)
    runRecord.mark_progress(StageName.STAGE_0_INGEST_VIDEO, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_1_result(runRecord: RunRecord, stage_0_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not SAM31_CHECKPOINT.exists():
        pytest.skip("needs the SAM 3.1 checkpoint (see README's Setup section)")

    return _run_heavy_stage(runRecord, StageName.STAGE_1_MASK_AND_TRACK)


@pytest.fixture(scope="session")
def stage_2_result(runRecord: RunRecord, stage_1_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    missing = [p.name for p in GVHMR_CHECKPOINTS if not p.exists()]
    if missing:
        pytest.skip(f"needs the GVHMR checkpoints (missing: {missing}; see README's Setup section)")

    return _run_heavy_stage(runRecord, StageName.STAGE_2_ESTIMATE_HUMAN_MOTION)


@pytest.fixture(scope="session")
def stage_3_result(runRecord: RunRecord, stage_1_result: dict[str, str]) -> dict[str, str]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    return _run_heavy_stage(runRecord, StageName.STAGE_3_ESTIMATE_DEPTH)


@pytest.fixture(scope="session")
def stage_4_result(
    runRecord: RunRecord, stage_1_result: dict[str, str], stage_2_result: dict[str, str]
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

    return _run_heavy_stage(runRecord, StageName.STAGE_4_ESTIMATE_HANDS)


@pytest.fixture(scope="session")
def stage_5_result(
    runRecord: RunRecord, stage_2_result: dict[str, str], stage_4_result: dict[str, str]
) -> dict[str, str]:
    # No GPU/checkpoints of its own, but SmplxSkeleton (for the kinematic tree)
    # and the optional preview both need the registration-gated SMPL-X model
    # file, checked via this file's own SMPLX_MODEL_PATH, not by importing
    # stage_6's module, to avoid its heavy transitive chain.
    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    from pipeline.stages import stage_5_retarget_hands

    outputs = stage_5_retarget_hands.run(runRecord)
    runRecord.mark_progress(StageName.STAGE_5_RETARGET_HANDS, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_6_result(
    runRecord: RunRecord, stage_2_result: dict[str, str], stage_3_result: dict[str, str]
) -> dict[str, str]:
    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    from pipeline.stages import stage_6_align_scene_scale

    outputs = stage_6_align_scene_scale.run(runRecord)
    runRecord.mark_progress(StageName.STAGE_6_ALIGN_SCENE_SCALE, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_7_result(
    runRecord: RunRecord, stage_2_result: dict[str, str], stage_5_result: dict[str, str]
) -> dict[str, str]:
    # Needs stage 5's retargeted motion + stage 1's object mask, plus (now)
    # stage 2's own pre-foot-lock translation directly, see
    # stage_7_annotate_contacts.py's own module docstring for why, and
    # progress_tracker.STAGE_DEPENDS_ON's comment for why this deliberately
    # doesn't wait on align_scene_scale (stage 6). `stage_2_result` is listed
    # explicitly even though `stage_5_result` already depends on it
    # transitively, since stage 7 now genuinely reads stage 2's own output.
    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    from pipeline.stages import stage_7_annotate_contacts

    outputs = stage_7_annotate_contacts.run(runRecord)
    runRecord.mark_progress(StageName.STAGE_7_ANNOTATE_CONTACTS, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_8_result(
    runRecord: RunRecord, stage_6_result: dict[str, str], stage_7_result: dict[str, str]
) -> dict[str, str]:
    if not SMPLX_MODEL_PATH.exists():
        pytest.skip("needs the SMPL-X model file (registration-gated, see README's Setup section)")

    from pipeline.stages import stage_8_optimize_hoi

    outputs = stage_8_optimize_hoi.run(runRecord)
    runRecord.mark_progress(StageName.STAGE_8_OPTIMIZE_HOI, StageStatus.COMPLETE, outputs=outputs)
    return outputs


@pytest.fixture(scope="session")
def stage_9_result(
    runRecord: RunRecord, stage_1_result: dict[str, str], stage_2_result: dict[str, str]
) -> dict[str, str]:
    # Depends on stage_2_result now (not just stage_1_result): the body-based
    # orientation prior feeds GVHMR's own tracked head rotation into the
    # FLAME fit as a regularizer (see face_landmark_fit.calibrate_rotation_
    # offset and this stage's own _body_head_rotation), so stage 2's real
    # output needs to
    # exist on disk, not just frames + the human mask. Loads three heavy
    # model stacks of its own (DECA, MICA, MediaPipe's FaceLandmarker), so
    # this runs via `_run_heavy_stage` like stages 1-4, not in-process.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    missing = [
        p.name for p in (DECA_CHECKPOINT, MICA_CHECKPOINT, FACE_LANDMARKER_CHECKPOINT, VITPOSE_CHECKPOINT)
        if not p.exists()
    ]
    if missing:
        pytest.skip(f"needs the DECA/MICA/MediaPipe/ViTPose checkpoints (missing: {missing}; see README's Setup section)")
    if not FLAME_MODEL_PATH.exists():
        pytest.skip("needs the FLAME model file (registration-gated, see README's Setup section)")

    return _run_heavy_stage(runRecord, StageName.STAGE_9_CAPTURE_FACE)
