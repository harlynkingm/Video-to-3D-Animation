"""Stage 0 regression test: extracts frames from the small test clip and
computes camera intrinsics from real (tiny) input data. Runs anywhere -- no
GPU or checkpoints needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from conftest import FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, TEST_VIDEO_FRAME_COUNT, TEST_VIDEO_FPS, TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH, make_run_input
from pipeline.create_run import create_run
from pipeline.stages import stage_0_ingest_video
from pipeline.stages.stage_0_ingest_video import _resolve_rotation


def test_extracts_every_frame_as_a_valid_jpeg(stage_0_result):
    frames_dir = Path(stage_0_result["frames_dir"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    assert len(frame_paths) == TEST_VIDEO_FRAME_COUNT

    for path in frame_paths:
        image = cv2.imread(str(path))
        assert image is not None, f"{path} did not decode as a valid image"
        assert image.shape[:2] == (TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH)


def test_scene_info_matches_the_real_video(runRecord, stage_0_result):
    assert runRecord.scene.frame_count == TEST_VIDEO_FRAME_COUNT
    assert runRecord.scene.width == TEST_VIDEO_WIDTH
    assert runRecord.scene.height == TEST_VIDEO_HEIGHT
    assert runRecord.scene.fps == pytest.approx(TEST_VIDEO_FPS, abs=0.1)


def test_intrinsics_matrix_is_built_from_the_given_lens_info(runRecord, stage_0_result):
    K = runRecord.scene.intrinsics_K
    expected_focal_px = FOCAL_LENGTH_MM * (TEST_VIDEO_WIDTH / SENSOR_WIDTH_MM)
    assert K[0][0] == pytest.approx(expected_focal_px)
    assert K[1][1] == pytest.approx(expected_focal_px)
    assert K[0][2] == pytest.approx(TEST_VIDEO_WIDTH / 2)
    assert K[1][2] == pytest.approx(TEST_VIDEO_HEIGHT / 2)
    assert K[2][2] == 1.0


def test_intrinsics_k_bypasses_the_computed_matrix_when_given(tmp_path):
    """A one-off RunRecord (not the shared session-scoped `runRecord` fixture,
    since this needs its own distinct RunInput) with a raw K given should use
    it as-is, ignoring focal_length_mm/sensor_width_mm entirely."""
    raw_k = [[123.0, 0.0, 45.0], [0.0, 123.0, 67.0], [0.0, 0.0, 1.0]]
    run_input = make_run_input(focal_length_mm=0.0, sensor_width_mm=0.0, intrinsics_k=raw_k)
    runRecord = create_run(tmp_path, run_input)

    stage_0_ingest_video.run(runRecord)

    assert runRecord.scene.intrinsics_K == raw_k


def test_resolve_rotation_is_a_noop_for_unrotated_video():
    rotate_code, width, height = _resolve_rotation(0.0, raw_width=1920, raw_height=1080)
    assert rotate_code is None
    assert (width, height) == (1920, 1080)


def test_resolve_rotation_90_degrees():
    """Confirmed against a real vertical phone clip by visual inspection:
    CAP_PROP_ORIENTATION_META == 90 requires ROTATE_90_CLOCKWISE to produce a
    correctly upright frame, not the other direction."""
    rotate_code, width, height = _resolve_rotation(90.0, raw_width=3840, raw_height=2160)
    assert rotate_code == cv2.ROTATE_90_CLOCKWISE
    assert (width, height) == (2160, 3840)


def test_resolve_rotation_270_degrees():
    rotate_code, width, height = _resolve_rotation(270.0, raw_width=3840, raw_height=2160)
    assert rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE
    assert (width, height) == (2160, 3840)


def test_resolve_rotation_180_degrees_keeps_dimensions():
    rotate_code, width, height = _resolve_rotation(180.0, raw_width=1920, raw_height=1080)
    assert rotate_code == cv2.ROTATE_180
    assert (width, height) == (1920, 1080)
