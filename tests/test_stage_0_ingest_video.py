"""Stage 0 regression test: extracts frames from the small test clip and
computes camera intrinsics from real (tiny) input data. Runs anywhere -- no
GPU or checkpoints needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from conftest import FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, TEST_VIDEO_FRAME_COUNT, TEST_VIDEO_FPS, TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH


def test_extracts_every_frame_as_a_valid_jpeg(stage_0_result):
    frames_dir = Path(stage_0_result["frames_dir"])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    assert len(frame_paths) == TEST_VIDEO_FRAME_COUNT

    for path in frame_paths:
        image = cv2.imread(str(path))
        assert image is not None, f"{path} did not decode as a valid image"
        assert image.shape[:2] == (TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH)


def test_scene_info_matches_the_real_video(progress, stage_0_result):
    assert progress.scene.frame_count == TEST_VIDEO_FRAME_COUNT
    assert progress.scene.width == TEST_VIDEO_WIDTH
    assert progress.scene.height == TEST_VIDEO_HEIGHT
    assert progress.scene.fps == pytest.approx(TEST_VIDEO_FPS, abs=0.1)


def test_intrinsics_matrix_is_built_from_the_given_lens_info(progress, stage_0_result):
    K = progress.scene.intrinsics_K
    expected_focal_px = FOCAL_LENGTH_MM * (TEST_VIDEO_WIDTH / SENSOR_WIDTH_MM)
    assert K[0][0] == pytest.approx(expected_focal_px)
    assert K[1][1] == pytest.approx(expected_focal_px)
    assert K[0][2] == pytest.approx(TEST_VIDEO_WIDTH / 2)
    assert K[1][2] == pytest.approx(TEST_VIDEO_HEIGHT / 2)
    assert K[2][2] == 1.0
