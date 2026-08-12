"""Unit tests for compute_intrinsics_matrix, pure math, always runs."""

from __future__ import annotations

import pytest

from pipeline.helpers.camera_info_helpers import compute_intrinsics_matrix


def test_focal_length_uses_sensor_width_px_not_image_width_px():
    """The regression case this split exists for: a rotated frame's own
    image_width_px is NOT the same axis sensor_width_mm was measured against
    (see stage_0_ingest_video.py's rotation handling), e.g. a 3840x2160
    sensor rotated 90 degrees produces a 2160x3840 output frame, but the
    focal length ratio must still use the sensor's real 3840px width."""
    K = compute_intrinsics_matrix(
        focal_length_mm=35.0, sensor_width_mm=36.0,
        sensor_width_px=3840, image_width_px=2160, image_height_px=3840,
    )
    expected_focal_px = 35.0 * (3840 / 36.0)
    assert K[0][0] == pytest.approx(expected_focal_px)
    assert K[1][1] == pytest.approx(expected_focal_px)


def test_principal_point_centers_on_the_final_image_not_the_sensor():
    K = compute_intrinsics_matrix(
        focal_length_mm=35.0, sensor_width_mm=36.0,
        sensor_width_px=3840, image_width_px=2160, image_height_px=3840,
    )
    assert K[0][2] == pytest.approx(2160 / 2)
    assert K[1][2] == pytest.approx(3840 / 2)


def test_unrotated_case_uses_the_same_width_for_both():
    K = compute_intrinsics_matrix(
        focal_length_mm=35.0, sensor_width_mm=36.0,
        sensor_width_px=1920, image_width_px=1920, image_height_px=1080,
    )
    expected_focal_px = 35.0 * (1920 / 36.0)
    assert K[0][0] == pytest.approx(expected_focal_px)
    assert K[0][2] == pytest.approx(1920 / 2)
    assert K[1][2] == pytest.approx(1080 / 2)
    assert K[2][2] == 1.0
